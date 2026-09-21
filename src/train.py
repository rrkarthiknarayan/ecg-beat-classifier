"""Train, evaluate, calibrate and draw all figures.   Run:  python src/train.py"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import novel
from data import (CAL_RECORDS, DS1, DS2, build_dataset, concat, features, load_record,
                  subset)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data" / "mitdb", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
ALPHA = 0.10
COL = {0: "#2a9d5c", 1: "#d62839", 2: "#f0a202"}


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", early_stopping=False, random_state=0)


def report(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=(tp + tn) / len(y), sensitivity=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), precision=tp / max(tp + fp, 1),
                auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def kept_stats(y, dec):
    k = dec != novel.UNCERTAIN
    if k.sum() == 0:
        return dict(kept=0.0, accuracy=np.nan, sensitivity=np.nan, specificity=np.nan)
    yk, dk = y[k], dec[k]
    return dict(kept=float(k.mean()), accuracy=float((yk == dk).mean()),
                sensitivity=float((dk[yk == 1] == 1).mean()) if (yk == 1).any() else np.nan,
                specificity=float((dk[yk == 0] == 0).mean()))


def main():
    print("Building datasets ...")
    ds1, ds2 = build_dataset(DS1, DATA), build_dataset(DS2, DATA)
    is_cal = np.isin(ds1["rec"], CAL_RECORDS)
    train, cal = subset(ds1, ~is_cal), subset(ds1, is_cal)
    print(f"train {len(train['y'])} | calibration {len(cal['y'])} | test {len(ds2['y'])} beats")

    # ---- 1. which information does the model need?  (ablation on the patient-wise test set)
    results, models, probs = {}, {}, {}
    for use in ("rr", "morph", "both"):
        m = make_model().fit(features(train, use), train["y"])
        probs[use] = m.predict_proba(features(ds2, use))[:, 1]
        results[use] = report(ds2["y"], probs[use])
        models[use] = m
        r = results[use]
        print(f"{use:>6}: acc {r['accuracy']:.3f}  sens {r['sensitivity']:.3f}  "
              f"spec {r['specificity']:.3f}  AUROC {r['auroc']:.3f}")
    model, p_test = models["both"], probs["both"]

    # ---- 2. the classic mistake: random beat-wise split (same patient in train AND test)
    pool = concat(ds1, ds2)
    tr, te = train_test_split(np.arange(len(pool["y"])), test_size=0.3,
                              stratify=pool["y"], random_state=0)
    leak = make_model().fit(features(subset(pool, tr)), pool["y"][tr])
    results["random_split"] = report(pool["y"][te], leak.predict_proba(features(subset(pool, te)))[:, 1])
    r = results["random_split"]
    print(f"random beat split: acc {r['accuracy']:.3f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")

    # ---- 3. novel feature: conformal 'uncertain' detector calibrated on unseen patients
    p_cal = model.predict_proba(features(cal))
    p_test2 = np.c_[1 - p_test, p_test]
    q = novel.calibrate(p_cal, cal["y"], ALPHA)
    dec = novel.decide(p_test2, q)
    conf = dict(alpha=ALPHA, q=q.tolist(),
                coverage_calibration=novel.coverage(p_cal, cal["y"], q),
                coverage_test=novel.coverage(p_test2, ds2["y"], q),
                uncertain_fraction=float((dec == 2).mean()), **kept_stats(ds2["y"], dec),
                abnormal_missed_rate=float((dec[ds2["y"] == 1] == 0).mean()),
                baseline_missed_rate=float(1 - results["both"]["sensitivity"]),
                normal_call_precision=float((ds2["y"][dec == 0] == 0).mean()))
    print("conformal:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in conf.items()})

    # honest check: is the review band better than simply moving one threshold?
    called_n = (dec == 0).mean()
    t_base = np.quantile(p_test, called_n)
    base = dict(threshold=float(t_base), called_normal=float(called_n),
                abnormal_missed_rate=float(((p_test < t_base) & (ds2["y"] == 1)).sum() / (ds2["y"] == 1).sum()),
                normal_call_precision=float((ds2["y"][p_test < t_base] == 0).mean()))
    conf["single_threshold_baseline"] = base
    print("single-threshold baseline at same operating point:", {k: round(v, 4) for k, v in base.items()})

    sweep = []
    for a in np.linspace(0.02, 0.15, 10):
        d = novel.decide(p_test2, novel.calibrate(p_cal, cal["y"], a))
        sweep.append(dict(alpha=float(a), **kept_stats(ds2["y"], d)))

    per_symbol = {s: float((dec[ds2["sym"] == s] == 2).mean()) for s in np.unique(ds2["sym"])
                  if (ds2["sym"] == s).sum() >= 40}
    per_record = {}
    for rec in np.unique(ds2["rec"]):
        mk = ds2["rec"] == rec
        rr = report(ds2["y"][mk], p_test[mk]) if 0 < ds2["y"][mk].sum() < mk.sum() else None
        per_record[int(rec)] = dict(n=int(mk.sum()), abnormal=int(ds2["y"][mk].sum()),
                                    sensitivity=None if rr is None else rr["sensitivity"],
                                    specificity=None if rr is None else rr["specificity"],
                                    uncertain=float((dec[mk] == 2).mean()))

    json.dump(dict(counts=dict(train=len(train["y"]), calibration=len(cal["y"]), test=len(ds2["y"]),
                               test_abnormal_pct=float(ds2["y"].mean() * 100)),
                   results=results, conformal=conf, sweep=sweep,
                   uncertain_by_symbol=per_symbol, per_record=per_record),
              open(OUT / "metrics.json", "w"), indent=1)
    joblib.dump(dict(model=model, q=q, alpha=ALPHA, p_cal=p_cal, y_cal=cal["y"]), OUT / "model.joblib")

    # ================================ figures ================================
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    t = novel.beat_time_axis()

    # fig 1: what do beats look like?
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for c, name in ((0, "normal"), (1, "abnormal")):
        b = train["Xm"][train["y"] == c]
        ax[0].plot(t, b.mean(0), color=COL[c], label=f"{name} (n={len(b):,})")
        ax[0].fill_between(t, b.mean(0) - b.std(0), b.mean(0) + b.std(0), color=COL[c], alpha=.15)
    ax[0].set(title="Average beat +/- 1 std", xlabel="ms from R-peak", ylabel="normalised amplitude")
    ax[0].legend(frameon=False)
    names = {"N": "N normal", "L": "L left bundle", "R": "R right bundle", "A": "A atrial premature",
             "V": "V ventricular premature", "F": "F fusion"}
    for s, lab in names.items():
        b = train["Xm"][train["sym"] == s]
        if len(b):
            ax[1].plot(t, b.mean(0), label=lab, lw=1.6, ls="-" if s in "NLR" else "--")
    ax[1].set(title="Average beat by type", xlabel="ms from R-peak")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_beats.png", dpi=150); plt.close(fig)

    # fig 2: confusion matrix + ROC
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = results["both"]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Abnormal"],
              yticklabels=["Normal", "Abnormal"], xlabel="Predicted", ylabel="True",
              title="Confusion matrix - unseen patients (DS2)")
    for use, lab in (("rr", "rhythm only"), ("morph", "waveform only"), ("both", "waveform + rhythm")):
        fpr, tpr, _ = roc_curve(ds2["y"], probs[use])
        ax[1].plot(fpr, tpr, label=f"{lab} (AUC {results[use]['auroc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig2_performance.png", dpi=150); plt.close(fig)

    # fig 3: leakage
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    keys = ["accuracy", "sensitivity", "specificity"]
    w = .38
    ax.bar(np.arange(3) - w / 2, [results["random_split"][k] for k in keys], w, label="random beat split (leaky)", color="#9aa5b1")
    ax.bar(np.arange(3) + w / 2, [results["both"][k] for k in keys], w, label="patient-wise split (honest)", color="#1d5fa8")
    for i, k in enumerate(keys):
        ax.text(i - w / 2, results["random_split"][k] + .01, f"{results['random_split'][k]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, results["both"][k] + .01, f"{results['both'][k]:.3f}", ha="center", fontsize=8)
    ax.set(xticks=range(3), xticklabels=keys, ylim=(0, 1.1), title="Same model, two ways of testing it")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_leakage.png", dpi=150); plt.close(fig)

    # fig 4: uncertainty trade-off
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    sw = [s for s in sweep if s["kept"] > 0]
    ab = [1 - s["kept"] for s in sw]
    ax[0].plot(ab, [s["accuracy"] for s in sw], label="accuracy", color="#1d5fa8")
    ax[0].plot(ab, [s["sensitivity"] for s in sw], label="sensitivity (abnormal caught)", color=COL[1])
    ax[0].plot(ab, [s["specificity"] for s in sw], label="specificity", color=COL[0])
    ax[0].scatter([conf["uncertain_fraction"]], [conf["sensitivity"]], color="k", zorder=5, s=25)
    ax[0].annotate(f"alpha={ALPHA}", (conf["uncertain_fraction"], conf["sensitivity"]),
                   textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax[0].set(xlabel="fraction of beats sent to 'uncertain'", ylabel="score on the beats the model keeps",
              title="Letting the model abstain", ylim=(0.6, 1.01))
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    syms = sorted(per_symbol, key=per_symbol.get, reverse=True)
    ax[1].bar(syms, [per_symbol[s] * 100 for s in syms], color=["#d62839" if s in "AaJSVEF" else "#2a9d5c" for s in syms])
    ax[1].set(ylabel="% flagged uncertain", title=f"Which beat types get flagged? (alpha={ALPHA})",
              xlabel="beat type (red = abnormal, green = normal)")
    fig.tight_layout(); fig.savefig(OUT / "fig4_uncertainty.png", dpi=150); plt.close(fig)

    # fig 5: ECG strip with predictions;  fig 6: explanations
    strip(ds2, dec, p_test)
    explain_fig(model, ds2, dec, p_test, q)
    print("done ->", OUT)


def strip(ds2, dec, p, seconds=10):
    """Find a 10 s window (in an unseen patient) that shows normal, abnormal AND uncertain beats."""
    best = None
    for rec in (200, 210, 213, 214, 221, 228, 233, 100, 119):
        mk = ds2["rec"] == rec
        if not mk.any():
            continue
        pos, d = ds2["pos"][mk], dec[mk]
        for lo in range(0, int(pos.max()) - seconds * 360, 360):
            w = (pos >= lo) & (pos < lo + seconds * 360)
            if {0, 1, 2} <= set(d[w]) and w.sum() >= 8:
                best = (rec, lo)
                break
        if best:
            break
    rec, lo = best
    sig, _, _ = load_record(rec, DATA)
    mk = ds2["rec"] == rec
    pos, y, sym, d = ds2["pos"][mk], ds2["y"][mk], ds2["sym"][mk], dec[mk]
    hi = lo + seconds * 360
    x = np.arange(lo, hi) / 360
    top = sig[lo:hi].max() + 1.2
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(x, sig[lo:hi], color="#333", lw=.9)
    for s_, sy, dd in zip(pos, sym, d):
        if lo <= s_ < hi:
            ax.scatter(s_ / 360, sig[s_], color=COL[dd], s=70, zorder=5, edgecolor="white")
            ax.text(s_ / 360, top, sy, ha="center", fontsize=9, color="#555")
    ax.text(x[0], top + .9, "true annotation:", fontsize=8, color="#555", va="bottom")
    for c in COL:
        ax.scatter([], [], color=COL[c], label="model says " + novel.LABELS[c].lower())
    ax.set(xlabel="time (s)", ylabel="amplitude (normalised)", ylim=(sig[lo:hi].min() - .8, top + 2.2),
           title=f"Record {rec} (patient never seen in training): decision for each beat")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig5_strip.png", dpi=150); plt.close(fig)


def explain_fig(model, ds2, dec, p, q):
    rng = np.random.default_rng(1)
    pick = []
    for cond in ((ds2["y"] == 1) & (dec == 1) & (ds2["sym"] == "V"),
                 (ds2["y"] == 0) & (dec == 0), dec == 2):
        idx = np.where(cond)[0]
        pick.append(int(rng.choice(idx)))
    t = novel.beat_time_axis()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, i, ttl in zip(ax, pick, ("Abnormal (V), confident", "Normal, confident", "Uncertain")):
        imp, cls, pa = novel.occlusion_map(model, ds2["Xm"][i], ds2["Xr"][i])
        m = np.abs(imp).max() + 1e-9
        lo_, hi_ = ds2["Xm"][i].min() - .5, ds2["Xm"][i].max() + .5
        a.imshow(imp[None], extent=(t[0], t[-1], lo_, hi_), aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m, alpha=.55)
        a.plot(t, ds2["Xm"][i], color="k", lw=1.6)
        a.set(title=f"{ttl}\ntrue={ds2['sym'][i]}  P(abnormal)={pa:.2f}", xlabel="ms from R-peak", ylim=(lo_, hi_))
    ax[0].set_ylabel("amplitude")
    fig.suptitle("Occlusion explanation: red = this part of the beat supports the model's call, blue = argues against", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig6_explain.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
