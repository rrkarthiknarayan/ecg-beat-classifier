"""Interactive demo.   Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import novel                                        # noqa: E402
from data import DS2, FS, build_dataset, features, load_record  # noqa: E402

DATA, OUT = ROOT / "data" / "mitdb", ROOT / "outputs"
COL = {0: "#2a9d5c", 1: "#d62839", 2: "#f0a202"}

st.set_page_config(page_title="ECG beat classifier", page_icon="🫀", layout="wide")
st.title("ECG heartbeat classifier - normal vs abnormal")
st.warning("Research / classroom benchmark on the public MIT-BIH database. "
           "Not a medical device and not a diagnosis.")

if not DATA.exists():
    st.error("Run `python download_data.py` first."); st.stop()

@st.cache_resource
def get_bundle():
    try:
        return joblib.load(OUT / "model.joblib")
    except Exception:
        from train import make_model
        recs = sorted(int(p.stem) for p in DATA.glob("*.dat"))
        ds = build_dataset(recs, DATA)
        idx = np.random.default_rng(0).permutation(len(ds["y"]))
        cut = int(len(idx) * 0.8)
        tr, ca = idx[:cut], idx[cut:]
        train_ds = {k: v[tr] for k, v in ds.items()}
        cal_ds = {k: v[ca] for k, v in ds.items()}
        m = make_model().fit(features(train_ds), train_ds["y"])
        p_cal = m.predict_proba(features(cal_ds))
        q = novel.calibrate(p_cal, cal_ds["y"], 0.10)
        return dict(model=m, q=q, alpha=0.10, p_cal=p_cal, y_cal=cal_ds["y"])
    
bundle = get_bundle()
model = bundle["model"]
metrics = json.load(open(OUT / "metrics.json")) if (OUT / "metrics.json").exists() else {"results": {}, "conformal": {}}


@st.cache_data
def load(rec):
    sig, _, _ = load_record(rec, DATA)
    d = build_dataset([rec], DATA)
    p = model.predict_proba(features(d))[:, 1]
    return sig, d, p


tab_demo, tab_bench = st.tabs(["Explore a recording", "Benchmark results"])

with tab_demo:
    avail = [r for r in DS2 if (DATA / f"{r}.dat").exists()]
    c1, c2, c3 = st.columns(3)
    rec = c1.selectbox("Recording (patient never seen in training)", avail, index=avail.index(200) if 200 in avail else 0)
    alpha = c2.slider("Review band strictness (alpha)", 0.02, 0.15, 0.10, 0.01,
                      help="Lower alpha = the model asks for human review more often, "
                           "but rarely misses an abnormal beat.")
    sig, d, p = load(rec)
    q = novel.calibrate(bundle["p_cal"], bundle["y_cal"], alpha)   # p_cal is already (n, 2)
    dec = novel.decide(np.c_[1 - p, p], q)
    dur = len(sig) / FS
    t0 = c3.slider("Start time (s)", 0, int(dur) - 10, 20, 5)
    lo, hi = t0 * FS, (t0 + 10) * FS

    fig, ax = plt.subplots(figsize=(12, 3.4))
    x = np.arange(lo, hi) / FS
    ax.plot(x, sig[lo:hi], color="#333", lw=.9)
    inwin = [i for i, s in enumerate(d["pos"]) if lo <= s < hi]
    for i in inwin:
        s = d["pos"][i]
        ax.scatter(s / FS, sig[s], color=COL[dec[i]], s=70, zorder=5, edgecolor="white")
    for c in COL:
        ax.scatter([], [], color=COL[c], label=novel.LABELS[c])
    ax.set(xlabel="time (s)", ylabel="amplitude (normalised)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)

    n = len(dec)
    m1, m2, m3 = st.columns(3)
    m1.metric("Beats in this recording", f"{n:,}")
    m2.metric("Sent to human review", f"{(dec == 2).mean():.0%}")
    m3.metric("Actually abnormal (annotation)", f"{d['y'].mean():.0%}")

    if inwin:
        rows = pd.DataFrame({
            "time (s)": [round(d["pos"][i] / FS, 2) for i in inwin],
            "true label": [d["sym"][i] for i in inwin],
            "P(abnormal)": [round(float(p[i]), 3) for i in inwin],
            "model says": [novel.LABELS[dec[i]] for i in inwin]})
        st.dataframe(rows, width="stretch", hide_index=True)
        pick = st.selectbox("Explain a beat", inwin, format_func=lambda i: f"t = {d['pos'][i] / FS:.2f} s  "
                            f"({d['sym'][i]}, {novel.LABELS[dec[i]]})")
        imp, cls, pa = novel.occlusion_map(model, d["Xm"][pick], d["Xr"][pick])
        t = novel.beat_time_axis()
        m = np.abs(imp).max() + 1e-9
        lo_, hi_ = d["Xm"][pick].min() - .5, d["Xm"][pick].max() + .5
        f2, a2 = plt.subplots(figsize=(6, 3))
        a2.imshow(imp[None], extent=(t[0], t[-1], lo_, hi_), aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m, alpha=.55)
        a2.plot(t, d["Xm"][pick], color="k", lw=1.6)
        a2.set(xlabel="ms from R-peak", ylim=(lo_, hi_), title=f"P(abnormal) = {pa:.2f}")
        a2.spines[["top", "right"]].set_visible(False)
        cA, cB = st.columns([1, 1])
        cA.pyplot(f2, clear_figure=True)
        cB.markdown("**How to read this:** red = this part of the beat pushed the model *toward* its "
                    "call, blue = it argued *against*. The model also uses rhythm (time since the "
                    "previous beat), which cannot be drawn on the waveform.")

with tab_bench:
    r, c = metrics["results"], metrics["conformal"]
    st.subheader("Test on 22 patients the model never saw (DS2)")
    df = pd.DataFrame({k: r[k] for k in ("rr", "morph", "both", "random_split")}).T
    df.index = ["rhythm only", "waveform only", "waveform + rhythm (final)", "random beat split (leaky!)"]
    st.dataframe(df[["accuracy", "sensitivity", "specificity", "precision", "auroc", "auprc"]].round(3))
    st.image(str(OUT / "fig2_performance.png"))
    st.image(str(OUT / "fig3_leakage.png"))
    st.subheader("Novel feature: the 'uncertain' band")
    st.write(f"At alpha = {c['alpha']}: {c['uncertain_fraction']:.0%} of beats go to review. "
             f"Only {c['abnormal_missed_rate']:.1%} of abnormal beats are confidently called normal "
             f"(vs {c['baseline_missed_rate']:.1%} for the plain model), and a confident 'normal' is right "
             f"{c['normal_call_precision']:.1%} of the time.")
    st.image(str(OUT / "fig4_uncertainty.png"))
    st.image(str(OUT / "fig6_explain.png"))
