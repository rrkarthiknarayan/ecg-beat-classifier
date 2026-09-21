"""The two novel features of this project.

1. Conformal "uncertain" detector
   Instead of forcing every beat into normal/abnormal, we calibrate the model on
   held-out *patients* so it can say "I'm not sure - a human should look at this".
   Class-conditional (Mondrian) split conformal prediction: for each class c we find
   a threshold q_c so that, on calibration data, the true class is kept in the
   prediction set at least (1 - alpha) of the time.

2. Occlusion explanations
   Hide small pieces of the heartbeat one at a time and see how much the model's
   confidence drops. The pieces that hurt most are the ones the model relied on.
   Works with any model, needs no deep-learning framework.
"""
import numpy as np

from data import DECIM, FS, PRE

NORMAL, ABNORMAL, UNCERTAIN = 0, 1, 2
LABELS = {0: "Normal", 1: "Abnormal", 2: "Uncertain"}


# ---------------------------------------------------------------- conformal
def calibrate(proba_cal, y_cal, alpha=0.10):
    """Per-class thresholds on the non-conformity score s = 1 - P(true class)."""
    q = np.zeros(2)
    for c in (0, 1):
        s = 1.0 - proba_cal[y_cal == c, c]
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[c] = np.quantile(s, level, method="higher")
    return q


def predict_sets(proba, q):
    """Boolean (n, 2): is class c inside the prediction set of each beat?"""
    return (1.0 - proba) <= q[None, :]


def decide(proba, q):
    """Final label per beat: 0 normal, 1 abnormal, 2 uncertain (both classes or neither)."""
    sets = predict_sets(proba, q)
    out = np.full(len(proba), UNCERTAIN)
    out[sets[:, 0] & ~sets[:, 1]] = NORMAL
    out[sets[:, 1] & ~sets[:, 0]] = ABNORMAL
    return out


def coverage(proba, y, q):
    sets = predict_sets(proba, q)
    return {c: float(sets[y == c, c].mean()) for c in (0, 1)}


# ---------------------------------------------------------------- occlusion
def occlusion_map(model, x_morph, x_rr, win=12, stride=3):
    """Importance of every time step of one beat for the model's own prediction.

    x_morph: (100,) beat vector, x_rr: (5,) rhythm features.
    Returns (importance (100,), predicted class, P(abnormal)).
    Positive = this part of the beat pushed the model toward its prediction.
    """
    n = len(x_morph)
    base = model.predict_proba(np.r_[x_morph, x_rr][None])[0]
    cls = int(base[1] >= 0.5)
    starts = list(range(0, n - win + 1, stride))
    batch = np.tile(np.r_[x_morph, x_rr], (len(starts), 1))
    for k, s in enumerate(starts):        # replace the window by a straight line
        batch[k, s:s + win] = np.linspace(x_morph[s], x_morph[s + win - 1], win)
    drop = base[cls] - model.predict_proba(batch)[:, cls]
    imp, cnt = np.zeros(n), np.zeros(n)
    for k, s in enumerate(starts):
        imp[s:s + win] += drop[k]
        cnt[s:s + win] += 1
    return imp / np.maximum(cnt, 1), cls, float(base[1])


def beat_time_axis(n=100):
    """Time (ms) relative to the R-peak for a beat vector."""
    return (np.arange(n) * DECIM - PRE) / FS * 1000
