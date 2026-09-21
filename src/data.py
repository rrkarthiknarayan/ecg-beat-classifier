"""Load MIT-BIH records, clean the signal, cut out heartbeats, build features.

Binary task:  0 = normal beat (AAMI class N: N, L, R, e, j)
              1 = abnormal beat (AAMI classes S, V, F: A, a, J, S, V, E, F)
Paced / unclassifiable beats (/, f, Q) are dropped, as in the usual AAMI protocol.
"""
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

FS = 360                    # MIT-BIH sampling rate (Hz)
PRE, POST = 90, 110         # samples kept before / after the R-peak (0.25 s / 0.31 s)
DECIM = 2                   # keep every 2nd sample -> 100-point beat vector

NORMAL = set("NLRej")
ABNORMAL = set("AaJSVEF")
UNUSED = set("/fQ")         # still used to compute RR intervals, never classified

# Inter-patient split of de Chazal et al. (2004). Paced records 102/104/107/217 are excluded.
# (114 is missing from the GitHub copy of the data that this was built on.)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213,
       214, 219, 221, 222, 228, 231, 232, 233, 234]
# Whole patients held out of DS1 to calibrate the "uncertain" detector (see train.py)
CAL_RECORDS = [106, 115, 118, 203, 209]

RR_NAMES = ["RR_prev", "RR_next", "RR_local_avg", "RR_prev/avg", "RR_next/avg"]


def bandpass(x, fs=FS, lo=0.5, hi=40.0):
    """Zero-phase band-pass: removes baseline wander (<0.5 Hz) and muscle/mains noise (>40 Hz)."""
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def load_record(rec, data_dir):
    """Return (clean signal, annotation sample positions, annotation symbols)."""
    path = str(Path(data_dir) / str(rec))
    r = wfdb.rdrecord(path)
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = bandpass(r.p_signal[:, ch])
    sig = (sig - sig.mean()) / sig.std()          # per-record amplitude normalisation
    ann = wfdb.rdann(path, "atr")
    return sig, np.asarray(ann.sample), np.asarray(ann.symbol)


def rr_features(peaks):
    """Rhythm context for every beat, from R-peak positions (seconds)."""
    rr = np.diff(peaks) / FS
    med = np.median(rr) if len(rr) else 0.8
    prev = np.r_[med, rr]
    nxt = np.r_[rr, med]
    local = np.array([prev[max(0, i - 9): i + 1].mean() for i in range(len(peaks))])
    return np.c_[prev, nxt, local, prev / local, nxt / local]


def beat_vector(sig, s):
    """The 100-point waveform snippet around R-peak position s."""
    return sig[s - PRE: s + POST][::DECIM]


def build_dataset(records, data_dir):
    Xm, Xr, y, sym, rec_id, pos = [], [], [], [], [], []
    for rec in records:
        if not (Path(data_dir) / f"{rec}.dat").exists():
            print(f"  (record {rec} not found - skipped)")
            continue
        sig, samp, symb = load_record(rec, data_dir)
        keep = np.isin(symb, list(NORMAL | ABNORMAL | UNUSED))
        peaks, beat_sym = samp[keep], symb[keep]
        rr = rr_features(peaks)
        for i, (s, c) in enumerate(zip(peaks, beat_sym)):
            if c in UNUSED or s - PRE < 0 or s + POST > len(sig):
                continue
            Xm.append(beat_vector(sig, s))
            Xr.append(rr[i])
            y.append(int(c in ABNORMAL))
            sym.append(c)
            rec_id.append(rec)
            pos.append(s)
    return dict(Xm=np.array(Xm, dtype=np.float32), Xr=np.array(Xr, dtype=np.float32),
                y=np.array(y), sym=np.array(sym), rec=np.array(rec_id), pos=np.array(pos))


def concat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def features(d, use="both"):
    if use == "rr":
        return d["Xr"]
    if use == "morph":
        return d["Xm"]
    return np.c_[d["Xm"], d["Xr"]]
