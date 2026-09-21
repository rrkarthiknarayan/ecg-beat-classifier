# ECG heartbeat classifier: normal vs abnormal

Classroom **benchmark** on the public MIT-BIH Arrhythmia Database. Not a medical device, not a diagnosis.

## Run it
```bash
pip install -r requirements.txt
python download_data.py        # MIT-BIH from PhysioNet (~100 MB)
python src/train.py            # trains, evaluates, writes outputs/ (figures, metrics.json, model)
streamlit run app.py           # interactive demo
```

## What it does
1. **Data**: 100-sample window around every annotated R-peak (band-pass 0.5-40 Hz), plus 5 rhythm features
   (time since previous beat, to next beat, ratios to the recent average).
2. **Labels**: AAMI grouping. Normal = N, L, R, e, j. Abnormal = A, a, J, S (supraventricular), V, E (ventricular), F (fusion).
   Paced/unclassifiable beats and paced records (102, 104, 107, 217) are dropped.
3. **Split by patient** (de Chazal DS1 train / DS2 test): no patient appears on both sides.
4. **Model**: gradient-boosted trees (scikit-learn). Ablation: rhythm only, waveform only, both.

## Results (22 unseen patients, 49,690 beats, 11% abnormal)
| Features | Accuracy | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|
| rhythm only | 0.868 | 0.857 | 0.869 | 0.928 |
| waveform only | 0.810 | 0.839 | 0.807 | 0.903 |
| **waveform + rhythm** | **0.905** | 0.806 | 0.917 | **0.948** |
| *same model, random beat split (leaky)* | *0.994* | *0.977* | *0.996* | *0.999* |

The last row is why the patient-wise split matters: mixing one patient's beats into both train and test
inflates accuracy from 90% to 99%. Precision of the abnormal class is only 0.54 (many false alarms).

## Novel features
**1. Conformal "uncertain" band.** The model may answer *normal / abnormal / uncertain*. Thresholds are calibrated
on 5 whole DS1 patients held out of training, per class (Mondrian split conformal), with target coverage 1 - alpha.
At alpha = 0.10: 23% of beats go to review, 64% are auto-called normal (99.4% of those really are normal),
13% are called abnormal. Only 3.7% of abnormal beats are confidently called normal (plain model: 19.4%).
Per-class coverage on the unseen patients was 94.6% / 96.3% (target 90%).
Trade-off: alpha 0.02 -> 54% reviewed, 97.2% sensitivity on kept beats; alpha 0.15 -> 5% reviewed, 90.3%.

**Honest caveat:** with two classes this is equivalent to a single threshold on P(abnormal) plus a review band.
Moving one plain threshold to the same operating point gives *identical* numbers (see
`conformal.single_threshold_baseline` in `metrics.json`). What conformal adds is a principled way to choose the
thresholds on unseen patients and a knob (alpha) with a defined meaning, not extra accuracy.

**2. Occlusion explanations.** Hide 12-sample pieces of a beat, see how much confidence drops, and paint the
result on the waveform (`fig6_explain.png`, and in the app). Model-agnostic.

## Known limitations (put these in your report)
- Beat positions come from the database annotations. A deployed system needs an R-peak detector first.
- Results vary a lot per patient (`per_record` in metrics.json): record 117 is all normal but specificity is 0.09;
  record 232 is 78% abnormal and sensitivity is 0.39. Pooled numbers are dominated by the big records.
- Review load is uneven: 0% of beats for some patients, 81% for record 213.
- Bundle-branch-block beats (L) are labelled "normal" under AAMI but look abnormal, so 53% of them get flagged uncertain.
- One database, one lead (MLII), 47 people. No external validation. Calibration used only 5 patients.
- Record 114 was missing from the copy of the data used to build this, so DS1 has 21 of 22 patients.
