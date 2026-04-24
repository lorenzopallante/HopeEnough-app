# HopeEnough — Web App

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hopeenough.streamlit.app/)

Streamlit app that predicts **poor outcome** for liver transplant recipients after HOPE/DHOPE (Hypothermic Oxygenated Perfusion / Dual-HOPE) machine perfusion.

Enter the four pre/intra-operative features for one patient — or upload a CSV with a whole cohort — and the app returns the predicted probability from a calibrated logistic-regression pipeline fitted on 476 HOPE/DHOPE transplants.

**⚠️ Research tool — not for clinical use.**

---

## 🌐 Live demo

👉 **[hopeenough.streamlit.app](https://hopeenough.streamlit.app/)**

Hosted on Streamlit Community Cloud, free tier. The app may take ~20 s to wake from sleep on the first request after a week of inactivity.

---

## What the app does

Two tabs:

| Tab | Input | Output |
| --- | --- | --- |
| **🧑‍⚕️ Single patient** | Sliders, toggles, and radios for the 4 features + a threshold slider | Live probability (calibrated, colour-coded), binary class, decision threshold |
| **📄 Batch (CSV)** | Upload a CSV or use the bundled example | Table with per-patient calibrated probabilities and classes, coloured by risk band, downloadable as CSV |

The sidebar lets an advanced user switch between any `*_full.joblib` model in `model/` (currently just the LR) and tune the decision threshold.

---

## Features expected

| Feature | Type | Description |
| --- | --- | --- |
| `life_supp` | binary | Life support pre-transplant (0 / 1) |
| `cit_real` | continuous (min) | Cold ischemia time |
| `meld_na` | continuous (pts) | MELD-Na score |
| `macro_30` | binary | Macrosteatosis ≥ 30% on donor biopsy (0 / 1) |

Inputs are passed in raw clinical units — the serialised pipeline handles imputation and standardisation internally. Missing values in CSV input are imputed automatically.

---

## The model

A scikit-learn `Pipeline` in [model/LR_full.joblib](model/LR_full.joblib) plus a post-hoc calibration layer in [model/LR_calibrator.json](model/LR_calibrator.json):

```
SimpleImputer(median) → StandardScaler    (continuous: cit_real, meld_na)
SimpleImputer(most_frequent)              (binary:     life_supp, macro_30)
          ↓
LogisticRegression(ElasticNet, class_weight='balanced')
          ↓
Post-hoc logistic recalibration            (intercept=-1.825, slope=0.930)
  p_calibrated = sigmoid(intercept + slope * logit(p_raw))
```

Fitted on the full 476-patient dataset (86.6% / 13.4% class split) with hyperparameters chosen via 5-fold stratified CV on ROC-AUC. The calibration layer maps raw `predict_proba` outputs (which sit on a fictitious 50/50 prior due to `class_weight='balanced'`) back onto the real 13.4% prevalence — so displayed probabilities are interpretable as actual frequencies of poor outcome. Method follows Ojeda et al. 2023 ([PMID 37849356](https://pubmed.ncbi.nlm.nih.gov/37849356/)).

---

## External validation and local recalibration

The shipped calibrator was fitted on a specific training population with a **13.4% prevalence of poor outcome**. Predictions are interpretable as real frequencies *only insofar as the target population resembles that of the training data*. If you plan to deploy the model in a new center or a different case mix, follow the TRIPOD recommendation and perform external validation + local recalibration.

**Minimal protocol for a new center:**

1. Collect at least ~100 consecutive HOPE/DHOPE patients with observed outcomes.
2. Run `app.py` in batch mode on those patients to obtain calibrated probabilities `p_i`.
3. Compute a reliability curve (observed-vs-predicted binned plot). If the points fall on the diagonal → the shipped calibrator is valid for your population; stop.
4. If the curve is off-diagonal in a parallel way (systematic over- or under-prediction) → **recalibration-in-the-large**: fit a single new intercept `a_new` via logistic regression `y ~ 1 + offset(logit(p))`, then apply `p_new = sigmoid(a_new + logit(p))` to every prediction.
5. If also the slope of the reliability curve differs from 1 → **logistic recalibration**: fit both intercept and slope via `y ~ logit(p)`, then apply `p_new = sigmoid(a_new + b_new * logit(p))`.

The code for steps 4–5 lives in the upstream [HopeEnough](https://github.com/lorenzopallante/HopeEnough) repo, `src/calibration.py::LogisticRecalibrator`. Swap the numbers in `model/LR_calibrator.json` and you are done — no retraining, no new model artefact.

**What the prevalence assumption means in practice.** If the true poor-outcome rate in your population is materially different from 13.4%, the shipped probabilities will be biased by a log-odds shift of `log(pi_yours / (1 - pi_yours)) - log(0.134 / 0.866)`. Slope distortions are rarer and require multivariable mis-specification, which external validation also surfaces.

---

## Run locally

```bash
git clone https://github.com/lorenzopallante/HopeEnough-app.git
cd HopeEnough-app

pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

## Repository structure

```text
HopeEnough-app/
├── app.py                       # Streamlit UI (two tabs: single-patient + batch CSV)
├── model/
│   ├── LR_full.joblib           # Serialised sklearn Pipeline (~4 KB)
│   └── LR_calibrator.json       # Post-hoc calibrator parameters (human-readable)
├── data/
│   └── example_patients.csv     # 5 fake patients — default batch-tab example
├── requirements.txt             # streamlit, scikit-learn, pandas, numpy, joblib, matplotlib
├── .gitignore
└── README.md
```

Everything in the repo is purely the deployable artifact — no training code, no intermediate files.

---

## Citation

If you use this app in research, please cite:

> *[Paper title, authors, journal, year — to be updated upon publication]*

## License

*To be defined — contact the authors before reuse.*
