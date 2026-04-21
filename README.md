# HopeEnough — Web App

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hopeenough.streamlit.app/)

Streamlit app that predicts **poor outcome** for liver transplant recipients after HOPE/DHOPE (Hypothermic Oxygenated Perfusion / Dual-HOPE) machine perfusion.

Enter the five pre/intra-operative features for one patient — or upload a CSV with a whole cohort — and the app returns the predicted probability from a logistic-regression pipeline fitted on 476 HOPE/DHOPE transplants.

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
| **🧑‍⚕️ Single patient** | Sliders, toggles, and a dropdown for the 5 features + a threshold slider | Live probability (colour-coded), binary class, decision threshold |
| **📄 Batch (CSV)** | Upload a CSV or use the bundled example | Table with per-patient probabilities and classes, coloured by risk band, downloadable as CSV |

The sidebar lets an advanced user switch between any `*_full.joblib` model in `model/` (currently just the LR) and tune the decision threshold.

---

## Features expected

| Feature | Type | Description |
| --- | --- | --- |
| `life_supp` | binary | Life support pre-transplant (0 / 1) |
| `cit_real` | continuous (min) | Cold ischemia time |
| `meld_na` | continuous (pts) | MELD-Na score |
| `status_pre` | categorical | Pre-LT location: `home` / `hospital` / `ICU` / `RIA` |
| `macro_15` | binary | Macrosteatosis ≥ 15% on donor biopsy (0 / 1) |

Inputs are passed in raw clinical units — the serialised pipeline handles imputation, standardisation, and one-hot encoding internally. Missing values in CSV input are imputed automatically.

---

## The model

A single scikit-learn `Pipeline` in [model/LR_full.joblib](model/LR_full.joblib):

```
SimpleImputer(median) → StandardScaler          (continuous: cit_real, meld_na)
SimpleImputer(most_frequent)                    (binary:    life_supp, macro_15)
SimpleImputer(most_frequent) → OneHotEncoder    (categorical: status_pre)
          ↓
LogisticRegression(ElasticNet, class_weight='balanced', C=0.1, l1_ratio=0.0)
```

Fitted on the full 476-patient dataset (86.6% / 13.4% class split) with hyperparameters chosen via 5-fold stratified CV on ROC-AUC.

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
│   └── LR_full.joblib           # Serialised sklearn Pipeline (~4 KB)
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
