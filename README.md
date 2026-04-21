# HopeEnough — Web App

Streamlit app that predicts **poor outcome** for liver transplant recipients after HOPE/DHOPE (Hypothermic Oxygenated Perfusion / Dual-HOPE) machine perfusion. Enter the five pre/intra-operative features for a patient — or upload a CSV with a cohort — and the app returns the predicted probability from a logistic-regression pipeline fitted on 476 HOPE/DHOPE transplants.

**⚠️ Research tool — not for clinical use.**

---

## Live demo

*Will be linked here once deployed on Streamlit Community Cloud.*

---

## Model

A single scikit-learn `Pipeline` serialised in [model/LR_full.joblib](model/LR_full.joblib):

```
SimpleImputer(median) → StandardScaler          (continuous)
SimpleImputer(most_frequent)                     (binary)
SimpleImputer(most_frequent) → OneHotEncoder     (categorical)
          ↓
LogisticRegression(ElasticNet, class_weight='balanced')
```

Fitted on the full 476-patient dataset with the best hyperparameters from 5-fold stratified CV (ROC-AUC).

### Features expected

| Feature | Type | Description |
| --- | --- | --- |
| `life_supp` | binary | Life support pre-transplant (0/1) |
| `cit_real` | continuous (min) | Cold ischemia time |
| `meld_na` | continuous (pts) | MELD-Na score |
| `status_pre` | categorical | Pre-LT location: `home` / `hospital` / `ICU` / `RIA` |
| `macro_15` | binary | Macrosteatosis ≥ 15% (0/1) |

Inputs are passed in raw clinical units — the pipeline handles imputation, scaling, and one-hot encoding internally.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

---

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Pick `lorenzopallante/HopeEnough-app`, branch `main`, main file `app.py`.
3. Deploy. Any push to `main` triggers an auto-redeploy.

---

## Citation

*[Paper title, authors, journal, year — to be updated upon publication]*
