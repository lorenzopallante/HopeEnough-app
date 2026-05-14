"""
HopeEnough — Streamlit web app for poor-outcome prediction after HOPE/DHOPE.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud: https://share.streamlit.io
  1. Push this repo (incl. model/LR_full.joblib) to GitHub
  2. "New app" → pick the repo → main file: app.py
  3. Deploy. Free, auto-redeploys on every push.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ── Paths ────────────────────────────────────────────────────────────────
# `model/` is the folder shipped to production (see .gitignore).
# Local training outputs in `results/` stay gitignored.
REPO_ROOT  = Path(__file__).parent
MODEL_DIR  = REPO_ROOT / "model"
DEFAULT_MODEL = "LR_full.joblib"
EXAMPLE_CSV = REPO_ROOT / "data" / "example_patients.csv"  # optional (gitignored)


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HopeEnough — Poor Outcome Predictor",
    page_icon="🫀",
    layout="centered",
)


# ── Cached loaders ───────────────────────────────────────────────────────
@st.cache_resource
def load_pipe(model_filename: str, _file_mtime: float):
    # _file_mtime is a cache-busting key: any file change invalidates the cache.
    return joblib.load(MODEL_DIR / model_filename)


@st.cache_resource
def load_calibrator(calibrator_filename: str, _file_mtime: float | None):
    """Load a portable JSON calibrator. Returns None if file is absent."""
    path = MODEL_DIR / calibrator_filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


def apply_calibration(p_raw: np.ndarray, cal: dict | None,
                       override_prevalence: float | None = None) -> np.ndarray:
    """Apply the calibration formula encoded in the JSON.

    Rationale: the underlying LR was trained with class_weight='balanced',
    which produces probabilities on a fictitious 50/50 prior. The calibrator
    (fitted via logistic or beta recalibration on OOF predictions) maps those
    raw scores back to real poor-outcome frequencies. Method and parameters
    are published in the JSON alongside the model. See upstream HopeEnough
    repo, src/calibration.py, and Ojeda et al. 2023 (PMID 37849356).

    If `override_prevalence` is given and differs from the training prevalence
    (cal['meta']['train_prevalence']), an additional log-odds shift is applied
    to move predictions onto a new population prior. This implements the
    King-Zeng prior correction for recalibration-in-the-large — useful when
    the app is applied to a center whose poor-outcome rate differs from the
    training population. Ranking among patients is preserved; only the
    absolute level shifts.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    if cal is None:
        return p_raw
    EPS = 1e-6
    p = np.clip(p_raw, EPS, 1.0 - EPS)
    if cal["method"] == "logistic":
        z = cal["intercept"] + cal["slope"] * np.log(p / (1.0 - p))
    elif cal["method"] == "beta":
        z = (cal["intercept"]
             + cal["coef_log_s"] * np.log(p)
             - cal["coef_neg_log_1ms"] * np.log(1.0 - p))
    else:
        return p_raw

    # Optional: shift predictions to a different target prevalence.
    if override_prevalence is not None:
        pi_train = cal.get("meta", {}).get("train_prevalence")
        if pi_train is not None and 0.0 < float(override_prevalence) < 1.0:
            pi_train = float(pi_train)
            pi_new = float(override_prevalence)
            if abs(pi_new - pi_train) > 1e-6:
                shift = (np.log(pi_new / (1.0 - pi_new))
                         - np.log(pi_train / (1.0 - pi_train)))
                z = z + shift

    return 1.0 / (1.0 + np.exp(-z))


def required_features(pipe) -> list[str]:
    feats = []
    for _, _, cols in pipe.named_steps["preprocessor"].transformers_:
        if cols == "drop":
            continue
        feats.extend(list(cols))
    return feats


def risk_colour(proba: float) -> str:
    if proba >= 0.66:
        return "#e74c3c"
    if proba >= 0.33:
        return "#f1c40f"
    return "#2ecc71"


# ── Sidebar: model ───────────────────────────────────────────────────────
st.sidebar.title("Model")

available_models = sorted(p.name for p in MODEL_DIR.glob("*_full.joblib")) \
    if MODEL_DIR.exists() else [DEFAULT_MODEL]
default_idx = available_models.index(DEFAULT_MODEL) \
    if DEFAULT_MODEL in available_models else 0

model_choice = st.sidebar.selectbox(
    "Trained pipeline",
    options=available_models,
    index=default_idx,
    help="`*_full.joblib` artifacts are refit on the entire dataset.",
)

pipe = load_pipe(model_choice, (MODEL_DIR / model_choice).stat().st_mtime)
FEATURES = required_features(pipe)

# Load companion calibrator: same basename with "_full" dropped + "_calibrator.json".
# e.g. "LR_full.joblib" → "LR_calibrator.json". Absence = uncalibrated deployment.
cal_filename = Path(model_choice).stem.replace("_full", "") + "_calibrator.json"
cal_path = MODEL_DIR / cal_filename
calibrator = load_calibrator(
    cal_filename,
    cal_path.stat().st_mtime if cal_path.exists() else None,
)

st.sidebar.caption(f"Model expects {len(FEATURES)} features: {', '.join(FEATURES)}")
if calibrator is not None:
    meta = calibrator.get("meta", {})
    st.sidebar.caption(
        f"Probabilities calibrated via **{calibrator['method']}** recalibration "
        f"(train prevalence: {meta.get('train_prevalence', '—')})."
    )
else:
    st.sidebar.caption(":warning: No calibrator — probabilities are on the raw "
                       "training scale (not interpretable as real frequencies).")

# ── Sidebar: local recalibration (prevalence shift) ──────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Local recalibration")

train_prev: float | None = None
if calibrator is not None:
    train_prev = calibrator.get("meta", {}).get("train_prevalence")

if calibrator is not None and train_prev is not None:
    default_pct = float(train_prev) * 100.0
    local_pct = st.sidebar.number_input(
        "Local population prevalence (%)",
        min_value=0.5, max_value=50.0,
        value=default_pct, step=0.5, format="%.1f",
        help=("If your target population has a different poor-outcome rate "
              "than the training population, enter it here. The app will "
              "shift every calibrated probability onto the new prior (King-"
              "Zeng recalibration-in-the-large). Assumes equal discrimination "
              "across populations — see README for the full protocol."),
    )
    local_prev = local_pct / 100.0
    shift_active = abs(local_prev - float(train_prev)) > 1e-4
    if shift_active:
        delta = (np.log(local_prev / (1.0 - local_prev))
                 - np.log(float(train_prev) / (1.0 - float(train_prev))))
        st.sidebar.caption(
            f":warning: Shifting from **{float(train_prev)*100:.1f}%** "
            f"(training) to **{local_pct:.1f}%** "
            f"(log-odds shift {delta:+.2f}). Rankings unchanged; "
            f"absolute levels scaled."
        )
    else:
        st.sidebar.caption(
            f"At training prevalence — no shift applied. "
            f"Set a different value to transport predictions."
        )
else:
    local_prev = None
    st.sidebar.caption(
        "Not available: model is deployed without a calibrator."
    )


# ── Header ───────────────────────────────────────────────────────────────
st.title("HopeEnough")
st.caption(
    "Poor-outcome prediction for liver transplant recipients after HOPE/DHOPE "
    "machine perfusion. **Research tool — not for clinical use.**"
)


# ── Tabs: single patient | batch CSV ─────────────────────────────────────
tab_single, tab_batch = st.tabs(["🧑‍⚕️ Single patient", "📄 Batch (CSV)"])


# ────────────────────────────────────────────────────────────────────────
# TAB 1 — single patient
# ────────────────────────────────────────────────────────────────────────
with tab_single:
    st.subheader("Enter patient features")

    with st.form("single_patient_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            life_supp = st.radio(
                "Life support pre-LT",
                options=[0, 1],
                format_func=lambda v: "No" if v == 0 else "Yes",
                horizontal=True,
            )
            macro_30 = st.radio(
                "Macrosteatosis ≥ 30%",
                options=[0, 1],
                format_func=lambda v: "< 30%" if v == 0 else "≥ 30%",
                horizontal=True,
            )

        with c2:
            cit_real = st.slider(
                "Cold ischemia time (min)",
                min_value=60, max_value=700, value=350, step=5,
            )
            meld_na = st.slider(
                "MELD-Na score",
                min_value=6, max_value=53, value=14, step=1,
            )

        submitted = st.form_submit_button("🔮 Predict", type="primary",
                                           width="stretch")

    if submitted:
        row = pd.DataFrame([{
            "life_supp": life_supp,
            "cit_real":  cit_real,
            "meld_na":   meld_na,
            "macro_30":  macro_30,
        }])[FEATURES]

        proba_raw = float(pipe.predict_proba(row)[0, 1])
        proba = float(apply_calibration(
            np.array([proba_raw]), calibrator,
            override_prevalence=local_prev,
        )[0])
        colour = risk_colour(proba)

        st.markdown(
            f"""
            <div style="padding:18px 22px; border-left:6px solid {colour};
                        background:#f7f7f7; border-radius:6px; margin-top:6px">
              <div style="font-size:13px; color:#666">
                  Predicted probability of poor outcome
              </div>
              <div style="font-size:42px; font-weight:700; color:{colour}; line-height:1.1">
                  {proba:.1%}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Show input row"):
            st.dataframe(row, hide_index=True, width="stretch")


# ────────────────────────────────────────────────────────────────────────
# TAB 2 — batch CSV
# ────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Predict from CSV")
    st.markdown(
        f"CSV must contain the columns **{', '.join(f'`{f}`' for f in FEATURES)}**. "
        "Any extra columns (e.g. `patient_id`) are preserved in the output."
    )

    uploaded = st.file_uploader("Upload a patients CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
    elif EXAMPLE_CSV.exists():
        st.caption(f"No file uploaded — using the bundled example: `{EXAMPLE_CSV.relative_to(REPO_ROOT)}`")
        df_in = pd.read_csv(EXAMPLE_CSV)
    else:
        df_in = None
        st.info(
            "Upload a CSV with the columns "
            + ", ".join(f"`{f}`" for f in FEATURES)
            + " (extra columns like `patient_id` are kept as-is)."
        )

    if df_in is not None:
        missing = [f for f in FEATURES if f not in df_in.columns]
        if missing:
            st.error(f"CSV is missing required feature columns: {missing}")
            st.stop()

        st.write(f"Loaded **{len(df_in)}** patient(s).")

        proba_raw = pipe.predict_proba(df_in[FEATURES])[:, 1]
        proba = apply_calibration(
            proba_raw, calibrator,
            override_prevalence=local_prev,
        )

        df_out = df_in.copy()
        df_out["predicted_proba"] = np.round(proba, 4)

        st.dataframe(
            df_out.style.format({"predicted_proba": "{:.3f}"})
                        .background_gradient(subset=["predicted_proba"],
                                              cmap="RdYlGn_r", vmin=0, vmax=1),
            hide_index=True,
            width="stretch",
        )

        st.info(
            f"Mean predicted probability of poor outcome across "
            f"**{len(df_in)}** patient(s): **{proba.mean():.1%}**."
        )

        st.download_button(
            "📥 Download predictions (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="hope_predictions.csv",
            mime="text/csv",
        )


# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Source: github.com/lorenzopallante/HopeEnough · "
    "Model: LR / DT / RF refit on 476 HOPE/DHOPE liver transplant cases "
    "(data_anonym_v3_clean). Research only."
)
