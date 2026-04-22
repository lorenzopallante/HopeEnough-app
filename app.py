"""
HopeEnough — Streamlit web app for poor-outcome prediction after HOPE/DHOPE.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud: https://share.streamlit.io
  1. Push this repo (incl. model/LR_full.joblib) to GitHub
  2. "New app" → pick the repo → main file: app.py
  3. Deploy. Free, auto-redeploys on every push.
"""

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
def load_pipe(model_filename: str):
    return joblib.load(MODEL_DIR / model_filename)


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


# ── Sidebar: model & threshold ───────────────────────────────────────────
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
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05, max_value=0.95, value=0.50, step=0.01,
    help="Probability ≥ threshold → predicted poor outcome. "
         "Use the `optimal_threshold` from run_log.json for Youden's J.",
)

pipe = load_pipe(model_choice)
FEATURES = required_features(pipe)

st.sidebar.caption(f"Model expects {len(FEATURES)} features: {', '.join(FEATURES)}")


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
                "Life support pre-LT (`life_supp`)",
                options=[0, 1],
                format_func=lambda v: "No" if v == 0 else "Yes",
                horizontal=True,
            )
            macro_30 = st.radio(
                "Macrosteatosis ≥ 30% (`macro_30`)",
                options=[0, 1],
                format_func=lambda v: "< 30%" if v == 0 else "≥ 30%",
                horizontal=True,
            )

        with c2:
            cit_real = st.slider(
                "Cold ischemia time (`cit_real`, min)",
                min_value=60, max_value=700, value=350, step=5,
            )
            meld_na = st.slider(
                "MELD-Na score (`meld_na`)",
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

        proba = float(pipe.predict_proba(row)[0, 1])
        pred  = int(proba >= threshold)
        colour = risk_colour(proba)
        label  = "POOR OUTCOME predicted" if pred == 1 else "no poor outcome predicted"

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
              <div style="font-size:14px; color:#333; margin-top:4px">
                  at threshold <b>{threshold:.2f}</b> → <b>{label}</b>
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

        proba = pipe.predict_proba(df_in[FEATURES])[:, 1]
        pred  = (proba >= threshold).astype(int)

        df_out = df_in.copy()
        df_out["predicted_proba"] = np.round(proba, 4)
        df_out["predicted_class"] = pred
        df_out["threshold_used"]  = threshold

        st.dataframe(
            df_out.style.format({"predicted_proba": "{:.3f}",
                                  "threshold_used":  "{:.2f}"})
                        .background_gradient(subset=["predicted_proba"],
                                              cmap="RdYlGn_r", vmin=0, vmax=1),
            hide_index=True,
            width="stretch",
        )

        n_pos = int(pred.sum())
        st.info(f"Predicted poor outcome: **{n_pos} / {len(df_in)}** "
                f"({n_pos/len(df_in)*100:.1f}%) at threshold {threshold:.2f}.")

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
