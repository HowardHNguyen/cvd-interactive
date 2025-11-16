
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------
st.set_page_config(page_title="CVD Risk Predictor - Interactive", layout="wide")

# ---------------------------------------------------------------------
# Modeling imports
# ---------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# ---------------------------------------------------------------------
# Constants / knobs
# ---------------------------------------------------------------------
TARGET_FHX_OR = 2.5        # Literature-aligned OR for family history (~2–3x)
SMOOTH_EPS = 1e-3          # Prevent degenerate 0/1 before isotonic
MIN_BASELINE = 0.02        # Enforce a minimum baseline risk floor (2%)

# ---------------------------------------------------------------------
# Training & artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def train_and_get_artifacts():
    """Train model, align FHx effect, fit isotonic calibration, and return artifacts."""
    st.info("Training model... (first time only, ~3 sec)")

    # Load data
    df = pd.read_csv("data_cvd_perfect_300.csv", encoding="latin-1", engine="python", on_bad_lines="skip").copy()
    y = df['cvd'].astype(int)
    X = df.drop('cvd', axis=1)

    # Column roles
    num_cont = ['age','sys_bp','dia_bp','cholesterol','glucose','bmi']
    bin_cols = ['smoke','family_hx']
    text_col = 'note'

    # Preprocess: scale continuous; passthrough binaries; compact TF-IDF for note
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cont),
        ('bin', 'passthrough', bin_cols),
        ('text', TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    # Interpretable baseline model
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000, random_state=42))
    ])

    # Cross-validated AUROC (pre-calibration; for signal strength)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

    # Fit on all data
    model.fit(X, y)

    # Coefficients → odds ratios for interpretability
    tfidf_names = model.named_steps['prep'].named_transformers_['text'].get_feature_names_out().tolist()
    feature_names = num_cont + bin_cols + tfidf_names
    coef = model.named_steps['clf'].coef_[0]
    coef_dict = dict(zip(feature_names, coef))

    smoke_or_model = float(np.exp(coef_dict.get('smoke', 0.0)))
    fhx_or_model = float(np.exp(coef_dict.get('family_hx', 0.0)))

    # --- Family history odds alignment (post-hoc) ---
    # We will multiply odds by this factor only when family_hx == 1
    adjust_factor = TARGET_FHX_OR / fhx_or_model if fhx_or_model > 0 else 1.0

    # --- Build adjusted probabilities on training data ---
    base = model.predict_proba(X)[:, 1]
    fhx = X['family_hx'].values.astype(int)

    odds = np.clip(base / (1 - base + 1e-12), 1e-9, 1e9)
    adjusted_odds = odds * np.where(fhx == 1, adjust_factor, 1.0)
    adjusted_proba = adjusted_odds / (1 + adjusted_odds)

    # Smooth to avoid perfect-0/1 collapse before isotonic calibration
    adjusted_proba = np.clip(adjusted_proba, SMOOTH_EPS, 1 - SMOOTH_EPS)

    # --- Isotonic calibration ---
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(adjusted_proba, y)

    # Report a simple calibrated Brier score on train (for sanity check)
    brier = float(brier_score_loss(y, calibrator.predict(adjusted_proba)))

    st.success(f"Model ready! CV AUC = {auc:.2f}")

    return {
        "model": model,
        "coef_dict": coef_dict,
        "auc": float(auc),
        "smoke_or": smoke_or_model,
        "fhx_or_model": fhx_or_model,
        "adjust_factor": adjust_factor,
        "calibrator": calibrator,
        "brier_cal_train": brier
    }

art = train_and_get_artifacts()
model = art["model"]
coef_dict = art["coef_dict"]
auc = art["auc"]
smoke_or = art["smoke_or"]
fhx_or_model = art["fhx_or_model"]
adjust_factor = art["adjust_factor"]
calibrator = art["calibrator"]
brier_cal_train = art["brier_cal_train"]

# ---------------------------------------------------------------------
# Sidebar — Patient inputs
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Profile")
    age = st.slider("Age", 20, 90, 50, step=1)
    sys_bp = st.slider("Systolic BP (mmHg)", 90, 200, 120, step=1)
    dia_bp = st.slider("Diastolic BP (mmHg)", 50, 120, 80, step=1)
    cholesterol = st.slider("Cholesterol (mg/dL)", 120, 360, 180, step=1)
    glucose = st.slider("Glucose (mg/dL)", 70, 250, 100, step=1)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)

    smoking_label = st.selectbox("Smoking", ["No", "Yes"])
    family_hx_label = st.selectbox("Family History of CVD", ["No", "Yes"])

    default_note = "no symptoms reported"
    note = st.text_area(
        "Clinical Note",
        placeholder="e.g., gets short of breath walking uphill, smokes 1ppd",
        height=100
    ) or default_note

# Convert to model-ready fields
smoke = 1 if smoking_label == "Yes" else 0
family_hx = 1 if family_hx_label == "Yes" else 0

# Diabetes hint if glucose >= 126
if glucose >= 126:
    note = (note + " diabetes suspected by glucose >= 126 mg/dL").strip()

# Single-row frame for inference
row = pd.DataFrame([{
    "note": note,
    "age": age,
    "sys_bp": sys_bp,
    "dia_bp": dia_bp,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "bmi": bmi,
    "smoke": smoke,
    "family_hx": family_hx,
}])

# ---------------------------------------------------------------------
# Inference: apply FHx alignment + isotonic calibration + baseline floor
# ---------------------------------------------------------------------
p_base = float(model.predict_proba(row)[0, 1])
odds = p_base / (1 - p_base + 1e-12)
odds_adj = odds * (adjust_factor if family_hx == 1 else 1.0)
p_adj = odds_adj / (1 + odds_adj)
p_adj = max(min(p_adj, 1 - SMOOTH_EPS), SMOOTH_EPS)   # same smoothing as train
p_cal = float(calibrator.predict([p_adj])[0])
p_cal = max(p_cal, MIN_BASELINE)                      # enforce non-zero baseline

risk_pct = 100.0 * p_cal

# Risk categories
if p_cal >= 0.60:
    verdict = ("Extremely High — Urgent referral", "danger")
elif p_cal >= 0.15:
    verdict = ("High — Start treatment", "warning")
elif p_cal >= 0.07:
    verdict = ("Moderate — Consider meds", "warning")
else:
    verdict = ("Low — Lifestyle focus", "success")

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
left, right = st.columns([1, 1.2])

with left:
    st.title("Input Summary")
    st.markdown(
        f"""
        - **Age:** {age}
        - **BP:** {sys_bp}/{dia_bp} mmHg
        - **Cholesterol:** {cholesterol} mg/dL
        - **Glucose:** {glucose} mg/dL
        - **BMI:** {bmi:.1f}
        - **Smoker:** {"Yes" if smoke else "No"}
        - **Family Hx:** {"Yes" if family_hx else "No"}
        """
    )

with right:
    st.subheader("CVD 10-Year Risk")
    risk_style = {
        "danger": "<span style='color:#B00020;font-weight:800;font-size:48px'>{:.1f}%</span>",
        "warning": "<span style='color:#E67E22;font-weight:800;font-size:48px'>{:.1f}%</span>",
        "success": "<span style='color:#2E7D32;font-weight:800;font-size:48px'>{:.1f}%</span>",
    }[verdict[1]]
    st.markdown(risk_style.format(risk_pct), unsafe_allow_html=True)
    st.caption(verdict[0])

    st.subheader("Risk Factor Impact")
    # Smoking: show learned model OR (true 0->1 effect); FHx: show literature-aligned OR
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Smoking", f"{smoke_or:.1f}× risk" if smoke else "No impact")
    with c2:
        st.metric("Family History", f"{TARGET_FHX_OR:.1f}× risk" if family_hx else "No impact")

    st.caption("Family History effect aligned to literature (~2–3×) via post‑hoc odds adjustment; probabilities are isotonic‑calibrated and include a small baseline floor to avoid 0%.")

with st.expander("Model Metrics & Notes"):
    st.markdown(
        f"""
        - **Cross‑validated AUROC (pre‑calibration):** **{auc:.2f}**
        - **Calibrated Brier score (train):** **{brier_cal_train:.4f}** (lower is better)
        - **Smoking OR (learned):** **{smoke_or:.2f}×**
        - **Family Hx OR (learned):** **{fhx_or_model:.2f}×** → **aligned to {TARGET_FHX_OR:.1f}×**

        **Calibration:** We apply isotonic regression on post‑adjusted probabilities.  
        **Caution:** Dataset appears highly separable (AUC≈1.00). Real‑world performance will be lower; validate prospectively.
        """
    )

with st.expander("About This App"):
    st.markdown(
        f"""
        **Model:** TF‑IDF (clinical note) + standardized vitals + Logistic Regression (`balanced`).  
        **Calibration:** Family history OR aligned to **{TARGET_FHX_OR:.1f}×**; outputs are isotonic‑calibrated.

        **Important:** Decision support only — not a diagnostic device. Consult a clinician for care decisions.
        """
    )

st.markdown(
    """
    <hr/>
    <div style='font-size:12px;color:#6b6b6b'>
    By Howard Nguyen, PhD, 2025. Baseline logistic model with literature‑aligned FHx effect, isotonic calibration, and non‑zero baseline risk floor.
    </div>
    """, unsafe_allow_html=True
)
