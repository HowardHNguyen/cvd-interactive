
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Page config MUST be the first Streamlit call
# -------------------------------------------------------------
st.set_page_config(
    page_title="CVD Risk Predictor - Interactive",
    layout="wide"
)

# -------------------------------------------------------------
# Training / modeling utilities
# -------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score

@st.cache_resource
def train_and_get_model():
    """Load data, build pipeline, compute CV AUC, fit, and return model + coefs."""
    st.info("Training model... (first time only, ~3 sec)")

    # Load attached dataset
    df = pd.read_csv(
        "data_cvd_perfect_300.csv",
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip"
    )

    # Columns by type
    num_cont = ['age','sys_bp','dia_bp','cholesterol','glucose','bmi']
    bin_cols = ['smoke','family_hx']
    text_col = 'note'

    # Preprocessor: scale ONLY continuous features; pass through binary 0/1 as-is;
    # a small TF-IDF space keeps the model compact & robust to phrasing.
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cont),
        ('bin', 'passthrough', bin_cols),
        ('text', TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    # Logistic Regression with class_weight balancing; deterministic
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000, random_state=42))
    ])

    X = df.drop('cvd', axis=1)
    y = df['cvd']

    # Robust CV AUC for display (not hard-coded)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

    # Fit once and extract coefficients mapped to feature names in correct order
    model.fit(X, y)

    # Build feature name list in the same order as the transformed matrix
    tfidf_names = model.named_steps['prep'].named_transformers_['text'].get_feature_names_out().tolist()
    feature_names = num_cont + bin_cols + tfidf_names

    # Coefficients (for binary columns 'smoke' and 'family_hx', these are true 0->1 effects now)
    coef = model.named_steps['clf'].coef_[0]
    coef_dict = dict(zip(feature_names, coef))

    st.success(f"Model ready! CV AUC = {auc:.2f}")
    return model, coef_dict, auc

# -------------------------------------------------------------
# Sidebar - Patient inputs
# -------------------------------------------------------------
model, coef_dict, auc = train_and_get_model()

st.title("Input Summary")

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

# Diabetes hint: append a clear phrase if glucose >= 126
if glucose >= 126:
    note = (note + " diabetes suspected by glucose >= 126 mg/dL").strip()

# One-row DataFrame for prediction
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

# Predict probability of 10-year CVD (class 1)
proba = model.predict_proba(row)[0][1]
risk_pct = 100.0 * proba

# Risk bucket mapping (tweakable)
if proba >= 0.60:
    verdict = ("Extremely High — Urgent referral", "danger")
elif proba >= 0.15:
    verdict = ("High — Start treatment", "warning")
elif proba >= 0.07:
    verdict = ("Moderate — Consider meds", "warning")
else:
    verdict = ("Low — Lifestyle focus", "success")

# -------------------------------------------------------------
# Layout
# -------------------------------------------------------------
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("Input Summary")
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

with col_right:
    st.subheader("CVD 10-Year Risk")
    risk_style = {
        "danger": "<span style='color:#B00020;font-weight:800;font-size:48px'>{:.1f}%</span>",
        "warning": "<span style='color:#E67E22;font-weight:800;font-size:48px'>{:.1f}%</span>",
        "success": "<span style='color:#2E7D32;font-weight:800;font-size:48px'>{:.1f}%</span>",
    }[verdict[1]]
    st.markdown(risk_style.format(risk_pct), unsafe_allow_html=True)
    st.caption(verdict[0])

    st.subheader("Risk Factor Impact")

    # Because we didn't scale binary columns, these are true 0->1 odds ratios
    smoke_or = float(np.exp(coef_dict.get('smoke', 0.0)))
    fhx_or = float(np.exp(coef_dict.get('family_hx', 0.0)))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Smoking", f"{smoke_or:.1f}× risk" if smoke else "No impact")
    with c2:
        st.metric("Family History", f"{fhx_or:.1f}× risk" if family_hx else "No impact")

    st.caption("Odds multipliers reflect model effects; 'No impact' means this factor is not present for the current patient.")

with st.expander("About This App"):
    st.markdown(
        f"""
        **Model:** TF‑IDF (clinical note) + Standardized vitals + Logistic Regression (`balanced`).
        **CV AUC:** {auc:.2f} on the attached dataset using 5‑fold Stratified CV.

        **Important:** This tool is for educational purposes and decision support only — not a diagnostic device. 
        Discuss any concerns with a licensed clinician.
        """
    )

st.markdown(
    """
    <hr/>
    <div style='font-size:12px;color:#6b6b6b'>
    By Howard Nguyen, PhD, 2025. Developed with TF‑IDF + Logistic Regression | CV AUC dynamically computed | 
    Smoking and Family Hx odds multipliers reflect true 0→1 effects (binary features are not scaled).
    </div>
    """,
    unsafe_allow_html=True
)
