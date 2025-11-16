
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CVD Risk Predictor - Interactive", layout="wide")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.isotonic import IsotonicRegression

TARGET_FHX_OR = 2.5  # literature-aligned target OR for family history

@st.cache_resource
def train_and_get_model():
    st.info("Training model... (first time only, ~3 sec)")

    df = pd.read_csv("data_cvd_perfect_300.csv", encoding="latin-1", engine="python", on_bad_lines="skip").copy()
    y = df['cvd'].astype(int)
    X = df.drop('cvd', axis=1)

    num_cont = ['age','sys_bp','dia_bp','cholesterol','glucose','bmi']
    bin_cols = ['smoke','family_hx']
    text_col = 'note'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cont),
        ('bin', 'passthrough', bin_cols),
        ('text', TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

    model.fit(X, y)

    tfidf_names = model.named_steps['prep'].named_transformers_['text'].get_feature_names_out().tolist()
    feature_names = num_cont + bin_cols + tfidf_names
    coef = model.named_steps['clf'].coef_[0]
    coef_dict = dict(zip(feature_names, coef))

    fhx_or_model = float(np.exp(coef_dict.get('family_hx', 0.0)))
    smoke_or_model = float(np.exp(coef_dict.get('smoke', 0.0)))

    adjust_factor = TARGET_FHX_OR / fhx_or_model if fhx_or_model > 0 else 1.0

    # Fit isotonic calibration on adjusted probabilities
    base = model.predict_proba(X)[:, 1]
    odds = np.clip(base / (1 - base + 1e-12), 1e-9, 1e9)
    fhx = X['family_hx'].values.astype(int)
    adjusted_odds = odds * np.where(fhx == 1, adjust_factor, 1.0)
    adjusted_proba = adjusted_odds / (1 + adjusted_odds)

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(adjusted_proba, y)

    st.success(f"Model ready! CV AUC = {auc:.2f}")

    return {
        'model': model,
        'coef_dict': coef_dict,
        'auc': auc,
        'smoke_or': smoke_or_model,
        'fhx_or_model': fhx_or_model,
        'adjust_factor': adjust_factor,
        'calibrator': calibrator
    }

art = train_and_get_model()
model = art['model']
coef_dict = art['coef_dict']
auc = art['auc']
smoke_or = art['smoke_or']
fhx_or_model = art['fhx_or_model']
adjust_factor = art['adjust_factor']
calibrator = art['calibrator']

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
    note = st.text_area("Clinical Note", placeholder="e.g., gets short of breath walking uphill, smokes 1ppd", height=100) or default_note

smoke = 1 if smoking_label == "Yes" else 0
family_hx = 1 if family_hx_label == "Yes" else 0

if glucose >= 126:
    note = (note + " diabetes suspected by glucose >= 126 mg/dL").strip()

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

p_base = float(model.predict_proba(row)[0, 1])
odds = p_base / (1 - p_base + 1e-12)
odds_adj = odds * (adjust_factor if family_hx == 1 else 1.0)
p_adj = odds_adj / (1 + odds_adj)
p_cal = float(calibrator.predict([p_adj])[0])
risk_pct = 100.0 * p_cal

if p_cal >= 0.60:
    verdict = ("Extremely High — Urgent referral", "danger")
elif p_cal >= 0.15:
    verdict = ("High — Start treatment", "warning")
elif p_cal >= 0.07:
    verdict = ("Moderate — Consider meds", "warning")
else:
    verdict = ("Low — Lifestyle focus", "success")

col_left, col_right = st.columns([1, 1.2])

with col_left:
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
    fhx_or_display = TARGET_FHX_OR
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Smoking", f"{smoke_or:.1f}× risk" if smoke else "No impact")
    with c2:
        st.metric("Family History", f"{fhx_or_display:.1f}× risk" if family_hx else "No impact")
    st.caption("Family History odds effect is aligned to literature (~2–3×) via post‑hoc odds adjustment; probabilities are isotonic‑calibrated.")

with st.expander("About This App"):
    st.markdown(
        f"""
        **Model:** TF‑IDF (clinical note) + Standardized vitals + Logistic Regression (`balanced`).  
        **CV AUC:** {auc:.2f} (5‑fold).  
        **Calibration:** Family history OR aligned to **{TARGET_FHX_OR:.1f}×**; isotonic calibration for reliable probabilities.

        **Important:** Decision support only — not a diagnostic device.
        """
    )

st.markdown(
    """
    <hr/>
    <div style='font-size:12px;color:#6b6b6b'>
    By Howard Nguyen, PhD, 2025. Logistic baseline with literature-aligned FHx effect and isotonic calibration.
    </div>
    """, unsafe_allow_html=True
)
