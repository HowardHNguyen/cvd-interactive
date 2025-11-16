
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="CVD Risk Predictor - Interactive (Clinical Grade)", layout="wide")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier

# Optional dependencies
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# Try xgboost; fallback to GradientBoosting if unavailable
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

TARGET_FHX_OR = 2.5
SMOOTH_EPS = 1e-3
BLEND_ALPHA = 0.05  # small Bayesian shrink toward prevalence (no static floor)

@st.cache_resource
def load_data():
    # Prefer uploaded/official file; else use synthetic if present
    for path in ["data_cvd_perfect_300.csv", "data_ehr_500.csv"]:
        try:
            df = pd.read_csv(path)
            if set(["note","age","sys_bp","dia_bp","cholesterol","glucose","bmi","smoke","family_hx","cvd"]).issubset(df.columns):
                return df.copy()
        except Exception:
            pass
    st.stop()

@st.cache_resource
def train_models(df, model_choice="Stacking"):
    st.info("Training models... (~5–10s first run)")

    y = df["cvd"].astype(int)
    X = df.drop("cvd", axis=1)

    num_cont = ['age','sys_bp','dia_bp','cholesterol','glucose','bmi']
    bin_cols = ['smoke','family_hx']
    text_col = 'note'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cont),
        ('bin', 'passthrough', bin_cols),
        ('text', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    lr = Pipeline([('prep', preprocessor),
                   ('clf', LogisticRegression(C=0.6, class_weight='balanced', max_iter=1500, random_state=42))])

    if HAVE_XGB:
        gbm_model = Pipeline([('prep', preprocessor),
                              ('clf', XGBClassifier(
                                  n_estimators=250, max_depth=4, learning_rate=0.07,
                                  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                  objective='binary:logistic', eval_metric='logloss', random_state=42))])
    else:
        gbm_model = Pipeline([('prep', preprocessor),
                              ('clf', GradientBoostingClassifier(random_state=42))])

    if model_choice == "Logistic":
        base_model = lr
        tree_model = gbm_model
    elif model_choice == "GBM/XGBoost":
        base_model = gbm_model
        tree_model = gbm_model
    else:
        # Stacking (LR + GBM) with LR meta-learner
        estimators = [('lr', lr), ('gbm', gbm_model)]
        base_model = StackingClassifier(estimators=estimators,
                                        final_estimator=LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
                                        stack_method='auto', passthrough=False, n_jobs=None)
        tree_model = gbm_model

    # Train/test split for calibration & metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Fit base model
    base_model.fit(X_train, y_train)

    # Family history OR alignment requires access to LR coefficients; if not LR pipeline, approximate using logistic on train
    # We'll compute FHx OR from a separate LR fitted on the same features for alignment magnitude.
    lr_for_or = lr
    try:
        lr_for_or.fit(X_train, y_train)
        # Build names
        tfidf_names = lr_for_or.named_steps['prep'].named_transformers_['text'].get_feature_names_out().tolist()
        feature_names = num_cont + bin_cols + tfidf_names
        coef = lr_for_or.named_steps['clf'].coef_[0]
        coef_dict = dict(zip(feature_names, coef))
        smoke_or_model = float(np.exp(coef_dict.get('smoke', 0.0)))
        fhx_or_model = float(np.exp(coef_dict.get('family_hx', 0.0)))
    except Exception:
        smoke_or_model = np.nan
        fhx_or_model = TARGET_FHX_OR

    adjust_factor = TARGET_FHX_OR / fhx_or_model if (fhx_or_model and fhx_or_model>0) else 1.0

    # Build adjusted probs on train, fit isotonic; blend toward prevalence (prevents 0% but remains dynamic)
    p_train = base_model.predict_proba(X_train)[:,1]
    odds = np.clip(p_train/(1-p_train+1e-12), 1e-9, 1e9)
    fhx_train = X_train['family_hx'].values.astype(int)
    odds_adj = odds * np.where(fhx_train==1, adjust_factor, 1.0)
    p_adj = odds_adj/(1+odds_adj)
    p_adj = np.clip(p_adj, SMOOTH_EPS, 1-SMOOTH_EPS)

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_adj, y_train)

    # Evaluate
    p_test = base_model.predict_proba(X_test)[:,1]
    odds_te = np.clip(p_test/(1-p_test+1e-12), 1e-9, 1e9)
    fhx_te = X_test['family_hx'].values.astype(int)
    p_adj_te = (odds_te*np.where(fhx_te==1, adjust_factor,1.0))/(1+odds_te*np.where(fhx_te==1, adjust_factor,1.0))
    p_adj_te = np.clip(p_adj_te, SMOOTH_EPS, 1-SMOOTH_EPS)
    p_cal_te = calibrator.predict(p_adj_te)
    prevalence = y_train.mean()
    p_blend_te = (1-BLEND_ALPHA)*p_cal_te + BLEND_ALPHA*prevalence

    auc_pre = roc_auc_score(y_test, p_test)
    auc_post = roc_auc_score(y_test, p_blend_te)
    brier = brier_score_loss(y_test, p_blend_te)

    return {
        "base_model": base_model,
        "tree_model": tree_model,
        "calibrator": calibrator,
        "adjust_factor": adjust_factor,
        "smoke_or_model": smoke_or_model,
        "fhx_or_model": fhx_or_model,
        "prevalence": float(prevalence),
        "metrics": {"AUC_pre": float(auc_pre), "AUC_post": float(auc_post), "Brier": float(brier)},
        "preprocessor_info": {"num_cont": num_cont, "bin_cols": bin_cols, "text_col": text_col}
    }

def predict_with_calibration(model_pack, row_df):
    base_model = model_pack["base_model"]
    calibrator = model_pack["calibrator"]
    adjust_factor = model_pack["adjust_factor"]
    prevalence = model_pack["prevalence"]

    p = float(base_model.predict_proba(row_df)[0,1])
    odds = p/(1-p+1e-12)
    if int(row_df.iloc[0]["family_hx"])==1:
        odds *= adjust_factor
    p_adj = odds/(1+odds)
    p_adj = float(np.clip(p_adj, SMOOTH_EPS, 1-SMOOTH_EPS))
    p_cal = float(calibrator.predict([p_adj])[0])
    p_blend = (1-BLEND_ALPHA)*p_cal + BLEND_ALPHA*prevalence
    return p_blend

def compute_counterfactuals(model_pack, row_df):
    deltas = {
        "SBP -10 mmHg": {"sys_bp": -10},
        "DBP -5 mmHg": {"dia_bp": -5},
        "Chol -20 mg/dL": {"cholesterol": -20},
        "BMI -2": {"bmi": -2},
        "Quit Smoking": {"smoke": -1},  # if 1 -> 0
    }
    base_p = predict_with_calibration(model_pack, row_df)
    out = []
    for name, change in deltas.items():
        cf = row_df.copy()
        for k,v in change.items():
            if k=="smoke":
                cf[k] = max(0, int(cf[k]) + v)
            else:
                cf[k] = float(cf[k]) + v
        cf_p = predict_with_calibration(model_pack, cf)
        out.append((name, base_p, cf_p, (base_p - cf_p)))
    out.sort(key=lambda x: x[3], reverse=True)
    return out

def build_pdf(row_df, risk_pct, impacts, metrics):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height-1*inch, "CVD 10-Year Risk Report")

    c.setFont("Helvetica", 11)
    y = height-1.3*inch
    for k,v in row_df.iloc[0].items():
        c.drawString(1*inch, y, f"{k}: {v}")
        y -= 0.2*inch

    y -= 0.2*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y, f"Predicted Risk: {risk_pct:.1f}%")
    y -= 0.3*inch

    c.setFont("Helvetica", 11)
    c.drawString(1*inch, y, "Counterfactuals (largest risk reduction first):")
    y -= 0.2*inch
    for name, base_p, cf_p, delta in impacts[:5]:
        c.drawString(1.2*inch, y, f"{name}: {100*cf_p:.1f}%  (Δ {100*delta:.1f}%)")
        y -= 0.18*inch

    y -= 0.2*inch
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, y, f"Model AUC (post-cal): {metrics['AUC_post']:.2f} | Brier: {metrics['Brier']:.3f}")
    y -= 0.18*inch
    c.drawString(1*inch, y, "Note: Decision support only — not for diagnosis.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --------------------- APP ----------------------
df = load_data()

st.sidebar.success(f"Dataset loaded • rows: {len(df)} • prevalence: {df['cvd'].mean():.2f}")
model_choice = st.sidebar.selectbox("Model", ["Stacking", "GBM/XGBoost", "Logistic"])
models = train_models(df, model_choice=model_choice)

# Sidebar inputs
st.sidebar.header("Patient Profile")
age = st.sidebar.slider("Age", 20, 90, 50, step=1)
sys_bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 200, 120, step=1)
dia_bp = st.sidebar.slider("Diastolic BP (mmHg)", 50, 120, 80, step=1)
chol = st.sidebar.slider("Cholesterol (mg/dL)", 120, 360, 180, step=1)
glu = st.sidebar.slider("Glucose (mg/dL)", 70, 250, 100, step=1)
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
smoke = 1 if st.sidebar.selectbox("Smoking", ["No","Yes"])=="Yes" else 0
fhx = 1 if st.sidebar.selectbox("Family History of CVD", ["No","Yes"])=="Yes" else 0
note = st.sidebar.text_area("Clinical Note", "no symptoms reported", height=100)

if glu >= 126 and "diabetes suspected" not in note:
    note = (note + " diabetes suspected by glucose >= 126 mg/dL").strip()

row = pd.DataFrame([{
    "note": note, "age": age, "sys_bp": sys_bp, "dia_bp": dia_bp,
    "cholesterol": chol, "glucose": glu, "bmi": bmi,
    "smoke": smoke, "family_hx": fhx
}])

# Prediction
p = predict_with_calibration(models, row)
risk_pct = 100*p

# Risk category
if p >= 0.60:
    verdict = ("Extremely High — Urgent referral", "danger")
elif p >= 0.15:
    verdict = ("High — Start treatment", "warning")
elif p >= 0.07:
    verdict = ("Moderate — Consider meds", "warning")
else:
    verdict = ("Low — Lifestyle focus", "success")

# Layout
left, right = st.columns([1, 1.3])
with left:
    st.title("Input Summary")
    st.markdown(f"""
- **Age:** {age}
- **BP:** {sys_bp}/{dia_bp} mmHg
- **Cholesterol:** {chol} mg/dL
- **Glucose:** {glu} mg/dL
- **BMI:** {bmi:.1f}
- **Smoker:** {"Yes" if smoke else "No"}
- **Family Hx:** {"Yes" if fhx else "No"}
""")

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
    smoke_or = models["smoke_or_model"]
    fhx_aligned = TARGET_FHX_OR
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Smoking", f"{smoke_or:.1f}× risk" if smoke_or==smoke_or and smoke else "No impact")
    with c2:
        st.metric("Family History", f"{fhx_aligned:.1f}× risk" if fhx else "No impact")
    st.caption("Family history effect aligned to literature (~2–3×) via odds adjustment; probabilities are isotonic‑calibrated and blended slightly toward cohort prevalence (no static floor).")

# Counterfactuals
st.subheader("What Lowers Risk Most? (Counterfactuals)")
cf = compute_counterfactuals(models, row)
for name, base_p, cf_p, delta in cf[:5]:
    st.write(f"**{name}** → {100*cf_p:.1f}%  (Δ {100*delta:.1f}%)")

# SHAP explanations (if available and using tree model)
if SHAP_OK:
    st.subheader("SHAP Explanations")
    try:
        # Build a small sample for speed
        sample = df.sample(200, random_state=42)
        explainer = shap.Explainer(models["tree_model"])
        shap_values = explainer(sample)
        st.pyplot(shap.plots.bar(shap_values, max_display=10, show=False))
    except Exception as e:
        st.info(f"SHAP not available in this environment: {e}")

# Comparison: simple "Traditional-like" score vs ML
st.subheader("Comparison: Traditional-like Score vs ML")
# A tiny hand-tuned logistic as a demo traditional score (NOT medical advice)
w = dict(age=0.03, sys_bp=0.012, dia_bp=0.004, cholesterol=0.004, glucose=0.003, bmi=0.03, smoke=0.9, family_hx=0.9, intercept=-14.0)
lin = (w['intercept'] + w['age']*age + w['sys_bp']*sys_bp + w['dia_bp']*dia_bp +
       w['cholesterol']*(chol/10) + w['glucose']*(glu/10) + w['bmi']*(bmi/2) + w['smoke']*smoke + w['family_hx']*fhx)
trad_p = 1/(1+np.exp(-lin))
st.write(f"**Traditional-like score:** {100*trad_p:.1f}%  |  **ML score (calibrated):** {risk_pct:.1f}%")

# Downloadable PDF
st.subheader("Download Report")
impacts = cf
pdf_buffer = build_pdf(row, risk_pct, impacts, models["metrics"])
st.download_button("Download PDF Report", data=pdf_buffer, file_name="cvd_risk_report.pdf", mime="application/pdf")

with st.expander("About This App"):
    st.markdown(f"""
**Models:** Logistic Regression, {'XGBoost' if HAVE_XGB else 'Gradient Boosting'}, and Stacking (LR+GBM).  
**Text:** TF‑IDF bigrams (200 features).  
**Calibration:** Family history aligned to {TARGET_FHX_OR:.1f}× OR; isotonic calibration; tiny Bayesian blend to cohort prevalence.  
**Metrics:** AUROC pre/post calibration and Brier on holdout.  
**Important:** Decision support only — not a diagnostic device.
""")
