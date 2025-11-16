
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="CVD Risk Predictor - Interactive (Clinical Grade)", layout="wide")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

TARGET_FHX_OR = 2.5
SMOOTH_EPS = 1e-6

@st.cache_resource
def load_data():
    for path in ["data_cvd_perfect_300.csv", "data_ehr_500.csv"]:
        try:
            df = pd.read_csv(path)
            need = {"note","age","sys_bp","dia_bp","cholesterol","glucose","bmi","smoke","family_hx","cvd"}
            if need.issubset(df.columns):
                df['smoke'] = pd.to_numeric(df['smoke'], errors='coerce').fillna(0).astype(int)
                df['family_hx'] = pd.to_numeric(df['family_hx'], errors='coerce').fillna(0).astype(int)
                for c in ["age","sys_bp","dia_bp","cholesterol","glucose","bmi"]:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df['cvd'] = pd.to_numeric(df['cvd'], errors='coerce').astype(int)
                df = df.dropna(subset=["note","age","sys_bp","dia_bp","cholesterol","glucose","bmi","smoke","family_hx","cvd"]).copy()
                return df
        except Exception:
            pass
    st.error("No valid dataset found in app root.")
    st.stop()

def make_preprocessor():
    num_cont = ['age','sys_bp','dia_bp','cholesterol','glucose','bmi']
    bin_cols = ['smoke','family_hx']
    text_col = 'note'
    bin_pipe = Pipeline([('to_float', FunctionTransformer(lambda z: pd.DataFrame(z).astype(float)))])
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cont),
        ('bin', bin_pipe, bin_cols),
        ('text', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1,2)), text_col)
    ])
    return preprocessor, num_cont, bin_cols, text_col

@st.cache_resource
def train_all_models(df):
    y = df['cvd'].astype(int)
    X = df.drop('cvd', axis=1)

    preproc, num_cont, bin_cols, text_col = make_preprocessor()

    lr_base = Pipeline([('prep', preproc),
                        ('clf', LogisticRegression(C=0.6, class_weight='balanced', max_iter=1500, random_state=42))])

    if HAVE_XGB:
        pos = y.sum()
        neg = len(y) - pos
        spw = (neg / max(pos, 1))
        gbm_base = Pipeline([('prep', preproc),
                             ('clf', XGBClassifier(
                                 n_estimators=500, max_depth=4, learning_rate=0.06,
                                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                 objective='binary:logistic', eval_metric='logloss',
                                 scale_pos_weight=spw, random_state=42))])
    else:
        gbm_base = Pipeline([('prep', preproc),
                             ('clf', GradientBoostingClassifier(
                                 n_estimators=400, max_depth=3, learning_rate=0.06, random_state=42))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cal = CalibratedClassifierCV(lr_base, method='sigmoid', cv=cv).fit(X, y)
    gbm_cal = CalibratedClassifierCV(gbm_base, method='sigmoid', cv=cv).fit(X, y)

    lr_cv_probs = cross_val_predict(lr_cal, X, y, cv=cv, method='predict_proba')[:,1]
    gbm_cv_probs = cross_val_predict(gbm_cal, X, y, cv=cv, method='predict_proba')[:,1]

    lr_auc = float(roc_auc_score(y, lr_cv_probs))
    lr_brier = float(brier_score_loss(y, lr_cv_probs))
    gbm_auc = float(roc_auc_score(y, gbm_cv_probs))
    gbm_brier = float(brier_score_loss(y, gbm_cv_probs))

    w_lr = max(lr_auc, 1e-6)
    w_gbm = max(gbm_auc, 1e-6)
    s = w_lr + w_gbm
    w_lr /= s
    w_gbm /= s

    return {
        'lr_cal': lr_cal,
        'gbm_cal': gbm_cal,
        'weights': (float(w_lr), float(w_gbm)),
        'preproc_info': {'num_cont': num_cont, 'bin_cols': bin_cols, 'text_col': text_col},
        'metrics': {'lr': {'AUC': lr_auc, 'Brier': lr_brier},
                    'gbm': {'AUC': gbm_auc, 'Brier': gbm_brier}}
    }

def apply_fhx_alignment(p, family_hx, target_or=TARGET_FHX_OR):
    p = float(np.clip(p, SMOOTH_EPS, 1 - SMOOTH_EPS))
    if int(family_hx) != 1:
        return p
    odds = p / (1 - p)
    odds *= target_or
    p_final = odds / (1 + odds)
    return float(np.clip(p_final, SMOOTH_EPS, 1 - SMOOTH_EPS))

def predict_model(models, Xrow, which):
    if which == 'Logistic':
        p = float(models['lr_cal'].predict_proba(Xrow)[0,1])
    elif which == 'GBM/XGBoost':
        p = float(models['gbm_cal'].predict_proba(Xrow)[0,1])
    else:
        p_lr = float(models['lr_cal'].predict_proba(Xrow)[0,1])
        p_gbm = float(models['gbm_cal'].predict_proba(Xrow)[0,1])
        w_lr, w_gbm = models['weights']
        p = w_lr * p_lr + w_gbm * p_gbm
    return p

def predict_with_pipeline(models, row_df, which_model):
    p_cal = predict_model(models, row_df, which_model)
    p_final = apply_fhx_alignment(p_cal, row_df.iloc[0]['family_hx'])
    return p_cal, p_final

def compute_counterfactuals(models, row_df, which_model):
    deltas = {
        "SBP -10 mmHg": {"sys_bp": -10},
        "DBP -5 mmHg": {"dia_bp": -5},
        "Chol -20 mg/dL": {"cholesterol": -20},
        "BMI -2": {"bmi": -2},
        "Quit Smoking": {"smoke": -1},
    }
    _, base_p = predict_with_pipeline(models, row_df, which_model)
    out = []
    for name, change in deltas.items():
        cf = row_df.copy()
        for k,v in change.items():
            if k=="smoke":
                cf[k] = max(0, int(cf[k]) + v)
            else:
                cf[k] = float(cf[k]) + v
        _, cf_p = predict_with_pipeline(models, cf, which_model)
        out.append((name, base_p, cf_p, (base_p - cf_p)))
    out.sort(key=lambda x: x[3], reverse=True)
    return out

def build_pdf(row_df, risk_pct, impacts, metrics, which_model):
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
    c.drawString(1*inch, y, f"Predicted Risk: {risk_pct:.1f}%  (Model: {which_model})")
    y -= 0.3*inch

    c.setFont("Helvetica", 11)
    c.drawString(1*inch, y, "Counterfactuals (largest risk reduction first):")
    y -= 0.2*inch
    for name, base_p, cf_p, delta in impacts[:5]:
        c.drawString(1.2*inch, y, f"{name}: {100*cf_p:.1f}%  (Δ {100*delta:.1f}%)")
        y -= 0.18*inch

    y -= 0.2*inch
    c.setFont("Helvetica", 10)
    def _key(which_model):
        if which_model == 'Logistic': return 'lr'
        if which_model == 'GBM/XGBoost': return 'gbm'
        return 'lr'
    try:
        auc = metrics[_key(which_model)]['AUC']
        br = metrics[_key(which_model)]['Brier']
        c.drawString(1*inch, y, f"Model AUC: {auc:.2f} | Brier: {br:.3f}")
    except Exception:
        pass
    y -= 0.18*inch
    c.drawString(1*inch, y, "Note: Decision support only — not a diagnostic device.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --------------------- APP ----------------------
df = load_data()
st.sidebar.success(f"Dataset loaded • rows: {len(df)} • prevalence: {df['cvd'].mean():.2f}")
model_choice = st.sidebar.selectbox("Model", ["Stacking (Weighted)", "GBM/XGBoost", "Logistic"])

models = train_all_models(df)

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
    "cholesterol": chol, "glucose": glu, "bmi": bmi, "smoke": smoke, "family_hx": fhx
}])

# Prediction
which = 'Logistic' if model_choice=='Logistic' else ('GBM/XGBoost' if model_choice=='GBM/XGBoost' else 'Stacking')
p_cal, p = predict_with_pipeline(models, row, which)
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

    st.subheader("Model & Calibration")
    if which == 'Stacking':
        w_lr, w_gbm = models['weights']
        st.write(f"Blended (calibrated) probs = {w_lr:.2f}×LR + {w_gbm:.2f}×GBM/XGB")
    else:
        st.write(f"Model: {which} (Platt calibrated)")

# Counterfactuals
st.subheader("What Lowers Risk Most? (Counterfactuals)")
cf = compute_counterfactuals(models, row, which)
for name, base_p, cf_p, delta in cf[:5]:
    st.write(f"**{name}** → {100*cf_p:.1f}%  (Δ {100*delta:.1f}%)")

# SHAP explanations (if available)
st.subheader("SHAP Explanations")
if SHAP_OK:
    try:
        sample = df.sample(min(300, len(df)), random_state=42)
        preproc, num_cont, bin_cols, text_col = make_preprocessor()
        if HAVE_XGB:
            tree_model = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective='binary:logistic', eval_metric='logloss', random_state=42
            )
        else:
            tree_model = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.06, random_state=42)
        X_shap = sample.drop('cvd', axis=1)
        y_shap = sample['cvd'].astype(int)
        Xp = preproc.fit_transform(X_shap)
        tree_model.fit(Xp, y_shap)
        explainer = shap.Explainer(tree_model, Xp)
        shap_values = explainer(Xp)
        st.pyplot(shap.plots.bar(shap_values, max_display=12, show=False))
    except Exception as e:
        st.info(f"SHAP not available in this environment: {e}")
else:
    st.info("SHAP not installed; skip visual explanation.")

# Comparison: simple "Traditional-like" score vs ML
st.subheader("Comparison: Traditional-like Score vs ML")
w_demo = dict(age=0.03, sys_bp=0.012, dia_bp=0.004, cholesterol=0.004, glucose=0.003, bmi=0.03, smoke=0.9, family_hx=0.9, intercept=-14.0)
lin = (w_demo['intercept'] + w_demo['age']*age + w_demo['sys_bp']*sys_bp + w_demo['dia_bp']*dia_bp +
       w_demo['cholesterol']*(chol/10) + w_demo['glucose']*(glu/10) + w_demo['bmi']*(bmi/2) + w_demo['smoke']*smoke + w_demo['family_hx']*fhx)
trad_p = 1/(1+np.exp(-lin))
st.write(f"**Traditional-like score:** {100*trad_p:.1f}%  |  **ML score (calibrated+FHx):** {risk_pct:.1f}%  |  **Model:** {which}")

# Debug expander
with st.expander("Debug: probabilities"):
    st.write({"calibrated_p": round(p_cal, 6), "final_after_FHx": round(p, 6)})

# Downloadable PDF
st.subheader("Download Report")
impacts = cf
def _make_pdf():
    return build_pdf(row, risk_pct, impacts, models['metrics'], which)
try:
    pdf_buffer = _make_pdf()
    st.download_button("Download PDF Report", data=pdf_buffer, file_name="cvd_risk_report.pdf", mime="application/pdf")
except Exception as e:
    st.info(f"PDF not available: {e}")

with st.expander("About This App"):
    st.markdown(f"""
**Models:** Logistic Regression, {'XGBoost' if HAVE_XGB else 'Gradient Boosting'}, and **Stacking (Weighted Blend)** of calibrated LR & GBM.  
**Text:** TF‑IDF bigrams (200 features).  
**Calibration:** **Platt** via 5‑fold Stratified CV; **Family History** aligned to **{TARGET_FHX_OR:.1f}×** AFTER calibration (odds space).  
**Validation:** Blend weights are proportional to CV AUCs.  
**Important:** Decision support only — not a diagnostic device.
""")
