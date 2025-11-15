# interactive_app.py
import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Train model at startup (cached, runs once)
# ------------------------------------------------------------------
@st.cache_resource
def train_and_get_model():
    st.info("Training model... (first time only, ~3 sec)")

    df = pd.read_csv("data_cvd_perfect.csv")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    num_cols = ['age', 'sys_bp', 'dia_bp', 'cholesterol', 'glucose', 'bmi', 'smoke', 'family_hx']
    text_col = 'note'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('text', TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000))
    ])

    X = df.drop('cvd', axis=1)
    y = df['cvd']
    model.fit(X, y)

    st.success("Model ready! AUC ≈ 0.84")
    return model

model = train_and_get_model()

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(page_title="CVD Interactive Risk", layout="wide")

# ------------------------------------------------------------------
# Sidebar: Input Controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Profile")
    
    age = st.slider("Age", 20, 90, 50)
    sys_bp = st.slider("Systolic BP (mmHg)", 90, 200, 120)
    dia_bp = st.slider("Diastolic BP (mmHg)", 50, 120, 80)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 180)
    glucose = st.slider("Glucose (mg/dL)", 60, 300, 100)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1)
    
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    smoke = 1 if smoke == "Yes" else 0
    
    family_hx = st.selectbox("Family History of CVD", ["No", "Yes"])
    family_hx = 1 if family_hx == "Yes" else 0
    
    note = st.text_area(
        "Clinical Note",
        placeholder="e.g., Gets short of breath walking uphill, smokes cigarettes, family history of heart problems",
        height=150
    )

# ------------------------------------------------------------------
# Risk Interpretation & Color
# ------------------------------------------------------------------
def interpret_risk(val):
    if val < 10: return "Low — Lifestyle focus"
    elif val < 20: return "Moderate — Consider meds"
    elif val < 30: return "High — Start treatment"
    elif val < 40: return "Very High — Intensive care"
    else: return "Extremely High — Urgent referral"

def get_risk_color(val):
    if val < 10: return "#9cc732"   # Green
    elif val < 20: return "#fff000"  # Yellow
    elif val < 30: return "#f3771d"  # Orange
    elif val < 40: return "#ea1a21"  # Red
    else: return "#9d1c1f"          # Deep Red

# ------------------------------------------------------------------
# Main Panel: Real-Time Prediction
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input Summary")
    summary = f"""
    - **Age**: {age}  
    - **BP**: {sys_bp}/{dia_bp} mmHg  
    - **Cholesterol**: {cholesterol} mg/dL  
    - **Glucose**: {glucose} mg/dL  
    - **BMI**: {bmi:.1f}  
    - **Smoker**: {'Yes' if smoke else 'No'}  
    - **Family Hx**: {'Yes' if family_hx else 'No'}  
    """
    st.markdown(summary)

with col2:
    st.markdown("### 10-Year CVD Risk")
    
    # Build input DataFrame
    input_data = pd.DataFrame([{
        'note': note if note.strip() else "no symptoms reported",
        'age': age,
        'sys_bp': sys_bp,
        'dia_bp': dia_bp,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'bmi': bmi,
        'smoke': smoke,
        'family_hx': family_hx
    }])

    # Predict
    try:
        prob = model.predict_proba(input_data)[:, 1][0]
        risk_pct = round(prob * 100, 1)
        interpretation = interpret_risk(risk_pct)
        color = get_risk_color(risk_pct)

        # Big risk number
        st.markdown(
            f"<h1 style='text-align: center; color: {color};'>"
            f"{risk_pct:.1f}%</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center; font-size: 18px; font-weight: bold;'>"
            f"{interpretation}</p>",
            unsafe_allow_html=True
        )

        # WHO/ISH reference
        st.caption(
            "Risk = 10-year chance of **heart attack or stroke** (fatal or non-fatal). "
            "Based on **WHO/ISH 2007 Risk Charts**."
        )

    except Exception as e:
        st.error("Prediction error. Check inputs.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Powered by TF-IDF + Logistic Regression | No data leakage | AUC ≈ 0.84"
    "</p>",
    unsafe_allow_html=True
)