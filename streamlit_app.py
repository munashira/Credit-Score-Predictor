import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="CIBIL Score Predictor", page_icon="💳", layout="wide")

# -------------------------------
# Custom CSS (Premium Fintech UI + Top Alignment Fix)
# -------------------------------
st.markdown("""
<style>

/* Import premium Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Full background */
.stApp {
    background: radial-gradient(circle at top left, rgba(59,130,246,0.35), transparent 60%),
                radial-gradient(circle at bottom right, rgba(168,85,247,0.35), transparent 60%),
                linear-gradient(135deg, #0b1220, #111827);
    background-attachment: fixed;
    color: white;
}

/* REMOVE STREAMLIT DEFAULT HEADER SPACE */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

/* REMOVE STREAMLIT DEFAULT TOP PADDING */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* FIX MAIN CONTENT ALIGNMENT */
div[data-testid="stAppViewContainer"] > .main {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Sidebar styling + alignment fix */
section[data-testid="stSidebar"] {
    width: 360px !important;
    min-width: 360px !important;
    max-width: 360px !important;

    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.12);

    padding-top: 0rem !important;
    margin-top: 0rem !important;
    padding-left: 20px;
    padding-right: 20px;
    padding-bottom: 20px;
}

/* Sidebar internal content spacing fix */
section[data-testid="stSidebar"] > div {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Sidebar text fix */
section[data-testid="stSidebar"] * {
    color: white !important;
    font-size: 15px;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #e0e7ff !important;
    font-weight: 700;
}

/* Main title */
.main-title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0px;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-title {
    font-size: 18px;
    text-align: center;
    color: rgba(255,255,255,0.75);
    margin-bottom: 40px;
    font-weight: 400;
}

/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 22px;
    padding: 25px;
    backdrop-filter: blur(22px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.55);
    margin-bottom: 20px;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.9), rgba(168,85,247,0.9));
    border-radius: 25px;
    padding: 35px;
    text-align: center;
    color: white;
    font-size: 52px;
    font-weight: 900;
    box-shadow: 0px 10px 45px rgba(0,0,0,0.6);
}

/* Score label */
.score-label {
    font-size: 16px;
    font-weight: 400;
    color: rgba(255,255,255,0.9);
    margin-top: 8px;
}

/* Tips box */
.tip-box {
    background: rgba(255, 255, 255, 0.06);
    border-left: 5px solid #fb7185;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 12px;
    font-size: 15px;
    color: rgba(255,255,255,0.88);
    box-shadow: 0px 5px 20px rgba(0,0,0,0.35);
}

/* Good box */
.good-box {
    background: rgba(34, 197, 94, 0.14);
    border-left: 5px solid #22c55e;
    padding: 15px;
    border-radius: 15px;
    font-size: 15px;
    color: rgba(255,255,255,0.92);
    box-shadow: 0px 5px 20px rgba(0,0,0,0.35);
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.55);
    margin-top: 40px;
    font-size: 14px;
}

/* Headings color */
h1, h2, h3, h4, h5 {
    color: white !important;
    font-weight: 700 !important;
}

/* Paragraph text */
p, div {
    color: rgba(255,255,255,0.85);
}

/* Slider labels */
label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 500 !important;
}

/* Slider value numbers */
div[data-testid="stSlider"] div {
    color: white !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(to right, #3b82f6, #a855f7);
    color: white;
    font-size: 17px;
    font-weight: 700;
    padding: 14px;
    border-radius: 16px;
    border: none;
    width: 100%;
    transition: 0.3s;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.45);
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #2563eb, #9333ea);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<div class="main-title">💳 CIBIL Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A premium ML-based credit score prediction dashboard with SHAP Explainability ✨</div>', unsafe_allow_html=True)

# -------------------------------
# Tips Function
# -------------------------------
def get_tip(feature):
    tips = {
        "Payment_History": "Pay EMIs & credit card bills on time consistently.",
        "Credit_Utilization": "Keep credit utilization below 30%.",
        "Credit_Age": "Maintain old accounts, avoid closing credit history.",
        "Number_of_Accounts": "Avoid opening many accounts quickly.",
        "Hard_Inquiries": "Limit loan/credit card applications in short time.",
        "Debt_to_Income_Ratio": "Reduce debt or increase income for better ratio."
    }
    return tips.get(feature, "Maintain financial discipline.")

# -------------------------------
# Model Training (Cached)
# -------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("synthetic_cibil_scores.csv")

    X = df[['Payment_History', 'Credit_Utilization', 'Credit_Age',
            'Number_of_Accounts', 'Hard_Inquiries', 'Debt_to_Income_Ratio']]
    y = df['CIBIL_Score']

    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)

    return model, explainer

# -------------------------------
# Predict Function
# -------------------------------
def predict_cibil(model, explainer, values):
    cols = ["Payment_History", "Credit_Utilization", "Credit_Age",
            "Number_of_Accounts", "Hard_Inquiries", "Debt_to_Income_Ratio"]

    df_input = pd.DataFrame([dict(zip(cols, values))])

    score = model.predict(df_input)[0]
    shap_values = explainer(df_input)

    return score, df_input, shap_values

# -------------------------------
# Suggestions Function
# -------------------------------
def get_suggestions(df_input, shap_values):
    suggestions = []
    shap_val = shap_values.values[0]
    feature_names = df_input.columns.tolist()

    for i, val in enumerate(shap_val):
        if val < 0:
            feature = feature_names[i]
            tip = get_tip(feature)
            suggestions.append(f"🔻 <b>{feature}</b> is reducing your score.<br>✅ Tip: {tip}")

    return suggestions

# -------------------------------
# Load Model
# -------------------------------
model, explainer = train_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Financial Inputs")
    st.write("Adjust sliders to simulate your credit profile:")

    ph = st.slider("📌 Payment History (%)", 0, 100, 80)
    cu = st.slider("📌 Credit Utilization (0.0 - 1.0)", 0.0, 1.0, 0.30)
    ca = st.slider("📌 Credit Age (Years)", 0, 50, 10)
    na = st.slider("📌 Number of Accounts", 1, 100, 8)
    hi = st.slider("📌 Hard Inquiries", 0, 10, 2)
    dr = st.slider("📌 Debt-to-Income Ratio (0.0 - 1.0)", 0.0, 1.0, 0.25)

    predict_btn = st.button("🚀 Predict My CIBIL Score")

# -------------------------------
# Layout Columns
# -------------------------------
col1, col2 = st.columns([1, 2])

# -------------------------------
# Prediction Output
# -------------------------------
if predict_btn:
    score, df_input, shap_values = predict_cibil(model, explainer, [ph, cu, ca, na, hi, dr])

    normalized_score = max(300, min(900, score))
    percent = int(((normalized_score - 300) / (900 - 300)) * 100)

    with col1:
        st.markdown(f"""
            <div class="score-card">
                {round(score)}
                <div class="score-label">Predicted CIBIL Score</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Score Strength Meter")
        st.progress(percent)

        if score >= 750:
            st.success("🔥 Excellent Score! You are financially strong.")
        elif score >= 650:
            st.warning("⚡ Good score, but improvement is possible.")
        else:
            st.error("🚨 Low score. Improve credit habits for better score.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Personalized Improvement Tips")

        suggestions = get_suggestions(df_input, shap_values)

        if suggestions:
            for tip in suggestions:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="good-box">✅ Great! No negative factors found.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 🔍 SHAP Explanation (Why this score?)")
        st.write("This graph shows which features increased or decreased your predicted score.")

        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="glass-card">
            <h2>👋 Welcome to the CIBIL Predictor</h2>
            <p style="color: rgba(255,255,255,0.75); font-size:17px;">
            Enter values from the sidebar and click <b>Predict My CIBIL Score</b> to view your credit score prediction and SHAP explanation.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown('<div class="footer">Made with ❤️ using Streamlit • XGBoost • SHAP | Premium Fintech UI</div>', unsafe_allow_html=True)