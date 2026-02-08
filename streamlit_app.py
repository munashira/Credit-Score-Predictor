import streamlit as st
import pandas as pd
import numpy as np


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3 {
            color: #00ffcc;
            text-align: center;
        }
        .stButton button {
            background: linear-gradient(90deg, #00ffcc, #0077ff);
            color: black;
            font-weight: bold;
            border-radius: 12px;
            height: 50px;
            width: 100%;
            font-size: 18px;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #0077ff, #00ffcc);
            color: white;
        }
        .result-box {
            background-color: #1f2937;
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #00ffcc;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1>💳 Credit Score Predictor</h1>", unsafe_allow_html=True)
st.write("### 📌 Enter the details below to predict your Credit Score")

st.markdown("---")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")   # Make sure model.pkl exists in GitHub
    return model

model = load_model()

# ------------------ INPUT FORM ------------------
st.subheader("📊 User Financial Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100, value=25)
    income = st.number_input("💰 Annual Income (in ₹)", min_value=10000, max_value=10000000, value=500000)
    loan_amount = st.number_input("🏦 Loan Amount (in ₹)", min_value=1000, max_value=10000000, value=200000)

with col2:
    credit_utilization = st.slider("📉 Credit Utilization (%)", 0, 100, 30)
    late_payments = st.number_input("⏳ Late Payments (Count)", min_value=0, max_value=50, value=2)
    credit_history = st.number_input("📅 Credit History (Years)", min_value=0, max_value=40, value=5)

st.markdown("---")

# ------------------ PREDICTION BUTTON ------------------
if st.button("🔮 Predict Credit Score"):
    try:
        # Convert inputs to model format
        input_data = np.array([[age, income, loan_amount, credit_utilization, late_payments, credit_history]])

        prediction = model.predict(input_data)[0]

        # Category based on score
        if prediction >= 750:
            status = "🟢 Excellent"
        elif prediction >= 650:
            status = "🟡 Good"
        elif prediction >= 550:
            status = "🟠 Average"
        else:
            status = "🔴 Poor"

        st.markdown(f"""
            <div class="result-box">
                📌 Predicted Credit Score: <span style="color:#00ffcc;">{prediction:.2f}</span> <br><br>
                ⭐ Credit Category: <span style="color:#ffd700;">{status}</span>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error occurred while predicting: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ by Munashira Farheen | Credit Score Predictor Project</p>", unsafe_allow_html=True)