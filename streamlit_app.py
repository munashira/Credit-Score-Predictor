import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="CIBIL Score Predictor", page_icon="📊", layout="centered")

st.title("📊 CIBIL Score Prediction")
st.write("Enter the following values to predict your CIBIL score:")

# -------------------------------
# Tips Dictionary
# -------------------------------
def get_tip(feature):
    tips = {
        "Payment_History": "Aim for a consistent payment history close to 100%.",
        "Credit_Utilization": "Try to keep credit utilization below 30%.",
        "Credit_Age": "Avoid closing old accounts to improve average credit age.",
        "Number_of_Accounts": "Avoid opening too many new accounts quickly.",
        "Hard_Inquiries": "Limit the number of credit inquiries over a short period.",
        "Debt_to_Income_Ratio": "Reduce debt or increase income to lower this ratio."
    }
    return tips.get(feature, "Maintain good financial discipline.")


# -------------------------------
# Train Model (Cached)
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
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)

    return model, explainer


# -------------------------------
# Prediction Function
# -------------------------------
def predict_cibil(model, explainer, values):
    cols = ["Payment_History", "Credit_Utilization", "Credit_Age",
            "Number_of_Accounts", "Hard_Inquiries", "Debt_to_Income_Ratio"]

    df_input = pd.DataFrame([dict(zip(cols, values))])

    score = model.predict(df_input)[0]

    shap_values = explainer(df_input)

    return score, df_input, shap_values


# -------------------------------
# Suggestions based on SHAP
# -------------------------------
def get_suggestions(df_input, shap_values):
    suggestions = []
    shap_val = shap_values.values[0]
    feature_names = df_input.columns.tolist()

    for i, val in enumerate(shap_val):
        if val < 0:
            feature = feature_names[i]
            tip = get_tip(feature)
            suggestions.append(f"🔻 **{feature}** is negatively affecting your score. Tip: {tip}")

    return suggestions


# -------------------------------
# Load Model and Explainer
# -------------------------------
model, explainer = train_model()

# -------------------------------
# Input Sliders
# -------------------------------
ph = st.slider("Payment History (0–100)%", 0, 100, 50)
cu = st.slider("Credit Utilization (0.0–1.0)", 0.0, 1.0, 0.5)
ca = st.slider("Credit Age (years)", 0, 50, 10)
na = st.slider("Number of Accounts", 1, 100, 10)
hi = st.slider("Hard Inquiries", 0, 10, 3)
dr = st.slider("Debt-to-Income Ratio (0.0–1.0)", 0.0, 1.0, 0.3)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):
    score, df_input, shap_values = predict_cibil(model, explainer, [ph, cu, ca, na, hi, dr])

    st.success(f"📈 Predicted CIBIL Score: **{round(score)}**")

    # -------------------------------
    # SHAP Waterfall Plot
    # -------------------------------
    st.write("### 🔍 Explanation of the Prediction (SHAP Waterfall Plot)")

    shap.plots.waterfall(shap_values[0], show=False)

    fig = plt.gcf()  # Get current SHAP plot figure
    st.pyplot(fig, clear_figure=True)

    # -------------------------------
    # Suggestions
    # -------------------------------
    suggestions = get_suggestions(df_input, shap_values)

    if suggestions:
        st.write("### 💡 Tips to Improve Your Score:")
        for tip in suggestions:
            st.markdown(tip)
    else:
        st.success("✅ Great job! All factors are contributing positively or neutrally.")