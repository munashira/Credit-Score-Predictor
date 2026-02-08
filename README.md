# 💳 CIBIL Score Prediction using Machine Learning (XGBoost + SHAP + Streamlit)

🚀 A modern and interactive **CIBIL Score Prediction Web Application** built using **Streamlit**, **XGBoost Regression**, and **SHAP Explainable AI**.

This project predicts a user's **CIBIL Score** based on important credit-related parameters like payment history, credit utilization, credit age, debt ratio, etc.  
It also provides a detailed **SHAP Waterfall Explanation Plot** to show how each feature affected the predicted score.

---

## 🌟 Project Highlights

✅ Predicts **CIBIL Score** instantly  
✅ Clean and interactive UI using **Streamlit sliders**  
✅ Uses **XGBoost Regressor** for strong prediction performance  
✅ SHAP Explanation (Waterfall Plot) for transparency  
✅ Provides **Tips to Improve CIBIL Score** automatically  
✅ Streamlit caching avoids retraining again and again  
✅ Simple, lightweight, fast and user-friendly  

---

## 🖥️ Output Preview

📌 **Predicted CIBIL Score**  
📌 **SHAP Waterfall Explanation Graph**  
📌 **Suggestions to improve score**

Example Output:

- 📈 Predicted CIBIL Score: **604**
- 🔍 SHAP Waterfall Plot shows which factors increased or decreased the score
- 💡 Tips displayed if any feature contributes negatively

---

## 🎯 Input Parameters

| Feature | Description |
|--------|-------------|
| Payment History | Timely repayment percentage (0–100%) |
| Credit Utilization | Ratio of credit used (0.0–1.0) |
| Credit Age | Total credit history age (years) |
| Number of Accounts | Total credit accounts held |
| Hard Inquiries | Number of credit inquiries |
| Debt-to-Income Ratio | Debt compared to income (0.0–1.0) |

---

## 🛠️ Tech Stack Used

| Technology | Purpose |
|----------|---------|
| **Python** | Core programming |
| **Streamlit** | Web UI framework |
| **XGBoost** | ML Regression Model |
| **SHAP** | Explainable AI |
| **Pandas / NumPy** | Data handling |
| **Matplotlib** | Plot visualization |
| **Scikit-learn** | Dataset split + evaluation |

---

## 📂 Project Structure

```bash
CIBIL-Score-Prediction/
│
├── main.py                     # Streamlit UI
├── model.py                    # Model training + prediction logic
├── synthetic_cibil_scores.csv  # Dataset used
├── requirements.txt            # Dependencies
└── README.md                   # Documentation