import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📞 Customer Churn Prediction")
st.markdown("Predict if a customer will leave your service")

# Generate synthetic data (no errors)
@st.cache_data
def create_data():
    np.random.seed(42)  # Corrected random seed
    size = 500
    data = {
        'tenure': np.random.randint(1, 72, size),
        'monthly_charges': np.random.uniform(20, 120, size).round(2),
        'total_usage': np.random.uniform(5, 500, size).round(2),
        'support_calls': np.random.randint(0, 10, size),
        'contract': np.random.choice(['Monthly', 'Yearly', 'Two-Year'], size),
        'churn': np.zeros(size)  # Proper initialization
    }
    
    # Correct churn probability calculation
    for i in range(size):
        churn_prob = (
            0.4 * (data['tenure'][i] < 6) +
            0.3 * (data['monthly_charges'][i] > 80) +
            0.2 * (data['support_calls'][i] > 3) -
            0.3 * (data['contract'][i] == 'Two-Year')
        )
        data['churn'][i] = 1 if np.random.random() < churn_prob else 0
    
    return pd.DataFrame(data)

df = create_data()

# Preprocess data
contract_mapping = {'Monthly': 0, 'Yearly': 1, 'Two-Year': 2}
df['contract'] = df['contract'].map(contract_mapping)

# Train-test split
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
total_usage = st.sidebar.slider("Total Usage (GB)", 5.0, 500.0, 150.0)
support_calls = st.sidebar.slider("Support Calls", 0, 10, 1)
contract = st.sidebar.selectbox("Contract Type", ["Monthly", "Yearly", "Two-Year"])

# Prediction
input_data = [[
    tenure,
    monthly_charges,
    total_usage,
    support_calls,
    contract_mapping[contract]
]]

churn_prob = model.predict_proba(input_data)[0][1]  # Correct probability extraction
prediction = model.predict(input_data)[0]  # Proper indexing

# Display results
st.subheader("🔮 Prediction Result")
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{churn_prob:.1%}")
with col2:
    if churn_prob > 0.7:
        st.error("🚨 High Risk of Churn")
    elif churn_prob > 0.4:
        st.warning("⚠️ Medium Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

# Model evaluation
st.subheader("📊 Model Performance")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"**Accuracy:** {accuracy:.2%}")

# Confusion matrix
st.write("**Confusion Matrix:**")
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Show sample data
if st.checkbox("Show sample data (10 random customers)"):
    st.dataframe(df.sample(10))
