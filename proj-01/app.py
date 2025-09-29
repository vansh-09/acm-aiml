import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature names
model = joblib.load("lg_churn.pkl")
feature_names = joblib.load("feature_names.pkl")  # should be a list

st.title("📊 Churn Prediction Dashboard")

st.markdown("""
Simple Streamlit app to predict churn probability based on user input.
""")

# Input form
st.header("Input Customer Data")
input_data = {}
with st.form(key="churn_form"):
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        input_data[feature] = value
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_class = model.predict(input_df)[0]

    st.subheader("Prediction Results")
    st.write(f" Churn Probability: **{churn_prob:.2f}**")
    st.write(f" Churn Prediction: **{'Yes' if churn_class==1 else 'No'}**")