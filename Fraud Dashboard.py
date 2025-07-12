import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("fraud_model.joblib")
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

st.title("💳 Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload a transaction CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

    # Preprocess
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    X_cluster = df[['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']].fillna(0)
    df['cluster'] = kmeans.predict(scaler.transform(X_cluster))

    model_features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
                      'hour', 'is_night', 'is_weekend', 'cluster']
    X = df[model_features]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df['fraud_pred'] = preds
    df['fraud_prob'] = probs

    st.write("🔍 Predicted Fraud Results")
    st.dataframe(df[['timestamp', 'amount', 'fraud_pred', 'fraud_prob']])

    st.write("📊 SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig = shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
