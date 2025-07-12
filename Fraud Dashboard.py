import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

st.set_page_config(page_title="Fraud Clustering & Analysis", layout="wide")
st.title("💳 Fraud Detection, Clustering & Exploratory Dashboard")

# Upload
uploaded_file = st.file_uploader("📁 Upload your Fraud_Detection.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

    # --- Preprocessing ---
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)

    features_to_cluster = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
    X = df[features_to_cluster].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # --- Cluster Summary ---
    st.subheader("🔍 Cluster Fraud Summary")
    cluster_summary = df.groupby('cluster').agg(
        total_transactions=('transaction_id', 'count'),
        fraud_transactions=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).reset_index()
    cluster_summary['fraud_rate'] = (cluster_summary['fraud_rate'] * 100).round(2)
    st.dataframe(cluster_summary)

    # --- Top Fraud Locations ---
    st.subheader("📍 Top Locations for Fraud")
    if 'location' in df.columns:
        city_counts = df[df['is_fraud'] == True]['location'].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax1)
        ax1.set_title("Top Locations for Fraud")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

    # --- Cluster-wise Fraud Count ---
    st.subheader("🧮 Fraud vs Non-Fraud by Cluster")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='cluster', hue='is_fraud', palette='Set2', ax=ax2)
    ax2.set_title("Fraud vs Non-Fraud per Cluster")
    st.pyplot(fig2)

    # --- Radar Plot: Normalized Cluster Profiles ---
    st.subheader("📊 Cluster Feature Radar")
    cluster_features = df.groupby('cluster')[features_to_cluster].mean().round(2)
    cluster_norm = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())

    labels = features_to_cluster
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(len(cluster_norm)):
        values = cluster_norm.iloc[i].tolist() + [cluster_norm.iloc[i, 0]]
        ax3.plot(angles, values, label=f'Cluster {i}')
        ax3.fill(angles, values, alpha=0.1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    ax3.set_title("Normalized Cluster Profiles")
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig3)

    # --- Cluster Center Heatmap ---
    st.subheader("🔥 Cluster Centers (KMeans)")
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_to_cluster)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cluster_centers, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title("KMeans Cluster Centers")
    st.pyplot(fig4)

    # --- Feature Importance (if model loaded) ---
    try:
        model = joblib.load("fraud_model.joblib")
        model_features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
                          'hour', 'is_night', 'is_weekend', 'cluster']
        X_model = df[model_features].fillna(0)

        st.subheader("📌 Feature Importance (Model)")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=model_features).sort_values(ascending=False)

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax5)
        ax5.set_title("🔍 Feature Importance (Random Forest)")
        st.pyplot(fig5)
    except Exception as e:
        st.info("⚠️ Could not load `fraud_model.joblib` for feature importance.")

    # --- Confusion Matrix (Optional if X_test & y_test provided) ---
    try:
        X_test = X_model
        y_test = df['is_fraud']
        st.subheader("📉 Confusion Matrix")
        fig6, ax6 = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax6)
        ax6.set_title("🧾 Confusion Matrix (Whole Dataset)")
        st.pyplot(fig6)
    except:
        st.info("⚠️ No test data or model to display confusion matrix.")

    # --- Fraud Trends ---
    st.subheader("📅 Fraud Trends")
    fig7, ax7 = plt.subplots()
    sns.countplot(data=df, x='hour', hue='is_fraud', ax=ax7)
    ax7.set_title("🕒 Fraud Frequency by Hour")
    st.pyplot(fig7)

    fig8, ax8 = plt.subplots()
    sns.barplot(data=df, x='is_weekend', y='is_fraud', ax=ax8)
    ax8.set_title("📆 Fraud Rate by Weekend")
    st.pyplot(fig8)

    # --- Heatmap of Fraud by Location & Hour ---
    if 'location' in df.columns:
        st.subheader("🌍 Heatmap of Fraud by Location and Hour")
        heatmap_data = df.groupby(['location', 'hour'])['is_fraud'].mean().unstack().fillna(0)
        fig9, ax9 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='Reds', ax=ax9)
        ax9.set_title("Fraud Rate by Location and Hour")
        st.pyplot(fig9)

    # --- Top Senders ---
    if 'sender_account' in df.columns and 'tx_per_sender' in df.columns:
        st.subheader("🏦 Top 20 Most Active Senders")
        top_senders = df.groupby('sender_account')['tx_per_sender'].max().sort_values(ascending=False).head(20)
        fig10, ax10 = plt.subplots(figsize=(10, 6))
        top_senders.plot(kind='barh', color='green', ax=ax10)
        ax10.set_title("Top 20 Most Active Senders")
        ax10.set_xlabel("Transaction Count")
        ax10.invert_yaxis()
        st.pyplot(fig10)

else:
    st.info("📌 Please upload your `Fraud_Detection.csv` to begin.")
