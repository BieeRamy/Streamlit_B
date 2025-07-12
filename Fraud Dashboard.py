import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit setup
st.set_page_config(page_title="Fraud Clustering Dashboard", layout="wide")
st.title("💳 Fraud Detection & Clustering Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your Fraud_Detection.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(10))

    # Feature selection
    features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
    X = df[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Cluster fraud stats
    cluster_summary = df.groupby('cluster').agg(
        total_transactions=('transaction_id', 'count'),
        fraud_transactions=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
    ).reset_index()
    cluster_summary['fraud_rate'] = (cluster_summary['fraud_rate'] * 100).round(2)

    st.subheader("🔍 Cluster Summary")
    st.dataframe(cluster_summary)

    # Top fraud locations
    if 'location' in df.columns:
        city_counts = df[df['is_fraud'] == True]['location'].value_counts().head(10)
        st.subheader("📍 Top Locations for Fraud")

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax1)
        ax1.set_title("Top Locations for Fraudulent Transactions")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

    # Fraud vs non-fraud per cluster
    st.subheader("🧮 Fraud vs Non-Fraud per Cluster")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='cluster', hue='is_fraud', palette='Set2', ax=ax2)
    ax2.set_title("Fraud vs Non-Fraud per Cluster")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Transaction Count")
    st.pyplot(fig2)

    # Radar chart of normalized cluster features
    st.subheader("📈 Cluster Profile Radar Chart")

    cluster_features = df.groupby('cluster')[features].mean().round(2)
    cluster_norm = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())

    labels = features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(len(cluster_norm)):
        values = cluster_norm.iloc[i].tolist()
        values += values[:1]
        ax3.plot(angles, values, label=f'Cluster {i}')
        ax3.fill(angles, values, alpha=0.1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.set_title("Normalized Cluster Profiles")
    st.pyplot(fig3)

else:
    st.info("Please upload your `Fraud_Detection.csv` file to begin.")
