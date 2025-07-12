    # Sidebar selection
    st.sidebar.header("📊 Select Visualization")
    selected_plot = st.sidebar.selectbox("Choose a chart to display:", [
        "Cluster Fraud Summary",
        "Top Fraud Locations",
        "Fraud vs Non-Fraud per Cluster",
        "Cluster Feature Radar",
        "Cluster Centers (Heatmap)",
        "Feature Importance",
        "Confusion Matrix",
        "Fraud by Hour",
        "Fraud by Weekend",
        "Fraud Heatmap by Location and Hour",
        "Top 20 Most Active Senders"
    ])

    # --- Visualizations ---
    if selected_plot == "Cluster Fraud Summary":
        st.subheader("🔍 Cluster Fraud Summary")
        st.dataframe(cluster_summary)

    elif selected_plot == "Top Fraud Locations" and 'location' in df.columns:
        st.subheader("📍 Top Locations for Fraud")
        city_counts = df[df['is_fraud'] == True]['location'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax)
        ax.set_title("Top Locations for Fraud")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    elif selected_plot == "Fraud vs Non-Fraud per Cluster":
        st.subheader("🧮 Fraud vs Non-Fraud per Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='cluster', hue='is_fraud', palette='Set2', ax=ax)
        ax.set_title("Fraud vs Non-Fraud per Cluster")
        st.pyplot(fig)

    elif selected_plot == "Cluster Feature Radar":
        st.subheader("📊 Cluster Feature Radar")
        cluster_features = df.groupby('cluster')[features_to_cluster].mean().round(2)
        cluster_norm = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())
        labels = features_to_cluster
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        for i in range(len(cluster_norm)):
            values = cluster_norm.iloc[i].tolist() + [cluster_norm.iloc[i, 0]]
            ax.plot(angles, values, label=f'Cluster {i}')
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Normalized Cluster Profiles")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)

    elif selected_plot == "Cluster Centers (Heatmap)":
        st.subheader("🔥 Cluster Centers (KMeans)")
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_to_cluster)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(cluster_centers, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("KMeans Cluster Centers")
        st.pyplot(fig)

    elif selected_plot == "Feature Importance":
        st.subheader("📌 Feature Importance (Model)")
        try:
            model = joblib.load("fraud_model.joblib")
            model_features = ['amount', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
                              'hour', 'is_night', 'is_weekend', 'cluster']
            X_model = df[model_features].fillna(0)
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=model_features).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
            ax.set_title("🔍 Feature Importance (Random Forest)")
            st.pyplot(fig)
        except Exception:
            st.info("⚠️ Could not load model for feature importance.")

    elif selected_plot == "Confusion Matrix":
        st.subheader("📉 Confusion Matrix")
        try:
            model = joblib.load("fraud_model.joblib")
            X_model = df[model_features].fillna(0)
            y_test = df['is_fraud']
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_estimator(model, X_model, y_test, ax=ax)
            ax.set_title("🧾 Confusion Matrix")
            st.pyplot(fig)
        except Exception:
            st.info("⚠️ Could not display confusion matrix.")

    elif selected_plot == "Fraud by Hour":
        st.subheader("🕒 Fraud by Hour")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df, x='hour', hue='is_fraud', ax=ax)
        ax.set_title("Fraud Frequency by Hour")
        st.pyplot(fig)

    elif selected_plot == "Fraud by Weekend":
        st.subheader("📆 Fraud by Weekend")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df, x='is_weekend', y='is_fraud', ax=ax)
        ax.set_title("Fraud Rate by Weekend")
        st.pyplot(fig)

    elif selected_plot == "Fraud Heatmap by Location and Hour" and 'location' in df.columns:
        st.subheader("🌍 Fraud Heatmap by Location and Hour")
        heatmap_data = df.groupby(['location', 'hour'])['is_fraud'].mean().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(heatmap_data, cmap='Reds', ax=ax)
        ax.set_title("Fraud Rate by Location and Hour")
        st.pyplot(fig)

    elif selected_plot == "Top 20 Most Active Senders" and 'sender_account' in df.columns and 'tx_per_sender' in df.columns:
        st.subheader("🏦 Top 20 Most Active Senders")
        top_senders = df.groupby('sender_account')['tx_per_sender'].max().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 4))
        top_senders.plot(kind='barh', color='green', ax=ax)
        ax.set_title("Top 20 Most Active Senders")
        ax.set_xlabel("Transaction Count")
        ax.invert_yaxis()
        st.pyplot(fig)
