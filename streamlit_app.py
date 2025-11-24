"""
Pakistani Diabetes Dataset - K-Means Clustering Analysis
Professional Streamlit Application for LinkedIn Portfolio
Author: [Syed Rizwan Ali Naqvi]
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Clustering Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 20px;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    h3 {
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔬 Pakistani Diabetes Dataset-K-Means Clustering Analysis")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-bottom: 30px;'>
An advanced machine learning analysis using K-Means clustering to identify patient segments in diabetes data.
This application demonstrates data preprocessing, clustering optimization, and comprehensive visualization techniques.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("###  Analysis Configuration")
    
    # Upload option
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    st.markdown("---")
    
    # Clustering parameters
    st.markdown("###  Clustering Parameters")
    max_clusters = st.slider("Maximum clusters for evaluation", 2, 15, 10)
    selected_k = st.slider("Number of clusters (K)", 2, 8, 3)
    
    st.markdown("---")
    
    # Feature selection
    st.markdown("###  Feature Selection for Default Dataset")
    st.info("Using features: Age, Weight, BMI, A1c, B.S.R")
    
    st.markdown("---")
    
    # About section
    st.markdown("### 👨‍💻 About")
    st.markdown("""
    **Created by:** Syed Rizwan Ali Naqvi
    **Purpose:** LinkedIn Portfolio  
    **Technologies:** Python, Streamlit, Scikit-learn  
    **Algorithm:** K-Means Clustering
    """)

# Load data function
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # For demo purposes - you can remove this and make upload mandatory
        try:
            df = pd.read_csv('Pakistani_Diabetes_Dataset.csv')
        except:
            st.error("Please upload the Pakistani Diabetes Dataset CSV file")
            st.stop()
    return df

# Main application
def main():
    # Load data
    df = load_data(uploaded_file)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Clustering Analysis", 
        "📈 Performance Metrics", 
        "🎯 Cluster Insights",
        "💡 Key Findings"
    ])
    
    # Prepare data
    feature_columns = ['Age', 'wt', 'BMI', 'A1c', 'B.S.R']
    X = df[feature_columns].fillna(df[feature_columns].mean())
    y = df['Outcome']
    
    # Split and scale data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features Used", len(feature_columns))
        with col3:
            st.metric("Diabetic Patients", f"{(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        with col4:
            st.metric("Non-Diabetic", f"{(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        
        st.markdown("---")
        
        # Data distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Statistics")
            st.dataframe(X.describe().round(2), use_container_width=True)
        
        with col2:
            st.subheader("Feature Correlations")
            fig_corr = px.imshow(X.corr().round(2), 
                               text_auto=True,
                               color_continuous_scale='RdBu',
                               title="Feature Correlation Matrix")
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions by Outcome")
        fig_dist = make_subplots(rows=2, cols=3, subplot_titles=feature_columns + [''])
        
        for i, feature in enumerate(feature_columns):
            row = i // 3 + 1
            col = i % 3 + 1
            
            for outcome in [0, 1]:
                data = df[df['Outcome'] == outcome][feature]
                fig_dist.add_trace(
                    go.Histogram(x=data, name=f"{'Diabetes' if outcome else 'No Diabetes'}", 
                               opacity=0.6, showlegend=(i==0)),
                    row=row, col=col
                )
        
        fig_dist.update_layout(height=600, title_text="Feature Distributions by Diabetes Status")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.header("🔍 K-Means Clustering Analysis")
        
        # Elbow method
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elbow Method")
            inertias = []
            silhouette_scores = []
            K = range(2, max_clusters + 1)
            
            progress_bar = st.progress(0)
            for i, k in enumerate(K):
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_temp.fit(X_train_scaled)
                inertias.append(kmeans_temp.inertia_)
                silhouette_scores.append(silhouette_score(X_train_scaled, kmeans_temp.labels_))
                progress_bar.progress((i + 1) / len(K))
            
            progress_bar.empty()
            
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(K), y=inertias, mode='lines+markers',
                                         name='Inertia', marker=dict(size=10)))
            fig_elbow.update_layout(title="Elbow Method for Optimal K",
                                  xaxis_title="Number of Clusters (K)",
                                  yaxis_title="Inertia")
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col2:
            st.subheader("Silhouette Score Analysis")
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(x=list(K), y=silhouette_scores, 
                                              mode='lines+markers',
                                              name='Silhouette Score', 
                                              marker=dict(size=10, color='red')))
            fig_silhouette.update_layout(title="Silhouette Score for Different K",
                                       xaxis_title="Number of Clusters (K)",
                                       yaxis_title="Silhouette Score")
            st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Apply K-means with selected K
        kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        
        # Predictions
        train_clusters = kmeans.predict(X_train_scaled)
        val_clusters = kmeans.predict(X_val_scaled)
        test_clusters = kmeans.predict(X_test_scaled)
        all_clusters = kmeans.predict(X_scaled)
        
        # PCA visualization
        st.subheader("Cluster Visualization (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': all_clusters.astype(str),
            'Outcome': y.map({0: 'No Diabetes', 1: 'Diabetes'})
        })
        
        fig_pca = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster',
                           symbol='Outcome', size_max=10,
                           title=f'K-Means Clustering Results (K={selected_k})',
                           labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                                  'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'})
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        for i in range(selected_k):
            fig_pca.add_trace(go.Scatter(x=[centers_pca[i, 0]], y=[centers_pca[i, 1]],
                                       mode='markers', marker=dict(size=20, symbol='star',
                                       color='black', line=dict(color='white', width=2)),
                                       name=f'Center {i}', showlegend=False))
        
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with tab3:
        st.header("📈 Performance Metrics")
        
        # Calculate metrics
        train_silhouette = silhouette_score(X_train_scaled, train_clusters)
        val_silhouette = silhouette_score(X_val_scaled, val_clusters)
        test_silhouette = silhouette_score(X_test_scaled, test_clusters)
        
        calinski = calinski_harabasz_score(X_test_scaled, test_clusters)
        davies_bouldin = davies_bouldin_score(X_test_scaled, test_clusters)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Silhouette", f"{train_silhouette:.3f}")
        with col2:
            st.metric("Validation Silhouette", f"{val_silhouette:.3f}")
        with col3:
            st.metric("Test Silhouette", f"{test_silhouette:.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Calinski-Harabasz Score", f"{calinski:.2f}", 
                     help="Higher is better - measures cluster separation")
        with col2:
            st.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}",
                     help="Lower is better - measures cluster similarity")
        
        # Performance comparison chart
        st.subheader("Performance Across Datasets")
        
        perf_data = pd.DataFrame({
            'Dataset': ['Training', 'Validation', 'Test'],
            'Silhouette Score': [train_silhouette, val_silhouette, test_silhouette],
            'Samples': [len(X_train), len(X_val), len(X_test)]
        })
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(x=perf_data['Dataset'], y=perf_data['Silhouette Score'],
                                text=perf_data['Silhouette Score'].round(3),
                                textposition='auto',
                                marker_color=['green', 'blue', 'red']))
        fig_perf.update_layout(title="Silhouette Scores Across Datasets",
                             yaxis_title="Silhouette Score")
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with tab4:
        st.header("🎯 Cluster Insights")
        
        # Cluster centers
        cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(cluster_centers_original, columns=feature_columns)
        centers_df.index = [f'Cluster {i}' for i in range(selected_k)]

        st.subheader("Cluster Centers")
        st.dataframe(
            centers_df.round(2),
            use_container_width=True,
            height=155
        )

        
        # Radar chart for cluster profiles
        st.subheader("Cluster Profiles (Radar Chart)")
        
        fig_radar = go.Figure()
        
        for i in range(selected_k):
            values = centers_df.iloc[i].tolist()
            values.append(values[0])  # Complete the circle
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=feature_columns + [feature_columns[0]],
                fill='toself',
                name=f'Cluster {i}'
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, centers_df.values.max() * 1.1]
                )),
            showlegend=True,
            title="Cluster Characteristics Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Diabetes distribution by cluster
        st.subheader("Diabetes Prevalence by Cluster")
        
        cluster_analysis = pd.DataFrame({
            'Cluster': all_clusters,
            'Outcome': y
        })
        
        diabetes_by_cluster = []
        for cluster in range(selected_k):
            cluster_data = cluster_analysis[cluster_analysis['Cluster'] == cluster]
            total = len(cluster_data)
            diabetic = (cluster_data['Outcome'] == 1).sum()
            diabetes_rate = (diabetic / total) * 100
            
            diabetes_by_cluster.append({
                'Cluster': f'Cluster {cluster}',
                'Total Samples': total,
                'Diabetic': diabetic,
                'Non-Diabetic': total - diabetic,
                'Diabetes Rate (%)': diabetes_rate
            })
        
        diabetes_df = pd.DataFrame(diabetes_by_cluster)
        
        fig_diabetes = go.Figure()
        fig_diabetes.add_trace(go.Bar(name='Non-Diabetic', x=diabetes_df['Cluster'], 
                                    y=diabetes_df['Non-Diabetic'],
                                    marker_color='lightblue'))
        fig_diabetes.add_trace(go.Bar(name='Diabetic', x=diabetes_df['Cluster'], 
                                    y=diabetes_df['Diabetic'],
                                    marker_color='lightcoral'))
        
        fig_diabetes.update_layout(barmode='stack', title='Diabetes Distribution Across Clusters',
                                 xaxis_title='Cluster', yaxis_title='Number of Patients')
        st.plotly_chart(fig_diabetes, use_container_width=True)
        
        # Detailed cluster statistics
        st.subheader("Detailed Cluster Statistics")
        st.data_editor(
            diabetes_df,
            use_container_width=True,
            disabled=True,    # makes it read-only
            height=155        # fixed height → no shaking
        )
    
    with tab5:
        st.header("💡 Key Findings and Insights")
        
        # Automated insights based on clustering results
        st.markdown("### 🔍 Clustering Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Algorithm:** K-Means Clustering  
            **Optimal K:** {selected_k} clusters  
            **Features:** {', '.join(feature_columns)}  
            **Test Silhouette Score:** {test_silhouette:.3f}
            """)
        
        with col2:
            st.success(f"""
            **Total Samples:** {len(df)}  
            **Training Set:** {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)  
            **Validation Set:** {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)  
            **Test Set:** {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)
            """)
        
        st.markdown("###  Cluster Characteristics")
        
        # Generate insights for each cluster
        for i in range(selected_k):
            cluster_mask = all_clusters == i
            diabetes_rate = (y[cluster_mask] == 1).sum() / cluster_mask.sum() * 100
            
            with st.expander(f"Cluster {i} Analysis ({cluster_mask.sum()} samples)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
                with col2:
                    st.metric("Avg Age", f"{X[cluster_mask]['Age'].mean():.1f} years")
                with col3:
                    st.metric("Avg BMI", f"{X[cluster_mask]['BMI'].mean():.1f}")
                
                # Determine cluster risk level
                if diabetes_rate > 70:
                    risk_level = "🔴 High Risk"
                elif diabetes_rate > 40:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🟢 Low Risk"
                
                st.markdown(f"""
                **Risk Level:** {risk_level}  
                **Average A1c:** {X[cluster_mask]['A1c'].mean():.2f}%  
                **Average B.S.R:** {X[cluster_mask]['B.S.R'].mean():.1f} mg/dL  
                **Average Weight:** {X[cluster_mask]['wt'].mean():.1f} kg
                
                **Clinical Insights:**
                - This cluster represents {'high' if diabetes_rate > 70 else 'moderate' if diabetes_rate > 40 else 'low'} diabetes prevalence
                - Patients show {'elevated' if X[cluster_mask]['A1c'].mean() > 7 else 'normal'} A1c levels
                - BMI indicates {'obese' if X[cluster_mask]['BMI'].mean() > 30 else 'overweight' if X[cluster_mask]['BMI'].mean() > 25 else 'normal'} weight category
                """)
        
        st.markdown("###  Recommendations")
        st.markdown("""
        Based on the clustering analysis, here are the key recommendations:
        
        1. **Patient Segmentation**: The identified clusters can be used for targeted intervention strategies
        2. **Risk Stratification**: High-risk clusters should receive priority screening and preventive care
        3. **Resource Allocation**: Healthcare resources can be optimized based on cluster characteristics
        4. **Personalized Care**: Treatment plans can be tailored according to cluster profiles
        5. **Early Detection**: Focus on clusters with borderline characteristics for early intervention
        """)
        
        # Download results
        st.markdown("###  Export Results")
        
        results_df = pd.DataFrame({
            'Age': X['Age'],
            'Weight': X['wt'],
            'BMI': X['BMI'],
            'A1c': X['A1c'],
            'B.S.R': X['B.S.R'],
            'Cluster': all_clusters,
            'Actual_Outcome': y
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label=" Download Clustering Results (CSV)",
            data=csv,
            file_name='kmeans_clustering_results.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p> Connect with me on <a href='https://www.linkedin.com/in/rizwan-ali-411446353' target='_blank'>LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
