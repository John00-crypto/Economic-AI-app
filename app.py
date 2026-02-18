"""
Advanced Economic Indicators AI System
A professional machine learning application for economic analysis and stock index prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
from datetime import datetime
from utils import load_data, train_models, load_model

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Tanzania Stock Index Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
    <style>
    /* Professional Color Palette */
    :root {
        --primary-navy: #1a365d;
        --secondary-slate: #2d3748;
        --accent-emerald: #2f855a;
        --soft-emerald: rgba(47, 133, 90, 0.1);
        --border-color: rgba(26, 54, 93, 0.1);
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1a365d 0%, #2f855a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .sub-header {
            color: #a0aec0;
        }
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #2d3748;
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Info boxes with refined support */
    .info-box {
        background-color: rgba(26, 54, 93, 0.04);
        border: 1px solid rgba(26, 54, 93, 0.1);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-navy);
        margin: 1.5rem 0;
    }
    .info-box b {
        color: var(--primary-navy);
        font-weight: 700;
    }
    .success-box {
        background-color: rgba(47, 133, 90, 0.04);
        border: 1px solid rgba(47, 133, 90, 0.1);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--accent-emerald);
    }
    .success-box b {
        color: var(--accent-emerald);
        font-weight: 700;
    }
    .warning-box {
        background-color: rgba(193, 128, 0, 0.04);
        border: 1px solid rgba(193, 128, 0, 0.1);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #c05621;
    }
    .warning-box b {
        color: #c05621;
        font-weight: 700;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-navy);
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetricValue"] {
            color: #e2e8f0;
        }
    }
    
    /* Premium Button styling */
    .stButton>button {
        width: 100%;
        background-color: var(--primary-navy);
        color: white !important;
        font-weight: 600;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        transition: all 0.25s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button:hover {
        background-color: var(--accent-emerald);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(47, 133, 90, 0.2);
    }
    /* Footer text */
    .footer-text {
        color: #718096 !important;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'trained_model' not in st.session_state:
    # Try loading from disk first
    loaded_model = load_model('model.pkl')
    st.session_state.trained_model = loaded_model
    if loaded_model:
         st.session_state.best_model_name = "Pre-trained Model (Loaded)"
         st.session_state.training_timestamp = datetime.fromtimestamp(os.path.getmtime('model.pkl')).strftime("%Y-%m-%d %H:%M:%S")

if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'training_timestamp' not in st.session_state:
    if st.session_state.trained_model is None:
        st.session_state.training_timestamp = None
if 'best_model_name' not in st.session_state:
    if st.session_state.trained_model is None:
        st.session_state.best_model_name = None

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">📊 Tanzania Stock Index Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced machine learning system for Tanzania\'s economic analysis and stock index prediction</p>', unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_dataset():
    """Load and cache the dataset"""
    try:
        return load_data("economic_indicator.csv")
    except FileNotFoundError:
        st.error("❌ **Error**: Dataset file 'economic_indicators_1000.csv' not found. Please run the setup notebook first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error loading data**: {str(e)}")
        st.stop()

data = load_dataset()

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/graph.png", width=80)
    st.markdown("### 🧭 Navigation")
    menu = st.selectbox(
        "Select Module",
        ["🏠 Dashboard", "🤖 Model Training", "🔮 Prediction", "📊 Analytics", "⬇️ Download"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Status Indicator
    st.markdown("### 🎯 System Status")
    if st.session_state.trained_model is not None:
        st.success(f"✅ Model Active")
        st.info(f"**Type**: {st.session_state.best_model_name}")
        if st.session_state.training_timestamp:
            st.caption(f"Trained: {st.session_state.training_timestamp}")
    else:
        st.warning("⚠️ No Model Trained")
        st.caption("Train a model in the Model Training section")
    
    st.markdown("---")
    
    # Dataset Info
    st.markdown("### 📈 Dataset Info")
    st.metric("Total Records", data.shape[0])
    st.metric("Features", data.shape[1] - 1)
    st.caption("Target: Stock_Index")
    
    st.markdown("---")
    st.caption("© 2026 Economic AI Platform")

# ==================== DASHBOARD MODULE ====================
if menu == "🏠 Dashboard":
    st.markdown("## 📌 Dataset Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{data.shape[0]:,}")
    with col2:
        st.metric("🔢 Features", data.shape[1] - 1)
    with col3:
        st.metric("📈 Avg Stock Index", f"{data['Stock_Index'].mean():.2f}")
    with col4:
        st.metric("📉 Std Deviation", f"{data['Stock_Index'].std():.2f}")
    
    st.markdown("---")
    
    # Data Preview
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📋 Data Preview")
        st.dataframe(data.head(20), use_container_width=True, height=400)
    
    with col_right:
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(data.describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["🔥 Correlation Heatmap", "📊 Distribution", "📈 Trends"])
    
    with tab1:
        st.markdown("### Correlation Analysis")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = data.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close()
        
        # Correlation insights
        stock_corr = correlation_matrix['Stock_Index'].drop('Stock_Index').sort_values(ascending=False)
        st.markdown("#### 🔍 Key Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Strongest Positive**: {stock_corr.idxmax()} ({stock_corr.max():.3f})")
        with col2:
            st.error(f"**Strongest Negative**: {stock_corr.idxmin()} ({stock_corr.min():.3f})")
    
    with tab2:
        st.markdown("### Feature Distributions")
        feature_to_plot = st.selectbox("Select Feature", data.columns.tolist())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(data[feature_to_plot], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax1.set_title(f'{feature_to_plot} - Histogram', fontsize=14, fontweight='bold')
        ax1.set_xlabel(feature_to_plot)
        ax1.set_ylabel('Frequency')
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2.boxplot(data[feature_to_plot], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#764ba2', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title(f'{feature_to_plot} - Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel(feature_to_plot)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown("### Stock Index Trends")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data.index, data['Stock_Index'], color='#667eea', linewidth=2, alpha=0.7)
        ax.fill_between(data.index, data['Stock_Index'], alpha=0.3, color='#667eea')
        ax.set_title('Stock Index Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Record Number')
        ax.set_ylabel('Stock Index')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
# ==================== MODEL TRAINING MODULE ====================
elif menu == "🤖 Model Training":
    st.markdown("## 🤖 Machine Learning Model Training")
    
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ Training Process</b><br>
    This module will train multiple machine learning algorithms and automatically select the best performing model based on R² score.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Training Options")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        random_state = st.number_input("Random State", 0, 100, 42, 1)
        
        train_button = st.button("🚀 Start Training", use_container_width=True)
    
    with col2:
        if train_button:
            with st.spinner("🔄 Training models... Please wait."):
                try:
                    results, best_model = train_models(data)
                    
                    # Store in session state
                    st.session_state.trained_model = best_model
                    st.session_state.training_results = results
                    st.session_state.training_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Find best model name
                    best_r2 = max(results.values(), key=lambda x: x['R2'])['R2']
                    st.session_state.best_model_name = [k for k, v in results.items() if v['R2'] == best_r2][0]
                    
                    st.success("✅ **Training Completed Successfully!**")
                except Exception as e:
                    st.error(f"❌ **Training Error**: {str(e)}")
    
    # Display Results
    if st.session_state.training_results is not None:
        st.markdown("---")
        st.markdown("### 📊 Training Results")
        
        results = st.session_state.training_results
        
        result_df = pd.DataFrame({
            "Model": list(results.keys()),
            "R² Score": [results[m]["R2"] for m in results],
            "MSE": [results[m]["MSE"] for m in results],
            "RMSE": [np.sqrt(results[m]["MSE"]) for m in results]
        }).sort_values("R² Score", ascending=False)
        
        # Highlight best model
        def highlight_best(row):
            if row["R² Score"] == result_df["R² Score"].max():
                return ['background-color: #d4edda; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled_df = result_df.style.apply(highlight_best, axis=1).format({
            "R² Score": "{:.4f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}"
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 R² Score Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#28a745' if x == result_df["R² Score"].max() else '#667eea' 
                     for x in result_df["R² Score"]]
            ax.barh(result_df["Model"], result_df["R² Score"], color=colors, alpha=0.8)
            ax.set_xlabel("R² Score", fontweight='bold')
            ax.set_title("Model Performance Comparison", fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 📉 Error Metrics")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(result_df))
            width = 0.35
            ax.bar(x - width/2, result_df["MSE"], width, label='MSE', color='#ff6b6b', alpha=0.8)
            ax.bar(x + width/2, result_df["RMSE"], width, label='RMSE', color='#feca57', alpha=0.8)
            ax.set_xlabel("Model", fontweight='bold')
            ax.set_ylabel("Error Value", fontweight='bold')
            ax.set_title("Error Comparison", fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(result_df["Model"], rotation=15, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # New Visualizations (Actual vs Predicted & Residuals)
        st.markdown("#### 🎯 Prediction Analysis (Best Model)")
        
        best_model_name = st.session_state.best_model_name
        if best_model_name in results:
            y_test = results[best_model_name]["y_test"]
            y_pred = results[best_model_name]["y_pred"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Actual vs Predicted**")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, color='#667eea')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title(f"Actual vs Predicted ({best_model_name})")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            with col2:
                st.markdown("**Residuals Distribution**")
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(residuals, kde=True, color='#ff6b6b', ax=ax)
                ax.axvline(0, color='r', linestyle='--')
                ax.set_xlabel("Residual (Actual - Predicted)")
                ax.set_title(f"Residuals Distribution ({best_model_name})")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        # Best model info
        st.markdown(f"""
        <div class="success-box">
        <b>🏆 Best Model Selected</b><br>
        <b>Algorithm</b>: {st.session_state.best_model_name}<br>
        <b>R² Score</b>: {result_df["R² Score"].max():.4f}<br>
        <b>RMSE</b>: {result_df["RMSE"].min():.2f}<br>
        <b>Status</b>: Ready for predictions
        </div>
        """, unsafe_allow_html=True)
                    
# ==================== PREDICTION MODULE ====================
elif menu == "🔮 Prediction":
    st.markdown("## 🔮 Stock Index Prediction Engine")
    
    if st.session_state.trained_model is None:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ No Model Available</b><br>
        Please train a model first in the <b>Model Training</b> section before making predictions.
        </div>
        """, unsafe_allow_html=True)
    else:
        model = st.session_state.trained_model
        
        st.markdown("""
        <div class="info-box">
        <b>ℹ️ Prediction System</b><br>
        Enter economic indicators below to predict the stock index value using the trained model.
        </div>
        """, unsafe_allow_html=True)

    
        st.markdown("### 📝 Input Economic Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gdp = st.number_input(
                "📊 GDP Growth (%)",
                min_value=0.0,
                max_value=15.0,
                value=6.0,
                step=0.1,
                help="Tanzania Gross Domestic Product growth rate"
            )
            
        with col2:
            inflation = st.number_input(
                "💰 Inflation Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=4.0,
                step=0.1,
                help="Consumer price inflation rate"
            )
            
        with col3:
            unemployment = st.number_input(
                "👥 Unemployment Rate (%)",
                min_value=0.0,
                max_value=25.0,
                value=10.0,
                step=0.1,
                help="Unemployment percentage"
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            interest = st.number_input(
                "💹 Interest Rate (%)",
                min_value=0.0,
                max_value=40.0,
                value=12.0,
                step=0.1,
                help="Central bank interest rate"
            )
            
        with col5:
            exchange = st.number_input(
                "💱 Exchange Rate (TZS)",
                min_value=500.0,
                max_value=5000.0,
                value=2500.0,
                step=10.0,
                help="Currency exchange rate (TZS/USD)"
            )
        
        st.markdown("---")
        
        col_predict, col_reset = st.columns([3, 1])
        
        with col_predict:
            predict_button = st.button("🔮 Generate Prediction", use_container_width=True)
        
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        if predict_button:
            # Validate inputs
            if any(val < 0 for val in [gdp, inflation, unemployment, interest, exchange]):
                st.error("❌ All values must be positive!")
            else:
                with st.spinner("🔄 Calculating prediction..."):
                    X = np.array([[gdp, inflation, unemployment, interest, exchange]])
                    prediction = model.predict(X)[0]
                    
                    # Display prediction with visual appeal
                    st.markdown("### 🎯 Prediction Result")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 15px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                        ">
                            <h1 style="margin: 0; font-size: 3rem;">{prediction:.2f}</h1>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Predicted Stock Index</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Show input summary
                    with st.expander("📋 View Input Summary", expanded=True):
                        input_df = pd.DataFrame({
                            "Indicator": ["GDP Growth", "Inflation", "Unemployment", "Interest Rate", "Exchange Rate"],
                            "Value": [f"{gdp}%", f"{inflation}%", f"{unemployment}%", f"{interest}%", f"{exchange}"],
                            "Impact": ["High", "Medium", "High", "Medium", "Low"]
                        })
                        st.dataframe(input_df, use_container_width=True, hide_index=True)
                    
                    # Comparison with dataset
                    avg_stock = data['Stock_Index'].mean()
                    diff = prediction - avg_stock
                    diff_pct = (diff / avg_stock) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Value", f"{prediction:.2f}")
                    col2.metric("Dataset Average", f"{avg_stock:.2f}")
                    col3.metric("Difference", f"{diff:.2f}", f"{diff_pct:+.2f}%")

# ==================== ANALYTICS MODULE ====================
elif menu == "📊 Analytics":
    st.markdown("## 📊 Model Analytics & Insights")
    
    if st.session_state.trained_model is None:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ No Model Available</b><br>
        Please train a model first in the <b>Model Training</b> section to view analytics.
        </div>
        """, unsafe_allow_html=True)
    else:
        model = st.session_state.trained_model
        
        st.markdown("### 🔍 Feature Importance Analysis")
        
        # Extract feature importance
        # Check if model is a pipeline
        if hasattr(model, 'named_steps'):
            regressor = model.named_steps['model']
        else:
            regressor = model

        if hasattr(regressor, "feature_importances_"):
            importance = regressor.feature_importances_
            importance_type = "Tree-based Model"
        elif hasattr(regressor, "coef_"):
            importance = np.abs(regressor.coef_)
            importance_type = "Linear Model (Absolute Coefficients)"
        else:
            st.error("❌ Model does not support feature importance analysis")
            st.stop()
        
        # Fix: Ensure features match the model input
        features = data.drop("Stock_Index", axis=1).columns
        
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        
        # Normalize importance to percentage
        imp_df["Importance %"] = (imp_df["Importance"] / imp_df["Importance"].sum()) * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### 📈 Importance Ranking ({importance_type})")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
            bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors, alpha=0.8)
            ax.set_xlabel("Importance Score", fontweight='bold', fontsize=12)
            ax.set_ylabel("Feature", fontweight='bold', fontsize=12)
            ax.set_title("Feature Importance Analysis", fontweight='bold', fontsize=14, pad=20)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, imp_df["Importance"])):
                ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}',
                       va='center', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 📊 Importance Table")
            styled_imp = imp_df.style.background_gradient(
                subset=["Importance"],
                cmap="YlGnBu"
            ).format({
                "Importance": "{:.4f}",
                "Importance %": "{:.2f}%"
            })
            st.dataframe(styled_imp, use_container_width=True, hide_index=True)
            
            # Key insights
            st.markdown("#### 🎯 Key Insights")
            st.success(f"**Most Important**: {imp_df.iloc[0]['Feature']}")
            st.info(f"**Contribution**: {imp_df.iloc[0]['Importance %']:.1f}%")
        
        st.markdown("---")
        
        # Pie chart
        st.markdown("### 🥧 Importance Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(imp_df)))
        wedges, texts, autotexts = ax.pie(
            imp_df["Importance %"],
            labels=imp_df["Feature"],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0.05 if i == 0 else 0 for i in range(len(imp_df))]
        )
        ax.set_title("Feature Importance Distribution", fontweight='bold', fontsize=14, pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ==================== DOWNLOAD MODULE ====================
elif menu == "⬇️ Download":
    st.markdown("## ⬇️ Download Center")
    
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ Export Options</b><br>
    Download the dataset and trained models for offline analysis and deployment.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Export")
        st.markdown("""
        Download the complete economic indicators dataset in CSV format.
        
        **Includes:**
        - All economic indicators
        - Stock index values
        - 1000 data records
        """)
        
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name=f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.metric("File Size", f"{len(csv_data) / 1024:.1f} KB")
        st.metric("Records", f"{data.shape[0]:,}")
    
    with col2:
        st.markdown("### 🤖 Model Export")
        
        if st.session_state.trained_model is not None:
            st.markdown(f"""
            Download the trained machine learning model for deployment.
            
            **Model Details:**
            - Algorithm: {st.session_state.best_model_name}
            - Format: Pickle (.pkl)
            - Training: {st.session_state.training_timestamp}
            """)
            
            # Serialize model
            buffer = io.BytesIO()
            joblib.dump(st.session_state.trained_model, buffer)
            model_bytes = buffer.getvalue()
            st.download_button(
                label="📥 Download Model (PKL)",
                data=model_bytes,
                file_name=f"economic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            
            st.metric("Model Size", f"{len(model_bytes) / 1024:.1f} KB")
            st.metric("Algorithm", st.session_state.best_model_name)
        else:
            st.markdown("""
            <div class="warning-box">
            <b>⚠️ No Model Available</b><br>
            Train a model first to enable download.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Usage instructions
    with st.expander("📖 How to Use Downloaded Files"):
        st.markdown("""
        ### Using the Dataset
        ```python
        import pandas as pd
        
        # Load the dataset
        data = pd.read_csv('economic_indicators_YYYYMMDD.csv')
        print(data.head())
        ```
        
        ### Using the Model
        ```python
        import joblib
        import numpy as np
        
        # Load the model
        model = joblib.load('economic_model_YYYYMMDD_HHMMSS.pkl')
        
        # Make predictions
        X = np.array([[3.0, 4.0, 6.5, 5.0, 2300.0]])  # GDP, Inflation, Unemployment, Interest, Exchange
        prediction = model.predict(X)[0]
        print(f"Predicted Stock Index: {prediction:.2f}")
        ```
        """)

# ==================== FOOTER ====================
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.markdown(
        "<p class='footer-text' style='text-align: center;'>© 2026 Tanzania Economic AI Analytics Platform</p>",
        unsafe_allow_html=True
    )

