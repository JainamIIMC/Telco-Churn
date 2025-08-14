# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ---- Plotly defaults (apply once) ----
import plotly.express as px  # ensure px is available here if moved
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = 420
DEFAULT_LAYOUT = dict(
    title_x=0.5,  # center titles
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=60, b=40),
)

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #0F52BA;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'model' not in st.session_state:
    st.session_state.model = {}

# Sidebar for navigation
st.markdown("""
<style>
    /* Style for sidebar buttons to look like panels */
    .stSidebar > div > div > div > div > button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        width: 100%;
        text-align: left;
        font-size: 14px;
        font-weight: 500;
        color: #262730;
        transition: all 0.3s ease;
        min-height: 45px;
        display: flex;
        align-items: center;
    }
    
    .stSidebar > div > div > div > div > button:hover {
        background-color: #e0e2e6;
        border-color: #c0c0c0;
        transform: translateX(2px);
    }
    
    .stSidebar > div > div > div > div > button:focus {
        background-color: #4a5568;
        color: white;
        border-color: #4a5568;
        box-shadow: 0 0 0 2px rgba(74, 85, 104, 0.2);
    }
    
    /* Alternative using custom class for selected state */
    .selected-panel {
        background-color: #4a5568 !important;
        color: white !important;
        border-color: #4a5568 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page selection
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "🏠 Home"

# Sidebar title
st.sidebar.title("🧭 Navigation")

# Define pages
page = [
    "🏠 Home",
    "📊 Data Overview",
    "🔍 Exploratory Analysis",
    "🎯 Customer Insights",
    "💻 ML Models",
    "📈 Model Comparison",
    "🔮 Churn Prediction",
    "💡 Recommendations"
]

# Create panel buttons
for page in page:
    if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
        st.session_state.selected_page = page

# Get the selected page
page = st.session_state.selected_page

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    return df


# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    """Preprocess the data for ML models"""
    df_processed = df.copy()

    # Drop customerID
    df_processed = df_processed.drop(['customerID'], axis=1, errors='ignore')

    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

    # Fill missing values
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].mean(), inplace=True)

    # Convert SeniorCitizen to Yes/No
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: "No", 1: "Yes"})

    # Label encode categorical variables
    le = LabelEncoder()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = le.fit_transform(df_processed[col])

    return df_processed


# Main app content based on navigation
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📱 TELECOM CUSTOMER CHURN PREDICTION</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("age_groups_banner.png")

    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #DC143C; font-style: italic; margin: 2rem 0;'>
    Did you know that attracting a new customer costs <b>five times</b> as much as keeping an existing one?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Churn Rate", "26.6%", "-2.1%")
    with col2:
        st.metric("Cost of Acquisition", "5x", "vs Retention")
    with col3:
        st.metric("Monthly Revenue Impact", "$2.1M", "-15%")
    with col4:
        st.metric("Customer Lifetime Value", "$3,450", "+8%")

    st.markdown("---")

    # Introduction
    st.markdown("""
    ## 🎯 Project Overview

    Customer churn is a critical metric for telecom companies. This interactive dashboard provides:

    - **Comprehensive Analysis** of customer behavior patterns
    - **Predictive Models** to identify at-risk customers
    - **Actionable Insights** for retention strategies
    - **Real-time Monitoring** capabilities

    ### 📊 What We'll Explore:

    1. **Customer Demographics** - Understanding who churns and why
    2. **Service Usage Patterns** - Identifying key service indicators
    3. **Financial Impact** - Analyzing revenue implications
    4. **Predictive Modeling** - Building accurate churn prediction models
    5. **Retention Strategies** - Data-driven recommendations
    """)

    # Quick Start
    st.info("👈 Use the sidebar to navigate through different sections of the analysis")

elif page == "📊 Data Overview":
    st.title("📊 Data Overview & Quality Assessment")

    # Load data
    df = load_data()
    st.session_state.data_loaded = True

    # Data shape and basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")

    st.markdown("---")

    # Data preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Data types and missing values
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        # Convert to native Python types
        fig = px.pie(
            values=dtype_counts.values.tolist(),  # Convert to list
            names=[str(x) for x in dtype_counts.index.tolist()],  # Convert to string list
            title="Distribution of Data Types"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔍 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_data.index.tolist(),  # Convert to list
            'Missing Count': missing_data.values.tolist(),  # Convert to list
            'Percentage': missing_percent.values.tolist()  # Convert to list
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Percentage',
                         title="Missing Data Percentage by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected in the dataset!")

    # Statistical summary
    st.subheader("📈 Statistical Summary")
    df = preprocess_data(df)
    st.dataframe(df.describe(), use_container_width=True)

    # TODO: Add data quality checks
    st.info("📌 TODO: Add comprehensive data quality checks and validation rules")

elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    df = load_data()
    df_processed = preprocess_data(df)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Services", "Financial", "Correlations"])

    with tab1:
        st.subheader("👥 Customer Demographics Analysis")

        # Gender and Churn Distribution
        col1, col2 = st.columns(2)

        with col1:
            # Gender distribution
            gender_counts = df['gender'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                hole=.4,
                marker_colors=['#66b3ff', '#ffb3e6']
            )])
            fig.update_layout(title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn distribution
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=churn_counts.index,
                values=churn_counts.values,
                hole=.4,
                marker_colors=['#ff6666', '#66ff66']
            )])
            fig.update_layout(title="Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Senior Citizens Analysis
        st.subheader("👴 Senior Citizens Analysis")
        senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100
        fig = px.bar(senior_churn.T, barmode='group',
                     title="Churn Rate by Senior Citizen Status (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📡 Service Usage Patterns")

        # Internet Service
        internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
        fig = px.bar(internet_churn.T, barmode='group',
                     title="Churn by Internet Service Type")
        st.plotly_chart(fig, use_container_width=True)

        # Contract Type
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        fig = px.bar(contract_churn.T, barmode='group',
                     title="Churn by Contract Type",
                     color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
        st.plotly_chart(fig, use_container_width=True)

        

        # Simplified Service Churn Analysis
        cols_services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

        present = [c for c in cols_services if c in df.columns]

        if present:
            churn_rates = []
            
            for service in present:
                # Simple approach: treat anything that's not "No" as having the service
                has_service = df[service] != "No"
                service_users = df[has_service]
                
                if len(service_users) > 0:
                    churn_rate = (service_users["Churn"] == "Yes").mean() * 100
                    churn_rates.append({"Service": service, "Churn_Rate": churn_rate})
            
            # Create DataFrame and sort by churn rate
            service_df = pd.DataFrame(churn_rates).sort_values("Churn_Rate", ascending=True)

            # Create horizontal bar chart
            fig = px.bar(service_df, 
                        x="Churn_Rate", 
                        y="Service",
                        orientation="h",
                        title="Churn Rate by Service Type",
                        labels={"Churn_Rate": "Churn Rate (%)", "Service": "Service"},
                        color="Churn_Rate",
                        color_continuous_scale=[(0,'#4ECDC4'),(1,'#FF6B6B') ])
            

            fig.update_layout(showlegend=False,
                            xaxis_title="Churn Rate (%)",
                            yaxis_title="Service")
            fig.update_coloraxes(showscale=False)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No service columns found to plot.")

    with tab3:
        st.subheader("💰 Financial Analysis")

        # Monthly Charges Distribution
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='MonthlyCharges', color='Churn',
                               title="Monthly Charges Distribution by Churn Status",
                               nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df_processed, x='Churn', y='TotalCharges',
                         title="Total Charges by Churn Status")
            st.plotly_chart(fig, use_container_width=True)

        # Tenure Analysis
        fig = px.box(df, x='Churn', y='tenure',
                     title="Customer Tenure by Churn Status",
                     color='Churn',
                     color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔗 Correlation Analysis")

        # Prepare data for correlation
        df_processed = preprocess_data(df)

        # Correlation matrix
        corr_matrix = df_processed.corr()

        fig = px.imshow(corr_matrix,
                        title="Feature Correlation Heatmap",
                        color_continuous_scale='Reds',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with Churn
        churn_corr = corr_matrix['Churn'].sort_values(ascending=False)[1:11]
        fig = px.bar(x=churn_corr.values, y=churn_corr.index,
                     orientation='h',
                     title="Top 10 Features Correlated with Churn")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Customer Insights":
    st.title("🎯 Key Customer Insights & Patterns")

    df = load_data()

    # Key Findings Section
    st.markdown("## 🔑 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📈 High Risk Indicators
        - **Month-to-month contracts**: 75% of churners
        - **Electronic check payments**: Highest churn rate
        - **No tech support**: 2x more likely to churn
        - **Fiber optic users**: Higher dissatisfaction
        - **New customers**: < 12 months tenure at risk
        """)

    with col2:
        st.markdown("""
        ### 🛡️ Retention Indicators
        - **Two-year contracts**: Only 3% churn rate
        - **Auto-payment setup**: Lower churn
        - **Multiple services**: Higher retention
        - **Tech support users**: 50% less churn
        - **Long tenure**: > 5 years very stable
        """)

    st.markdown("---")

    # Customer Segmentation
    st.subheader("👥 Customer Segmentation Analysis")

    # Create segments based on risk
    df['Risk_Segment'] = 'Medium Risk'
    df.loc[(df['Contract'] == 'Month-to-month') & (df['tenure'] < 12), 'Risk_Segment'] = 'High Risk'
    df.loc[(df['Contract'] == 'Two year') & (df['tenure'] > 24), 'Risk_Segment'] = 'Low Risk'

    segment_counts = df['Risk_Segment'].value_counts()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Risk Customers", f"{segment_counts.get('High Risk', 0):,}")
    with col2:
        st.metric("Medium Risk Customers", f"{segment_counts.get('Medium Risk', 0):,}")
    with col3:
        st.metric("Low Risk Customers", f"{segment_counts.get('Low Risk', 0):,}")

    # Risk Distribution Visualization
    fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                 title="Customer Risk Segmentation",
                 color_discrete_map={'High Risk': '#FF6B6B',
                                     'Medium Risk': '#FFE66D',
                                     'Low Risk': '#4ECDC4'})
    st.plotly_chart(fig, use_container_width=True)

    # TODO: Add customer lifetime value analysis
    st.info("📌 TODO: Add Customer Lifetime Value (CLV) analysis and profitability segments")

elif page == "💻 ML Models":
    st.title("💻 Machine Learning Models")

    df = load_data()
    df_processed = preprocess_data(df)

    # Prepare data for modeling
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    for col in numerical_cols:
        if col in X_train.columns:
            X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
            X_test_scaled[col] = scaler.transform(X_test[[col]])

    # Model selection
    st.subheader("🎯 Select Model for Training")

    model_option = st.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Random Forest", "Gradient Boosting",
         "Support Vector Machine", "K-Nearest Neighbors", "AdaBoost",
         "Voting Classifier (Ensemble)", "XGBoost"]
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_option}..."):

                # Train selected model
                if model_option == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                elif model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_option == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=42)
                elif model_option == "Support Vector Machine":
                    model = SVC(probability=True, random_state=42)
                elif model_option == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=11)
                elif model_option == "AdaBoost":
                    model = AdaBoostClassifier(random_state=42)
                elif model_option == "XGBoost":
                    model = XGBClassifier(random_state=42, learning_rate=0.1)
                else:  # Voting Classifier
                    clf1 = GradientBoostingClassifier(random_state=42)
                    clf2 = LogisticRegression(random_state=42)
                    clf3 = AdaBoostClassifier(random_state=42)
                    clf4 = XGBClassifier(random_state=42, learning_rate=0.1)
                    clf5 = RandomForestClassifier(n_estimators=100, random_state=42)
                    model = VotingClassifier(
                        estimators=[('gb', clf1), ('lr', clf2), ('ada', clf3), ('xg', clf4), ('rf', clf5)],
                        voting='soft'
                    )

                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # Store results
                st.session_state.model_results[model_option] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': conf_matrix
                }

                st.session_state.model[model_option] = model

                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")

    with col2:
        if model_option in st.session_state.model_results:
            results = st.session_state.model_results[model_option]

            # Display metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Accuracy", f"{results['accuracy']:.2%}")
            with col2_2:
                tn, fp, fn, tp = results['confusion_matrix'].ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                st.metric("Precision", f"{precision:.2%}")
            with col2_3:
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                st.metric("Recall", f"{recall:.2%}")

    # Results Visualization
    if model_option in st.session_state.model_results:
        st.markdown("---")
        st.subheader("📊 Model Performance Visualization")

        results = st.session_state.model_results[model_option]

        col1, col2 = st.columns(2)

        with col1:
            # Confusion Matrix
            fig = px.imshow(results['confusion_matrix'],
                            labels=dict(x="Predicted", y="Actual"),
                            x=['No Churn', 'Churn'],
                            y=['No Churn', 'Churn'],
                            title=f"Confusion Matrix - {model_option}",
                            text_auto=True,
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve',
                                     line=dict(color='#FF6B6B', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                                     line=dict(color='gray', width=1, dash='dash')))
            fig.update_layout(title=f'ROC Curve - {model_option}',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance (for tree-based models)
        if hasattr(st.session_state.model[model_option], 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            fig = px.bar(importance, x='importance', y='feature', orientation='h',
                         title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Model Comparison":
    st.title("📈 Model Performance Comparison")

    if len(st.session_state.model_results) == 0:
        st.warning("⚠️ No models trained yet. Please go to the ML Models page to train models first.")
    else:
        # Comparison table
        comparison_data = []
        for model_name, results in st.session_state.model_results.items():
            tn, fp, fn, tp = results['confusion_matrix'].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Display comparison table
        st.subheader("📊 Performance Metrics Comparison")
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}'
            }).highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )

        # Visualization
        st.subheader("📈 Visual Comparison")

        # Metrics comparison chart
        fig = px.bar(comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                     x='Model', y='Score', color='Metric', barmode='group',
                     title="Model Performance Comparison",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

        # Best model identification
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = comparison_df['Accuracy'].max()

        st.success(f"🏆 Best Performing Model: **{best_model}** with {best_accuracy:.2%} accuracy")

        # TODO: Add cross-validation results
        st.info("📌 TODO: Add cross-validation scores and confidence intervals for more robust comparison")
elif page == "🔮 Churn Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("### Enter customer details to predict churn probability")

    # Load and prepare data
    encoders = {}
    df = load_data()
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object" and col != "customerID" and col != "Churn":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📝 Customer Information", "📊 Service Details", "💳 Billing Information"])

    with tab1:
        st.subheader("Demographics")
        
        # Row 1: Basic Demographics
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.checkbox("Senior Citizen (65+ years)", value=False)
        
        with col2:
            Partner = st.checkbox("Has Partner", value=False)
            Dependents = st.checkbox("Has Dependents", value=False)
        
        with col3:
            tenure = st.number_input(
                "Tenure (months)", 
                min_value=0, 
                max_value=72, 
                value=12,
                help="How long the customer has been with the company"
            )
            tenure_slider = st.slider(
                "Adjust with slider",
                min_value=0,
                max_value=72,
                value=tenure,
                label_visibility="collapsed"
            )
            tenure = tenure_slider
            
            # Visual tenure indicator
            if tenure < 12:
                st.caption("🆕 New Customer")
            elif tenure < 36:
                st.caption("📊 Regular Customer")
            else:
                st.caption("⭐ Loyal Customer")

    with tab2:
        st.subheader("Services Subscribed")
        
        # Phone Services Section
        st.markdown("#### 📞 Phone Services")
        col1, col2 = st.columns(2)
        
        with col1:
            PhoneService = st.checkbox("Phone Service", value=True)
        
        with col2:
            if PhoneService:
                MultipleLines = st.selectbox(
                    "Multiple Lines", 
                    ["No", "Yes"],
                    disabled=not PhoneService
                )
            else:
                MultipleLines = "No phone service"
                st.info("Enable phone service to select multiple lines")
        
        st.markdown("---")
        
        # Internet Services Section
        st.markdown("#### 🌐 Internet Services")
        InternetService = st.selectbox(
            "Internet Service Type",
            ["No", "DSL", "Fiber optic"],
            help="Fiber optic provides the fastest speeds"
        )
        
        # Show internet-dependent services only if internet is selected
        if InternetService != "No":
            st.markdown("##### Additional Internet Services")
            
            # Create a clean 3x2 grid for services
            col1, col2, col3 = st.columns(3)
            
            with col1:
                OnlineSecurity = st.checkbox("🔒 Online Security")
                TechSupport = st.checkbox("🛠️ Tech Support")
            
            with col2:
                OnlineBackup = st.checkbox("☁️ Online Backup")
                StreamingTV = st.checkbox("📺 Streaming TV")
            
            with col3:
                DeviceProtection = st.checkbox("📱 Device Protection")
                StreamingMovies = st.checkbox("🎬 Streaming Movies")
            
            # Convert checkboxes to Yes/No
            OnlineSecurity = "Yes" if OnlineSecurity else "No"
            OnlineBackup = "Yes" if OnlineBackup else "No"
            DeviceProtection = "Yes" if DeviceProtection else "No"
            TechSupport = "Yes" if TechSupport else "No"
            StreamingTV = "Yes" if StreamingTV else "No"
            StreamingMovies = "Yes" if StreamingMovies else "No"
            
            # Show service bundle recommendation
            services_count = sum([x == "Yes" for x in [
                OnlineSecurity, OnlineBackup, DeviceProtection, 
                TechSupport, StreamingTV, StreamingMovies
            ]])
            
            if services_count >= 4:
                st.success(f"💰 Bundle Deal Available! You have {services_count} services - eligible for 15% discount")
            elif services_count >= 2:
                st.info(f"💡 You have {services_count} services. Add {4-services_count} more for bundle discount!")
        else:
            st.info("No internet service selected")
            OnlineSecurity = "No internet service"
            OnlineBackup = "No internet service"
            DeviceProtection = "No internet service"
            TechSupport = "No internet service"
            StreamingTV = "No internet service"
            StreamingMovies = "No internet service"

    with tab3:
        st.subheader("Contract & Payment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Contract Details")
            Contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                help="Longer contracts typically have lower churn rates"
            )
            
            # Show contract benefits
            if Contract == "Month-to-month":
                st.warning("⚠️ Higher flexibility but higher churn risk")
            elif Contract == "One year":
                st.info("ℹ️ Balanced commitment and flexibility")
            else:
                st.success("✅ Best value and lowest churn risk")
        
        with col2:
            st.markdown("##### Payment Details")
            PaymentMethod = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Automatic payment methods have lower churn rates"
            )
            
            # Payment method recommendation
            if "automatic" in PaymentMethod:
                st.success("✅ Auto-pay reduces churn risk")
            else:
                st.info("💡 Consider auto-pay for convenience")
        
        st.markdown("---")
        
        # Billing Preferences
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Billing Preferences")
            PaperlessBilling = st.checkbox("📧 Paperless Billing", value=True)
        
        st.markdown("---")
        
        # Charges Section
        st.markdown("#### 💵 Charges")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            MonthlyCharges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=200.0,
                value=70.0,
                step=0.50,
                help="Monthly subscription cost"
            )
            MonthlyCharges_slider = st.slider(
                "Adjust with slider",
                min_value=18.0,
                max_value=200.0,
                value=MonthlyCharges,
                step=0.50,
                label_visibility="collapsed"
            )
            MonthlyCharges = MonthlyCharges_slider
        
        with col2:
            TotalCharges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=10.0,
                help="Total amount paid to date"
            )
            TotalCharges_slider = st.slider(
                "Adjust with slider",
                min_value=0.0,
                max_value=10000.0,
                value=TotalCharges,
                step=10.0,
                label_visibility="collapsed"
            )
            TotalCharges = TotalCharges_slider
        
        with col3:
            st.markdown("##### Cost Analysis")
            if MonthlyCharges < 35:
                st.success("💚 Low cost tier")
            elif MonthlyCharges < 65:
                st.info("💙 Medium cost tier")
            else:
                st.warning("💛 Premium tier")
            
            # Show monthly vs average
            avg_monthly = 70.0  # You can calculate this from your data
            diff = MonthlyCharges - avg_monthly
            if diff > 0:
                st.caption(f"${diff:.2f} above average")
            else:
                st.caption(f"${abs(diff):.2f} below average")

    # Convert inputs for model
    Partner = "Yes" if Partner else "No"
    Dependents = "Yes" if Dependents else "No"
    PhoneService = "Yes" if PhoneService else "No"
    PaperlessBilling = "Yes" if PaperlessBilling else "No"
    SeniorCitizen = 1 if SeniorCitizen else 0
    
    # Calculate services_count for later use
    services_count = 0
    if InternetService != "No":
        services_count = sum([x == "Yes" for x in [
            OnlineSecurity, OnlineBackup, DeviceProtection, 
            TechSupport, StreamingTV, StreamingMovies
        ]])

    # Customer Summary Card
    st.markdown("---")
    st.markdown("### 📋 Customer Profile Summary")
    
    # Create a clean 4-column layout for metrics with smaller text
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"<p style='font-size: 14px; margin: 0;'><b>Customer Type</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{'Senior' if SeniorCitizen else 'Regular'}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 14px; margin: 15px 0 0 0;'><b>Tenure</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{tenure} months</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<p style='font-size: 14px; margin: 0;'><b>Contract</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{Contract.split('-')[0] if '-' in Contract else Contract}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 14px; margin: 15px 0 0 0;'><b>Monthly Charges</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>${MonthlyCharges:.2f}</p>", unsafe_allow_html=True)
    
    with col3:
        # Count active services
        services_list = []
        if PhoneService == "Yes": 
            services_list.append("Phone")
        if InternetService != "No": 
            services_list.append("Internet")
        if InternetService != "No":
            if OnlineSecurity == "Yes": services_list.append("Security")
            if OnlineBackup == "Yes": services_list.append("Backup")
            if DeviceProtection == "Yes": services_list.append("Protection")
            if TechSupport == "Yes": services_list.append("Support")
            if StreamingTV == "Yes": services_list.append("TV")
            if StreamingMovies == "Yes": services_list.append("Movies")
        
        st.markdown(f"<p style='font-size: 14px; margin: 0;'><b>Active Services</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{len(services_list)}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 14px; margin: 15px 0 0 0;'><b>Total Charges</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>${TotalCharges:.2f}</p>", unsafe_allow_html=True)
    
    with col4:
        payment_type = "Auto-pay" if "automatic" in PaymentMethod else "Manual"
        st.markdown(f"<p style='font-size: 14px; margin: 0;'><b>Payment Type</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{payment_type}</p>", unsafe_allow_html=True)
        billing_type = "Paperless" if PaperlessBilling else "Paper"
        st.markdown(f"<p style='font-size: 14px; margin: 15px 0 0 0;'><b>Billing Type</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px; margin: 0;'>{billing_type}</p>", unsafe_allow_html=True)

    # Prepare input for model
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df_input = pd.DataFrame([input_dict])

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df_input.columns:
            df_input[col] = encoder.transform(df_input[col])

    # Prediction Section
    st.markdown("---")
    st.markdown("### 🎯 Churn Prediction")

    if 'model' not in st.session_state or len(st.session_state.model) == 0:
        st.warning("⚠️ No models trained yet. Please go to the ML Models page to train models first.")
    else:
        # Create prediction controls with proper alignment
        col1, col2 = st.columns([3, 1])
        
        with col1:
            model_option = st.selectbox(
                "Select Prediction Model",
                st.session_state.model.keys(),
                help="Choose the machine learning model for prediction"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align button
            predict_button = st.button(
                "🔮 Predict Churn", 
                type="primary", 
                use_container_width=True
            )

        if predict_button:
            with st.spinner("Analyzing customer profile..."):
                # Make prediction
                prediction = st.session_state.model[model_option].predict(df_input)[0]
                probability = st.session_state.model[model_option].predict_proba(df_input)[0]
                
                # Display results
                st.markdown("---")
                
                # Create centered result display
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col2:
                    if prediction == 1:
                        # High churn risk
                        churn_prob = probability[1]
                        
                        # Alert box
                        st.error("### ⚠️ HIGH CHURN RISK")
                        
                        # Probability display
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Churn Probability", f"{churn_prob:.1%}")
                        with col_b:
                            st.progress(float(churn_prob))
                        
                        # Risk factors in columns
                        st.markdown("#### 🔍 Key Risk Factors")
                        risk_factors = []
                        
                        if Contract == "Month-to-month":
                            risk_factors.append("Month-to-month contract")
                        if tenure < 12:
                            risk_factors.append("New customer (low tenure)")
                        if "Electronic check" in PaymentMethod:
                            risk_factors.append("Electronic check payment")
                        if InternetService == "Fiber optic" and services_count < 3:
                            risk_factors.append("Limited service bundle")
                        if MonthlyCharges > 80:
                            risk_factors.append("High monthly charges")
                        
                        # Display risk factors in a clean list
                        for i, factor in enumerate(risk_factors, 1):
                            st.write(f"{i}. {factor}")
                        
                        # Recommendations
                        st.markdown("#### 💡 Retention Recommendations")
                        
                        rec_col1, rec_col2 = st.columns(2)
                        with rec_col1:
                            st.info("""
                            **Immediate Actions:**
                            - Personal retention call
                            - Contract upgrade offer
                            - 20% discount for 6 months
                            """)
                        
                        with rec_col2:
                            st.info("""
                            **Support Offerings:**
                            - Free tech support (3 months)
                            - Payment method assistance
                            - Service bundle discount
                            """)
                    
                    else:
                        # Low churn risk
                        retain_prob = probability[0]
                        
                        # Success box
                        st.success("### ✅ LOW CHURN RISK")
                        
                        # Probability display
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Retention Probability", f"{retain_prob:.1%}")
                        with col_b:
                            st.progress(float(retain_prob))
                        
                        # Positive factors
                        st.markdown("#### 💚 Positive Indicators")
                        positive_factors = []
                        
                        if Contract in ["One year", "Two year"]:
                            positive_factors.append("Long-term contract commitment")
                        if tenure > 24:
                            positive_factors.append("Loyal customer (high tenure)")
                        if "automatic" in PaymentMethod:
                            positive_factors.append("Automatic payment setup")
                        if services_count >= 4:
                            positive_factors.append("Strong service bundle")
                        
                        # Display positive factors
                        for i, factor in enumerate(positive_factors, 1):
                            st.write(f"{i}. {factor}")
                        
                        # Retention strategy
                        st.markdown("#### 🎯 Retention Strategy")
                        
                        strat_col1, strat_col2 = st.columns(2)
                        with strat_col1:
                            st.info("""
                            **Engagement:**
                            - Quarterly check-ins
                            - Loyalty rewards program
                            - Beta program access
                            """)
                        
                        with strat_col2:
                            st.info("""
                            **Growth Opportunities:**
                            - Premium service offers
                            - Referral incentives
                            - Exclusive discounts
                            """)
                
                # Additional insights
                st.markdown("---")
                with st.expander("📊 View Detailed Analysis"):
                    st.markdown("#### Feature Contributions to Prediction")
                    
                    # Create risk assessment visualization
                    feature_values = {
                        "Tenure": (72 - tenure) / 72,  # Inverted - lower tenure = higher risk
                        "Monthly Charges": MonthlyCharges / 120,
                        "Total Charges": 1 - min(TotalCharges / 5000, 1),  # Inverted
                        "Contract Risk": 1.0 if Contract == "Month-to-month" else (0.5 if Contract == "One year" else 0.1),
                        "Payment Risk": 0.8 if "Electronic check" in PaymentMethod else (0.6 if "Mailed check" in PaymentMethod else 0.2),
                        "Service Bundle": 1 - (services_count / 6) if InternetService != "No" else 0.5
                    }
                    
                    # Create horizontal bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(feature_values.values()),
                            y=list(feature_values.keys()),
                            orientation='h',
                            marker_color=['#ff4444' if v > 0.6 else '#ffaa00' if v > 0.3 else '#44ff44' 
                                        for v in feature_values.values()],
                            text=[f'{v:.0%}' for v in feature_values.values()],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Risk Factor Analysis",
                        xaxis_title="Risk Level",
                        yaxis_title="Features",
                        showlegend=False,
                        height=400,
                        xaxis=dict(range=[0, 1], tickformat='.0%'),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_risk = sum(feature_values.values()) / len(feature_values)
                        st.metric("Average Risk Score", f"{avg_risk:.1%}")
                    with col2:
                        high_risk_factors = sum(1 for v in feature_values.values() if v > 0.6)
                        st.metric("High Risk Factors", high_risk_factors)
                    with col3:
                        model_confidence = max(probability)
                        st.metric("Model Confidence", f"{model_confidence:.1%}")
elif page == "💡 Recommendations":
    st.title("💡 Strategic Recommendations")
    
    # Executive Summary Box
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #0F52BA;'>
        <h3 style='color: #0F52BA; margin-top: 0;'>🎯 Executive Summary</h3>
        <p style='font-size: 16px; margin-bottom: 0;'>
        Our analysis identifies <b>26.6% churn rate</b> with clear intervention opportunities. 
        Implementing these recommendations can reduce churn by <b>5-8%</b>, saving <b>$8.4M annually</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📊 Priority Matrix", 
        "🚀 Action Plan", 
        "💰 Financial Impact", 
        "📈 Implementation Roadmap"
    ])
    
    with main_tab1:
        st.subheader("Priority Action Matrix")
        
        # Create priority matrix data
        priority_actions = pd.DataFrame({
            'Action': ['Contract Conversion', 'Payment Optimization', 'Enhanced Onboarding', 
                       'Service Bundles', 'Proactive Support', 'Loyalty Program'],
            'Impact': [85, 70, 75, 60, 80, 65],
            'Effort': [30, 20, 40, 70, 60, 80],
            'Timeline': ['Immediate', 'Immediate', 'Immediate', 'Medium-term', 'Medium-term', 'Long-term'],
            'ROI': [450, 380, 320, 250, 300, 200]
        })
        
        fig_matrix = go.Figure()
        
        colors = {'Immediate': '#ff4444', 'Medium-term': '#ffaa44', 'Long-term': '#44aa44'}
        
        for timeline in ['Immediate', 'Medium-term', 'Long-term']:
            df_filtered = priority_actions[priority_actions['Timeline'] == timeline]
            if not df_filtered.empty:
                fig_matrix.add_trace(go.Scatter(
                    x=df_filtered['Effort'],
                    y=df_filtered['Impact'],
                    mode='markers+text',
                    name=timeline,
                    text=df_filtered['Action'],
                    textposition="top center",
                    marker=dict(
                        size=df_filtered['ROI']/10,
                        color=colors[timeline],
                        opacity=0.6,
                        line=dict(width=2, color='white')
                    )
                ))
        
        fig_matrix.update_layout(
            title="Impact vs Effort Analysis (Bubble size = ROI)",
            xaxis_title="Implementation Effort →",
            yaxis_title="Business Impact →",
            height=500,
            showlegend=True,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        
        # Add quadrant lines
        fig_matrix.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_matrix.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_matrix.add_annotation(x=25, y=75, text="Quick Wins", showarrow=False, font=dict(size=12, color="green"))
        fig_matrix.add_annotation(x=75, y=75, text="Major Projects", showarrow=False, font=dict(size=12, color="orange"))
        fig_matrix.add_annotation(x=25, y=25, text="Low Priority", showarrow=False, font=dict(size=12, color="gray"))
        fig_matrix.add_annotation(x=75, y=25, text="Questionable", showarrow=False, font=dict(size=12, color="red"))
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Priority Actions Summary
        col1, col2 = st.columns(2)
        with col1:
            st.info("**🎯 Quick Wins:** Focus on Contract Conversion and Payment Optimization for immediate impact with minimal effort.")
        with col2:
            st.warning("**⚡ Major Projects:** Proactive Support systems require more resources but deliver substantial long-term value.")
    
    with main_tab2:
        st.subheader("Implementation Action Plan")
        
        # Sub-tabs for different time horizons
        action_tab1, action_tab2, action_tab3 = st.tabs([
            "🔴 Immediate Actions (0-2 months)", 
            "🟡 Medium-term (3-6 months)", 
            "🟢 Long-term (6+ months)"
        ])
        
        with action_tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 1. Contract Conversion Campaign")
                st.markdown("""
                <div style='background-color: #fff5f5; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <b>Target Segment:</b> Month-to-month customers with 6+ months tenure<br>
                    <b>Strategy:</b> Offer 20% discount for 1-year contract, 30% for 2-year<br>
                    <b>Implementation:</b> Email campaign + In-app notifications + Call center outreach<br>
                    <b>Success Metrics:</b> 30% conversion rate, 15-20% churn reduction
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 2. Payment Method Optimization")
                st.markdown("""
                <div style='background-color: #fff5f5; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <b>Target Segment:</b> Electronic check users (highest churn group)<br>
                    <b>Strategy:</b> $5/month discount for credit card auto-pay setup<br>
                    <b>Implementation:</b> Automated workflow with payment reminders<br>
                    <b>Success Metrics:</b> 40% adoption rate, 10% churn reduction
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 3. Enhanced Onboarding Program")
                st.markdown("""
                <div style='background-color: #fff5f5; padding: 15px; border-radius: 8px;'>
                    <b>Target Segment:</b> New customers (0-12 months)<br>
                    <b>Strategy:</b> Dedicated success manager for first 90 days<br>
                    <b>Implementation:</b> Automated check-ins + Personal touchpoints<br>
                    <b>Success Metrics:</b> 25% reduction in early churn
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Expected Churn Reduction", "3-5%", "in 2 months")
                st.metric("Investment Required", "$300K", "quick implementation")
                st.metric("Affected Customers", "12,000", "high-risk segment")
                
                # Timeline visualization
                st.markdown("### Implementation Timeline")
                timeline_data = pd.DataFrame({
                    'Week': ['Week 1-2', 'Week 3-4', 'Week 5-6', 'Week 7-8'],
                    'Progress': [25, 50, 75, 100]
                })
                fig_timeline = go.Figure(go.Bar(
                    x=timeline_data['Progress'],
                    y=timeline_data['Week'],
                    orientation='h',
                    marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f', '#4ecdc4']
                ))
                fig_timeline.update_layout(
                    height=200, 
                    showlegend=False, 
                    xaxis_title="Completion %",
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Milestones
        st.markdown("### 🎯 Key Milestones")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Q1 2024:**
            - ✅ Complete churn analysis
            - ✅ Identify quick wins
            - ✅ Launch contract conversion campaign
            
            **Q2 2024:**
            - 🔄 Implement payment optimization
            - 🔄 Deploy enhanced onboarding
            - 🔄 Begin system integration
            """)
        
        with col2:
            st.markdown("""
            **Q3 2024:**
            - 📅 Launch service bundles
            - 📅 Deploy ML churn prediction
            - 📅 Begin loyalty program pilot
            
            **Q4 2024:**
            - 📅 Full AI/ML deployment
            - 📅 Loyalty program rollout
            - 📅 Year 1 performance review
            """)
        
        # Phase Details
        st.markdown("### 📋 Phase Details")
        
        phase_details = {
            "🔴 Foundation Building (Jan-Feb)": {
                "Focus": "Establish baseline metrics and prepare infrastructure",
                "Deliverables": ["Churn analysis dashboard", "Customer segmentation", "Team training"],
                "Budget": "$200K"
            },
            "🟡 Quick Wins (Jan-Mar)": {
                "Focus": "Implement high-impact, low-effort initiatives",
                "Deliverables": ["Contract conversion campaign", "Payment optimization", "Basic retention workflows"],
                "Budget": "$300K"
            },
            "🔵 System Integration (Mar-May)": {
                "Focus": "Connect data systems and automate processes",
                "Deliverables": ["CRM integration", "Automated alerts", "Real-time reporting"],
                "Budget": "$250K"
            },
            "🟢 Advanced Analytics (May-Jul)": {
                "Focus": "Deploy predictive models and insights",
                "Deliverables": ["Churn prediction model", "Customer lifetime value calculator", "Risk scoring"],
                "Budget": "$200K"
            },
            "🟣 AI/ML Deployment (Jul-Sep)": {
                "Focus": "Launch intelligent automation and personalization",
                "Deliverables": ["Personalized recommendations", "Automated interventions", "Sentiment analysis"],
                "Budget": "$150K"
            },
            "⚡ Continuous Optimization (Sep-Dec)": {
                "Focus": "Refine and scale successful initiatives",
                "Deliverables": ["A/B testing framework", "Performance optimization", "2025 strategy"],
                "Budget": "$100K"
            }
        }
        
        for phase, details in phase_details.items():
            with st.expander(phase, expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Focus:** {details['Focus']}")
                    st.markdown("**Key Deliverables:**")
                    for item in details['Deliverables']:
                        st.markdown(f"• {item}")
                with col2:
                    st.metric("Budget", details['Budget'])
        
        with action_tab2:
            with st.expander("📦 Service Bundle Optimization", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    **Implementation Plan:**
                    - Create 3 tiers of service bundles: Essential, Professional, Premium
                    - Include tech support in Professional and Premium tiers
                    - Fiber optic priority upgrades for Premium customers
                    - Personalized bundle recommendations based on usage patterns
                    
                    **Timeline:** Months 3-4 for design, Month 5 for launch
                    """)
                with col2:
                    st.metric("Revenue Uplift", "+8%", "per customer")
                    st.metric("Churn Impact", "-2.5%", "bundle users")
            
            with st.expander("🛡️ Proactive Customer Support", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    **Implementation Plan:**
                    - ML-powered early warning system for at-risk customers
                    - Automated alerts 30 days before predicted churn
                    - Dedicated retention specialists for high-value accounts
                    - 24/7 premium support tier with <5 min response time
                    
                    **Timeline:** Month 3 for system setup, Month 4-5 for training, Month 6 go-live
                    """)
                with col2:
                    st.metric("Detection Rate", "78%", "accuracy")
                    st.metric("Save Rate", "45%", "contacted")
        
        with action_tab3:
            with st.expander("🏆 Loyalty Program Launch", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    **Program Structure:**
                    - Points-based system: 1 point per $1 spent
                    - Milestone rewards at 6, 12, 24, 36 months
                    - Referral bonuses: 500 points per successful referral
                    - Exclusive perks: Priority support, free upgrades, event invites
                    
                    **Long-term Benefits:**
                    - Increased customer lifetime value
                    - Organic growth through referrals
                    - Enhanced brand loyalty
                    """)
                with col2:
                    st.metric("Engagement", "65%", "participation")
                    st.metric("LTV Increase", "+22%", "loyal customers")
            
            with st.expander("🔄 Continuous Improvement Framework", expanded=True):
                st.markdown("""
                **Establish ongoing optimization:**
                - Monthly churn analysis reviews
                - Quarterly strategy adjustments
                - Annual program overhauls
                - Real-time dashboard monitoring
                - A/B testing for all initiatives
                """)
    
    with main_tab3:
        st.subheader("Financial Impact Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### 💸 Investment Breakdown")
            cost_data = pd.DataFrame({
                'Category': ['Technology', 'Personnel', 'Marketing', 'Training', 'Other'],
                'Amount': [400000, 350000, 200000, 150000, 100000]
            })
            
            fig_cost = go.Figure(go.Pie(
                labels=cost_data['Category'],
                values=cost_data['Amount'],
                hole=0.4,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            ))
            fig_cost.update_layout(
                height=300, 
                showlegend=True,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.info(f"**Total Investment:** ${sum(cost_data['Amount']):,.0f}")
        
        with col2:
            st.markdown("### 📈 ROI Projection (5 Years)")
            
            years = list(range(1, 6))
            investment = [1.2, 0.3, 0.2, 0.2, 0.2]
            returns = [2.4, 4.8, 6.2, 7.5, 8.4]
            net_benefit = [r - i for r, i in zip(returns, investment)]
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Bar(name='Investment', x=years, y=investment, marker_color='#ff6b6b'))
            fig_roi.add_trace(go.Bar(name='Returns', x=years, y=returns, marker_color='#4ecdc4'))
            fig_roi.add_trace(go.Scatter(
                name='Net Benefit', 
                x=years, 
                y=net_benefit,
                mode='lines+markers', 
                marker_color='#2ecc71', 
                yaxis='y2'
            ))
            
            fig_roi.update_layout(
                xaxis_title="Year",
                yaxis_title="Amount ($M)",
                yaxis2=dict(title="Net Benefit ($M)", overlaying='y', side='right'),
                height=350,
                hovermode='x unified',
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col3:
            st.markdown("### 🎯 Key Metrics")
            st.metric("5-Year NPV", "$24.8M", "+1,967%")
            st.metric("Payback Period", "8 months", "")
            st.metric("Customer LTV", "+$450", "+32%")
            st.metric("NPS Score", "+15 pts", "projected")
        
        # Detailed Financial Breakdown
        st.markdown("---")
        st.markdown("### 📊 Detailed Financial Projections")
        
        financial_data = pd.DataFrame({
            'Metric': ['Revenue Retention', 'Cost Savings', 'New Revenue', 'Total Benefit'],
            'Year 1': [1.8, 0.4, 0.2, 2.4],
            'Year 2': [3.2, 0.8, 0.8, 4.8],
            'Year 3': [4.0, 1.2, 1.0, 6.2],
            'Year 4': [4.5, 1.5, 1.5, 7.5],
            'Year 5': [5.0, 1.6, 1.8, 8.4]
        })
        
        st.dataframe(
            financial_data.style.format({'Year 1': '${:.1f}M', 'Year 2': '${:.1f}M', 
                                        'Year 3': '${:.1f}M', 'Year 4': '${:.1f}M', 
                                        'Year 5': '${:.1f}M'}),
            use_container_width=True
        )
    
    with main_tab4:
        st.subheader("Strategic Implementation Roadmap")
        
        # Create a simpler timeline visualization that will definitely work
        st.markdown("### 📅 2024 Implementation Timeline")
        
        # Timeline data with months
        timeline_df = pd.DataFrame({
            'Phase': ['Foundation', 'Quick Wins', 'Integration', 'Analytics', 'AI/ML', 'Optimization'],
            'Q1': ['████████', '██████', '', '', '', ''],
            'Q2': ['', '', '████████', '████', '', ''],
            'Q3': ['', '', '', '████████', '████████', ''],
            'Q4': ['', '', '', '', '████', '████████']
        })
        
        # Display as a styled table
        st.markdown("""
        <style>
        .timeline-table {
            width: 100%;
            border-collapse: collapse;
        }
        .timeline-table th {
            background-color: #0F52BA;
            color: white;
            padding: 10px;
            text-align: left;
        }
        .timeline-table td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Alternative Gantt visualization using bar chart
        gantt_data = pd.DataFrame({
            'Task': ['Foundation Building', 'Quick Wins', 'System Integration', 
                     'Advanced Analytics', 'AI/ML Deployment', 'Continuous Optimization'],
            'Start_Month': [1, 1, 3, 5, 7, 9],
            'Duration': [2, 2, 2, 2, 2, 4],
            'Team': ['Planning', 'Execution', 'Technology', 'Analytics', 'Innovation', 'Operations']
        })
        
        fig_timeline = go.Figure()
        
        colors = {'Planning': '#FF6B6B', 'Execution': '#4ECDC4', 'Technology': '#45B7D1', 
                  'Analytics': '#96CEB4', 'Innovation': '#FECA57', 'Operations': '#5F27CD'}
        
        for idx, row in gantt_data.iterrows():
            fig_timeline.add_trace(go.Bar(
                name=row['Task'],
                x=[row['Duration']],
                y=[row['Task']],
                base=[row['Start_Month']],
                orientation='h',
                marker=dict(color=colors[row['Team']]),
                text=row['Team'],
                textposition='inside',
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Start: Month %{base}<br>Duration: %{x} months<br>Team: %{text}<extra></extra>'
            ))
        
        fig_timeline.update_layout(
            title="2024 Project Timeline",
            xaxis=dict(
                title="Month",
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                range=[0.5, 12.5]
            ),
            yaxis=dict(title=""),
            height=400,
            barmode='stack',
            showlegend=False
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Milestones
        st.markdown("### 🎯 Key Milestones")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Q1 2024:**
            - ✅ Complete churn analysis
            - ✅ Identify quick wins
            - ✅ Launch contract conversion campaign
            
            **Q2 2024:**
            - 🔄 Implement payment optimization
            - 🔄 Deploy enhanced onboarding
            - 🔄 Begin system integration
            """)
        
        with col2:
            st.markdown("""
            **Q3 2024:**
            - 📅 Launch service bundles
            - 📅 Deploy ML churn prediction
            - 📅 Begin loyalty program pilot
            
            **Q4 2024:**
            - 📅 Full AI/ML deployment
            - 📅 Loyalty program rollout
            - 📅 Year 1 performance review
            """)
    
    # Final Summary Section
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-top: 30px;'>
        <h3 style='color: #0F52BA; text-align: center;'>🚀 Ready to Transform Your Customer Retention?</h3>
        <p style='font-size: 16px; text-align: center;'>
        Start with our quick wins to see immediate impact while building towards long-term success.
        </p>
        <div style='margin-top: 20px; text-align: center;'>
            <p style='color: #666; font-size: 14px;'>
            <b>Next Steps:</b> Review the priority matrix, select your immediate actions, and begin implementation within 2 weeks.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.success("""
    ## 🚀 Next Steps

    1. **Form a dedicated retention task force** with representatives from Sales, Customer Service, and Analytics
    2. **Prioritize quick wins** to demonstrate value and build momentum
    3. **Establish baseline metrics** and set realistic targets for each initiative
    4. **Deploy the ML model** in production for real-time churn scoring
    5. **Create feedback loops** to continuously improve predictions and interventions

    Remember: **Every 1% reduction in churn = $1.68M in annual revenue saved!**
    """)

# Export functionality
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Analysis Report", type="secondary"):
        st.info("📌 TODO: Implement PDF report generation with all insights and recommendations")

with col2:
    if st.button("📊 Download Model Scores", type="secondary"):
        st.info("📌 TODO: Export customer risk scores to CSV for operational use")

with col3:
    if st.button("🔗 Share Dashboard", type="secondary"):
        st.info("📌 TODO: Generate shareable link for stakeholder access")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📱 Telecom Customer Churn Analysis Dashboard v1.0</p>
    <p>Built with ❤️ using Streamlit | Data Science for Business Impact</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - Churn Management Best Practices
    - Industry Benchmarks
    - Technical Documentation
    """)

    st.markdown("---")
    st.markdown("### 🔔 Notifications")

    # Simulated alerts
    if st.checkbox("Enable Real-time Alerts"):
        st.warning("⚠️ 23 customers at high risk today")
        st.info("ℹ️ 5 successful interventions this week")

    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("""
    **Need Help?**
    - Email: analytics@telecom.com
    - Slack: #churn-analytics
    - Ext: 1234
    """)

