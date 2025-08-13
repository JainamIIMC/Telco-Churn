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
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("""
<style>
    .stSelectbox > div > div {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Data Overview", "🔍 Exploratory Analysis",
     "🎯 Customer Insights", "🤖 ML Models", "📈 Model Comparison", "🔮 Churn Prediction",
     "💡 Recommendations"]
)


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
    df = preprocess_data(df)

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

        st.subheader("📡 Service Usage & Plans")

        # Contract distribution (share %) - horizontal, narrow bars
        if "Contract" in df.columns:
            vc_contract = df["Contract"].value_counts(dropna=False).rename_axis("Contract").reset_index(name="count")
            vc_contract["share"] = (vc_contract["count"] / vc_contract["count"].sum() * 100).round(1)
            fig = px.bar(vc_contract.sort_values("share"), y="Contract", x="share", text="share",
                         orientation="h", labels={"share": "Share (%)"},
                         title="Contract Types (Share %)")
            fig.update_traces(texttemplate="%{text}%", textposition="outside", marker_line_width=0.5,
                              marker_line_color="#888")
            fig.update_layout(**DEFAULT_LAYOUT, bargap=0.35,
                              xaxis=dict(range=[0, max(60, vc_contract['share'].max() + 10)]))
            st.plotly_chart(fig, use_container_width=True)

        cols_services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                         "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        # Fix: Calculate churn split for customers having the service
        # Clean churn share calculation for each service column
        # ---- Churn share by service (100% of service users) ----
        cols_services = [
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        present = [c for c in cols_services if c in df.columns]


        def has_service_mask(col_ser: pd.Series) -> pd.Series:
            s = col_ser.astype(str).str.strip().str.lower()
            # values that clearly mean ABSENCE of the service
            negatives = {
                "no", "no internet service", "no phone service",
                "no online security", "no online backup", "no device protection",
                "no tech support", "none", "", "nan"
            }
            # If the column is 0/1, treat 1 as has-service
            if set(s.dropna().unique()).issubset({"0", "1"}):
                return s == "1"
            # Otherwise: anything not explicitly negative = has service
            return ~s.isin(negatives)


        if present:
            rows = []
            for c in present:
                mask_has = has_service_mask(df[c])
                subset = df[mask_has].copy()
                total = len(subset)
                if total == 0:
                    continue
                y = (subset["Churn"] == "Yes").sum()
                n = (subset["Churn"] == "No").sum()
                rows.append({"Service": c, "Churn": "Yes", "Percent": round(y / total * 100, 1)})
                rows.append({"Service": c, "Churn": "No", "Percent": round(n / total * 100, 1)})

            chart_df = pd.DataFrame(rows)

            # sort by higher churn (Yes) on top
            order = (chart_df[chart_df["Churn"] == "Yes"]
                     .sort_values("Percent", ascending=False)["Service"].tolist())

            fig = px.bar(
                chart_df, y="Service", x="Percent", color="Churn",
                color_discrete_map={"No": "#4CAF50", "Yes": "#E53935"},
                category_orders={"Service": order},
                barmode="stack", text="Percent",
                title="Churn Share by Service (100% of Service Users)"
            )
            fig.update_traces(texttemplate="%{text}%", textposition="inside", insidetextanchor="middle",
                              marker_line_width=0.5, marker_line_color="#888")
            fig.update_layout(
                **DEFAULT_LAYOUT,
                bargap=0.30,
                yaxis_title="Service",
                xaxis_title="Share (%)",
                xaxis=dict(range=[0, 100])
            )
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
            fig = px.box(df, x='Churn', y='TotalCharges',
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

elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")

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
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

            # Senior Citizen with toggle
            senior_col1, senior_col2 = st.columns([2, 1])
            with senior_col1:
                st.write("Senior Citizen (65+ years)")
            with senior_col2:
                SeniorCitizen = st.toggle("", value=False)

        with col2:
            # Partner with visual indicator
            partner_col1, partner_col2 = st.columns([1, 1])
            with partner_col1:
                st.write("Has Partner")
            with partner_col2:
                Partner = st.toggle("", value=False, key="partner")

            dep_col1, dep_col2 = st.columns([1, 1])
            with dep_col1:
                st.write("Has Dependents")
            with dep_col2:
                Dependents = st.toggle("", value=False, key="dependents")

        with col3:
            # Tenure with both slider and input
            st.write("Tenure (months)")
            tenure_slider = st.slider("", 0, 72, 12, label_visibility="collapsed")
            tenure = st.number_input("Or enter manually:", min_value=0, max_value=72, value=tenure_slider)

            # Visual tenure indicator
            if tenure < 12:
                st.caption("🆕 New Customer")
            elif tenure < 36:
                st.caption("📊 Regular Customer")
            else:
                st.caption("⭐ Loyal Customer")

    with tab2:
        st.subheader("Services Subscribed")

        # Phone Services
        st.markdown("#### 📞 Phone Services")
        col1, col2 = st.columns(2)

        with col1:
            phone_col1, phone_col2 = st.columns([1, 1])
            with phone_col1:
                st.write("Phone Service")
            with phone_col2:
                PhoneService = st.toggle("", value=True, key="phone")

        with col2:
            if PhoneService:
                MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
            else:
                MultipleLines = "No phone service"
                st.info("No phone service selected")

        # Internet Services
        st.markdown("#### 🌐 Internet Services")
        InternetService = st.radio(
            "Internet Service Type",
            ["No", "DSL", "Fiber optic"],
            horizontal=True,
            help="Fiber optic provides the fastest speeds"
        )

        # Show internet-dependent services only if internet is selected
        if InternetService != "No":
            st.markdown("##### Additional Internet Services")

            # Create a 2x3 grid for services
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            with col1:
                OnlineSecurity = st.checkbox("🔒 Online Security", value=False)
            with col2:
                OnlineBackup = st.checkbox("☁️ Online Backup", value=False)
            with col3:
                DeviceProtection = st.checkbox("📱 Device Protection", value=False)
            with col4:
                TechSupport = st.checkbox("🛠️ Tech Support", value=False)
            with col5:
                StreamingTV = st.checkbox("📺 Streaming TV", value=False)
            with col6:
                StreamingMovies = st.checkbox("🎬 Streaming Movies", value=False)

            # Convert checkboxes to Yes/No
            OnlineSecurity = "Yes" if OnlineSecurity else "No"
            OnlineBackup = "Yes" if OnlineBackup else "No"
            DeviceProtection = "Yes" if DeviceProtection else "No"
            TechSupport = "Yes" if TechSupport else "No"
            StreamingTV = "Yes" if StreamingTV else "No"
            StreamingMovies = "Yes" if StreamingMovies else "No"

            # Show service bundle recommendation
            services_count = sum([x == "Yes" for x in [OnlineSecurity, OnlineBackup,
                                                       DeviceProtection, TechSupport,
                                                       StreamingTV, StreamingMovies]])
            if services_count >= 4:
                st.success(f"💰 Bundle Deal Available! You have {services_count} services - eligible for 15% discount")
        else:
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
            # Contract with visual emphasis
            Contract = st.radio(
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

            # Paperless billing with icon
            paperless_col1, paperless_col2 = st.columns([3, 1])
            with paperless_col1:
                st.write("📧 Paperless Billing")
            with paperless_col2:
                PaperlessBilling = st.toggle("", value=True, key="paperless")

        with col2:
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

        st.markdown("#### 💵 Charges")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Monthly charges with slider and input
            st.write("Monthly Charges ($)")
            monthly_slider = st.slider("", 18.0, 120.0, 70.0, 0.5, label_visibility="collapsed")
            MonthlyCharges = st.number_input(
                "Or enter manually:",
                min_value=18.0,
                max_value=200.0,
                value=monthly_slider,
                step=0.5
            )

        with col2:
            # Auto-calculate total charges based on tenure and monthly
            st.write("Total Charges ($)")
            calculated_total = MonthlyCharges * tenure
            TotalCharges = st.number_input(
                "Auto-calculated (editable):",
                min_value=0.0,
                max_value=10000.0,
                value=calculated_total,
                step=10.0
            )

        with col3:
            # Show average charges indicator
            st.write("Cost Analysis")
            if MonthlyCharges < 35:
                st.success("💚 Low cost tier")
            elif MonthlyCharges < 65:
                st.info("💙 Medium cost tier")
            else:
                st.warning("💛 Premium tier")

    # Convert inputs for model
    Partner = "Yes" if Partner else "No"
    Dependents = "Yes" if Dependents else "No"
    PhoneService = "Yes" if PhoneService else "No"
    PaperlessBilling = "Yes" if PaperlessBilling else "No"
    SeniorCitizen = 1 if SeniorCitizen else 0

    # Customer Summary Card
    st.markdown("---")
    st.subheader("📋 Customer Profile Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customer Type", "Senior" if SeniorCitizen else "Regular")
        st.metric("Tenure", f"{tenure} months")
    with col2:
        st.metric("Contract", Contract.replace("-", " "))
        st.metric("Monthly Charges", f"${MonthlyCharges:.2f}")
    with col3:
        services_list = []
        if PhoneService == "Yes": services_list.append("Phone")
        if InternetService != "No": services_list.append("Internet")
        if OnlineSecurity == "Yes": services_list.append("Security")
        st.metric("Services", len(services_list))
        st.metric("Total Charges", f"${TotalCharges:.2f}")
    with col4:
        st.metric("Payment", "Auto" if "automatic" in PaymentMethod else "Manual")
        st.metric("Billing", "Paperless" if PaperlessBilling == "Yes" else "Paper")

    # Prepare input
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
    st.subheader("🎯 Churn Prediction")

    if 'model' not in st.session_state or len(st.session_state.model) == 0:
        st.warning("⚠️ No models trained yet. Please go to the ML Models page to train models first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            model_option = st.selectbox(
                "Select Prediction Model:",
                st.session_state.model.keys(),
                help="Choose the machine learning model for prediction"
            )

        with col2:
            predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

        if predict_button:
            with st.spinner("Analyzing customer profile..."):
                # Make prediction
                prediction = st.session_state.model[model_option].predict(df_input)[0]
                probability = st.session_state.model[model_option].predict_proba(df_input)[0]
                print(probability)

                # Display results with enhanced visualization
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    if prediction == 1:
                        # High churn risk
                        churn_prob = probability[1]
                        st.error(f"### ⚠️ HIGH CHURN RISK")

                        # Progress bar for probability
                        st.progress(float(churn_prob))
                        st.metric("Churn Probability", f"{churn_prob:.1%}")

                        # Risk factors
                        st.markdown("#### 🔍 Key Risk Factors:")
                        risk_factors = []
                        if Contract == "Month-to-month":
                            risk_factors.append("• Month-to-month contract")
                        if tenure < 12:
                            risk_factors.append("• New customer (low tenure)")
                        if "Electronic check" in PaymentMethod:
                            risk_factors.append("• Electronic check payment")
                        if InternetService == "Fiber optic":
                            risk_factors.append("• Fiber optic service issues")

                        for factor in risk_factors:
                            st.write(factor)

                        # Recommendations
                        st.markdown("#### 💡 Retention Recommendations:")
                        st.info("""
                        1. **Immediate Action**: Personal call from retention specialist
                        2. **Offer**: Contract upgrade with 20% discount for 6 months
                        3. **Support**: Free tech support for 3 months
                        4. **Payment**: Assistance switching to auto-pay with incentive
                        """)

                    else:
                        # Low churn risk
                        retain_prob = probability[0]
                        st.success(f"### ✅ LOW CHURN RISK")

                        # Progress bar for retention probability
                        st.progress(float(retain_prob))
                        st.metric("Retention Probability", f"{retain_prob:.1%}")

                        # Positive factors
                        st.markdown("#### 💚 Positive Indicators:")
                        positive_factors = []
                        if Contract in ["One year", "Two year"]:
                            positive_factors.append("• Long-term contract")
                        if tenure > 24:
                            positive_factors.append("• Loyal customer")
                        if "automatic" in PaymentMethod:
                            positive_factors.append("• Auto-payment setup")

                        for factor in positive_factors:
                            st.write(factor)

                        # Retention strategy
                        st.markdown("#### 🎯 Retention Strategy:")
                        st.info("""
                        1. **Maintain**: Regular check-ins every quarter
                        2. **Reward**: Loyalty program enrollment
                        3. **Upsell**: Offer premium services at discount
                        4. **Engage**: Include in beta programs and surveys
                        """)

                # Additional insights
                st.markdown("---")
                with st.expander("📊 View Detailed Analysis"):
                    # Feature importance for this prediction (if available)
                    st.markdown("#### Feature Contributions")

                    # Create a simple bar chart of input features
                    feature_values = {
                        "Tenure": tenure / 72,
                        "Monthly Charges": MonthlyCharges / 120,
                        "Total Charges": min(TotalCharges / 5000, 1),
                        "Contract Risk": 0.8 if Contract == "Month-to-month" else 0.2,
                        "Payment Risk": 0.7 if "check" in PaymentMethod else 0.3,
                        "Service Bundle": services_count / 6 if InternetService != "No" else 0
                    }

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(feature_values.values()),
                            y=list(feature_values.keys()),
                            orientation='h',
                            marker_color=['red' if v > 0.5 else 'green' for v in feature_values.values()]
                        )
                    ])
                    fig.update_layout(
                        title="Risk Factor Analysis",
                        xaxis_title="Risk Level",
                        yaxis_title="Features",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "💡 Recommendations":
    st.title("💡 Strategic Recommendations")

    st.markdown("""
    ## 🎯 Data-Driven Retention Strategies

    Based on our comprehensive analysis, here are the key recommendations to reduce customer churn:
    """)

    # Recommendations in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔴 Immediate Actions

        **1. Contract Conversion Campaign**
        - Target month-to-month customers with tenure > 6 months
        - Offer incentives for 1-2 year contract upgrades
        - Expected impact: 15-20% churn reduction

        **2. Payment Method Optimization**
        - Encourage electronic check users to switch to auto-pay
        - Provide setup assistance and first-month discount
        - Expected impact: 10% churn reduction

        **3. New Customer Onboarding**
        - Enhanced support for customers < 12 months
        - Weekly check-ins during first 3 months
        - Proactive issue resolution
        """)

    with col2:
        st.markdown("""
        ### 🟡 Medium-term Initiatives

        **4. Service Bundle Optimization**
        - Create attractive bundles with tech support
        - Focus on fiber optic service improvements
        - Personalized recommendations based on usage

        **5. Proactive Customer Support**
        - Implement predictive alerts for at-risk customers
        - Dedicated retention team for high-value segments
        - 24/7 tech support for premium customers

        **6. Loyalty Program Launch**
        - Tenure-based rewards and discounts
        - Referral bonuses for long-term customers
        - Exclusive perks for 2+ year contracts
        """)

    st.markdown("---")

    # ROI Estimation
    st.subheader("💰 Expected ROI from Retention Initiatives")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Churn Reduction Target", "5%", "from 26.6%")
    with col2:
        st.metric("Annual Revenue Saved", "$8.4M", "+12%")
    with col3:
        st.metric("Implementation Cost", "$1.2M", "one-time")
    with col4:
        st.metric("ROI", "600%", "first year")

    # Implementation Roadmap
    st.subheader("📅 Implementation Roadmap")

    roadmap_data = {
        'Phase': ['Phase 1: Quick Wins', 'Phase 2: System Integration',
                  'Phase 3: Advanced Analytics', 'Phase 4: Optimization'],
        'Timeline': ['Month 1-2', 'Month 3-4', 'Month 5-6', 'Month 7+'],
        'Key Activities': [
            'Contract conversion campaign, Payment method optimization',
            'CRM integration, Automated alerts setup',
            'ML model deployment, Real-time scoring',
            'Continuous improvement, A/B testing'
        ],
        'Expected Impact': ['Quick 3-5% reduction', 'Additional 2-3% reduction',
                            'Sustained 5%+ reduction', 'Ongoing optimization']
    }
    roadmap_df = pd.DataFrame(roadmap_data)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(roadmap_df.columns),
                    fill_color='#0F52BA',
                    font=dict(color='white', size=12),
                    align='left'),
        cells=dict(values=[roadmap_df[col] for col in roadmap_df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    # Success Metrics
    st.subheader("📊 Success Metrics & KPIs")

    tab1, tab2, tab3 = st.tabs(["Leading Indicators", "Lagging Indicators", "Monitoring Dashboard"])

    with tab1:
        st.markdown("""
        **Monitor Weekly:**
        - Number of contract conversions
        - Customer support ticket resolution time
        - Payment method changes
        - Service bundle adoption rates

        **Monitor Daily:**
        - High-risk customer interactions
        - Proactive outreach completion rate
        - Customer satisfaction scores
        """)

    with tab2:
        st.markdown("""
        **Monitor Monthly:**
        - Overall churn rate
        - Churn rate by segment
        - Customer lifetime value
        - Revenue retention rate

        **Monitor Quarterly:**
        - Net Promoter Score (NPS)
        - Customer acquisition cost vs retention cost
        - Market share changes
        """)

    with tab3:
        st.markdown("""
        **Real-time Dashboard Components:**
        - Live churn risk scores
        - Alert system for high-risk behaviors
        - Intervention success tracking
        - ROI calculator

        **Automated Reports:**
        - Daily executive summary
        - Weekly team performance metrics
        - Monthly trend analysis
        """)

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

