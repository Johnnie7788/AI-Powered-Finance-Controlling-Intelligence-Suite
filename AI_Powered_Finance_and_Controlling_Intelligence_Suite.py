#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import joblib
from prophet import Prophet  
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text
import os
import io
import xlsxwriter
import shutil

# Set up Streamlit Page
st.set_page_config(page_title="AI-Powered Finance & Controlling Suite", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        .title-container {
            background-color: #1E3A8A;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
        }
        .title-text {
            color: white;
            font-size: 32px;
            font-weight: bold;
        }
        .stApp {
            background-color: #F5F5F5;
        }
    </style>
    <div class='title-container'>
        <span class='title-text'>AI-Powered Finance & Controlling Intelligence Suite</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Database Connection
def get_db_connection():
    engine = create_engine("sqlite:///finance_data.db")
    return engine

# Initialize Database
def initialize_database():
    conn = get_db_connection()
    with conn.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_data (
                Date TEXT PRIMARY KEY,
                Revenue REAL,
                Expenses REAL,
                Profit REAL,
                Operating_Cash_Flow REAL,
                Debt_to_Equity_Ratio REAL,
                EBITDA REAL,
                Net_Profit_Margin REAL,
                ROI REAL,
                ROE REAL,
                Region TEXT,
                Transaction_Volume INTEGER,
                Compliance_Check TEXT,
                Anomaly_Score INTEGER,
                Industry_Inflation_Rate REAL,
                Market_Trend_Index REAL,
                Competitor_Pricing_Index REAL
            )
        """))
        connection.commit()

initialize_database()

# Sidebar File Upload
st.sidebar.header("Upload Your Financial Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

df = load_data(uploaded_file)

# Ensure Data is Loaded
if df.empty:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Sidebar Filters
if 'Region' in df.columns:
    st.sidebar.header("Filters")
    selected_regions = st.sidebar.multiselect("Select Regions", df['Region'].unique(), default=df['Region'].unique())

    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=df['Date'].min().to_pydatetime(),
        max_value=df['Date'].max().to_pydatetime(),
        value=(df['Date'].min().to_pydatetime(), df['Date'].max().to_pydatetime())
    )

    df = df[df['Region'].isin(selected_regions)]
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Sidebar Navigation
st.sidebar.header("Navigation")
selected_option = st.sidebar.radio("Select a Section", ["Dashboard", "AI Forecasting", "Data Quality & Governance", "KPI Monitoring", "Anomaly Detection", "Financial Compliance", "Export Reports"])

# Navigation Functions
def show_dashboard():
    st.subheader("Business Intelligence Dashboard")
    col1, col2 = st.columns(2)

    if "Revenue" in df.columns and "Profit" in df.columns:
        with col1:
            st.markdown("### Revenue Trends")
            fig1 = px.line(df, x='Date', y='Revenue', title="Monthly Revenue Trends", markers=True)
            st.plotly_chart(fig1)

        with col2:
            st.markdown("### Profit Analysis")
            fig2 = px.bar(df, x='Date', y='Profit', title="Monthly Profit Trends", color="Profit", color_continuous_scale="Blues")
            st.plotly_chart(fig2)

    if "Region" in df.columns and "Revenue" in df.columns:
        st.markdown("### Regional Performance")
        fig3 = px.treemap(df, path=['Region'], values='Revenue', title="Revenue Contribution by Region")
        st.plotly_chart(fig3)


# AI Forecasting with Random Forest
def show_forecasting(df):
    if df.empty:
        st.error("⚠️ No data available. Please upload a dataset first.")
        return

    if len(df) < 15:
        st.error("⚠️ Not enough data points for forecasting. Please upload at least 15 records.")
        return

    df = df.dropna(subset=['Date'])
    df['Timestamp'] = df['Date'].map(pd.Timestamp.toordinal)
    X = df[['Timestamp']]
    y = df['Revenue']

    # Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Generate Future Dates
    future_dates = pd.date_range(start=df['Date'].max(), periods=12, freq='M')
    future_timestamps = np.array([pd.Timestamp(date).toordinal() for date in future_dates]).reshape(-1, 1)

    # Make Predictions
    future_predictions = model.predict(future_timestamps)

    # Compute Confidence Intervals
    pred_std = np.std(y - model.predict(X))
    lower_bound = future_predictions - (1.96 * pred_std)
    upper_bound = future_predictions + (1.96 * pred_std)

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], mode='markers', name='Actual', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', name='Upper Bound', line=dict(color='lightblue', dash='dot')))
    fig.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='lines', name='Lower Bound', line=dict(color='lightblue', dash='dot')))
    fig.update_layout(title="📈 AI-Driven Sales & Revenue Forecasting", xaxis_title="Date", yaxis_title="Revenue ($)", legend_title="Legend", template="plotly_white")

    st.plotly_chart(fig)



def show_data_quality():
    st.subheader("📊 Data Quality & Governance")

    # ✅ 1. Display Basic Data Summary
    st.markdown("### 🔍 Overview of Data")
    st.write(df.describe())

    # ✅ 2. Missing Values Check
    st.markdown("### ⚠️ Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percent})
    st.write(missing_df[missing_df["Missing Values"] > 0])

    # ✅ 3. Duplicate Rows Check
    st.markdown("### 🔄 Duplicate Data Check")
    duplicate_count = df.duplicated().sum()
    st.write(f"Total Duplicate Rows: **{duplicate_count}**")

    # ✅ 4. Data Type Validation
    st.markdown("### 📌 Data Type Validation")
    dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.write(dtype_df)

    # ✅ 5. Outlier Detection (Using Z-Score)
    st.markdown("### 🚨 Outlier Detection")
    numeric_columns = df.select_dtypes(include=['number']).columns
    z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
    outliers = (z_scores > 3).sum()
    st.write(pd.DataFrame({"Outlier Count": outliers}))

    # ✅ 6. Data Distribution Visualization
    st.markdown("### 📈 Data Distribution")
    selected_column = st.selectbox("Select a Numeric Column", numeric_columns)
    fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
    st.plotly_chart(fig)



def show_kpi_monitoring():
    st.subheader("📊 Key Performance Indicators (KPI) Monitoring")

    # ✅ Check if required KPIs are available
    kpi_columns = {'Revenue', 'Profit', 'ROI', 'ROE'}
    available_kpis = kpi_columns.intersection(df.columns)

    if not available_kpis:
        st.warning("⚠️ No relevant KPI columns found. Please upload a dataset with Revenue, Profit, ROI, or ROE.")
        return

    # ✅ 1. KPI Selection
    selected_kpi = st.selectbox("📌 Select a KPI to Monitor", list(available_kpis))

    # ✅ 2. KPI Summary & Threshold Comparison
    kpi_mean = df[selected_kpi].mean()
    kpi_median = df[selected_kpi].median()
    kpi_min = df[selected_kpi].min()
    kpi_max = df[selected_kpi].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"📈 Avg {selected_kpi}", f"{kpi_mean:,.2f}")
    col2.metric(f"🔍 Median {selected_kpi}", f"{kpi_median:,.2f}")
    col3.metric(f"⬇️ Min {selected_kpi}", f"{kpi_min:,.2f}")
    col4.metric(f"⬆️ Max {selected_kpi}", f"{kpi_max:,.2f}")

    # ✅ 3. KPI Trend Visualization
    st.markdown(f"### 📉 {selected_kpi} Trend Over Time")
    fig = px.line(df, x="Date", y=selected_kpi, title=f"{selected_kpi} Trend", markers=True)
    st.plotly_chart(fig)

    # ✅ 4. KPI Performance (Last 6 Months vs. Previous 6 Months)
    st.markdown("### 🔄 Comparative KPI Performance")
    df["YearMonth"] = df["Date"].dt.to_period("M")  # Convert Date to Year-Month Format
    recent_months = df["YearMonth"].unique()[-12:]  # Get last 12 months
    
    if len(recent_months) >= 12:
        last_6_months = df[df["YearMonth"].isin(recent_months[-6:])][selected_kpi].mean()
        prev_6_months = df[df["YearMonth"].isin(recent_months[:6])][selected_kpi].mean()
        change_percentage = ((last_6_months - prev_6_months) / prev_6_months) * 100

        col1, col2 = st.columns(2)
        col1.metric("📅 Last 6 Months", f"{last_6_months:,.2f}")
        col2.metric("⏳ Previous 6 Months", f"{prev_6_months:,.2f}", f"{change_percentage:.2f}% change")

    # ✅ 5. KPI Bar Chart (Revenue & Profit Comparison)
    if {'Revenue', 'Profit'}.issubset(df.columns):
        st.markdown("### 📊 Revenue vs. Profit Comparison")
        fig = px.bar(df, x="Date", y=["Revenue", "Profit"], barmode="group", title="Revenue vs. Profit Trends")
        st.plotly_chart(fig)



# Define Anomaly Detection Function Before Calling It
def show_anomaly_detection():
    st.subheader('🚨 AI-Powered Anomaly Detection in Financial Data')

    # Ensure required features exist
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_features = st.multiselect("📌 Select Features for Anomaly Detection", list(numeric_columns), default=['Revenue', 'Expenses', 'Profit'])

    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least two numerical features for anomaly detection.")
        st.stop()

    # Handle missing values
    df_clean = df.dropna(subset=selected_features)

    # Normalize Data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean[selected_features])

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_scores = iso_forest.fit_predict(df_scaled)
    df_clean['Anomaly_Score'] = np.where(anomaly_scores == -1, "Anomaly", "Normal")

    # Display Anomalies
    anomalies = df_clean[df_clean['Anomaly_Score'] == "Anomaly"]
    st.markdown("### 🚨 Detected Anomalies")
    st.dataframe(anomalies)

    # Visualization of Anomalies
    if len(selected_features) >= 2:
        st.markdown("### 📉 Anomaly Detection Scatter Plot")
        fig = px.scatter(df_clean, x=selected_features[0], y=selected_features[1], 
                         color=df_clean['Anomaly_Score'], 
                         title="Detected Anomalies", 
                         color_discrete_map={"Normal": "blue", "Anomaly": "red"})
        st.plotly_chart(fig)

# Ensure the Function is Called Correctly
if selected_option == "Anomaly Detection":
    show_anomaly_detection()



elif selected_option == 'Financial Compliance':  
    st.subheader('📑 AI-Powered Financial Compliance Analysis')

    # Handle Missing Compliance Data
    df = df.dropna(subset=['Compliance_Check'])

    # Compliance Summary (Count and Percentage)
    compliance_summary = df['Compliance_Check'].value_counts()
    compliance_percentage = df['Compliance_Check'].value_counts(normalize=True) * 100

    # Display Compliance Breakdown
    st.markdown("### 🏆 Compliance Status Breakdown")
    fig = px.bar(compliance_summary, x=compliance_summary.index, y=compliance_summary.values, 
                 labels={'x': 'Compliance Status', 'y': 'Count'}, title="Financial Compliance Overview")
    st.plotly_chart(fig)

    # Show Compliance Percentage
    st.markdown("### 📊 Compliance Percentage Breakdown")
    compliance_df = pd.DataFrame({'Compliance Status': compliance_percentage.index, 'Percentage': compliance_percentage.values})
    st.dataframe(compliance_df)

    # Categorize Compliance Failures by Severity
    compliance_levels = {
        "Minor": ["Late Filing", "Incorrect Invoice"],
        "Moderate": ["Policy Violation", "Data Breach"],
        "Critical": ["Fraud", "Regulatory Breach"]
    }

    def classify_compliance(issue):
        if pd.isna(issue):
            return "Unknown"
        for level, issues in compliance_levels.items():
            if any(iss in str(issue) for iss in issues):
                return level
        return "Unknown"

    df['Compliance_Severity'] = df['Compliance_Check'].astype(str).apply(classify_compliance)

    # Display Compliance Severity Breakdown
    st.markdown("### 🚨 Compliance Violation Severity Breakdown")
    severity_counts = df['Compliance_Severity'].value_counts().reset_index()
    fig2 = px.pie(severity_counts, names='index', values='Compliance_Severity', title="Compliance Violation Severity Breakdown")
    st.plotly_chart(fig2)




def show_export():
    st.subheader("📤 Export Reports")

    # ✅ User Selects File Format
    export_format = st.radio("📌 Select Export Format", ["Excel (.xlsx)", "CSV (.csv)", "ZIP (Compressed)"])

    # ✅ User Selects Sections to Export
    sections = {
        "📊 Financial Data": df,
        "📉 KPI Summary": df[['Date', 'Revenue', 'Profit', 'ROI', 'ROE']] if {'Revenue', 'Profit', 'ROI', 'ROE'}.issubset(df.columns) else None,
        "🚨 Anomaly Report": df[df["Anomaly_Score"] == "Anomaly"] if "Anomaly_Score" in df.columns else None,
        "📑 Compliance Failures": df[df["Compliance_Check"] == "Failed"] if "Compliance_Check" in df.columns else None
    }
    
    selected_sections = st.multiselect("✅ Select Reports to Export", list(sections.keys()), default=["📊 Financial Data"])

    if not selected_sections:
        st.warning("⚠️ Please select at least one report to export.")
        return

    # ✅ Export to Excel (Multi-Sheet)
    if export_format == "Excel (.xlsx)":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for section in selected_sections:
                if sections[section] is not None:
                    sections[section].to_excel(writer, sheet_name=section.replace("📊 ", ""), index=False)
        output.seek(0)
        st.download_button(label="📥 Download Excel", data=output, file_name="Financial_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ✅ Export to CSV (Single File)
    elif export_format == "CSV (.csv)":
        output = io.StringIO()
        sections[selected_sections[0]].to_csv(output, index=False)
        st.download_button(label="📥 Download CSV", data=output.getvalue(), file_name="Financial_Report.csv", mime="text/csv")

    # ✅ Export as ZIP (Multiple CSVs)
    elif export_format == "ZIP (Compressed)":
        zip_output = io.BytesIO()
        with zipfile.ZipFile(zip_output, "w") as zipf:
            for section in selected_sections:
                if sections[section] is not None:
                    csv_data = sections[section].to_csv(index=False).encode("utf-8")
                    zipf.writestr(f"{section.replace('📊 ', '')}.csv", csv_data)
        zip_output.seek(0)
        st.download_button(label="📥 Download ZIP", data=zip_output, file_name="Financial_Reports.zip", mime="application/zip")

    # ✅ Confirmation Message
    st.success("✅ Report generated successfully! 🎉")

# Call the Appropriate Function Based on Selection
if selected_option == "Dashboard":
    show_dashboard()
if selected_option == "AI Forecasting":
    show_forecasting(df)
elif selected_option == "Data Quality & Governance":
    show_data_quality()
elif selected_option == "KPI Monitoring":
    show_kpi_monitoring()
elif selected_option == "Anomaly Detection":
    show_anomaly_detection()
elif selected_option == "Financial Compliance":
    show_compliance()
elif selected_option == "Export Reports":
    show_export()


# Footer
st.sidebar.markdown("---")
st.sidebar.write("AI-Powered Finance & Controlling Suite | Developed by John Johnson O.")   

