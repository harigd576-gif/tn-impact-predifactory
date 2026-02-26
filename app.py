import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

st.set_page_config(page_title="PrediFactory", layout="wide")
st.title("🛠️ PrediFactory - Predictive Maintenance Dashboard")
st.subheader("Solving real machine breakdown problems for Tamil Nadu industries")

# File upload
uploaded_file = st.file_uploader("Upload your machine_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Make timestamp nice if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    st.success(f"Loaded {len(df)} rows of machine data! ✓")
    
    # Quick stats cards
    col1, col2, col3 = st.columns(3)
    if 'temperature' in df.columns:
        col1.metric("Average Temperature", f"{df['temperature'].mean():.1f} °C")
    if 'vibration' in df.columns:
        col2.metric("Highest Vibration", f"{df['vibration'].max():.2f}")
    if 'running_hours' in df.columns:
        col3.metric("Max Running Hours", int(df['running_hours'].max()))
    
    # Trend chart
    st.subheader("📈 Live Sensor Trends")
    numeric_cols = [col for col in ['temperature', 'vibration', 'pressure', 'running_hours'] if col in df.columns]
    if 'timestamp' in df.columns and numeric_cols:
        fig = px.line(df, x='timestamp', y=numeric_cols,
                      title="How sensors change over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Showing basic table (no timestamp found)")
        st.dataframe(df)
    
    # AI part - find risky points
  # ──────────────────────────────────────────────────────────────
# AI Anomaly / Failure Prediction section – upgraded to LOF
# ──────────────────────────────────────────────────────────────
# AI Anomaly Detection with Model Comparison Toggle
st.subheader("🔴 AI Failure Risk Detection")

# Toggle to choose model – great for demo / comparison
model_choice = st.radio(
    "Select Anomaly Detection Model",
    options=["Isolation Forest", "Local Outlier Factor (LOF)"],
    index=0,  # default to Isolation Forest
    horizontal=True
)

st.markdown(f"**Using:** {model_choice}")

# Features to use (same for both models)
feature_cols = [col for col in ['temperature', 'vibration', 'pressure'] if col in df.columns]

if len(feature_cols) >= 2:
    features = df[feature_cols].dropna()
    
    if len(features) >= 5:
        if model_choice == "Isolation Forest":
            # ─── Isolation Forest ───────────────────────────────────────
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(features)
            
            title_suffix = "(Isolation Forest)"
            color_map = {"HIGH RISK ⚠️": "red", "Normal ✅": "green"}
        
        else:
            # ─── Local Outlier Factor ───────────────────────────────────
            from sklearn.neighbors import LocalOutlierFactor
            
            model = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
            preds = model.fit_predict(features)
            
            title_suffix = "(Local Outlier Factor)"
            color_map = {"HIGH RISK ⚠️": "orange", "Normal ✅": "green"}   # different color to visually distinguish

        # Common prediction logic
        df.loc[features.index, 'risk'] = np.where(preds == -1, "HIGH RISK ⚠️", "Normal ✅")
        
        high_risk_count = (df['risk'] == "HIGH RISK ⚠️").sum()
        
        if high_risk_count > 0:
            st.warning(f"🚨 Detected **{high_risk_count}** high-risk points!")
            st.success("Early warning → can prevent costly downtime and repairs.")
        else:
            st.success("✅ All readings appear normal under current model.")
        
        # Visualization
        fig_risk = px.scatter(
            df,
            x='timestamp' if 'timestamp' in df.columns else df.index,
            y=feature_cols[0],
            color='risk',
            hover_data=feature_cols + ['risk'],
            title=f"Anomaly Detection {title_suffix} – Red/Orange = Potential Failure Risk",
            color_discrete_map=color_map,
            opacity=0.8
        )
        
        # Make points stand out more
        fig_risk.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Bonus: show how many points each model would flag (if you want to compare side-by-side later)
        # st.caption(f"Model flagged {high_risk_count} outliers out of {len(features)} points.")
    
    else:
        st.info("Not enough data points (need ≥5 clean rows) for reliable anomaly detection.")
else:
    st.info("Need at least 2 numeric sensor columns (e.g. temperature + vibration) to run AI detection.")# ──────────────────────────────────────────────────────────────
# AI Anomaly Detection with Model Comparison + Contamination Slider
st.subheader("🔴 AI Failure Risk Detection")

# Model selection
model_choice = st.radio(
    "Select Anomaly Detection Model",
    options=["Isolation Forest", "Local Outlier Factor (LOF)"],
    index=0,
    horizontal=True
)

# Contamination slider – controls expected % of anomalies (0.01 = very strict, 0.25 = more lenient)
contamination = st.slider(
    "Contamination (expected anomaly ratio)",
    min_value=0