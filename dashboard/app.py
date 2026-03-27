import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
page_title="Smart Grid Energy Forecast",
layout="wide"
)

st.title("⚡ Smart Grid Energy Consumption Forecasting Dashboard")

st.markdown(
"""
This dashboard shows electricity demand forecasts generated using machine learning models.
"""
)

# --------------------------------

# Sidebar controls

# --------------------------------

st.sidebar.header("Controls")

forecast_horizon = st.sidebar.slider(
"Forecast Horizon (hours)",
min_value=6,
max_value=168,
value=24
)

model_choice = st.sidebar.selectbox(
"Select Model",
["LSTM", "Temporal Fusion Transformer", "Prophet"]
)

# --------------------------------

# Load data

# --------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/predictions.csv")
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

df = load_data()

# Filter horizon

df = df.tail(forecast_horizon)

# --------------------------------

# Main layout

# --------------------------------

col1, col2 = st.columns([2,1])

# --------------------------------

# Forecast Plot

# --------------------------------

with col1:

    st.subheader("Energy Demand Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["actual"],
            mode="lines",
            name="Actual Demand"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["prediction"],
            mode="lines",
            name="Predicted Demand"
        )
    )

    # Prediction interval
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["p90"],
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["p10"],
            fill="tonexty",
            name="Prediction Interval"
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Energy Consumption",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------

# Metrics

# --------------------------------

with col2:


    st.subheader("Model Metrics")

    mae = np.mean(np.abs(df["actual"] - df["prediction"]))
    rmse = np.sqrt(np.mean((df["actual"] - df["prediction"])**2))
    mape = np.mean(np.abs((df["actual"] - df["prediction"]) / df["actual"])) * 100

    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAPE (%)", f"{mape:.2f}")


# --------------------------------

# Feature Importance (example)

# --------------------------------

st.subheader("Feature Importance")

feature_importance = pd.DataFrame({
"feature": [
"Temperature",
"Hour of Day",
"Day of Week",
"Previous Demand",
"Holiday"
],
"importance": [0.32, 0.25, 0.18, 0.20, 0.05]
})

fig2 = px.bar(
feature_importance,
x="importance",
y="feature",
orientation="h",
title="Model Feature Importance"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------

# Data Table

# --------------------------------

st.subheader("Forecast Data")

st.dataframe(df)
