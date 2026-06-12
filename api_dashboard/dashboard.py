import streamlit as st
import time
import requests
import pandas as pd
from streamlit_echarts import st_echarts
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join("..")))
from src.LSTM.preprocess import load_data, encode_features, create_sequences

# disclaimer: gemini was used for page setup with css
st.set_page_config(
    page_title="Live Energy Forecast", layout="wide", initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* 1. Balanced padding to fit the screen without cramping the top */
    .block-container {
        padding-top: 2.5rem !important; /* Gave the top some breathing room */
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* 2. Keep the Left Column KPI Metrics compact */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- UI PLACEHOLDERS & LAYOUT ---
left_col, right_col = st.columns([1, 4])
with left_col:
    st.markdown("### Live Energy Demand")
    time_placeholder = st.empty()
    st.markdown("---")

    actual_metric = st.empty()
    pred_metric = st.empty()
    st.markdown("---")

    temp_metric = st.empty()
    rad_metric = st.empty()
    latency_metric = st.empty()
    st.markdown("---")

    with st.expander("Model Monitoring", expanded=True):
        mae_metric = st.empty()
        status_metric = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    start_stream = st.button("▶ Start Stream", type="primary", use_container_width=True)
with right_col:
    line_chart_placeholder = st.empty()
    flow_placeholder = st.empty()


def render_echarts_flow(actual_total, heat_pump, wash, other):
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "coordinateSystem": "cartesian2d",
                "symbolSize": 110,
                "label": {
                    "show": True,
                    "color": "#333333",
                    "position": "inside",
                    "fontSize": 18,
                    "align": "center",
                    "formatter": "{b}",
                },
                "data": [
                    {
                        "name": f"⚡\n{actual_total:.2f} kW\nTotal",
                        "value": [10, 50],
                        "symbolSize": 135,
                        "itemStyle": {
                            "color": "transparent",
                            "borderColor": "#00E676",
                            "borderWidth": 5,
                        },
                    },
                    {
                        "name": f"🌡🏠\n{heat_pump:.2f} kW\nHeat Pump",
                        "value": [70, 50],
                        "itemStyle": {
                            "color": "transparent",
                            "borderColor": "#FF7043",
                            "borderWidth": 4,
                        },
                    },
                    {
                        "name": f"🫧👕\n{wash:.2f} kW\nWasher",
                        "value": [85, 100],
                        "itemStyle": {
                            "color": "transparent",
                            "borderColor": "#4285F4",
                            "borderWidth": 4,
                        },
                    },
                    {
                        "name": f"🔌🔋\n{other:.2f} kW\nOther",
                        "value": [85, 0],
                        "itemStyle": {
                            "color": "transparent",
                            "borderColor": "#AB47BC",
                            "borderWidth": 4,
                        },
                    },
                ],
            },
            {
                "type": "lines",
                "coordinateSystem": "cartesian2d",
                "effect": {
                    "show": True,
                    "period": 1.5,
                    "trailLength": 0.05,
                    "color": "#00E676",
                    "symbolSize": 8,
                },
                "lineStyle": {"color": "#555", "width": 2},
                "data": [
                    {"coords": [[10, 50], [70, 50]], "lineStyle": {"curveness": 0}},
                    {"coords": [[10, 50], [85, 100]], "lineStyle": {"curveness": 0.25}},
                    {"coords": [[10, 50], [85, 0]], "lineStyle": {"curveness": -0.25}},
                ],
            },
        ],
        "xAxis": {"show": False, "min": 0, "max": 125},
        "yAxis": {"show": False, "min": -30, "max": 130},
    }


# --- MAIN STREAMING LOOP ---
if start_stream:
    df_raw = pd.read_csv("../data/processed/residential4_energy_demand_daily_test.csv")

    with st.spinner("Preprocessing..."):
        train_path = "../data/processed/residential4_energy_demand_daily_train.csv"
        test_path = "../data/processed/residential4_energy_demand_daily_test.csv"

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        RESOLUTION = "daily"
        SEQUENCE_LENGTH = 2

        train_df, test_df = encode_features(train_df, test_df, resolution=RESOLUTION)

        scaler_path = (
            "artifacts/trained_model_LSTMAttentionModel_lr0.001:v1/scaler.joblib"
        )
        saved_scaler = joblib.load(scaler_path)

        columns = train_df.columns
        train_df[columns] = saved_scaler.transform(train_df[columns])
        test_df[columns] = saved_scaler.transform(test_df[columns])

        _, _, _, X_past_test, X_future_test, _ = create_sequences(
            train_df, test_df, k=SEQUENCE_LENGTH, resolution=RESOLUTION
        )

    time.sleep(5)

    history_actual = []
    history_pred = []

    for index in range(len(X_past_test)):
        start_time = time.time()

        target_row = df_raw.iloc[index]

        current_time = target_row["utc_timestamp"]
        time_placeholder.markdown(f"**🕒 {current_time[:10]}**")

        temp = target_row["temperature"]
        rad = target_row["radiation_direct_horizontal"]
        temp_metric.metric("🌡️ Temp", f"{temp:.1f} °C")
        rad_metric.metric("☀️ Solar", f"{rad:.1f} W")

        payload = {
            "past_sequences": X_past_test[index].tolist(),
            "future_features": X_future_test[index].tolist(),
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                prediction = response.json()["predicted_kWh"]
            else:
                prediction = 0.0
        except requests.exceptions.ConnectionError:
            prediction = 0.0

        latency = int((time.time() - start_time) * 1000)

        actual_total = target_row["energy_demand"]
        heat_pump = target_row["heat_pump"]
        wash = target_row["washing_machine"]
        other = max(0, actual_total - (heat_pump + wash))

        history_actual.append(actual_total)
        history_pred.append(prediction)
        if len(history_actual) > 20:
            history_actual.pop(0)
            history_pred.pop(0)

        actual_metric.metric("Actual", f"{actual_total:.1f} kW")
        pred_metric.metric(
            "Predicted",
            f"{prediction:.1f} kW",
            delta=f"{prediction - actual_total:.1f}",
            delta_color="inverse",
        )
        latency_metric.metric("API Latency", f"{latency} ms")

        # Rolling MAE for basic Monitoring - last 20 observations
        import numpy as np

        if len(history_actual) > 0:
            rolling_mae = np.mean(
                np.abs(np.array(history_actual) - np.array(history_pred))
            )
            mae_metric.metric("Rolling Error (MAE)", f"{rolling_mae:.2f} kW")

            # Simple Drift Alert Threshold
            if rolling_mae > 15.0:  # Adjust this threshold based on your normal error
                status_metric.error("HIGH ERROR: Possible Data Drift!")
            else:
                status_metric.success("Model Health: Stable")

        df_chart = pd.DataFrame({"Actual": history_actual, "Predicted": history_pred})

        line_chart_placeholder.line_chart(
            df_chart, color=["#1f77b4", "#00E676"], height=250
        )

        with flow_placeholder:
            st_echarts(
                options=render_echarts_flow(actual_total, heat_pump, wash, other),
                height="350px",
                key=f"flow_{index}",
            )

        time.sleep(4)
