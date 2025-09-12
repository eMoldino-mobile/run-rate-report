
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# --- Helper Functions ---
def format_time(minutes):
    seconds = int(minutes * 60)
    return str(timedelta(seconds=seconds))

def calculate_run_rate_excel_like(df):
    df = df.copy()
    df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
    df["CT_diff_sec"] = df["SHOT TIME"].diff().dt.total_seconds()

    # Mode CT (seconds)
    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05

    # STOP flag
    df["STOP_FLAG"] = np.where(
        (df["CT_diff_sec"].notna()) & 
        ((df["CT_diff_sec"] < lower_limit) | (df["CT_diff_sec"] > upper_limit)) & 
        (df["CT_diff_sec"] <= 28800),
        1, 0
    )
    df.loc[df.index[0], "STOP_FLAG"] = 0

    # Adjust for back-to-back stops
    df["STOP_ADJ"] = df["STOP_FLAG"]
    df.loc[(df["STOP_FLAG"] == 1) & (df["STOP_FLAG"].shift(fill_value=0) == 1), "STOP_ADJ"] = 0

    # Metrics
    total_shots = len(df)
    normal_shots = (df["STOP_ADJ"] == 0).sum()
    df["STOP_EVENT"] = (df["STOP_ADJ"].shift(fill_value=0) == 0) & (df["STOP_ADJ"] == 1)
    stop_events = df["STOP_EVENT"].sum()

    run_hours = df["TOTAL RUN TIME"].iloc[0] / 60
    gross_rate = total_shots / run_hours if run_hours else None
    net_rate = normal_shots / run_hours if run_hours else None
    efficiency = normal_shots / total_shots if total_shots else None

    # Extra metrics
    production_time = df["PRODUCTION TIME"].iloc[0]
    downtime = df["TOTAL DOWN TIME"].iloc[0]
    total_runtime = df["TOTAL RUN TIME"].iloc[0]

    # Time bucket analysis
    df["RUN_DURATION"] = np.where(df["STOP_ADJ"] == 1, df["CT_diff_sec"]/60, np.nan)
    df["TIME_BUCKET"] = pd.cut(df["RUN_DURATION"], 
                               bins=[0,1,2,3,5,10,20,30,60,120,999999],
                               labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"])
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # Per-hour aggregation for MTTR / MTBF
    df["HOUR"] = df["SHOT TIME"].dt.hour
    hourly = df.groupby("HOUR").agg(
        mttr=("RUN_DURATION", lambda x: np.nanmean(x) if len(x) > 0 else np.nan),
        mtbf=("CT_diff_sec", lambda x: np.nanmean(x) if len(x) > 0 else np.nan)
    ).reset_index()
    hourly["stability_index"] = (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100

    results = {
        "mode_ct": mode_ct,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "total_shots": total_shots,
        "normal_shots": normal_shots,
        "stop_events": stop_events,
        "run_hours": run_hours,
        "gross_rate": gross_rate,
        "net_rate": net_rate,
        "efficiency": efficiency,
        "production_time": production_time,
        "downtime": downtime,
        "total_runtime": total_runtime,
        "bucket_counts": bucket_counts,
        "hourly": hourly
    }

    return results, df

# --- Streamlit UI ---
st.sidebar.title("Run Rate Report Generator")

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
    tool = st.sidebar.selectbox("Select Tool / Equipment Code", df["EQUIPMENT CODE"].unique())
    date = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT TIME"]).dt.date.min())

    if st.sidebar.button("Generate Report"):
        mask = (df["EQUIPMENT CODE"] == tool) & (pd.to_datetime(df["SHOT TIME"]).dt.date == date)
        df_filtered = df.loc[mask].copy()
        df_filtered["SHOT TIME"] = pd.to_datetime(df_filtered["SHOT TIME"], errors="coerce")

        if df_filtered.empty:
            st.warning("No data found for this selection.")
        else:
            results, processed_df = calculate_run_rate_excel_like(df_filtered)

            st.title("📊 Run Rate Report")
            st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")

            # --- Graphs ---
            st.header("📈 Visual Analysis")

            # 1. Time Bucket Analysis (Bar)
            bucket_df = results['bucket_counts'].drop("Grand Total")
            fig1 = px.bar(bucket_df, x=bucket_df.index, y=bucket_df.values, text=bucket_df.values)
            fig1.update_traces(marker_color="steelblue")
            fig1.update_layout(title="Time Bucket Analysis", xaxis_title="Time Bucket", yaxis_title="Occurrences")
            st.plotly_chart(fig1, use_container_width=True)

            # 2. Time Bucket Trend (Stacked Bar by Hour, 0-23 always shown)
            trend_df = processed_df.dropna(subset=["TIME_BUCKET"]).groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
            all_hours = pd.DataFrame({"HOUR": range(24)})
            trend_df = all_hours.merge(trend_df, on="HOUR", how="left").fillna(0)
            fig2 = px.bar(trend_df, x="HOUR", y="count", color="TIME_BUCKET", title="Time Bucket Trend by Hour", barmode="stack")
            st.plotly_chart(fig2, use_container_width=True)

            # 3. MTTR & MTBF Trend (Line)
            hourly = results['hourly']
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mttr"], mode="lines+markers", name="MTTR",
                                      line=dict(color="red", width=4)))
            fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mtbf"], mode="lines+markers", name="MTBF",
                                      line=dict(color="green", width=4)))
            fig3.update_layout(title="MTTR & MTBF Trend per Hour", xaxis_title="Hour of Day", yaxis_title="Minutes")
            st.plotly_chart(fig3, use_container_width=True)

            # 4. Stability Index (Line with Zones)
            fig4 = go.Figure()
            # add colored zones
            fig4.add_hrect(y0=70, y1=90, fillcolor="lightgreen", opacity=0.3, line_width=0)
            fig4.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.3, line_width=0)
            fig4.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.3, line_width=0)
            # add line
            fig4.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["stability_index"], mode="lines+markers",
                                      name="Stability Index", line=dict(color="blue", width=4)))
            fig4.update_layout(title="Stability Index per Hour", xaxis_title="Hour of Day", yaxis_title="Index (0-100)")
            st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("👈 Upload a cleaned run rate Excel file to begin.")
