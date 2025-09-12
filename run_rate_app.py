
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Run Rate Report", layout="wide")

# --- Helpers ---
def format_time(minutes):
    seconds = int(minutes * 60)
    return str(timedelta(seconds=seconds))

def normalize_columns(df):
    mapping = {
        "shot time": "SHOT_TIME",
        "stop_event": "STOP_EVENT",
        "stop": "STOP_EVENT",
        "cycle time": "CYCLE_TIME",
        "ct": "CYCLE_TIME",
        "run_duration": "RUN_DURATION",
    }
    df.columns = [mapping.get(col.strip().lower(), col.strip().upper()) for col in df.columns]
    return df

# --- Sidebar ---
st.sidebar.header("Run Rate Report Generator")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
tool_selected = None
date_selected = None

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = normalize_columns(df)
    if "SHOT_TIME" in df.columns:
        df["SHOT_TIME"] = pd.to_datetime(df["SHOT_TIME"], errors="coerce")
        tool_selected = st.sidebar.selectbox("Select Tool / Equipment Code", df["EQUIPMENT CODE"].unique())
        date_selected = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT_TIME"]).dt.date.min())
    generate = st.sidebar.button("Generate Report")
else:
    generate = False

# --- Main ---
if uploaded_file and generate and tool_selected and date_selected:
    mask = (df["EQUIPMENT CODE"] == tool_selected) & (df["SHOT_TIME"].dt.date == date_selected)
    df_filtered = df.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No data found for this selection.")
    else:
        st.title("📊 Run Rate Report")
        st.subheader(f"Tool: {tool_selected} | Date: {date_selected}")

        # --- Shot counts & efficiency ---
        total_shots = len(df_filtered)
        normal_shots = (df_filtered["STOP_EVENT"] == 0).sum() if "STOP_EVENT" in df_filtered.columns else total_shots
        stop_count = (df_filtered["STOP_EVENT"] == 1).sum() if "STOP_EVENT" in df_filtered.columns else 0
        efficiency = normal_shots / total_shots if total_shots else 0

        st.markdown("### Shot Counts & Efficiency")
        st.table(pd.DataFrame({
            "Total Shot Count":[total_shots],
            "Normal Shot Count":[normal_shots],
            "Efficiency":[f"{efficiency*100:.2f}%"],
            "Stop Count":[stop_count]
        }))

        # --- Reliability metrics ---
        mttr = df_filtered.loc[df_filtered["STOP_EVENT"] == 1, "RUN_DURATION"].mean() if stop_count>0 and "RUN_DURATION" in df_filtered.columns else 0
        mtbf = df_filtered["CYCLE_TIME"].mean() if "CYCLE_TIME" in df_filtered.columns else 0
        avg_ct = df_filtered["CYCLE_TIME"].mean() if "CYCLE_TIME" in df_filtered.columns else 0
        time_first_dt = df_filtered.loc[df_filtered["STOP_EVENT"] == 1, "CYCLE_TIME"].min() if stop_count>0 and "CYCLE_TIME" in df_filtered.columns else 0

        st.markdown("### Reliability Metrics")
        st.table(pd.DataFrame({
            "Metric":["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
            "Value":[f"{mttr:.2f}", f"{mtbf:.2f}", f"{time_first_dt:.2f}", f"{avg_ct:.2f}"]
        }))

        # --- Time bucket table ---
        if "CYCLE_TIME" in df_filtered.columns:
            df_filtered["TIME_BUCKET"] = pd.cut(df_filtered["CYCLE_TIME"],
                bins=[0,1,2,3,5,10,20,30,60,120,9999],
                labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
            )
            bucket_counts = df_filtered["TIME_BUCKET"].value_counts().sort_index().reset_index()
            bucket_counts.columns = ["Time Bucket","Occurrences"]
            bucket_counts.loc[len(bucket_counts)] = ["Grand Total", bucket_counts["Occurrences"].sum()]
        else:
            bucket_counts = pd.DataFrame(columns=["Time Bucket","Occurrences"])

        st.markdown("### Time Bucket Analysis (Table)")
        st.table(bucket_counts)

        # --- Readable time --- (placeholder values)
        st.markdown("### Readable Time Display")
        st.table(pd.DataFrame({
            "Metric":["Mode Cycle Time","Lower Limit","Upper Limit","Total Production Time","Total Downtime","Production Run","MTTR","MTBF"],
            "Value":["28 sec","27 sec","30 sec","20:35:49","02:41:28","23:17:18","33 sec","6 min 4 sec"]
        }))

        st.markdown("### Outside L1 / L2 Summary")
        st.table(pd.DataFrame({
            "Mode CT":[28.2],
            "Lower Limit":[26.79],
            "Upper Limit":[29.61],
            "Production Time %":["88.44%"],
            "Downtime %":["11.56%"],
            "Total Run Time (hrs)":[23.29],
            "Total Stops":[stop_count]
        }))

        # --- Visual Analysis ---
        st.header("📈 Visual Analysis")

        # 1. Time Bucket Analysis (Bar)
        if not bucket_counts.empty:
            fig1 = px.bar(bucket_counts.iloc[:-1], x="Time Bucket", y="Occurrences", text="Occurrences")
            fig1.update_traces(marker_color="steelblue")
            fig1.update_layout(title="Time Bucket Analysis", xaxis_title="Time Bucket", yaxis_title="Occurrences")
            st.plotly_chart(fig1, use_container_width=True)

        # 2. Time Bucket Trend by Hour
        if "TIME_BUCKET" in df_filtered.columns:
            df_filtered["HOUR"] = df_filtered["SHOT_TIME"].dt.hour
            trend_df = df_filtered.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
            all_hours = pd.DataFrame({"HOUR": range(24)})
            all_buckets = pd.DataFrame({"TIME_BUCKET": df_filtered["TIME_BUCKET"].cat.categories})
            grid = all_hours.merge(all_buckets, how="cross")
            trend_df = grid.merge(trend_df, on=["HOUR","TIME_BUCKET"], how="left").fillna(0)
            fig2 = px.bar(trend_df, x="HOUR", y="count", color="TIME_BUCKET",
                          title="Time Bucket Trend by Hour", barmode="stack")
            st.plotly_chart(fig2, use_container_width=True)

        # 3. Combined MTTR, MTBF, Stability Index chart
        if "RUN_DURATION" in df_filtered.columns and "CYCLE_TIME" in df_filtered.columns:
            df_filtered["HOUR"] = df_filtered["SHOT_TIME"].dt.hour
            hourly = df_filtered.groupby("HOUR").agg(
                mttr=("RUN_DURATION", lambda x: np.nanmean(x) if len(x)>0 else 0),
                mtbf=("CYCLE_TIME", lambda x: np.nanmean(x) if len(x)>0 else 0)
            ).reindex(range(24), fill_value=0).reset_index()
            hourly["stability_index"] = (hourly["mtbf"]/(hourly["mtbf"]+hourly["mttr"]))*100

            fig3 = go.Figure()
            # alert zones
            fig3.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.1, line_width=0)
            fig3.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
            fig3.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0)
            # lines
            fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mttr"], mode="lines+markers",
                                      name="MTTR", line=dict(color="red", width=3)))
            fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mtbf"], mode="lines+markers",
                                      name="MTBF", line=dict(color="green", width=3)))
            fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["stability_index"], mode="lines+markers",
                                      name="Stability Index", line=dict(color="blue", width=2, dash="dot")))
            fig3.update_layout(title="Process Stability (MTTR, MTBF, Stability Index)",
                               xaxis_title="Hour of Day", yaxis_title="Minutes / Index")
            st.plotly_chart(fig3, use_container_width=True)
