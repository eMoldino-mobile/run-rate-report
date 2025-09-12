
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Run Rate Report Generator", layout="wide")

st.title("📊 Run Rate Report Generator")

# File upload
uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

def detect_column(df, candidates):
    for cand in candidates:
        for col in df.columns:
            if cand in col.upper():
                return col
    return None

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Detect columns
    tool_col = detect_column(df, ["TOOL", "EQUIP", "MOLD"])
    shot_time_col = detect_column(df, ["SHOT TIME", "SHOT"])
    stop_col = detect_column(df, ["STOP"])
    cycle_time_col = detect_column(df, ["CYCLE TIME", "CT", "APPROVED CT"])
    date_candidates = [c for c in df.columns if "DATE" in c.upper()]
    date_col = date_candidates[0] if date_candidates else None

    # Normalize datetime
    if shot_time_col:
        df[shot_time_col] = pd.to_datetime(df[shot_time_col], errors="coerce")

    # If no date col, fallback to SHOT TIME date
    if not date_col and shot_time_col:
        df["__DATE__"] = df[shot_time_col].dt.date
        date_col = "__DATE__"

    # Sidebar selectors
    tools = df[tool_col].dropna().unique() if tool_col else []
    selected_tool = st.sidebar.selectbox("Select Tool / Equipment Code", tools)
    dates = df[date_col].dropna().unique() if date_col else []
    selected_date = st.sidebar.selectbox("Select Date", sorted(dates))

    if st.sidebar.button("Generate Report"):
        df_filtered = df[(df[tool_col] == selected_tool) & (df[date_col] == selected_date)]

        if not df_filtered.empty:
            st.subheader(f"Tool: {selected_tool} | Date: {selected_date}")

            # --- Summaries ---
            total_shots = len(df_filtered)
            normal_shots = (df_filtered[stop_col] == 0).sum() if stop_col else total_shots
            stop_events = (df_filtered[stop_col] == 1).sum() if stop_col else 0
            efficiency = (normal_shots / total_shots * 100) if total_shots else 0

            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count": [total_shots],
                "Normal Shot Count": [normal_shots],
                "Efficiency": [f"{efficiency:.2f}%"],
                "Stop Count": [stop_events]
            }))

            avg_ct = df_filtered[cycle_time_col].mean() if cycle_time_col else 0

            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric": ["Avg Cycle Time (Avg)"],
                "Value": [round(avg_ct, 2)]
            }))

            # --- Visual Analysis ---
            st.markdown("## 📉 Visual Analysis")

            if shot_time_col and stop_col:
                df_filtered["HOUR"] = df_filtered[shot_time_col].dt.hour

                # Calculate durations between consecutive shots
                df_filtered = df_filtered.sort_values(shot_time_col)
                df_filtered["DIFF"] = df_filtered[shot_time_col].diff().dt.total_seconds() / 60

                # MTTR = average downtime when STOP = 1
                # MTBF = average run time when STOP = 0
                hourly = df_filtered.groupby("HOUR").agg(
                    MTTR=("DIFF", lambda x: x[df_filtered[stop_col] == 1].mean() if any(df_filtered[stop_col] == 1) else 0),
                    MTBF=("DIFF", lambda x: x[df_filtered[stop_col] == 0].mean() if any(df_filtered[stop_col] == 0) else 0)
                ).reindex(range(24), fill_value=0).reset_index()

                hourly["Stability"] = (hourly["MTBF"] / (hourly["MTBF"] + hourly["MTTR"])) * 100

                fig = go.Figure()
                fig.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.2, line_width=0)
                fig.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.2, line_width=0)
                fig.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.2, line_width=0)

                fig.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["MTTR"], mode="lines+markers",
                                         name="MTTR", line=dict(color="red", width=3)))
                fig.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["MTBF"], mode="lines+markers",
                                         name="MTBF", line=dict(color="green", width=3)))
                fig.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["Stability"], mode="lines+markers",
                                         name="Stability Index", line=dict(color="blue", width=2, dash="dot")))

                fig.update_layout(title="Process Stability (MTTR, MTBF, Stability Index)",
                                  xaxis_title="Hour of Day", yaxis_title="Minutes / Index")
                st.plotly_chart(fig, use_container_width=True)
