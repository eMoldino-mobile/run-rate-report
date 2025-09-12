
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Run Rate Report", layout="wide")

def format_seconds(seconds):
    if pd.isna(seconds):
        return "0 sec"
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if h > 0:
        return f"{h} h {m} min {s} sec"
    elif m > 0:
        return f"{m} min {s} sec"
    else:
        return f"{s} sec"

st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.upper().str.strip()
    if "SHOT TIME" not in df.columns:
        st.error("SHOT TIME column is missing from the uploaded file.")
    else:
        df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")

    tools = df["EQUIPMENT CODE"].unique() if "EQUIPMENT CODE" in df.columns else ["Unknown"]
    selected_tool = st.sidebar.selectbox("Select Tool / Equipment Code", tools)
    dates = pd.to_datetime(df["SHOT TIME"].dt.date.unique())
    selected_date = st.sidebar.date_input("Select Date", min(dates))

    if st.sidebar.button("Generate Report"):
        df_filtered = df[df["EQUIPMENT CODE"] == selected_tool]
        df_filtered = df_filtered[df_filtered["SHOT TIME"].dt.date == pd.to_datetime(selected_date).date()]

        if df_filtered.empty:
            st.warning("No data available for the selected tool/date.")
        else:
            # Metrics
            total_shots = len(df_filtered)
            stop_count = df_filtered["STOP_EVENT"].sum() if "STOP_EVENT" in df_filtered.columns else 0
            normal_shots = total_shots - stop_count
            efficiency = round((normal_shots / total_shots) * 100, 2) if total_shots > 0 else 0

            # Cycle times
            cycle_times = df_filtered["CYCLE_TIME"].dropna() if "CYCLE_TIME" in df_filtered.columns else pd.Series()
            mode_ct = cycle_times.mode().iloc[0] if not cycle_times.empty else 0
            lower_limit, upper_limit = round(mode_ct * 0.95, 2), round(mode_ct * 1.05, 2)

            # Reliability
            mttr = df_filtered.loc[df_filtered["STOP_EVENT"] == 1, "RUN_DURATION"].mean() if "RUN_DURATION" in df_filtered.columns else 0
            mtbf = cycle_times.mean() if not cycle_times.empty else 0
            time_to_first = df_filtered["CYCLE_TIME"].iloc[0] if not df_filtered.empty else 0

            # --- Summaries ---
            st.subheader(f"Tool: {selected_tool} | Date: {selected_date}")
            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count": [total_shots],
                "Normal Shot Count": [normal_shots],
                "Efficiency": [f"{efficiency}%"],
                "Stop Count": [stop_count]
            }))

            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
                "Value": [round(mttr, 2), round(mtbf, 2), round(time_to_first, 2), round(cycle_times.mean(), 2)]
            }))

            st.markdown("### Time Bucket Analysis (Table)")
            if "TIME_BUCKET" in df_filtered.columns:
                bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
                bucket_counts.columns = ["Time Bucket", "Occurrences"]
                st.table(bucket_counts)

            st.markdown("### Readable Time Display")
            st.table(pd.DataFrame({
                "Metric": [
                    "Mode Cycle Time", "Lower Limit", "Upper Limit",
                    "Total Production Time", "Total Downtime", "Production Run",
                    "MTTR", "MTBF"
                ],
                "Value": [
                    f"{round(mode_ct,2)} sec",
                    f"{lower_limit} sec", f"{upper_limit} sec",
                    format_seconds(df_filtered["CYCLE_TIME"].sum() if "CYCLE_TIME" in df_filtered.columns else 0),
                    format_seconds(df_filtered["RUN_DURATION"].sum() if "RUN_DURATION" in df_filtered.columns else 0),
                    format_seconds(df_filtered["CYCLE_TIME"].sum() if "CYCLE_TIME" in df_filtered.columns else 0),
                    format_seconds(mttr), format_seconds(mtbf)
                ]
            }))

            st.markdown("### Outside L1 / L2 Summary")
            st.table(pd.DataFrame({
                "Mode CT": [round(mode_ct, 2)],
                "Lower Limit": [lower_limit],
                "Upper Limit": [upper_limit],
                "Production Time %": [f"{round((normal_shots/total_shots)*100, 2) if total_shots>0 else 0}%"],
                "Downtime %": [f"{100-efficiency}%"],
                "Total Run Time (hrs)": [round(df_filtered["CYCLE_TIME"].sum()/3600, 2) if "CYCLE_TIME" in df_filtered.columns else 0],
                "Total Stops": [stop_count]
            }))

            # ----------------------
            # Visual Analysis Section
            # ----------------------
            st.markdown("## 📉 Visual Analysis")

            # 1. Time Bucket Chart
            if "TIME_BUCKET" in df_filtered.columns:
                bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
                bucket_counts.columns = ["Time Bucket", "Occurrences"]
                fig = px.bar(bucket_counts, x="Time Bucket", y="Occurrences", text="Occurrences")
                st.plotly_chart(fig, use_container_width=True)

            # 2. Time Bucket Trend
            if "TIME_BUCKET" in df_filtered.columns:
                df_filtered["HOUR"] = df_filtered["SHOT TIME"].dt.hour
                trend_df = df_filtered.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
                all_hours = pd.DataFrame({"HOUR": range(24)})
                all_buckets = pd.DataFrame({"TIME_BUCKET": df_filtered["TIME_BUCKET"].cat.categories})
                grid = all_hours.merge(all_buckets, how="cross")
                trend_df = grid.merge(trend_df, on=["HOUR","TIME_BUCKET"], how="left").fillna(0)
                fig2 = px.bar(trend_df, x="HOUR", y="count", color="TIME_BUCKET",
                              title="Time Bucket Trend by Hour", barmode="stack")
                st.plotly_chart(fig2, use_container_width=True)

            # 3. Combined MTTR, MTBF, Stability Index
            if "RUN_DURATION" in df_filtered.columns and "CYCLE_TIME" in df_filtered.columns:
                df_filtered["HOUR"] = df_filtered["SHOT TIME"].dt.hour
                hourly = df_filtered.groupby("HOUR").agg(
                    mttr=("RUN_DURATION", lambda x: np.nanmean(x) if len(x)>0 else 0),
                    mtbf=("CYCLE_TIME", lambda x: np.nanmean(x) if len(x)>0 else 0)
                ).reindex(range(24), fill_value=0).reset_index()
                hourly["stability_index"] = (hourly["mtbf"]/(hourly["mtbf"]+hourly["mttr"]))*100

                fig3 = go.Figure()
                # Alert zones
                fig3.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.1, line_width=0)
                fig3.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
                fig3.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0)

                # Lines with hover tooltips
                fig3.add_trace(go.Scatter(
                    x=hourly["HOUR"], y=hourly["mttr"], mode="lines+markers",
                    name="MTTR", line=dict(color="red", width=3),
                    hovertemplate="Hour %{x}<br>MTTR: %{y:.2f} min<extra></extra>"
                ))
                fig3.add_trace(go.Scatter(
                    x=hourly["HOUR"], y=hourly["mtbf"], mode="lines+markers",
                    name="MTBF", line=dict(color="green", width=3),
                    hovertemplate="Hour %{x}<br>MTBF: %{y:.2f} min<extra></extra>"
                ))
                fig3.add_trace(go.Scatter(
                    x=hourly["HOUR"], y=hourly["stability_index"], mode="lines+markers",
                    name="Stability Index", line=dict(color="blue", width=2, dash="dot"),
                    hovertemplate="Hour %{x}<br>Stability Index: %{y:.2f}<extra></extra>"
                ))

                fig3.update_layout(
                    title="Process Stability (MTTR, MTBF, Stability Index)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Minutes / Index",
                    legend_title="Metrics",
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)
