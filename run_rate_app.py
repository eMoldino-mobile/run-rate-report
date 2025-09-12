
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Utility functions
# ----------------------
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

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Run Rate Report", layout="wide")
st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Detect columns dynamically
    df.columns = df.columns.str.upper().str.strip()
    if "SHOT TIME" not in df.columns:
        st.error("SHOT TIME column is missing from the uploaded file.")
    else:
        df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")

    # Sidebar selectors
    tools = df["TOOL"].unique() if "TOOL" in df.columns else ["Unknown"]
    selected_tool = st.sidebar.selectbox("Select Tool / Equipment Code", tools)
    dates = pd.to_datetime(df["SHOT TIME"].dt.date.unique())
    selected_date = st.sidebar.date_input("Select Date", min(dates))

    if st.sidebar.button("Generate Report"):
        df_filtered = df[df["TOOL"] == selected_tool]
        df_filtered = df_filtered[df_filtered["SHOT TIME"].dt.date == pd.to_datetime(selected_date).date()]

        if df_filtered.empty:
            st.warning("No data available for the selected tool/date.")
        else:
            # Metrics calculations
            total_shots = len(df_filtered)
            stop_count = df_filtered["STOP_EVENT"].sum() if "STOP_EVENT" in df_filtered.columns else 0
            normal_shots = total_shots - stop_count
            efficiency = round((normal_shots / total_shots) * 100, 2) if total_shots > 0 else 0

            # Cycle times
            cycle_times = df_filtered["CYCLE_TIME"].dropna() if "CYCLE_TIME" in df_filtered.columns else pd.Series()
            mode_ct = cycle_times.mode().iloc[0] if not cycle_times.empty else 0
            lower_limit, upper_limit = round(mode_ct * 0.95, 2), round(mode_ct * 1.05, 2)

            # Reliability Metrics
            mttr = df_filtered["DURATION"][df_filtered["STOP_EVENT"] == 1].mean() if "DURATION" in df_filtered.columns else 0
            mtbf = cycle_times.mean() if not cycle_times.empty else 0
            time_to_first = df_filtered["CYCLE_TIME"].iloc[0] if not df_filtered.empty else 0

            # Display Summaries
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

            # Time Bucket Analysis
            st.markdown("### Time Bucket Analysis (Table)")
            if "TIME_BUCKET" in df_filtered.columns:
                bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
                bucket_counts.columns = ["Time Bucket", "Occurrences"]
                st.table(bucket_counts)

            # Readable Time Display
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
                    format_seconds(df_filtered["DURATION"].sum() if "DURATION" in df_filtered.columns else 0),
                    format_seconds(df_filtered["CYCLE_TIME"].sum() if "CYCLE_TIME" in df_filtered.columns else 0),
                    format_seconds(mttr), format_seconds(mtbf)
                ]
            }))

            # Outside L1 / L2 Summary
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

            # Time Bucket Chart
            if "TIME_BUCKET" in df_filtered.columns:
                bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
                bucket_counts.columns = ["Time Bucket", "Occurrences"]
                fig = px.bar(bucket_counts, x="Time Bucket", y="Occurrences", text="Occurrences")
                st.plotly_chart(fig, use_container_width=True)
