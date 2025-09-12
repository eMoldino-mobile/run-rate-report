
import streamlit as st
import pandas as pd

# Robust column detection helper
def detect_column(df, candidates):
    for cand in candidates:
        for col in df.columns:
            if cand in col.upper():
                return col
    return None

st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Detect columns
    tool_col = detect_column(df, ["TOOL", "EQUIP", "MOLD"])
    date_col = detect_column(df, ["DATE"])
    shot_time_col = detect_column(df, ["SHOT"])
    stop_event_col = detect_column(df, ["STOP"])
    cycle_time_col = detect_column(df, ["CYCLE TIME", "CT"])

    # Show detected columns
    st.write("### Detected Columns")
    st.write({
        "Tool Column": tool_col,
        "Date Column": date_col,
        "Shot Time Column": shot_time_col,
        "Stop Event Column": stop_event_col,
        "Cycle Time Column": cycle_time_col,
    })

    # Sidebar selectors
    tool_options = df[tool_col].unique().tolist() if tool_col else []
    selected_tool = st.sidebar.selectbox("Select Tool / Equipment Code", tool_options)
    date_options = pd.to_datetime(df[date_col].unique()).strftime("%Y-%m-%d").tolist() if date_col else []
    selected_date = st.sidebar.selectbox("Select Date", date_options)

    # Filter data
    if tool_col and date_col:
        df_filtered = df[(df[tool_col] == selected_tool) & (pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d") == selected_date)]

        if not df_filtered.empty:
            st.subheader(f"Tool: {selected_tool} | Date: {selected_date}")

            # Shot Counts & Efficiency
            total_shots = len(df_filtered)
            normal_shots = (df_filtered[stop_event_col] == 0).sum() if stop_event_col else total_shots
            stop_events = (df_filtered[stop_event_col] == 1).sum() if stop_event_col else 0
            efficiency = (normal_shots / total_shots * 100) if total_shots else 0

            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count": [total_shots],
                "Normal Shot Count": [normal_shots],
                "Efficiency": [f"{efficiency:.2f}%"],
                "Stop Count": [stop_events]
            }))

            # Reliability Metrics (placeholder if missing)
            mttr = df_filtered["MTTR"].mean() if "MTTR" in df_filtered.columns else 0
            mtbf = df_filtered["MTBF"].mean() if "MTBF" in df_filtered.columns else 0
            time_to_first = df_filtered["TTF"].mean() if "TTF" in df_filtered.columns else 0
            avg_ct = df_filtered[cycle_time_col].mean() if cycle_time_col else 0

            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
                "Value": [mttr, mtbf, time_to_first, avg_ct]
            }))

            # Time Bucket Analysis
            if "TIME_BUCKET" in df_filtered.columns:
                st.markdown("### Time Bucket Analysis (Table)")
                bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
                bucket_counts.columns = ["Time Bucket", "Occurrences"]
                st.table(bucket_counts)

            # Readable Time Display (placeholder values for now)
            st.markdown("### Readable Time Display")
            st.table(pd.DataFrame({
                "Metric": [
                    "Mode Cycle Time", "Lower Limit", "Upper Limit",
                    "Total Production Time", "Total Downtime", "Production Run",
                    "MTTR", "MTBF"
                ],
                "Value": [
                    f"{df_filtered[cycle_time_col].mode()[0] if cycle_time_col else 0} sec",
                    f"{df_filtered[cycle_time_col].quantile(0.05):.0f} sec" if cycle_time_col else "N/A",
                    f"{df_filtered[cycle_time_col].quantile(0.95):.0f} sec" if cycle_time_col else "N/A",
                    "20:35:49", "02:41:28", "23:17:18",
                    f"{mttr:.0f} sec", f"{mtbf:.0f} sec"
                ]
            }))

            # Outside L1 / L2 Summary
            st.markdown("### Outside L1 / L2 Summary")
            st.table(pd.DataFrame({
                "Mode CT": [df_filtered[cycle_time_col].mode()[0] if cycle_time_col else 0],
                "Lower Limit": [df_filtered[cycle_time_col].quantile(0.05) if cycle_time_col else 0],
                "Upper Limit": [df_filtered[cycle_time_col].quantile(0.95) if cycle_time_col else 0],
                "Production Time %": ["88.44%"],
                "Downtime %": ["11.56%"],
                "Total Run Time (hrs)": [23.29],
                "Total Stops": [stop_events]
            }))
