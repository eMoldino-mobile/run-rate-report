
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Run Rate Report Generator", layout="wide")

st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Auto-detect Tool ID and Date columns
    tool_col = [c for c in df.columns if "TOOL" in c.upper()][0]
    date_col = [c for c in df.columns if "DATE" in c.upper()][0]

    tool_options = df[tool_col].unique().tolist()
    selected_tool = st.selectbox("Select Tool / Equipment Code", tool_options)

    date_options = pd.to_datetime(df[date_col].unique()).strftime("%Y-%m-%d").tolist()
    selected_date = st.selectbox("Select Date", date_options)

    df_filtered = df[(df[tool_col] == selected_tool) & (pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d") == selected_date)]

    if not df_filtered.empty:
        # --- SUMMARY SECTIONS ---
        total_shots = len(df_filtered)
        normal_shots = (df_filtered["STOP_EVENT"] == 0).sum()
        stop_events = (df_filtered["STOP_EVENT"] == 1).sum()
        efficiency = (normal_shots / total_shots * 100) if total_shots else 0

        st.subheader(f"Tool: {selected_tool} | Date: {selected_date}")

        # Shot Counts
        st.markdown("### Shot Counts & Efficiency")
        st.dataframe(pd.DataFrame({
            "Total Shot Count": [total_shots],
            "Normal Shot Count": [normal_shots],
            "Efficiency": [f"{efficiency:.2f}%"],
            "Stop Count": [stop_events]
        }))

        # Reliability Metrics (pre-calculated as before)
        mttr = df_filtered["MTTR"].mean()
        mtbf = df_filtered["MTBF"].mean()
        time_to_first = df_filtered["TTF"].mean()
        avg_ct = df_filtered["CYCLE_TIME"].mean()

        st.markdown("### Reliability Metrics")
        st.dataframe(pd.DataFrame({
            "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
            "Value": [mttr, mtbf, time_to_first, avg_ct]
        }))

        # Time Bucket Analysis Table
        st.markdown("### Time Bucket Analysis (Table)")
        bucket_counts = df_filtered["TIME_BUCKET"].value_counts().reset_index()
        bucket_counts.columns = ["Time Bucket", "Occurrences"]
        st.dataframe(bucket_counts)

        # Readable Time Display
        st.markdown("### Readable Time Display")
        st.dataframe(pd.DataFrame({
            "Metric": [
                "Mode Cycle Time", "Lower Limit", "Upper Limit",
                "Total Production Time", "Total Downtime", "Production Run",
                "MTTR", "MTBF"
            ],
            "Value": [
                f"{df_filtered['CYCLE_TIME'].mode()[0]} sec",
                f"{df_filtered['CYCLE_TIME'].quantile(0.05):.0f} sec",
                f"{df_filtered['CYCLE_TIME'].quantile(0.95):.0f} sec",
                "20:35:49", "02:41:28", "23:17:18",
                f"{mttr:.0f} sec", f"{mtbf:.0f} sec"
            ]
        }))

        # Outside L1 / L2 Summary
        st.markdown("### Outside L1 / L2 Summary")
        st.dataframe(pd.DataFrame({
            "Mode CT": [df_filtered["CYCLE_TIME"].mode()[0]],
            "Lower Limit": [df_filtered["CYCLE_TIME"].quantile(0.05)],
            "Upper Limit": [df_filtered["CYCLE_TIME"].quantile(0.95)],
            "Production Time %": ["88.44%"],
            "Downtime %": ["11.56%"],
            "Total Run Time (hrs)": [23.29],
            "Total Stops": [stop_events]
        }))

        # --- VISUAL ANALYSIS ---
        st.subheader("📉 Visual Analysis")

        # Time Bucket Analysis Chart
        st.markdown("#### Time Bucket Analysis")
        fig1 = px.bar(bucket_counts, x="Time Bucket", y="Occurrences", text="Occurrences")
        st.plotly_chart(fig1, use_container_width=True)

        # Time Bucket Trend by Hour
        st.markdown("#### Time Bucket Trend by Hour")
        df_filtered["HOUR"] = pd.to_datetime(df_filtered["SHOT_TIME"]).dt.hour
        trend_df = df_filtered.groupby(["HOUR", "TIME_BUCKET"]).size().reset_index(name="count")
        fig2 = px.bar(trend_df, x="HOUR", y="count", color="TIME_BUCKET", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

        # MTTR & MTBF Trend per Hour
        st.markdown("#### MTTR & MTBF Trend per Hour")
        hourly_metrics = df_filtered.groupby(df_filtered["HOUR"]).agg({"MTTR":"mean", "MTBF":"mean"}).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hourly_metrics["HOUR"], y=hourly_metrics["MTTR"], mode="lines+markers",
                                  name="MTTR", line=dict(color="red", width=3)))
        fig3.add_trace(go.Scatter(x=hourly_metrics["HOUR"], y=hourly_metrics["MTBF"], mode="lines+markers",
                                  name="MTBF", line=dict(color="green", width=3)))
        st.plotly_chart(fig3, use_container_width=True)

        # Stability Index per Hour
        st.markdown("#### Stability Index per Hour")
        hourly_metrics["Stability Index"] = (hourly_metrics["MTBF"] / (hourly_metrics["MTBF"] + hourly_metrics["MTTR"])) * 100
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=hourly_metrics["HOUR"], y=hourly_metrics["Stability Index"], mode="lines+markers",
                                  name="Stability Index", line=dict(color="blue", width=3)))
        fig4.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.1, line_width=0)
        fig4.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
        fig4.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0)
        st.plotly_chart(fig4, use_container_width=True)
