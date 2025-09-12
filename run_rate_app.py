
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Run Rate Report Generator", layout="wide")

# --- File Upload ---
st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.upper()

    # Detect columns dynamically
    tool_col = [c for c in df.columns if "TOOL" in c or "EQUIPMENT" in c][0]
    shot_time_col = [c for c in df.columns if "SHOT" in c and "TIME" in c][0]
    stop_col = [c for c in df.columns if "STOP" in c][0]
    cycle_time_col = [c for c in df.columns if "CT" in c][0]

    df[shot_time_col] = pd.to_datetime(df[shot_time_col], errors="coerce")

    tools = df[tool_col].dropna().unique()
    selected_tool = st.sidebar.selectbox("Select Tool / Equipment Code", tools)

    dates = pd.to_datetime(df[shot_time_col].dt.date.unique())
    selected_date = st.sidebar.selectbox("Select Date", dates.astype(str))

    df_filtered = df[(df[tool_col] == selected_tool) & (df[shot_time_col].dt.date.astype(str) == selected_date)]

    if not df_filtered.empty:
        # --- Core Calculations ---
        total_shots = len(df_filtered)
        normal_shots = (df_filtered[stop_col] == 0).sum()
        stop_count = (df_filtered[stop_col] == 1).sum()
        efficiency = (normal_shots / total_shots) * 100 if total_shots > 0 else 0

        cycle_times = df_filtered[cycle_time_col].dropna()
        mode_ct = cycle_times.mode().iloc[0] if not cycle_times.empty else 0
        lower_limit = mode_ct * 0.95
        upper_limit = mode_ct * 1.05

        df_filtered = df_filtered.copy()
        df_filtered["DURATION"] = df_filtered[cycle_time_col]
        df_filtered["TIME_BUCKET"] = pd.cut(
            df_filtered["DURATION"],
            bins=[0,1,2,3,5,10,20,30,60,120,999999],
            labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
        )

        bucket_counts = df_filtered["TIME_BUCKET"].value_counts().sort_index()

        # MTTR / MTBF approximations
        mttr = df_filtered.loc[df_filtered[stop_col] == 1, "DURATION"].mean() if stop_count > 0 else 0
        mtbf = df_filtered.loc[df_filtered[stop_col] == 0, "DURATION"].mean() if normal_shots > 0 else 0

        # --- Tabs ---
        summary_tab, visual_tab = st.tabs(["📑 Summary", "📈 Visual Analysis"])

        with summary_tab:
            st.subheader(f"Tool: {selected_tool} | Date: {selected_date}")

            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count": [total_shots],
                "Normal Shot Count": [normal_shots],
                "Efficiency": [f"{efficiency:.2f}%"],
                "Stop Count": [stop_count]
            }))

            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
                "Value": [round(mttr,2), round(mtbf,2), 0 if df_filtered.empty else round(df_filtered["DURATION"].iloc[0],2), round(df_filtered["DURATION"].mean(),2)]
            }))

            st.markdown("### Time Bucket Analysis (Table)")
            st.table(bucket_counts.reset_index().rename(columns={"index":"Time Bucket", "TIME_BUCKET":"Occurrences"}))

            st.markdown("### Readable Time Display")
            total_production_time = df_filtered["DURATION"].sum()
            total_downtime = df_filtered.loc[df_filtered[stop_col] == 1, "DURATION"].sum()
            production_run = df_filtered.loc[df_filtered[stop_col] == 0, "DURATION"].sum()

            readable_df = pd.DataFrame({
                "Metric": [
                    "Mode Cycle Time","Lower Limit","Upper Limit",
                    "Total Production Time","Total Downtime","Production Run","MTTR","MTBF"
                ],
                "Value": [
                    f"{mode_ct:.0f} sec",f"{lower_limit:.0f} sec",f"{upper_limit:.0f} sec",
                    f"{pd.to_timedelta(total_production_time, unit='s')}",
                    f"{pd.to_timedelta(total_downtime, unit='s')}",
                    f"{pd.to_timedelta(production_run, unit='s')}",
                    f"{mttr:.0f} sec",f"{mtbf:.0f} sec"
                ]
            })
            st.table(readable_df)

            st.markdown("### Outside L1 / L2 Summary")
            st.table(pd.DataFrame({
                "Mode CT": [round(mode_ct,2)],
                "Lower Limit": [round(lower_limit,2)],
                "Upper Limit": [round(upper_limit,2)],
                "Production Time %": [f"{(production_run/total_production_time*100):.2f}%" if total_production_time>0 else "0%"],
                "Downtime %": [f"{(total_downtime/total_production_time*100):.2f}%" if total_production_time>0 else "0%"],
                "Total Run Time (hrs)": [round(total_production_time/3600,2)],
                "Total Stops": [stop_count]
            }))

        with visual_tab:
            st.subheader("Time Bucket Analysis (Chart)")
            fig1 = px.bar(bucket_counts.reset_index(), x="index", y="TIME_BUCKET",
                          labels={"index":"Time Bucket","TIME_BUCKET":"Occurrences"})
            st.plotly_chart(fig1, use_container_width=True)
