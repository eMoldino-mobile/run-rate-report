
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Run Rate Report", layout="wide")

st.title("📊 Run Rate Report Generator")

uploaded_file = st.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
selected_tool = st.text_input("Select Tool / Equipment Code", "6700-074")
selected_date = st.date_input("Select Date")

if uploaded_file and selected_tool and selected_date:
    df = pd.read_excel(uploaded_file)
    df['SHOT TIME'] = pd.to_datetime(df['SHOT TIME'])
    df['HOUR'] = df['SHOT TIME'].dt.hour

    # Filter by selected date
    df_filtered = df[df['SHOT TIME'].dt.date == selected_date]

    # Core metrics
    total_shots = len(df_filtered)
    normal_shots = (df_filtered['STOP_EVENT'] == 0).sum()
    stop_count = (df_filtered['STOP_EVENT'] == 1).sum()
    efficiency = normal_shots / total_shots * 100 if total_shots > 0 else 0

    # Reliability
    mttr = df_filtered.loc[df_filtered['STOP_EVENT'] == 1, 'RUN_DURATION'].mean() / 60 if stop_count > 0 else 0
    mtbf = df_filtered['CT_diff_sec'].mean() / 60 if stop_count > 0 else 0
    avg_ct = df_filtered['CT_diff_sec'].mean() if total_shots > 0 else 0
    time_first_dt = df_filtered.loc[df_filtered['STOP_EVENT'] == 1, 'CT_diff_sec'].min() / 60 if stop_count > 0 else 0

    # Mode CT and limits
    mode_ct = df_filtered['CT_diff_sec'].mode().iloc[0] / 60 if not df_filtered.empty else 0
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05

    # Time calculations
    prod_time = df_filtered.loc[df_filtered['STOP_EVENT'] == 0, 'CT_diff_sec'].sum() / 60
    down_time = df_filtered.loc[df_filtered['STOP_EVENT'] == 1, 'CT_diff_sec'].sum() / 60
    total_run_time = prod_time + down_time

    # --- Summaries ---
    st.header(f"Tool: {selected_tool} | Date: {selected_date}")

    st.subheader("Shot Counts & Efficiency")
    st.table({
        "Total Shot Count": [total_shots],
        "Normal Shot Count": [normal_shots],
        "Efficiency": [f"{efficiency:.2f}%"],
        "Stop Count": [stop_count]
    })

    st.subheader("Reliability Metrics")
    st.table(pd.DataFrame({
        "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
        "Value": [f"{mttr:.2f}", f"{mtbf:.2f}", f"{time_first_dt:.2f}", f"{avg_ct:.2f}"]
    }))

    st.subheader("Time Bucket Analysis (Table)")
    df_filtered["TIME_BUCKET"] = pd.cut(df_filtered["CT_diff_sec"]/60,
        bins=[0,1,2,3,5,10,20,30,60,120,9999],
        labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
    )
    bucket_table = df_filtered.groupby("TIME_BUCKET").size().reset_index(name="count")
    bucket_table.loc[len(bucket_table)] = ["Grand Total", bucket_table['count'].sum()]
    st.table(bucket_table)

    st.subheader("Readable Time Display")
    readable = pd.DataFrame({
        "Metric": ["Mode Cycle Time","Lower Limit","Upper Limit","Total Production Time",
                   "Total Downtime","Production Run","MTTR","MTBF"],
        "Value": [f"{mode_ct:.0f} sec", f"{lower_limit:.0f} sec", f"{upper_limit:.0f} sec",
                  f"{prod_time:.0f} min", f"{down_time:.0f} min", f"{total_run_time:.0f} min",
                  f"{mttr*60:.0f} sec", f"{mtbf*60:.0f} sec"]
    })
    st.table(readable)

    st.subheader("Outside L1 / L2 Summary")
    st.table({
        "Mode CT": [round(mode_ct,2)],
        "Lower Limit": [round(lower_limit,2)],
        "Upper Limit": [round(upper_limit,2)],
        "Production Time %": [f"{prod_time/total_run_time*100:.2f}%"],
        "Downtime %": [f"{down_time/total_run_time*100:.2f}%"],
        "Total Run Time (hrs)": [round(total_run_time/60,2)],
        "Total Stops": [stop_count]
    })

    # --- Visual Analysis ---
    st.header("📈 Visual Analysis")

    # 1. Time Bucket Analysis (Bar)
    fig1 = px.bar(bucket_table[:-1], x="TIME_BUCKET", y="count",
                  title="Time Bucket Analysis", text="count")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Time Bucket Trend by Hour
    trend_df = df_filtered.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
    all_hours = pd.DataFrame({"HOUR": range(24)})
    all_buckets = pd.DataFrame({"TIME_BUCKET": df_filtered["TIME_BUCKET"].cat.categories})
    grid = all_hours.merge(all_buckets, how="cross")
    trend_df = grid.merge(trend_df, on=["HOUR","TIME_BUCKET"], how="left").fillna(0)
    fig2 = px.bar(trend_df, x="HOUR", y="count", color="TIME_BUCKET",
                  title="Time Bucket Trend by Hour", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Combined MTTR, MTBF, Stability Index
    hourly = df_filtered.groupby("HOUR").agg(
        mttr=("RUN_DURATION", lambda x: np.nanmean(x)/60 if len(x)>0 else 0),
        mtbf=("CT_diff_sec", lambda x: np.nanmean(x)/60 if len(x)>0 else 0)
    ).reindex(range(24), fill_value=0).reset_index()

    hourly["stability_index"] = (hourly["mtbf"]/(hourly["mtbf"]+hourly["mttr"]))*100
    fig3 = go.Figure()

    # Alert zones
    fig3.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.1, line_width=0)
    fig3.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
    fig3.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0)

    # Lines
    fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mttr"],
                              mode="lines+markers", name="MTTR",
                              line=dict(color="red", width=4)))
    fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mtbf"],
                              mode="lines+markers", name="MTBF",
                              line=dict(color="green", width=4)))
    fig3.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["stability_index"],
                              mode="lines+markers", name="Stability Index",
                              line=dict(color="blue", width=3, dash="dot")))
    fig3.update_layout(title="Process Stability (MTTR, MTBF, Stability Index)",
                       xaxis_title="Hour of Day", yaxis_title="Minutes / Index")
    st.plotly_chart(fig3, use_container_width=True)
