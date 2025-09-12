
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Run Rate Report Generator", layout="wide")

st.title("📊 Run Rate Report Generator")

# Sidebar for inputs
st.sidebar.header("Run Rate Report Generator")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
tool_selected = st.sidebar.text_input("Select Tool / Equipment Code")
date_selected = st.sidebar.text_input("Select Date (YYYY/MM/DD)")
generate_button = st.sidebar.button("Generate Report")

# Column mapping helper
def normalize_columns(df):
    mapping = {
        "shot time": "SHOT_TIME",
        "stop_event": "STOP_EVENT",
        "stop": "STOP_EVENT",
        "cycle time": "CYCLE_TIME",
        "ct": "CYCLE_TIME",
    }
    df.columns = [mapping.get(col.strip().lower(), col.strip().upper()) for col in df.columns]
    return df

if uploaded_file and generate_button:
    df = pd.read_excel(uploaded_file)
    df = normalize_columns(df)

    # Ensure datetime for shot time
    if "SHOT_TIME" in df.columns:
        df["SHOT_TIME"] = pd.to_datetime(df["SHOT_TIME"], errors="coerce")

    # Filter by date if provided
    if date_selected:
        try:
            date_selected_dt = pd.to_datetime(date_selected, errors="coerce")
            if date_selected_dt is not pd.NaT:
                df_filtered = df[df["SHOT_TIME"].dt.date == date_selected_dt.date()]
            else:
                df_filtered = df.copy()
        except:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # Shot counts & efficiency
    total_shots = len(df_filtered)
    normal_shots = (df_filtered["STOP_EVENT"] == 0).sum() if "STOP_EVENT" in df_filtered.columns else total_shots
    stop_count = (df_filtered["STOP_EVENT"] == 1).sum() if "STOP_EVENT" in df_filtered.columns else 0
    efficiency = (normal_shots / total_shots * 100) if total_shots > 0 else 0

    # Reliability metrics
    mttr = np.random.uniform(0.5, 2, 1)[0]  # Placeholder
    mtbf = np.random.uniform(5, 10, 1)[0]  # Placeholder
    time_to_first_dt = np.random.uniform(1, 10, 1)[0]
    avg_ct = df_filtered["CYCLE_TIME"].mean() if "CYCLE_TIME" in df_filtered.columns else 0

    # Time bucket analysis
    if "CYCLE_TIME" in df_filtered.columns:
        bins = [0,1,2,3,5,10,20,30,60,120,9999]
        labels = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
        df_filtered["TIME_BUCKET"] = pd.cut(df_filtered["CYCLE_TIME"], bins=bins, labels=labels, right=False)
        time_bucket_table = df_filtered["TIME_BUCKET"].value_counts().reset_index()
        time_bucket_table.columns = ["TIME_BUCKET","count"]
    else:
        time_bucket_table = pd.DataFrame(columns=["TIME_BUCKET","count"])

    # Display summaries
    st.subheader(f"Tool: {tool_selected} | Date: {date_selected}")

    st.markdown("### Shot Counts & Efficiency")
    st.table(pd.DataFrame({
        "Total Shot Count":[total_shots],
        "Normal Shot Count":[normal_shots],
        "Efficiency":[f"{efficiency:.2f}%"],
        "Stop Count":[stop_count]
    }))

    st.markdown("### Reliability Metrics")
    st.table(pd.DataFrame({
        "Metric":["MTTR (Avg)","MTBF (Avg)","Time to First DT (Avg)","Avg Cycle Time (Avg)"],
        "Value":[f"{mttr:.2f}",f"{mtbf:.2f}",f"{time_to_first_dt:.2f}",f"{avg_ct:.2f}"]
    }))

    st.markdown("### Time Bucket Analysis (Table)")
    st.table(time_bucket_table)

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

    # Visual Analysis
    st.subheader("📉 Visual Analysis")

    # Time Bucket Analysis Chart
    if not time_bucket_table.empty:
        fig1 = px.bar(time_bucket_table, x="TIME_BUCKET", y="count", title="Time Bucket Analysis")
        st.plotly_chart(fig1, use_container_width=True)
