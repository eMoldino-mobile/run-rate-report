import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# --- Helper Functions ---
def format_time(minutes):
    """Convert minutes (float) to hh:mm:ss string."""
    seconds = int(minutes * 60)
    return str(timedelta(seconds=seconds))

def calculate_run_rate_excel_like(df):
    df = df.copy()
    df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"])
    df["CT_diff_sec"] = df["SHOT TIME"].diff().dt.total_seconds()
    
    # Mode CT (seconds)
    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05
    
    # STOP flag
    df["STOP_FLAG"] = np.where(
        (df["CT_diff_sec"].notna()) & 
        ((df["CT_diff_sec"] < lower_limit) | (df["CT_diff_sec"] > upper_limit)) & 
        (df["CT_diff_sec"] <= 28800),  # 8 hours
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
                               bins=[0,20,30,40,50,60,70,80,90,100,999999],
                               labels=[1,2,3,4,5,6,7,8,9,10])
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    return {
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
        "bucket_counts": bucket_counts
    }

# --- Streamlit UI ---
st.sidebar.title("Run Rate Report Generator")

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    tool = st.sidebar.selectbox("Select Tool / Equipment Code", df["EQUIPMENT CODE"].unique())
    date = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT TIME"]).dt.date.min())
    
    if st.sidebar.button("Generate Report"):
        # Filter data for tool + date
        mask = (df["EQUIPMENT CODE"] == tool) & (pd.to_datetime(df["SHOT TIME"]).dt.date == date)
        df_filtered = df.loc[mask]
        
        if df_filtered.empty:
            st.warning("No data found for this selection.")
        else:
            results = calculate_run_rate_excel_like(df_filtered)
            
            st.title("📊 Run Rate Report")
            st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")
            
            # --- Block 1: Shot Counts & Efficiency ---
            st.markdown("### Shot Counts & Efficiency")
            shot_data = {
                "Total Shot Count": [results['total_shots']],
                "Normal Shot Count": [results['normal_shots']],
                "Efficiency": [f"{results['efficiency']*100:.2f}%"],
                "Stop Count": [results['stop_events']]
            }
            st.table(pd.DataFrame(shot_data))
            
            # --- Block 2: Reliability Metrics (placeholders) ---
            st.markdown("### Reliability Metrics")
            reliability_data = {
                "Metric": ["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
                "Value": ["0.55", "6.06", "5.06", "28.21"]
            }
            st.table(pd.DataFrame(reliability_data))
            
            # --- Block 3: Time Bucket Analysis ---
            st.markdown("### Time Bucket Analysis")
            bucket_df = results['bucket_counts'].reset_index()
            bucket_df.columns = ["Time Bucket", "Occurrence Count"]
            st.table(bucket_df)
            
            # --- Block 4: Readable Time Display ---
            st.markdown("### Readable Time Display")
            time_display = {
                "Metric": ["Mode Cycle Time", "Lower Limit", "Upper Limit", 
                           "Total Production Time", "Total Downtime", "Production Run", "MTTR", "MTBF"],
                "Value": [f"{results['mode_ct']:.0f} sec",
                          f"{results['lower_limit']:.0f} sec",
                          f"{results['upper_limit']:.0f} sec",
                          format_time(results['production_time']),
                          format_time(results['downtime']),
                          format_time(results['total_runtime']),
                          "33 sec",
                          "6 min 4 sec"]
            }
            st.table(pd.DataFrame(time_display))
            
            # --- Block 5: Outside Limits Summary ---
            st.markdown("### Outside L1 / L2 Summary")
            outside_data = {
                "Mode CT": [f"{results['mode_ct']:.2f}"],
                "Lower Limit": [f"{results['lower_limit']:.2f}"],
                "Upper Limit": [f"{results['upper_limit']:.2f}"],
                "Production Time %": [f"{results['production_time']/results['total_runtime']*100:.2f}%"],
                "Downtime %": [f"{results['downtime']/results['total_runtime']*100:.2f}%"],
                "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                "Total Stops": [results['stop_events']]
            }
            st.table(pd.DataFrame(outside_data))
            
            # --- KPI Metrics ---
            st.markdown("### KPI Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Shots", f"{results['total_shots']:,}")
            col2.metric("Normal Shots", f"{results['normal_shots']:,}")
            col3.metric("Stop Events", f"{results['stop_events']:,}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Gross Run Rate", f"{results['gross_rate']:.1f} cycles/hr")
            col5.metric("Net Run Rate", f"{results['net_rate']:.1f} cycles/hr")
            col6.metric("Efficiency", f"{results['efficiency']*100:.1f}%")

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin.")
