import streamlit as st
import pandas as pd
import numpy as np

# --- Functions ---
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
    
    return {
        "mode_ct": mode_ct,
        "total_shots": total_shots,
        "normal_shots": normal_shots,
        "stop_events": stop_events,
        "run_hours": run_hours,
        "gross_rate": gross_rate,
        "net_rate": net_rate,
        "efficiency": efficiency
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
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Shots", f"{results['total_shots']:,}")
            col2.metric("Normal Shots", f"{results['normal_shots']:,}")
            col3.metric("Stop Events", f"{results['stop_events']:,}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Gross Run Rate", f"{results['gross_rate']:.1f} shots/hr")
            col5.metric("Net Run Rate", f"{results['net_rate']:.1f} shots/hr")
            col6.metric("Efficiency", f"{results['efficiency']*100:.1f}%")
            
            st.caption(f"Run Hours: {results['run_hours']:.2f} h | Mode CT: {results['mode_ct']:.1f} sec")

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin.")
