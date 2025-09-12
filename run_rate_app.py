
import streamlit as st
import pandas as pd
import plotly.express as px

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

    # Detect columns flexibly
    tool_col = detect_column(df, ["TOOL", "EQUIP", "MOLD"])
    date_col = detect_column(df, ["DATE"])
    shot_time_col = detect_column(df, ["SHOT", "CYCLE"])
    stop_event_col = detect_column(df, ["STOP"])
    cycle_time_col = detect_column(df, ["CYCLE TIME", "CT"])

    st.write("### Detected Columns")
    st.write({
        "Tool Column": tool_col,
        "Date Column": date_col,
        "Shot Time Column": shot_time_col,
        "Stop Event Column": stop_event_col,
        "Cycle Time Column": cycle_time_col,
    })

    # Preview first rows
    st.write("### Data Preview")
    st.dataframe(df.head())
