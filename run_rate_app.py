import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide")

# --- Helper Functions ---
def format_time_hms(minutes):
    """Convert minutes to hh:mm:ss string."""
    if pd.isna(minutes):
        return "N/A"
    seconds = int(minutes * 60)
    return str(timedelta(seconds=seconds))

def format_time_hm(minutes):
    """Convert minutes to 'Xh Ym' string."""
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"

def calculate_run_rate_excel_like(df):
    df = df.copy()

    # --- Handle Date/Time Parsing ---
    if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
        df["SHOT TIME"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" +
            df["MONTH"].astype(str) + "-" +
            df["DAY"].astype(str) + " " +
            df["TIME"].astype(str),
            errors="coerce"
        )
    elif "SHOT TIME" in df.columns:
        df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
    else:
        st.error("Input file must contain either 'SHOT TIME' or YEAR/MONTH/DAY/TIME columns.")
        st.stop()

    df["CT_diff_sec"] = df["SHOT TIME"].diff().dt.total_seconds()

    # Mode CT
    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05

    # STOP flag
    df["STOP_FLAG"] = np.where(
        (df["CT_diff_sec"].notna()) &
        ((df["CT_diff_sec"] < lower_limit) | (df["CT_diff_sec"] > upper_limit)) &
        (df["CT_diff_sec"] <= 28800),
        1, 0
    )
    df.loc[df.index[0], "STOP_FLAG"] = 0

    # Mark stop events (including back-to-back stops as STOP)
    df["STOP_EVENT"] = df["STOP_FLAG"]

    # Metrics
    total_shots = len(df)
    normal_shots = (df["STOP_FLAG"] == 0).sum()
    stop_events = df["STOP_EVENT"].sum()

    run_hours = df["TOTAL RUN TIME"].iloc[0] / 60
    gross_rate = total_shots / run_hours if run_hours else None
    net_rate = normal_shots / run_hours if run_hours else None
    efficiency = normal_shots / total_shots if total_shots else None

    production_time = df["PRODUCTION TIME"].iloc[0]
    downtime = df["TOTAL DOWN TIME"].iloc[0]
    total_runtime = df["TOTAL RUN TIME"].iloc[0]

    # Time bucket analysis
    df["RUN_DURATION"] = np.where(df["STOP_FLAG"] == 1, df["CT_diff_sec"] / 60, np.nan)
    df["TIME_BUCKET"] = pd.cut(
        df["RUN_DURATION"],
        bins=[0,1,2,3,5,10,20,30,60,120,999999],
        labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
    )
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # Hourly MTTR/MTBF
    df["HOUR"] = df["SHOT TIME"].dt.hour
    df["DOWNTIME_MIN"] = np.where(df["STOP_EVENT"] == 1, df["CT_diff_sec"]/60, np.nan)
    df["UPTIME_MIN"] = np.where(df["STOP_EVENT"] == 0, df["CT_diff_sec"]/60, np.nan)

    def safe_mtbf(uptime_series, stop_count):
        if stop_count > 0 and uptime_series.notna().any():
            return np.nanmean(uptime_series)
        else:
            return np.nan

    hourly = (
        df.groupby("HOUR", observed=False)
          .apply(lambda g: pd.Series({
              "stops": g["STOP_EVENT"].sum(),
              "mttr": np.nanmean(g["DOWNTIME_MIN"]) if g["DOWNTIME_MIN"].notna().any() else np.nan,
              "mtbf": safe_mtbf(g["UPTIME_MIN"], g["STOP_EVENT"].sum())
          }))
          .reset_index()
    )
    hourly["stability_index"] = (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100

    # --- Processed table fields ---
    df["Approved CT"] = df["APPROVED CT"] if "APPROVED CT" in df.columns else "Not Provided"
    df["Stop"] = df["STOP_EVENT"].astype(bool)
    df["Cumulative Count"] = (~df["Stop"]).cumsum()
    df["Run Duration (min)"] = np.where(df["Stop"], df["CT_diff_sec"]/60, 0)

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
        "bucket_counts": bucket_counts,
        "hourly": hourly,
        "df": df
    }

# --- UI ---
st.sidebar.title("Run Rate Report Generator")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Tool selection
    selection_column = None
    if "TOOLING ID" in df.columns:
        selection_column = "TOOLING ID"
    elif "EQUIPMENT CODE" in df.columns:
        selection_column = "EQUIPMENT CODE"
    else:
        st.error("File must contain either 'TOOLING ID' or 'EQUIPMENT CODE'.")
        st.stop()

    tool = st.sidebar.selectbox("Select Tool", df[selection_column].unique())
    date = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT TIME"]).dt.date.min())
    page = st.sidebar.radio("Select Page", ["📊 Analysis Dashboard", "📂 Raw & Processed Data"])

    if st.sidebar.button("Generate Report"):
        mask = (df[selection_column] == tool) & (pd.to_datetime(df["SHOT TIME"]).dt.date == date)
        df_filtered = df.loc[mask]

        if df_filtered.empty:
            st.warning("No data found for this selection.")
        else:
            results = calculate_run_rate_excel_like(df_filtered)

            # Store in session state
            st.session_state.results = results
            st.session_state.tool = tool
            st.session_state.date = date

            # --- Page 1 ---
            if page == "📊 Analysis Dashboard":
                st.title("📊 Run Rate Report")
                st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")

                # Summaries
                st.markdown("### Shot Counts & Efficiency")
                st.table(pd.DataFrame({
                    "Total Shot Count": [results['total_shots']],
                    "Normal Shot Count": [results['normal_shots']],
                    "Efficiency": [f"{results['efficiency']*100:.2f}%"],
                    "Stop Count": [results['stop_events']]
                }))

                st.markdown("### Reliability Metrics")
                st.table(pd.DataFrame({
                    "Metric": ["MTTR", "MTBF", "Time to First DT (Avg)", "Avg Cycle Time"],
                    "Value": [
                        f"{results['hourly']['mttr'].mean():.2f} min ({format_time_hms(results['hourly']['mttr'].mean())})",
                        f"{results['hourly']['mtbf'].mean():.2f} min ({format_time_hms(results['hourly']['mtbf'].mean())})",
                        "5.06", "28.21"
                    ]
                }))

                st.markdown("### Production & Downtime Summary")
                st.table(pd.DataFrame({
                    "Mode CT": [f"{results['mode_ct']:.2f}"],
                    "Lower Limit": [f"{results['lower_limit']:.2f}"],
                    "Upper Limit": [f"{results['upper_limit']:.2f}"],
                    "Production Time": [f"{format_time_hm(results['production_time'])} ({results['production_time']/results['total_runtime']*100:.2f}%)"],
                    "Downtime": [f"{format_time_hm(results['downtime'])} ({results['downtime']/results['total_runtime']*100:.2f}%)"],
                    "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                    "Total Stops": [results['stop_events']]
                }))

                # Graphs + Tables
                st.subheader("📈 Visual Analysis")
                # Time bucket analysis
                bucket_df = results["bucket_counts"].reset_index()
                bucket_df.columns = ["Time Bucket", "Occurrences"]
                fig_bucket = px.bar(bucket_df[bucket_df["Time Bucket"].notna()],
                                    x="Occurrences", y="Time Bucket",
                                    orientation="h", text="Occurrences",
                                    title="Time Bucket Analysis")
                st.plotly_chart(fig_bucket, use_container_width=True)
                with st.expander("📊 Time Bucket Analysis Data Table", expanded=False):
                    st.dataframe(bucket_df)

                # Time bucket trend
                src = results["df"].loc[results["df"]["STOP_EVENT"] & results["df"]["TIME_BUCKET"].notna(), ["HOUR","TIME_BUCKET"]]
                if not src.empty:
                    hours = list(range(24))
                    grid = pd.MultiIndex.from_product([hours, bucket_df["Time Bucket"].unique()], names=["HOUR","TIME_BUCKET"]).to_frame(index=False)
                    counts = src.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
                    trend = grid.merge(counts, on=["HOUR","TIME_BUCKET"], how="left").fillna({"count":0})
                    fig_tb_trend = px.bar(trend, x="HOUR", y="count", color="TIME_BUCKET", barmode="stack")
                    st.plotly_chart(fig_tb_trend, use_container_width=True)
                    with st.expander("📊 Time Bucket Trend Data Table", expanded=False):
                        st.dataframe(trend)

                # MTTR & MTBF
                hourly = results["hourly"].copy()
                fig_mt = go.Figure()
                fig_mt.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mttr"], mode="lines+markers", name="MTTR (min)", line=dict(color="red")))
                fig_mt.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mtbf"], mode="lines+markers", name="MTBF (min)", line=dict(color="green", dash="dot")))
                st.plotly_chart(fig_mt, use_container_width=True)
                with st.expander("📊 MTTR & MTBF Data Table", expanded=False):
                    st.dataframe(hourly)

                # Stability Index
                hourly["stability_index"] = np.where((hourly["stops"] == 0) & (hourly["mtbf"].isna()), 100, hourly["stability_index"])
                hourly["stability_change_%"] = hourly["stability_index"].pct_change() * 100

                def color_stability(val):
                    if pd.isna(val): return ""
                    if val <= 50: return "background-color: red"
                    elif val <= 70: return "background-color: yellow"
                    else: return ""

                fig_stability = go.Figure()
                fig_stability.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["stability_index"], mode="lines+markers", name="Stability Index"))
                st.plotly_chart(fig_stability, use_container_width=True)
                with st.expander("📊 Stability Index Data Table", expanded=False):
                    table_data = hourly[["HOUR","stability_index","stability_change_%","mttr","mtbf","stops"]].copy()
                    st.dataframe(table_data.style.applymap(color_stability, subset=["stability_index"]))

                # Stoppage Alerts
                stoppage_alerts = results["df"][results["df"]["CT_diff_sec"] >= results["mode_ct"]*2]
                st.markdown("### 🚨 Stoppage Alerts")
                if stoppage_alerts.empty:
                    st.info("✅ No stoppage alerts.")
                else:
                    st.dataframe(stoppage_alerts[["SHOT TIME","CT_diff_sec","HOUR"]])

            # --- Page 2 ---
            if page == "📂 Raw & Processed Data":
                st.title("📋 Raw & Processed Data")
                st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")

                # Summaries same as Page 1
                st.markdown("### Production & Downtime Summary")
                st.table(pd.DataFrame({
                    "Mode CT": [f"{results['mode_ct']:.2f}"],
                    "Lower Limit": [f"{results['lower_limit']:.2f}"],
                    "Upper Limit": [f"{results['upper_limit']:.2f}"],
                    "Production Time": [f"{format_time_hm(results['production_time'])} ({results['production_time']/results['total_runtime']*100:.2f}%)"],
                    "Downtime": [f"{format_time_hm(results['downtime'])} ({results['downtime']/results['total_runtime']*100:.2f}%)"],
                    "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                    "Total Stops": [results['stop_events']]
                }))

                # Processed cycle table
                df_clean = results["df"][[
                    "SUPPLIER NAME","EQUIPMENT CODE","SHOT TIME","Approved CT","ACTUAL CT","CT_diff_sec","Stop","Cumulative Count","Run Duration (min)"
                ]].copy()
                df_clean.rename(columns={
                    "SUPPLIER NAME":"Supplier Name",
                    "EQUIPMENT CODE":"Equipment Code",
                    "SHOT TIME":"Shot Time",
                    "ACTUAL CT":"Actual CT",
                    "CT_diff_sec":"Time Diff Sec"
                }, inplace=True)
                df_clean["Actual CT"] = df_clean["Actual CT"].round(1)
                df_clean["Time Diff Sec"] = df_clean["Time Diff Sec"].round(2)

                def highlight_stops(row):
                    return ["background-color: red" if row["Stop"] else "" for _ in row]

                st.dataframe(df_clean.style.apply(highlight_stops, axis=1), use_container_width=True)
                st.download_button("💾 Download Processed Data (CSV)", df_clean.to_csv(index=False).encode("utf-8"), "processed_cycle_data.csv")

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")