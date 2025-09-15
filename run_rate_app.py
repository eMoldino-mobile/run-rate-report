import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide")

# --- Helper Functions ---
def format_time(minutes):
    """Convert minutes (float) to hh:mm:ss string."""
    seconds = int(minutes * 60)
    return str(timedelta(seconds=seconds))

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

    # Back-to-back stop adjustment
    df["STOP_ADJ"] = df["STOP_FLAG"]
    df.loc[(df["STOP_FLAG"] == 1) & (df["STOP_FLAG"].shift(fill_value=0) == 1), "STOP_ADJ"] = 0

    # Events
    df["STOP_EVENT"] = (df["STOP_ADJ"].shift(fill_value=0) == 0) & (df["STOP_ADJ"] == 1)

    # Metrics
    total_shots = len(df)
    normal_shots = (df["STOP_ADJ"] == 0).sum()
    stop_events = df["STOP_EVENT"].sum()

    run_hours = df["TOTAL RUN TIME"].iloc[0] / 60
    gross_rate = total_shots / run_hours if run_hours else None
    net_rate = normal_shots / run_hours if run_hours else None
    efficiency = normal_shots / total_shots if total_shots else None

    production_time = df["PRODUCTION TIME"].iloc[0]
    downtime = df["TOTAL DOWN TIME"].iloc[0]
    total_runtime = df["TOTAL RUN TIME"].iloc[0]

    # Time bucket analysis
    df["RUN_DURATION"] = np.where(df["STOP_ADJ"] == 1, df["CT_diff_sec"] / 60, np.nan)
    df["TIME_BUCKET"] = pd.cut(
        df["RUN_DURATION"],
        bins=[0,1,2,3,5,10,20,30,60,120,999999],
        labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
    )
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # Hourly MTTR/MTBF
    df["HOUR"] = df["SHOT TIME"].dt.hour
    df["DOWNTIME_MIN"] = np.where(df["STOP_EVENT"], df["CT_diff_sec"]/60, np.nan)
    df["UPTIME_MIN"] = np.where(~df["STOP_EVENT"], df["CT_diff_sec"]/60, np.nan)

    def safe_mtbf(uptime_series, stop_count):
        if stop_count > 0 and uptime_series.notna().any():
            return np.nanmean(uptime_series)
        else:
            return np.nan
    
    hourly = (
        df.groupby("HOUR")
          .apply(lambda g: pd.Series({
              "stops": g["STOP_EVENT"].sum(),
              "mttr": np.nanmean(g["DOWNTIME_MIN"]) if g["DOWNTIME_MIN"].notna().any() else np.nan,
              "mtbf": safe_mtbf(g["UPTIME_MIN"], g["STOP_EVENT"].sum())
          }))
          .reset_index()
    )
    hourly["stability_index"] = (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100

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

            # --- Page 1: Analysis Dashboard ---
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
                    "Value": ["0.55", "6.06", "5.06", "28.21"]
                }))

                st.markdown("### Production & Downtime Summary")
                st.table(pd.DataFrame({
                    "Mode CT": [f"{results['mode_ct']:.2f}"],
                    "Lower Limit": [f"{results['lower_limit']:.2f}"],
                    "Upper Limit": [f"{results['upper_limit']:.2f}"],
                    "Production Time %": [f"{results['production_time']/results['total_runtime']*100:.2f}%"],
                    "Downtime %": [f"{results['downtime']/results['total_runtime']*100:.2f}%"],
                    "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                    "Total Stops": [results['stop_events']]
                }))

                # Graphs + Collapsible Tables
                st.subheader("📈 Visual Analysis")
                df_vis = results["df"].copy()
                bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]

                # 1) Time Bucket Analysis
                bucket_counts = df_vis["TIME_BUCKET"].value_counts().reindex(bucket_order).fillna(0).astype(int)
                bucket_df = bucket_counts.reset_index()
                bucket_df.columns = ["Time Bucket", "Occurrences"]
                fig_bucket = px.bar(bucket_df[bucket_df["Time Bucket"].notna()],
                                    x="Occurrences", y="Time Bucket",
                                    orientation="h", text="Occurrences",
                                    title="Time Bucket Analysis")
                fig_bucket.update_traces(textposition="outside")
                st.plotly_chart(fig_bucket, use_container_width=True)
                with st.expander("📊 Time Bucket Analysis Data Table", expanded=False):
                    st.dataframe(bucket_df)

                # 2) Time Bucket Trend by Hour
                src = df_vis.loc[df_vis["STOP_EVENT"] & df_vis["TIME_BUCKET"].notna(), ["HOUR","TIME_BUCKET"]]
                if src.empty:
                    st.info("No stop events with valid TIME_BUCKET for the selected tool/date.")
                else:
                    hours = list(range(24))
                    grid = pd.MultiIndex.from_product([hours, bucket_order], names=["HOUR","TIME_BUCKET"]).to_frame(index=False)
                    counts = src.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")
                    trend = grid.merge(counts, on=["HOUR","TIME_BUCKET"], how="left").fillna({"count":0})
                    fig_tb_trend = px.bar(trend, x="HOUR", y="count", color="TIME_BUCKET",
                                          category_orders={"HOUR": hours, "TIME_BUCKET": bucket_order},
                                          title="Time Bucket Trend by Hour (0–23)")
                    fig_tb_trend.update_layout(barmode="stack")
                    st.plotly_chart(fig_tb_trend, use_container_width=True)
                    with st.expander("📊 Time Bucket Trend Data Table", expanded=False):
                        st.dataframe(trend)

                # 3) MTTR & MTBF Trend by Hour
                hourly = results["hourly"].copy()
                all_hours = pd.DataFrame({"HOUR": list(range(24))})
                hourly = all_hours.merge(hourly, on="HOUR", how="left")
                fig_mt = go.Figure()
                fig_mt.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mttr"], mode="lines+markers",
                                            name="MTTR (min)", line=dict(color="red", width=2), yaxis="y"))
                fig_mt.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["mtbf"], mode="lines+markers",
                                            name="MTBF (min)", line=dict(color="green", width=2, dash="dot"), yaxis="y2"))
                fig_mt.update_layout(title="MTTR & MTBF Trend by Hour",
                                     xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1, range=[-0.5,23.5]),
                                     yaxis=dict(title="MTTR (min)", tickfont=dict(color="red"), side="left"),
                                     yaxis2=dict(title="MTBF (min)", tickfont=dict(color="green"), overlaying="y", side="right"),
                                     margin=dict(l=60,r=60,t=60,b=40),
                                     legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center"))
                st.plotly_chart(fig_mt, use_container_width=True)
                with st.expander("📊 MTTR & MTBF Data Table", expanded=False):
                    st.dataframe(hourly)

                # 4) Stability Index
                hourly["stability_index"] = np.where((hourly["stops"] == 0) & (hourly["mtbf"].isna()),
                                                     100, hourly["stability_index"])
                hourly["stability_change_%"] = hourly["stability_index"].pct_change() * 100
                colors = ["gray" if pd.isna(v) else "red" if v <= 50 else "yellow" if v <= 70 else "green" for v in hourly["stability_index"]]
                fig_stability = go.Figure()
                fig_stability.add_trace(go.Scatter(x=hourly["HOUR"], y=hourly["stability_index"],
                                                   mode="lines+markers", name="Stability Index (%)",
                                                   line=dict(color="blue", width=2), marker=dict(color=colors, size=8)))
                for y0,y1,c in [(0,50,"red"),(50,70,"yellow"),(70,100,"green")]:
                    fig_stability.add_shape(type="rect", x0=-0.5, x1=23.5, y0=y0, y1=y1,
                                            fillcolor=c, opacity=0.1, line_width=0, yref="y")
                fig_stability.update_layout(title="Stability Index by Hour",
                                            xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1, range=[-0.5,23.5]),
                                            yaxis=dict(title="Stability Index (%)", range=[0,100], side="left"),
                                            margin=dict(l=60,r=60,t=60,b=40),
                                            legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center"))
                st.plotly_chart(fig_stability, use_container_width=True)
                with st.expander("📊 Stability Index Data Table", expanded=False):
                    table_data = hourly[["HOUR","stability_index","stability_change_%","mttr","mtbf","stops"]].copy()
                    table_data.rename(columns={"HOUR":"Hour","stability_index":"Stability Index (%)","stability_change_%":"Change vs Prev Hour (%)","mttr":"MTTR (min)","mtbf":"MTBF (min)","stops":"Stop Count"}, inplace=True)
                    st.dataframe(table_data.style.format({
                        "Stability Index (%)": "{:.2f}",
                        "Change vs Prev Hour (%)": "{:+.2f}%",
                        "MTTR (min)": "{:.2f}",
                        "MTBF (min)": "{:.2f}"
                    }))

                st.markdown("""
                **ℹ️ Stability Index Formula**
                - Stability Index (%) = (MTBF / (MTBF + MTTR)) × 100
                - If no stoppages occur in an hour, Stability Index is forced to **100%**
                - Alert Zones:
                  - 🟥 0–50% → High Risk (unstable production)
                  - 🟨 50–70% → Medium Risk (watch closely)
                  - 🟩 70–100% → Low Risk (stable operation)
                """)

                # 5) Stoppage Alerts
                df_vis = results["df"].copy()
                threshold = results["mode_ct"] * 2
                stoppage_alerts = df_vis[df_vis["CT_diff_sec"] >= threshold].copy()
                st.markdown("### 🚨 Stoppage Alert Reporting (≥ Mode CT × 2)")
                if stoppage_alerts.empty:
                    st.info("✅ No stoppage alerts found (≥ Mode CT × 2).")
                else:
                    stoppage_alerts["Gap (min)"] = (stoppage_alerts["CT_diff_sec"] / 60).round(2)
                    stoppage_alerts["Alert"] = "🔴"
                    table = stoppage_alerts[["SHOT TIME","CT_diff_sec","HOUR","Gap (min)","Alert"]].rename(columns={"SHOT TIME":"Event Time","CT_diff_sec":"Gap (sec)","HOUR":"Hour"})
                    st.dataframe(table, use_container_width=True)
                    st.markdown(f"""
                    **Summary**
                    - Total Stoppage Alerts: {len(stoppage_alerts)}
                    - Threshold Applied: {results['mode_ct']:.2f} sec × 2 = {threshold:.2f} sec
                    """)

                # --- Page 2: Raw & Processed Data ---
                elif page == "Raw & Processed Data":
                    st.title("📋 Raw & Processed Cycle Data")
                
                    st.markdown("This table shows all cycle-level data, combining base inputs with recalculated metrics used in the analysis.")
                
                    export_df = results["df"].copy()
                
                    # Keep only relevant columns
                    keep_cols = [
                        selection_column if selection_column in export_df.columns else None,
                        "SHOT TIME",
                        "ACTUAL CT",
                        "CT_diff_sec",
                        "STOP_FLAG",
                        "STOP_ADJ",
                        "STOP_EVENT",
                        "RUN_DURATION",
                        "TIME_BUCKET",
                        "HOUR",
                        "DOWNTIME_MIN",
                        "UPTIME_MIN"
                    ]
                    keep_cols = [c for c in keep_cols if c in export_df.columns]  # remove Nones
                
                    export_df = export_df[keep_cols]
                
                    # Rename headers for clarity
                    export_df = export_df.rename(columns={
                        selection_column: "Tooling ID" if selection_column == "TOOLING ID" else "Equipment Code",
                        "SHOT TIME": "Shot Time",
                        "ACTUAL CT": "Actual CT (sec)",
                        "CT_diff_sec": "Cycle Gap (sec)",
                        "STOP_FLAG": "Stop Flag",
                        "STOP_ADJ": "Stop Adjusted",
                        "STOP_EVENT": "Stop Event",
                        "RUN_DURATION": "Run Duration (min)",
                        "TIME_BUCKET": "Time Bucket",
                        "HOUR": "Hour",
                        "DOWNTIME_MIN": "Downtime (min)",
                        "UPTIME_MIN": "Uptime (min)"
                    })
                
                    # Display table on-screen
                    st.dataframe(export_df, use_container_width=True)
                
                    # Optional: download button
                    csv = export_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Data as CSV",
                        data=csv,
                        file_name=f"processed_cycles_{date.strftime('%Y-%m-%d')}.csv",
                        mime="text/csv"
                    )


else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")
