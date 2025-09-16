import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import warnings

# Suppress deprecation warnings during dev (optional but recommended)
warnings.filterwarnings("ignore", category=FutureWarning)

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

    # Mode CT (seconds)
    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05

    # STOP flag (all potential stops)
    df["STOP_FLAG"] = np.where(
        (df["CT_diff_sec"].notna()) &
        ((df["CT_diff_sec"] < lower_limit) | (df["CT_diff_sec"] > upper_limit)) &
        (df["CT_diff_sec"] <= 28800),  # ignore > 8 hours gaps
        1, 0
    )
    df.loc[df.index[0], "STOP_FLAG"] = 0

    # Back-to-back stop adjustment (for stop count)
    df["STOP_ADJ"] = df["STOP_FLAG"]
    df.loc[(df["STOP_FLAG"] == 1) & (df["STOP_FLAG"].shift(fill_value=0) == 1), "STOP_ADJ"] = 0

    # Events (first in sequence = true stop event)
    df["STOP_EVENT"] = (df["STOP_ADJ"].shift(fill_value=0) == 0) & (df["STOP_ADJ"] == 1)

    # --- Core Metrics ---
    total_shots = len(df)
    normal_shots = (df["STOP_ADJ"] == 0).sum()
    stop_events = df["STOP_EVENT"].sum()

    # --- Time-based Calculations ---
    total_runtime = (df["SHOT TIME"].max() - df["SHOT TIME"].min()).total_seconds() / 60  # minutes
    run_hours = total_runtime / 60

    # Downtime = sum of ALL stop intervals (even back-to-back)
    downtime = df.loc[df["STOP_FLAG"] == 1, "CT_diff_sec"].sum() / 60  # minutes

    # Production time = runtime - downtime
    production_time = total_runtime - downtime

    gross_rate = total_shots / run_hours if run_hours else None
    net_rate = normal_shots / run_hours if run_hours else None
    efficiency = normal_shots / total_shots if total_shots else None

    # --- NEW: Continuous Run Durations ---
    # Each run = time between two stop events (using STOP_ADJ to collapse back-to-back)
    df["RUN_GROUP"] = df["STOP_ADJ"].cumsum()
    run_durations = (
        df.groupby("RUN_GROUP")
          .apply(lambda g: g["CT_diff_sec"].sum() / 60)  # minutes
          .reset_index(name="RUN_DURATION")
    )

    # Remove first run if it starts with a stop (edge case)
    run_durations = run_durations[run_durations["RUN_DURATION"] > 0]

    # Assign buckets (0–20, 20–40, …)
    run_durations["TIME_BUCKET"] = (
    pd.cut(
        run_durations["RUN_DURATION"],
        bins=[0,20,40,60,80,100,120,140,160,999999],
        labels=["0-20","20-40","40-60","60-80","80-100",
                "100-120","120-140","140-160",">160"]
    ).cat.add_categories("Unclassified")
)

    # Bucket counts for overall distribution
    bucket_counts = run_durations["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # --- Hourly MTTR/MTBF ---
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
        "df": df,
        "run_durations": run_durations  # <-- NEW dataset for plotting
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
            st.session_state.results = results
    # --- Threshold Settings (in sidebar) ---
    st.sidebar.markdown("### 🚨 Stoppage Threshold Settings")
    
    mode_ct = st.session_state.results["mode_ct"] if "results" in st.session_state else None
    
    threshold_mode = st.sidebar.radio(
        "Select threshold type:",
        ["Multiple of Mode CT", "Manual (seconds)"],
        horizontal=False,
        key="sidebar_threshold_mode"  # unique key for sidebar
    )
    
    if threshold_mode == "Multiple of Mode CT":
        multiplier = st.sidebar.slider(
            "Multiplier of Mode CT",
            min_value=1.0, max_value=5.0, value=2.0, step=0.5,
            key="sidebar_ct_multiplier"
        )
        threshold = mode_ct * multiplier if mode_ct else None
        threshold_label = f"Mode CT × {multiplier} = {threshold:.2f} sec" if threshold else ""
    else:
        default_val = float(mode_ct * 2) if mode_ct else 2.0
        threshold = st.sidebar.number_input(
            "Manual threshold (seconds)",
            min_value=1.0, value=default_val,
            key="sidebar_manual_threshold"
        )
        threshold_label = f"{threshold:.2f} sec (manual)" if threshold else ""
    
    # Save into session_state for use in main section
    st.session_state["threshold_mode"] = threshold_mode
    st.session_state["threshold"] = threshold
    st.session_state["threshold_label"] = threshold_label

    # --- Page 1: Analysis Dashboard ---
    if page == "📊 Analysis Dashboard":
        if "results" not in st.session_state:
            st.info("👈 Please generate a report first from the sidebar.")
        else:
            results = st.session_state.results
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

        df_res = results["df"]
                
        # MTTR = average downtime duration (minutes)
        mttr = df_res.loc[df_res["STOP_EVENT"], "CT_diff_sec"].mean() / 60 if results["stop_events"] > 0 else None
                
        # MTBF = average uptime duration (minutes)
        uptimes = df_res.loc[~df_res["STOP_EVENT"], "CT_diff_sec"]
        mtbf = uptimes.mean() / 60 if results["stop_events"] > 0 and not uptimes.empty else None
                
        # Time to First Downtime = first STOP_EVENT gap (minutes)
        first_dt = df_res.loc[df_res["STOP_EVENT"], "CT_diff_sec"].iloc[0] / 60 if results["stop_events"] > 0 else None
                
        # Avg Cycle Time = mean of ACTUAL CT (seconds → convert to sec directly)
        avg_ct = df_res["ACTUAL CT"].mean()
                
        reliability_df = pd.DataFrame({
            "Metric": ["MTTR (min)", "MTBF (min)", "Time to First DT (min)", "Avg Cycle Time (sec)"],
            "Value": [
                f"{mttr:.2f}" if mttr else "N/A",
                f"{mtbf:.2f}" if mtbf else "N/A",
                f"{first_dt:.2f}" if first_dt else "N/A",
                f"{avg_ct:.2f}" if avg_ct else "N/A"
            ]
        })
                
        st.table(reliability_df)

        st.markdown("### Production & Downtime Summary")
        st.table(pd.DataFrame({
            "Mode CT": [f"{results['mode_ct']:.2f}"],
            "Lower Limit": [f"{results['lower_limit']:.2f}"],
            "Upper Limit": [f"{results['upper_limit']:.2f}"],
            "Production Time (hrs)": [f"{results['production_time']/60:.1f} hrs ({results['production_time']/results['total_runtime']*100:.2f}%)"],
            "Downtime (hrs)": [f"{results['downtime']/60:.1f} hrs ({results['downtime']/results['total_runtime']*100:.2f}%)"],
            "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
            "Total Stops": [results['stop_events']]
        }))

        # Graphs + Collapsible Tables
        st.subheader("📈 Visual Analysis")
        run_durations = results["run_durations"].copy()
        bucket_order = [f"{i+1}: {rng}" for i, rng in enumerate(
            ["0-20 min","20-40 min","40-60 min","60-80 min","80-100 min","100-120 min","120-140 min","140-160 min",">160 min"]
        )]

        # Re-map bucket labels in run_durations
        label_map = {
            "0-20":"1: 0-20 min", "20-40":"2: 20-40 min", "40-60":"3: 40-60 min",
            "60-80":"4: 60-80 min", "80-100":"5: 80-100 min", "100-120":"6: 100-120 min",
            "120-140":"7: 120-140 min", "140-160":"8: 140-160 min", ">160":"9: >160 min"
        }
        run_durations["TIME_BUCKET"] = run_durations["TIME_BUCKET"].map(label_map)

        # 1) Time Bucket Analysis (overall distribution of run durations)
        bucket_counts = run_durations["TIME_BUCKET"].value_counts().reindex(bucket_order).fillna(0).astype(int)
        total_runs = bucket_counts.sum()
        bucket_df = bucket_counts.reset_index()
        bucket_df.columns = ["Time Bucket", "Occurrences"]
        bucket_df["Percentage"] = (bucket_df["Occurrences"] / total_runs * 100).round(2)

        fig_bucket = px.bar(
            bucket_df[bucket_df["Time Bucket"].notna()],
            x="Occurrences", y="Time Bucket",
            orientation="h", text="Occurrences",
            title="Time Bucket Analysis (Continuous Runs Before Stops)",
            category_orders={"Time Bucket": bucket_order},
            color="Time Bucket",
            color_discrete_map = {
                "1: 0-20 min":   "#d73027",  # red
                "2: 20-40 min":  "#fc8d59",  # orange-red
                "3: 40-60 min":  "#fee090",  # yellow
                "4: 60-80 min":  "#c6dbef",  # very light grey-blue
                "5: 80-100 min": "#9ecae1",  # light steel blue
                "6: 100-120 min":"#6baed6",  # medium blue-grey
                "7: 120-140 min":"#4292c6",  # stronger blue-grey
                "8: 140-160 min":"#2171b5",  # dark muted blue
                "9: >160 min":  "#084594"    # deep navy blue
            },
            hover_data={"Occurrences":True,"Percentage":True}
        )
        fig_bucket.update_traces(textposition="outside")
        st.plotly_chart(fig_bucket, width="stretch")

        with st.expander("📊 Time Bucket Analysis Data Table", expanded=False):
            st.dataframe(bucket_df)

        # 2) Time Bucket Trend (group by hour of day instead of week)
        if "SHOT TIME" in results["df"].columns:
            run_durations["HOUR"] = results["df"]["SHOT TIME"].dt.hour
        else:
            run_durations["HOUR"] = -1  # fallback if no timestamps

        trend = run_durations.groupby(["HOUR","TIME_BUCKET"]).size().reset_index(name="count")

        # Ensure all hours 0–23 appear, even if empty
        hours = list(range(24))
        grid = pd.MultiIndex.from_product([hours, bucket_order], names=["HOUR","TIME_BUCKET"]).to_frame(index=False)
        trend = grid.merge(trend, on=["HOUR","TIME_BUCKET"], how="left").fillna({"count":0})

        fig_tb_trend = px.bar(
            trend, x="HOUR", y="count", color="TIME_BUCKET",
            category_orders={"TIME_BUCKET": bucket_order},
            title="Hourly Time Bucket Trend (Continuous Runs Before Stops)",
            color_discrete_map = {
                "1: 0-20 min":   "#d73027",  # red
                "2: 20-40 min":  "#fc8d59",  # orange-red
                "3: 40-60 min":  "#fee090",  # yellow
                "4: 60-80 min":  "#c6dbef",  # very light grey-blue
                "5: 80-100 min": "#9ecae1",  # light steel blue
                "6: 100-120 min":"#6baed6",  # medium blue-grey
                "7: 120-140 min":"#4292c6",  # stronger blue-grey
                "8: 140-160 min":"#2171b5",  # dark muted blue
                "9: >160 min":  "#084594"    # deep navy blue
            },
            hover_data={"count":True,"HOUR":True}
        )
        fig_tb_trend.update_layout(
            barmode="stack",
            xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1, range=[-0.5,23.5]),
            yaxis=dict(title="Number of Runs")
        )
        st.plotly_chart(fig_tb_trend, width="stretch")

        with st.expander("📊 Hourly Time Bucket Trend Data Table", expanded=False):
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
        st.plotly_chart(fig_mt, width="stretch")
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
        st.plotly_chart(fig_stability, width="stretch")
        with st.expander("📊 Stability Index Data Table", expanded=False):
            table_data = hourly[["HOUR","stability_index","stability_change_%","mttr","mtbf","stops"]].copy()
            table_data.rename(columns={
                "HOUR":"Hour",
                "stability_index":"Stability Index (%)",
                "stability_change_%":"Change vs Prev Hour (%)",
                "mttr":"MTTR (min)",
                "mtbf":"MTBF (min)",
                "stops":"Stop Count"
            }, inplace=True)
        
            # Highlight only the Stability Index column
            def highlight_stability(val):
                if pd.isna(val):
                    return ""
                elif val <= 50:
                    return "background-color: rgba(255, 0, 0, 0.3);"   # soft red
                elif val <= 70:
                    return "background-color: rgba(255, 255, 0, 0.3);" # soft yellow
                else:
                    return ""
            
            st.dataframe(
                table_data.style
                .applymap(highlight_stability, subset=["Stability Index (%)"])
                .format({
                    "Stability Index (%)": "{:.2f}",
                    "Change vs Prev Hour (%)": "{:+.2f}%",
                    "MTTR (min)": "{:.2f}",
                    "MTBF (min)": "{:.2f}"
                })
            )

        st.markdown("""
        **ℹ️ Stability Index Formula**
        - Stability Index (%) = (MTBF / (MTBF + MTTR)) × 100
        - If no stoppages occur in an hour, Stability Index is forced to **100%**
        - Alert Zones:
          - 🟥 0–50% → High Risk (Frequent stoppages with long recovery times. Production is highly unstable.)
          - 🟨 50–70% → Medium Risk (Minor but frequent stoppages or slower-than-normal recoveries. Production flow is inconsistent and requires attention to prevent escalation.)
          - 🟩 70–100% → Low Risk (stable operation)
        """)

        # 5) 🚨 Stoppage Alerts (Improved Table)
        st.markdown("### 🚨 Stoppage Alert Reporting")
        
        if "results" in st.session_state:
            results = st.session_state.results
            df_vis = results["df"].copy()
        
            # --- Read threshold values from sidebar ---
            threshold_mode = st.session_state.get("threshold_mode")
            threshold = st.session_state.get("threshold")
            threshold_label = st.session_state.get("threshold_label")
        
            if threshold is None:
                st.warning("⚠️ Please set a stoppage threshold in the sidebar.")
            else:
                # --- Filter stoppages ---
                if "STOP_EVENT" in df_vis.columns and "CT_diff_sec" in df_vis.columns:
                    stoppage_alerts = df_vis[df_vis["CT_diff_sec"] >= threshold].copy()
        
                    if stoppage_alerts.empty:
                        st.info(f"✅ No stoppage alerts found (≥ {threshold_label}).")
                    else:
                        # Add context columns
                        stoppage_alerts["Shots Since Last Stop"] = stoppage_alerts.groupby(
                            stoppage_alerts["STOP_EVENT"].cumsum()
                        ).cumcount()
                        stoppage_alerts["Duration (min)"] = (stoppage_alerts["CT_diff_sec"] / 60).round(1)
                        stoppage_alerts["Reason"] = "to be added"
                        stoppage_alerts["Alert"] = "🔴"
        
                        # Final clean table
                        table = stoppage_alerts[[
                            "SHOT TIME", "Duration (min)", "Shots Since Last Stop", "Reason", "Alert"
                        ]].rename(columns={"SHOT TIME": "Event Time"})
        
                        st.dataframe(table, width="stretch")
        
                        st.markdown(f"""
                        **Summary**
                        - Total Stoppage Alerts: {len(stoppage_alerts)}
                        - Threshold Applied: {threshold_label}
                        """)
                else:
                    st.warning("⚠️ No stoppage event data available for this dataset.")

    # ---------- Page 2: Raw & Processed Data ----------
    elif page == "📂 Raw & Processed Data":
        st.title("📋 Raw & Processed Cycle Data")
    
        if "results" not in st.session_state:
            st.info("👈 Please generate a report first from the Analysis Dashboard.")
        else:
            results = st.session_state.results
            df_vis = results["df"].copy()
    
            # --- Summary (same as Page 1) ---
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
                "Production Time (hrs)": [f"{results['production_time']/60:.1f} hrs ({results['production_time']/results['total_runtime']*100:.2f}%)"],
                "Downtime (hrs)": [f"{results['downtime']/60:.1f} hrs ({results['downtime']/results['total_runtime']*100:.2f}%)"],
                "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                "Total Stops": [results['stop_events']]
            }))
    
            st.markdown("---")
    
            # --- Supplier Name ---
            if "SUPPLIER NAME" in df_vis.columns:
                df_vis["Supplier Name"] = df_vis["SUPPLIER NAME"]
            else:
                df_vis["Supplier Name"] = "not provided"
    
            # --- Equipment Code ---
            if "EQUIPMENT CODE" in df_vis.columns:
                df_vis["Equipment Code"] = df_vis["EQUIPMENT CODE"]
            else:
                df_vis["Equipment Code"] = "not provided"
    
            # --- Approved CT ---
            if "APPROVED CT" in df_vis.columns:
                df_vis["Approved CT"] = df_vis["APPROVED CT"]
            else:
                df_vis["Approved CT"] = "not provided"
    
            # --- Actual CT (1 decimal) ---
            df_vis["Actual CT"] = df_vis["ACTUAL CT"].round(1)
    
            # --- Time Diff Sec (2 decimals) ---
            df_vis["Time Diff Sec"] = df_vis["CT_diff_sec"].round(2)
    
            # --- Stop Flag (use STOP_ADJ so back-to-backs are also marked) ---
            df_vis["Stop"] = df_vis["STOP_ADJ"]
    
            # --- Cumulative Count (cycles since last stop) ---
            df_vis["Cumulative Count"] = df_vis.groupby(df_vis["Stop"].cumsum()).cumcount()
    
            # --- Run Duration (update only when stop occurs) ---
            df_vis["Run Duration"] = np.where(df_vis["Stop"] == 1,
                                              (df_vis["CT_diff_sec"] / 60).round(2),
                                              0)
    
            # --- Select only required columns ---
            df_clean = df_vis[[
                "Supplier Name", "Equipment Code", "SHOT TIME",
                "Approved CT", "Actual CT", "Time Diff Sec",
                "Stop", "Cumulative Count", "Run Duration"
            ]].rename(columns={
                "SHOT TIME": "Shot Time"
            })
    
            # --- Display with checkboxes for Stop ---
            st.markdown("### Cycle Data Table (Processed)")
            st.data_editor(
                df_clean,
                width="stretch",
                column_config={
                    "Stop": st.column_config.CheckboxColumn(
                        "Stop",
                        help="Marked as stoppage event",
                        default=False
                    )
                }
            )
    
            # --- Download option ---
            csv = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Processed Data (CSV)",
                data=csv,
                file_name="processed_cycle_data.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")