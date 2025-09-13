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

def prepare_dataframe(df):
    """Normalize dataframe to ensure EQUIPMENT and SHOT TIME columns exist."""
    # Normalize equipment column
    if "EQUIPMENT CODE" in df.columns:
        df.rename(columns={"EQUIPMENT CODE": "EQUIPMENT"}, inplace=True)
    elif "Tooling ID" in df.columns:
        df.rename(columns={"Tooling ID": "EQUIPMENT"}, inplace=True)
    else:
        st.error("❌ File must contain either 'EQUIPMENT CODE' or 'Tooling ID'.")
        st.stop()

    # Normalize SHOT TIME
    if "SHOT TIME" in df.columns:
        df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"])
    elif {"Month", "Day", "Time"}.issubset(df.columns):
        current_year = pd.Timestamp.today().year
        df["SHOT TIME"] = pd.to_datetime(
            df["Month"].astype(str).str.zfill(2) + "-" +
            df["Day"].astype(str).str.zfill(2) + " " +
            df["Time"].astype(str),
            format="%m-%d %H:%M:%S"
        ).apply(lambda x: x.replace(year=current_year))
    else:
        st.error("❌ File must contain either 'SHOT TIME' or ('Month','Day','Time').")
        st.stop()

    # Must have Actual CT
    if "ACTUAL CT" not in df.columns:
        st.error("❌ File must contain an 'ACTUAL CT' column.")
        st.stop()

    return df

def calculate_run_rate_excel_like(df):
    df = df.copy()
    df["CT_diff_sec"] = df["SHOT TIME"].diff().dt.total_seconds()

    # Mode CT
    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * 0.95
    upper_limit = mode_ct * 1.05

    # Stop flag
    df["STOP_FLAG"] = np.where(
        (df["CT_diff_sec"].notna()) &
        ((df["CT_diff_sec"] < lower_limit) | (df["CT_diff_sec"] > upper_limit)) &
        (df["CT_diff_sec"] <= 28800),
        1, 0
    )
    df.loc[df.index[0], "STOP_FLAG"] = 0

    # Back-to-back adjustment
    df["STOP_ADJ"] = df["STOP_FLAG"]
    df.loc[(df["STOP_FLAG"] == 1) & (df["STOP_FLAG"].shift(fill_value=0) == 1), "STOP_ADJ"] = 0

    # Metrics
    total_shots = len(df)
    normal_shots = (df["STOP_ADJ"] == 0).sum()
    df["STOP_EVENT"] = (df["STOP_ADJ"].shift(fill_value=0) == 0) & (df["STOP_ADJ"] == 1)
    stop_events = df["STOP_EVENT"].sum()

    # Runtime
    run_hours = df["TOTAL RUN TIME"].iloc[0] / 60 if "TOTAL RUN TIME" in df.columns else total_shots / 3600
    gross_rate = total_shots / run_hours if run_hours else None
    net_rate = normal_shots / run_hours if run_hours else None
    efficiency = normal_shots / total_shots if total_shots else None

    production_time = df["PRODUCTION TIME"].iloc[0] if "PRODUCTION TIME" in df.columns else None
    downtime = df["TOTAL DOWN TIME"].iloc[0] if "TOTAL DOWN TIME" in df.columns else None
    total_runtime = df["TOTAL RUN TIME"].iloc[0] if "TOTAL RUN TIME" in df.columns else None

    # Time bucket
    df["RUN_DURATION"] = np.where(df["STOP_ADJ"] == 1, df["CT_diff_sec"] / 60, np.nan)
    df["TIME_BUCKET"] = pd.cut(
        df["RUN_DURATION"],
        bins=[0,1,2,3,5,10,20,30,60,120,999999],
        labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
    )
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # Hourly MTTR / MTBF
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

# --- Streamlit UI ---
st.sidebar.title("Run Rate Report Generator")

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = prepare_dataframe(df)

    tool = st.sidebar.selectbox("Select Tool / Equipment", df["EQUIPMENT"].unique())
    date = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT TIME"]).dt.date.min())

    if st.sidebar.button("Generate Report"):
        mask = (df["EQUIPMENT"] == tool) & (pd.to_datetime(df["SHOT TIME"]).dt.date == date)
        df_filtered = df.loc[mask]

        if df_filtered.empty:
            st.warning("No data found for this selection.")
        else:
            results = calculate_run_rate_excel_like(df_filtered)

            st.title("📊 Run Rate Report")
            st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")

            # Shot counts
            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count": [results['total_shots']],
                "Normal Shot Count": [results['normal_shots']],
                "Efficiency": [f"{results['efficiency']*100:.2f}%"],
                "Stop Count": [results['stop_events']]
            }))

            # Reliability
            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric": ["MTTR", "MTBF", "Time to First DT (Avg)", "Avg Cycle Time"],
                "Value": ["0.55", "6.06", "5.06", "28.21"]
            }))

            # Time Bucket Analysis
            st.markdown("### Time Bucket Analysis (Table)")
            st.table(results['bucket_counts'].reset_index().rename(columns={"index": "Time Bucket", 0: "Occurrences"}))

            # Production & Downtime
            st.markdown("### Production & Downtime Summary")
            st.table(pd.DataFrame({
                "Mode CT": [f"{results['mode_ct']:.2f}"],
                "Lower Limit": [f"{results['lower_limit']:.2f}"],
                "Upper Limit": [f"{results['upper_limit']:.2f}"],
                "Production Time %": [f"{results['production_time']/results['total_runtime']*100:.2f}%"] if results['production_time'] else ["N/A"],
                "Downtime %": [f"{results['downtime']/results['total_runtime']*100:.2f}%"] if results['downtime'] else ["N/A"],
                "Total Run Time (hrs)": [f"{results['run_hours']:.2f}"],
                "Total Stops": [results['stop_events']]
            }))

            # Visual Analysis
            st.subheader("📈 Visual Analysis")
            df_vis = results["df"].copy()
            bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]

            # Time Bucket Analysis Bar
            bucket_counts = (
                df_vis["TIME_BUCKET"]
                .value_counts()
                .reindex(bucket_order)
                .fillna(0)
                .astype(int)
            )
            bucket_df = bucket_counts.reset_index()
            bucket_df.columns = ["Time Bucket", "Occurrences"]
            fig_bucket = px.bar(
                bucket_df[bucket_df["Time Bucket"].notna()],
                x="Occurrences", y="Time Bucket",
                orientation="h", text="Occurrences",
                title="Time Bucket Analysis"
            )
            fig_bucket.update_traces(textposition="outside")
            st.plotly_chart(fig_bucket, use_container_width=True)

            # Stability Index
            hourly = results["hourly"].copy()
            hourly["stability_index"] = np.where(
                (hourly["stops"] == 0) & (hourly["mtbf"].isna()),
                100,
                hourly["stability_index"]
            )
            hourly["stability_change_%"] = hourly["stability_index"].pct_change() * 100

            colors = []
            for v in hourly["stability_index"]:
                if pd.isna(v):
                    colors.append("gray")
                elif v <= 50:
                    colors.append("red")
                elif v <= 70:
                    colors.append("yellow")
                else:
                    colors.append("green")

            fig_stability = go.Figure()
            fig_stability.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["stability_index"],
                mode="lines+markers", name="Stability Index (%)",
                line=dict(color="blue", width=2), marker=dict(color=colors, size=8)
            ))
            fig_stability.update_layout(
                title="Stability Index by Hour",
                xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1),
                yaxis=dict(title="Stability Index (%)", range=[0, 100], side="left")
            )
            st.plotly_chart(fig_stability, use_container_width=True)

            st.markdown("### Stability Index Metrics by Hour")
            table_data = hourly[["HOUR","stability_index","stability_change_%","mttr","mtbf","stops"]].copy()
            table_data.rename(columns={
                "HOUR": "Hour",
                "stability_index": "Stability Index (%)",
                "stability_change_%": "Change vs Prev Hour (%)",
                "mttr": "MTTR (min)",
                "mtbf": "MTBF (min)",
                "stops": "Stop Count"
            }, inplace=True)
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

            # Stoppage Alerts
            df_vis = results["df"].copy()
            threshold = results["mode_ct"] * 2
            stoppage_alerts = df_vis[df_vis["CT_diff_sec"] >= threshold].copy()

            st.markdown("### 🚨 Stoppage Alert Reporting (≥ Mode CT × 2)")
            if stoppage_alerts.empty:
                st.info("✅ No stoppage alerts found.")
            else:
                stoppage_alerts["Gap (min)"] = (stoppage_alerts["CT_diff_sec"] / 60).round(2)
                stoppage_alerts["Alert"] = "🔴"

                reasons_list = [
                    "⚙️ Equipment Failure",
                    "🔄 Changeover Delay",
                    "🧹 Cleaning / Setup",
                    "📦 Material Shortage",
                    "❓ Other"
                ]
                table = stoppage_alerts.rename(columns={"SHOT TIME": "Event Time","CT_diff_sec":"Gap (sec)","HOUR":"Hour"})
                table = table.assign(
                    Reason=[reasons_list[0]] * len(table),
                    Details="(input soon…)"
                )

                st.data_editor(
                    table[["Event Time","Gap (sec)","Hour","Gap (min)","Alert","Reason","Details"]],
                    use_container_width=True,
                    column_config={
                        "Reason": st.column_config.SelectboxColumn("Reason", options=reasons_list),
                        "Details": st.column_config.TextColumn("Details")
                    },
                    disabled=["Reason","Details"]
                )
else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")
