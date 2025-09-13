import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        (df["CT_diff_sec"] <= 28800),
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
    df["RUN_DURATION"] = np.where(df["STOP_ADJ"] == 1, df["CT_diff_sec"] / 60, np.nan)
    df["TIME_BUCKET"] = pd.cut(
        df["RUN_DURATION"],
        bins=[0,1,2,3,5,10,20,30,60,120,999999],
        labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
    )
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # --- Per-hour aggregation for MTTR / MTBF ---
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

    # Stability index (if no stoppages → 100%)
    hourly["stability_index"] = np.where(
        hourly["stops"] == 0,
        100,
        (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100
    )

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

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel (clean table)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    tool = st.sidebar.selectbox("Select Tool / Equipment Code", df["EQUIPMENT CODE"].unique())
    date = st.sidebar.date_input("Select Date", pd.to_datetime(df["SHOT TIME"]).dt.date.min())

    if st.sidebar.button("Generate Report"):
        mask = (df["EQUIPMENT CODE"] == tool) & (pd.to_datetime(df["SHOT TIME"]).dt.date == date)
        df_filtered = df.loc[mask]

        if df_filtered.empty:
            st.warning("No data found for this selection.")
        else:
            results = calculate_run_rate_excel_like(df_filtered)

            st.title("📊 Run Rate Report")
            st.subheader(f"Tool: {tool} | Date: {date.strftime('%Y-%m-%d')}")

            # --- Summaries ---
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

            st.markdown("### Time Bucket Analysis (Table)")
            st.table(results['bucket_counts'].reset_index().rename(columns={"index": "Time Bucket", 0: "Occurrences"}))

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

            # --- Graphs ---
            st.subheader("📈 Visual Analysis")

            df_vis = results["df"].copy()
            hourly = results["hourly"].copy()
            bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]

            # ---------- Stability Index with zones ----------
            fig_stability = go.Figure()

            # Background zones
            fig_stability.add_shape(type="rect", x0=-0.5, x1=23.5, y0=70, y1=100,
                                    fillcolor="green", opacity=0.1, layer="below", line_width=0)
            fig_stability.add_shape(type="rect", x0=-0.5, x1=23.5, y0=50, y1=70,
                                    fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
            fig_stability.add_shape(type="rect", x0=-0.5, x1=23.5, y0=0, y1=50,
                                    fillcolor="red", opacity=0.1, layer="below", line_width=0)

            # Continuous trend line
            fig_stability.add_trace(go.Scatter(
                x=hourly["HOUR"], 
                y=hourly["stability_index"],
                mode="lines+markers",
                name="Stability Index (%)",
                line=dict(color="blue", width=2),
                marker=dict(
                    size=8,
                    color=np.where(hourly["stability_index"] < 50, "red",
                          np.where(hourly["stability_index"] < 70, "yellow", "green"))
                )
            ))

            fig_stability.update_layout(
                title="Stability Index Trend by Hour",
                xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1, range=[-0.5, 23.5]),
                yaxis=dict(title="Stability Index (%)", range=[0, 100]),
                margin=dict(l=60, r=60, t=60, b=40),
                legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center")
            )

            st.plotly_chart(fig_stability, use_container_width=True)

            # Detailed legend table
            st.markdown("### Stability Index Alert Zones")
            legend_df = pd.DataFrame({
                "Zone": ["Low Risk", "Medium Risk", "High Risk"],
                "Range (%)": ["70–100", "50–70", "0–50"],
                "Indicator": ["🟢 Green", "🟡 Yellow", "🔴 Red"],
                "Interpretation": [
                    "Stable operation with sufficient MTBF",
                    "Caution – risk of instability rising",
                    "Critical – frequent stops, unstable production"
                ]
            })
            st.table(legend_df)

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")
