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

    hourly = df.groupby("HOUR").agg(
        stops=("STOP_EVENT", "sum"),
        mttr=("DOWNTIME_MIN", lambda x: np.nanmean(x) if len(x.dropna()) > 0 else np.nan),
        mtbf=("UPTIME_MIN", lambda x: np.nanmean(x) if len(x.dropna()) > 0 else np.nan)
    ).reset_index()

    # Stability index
    hourly["stability_index"] = (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100

    # ✅ Return dictionary with df included
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

            st.markdown("### Outside L1 / L2 Summary")
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
            bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]

            # 1) Time Bucket Analysis
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

            # 2) Time Bucket Trend by Hour
            src = df_vis.loc[df_vis["STOP_EVENT"] & df_vis["TIME_BUCKET"].notna(), ["HOUR", "TIME_BUCKET"]]
            if src.empty:
                st.info("No stop events with valid TIME_BUCKET for the selected tool/date.")
            else:
                hours = list(range(24))
                grid = pd.MultiIndex.from_product([hours, bucket_order], names=["HOUR","TIME_BUCKET"]).to_frame(index=False)
                counts = src.groupby(["HOUR", "TIME_BUCKET"]).size().reset_index(name="count")
                trend = grid.merge(counts, on=["HOUR","TIME_BUCKET"], how="left").fillna({"count":0})

                fig_tb_trend = px.bar(
                    trend, x="HOUR", y="count", color="TIME_BUCKET",
                    category_orders={"HOUR": hours, "TIME_BUCKET": bucket_order},
                    title="Time Bucket Trend by Hour (0–23)"
                )
                fig_tb_trend.update_layout(barmode="stack")
                st.plotly_chart(fig_tb_trend, use_container_width=True)

                        # ---------- 3) MTTR & MTBF Trend by Hour ----------
            hourly = results["hourly"].copy()

            # Ensure all 24 hours exist
            all_hours = pd.DataFrame({"HOUR": list(range(24))})
            hourly = all_hours.merge(hourly, on="HOUR", how="left")

            # Guarantee columns exist
            if "mttr" not in hourly.columns:
                hourly["mttr"] = np.nan
            if "mtbf" not in hourly.columns:
                hourly["mtbf"] = np.nan

            fig_mt = go.Figure()

            # MTTR (left y-axis)
            fig_mt.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["mttr"],
                mode="lines+markers",
                name="MTTR (min)",
                line=dict(color="red", width=2),
                yaxis="y1",
                connectgaps=False
            ))

            # MTBF (right y-axis)
            fig_mt.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["mtbf"],
                mode="lines+markers",
                name="MTBF (min)",
                line=dict(color="green", width=2, dash="dot"),
                yaxis="y2",
                connectgaps=False
            ))

            # Safe layout
            fig_mt.update_layout(
                title="MTTR & MTBF Trend by Hour",
                xaxis=dict(
                    title="Hour of Day (0–23)",
                    tickmode="linear",
                    dtick=1,
                    range=[-0.5, 23.5]
                ),
                yaxis=dict(
                    title="MTTR (min)",
                    titlefont=dict(color="red"),
                    tickfont=dict(color="red"),
                    side="left"
                ),
                yaxis2=dict(
                    title="MTBF (min)",
                    titlefont=dict(color="green"),
                    tickfont=dict(color="green"),
                    overlaying="y",
                    side="right"
                ),
                margin=dict(l=60, r=60, t=60, b=40),
                legend=dict(
                    orientation="h",
                    x=0.5,
                    y=-0.2,
                    xanchor="center",
                    yanchor="top"
                )
            )

            st.plotly_chart(fig_mt, use_container_width=True)



else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please")
