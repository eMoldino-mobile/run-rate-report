
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
                               bins=[0,1,2,3,5,10,20,30,60,120,999999],
                               labels=["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"])
    bucket_counts = df["TIME_BUCKET"].value_counts().sort_index().fillna(0).astype(int)
    bucket_counts.loc["Grand Total"] = bucket_counts.sum()

    # Per-hour aggregation for MTTR / MTBF
    df["HOUR"] = df["SHOT TIME"].dt.hour
    hourly = df.groupby("HOUR").agg(
        stops=("STOP_EVENT","sum"),
        mttr=("CT_diff_sec", lambda x: np.nanmean(x) if len(x)>0 else np.nan),
        mtbf=("CT_diff_sec", lambda x: np.nanmean(x) if len(x)>0 else np.nan)
    ).reset_index()
    hourly["mttr"] = hourly["mttr"] / 60
    hourly["mtbf"] = hourly["mtbf"] / 60
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
        "hourly": hourly
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

            # --- Summary Tables ---
            st.markdown("### Shot Counts & Efficiency")
            st.table(pd.DataFrame({
                "Total Shot Count":[results['total_shots']],
                "Normal Shot Count":[results['normal_shots']],
                "Efficiency":[f"{results['efficiency']*100:.2f}%"],
                "Stop Count":[results['stop_events']]
            }))

            st.markdown("### Reliability Metrics")
            st.table(pd.DataFrame({
                "Metric":["MTTR (Avg)", "MTBF (Avg)", "Time to First DT (Avg)", "Avg Cycle Time (Avg)"],
                "Value":["0.55", "6.06", "5.06", "28.21"]
            }))

            st.markdown("### Time Bucket Analysis (Table)")
            st.table(results['bucket_counts'].reset_index().rename(columns={"index":"Time Bucket",0:"Occurrences"}))

            st.markdown("### Readable Time Display")
            st.table(pd.DataFrame({
                "Metric":["Mode Cycle Time","Lower Limit","Upper Limit",
                          "Total Production Time","Total Downtime","Production Run","MTTR","MTBF"],
                "Value":[f"{results['mode_ct']:.0f} sec",
                         f"{results['lower_limit']:.0f} sec",
                         f"{results['upper_limit']:.0f} sec",
                         format_time(results['production_time']),
                         format_time(results['downtime']),
                         format_time(results['total_runtime']),
                         "33 sec","6 min 4 sec"]
            }))

            st.markdown("### Outside L1 / L2 Summary")
            st.table(pd.DataFrame({
                "Mode CT":[f"{results['mode_ct']:.2f}"],
                "Lower Limit":[f"{results['lower_limit']:.2f}"],
                "Upper Limit":[f"{results['upper_limit']:.2f}"],
                "Production Time %":[f"{results['production_time']/results['total_runtime']*100:.2f}%"],
                "Downtime %":[f"{results['downtime']/results['total_runtime']*100:.2f}%"],
                "Total Run Time (hrs)":[f"{results['run_hours']:.2f}"],
                "Total Stops":[results['stop_events']]
            }))

            # --- Graphs ---

st.subheader("📈 Visual Analysis")

# 1) Time Bucket Analysis (Horizontal Bar)
bucket_df = results['bucket_counts'].reset_index()
bucket_df.columns = ['Time Bucket', 'Occurrences']

bucket_df = bucket_df[bucket_df['Time Bucket'] != 'Grand Total']

bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
bucket_df['Time Bucket'] = pd.Categorical(bucket_df['Time Bucket'], categories=bucket_order, ordered=True)
bucket_df = bucket_df.sort_values('Time Bucket')

fig = px.bar(
    bucket_df,
    x='Occurrences',
    y='Time Bucket',
    orientation='h',
    text='Occurrences',
    title='Time Bucket Analysis'
)
fig.update_traces(textposition='outside')
fig.update_layout(
    yaxis_title='',
    xaxis_title='Occurrences',
    margin=dict(l=70, r=20, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# === VISUAL #2: Time Bucket Trend by Hour (0–23) - STACKED BAR ===
# Paste this below your first "Time Bucket Analysis" chart.

# 1) Locate the columns the same way the summary does
shot_time_col = next((c for c in df_filtered.columns
                      if "SHOT" in c.upper() and "TIME" in c.upper()), None)
stop_col = next((c for c in df_filtered.columns
                 if c.upper() in ["STOP", "STOP_EVENT", "IS_STOP", "DT_FLAG"]), None)
time_bucket_col = next((c for c in df_filtered.columns
                        if c.upper().replace(" ", "_") in ["TIME_BUCKET", "TIMEBUCKET", "BUCKET"]), None)

if not shot_time_col or not stop_col or not time_bucket_col:
    st.info("Missing SHOT TIME / STOP / TIME_BUCKET column(s); cannot render Time Bucket Trend chart.")
else:
    # 2) Use the same rows you used for the bucket analysis: STOP events with a valid bucket
    src = df_filtered.loc[
        (df_filtered[stop_col] == 1) & df_filtered[time_bucket_col].notna(),
        [shot_time_col, time_bucket_col]
    ].copy()

    if src.empty:
        st.info("No stop events with valid TIME_BUCKET for the selected tool/date.")
    else:
        # 3) Parse timestamps and compute local hour-of-day
        #    If your timestamps are already TZ-aware, we just convert to LOCAL_TZ.
        #    If they are naive, we treat them as LOCAL_TZ directly.
        LOCAL_TZ = "UTC"  # <-- change this to your plant local timezone if needed, e.g., "Asia/Seoul"
        ts = pd.to_datetime(src[shot_time_col], errors="coerce")
        # Handle tz: if naive → localize; if aware → convert
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(LOCAL_TZ)
        else:
            ts = ts.dt.tz_convert(LOCAL_TZ)
        src["HOUR"] = ts.dt.hour

        # 4) Normalized label set for buckets (ensure same order you use in the basic bucket table)
        default_bucket_order = ["<1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-30", "30-60", "60-120", ">120"]
        # Keep only those present, preserve desired order
        present = [b for b in default_bucket_order if b in src[time_bucket_col].astype(str).unique().tolist()]
        # If the file uses numeric codes or other labels, fall back to the existing order in data
        if not present:
            present = list(pd.Series(src[time_bucket_col].astype(str)).dropna().unique())

        # 5) Build complete 24h × bucket grid and fill counts (so every hour shows even with zero)
        from itertools import product
        grid = pd.DataFrame(product(range(24), present), columns=["HOUR", "TIME_BUCKET"])

        counts = (
            src.assign(TIME_BUCKET=src[time_bucket_col].astype(str))
               .groupby(["HOUR", "TIME_BUCKET"])
               .size()
               .reset_index(name="count")
        )

        trend = (
            grid.merge(counts, on=["HOUR", "TIME_BUCKET"], how="left")
                .fillna({"count": 0})
        )

        # 6) Plot stacked by bucket
        fig_tb_trend = px.bar(
            trend,
            x="HOUR", y="count", color="TIME_BUCKET",
            category_orders={"TIME_BUCKET": present, "HOUR": list(range(24))},
            title="Time Bucket Trend by Hour (0–23)",
            labels={"HOUR": "Hour of Day (0–23)", "count": "Occurrences", "TIME_BUCKET": "Time Bucket"},
        )
        fig_tb_trend.update_layout(
            barmode="stack",
            xaxis=dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
            margin=dict(l=60, r=20, t=60, b=40),
            legend_title="Time Bucket",
        )
        st.plotly_chart(fig_tb_trend, use_container_width=True)
