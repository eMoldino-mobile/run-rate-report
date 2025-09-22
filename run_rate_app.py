import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import warnings
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide")

# --- Helper Functions ---
def build_20_min_bins(max_minutes: float):
    """
    Return (edges, labels_no_prefix, labels_with_prefix)
    for 20-min bins up to ceil(max/20)*20.
    """
    base_bucket_colors = [
        "#d73027", "#fc8d59", "#fee090", "#c6dbef", "#9ecae1",
        "#6baed6", "#4292c6", "#2171b5", "#084594"
    ]
    if pd.isna(max_minutes) or max_minutes <= 0:
        edges = [0, 20]
    else:
        upper = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper + 20, 20))
    
    labels_np = [f"{edges[i]}-{edges[i+1]} min" for i in range(len(edges)-1)]
    labels_wp = [f"{i+1}: {labels_np[i]}" for i in range(len(labels_np))]
    
    return edges, labels_np, labels_wp, base_bucket_colors

def lighten_hex(hex_color, factor=0.2):
    """Lighten a hex color by blending it with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def make_bucket_color_map(active_labels, all_labels_with_prefix, base_colors):
    """Assigns colors to active time buckets based on their order in the full list."""
    full_label_to_index = {lbl: i for i, lbl in enumerate(all_labels_with_prefix)}
    
    color_map = {}
    for label in active_labels:
        if label in full_label_to_index:
            idx = full_label_to_index[label]
            if idx < len(base_colors):
                color_map[label] = base_colors[idx]
            else:
                base_color = base_colors[idx % len(base_colors)]
                factor = 0.2 * ((idx // len(base_colors)) + 1)
                factor = min(factor, 0.8)
                color_map[label] = lighten_hex(base_color, factor)
    return color_map

@st.cache_data
def calculate_run_rate_metrics(df, tolerance: float):
    """Calculates all run rate metrics from a dataframe."""
    df = df.copy()
    
    if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
        df["shot_time"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" +
            df["DAY"].astype(str) + " " + df["TIME"].astype(str),
            errors="coerce"
        )
    elif "SHOT TIME" in df.columns:
        df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
    else:
        return None

    df["ct_diff_raw"] = df["shot_time"].diff().dt.total_seconds()
    df["ct_diff_sec"] = df["ACTUAL CT"].shift()
    
    mask_maxed = df["ct_diff_sec"] == 999.9
    df.loc[mask_maxed, "ct_diff_sec"] = df.loc[mask_maxed, "ct_diff_raw"]
    df.loc[df.index[0], "ct_diff_sec"] = df.loc[df.index[0], "ct_diff_raw"]

    if "ACTUAL CT" not in df.columns: return None

    mode_ct = df["ACTUAL CT"].mode().iloc[0]
    lower_limit = mode_ct * (1 - tolerance)
    upper_limit = mode_ct * (1 + tolerance)

    df["stop_flag"] = np.where(
        (df["ct_diff_sec"].notna()) &
        ((df["ct_diff_sec"] < lower_limit) | (df["ct_diff_sec"] > upper_limit)) &
        (df["ct_diff_sec"] <= 28800), 1, 0
    )
    df.loc[df.index[0], "stop_flag"] = 0

    df["stop_adj"] = df["stop_flag"]
    df.loc[(df["stop_flag"] == 1) & (df["stop_flag"].shift(fill_value=0) == 1), "stop_adj"] = 0
    df["stop_event"] = (df["stop_adj"].shift(fill_value=0) == 0) & (df["stop_adj"] == 1)

    total_shots = len(df)
    normal_shots = ((df["ct_diff_sec"] >= lower_limit) & (df["ct_diff_sec"] <= upper_limit)).sum()
    stop_events = df["stop_event"].sum()
    
    efficiency = normal_shots / total_shots if total_shots > 0 else 0

    total_runtime = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() / 60
    run_hours = total_runtime / 60
    downtime = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum() / 60
    production_time = total_runtime - downtime
    
    if stop_events > 0:
        downtime_events = df.loc[df["stop_event"], "ct_diff_sec"] / 60
        mttr = downtime_events.mean()
        total_uptime = df.loc[df["stop_event"] == 0, "ct_diff_sec"].sum() / 60
        mtbf = total_uptime / stop_events
    else:
        mttr, mtbf = np.nan, np.nan
        total_uptime = total_runtime

    df["run_group"] = df["stop_adj"].cumsum()
    run_durations = df.groupby("run_group").apply(lambda g: g["ct_diff_sec"].sum() / 60).reset_index(name="run_duration")
    run_end_times = df.groupby("run_group")["shot_time"].max().reset_index(name="run_end")
    run_durations = run_durations.merge(run_end_times, on="run_group", how="left")
    
    run_durations = run_durations[run_durations["run_duration"] > 0]
    if df["stop_adj"].iloc[-1] == 0:
        last_group = df["run_group"].iloc[-1]
        run_durations = run_durations[run_durations["run_group"] != last_group]
    
    max_minutes = min(run_durations["run_duration"].max() if not run_durations.empty else 0, 240)
    edges, labels_np, labels_wp, base_colors = build_20_min_bins(max_minutes)
    
    run_durations["time_bucket_raw"] = pd.cut(run_durations["run_duration"], bins=edges, right=False, labels=labels_np)
    map_np_to_wp = {np_lbl: wp_lbl for np_lbl, wp_lbl in zip(labels_np, labels_wp)}
    run_durations["time_bucket"] = run_durations["time_bucket_raw"].map(map_np_to_wp)
    
    active_buckets = run_durations["time_bucket"].dropna().unique().tolist()
    bucket_color_map = make_bucket_color_map(active_buckets, labels_wp, base_colors)

    df["hour"] = df["shot_time"].dt.hour
    df["downtime_min"] = np.where(df["stop_event"], df["ct_diff_sec"] / 60, np.nan)
    
    hourly = df.groupby("hour").apply(lambda g: pd.Series({
        "stops": g["stop_event"].sum(),
        "mttr": np.nanmean(g["downtime_min"]),
        "mtbf": np.nanmean(g.loc[g["stop_event"] == 0, "ct_diff_sec"] / 60) if g["stop_event"].sum() > 0 else np.nan
    })).reset_index()
    
    hourly["stability_index"] = (hourly["mtbf"] / (hourly["mtbf"] + hourly["mttr"])) * 100
    hourly.loc[hourly["stops"] == 0, "stability_index"] = 100.0

    return {
        "df": df,
        "mode_ct": mode_ct,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "total_shots": total_shots,
        "normal_shots": normal_shots,
        "bad_shots": total_shots - normal_shots,
        "efficiency": efficiency,
        "stop_events": stop_events,
        "run_hours": run_hours,
        "production_time": production_time,
        "downtime": downtime,
        "total_runtime": total_runtime,
        "run_durations": run_durations,
        "bucket_order": labels_wp,
        "bucket_color_map": bucket_color_map,
        "hourly": hourly,
        "mttr": mttr,
        "mtbf": mtbf
    }

@st.cache_data
def export_to_excel(df, results):
    """Generates and returns an Excel file from dataframes and results."""
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Dashboard"
    
    summary_data = [
        ["Total Shot Count", results.get("total_shots", 0)],
        ["Normal Shot Count", results.get("normal_shots", 0)],
        ["Efficiency", f"{results.get('efficiency', 0) * 100:.2f}%"],
        ["Stop Count", results.get("stop_events", 0)],
        ["Mode CT", f"{results.get('mode_ct', 0):.2f}"],
        ["Lower Limit", f"{results.get('lower_limit', 0):.2f}"],
        ["Upper Limit", f"{results.get('upper_limit', 0):.2f}"]
    ]
    for row in summary_data: ws1.append(row)

    ws2 = wb.create_sheet("Processed Data")
    for r in dataframe_to_rows(df, index=False, header=True): ws2.append(r)

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

@st.cache_data
def display_summary_table(results):
    """Displays a consistent summary table."""
    st.markdown("### Shot Counts & Efficiency")
    table_data = {
        "Total Shots": [results.get("total_shots", 0)],
        "Normal Shots": [results.get("normal_shots", 0)],
        "Bad Shots": [results.get("bad_shots", 0)],
        "Efficiency": [f"{results.get('efficiency', 0) * 100:.2f}%"],
        "Stop Events": [results.get("stop_events", 0)],
    }
    st.table(pd.DataFrame(table_data))
    
    st.markdown("### Reliability & Runtime Metrics")
    runtime_data = {
        "Mode CT (sec)": [f"{results.get('mode_ct', 0):.2f}"],
        "Lower Limit (sec)": [f"{results.get('lower_limit', 0):.2f}"],
        "Upper Limit (sec)": [f"{results.get('upper_limit', 0):.2f}"],
        "Production Time (hrs)": [f"{results.get('production_time', 0)/60:.1f} hrs"],
        "Downtime (hrs)": [f"{results.get('downtime', 0)/60:.1f} hrs"],
        "Total Runtime (hrs)": [f"{results.get('run_hours', 0):.2f}"],
        "MTTR (min)": [f"{results.get('mttr', np.nan):.2f}" if pd.notna(results.get('mttr')) else "N/A"],
        "MTBF (min)": [f"{results.get('mtbf', np.nan):.2f}" if pd.notna(results.get('mtbf')) else "N/A"]
    }
    st.table(pd.DataFrame(runtime_data))

@st.cache_data
def plot_time_bucket_analysis(run_durations, bucket_order, bucket_color_map, title_suffix=""):
    """Generates and displays the time bucket bar chart."""
    bucket_counts = run_durations["time_bucket"].value_counts().reindex(bucket_order).fillna(0).astype(int)
    total_runs = bucket_counts.sum()
    bucket_df = bucket_counts.reset_index()
    bucket_df.columns = ["Time Bucket", "Occurrences"]
    bucket_df["Percentage"] = np.where(total_runs > 0, (bucket_df["Occurrences"] / total_runs * 100).round(2), 0.0)
    
    fig = px.bar(
        bucket_df[bucket_df["Time Bucket"].notna()],
        x="Occurrences", y="Time Bucket",
        orientation="h", text="Occurrences",
        title=f"Time Bucket Analysis (Continuous Runs Before Stops){title_suffix}",
        category_orders={"Time Bucket": bucket_order},
        color="Time Bucket",
        color_discrete_map=bucket_color_map,
        hover_data={"Occurrences": True, "Percentage": True}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(legend=dict(orientation="h", x=0.5, y=-0.2, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📊 Time Bucket Analysis Data Table", expanded=False):
        st.dataframe(bucket_df)
    
@st.cache_data
def plot_mt_trend(df, time_col, mttr_col, mtbf_col, title_suffix=""):
    """Generates and displays the MTTR & MTBF trend chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[mttr_col], mode="lines+markers",
                             name="MTTR (min)", line=dict(color="red", width=2), yaxis="y"))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[mtbf_col], mode="lines+markers",
                             name="MTBF (min)", line=dict(color="green", width=2, dash="dot"), yaxis="y2"))
    fig.update_layout(
        title=f"MTTR & MTBF Trend{title_suffix}",
        xaxis=dict(title=time_col.replace('_', ' ').title(), tickmode="linear", dtick=1),
        yaxis=dict(title="MTTR (min)", tickfont=dict(color="red"), side="left"),
        yaxis2=dict(title="MTBF (min)", tickfont=dict(color="green"), overlaying="y", side="right"),
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📊 MTTR & MTBF Data Table", expanded=False):
        st.dataframe(df)

@st.cache_data
def plot_stability_trend(df, time_col, stability_col, title_suffix=""):
    """Generates and displays the Stability Index trend chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[stability_col], mode="lines+markers",
                             name="Stability Index (%)", line=dict(color="blue", width=2),
                             marker=dict(color=["red" if pd.notna(v) and v <= 50 else "yellow" if pd.notna(v) and v <= 70 else "green" for v in df[stability_col]], size=8)))
    for y0, y1, c in [(0, 50, "red"), (50, 70, "yellow"), (70, 100, "green")]:
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=c, opacity=0.1, line_width=0, layer="below")
    fig.update_layout(
        title=f"Stability Index Trend{title_suffix}",
        xaxis=dict(title=time_col.replace('_', ' ').title(), tickmode="linear", dtick=1),
        yaxis=dict(title="Stability Index (%)", range=[0, 100]),
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📊 Stability Index Data Table", expanded=False):
        st.dataframe(df)

@st.cache_data
def calculate_daily_metrics(df, tolerance):
    """Calculates key metrics for a single day based on that day's data."""
    daily_results = calculate_run_rate_metrics(df, tolerance)
    return pd.Series({
        "Total Shots": daily_results["total_shots"],
        "Normal Shots": daily_results["normal_shots"],
        "Bad Shots": daily_results["bad_shots"],
        "Efficiency (%)": round(daily_results["efficiency"] * 100, 2),
        "Stops": daily_results["stop_events"],
        "MTTR (min)": round(daily_results["mttr"], 2) if pd.notna(daily_results["mttr"]) else np.nan,
        "MTBF (min)": round(daily_results["mtbf"], 2) if pd.notna(daily_results["mtbf"]) else np.nan,
        "Daily Mode CT (sec)": round(daily_results["mode_ct"], 2),
        "Stability Index (%)": round((daily_results["mtbf"] / (daily_results["mtbf"] + daily_results["mttr"])) * 100, 2)
                        if pd.notna(daily_results["mtbf"]) and pd.notna(daily_results["mttr"]) else 100.0
    })

@st.cache_data
def calculate_weekly_metrics(df, tolerance):
    """Calculates key metrics for a single week based on that week's data."""
    weekly_results = calculate_run_rate_metrics(df, tolerance)
    return pd.Series({
        "Total Shots": weekly_results["total_shots"],
        "Normal Shots": weekly_results["normal_shots"],
        "Bad Shots": weekly_results["bad_shots"],
        "Efficiency (%)": round(weekly_results["efficiency"] * 100, 2),
        "Stops": weekly_results["stop_events"],
        "MTTR (min)": round(weekly_results["mttr"], 2) if pd.notna(weekly_results["mttr"]) else np.nan,
        "MTBF (min)": round(weekly_results["mtbf"], 2) if pd.notna(weekly_results["mtbf"]) else np.nan,
        "Weekly Mode CT (sec)": round(weekly_results["mode_ct"], 2),
        "Stability Index (%)": round((weekly_results["mtbf"] / (weekly_results["mtbf"] + weekly_results["mttr"])) * 100, 2)
                        if pd.notna(weekly_results["mtbf"]) and pd.notna(weekly_results["mttr"]) else 100.0
    })


# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx"])

if uploaded_file:
    # Use cache to avoid re-reading file on every interaction
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        return df

    df_raw = load_data(uploaded_file)
    
    selection_column = "TOOLING ID" if "TOOLING ID" in df_raw.columns else "EQUIPMENT CODE" if "EQUIPMENT CODE" in df_raw.columns else None
    
    if selection_column is None:
        st.error("File must contain either 'TOOLING ID' or 'EQUIPMENT CODE'.")
        st.stop()
    
    tool = st.sidebar.selectbox("Select Tool", df_raw[selection_column].unique())
    
    # Filter data for the selected tool
    df_tool = df_raw.loc[df_raw[selection_column] == tool].copy()
    
    if df_tool.empty:
        st.warning(f"No data found for tool: {tool}")
        st.stop()

    st.sidebar.markdown("### ⚙️ Cycle Time Tolerance Settings")
    tolerance = st.sidebar.slider(
        "Tolerance Band (% of Mode CT)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Defines the ±% around Mode CT to classify normal vs. stoppage cycles"
    )
    
    # Process full dataset and cache
    results_full = calculate_run_rate_metrics(df_tool, tolerance)

    page = st.sidebar.radio(
        "Select Page",
        ["📊 Analysis Dashboard", "📂 Raw & Processed Data", "📅 Daily Analysis", "🗓️ Weekly Analysis"]
    )
    
    # --- Page 1: Analysis Dashboard ---
    if page == "📊 Analysis Dashboard":
        st.title("📊 Daily Run Rate Report")
        
        df_full = results_full["df"]
        min_date = df_full["shot_time"].min().date()
        max_date = df_full["shot_time"].max().date()
        selected_date = st.date_input("Select Date", min_value=min_date, max_value=max_date, value=max_date)

        df_day = df_full[df_full["shot_time"].dt.date == selected_date].copy()

        if df_day.empty:
            st.warning("No data found for this date.")
        else:
            daily_results = calculate_run_rate_metrics(df_day, tolerance)
            
            st.subheader(f"Tool: {tool} | Date: {selected_date.strftime('%Y-%m-%d')}")
            
            display_summary_table(daily_results)
            
            st.markdown("---")
            st.subheader("📈 Visual Analysis")
            
            plot_time_bucket_analysis(
                daily_results["run_durations"],
                daily_results["bucket_order"],
                daily_results["bucket_color_map"]
            )
            
            plot_mt_trend(daily_results["hourly"], "hour", "mttr", "mtbf", title_suffix=" by Hour")
            plot_stability_trend(daily_results["hourly"], "hour", "stability_index", title_suffix=" by Hour")

    # --- Page 2: Raw & Processed Data ---
    elif page == "📂 Raw & Processed Data":
        st.title("📋 Raw & Processed Cycle Data")
        
        display_summary_table(results_full)
        
        st.markdown("---")
        
        st.markdown("### Processed Cycle Data Table")
        df_processed = results_full["df"].copy()
        
        df_processed["Actual CT (sec)"] = df_processed["ACTUAL CT"].round(1)
        df_processed["Time Diff (sec)"] = df_processed["ct_diff_sec"].round(2)
        df_processed["Stop"] = np.where(df_processed["stop_flag"] == 1, "🔴", "")
        df_processed["Stop Event"] = np.where(df_processed["stop_event"] == 1, "🛑", "")

        df_display = df_processed[[
            "shot_time", "Actual CT (sec)", "Time Diff (sec)",
            "Stop", "Stop Event"
        ]].rename(columns={"shot_time": "Shot Time"})
        
        st.dataframe(df_display, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df_processed.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Processed Data (CSV)",
                data=csv,
                file_name=f"{tool}_processed_data.csv",
                mime="text/csv"
            )
        with col2:
            excel_buffer = export_to_excel(df_processed, results_full)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer,
                file_name=f"{tool}_run_rate_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --- Page 3: Daily Analysis ---
    elif page == "📅 Daily Analysis":
        st.title("📅 Weekly Analysis by Day")
        st.subheader(f"Tool: {tool}")

        df_full = results_full["df"]
        df_full["day"] = df_full["shot_time"].dt.date

        min_date, max_date = df_full["day"].min(), df_full["day"].max()
        week_options = pd.date_range(min_date, max_date, freq="W-MON").date
        selected_week = st.selectbox("Select Week", week_options, format_func=lambda d: f"Week of {d}")

        week_mask = (df_full["day"] >= selected_week) & (df_full["day"] < (selected_week + timedelta(days=7)))
        df_week = df_full.loc[week_mask].copy()

        if df_week.empty:
            st.warning("No data found for this week.")
        else:
            daily_summary = df_week.groupby("day").apply(
                lambda g: calculate_daily_metrics(g, tolerance)
            ).reset_index()

            st.markdown("### 📋 Weekly Summary Table (Daily Breakdown)")
            st.dataframe(daily_summary, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Weekly Trends")
            
            plot_mt_trend(daily_summary, "day", "MTTR (min)", "MTBF (min)", title_suffix=" by Day")
            plot_stability_trend(daily_summary, "day", "Stability Index (%)", title_suffix=" by Day")

            run_durations_weekly = calculate_run_rate_metrics(df_week, tolerance)["run_durations"]
            bucket_trend = run_durations_weekly.groupby(["run_end", "time_bucket"]).size().reset_index(name="count")
            bucket_trend["day"] = bucket_trend["run_end"].dt.date
            bucket_trend_day = bucket_trend.groupby(["day", "time_bucket"])["count"].sum().reset_index()
            
            fig_tb_trend = px.bar(
                bucket_trend_day, x="day", y="count", color="time_bucket",
                category_orders={"time_bucket": results_full["bucket_order"]},
                color_discrete_map=results_full["bucket_color_map"],
                title="Daily Time Bucket Trend (Selected Week)",
                hover_data={"count": True, "day": True}
            )
            fig_tb_trend.update_layout(barmode="stack", xaxis=dict(title="Day"))
            st.plotly_chart(fig_tb_trend, use_container_width=True)
            with st.expander("📊 Daily Time Bucket Trend – Data", expanded=False):
                st.dataframe(bucket_trend_day)

    # --- Page 4: Weekly Analysis ---
    elif page == "🗓️ Weekly Analysis":
        st.title("🗓️ Monthly Analysis by Week")
        st.subheader(f"Tool: {tool}")

        df_full = results_full["df"]
        df_full["week_start"] = df_full["shot_time"].dt.to_period('W-MON').dt.start_time.dt.date
        df_full["month_start"] = df_full["shot_time"].dt.to_period('M').dt.start_time.dt.date

        unique_months = df_full["month_start"].unique()
        selected_month = st.selectbox("Select Month", unique_months, format_func=lambda d: pd.to_datetime(d).strftime('%B %Y'))

        df_month = df_full.loc[df_full["month_start"] == selected_month].copy()
        
        if df_month.empty:
            st.warning("No data found for this month.")
        else:
            weekly_summary = df_month.groupby("week_start").apply(
                lambda g: calculate_weekly_metrics(g, tolerance)
            ).reset_index()
            
            st.markdown("### 📋 Monthly Summary Table (Weekly Breakdown)")
            st.dataframe(weekly_summary, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Monthly Trends")
            
            plot_mt_trend(weekly_summary, "week_start", "MTTR (min)", "MTBF (min)", title_suffix=" by Week")
            plot_stability_trend(weekly_summary, "week_start", "Stability Index (%)", title_suffix=" by Week")
            
            run_durations_monthly = calculate_run_rate_metrics(df_month, tolerance)["run_durations"]
            bucket_trend = run_durations_monthly.groupby(["run_end", "time_bucket"]).size().reset_index(name="count")
            bucket_trend["week_start"] = bucket_trend["run_end"].dt.to_period('W-MON').dt.start_time.dt.date
            bucket_trend_week = bucket_trend.groupby(["week_start", "time_bucket"])["count"].sum().reset_index()
            
            fig_tb_trend = px.bar(
                bucket_trend_week, x="week_start", y="count", color="time_bucket",
                category_orders={"time_bucket": results_full["bucket_order"]},
                color_discrete_map=results_full["bucket_color_map"],
                title="Weekly Time Bucket Trend (Selected Month)",
                hover_data={"count": True, "week_start": True}
            )
            fig_tb_trend.update_layout(barmode="stack", xaxis=dict(title="Week Start"))
            st.plotly_chart(fig_tb_trend, use_container_width=True)
            with st.expander("📊 Weekly Time Bucket Trend – Data", expanded=False):
                st.dataframe(bucket_trend_week)

else:
    st.info("👈 Upload a cleaned run rate Excel file to begin. Headers in ROW 1 please.")