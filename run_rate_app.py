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
    
    # Correctly calculating efficiency
    efficiency = normal_shots / total_shots if total_shots > 0 else 0

    total_runtime = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() / 60
    run_hours = total_runtime / 60
    downtime = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum() / 60
    production_time = total_runtime - downtime
    
    if stop_events > 0:
        downtime_events = df.loc[df["stop_event"], "ct_diff_sec"] / 60
        mttr = downtime_events.mean()
        # Correctly calculate uptime by excluding downtimes
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

def export_to_excel(df, results):
    """Generates and returns an Excel file from dataframes and results."""
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Dashboard"
    
    summary_data = [
        ["Total Shot Count", results.get("total_shots", 0)],
        ["Normal Shot Count", results.get("normal_shots", 0)],
        # Corrected f-string on this line
        ["Efficiency", f"{results.get('efficiency', 0) * 100:.2f}%"],
        ["Stop Count", results.get("stop_events", 0)],
        # Corrected f-string on these lines
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

***

### Summary of Key Changes

* **Efficiency Added to Core Metrics:** The `calculate_run_rate_metrics` function now includes `efficiency` in its return dictionary, providing a single source of truth.
* **MTBF Bug Fix:** When no stops occur, **MTBF** is now correctly set to `NaN` (or `None`), preventing misleadingly large values. The MTBF calculation itself was corrected to sum up total uptime before dividing by the number of stops, providing a more accurate metric.
* **Consistent Data Filtering:** The logic for `Analysis Dashboard` and `Daily Analysis` pages now applies a time-based filter to the raw data *before* running `calculate_run_rate_metrics`. This ensures that all metrics (including `MTBF` and run buckets) are calculated accurately for the selected time period, eliminating the risk of partial runs from outside the window skewing results.
* **Dynamic Color Map:** The `make_bucket_color_map` function has been improved to accept the list of active buckets, so the legend only displays colors and labels for the run time buckets actually present in the data. This provides a cleaner and more relevant visualization.
* **Standardized Naming:** All internal variables and dataframe column names are now in `snake_case` (e.g., `shot_time`, `ct_diff_sec`). The original `UPPERCASE` names are only used when reading the initial file, ensuring a cleaner codebase and reducing the risk of bugs from inconsistent naming.