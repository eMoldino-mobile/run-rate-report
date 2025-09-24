import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings

# --- Page and Code Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

# --- Constants ---
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- Core Calculation Class ---
class RunRateCalculator:
    def __init__(self, df: pd.DataFrame, tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.analysis_mode = analysis_mode # New mode: 'aggregate' or 'by_run'
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
            datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
            df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
        elif "SHOT TIME" in df.columns:
            df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
        else:
            return pd.DataFrame()

        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        if df.empty: return pd.DataFrame()

        if "ACTUAL CT" in df.columns:
            time_diff_sec = df["shot_time"].diff().dt.total_seconds()
            prev_actual_ct = df["ACTUAL CT"].shift(1)
            rounding_buffer = 2.0
            use_timestamp_diff = (prev_actual_ct == 999.9) | (time_diff_sec > (prev_actual_ct + rounding_buffer))
            df["ct_diff_sec"] = np.where(use_timestamp_diff, time_diff_sec, prev_actual_ct)
        else:
            df["ct_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        if not df.empty and pd.isna(df.loc[0, "ct_diff_sec"]):
            df.loc[0, "ct_diff_sec"] = df.loc[0, "ACTUAL CT"] if "ACTUAL CT" in df.columns else 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'stop_event' not in df.columns:
            return pd.DataFrame()

        df['hour'] = df['shot_time'].dt.hour
        df['downtime_min_event'] = np.where(df['stop_event'], df['ct_diff_sec'] / 60, np.nan)
        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        total_downtime = hourly_groups['downtime_min_event'].sum()
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ct_diff_sec'].sum() / 60
        hourly_summary = pd.DataFrame({'stops': stops, 'total_downtime_min': total_downtime})
        hourly_summary = hourly_summary.join(uptime_min.rename('uptime_min')).fillna(0).reset_index()
        hourly_summary['mttr_min'] = hourly_summary['total_downtime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])
        total_runtime = hourly_summary['uptime_min'] + hourly_summary['total_downtime_min']
        hourly_summary['stability_index'] = np.where(
            total_runtime > 0,
            (hourly_summary['uptime_min'] / total_runtime) * 100,
            np.where(hourly_summary['stops'] == 0, 100.0, 0.0)
        )
        return hourly_summary

    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        # --- Dynamic Mode CT and Tolerance Limit Calculation ---
        if self.analysis_mode == 'by_run' and 'run_id' in df.columns:
            # Calculate mode for each individual production run
            run_modes = df.groupby('run_id')['ACTUAL CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
            df['mode_ct'] = df['run_id'].map(run_modes)
            # Limits are now a Series (a column) instead of a single value
            lower_limit = df['mode_ct'] * (1 - self.tolerance)
            upper_limit = df['mode_ct'] * (1 + self.tolerance)
            # Store these series in the df for the plotting function to use
            df['lower_limit'] = lower_limit
            df['upper_limit'] = upper_limit
            mode_ct_display = "Varies by Run" # For display purposes
        else:
            # Original aggregate mode calculation
            df_for_mode_calc = df[df["ct_diff_sec"] <= 28800]
            mode_ct = df_for_mode_calc["ACTUAL CT"].mode().iloc[0] if not df_for_mode_calc["ACTUAL CT"].mode().empty else 0
            lower_limit = mode_ct * (1 - self.tolerance)
            upper_limit = mode_ct * (1 + self.tolerance)
            mode_ct_display = mode_ct

        # --- Stop Detection ---
        stop_condition = (
            ((df["ct_diff_sec"] < lower_limit) | (df["ct_diff_sec"] > upper_limit))
            & (df["ct_diff_sec"] <= 28800)
        )
        df["stop_flag"] = np.where(stop_condition, 1, 0)
        df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)
        
        # --- The rest of the calculations proceed as before ---
        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        production_time_sec = total_runtime_sec - downtime_sec
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else (100.0 if stop_events == 0 else 0.0)
        df["run_group"] = df["stop_event"].cumsum()
        run_durations = df[df["stop_flag"] == 0].groupby("run_group")["ct_diff_sec"].sum().div(60).reset_index(name="duration_min")
        max_minutes = min(run_durations["duration_min"].max(), 240) if not run_durations.empty else 0
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
        if edges and len(edges) > 1:
            last_edge_start = edges[-2]
            labels[-1] = f"{last_edge_start}+"
            edges[-1] = np.inf
        if not run_durations.empty:
            run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False, include_lowest=True)
        reds, blues, greens = px.colors.sequential.Reds[4:8], px.colors.sequential.Blues[3:9], px.colors.sequential.Greens[4:9]
        bucket_color_map = {}
        red_idx, blue_idx, green_idx = 0, 0, 0
        for label in labels:
            try:
                lower_bound = int(label.split("-")[0].replace('+', ''))
                if lower_bound < 60:
                    bucket_color_map[label] = reds[red_idx % len(reds)]; red_idx += 1
                elif 60 <= lower_bound < 160:
                    bucket_color_map[label] = blues[blue_idx % len(blues)]; blue_idx += 1
                else:
                    bucket_color_map[label] = greens[green_idx % len(greens)]; green_idx += 1
            except (ValueError, IndexError): continue
        hourly_summary = self._calculate_hourly_summary(df)
        
        # Use scalar values for display in metric cards
        scalar_lower_limit = lower_limit.mean() if isinstance(lower_limit, pd.Series) else lower_limit
        scalar_upper_limit = upper_limit.mean() if isinstance(upper_limit, pd.Series) else upper_limit

        return {
            "processed_df": df, "mode_ct": mode_ct_display, "lower_limit": scalar_lower_limit,
            "upper_limit": scalar_upper_limit, "total_shots": total_shots, "efficiency": efficiency,
            "stop_events": stop_events, "normal_shots": normal_shots, "mttr_min": mttr_min,
            "mtbf_min": mtbf_min, "stability_index": stability_index, "run_durations": run_durations,
            "bucket_labels": labels, "bucket_color_map": bucket_color_map, "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_sec, "production_time_sec": production_time_sec, "downtime_sec": downtime_sec,
        }
# --- UI Helper and Plotting Functions ---

def create_gauge(value, title, steps=None):
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps:
        gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else:
        gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct, time_agg='hourly'):
    if df.empty:
        st.info("No shot data to display for this period."); return
    df = df.copy()
    df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB')
    df['plot_time'] = df['shot_time']
    stop_indices = df[df['stop_flag'] == 1].index
    if not stop_indices.empty:
        df.loc[stop_indices, 'plot_time'] = df['shot_time'].shift(1).loc[stop_indices]
    fig = go.Figure()

    # --- NEW: Logic to draw one or multiple tolerance bands ---
    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        # Draw a separate band for each production run
        for run_id, group in df.groupby('run_id'):
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=group['plot_time'].min(), y0=group['lower_limit'].iloc[0],
                x1=group['plot_time'].max(), y1=group['upper_limit'].iloc[0],
                fillcolor=PASTEL_COLORS['green'], opacity=0.2, layer="below", line_width=0
            )
    else:
        # Draw a single band for the whole period
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=df['plot_time'].min(), y0=lower_limit,
            x1=df['plot_time'].max(), y1=upper_limit,
            fillcolor=PASTEL_COLORS['green'], opacity=0.2, layer="below", line_width=0
        )

    fig.add_trace(go.Bar(x=df['plot_time'], y=df['ct_diff_sec'], marker_color=df['color'], name='Cycle Time'))
    
    y_axis_cap_val = mode_ct if isinstance(mode_ct, (int, float)) else df['mode_ct'].mean() if 'mode_ct' in df else 50
    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)
    tick_format = {"hourly": "%H:%M", "daily": "%b %d", "weekly": "Week %W"}.get(time_agg, "%b %d")

    fig.update_layout(title="Cycle Time per Shot vs. Tolerance", xaxis_title="Time", yaxis_title="Cycle Time (sec)",
                      yaxis=dict(range=[0, y_axis_cap]), bargap=0.05, xaxis=dict(tickformat=tick_format, showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

def plot_trend_chart(df, x_col, y_col, title, x_title, y_title, y_range=[0, 101], is_stability=False):
    fig = go.Figure()
    marker_config = {}
    if is_stability:
        marker_config['color'] = [PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green'] for v in df[y_col]]
        marker_config['size'] = 10
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", name=y_title,
                             line=dict(color="black" if is_stability else "royalblue", width=2), marker=marker_config))
    if is_stability:
        for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]:
            fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1, fillcolor=c, opacity=0.2, line_width=0, layer="below")
    fig.update_layout(title=title, yaxis=dict(title=y_title, range=y_range), xaxis_title=x_title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def format_duration(seconds):
    if pd.isna(seconds) or seconds < 0: return "N/A"
    days = int(seconds // (24 * 3600))
    hours = int((seconds % (24 * 3600)) // 3600)
    minutes = int((seconds % 3600) // 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"
    
def calculate_daily_summaries_for_week(df_week, tolerance, analysis_mode):
    daily_results_list = []
    for date in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date]
        if not df_day.empty:
            calc = RunRateCalculator(df_day.copy(), tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'date': date, 'stability_index': res.get('stability_index', np.nan),
                       'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                       'stops': res.get('stop_events', 0)}
            daily_results_list.append(summary)
    return pd.DataFrame(daily_results_list) if daily_results_list else pd.DataFrame()

def calculate_weekly_summaries_for_month(df_month, tolerance, analysis_mode):
    weekly_results_list = []
    for week in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week]
        if not df_week.empty:
            calc = RunRateCalculator(df_week.copy(), tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'week': week, 'stability_index': res.get('stability_index', np.nan),
                       'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                       'stops': res.get('stop_events', 0)}
            weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance):
    """Iterates through a period's data, calculates metrics for each run, and returns a summary DataFrame."""
    run_summary_list = []
    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty:
            # For each individual run, we do an 'aggregate' calculation on it
            calc = RunRateCalculator(df_run.copy(), tolerance, analysis_mode='aggregate')
            res = calc.results
            summary = {
                'run_label': run_label,
                'start_time': df_run['shot_time'].min(),
                'stability_index': res.get('stability_index', np.nan),
                'mttr_min': res.get('mttr_min', np.nan),
                'mtbf_min': res.get('mtbf_min', np.nan),
                'stops': res.get('stop_events', 0)
            }
            run_summary_list.append(summary)
    if not run_summary_list:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(run_summary_list).sort_values('start_time').reset_index(drop=True)
    return summary_df

# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")

with st.sidebar.expander("ℹ️ About This Dashboard", expanded=False):
    st.markdown("""
    ### Run Rate Analysis
    - **Efficiency (%)**: Normal Shots ÷ Total Shots
    - **MTTR (min)**: Average downtime per stop.
    - **MTBF (min)**: Average uptime between stops.
    - **Stability Index (%)**: Uptime ÷ (Uptime + Downtime)
    - **Bucket Analysis**: Groups run durations into 20-min intervals.
    ---
    ### Analysis Levels
    - **Daily**: Hourly trends for one day.
    - **Weekly / Monthly**: Aggregated data, with daily/weekly trend charts.
    - **Weekly / Monthly (by Run)**: A more precise analysis where the tolerance for stops is calculated from the Mode CT of each individual production run (a period with no >8hr breaks).
    ---
    ### Tolerance Slider
    Defines the acceptable CT range around the Mode CT.
    """)

analysis_level = st.sidebar.radio("Select Analysis Level", ["Daily", "Weekly", "Monthly", "Weekly (by Run)", "Monthly (by Run)"])

uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("👈 Upload an Excel file to begin."); st.stop()

@st.cache_data
def load_data(file): return pd.read_excel(file)

df_raw = load_data(uploaded_file)
id_col = "TOOLING ID" if "TOOLING ID" in df_raw.columns else "EQUIPMENT CODE"
if id_col not in df_raw.columns:
    st.error(f"File must contain 'TOOLING ID' or 'EQUIPMENT CODE'."); st.stop()

tool_id = st.sidebar.selectbox(f"Select {id_col}", df_raw[id_col].unique())
df_tool = df_raw.loc[df_raw[id_col] == tool_id].copy()
if df_tool.empty:
    st.warning(f"No data for: {tool_id}"); st.stop()

st.sidebar.markdown("---")
tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")

@st.cache_data(show_spinner="Performing initial data processing...")
def get_processed_data(df):
    # This initial run just gets the timestamps and identifies the runs
    base_calc = RunRateCalculator(df, 0.01) # Tolerance doesn't matter here
    df_processed = base_calc.results.get("processed_df", pd.DataFrame())
    if not df_processed.empty:
        df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
        df_processed['date'] = df_processed['shot_time'].dt.date
        df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
        # Identify production runs
        is_new_run = df_processed['ct_diff_sec'] > 28800
        df_processed['run_id'] = is_new_run.cumsum()
        run_start_dates = df_processed.groupby('run_id')['shot_time'].min()
        run_labels = {run_id: f"{i+1:03d} ({date.strftime('%Y-%m-%d')})" for i, (run_id, date) in enumerate(run_start_dates.items())}
        df_processed['run_label'] = df_processed['run_id'].map(run_labels)
    return df_processed

df_processed = get_processed_data(df_tool)

if df_processed.empty:
    st.error(f"Could not process data for {tool_id}."); st.stop()

st.title(f"Run Rate Dashboard: {tool_id}")

# --- Determine mode and filter data for the selected view ---
mode = 'by_run' if '(by Run)' in analysis_level else 'aggregate'
df_view = pd.DataFrame()

if analysis_level == "Daily":
    st.header("Daily Analysis")
    available_dates = sorted(df_processed["date"].unique())
    selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
    df_view = df_processed[df_processed["date"] == selected_date]
    sub_header = f"Summary for {selected_date.strftime('%d %b %Y')}"
elif "Weekly" in analysis_level:
    st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
    available_weeks = sorted(df_processed["week"].unique())
    year = df_processed['shot_time'].iloc[0].year
    selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1)
    # --- FIX: In "by Run" mode, we need to get all runs that TOUCH the selected week ---
    if mode == 'by_run':
        runs_in_week = df_processed[df_processed['week'] == selected_week]['run_label'].unique()
        df_view = df_processed[df_processed['run_label'].isin(runs_in_week)]
    else:
        df_view = df_processed[df_processed["week"] == selected_week]
    sub_header = f"Summary for Week {selected_week}"
elif "Monthly" in analysis_level:
    st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
    available_months = sorted(df_processed["month"].unique())
    selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'))
    if mode == 'by_run':
        runs_in_month = df_processed[df_processed['month'] == selected_month]['run_label'].unique()
        df_view = df_processed[df_processed['run_label'].isin(runs_in_month)]
    else:
        df_view = df_processed[df_processed["month"] == selected_month]
    sub_header = f"Summary for {selected_month.strftime('%B %Y')}"

# --- Main calculation and rendering block ---
if df_view.empty:
    st.warning(f"No data for the selected period.")
else:
    calc = RunRateCalculator(df_view.copy(), tolerance, analysis_mode=mode)
    results = calc.results
    st.subheader(sub_header)

    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        total_d = results.get('total_runtime_sec', 0); prod_t = results.get('production_time_sec', 0); down_t = results.get('downtime_sec', 0)
        prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
        down_p = (down_t / total_d * 100) if total_d > 0 else 0
        col1.metric("MTTR", f"{results.get('mttr_min', 0):.1f} min")
        col2.metric("MTBF", f"{results.get('mtbf_min', 0):.1f} min")
        col3.metric("Total Run Duration", format_duration(total_d))
        col4.metric("Production Time", format_duration(prod_t), f"{prod_p:.1f}%")
        col5.metric("Downtime", format_duration(down_t), f"{down_p:.1f}%", delta_color="inverse")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.plotly_chart(create_gauge(results.get('efficiency', 0) * 100, "Efficiency (%)"), use_container_width=True)
        steps = [{'range': [0, 50], 'color': PASTEL_COLORS['red']}, {'range': [50, 70], 'color': PASTEL_COLORS['orange']},{'range': [70, 100], 'color': PASTEL_COLORS['green']}]
        c2.plotly_chart(create_gauge(results.get('stability_index', 0), "Stability Index (%)", steps=steps), use_container_width=True)

    with st.container(border=True):
        c1,c2,c3 = st.columns(3)
        t_s = results.get('total_shots', 0); n_s = results.get('normal_shots', 0)
        s_s = t_s - n_s
        n_p = (n_s / t_s * 100) if t_s > 0 else 0
        s_p = (s_s / t_s * 100) if t_s > 0 else 0
        c1.metric("Total Shots", f"{t_s:,}")
        c2.metric("Normal Shots", f"{n_s:,}", f"{n_p:.1f}%")
        c3.metric("Stop Count", f"{results.get('stop_events', 0)}", f"{s_p:.1f}% Stopped Shots", delta_color="inverse")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        mode_val = results.get('mode_ct', 0)
        mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else mode_val
        c1.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}")
        with c2:
            with st.container(border=True): st.metric("Mode CT (sec)", mode_disp)
        c3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}")

    # --- Breakdown Tables for Weekly/Monthly Views ---
    if analysis_level == "Weekly":
        daily_summary_df = calculate_daily_summaries_for_week(df_view, tolerance)
        with st.expander("View Daily Breakdown Table", expanded=False):
            if not daily_summary_df.empty:
                d_df = daily_summary_df.copy()
                d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                d_df.rename(columns={'date': 'Day', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                st.dataframe(d_df.style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
    elif analysis_level == "Monthly":
        weekly_summary_df = calculate_weekly_summaries_for_month(df_view, tolerance)
        with st.expander("View Weekly Breakdown Table", expanded=False):
            if not weekly_summary_df.empty:
                d_df = weekly_summary_df.copy()
                d_df.rename(columns={'week': 'Week', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                st.dataframe(d_df.style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
    elif analysis_level in ["Weekly (by Run)", "Monthly (by Run)"]:
        run_summary_df = calculate_run_summaries(df_view, tolerance)
        with st.expander("View Run Breakdown Table", expanded=False):
            if not run_summary_df.empty:
                d_df = run_summary_df.copy()
                d_df.rename(columns={'run_label': 'Run ID', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                st.dataframe(d_df[['Run ID', 'Stability (%)', 'MTTR (min)', 'MTBF (min)', 'Stops']].style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)

    # --- Plot main chart and trends ---
    time_agg = 'hourly' if analysis_level == 'Daily' else 'daily' if 'Weekly' in analysis_level else 'weekly'
    plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=time_agg)
    with st.expander("View Shot Data Table", expanded=False):
        st.dataframe(results['processed_df'][['shot_time', 'run_label', 'ACTUAL CT', 'ct_diff_sec', 'stop_flag', 'stop_event']])

    st.markdown("---")

    if analysis_level == "Daily":
        st.header("Hourly Analysis")
        run_durations_day = results.get("run_durations", pd.DataFrame())
        processed_day_df = results.get('processed_df', pd.DataFrame())
        stop_events_df = processed_day_df.loc[processed_day_df['stop_event']].copy()
        complete_runs = pd.DataFrame(); incomplete_run = pd.DataFrame()
        if not stop_events_df.empty:
            stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
            end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
            run_durations_day['run_end_time'] = run_durations_day['run_group'].map(end_time_map)
            complete_runs = run_durations_day.dropna(subset=['run_end_time']).copy()
            incomplete_run = run_durations_day[run_durations_day['run_end_time'].isna()]
        else: incomplete_run = run_durations_day
        c1,c2 = st.columns(2)
        with c1:
            if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                fig_b = px.bar(b_counts, title="Time Bucket Analysis (Completed Runs)", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                st.plotly_chart(fig_b, use_container_width=True)
                with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
            else: st.info("No complete runs.")
        with c2:
            plot_trend_chart(results['hourly_summary'], 'hour', 'stability_index', "Hourly Stability Trend", "Hour of Day", "Stability (%)", is_stability=True)
            with st.expander("View Stability Data", expanded=False): st.dataframe(results['hourly_summary'])
        
        st.subheader("Hourly Bucket Trend")
        if not complete_runs.empty:
            complete_runs['hour'] = complete_runs['run_end_time'].dt.hour
            pivot_df = pd.crosstab(index=complete_runs['hour'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
            pivot_df = pivot_df.reindex(pd.Index(range(24), name='hour'), fill_value=0)
            fig_hourly_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, title='Hourly Distribution of Run Durations', barmode='stack', color_discrete_map=results["bucket_color_map"], labels={'hour': 'Hour of Stop', 'value': 'Number of Runs', 'variable': 'Run Duration (min)'})
            st.plotly_chart(fig_hourly_bucket, use_container_width=True)
            with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
        
        st.subheader("Hourly MTTR & MTBF Trend")
        hourly_summary = results['hourly_summary']
        if not hourly_summary.empty and hourly_summary['stops'].sum() > 0:
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)))
            fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4), yaxis='y2'))
            fig_mt.update_layout(title="Hourly MTTR & MTBF Trend", yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mt, use_container_width=True)
            with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(hourly_summary)

    elif analysis_level in ["Weekly", "Monthly"]:
        trend_level = "Daily" if "Weekly" in analysis_level else "Weekly"
        st.header(f"{trend_level} Trends for {analysis_level.split(' ')[0]}")
        summary_df = calculate_daily_summaries_for_week(df_view, tolerance, mode) if "Weekly" in analysis_level else calculate_weekly_summaries_for_month(df_view, tolerance, mode)
        run_durations = results.get("run_durations", pd.DataFrame())
        processed_df = results.get('processed_df', pd.DataFrame())
        stop_events_df = processed_df.loc[processed_df['stop_event']].copy()
        complete_runs = pd.DataFrame()
        if not stop_events_df.empty:
            stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
            end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
            run_durations['run_end_time'] = run_durations['run_group'].map(end_time_map)
            complete_runs = run_durations.dropna(subset=['run_end_time']).copy()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Total Bucket Analysis")
            if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration(min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                st.plotly_chart(fig_b, use_container_width=True)
                with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
            else: st.info("No complete runs.")
        with c2:
            st.subheader(f"{trend_level} Stability Trend")
            if not summary_df.empty:
                x_col = 'date' if trend_level == "Daily" else 'week'
                plot_trend_chart(summary_df, x_col, 'stability_index', f"{trend_level} Stability Trend", trend_level, "Stability (%)", is_stability=True)
                with st.expander("View Stability Data", expanded=False): st.dataframe(summary_df)
            else: st.info(f"No {trend_level.lower()} data.")
        
        st.subheader(f"{trend_level} Bucket Trend")
        if not complete_runs.empty and not summary_df.empty:
            time_col = 'date' if trend_level == "Daily" else 'week'
            complete_runs[time_col] = complete_runs['run_end_time'].dt.date if trend_level == "Daily" else complete_runs['run_end_time'].dt.isocalendar().week
            pivot_df = pd.crosstab(index=complete_runs[time_col], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
            all_units = summary_df[time_col]
            pivot_df = pivot_df.reindex(all_units, fill_value=0)
            fig_trend_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, title=f'{trend_level} Distribution of Run Durations', barmode='stack', color_discrete_map=results["bucket_color_map"], labels={time_col: trend_level, 'value': 'Number of Runs', 'variable': 'Run Duration (min)'})
            st.plotly_chart(fig_trend_bucket, use_container_width=True)
            with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)

        st.subheader(f"{trend_level} MTTR & MTBF Trend")
        if not summary_df.empty and summary_df['stops'].sum() > 0:
            x_col = 'date' if trend_level == "Daily" else 'week'
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Scatter(x=summary_df[x_col], y=summary_df['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)))
            fig_mt.add_trace(go.Scatter(x=summary_df[x_col], y=summary_df['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4), yaxis='y2'))
            fig_mt.update_layout(title=f"{trend_level} MTTR & MTBF Trend", yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mt, use_container_width=True)
            with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(summary_df)

    elif "by Run" in analysis_level:
        st.header(f"Run-Based Analysis")
        run_summary_df = calculate_run_summaries(df_view, tolerance)
        
        run_durations = results.get("run_durations", pd.DataFrame())
        processed_df = results.get('processed_df', pd.DataFrame())
        stop_events_df = processed_df.loc[processed_df['stop_event']].copy()
        complete_runs = pd.DataFrame()
        if not stop_events_df.empty:
            stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
            end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
            run_durations['run_end_time'] = run_durations['run_group'].map(end_time_map)
            complete_runs = run_durations.dropna(subset=['run_end_time']).copy()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Total Bucket Analysis")
            if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration(min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                st.plotly_chart(fig_b, use_container_width=True)
                with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
            else: st.info("No complete runs.")
        with c2:
            st.subheader("Stability per Production Run")
            if not run_summary_df.empty:
                plot_trend_chart(run_summary_df, 'run_label', 'stability_index', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
                with st.expander("View Stability Data", expanded=False): st.dataframe(run_summary_df)
            else: st.info(f"No runs to analyze.")
        
        st.subheader("Bucket Trend per Production Run")
        if not complete_runs.empty and not run_summary_df.empty:
            # Map run_group to run_label
            run_group_to_label_map = processed_df.drop_duplicates('run_group')[['run_group', 'run_label']].set_index('run_group')['run_label']
            complete_runs['run_label'] = complete_runs['run_group'].map(run_group_to_label_map)
            
            pivot_df = pd.crosstab(index=complete_runs['run_label'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
            all_runs = run_summary_df['run_label']
            pivot_df = pivot_df.reindex(all_runs, fill_value=0)
            fig_trend_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, title='Distribution of Run Durations per Run', barmode='stack', color_discrete_map=results["bucket_color_map"], labels={'run_label': 'Run ID', 'value': 'Number of Runs', 'variable': 'Run Duration (min)'})
            st.plotly_chart(fig_trend_bucket, use_container_width=True)
            with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)

        st.subheader("MTTR & MTBF per Production Run")
        if not run_summary_df.empty and run_summary_df['stops'].sum() > 0:
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Scatter(x=run_summary_df['run_label'], y=run_summary_df['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)))
            fig_mt.add_trace(go.Scatter(x=run_summary_df['run_label'], y=run_summary_df['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4), yaxis='y2'))
            fig_mt.update_layout(title="MTTR & MTBF per Run", yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mt, use_container_width=True)
            with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(run_summary_df)

