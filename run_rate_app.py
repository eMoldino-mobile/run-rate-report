import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import warnings
import streamlit.components.v1 as components # Import v1 specifically
import xlsxwriter
from datetime import datetime, timedelta, date # Import date specifically
import traceback # Import traceback for better error logging

# --- Page and Code Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

# --- Constants ---
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- [UNIFIED] Core Calculation Class (Used for BOTH Dashboard & Export Prep) ---
class RunRateCalculator:
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.analysis_mode = analysis_mode
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        # Ensure 'shot_time' column exists or can be created
        if "shot_time" not in df.columns:
            if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
                datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
                df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
            elif "SHOT TIME" in df.columns:
                df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
            else:
                return pd.DataFrame() # Cannot proceed without time

        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        if df.empty: return pd.DataFrame()

        # Always calculate the raw time difference. Crucial for Excel formula and gap logic.
        df["time_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        # Handle the NaN for the first shot.
        if not df.empty and pd.isna(df.loc[0, "time_diff_sec"]):
            if "ACTUAL CT" in df.columns:
                # Use ACTUAL CT for the first shot's "diff" if available, conceptually representing its duration
                df.loc[0, "time_diff_sec"] = df.loc[0, "ACTUAL CT"]
            else:
                # If no ACTUAL CT, the first diff is treated as 0 (consistent with Excel formula)
                df.loc[0, "time_diff_sec"] = 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'stop_event' not in df.columns: return pd.DataFrame()
        df['hour'] = df['shot_time'].dt.hour
        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        # Use adj_ct_sec for downtime sum
        total_downtime_sec = hourly_groups.apply(lambda x: x[x['stop_flag'] == 1]['adj_ct_sec'].sum())
        # Use ACTUAL CT for uptime sum
        uptime_min = 0 # Default
        if 'ACTUAL CT' in df.columns: # Check if column exists before grouping
            uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ACTUAL CT'].sum() / 60
        else: # Estimate uptime based on adj_ct_sec if ACTUAL CT is missing
            uptime_min = df[df['stop_flag'] == 0].groupby('hour')['adj_ct_sec'].sum() / 60

        shots = hourly_groups.size().rename('total_shots')

        hourly_summary = pd.DataFrame(index=range(24))
        hourly_summary['hour'] = hourly_summary.index
        hourly_summary = hourly_summary.join(stops.rename('stops')).join(shots).join(uptime_min.rename('uptime_min')).fillna(0)
        hourly_summary = hourly_summary.join(total_downtime_sec.rename('total_downtime_sec')).fillna(0)

        hourly_summary['mttr_min'] = (hourly_summary['total_downtime_sec'] / 60) / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])

        effective_runtime_min = hourly_summary['uptime_min'] + (hourly_summary['total_downtime_sec'] / 60)
        hourly_summary['stability_index'] = np.where(
            effective_runtime_min > 0,
            (hourly_summary['uptime_min'] / effective_runtime_min) * 100,
            np.where(hourly_summary['stops'] == 0, 100.0, 0.0)
        )
        return hourly_summary.fillna(0)

    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        # Allow processing even if ACTUAL CT is missing, but metrics will be limited
        if df.empty or ("time_diff_sec" not in df.columns):
            return {}

        # Determine Mode CT and Limits (only if ACTUAL CT exists)
        mode_ct = 0
        lower_limit = 0
        upper_limit = np.inf # Default to infinity if no ACTUAL CT
        mode_ct_display = "N/A"

        if "ACTUAL CT" in df.columns:
            if self.analysis_mode == 'by_run' and 'run_id' in df.columns:
                run_modes = df.groupby('run_id')['ACTUAL CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
                df['mode_ct'] = df['run_id'].map(run_modes)
                lower_limit = df['mode_ct'] * (1 - self.tolerance)
                upper_limit = df['mode_ct'] * (1 + self.tolerance)
                df['lower_limit'] = lower_limit
                df['upper_limit'] = upper_limit
                mode_ct_display = "Varies by Run"
            else:
                df_for_mode_calc = df[df["ACTUAL CT"] < 999.9].copy()
                df_for_mode_calc['rounded_ct'] = df_for_mode_calc['ACTUAL CT'].round(0)
                mode_ct = df_for_mode_calc['rounded_ct'].mode().iloc[0] if not df_for_mode_calc['rounded_ct'].mode().empty else 0
                lower_limit = mode_ct * (1 - self.tolerance)
                upper_limit = mode_ct * (1 + self.tolerance)
                mode_ct_display = mode_ct
        else:
            lower_limit = 0 # Define explicitly
            upper_limit = np.inf


        # --- Stop Detection Logic ---
        df["stop_flag"] = 0
        df["stop_event"] = False

        if "ACTUAL CT" in df.columns:
            # Need to handle cases where lower/upper limit might be Series (in 'by_run' mode)
            if isinstance(lower_limit, pd.Series):
                is_abnormal_cycle = (df["ACTUAL CT"] < df['lower_limit']) | (df["ACTUAL CT"] > df['upper_limit'])
            else:
                is_abnormal_cycle = (df["ACTUAL CT"] < lower_limit) | (df["ACTUAL CT"] > upper_limit)

            prev_actual_ct = df["ACTUAL CT"].shift(1)
            is_downtime_gap = df["time_diff_sec"] > (prev_actual_ct.fillna(0) + self.downtime_gap_tolerance)
            df["stop_flag"] = np.where(is_abnormal_cycle | is_downtime_gap.fillna(False), 1, 0)
        else:
            df["stop_flag"] = 0 # Default to no stops if ACTUAL CT missing


        if not df.empty:
            df.loc[0, "stop_flag"] = 0 # First shot is never a stop

        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # Adjusted CT: Use time_diff_sec during stops, ACTUAL CT otherwise (if available)
        if "ACTUAL CT" in df.columns:
            df["adj_ct_sec"] = np.where(df["stop_flag"] == 1, df["time_diff_sec"], df["ACTUAL CT"])
        else:
            # If no ACTUAL CT, use time_diff_sec always for adj_ct_sec
            df["adj_ct_sec"] = df["time_diff_sec"]


        # --- Calculate Metrics ---
        total_shots = len(df)
        stop_events = df["stop_event"].sum()

        downtime_sec = df.loc[df['stop_flag'] == 1, 'adj_ct_sec'].sum()
        mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0

        production_time_sec = 0
        if "ACTUAL CT" in df.columns:
            production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum()
        else:
            production_time_sec = df.loc[df['stop_flag'] == 0, 'adj_ct_sec'].sum()


        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)

        total_runtime_calc = production_time_sec + downtime_sec # Consistent definition
        stability_index = (production_time_sec / total_runtime_calc * 100) if total_runtime_calc > 0 else (100.0 if stop_events == 0 else 0.0)

        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        # Run durations
        run_duration_col = "ACTUAL CT" if "ACTUAL CT" in df.columns else "adj_ct_sec"
        df_for_runs = df[df['adj_ct_sec'] <= (24 * 3600)].copy() # Use a reasonable upper limit
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")[run_duration_col].sum().div(60).reset_index(name="duration_min")


        # --- Calcs needed specifically for the Excel Exporter ---
        avg_cycle_time_sec = production_time_sec / normal_shots if normal_shots > 0 else 0
        first_stop_index = df[df['stop_event']].index.min()
        time_to_first_dt_sec = df.loc[:first_stop_index-1, 'adj_ct_sec'].sum() if pd.notna(first_stop_index) and first_stop_index > 0 else production_time_sec
        production_run_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0

        # --- Bucket Analysis ---
        labels = []
        bucket_color_map = {}
        if not run_durations.empty and 'duration_min' in run_durations.columns: # Check column exists
            # Ensure duration_min is numeric before finding max
            run_durations['duration_min'] = pd.to_numeric(run_durations['duration_min'], errors='coerce')
            run_durations.dropna(subset=['duration_min'], inplace=True)

            if not run_durations.empty: # Check again after potential drops
                max_minutes = min(run_durations["duration_min"].max(), 240)
                upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
                edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
                labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
                if edges and len(edges) > 1:
                    last_edge_start = edges[-2]; labels[-1] = f"{last_edge_start}+"; edges[-1] = np.inf
                # Ensure labels list is not empty before using pd.cut
                if labels:
                    try:
                        run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False, include_lowest=True)
                    except ValueError as e:
                        print(f"Warning during bucket creation: {e}.") # Print instead of st.warning inside class
                        run_durations["time_bucket"] = "Error" # Assign default if cut fails

                reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
                red_labels, blue_labels, green_labels = [], [], []
                for label in labels:
                    try:
                        lower_bound = int(str(label).split('-')[0].replace('+', ''))
                        if lower_bound < 60: red_labels.append(label)
                        elif 60 <= lower_bound < 160: blue_labels.append(label)
                        else: green_labels.append(label)
                    except (ValueError, IndexError, TypeError):
                        continue # Skip bad labels

                for i, label in enumerate(red_labels): bucket_color_map[label] = reds[i % len(reds)]
                for i, label in enumerate(blue_labels): bucket_color_map[label] = blues[i % len(blues)]
                for i, label in enumerate(green_labels): bucket_color_map[label] = greens[i % len(greens)]

        hourly_summary = self._calculate_hourly_summary(df)

        final_results = {
            "processed_df": df, "mode_ct": mode_ct_display, "total_shots": total_shots, "efficiency": efficiency,
            "stop_events": stop_events, "normal_shots": normal_shots, "mttr_min": mttr_min,
            "mtbf_min": mtbf_min, "stability_index": stability_index, "run_durations": run_durations,
            "bucket_labels": labels, "bucket_color_map": bucket_color_map, "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_calc, # Uptime + Downtime
            "production_time_sec": production_time_sec, "downtime_sec": downtime_sec,
            "avg_cycle_time_sec": avg_cycle_time_sec, "time_to_first_dt_min": time_to_first_dt_sec / 60,
            "production_run_sec": production_run_sec, # Wall-clock
            "tot_down_time_sec": downtime_sec # Alias
        }

        # Update results with specific limits based on analysis mode
        if self.analysis_mode == 'by_run' and isinstance(lower_limit, pd.Series) and not df.empty and 'mode_ct' in df.columns:
            # Check if mode_ct column actually exists before trying to access min/max
            final_results.update({"min_lower_limit": lower_limit.min(), "max_lower_limit": lower_limit.max(),
                                  "min_upper_limit": upper_limit.min(), "max_upper_limit": upper_limit.max(),
                                  "min_mode_ct": df['mode_ct'].min(), "max_mode_ct": df['mode_ct'].max()})
        elif "ACTUAL CT" in df.columns :
            final_results.update({"lower_limit": lower_limit, "upper_limit": upper_limit})
        # Add placeholder limits if ACTUAL CT was missing entirely
        elif "ACTUAL CT" not in df.columns:
            final_results.update({"lower_limit": 0, "upper_limit": np.inf})


        return final_results


# --- UI Helper and Plotting Functions ---
def create_gauge(value, title, steps=None):
    numeric_value = pd.to_numeric(value, errors='coerce'); gauge_value = 0 if pd.isna(numeric_value) else numeric_value
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps: gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else: gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=gauge_value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20)); return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct, time_agg='hourly'):
    if df.empty or 'adj_ct_sec' not in df.columns: st.info("No shot data with cycle times to display."); return
    df = df.copy(); df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB')
    df['plot_time'] = df['shot_time']
    stop_indices = df[df['stop_flag'] == 1].index
    if not stop_indices.empty: valid_stop_indices = stop_indices[stop_indices > 0]; df.loc[valid_stop_indices, 'plot_time'] = df['shot_time'].shift(1).loc[valid_stop_indices]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['plot_time'], y=df['adj_ct_sec'], marker_color=df['color'], name='Cycle Time', showlegend=False))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Normal Shot", marker_color='#3498DB', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Stopped Shot", marker_color=PASTEL_COLORS['red'], showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(119, 221, 119, 0.3)', name='Tolerance Band', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='New Run Start', line=dict(color='purple', dash='dash', width=2), showlegend=True))

    # Ensure lower_limit and upper_limit are treated correctly (scalar or Series)
    ll = lower_limit if lower_limit is not None else 0
    ul = upper_limit if upper_limit is not None else np.inf

    # Check if 'lower_limit' column exists (implies by_run mode limits were calculated)
    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        for run_id, group in df.groupby('run_id'):
            if not group.empty:
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=group['shot_time'].min(), y0=group['lower_limit'].iloc[0],
                              x1=group['shot_time'].max(), y1=group['upper_limit'].iloc[0],
                              fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0)
    # Fallback for aggregate mode or if limits are scalar
    elif pd.notna(ll) and pd.notna(ul) and ul != np.inf:
        if not df.empty:
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=df['shot_time'].min(), y0=ll,
                          x1=df['shot_time'].max(), y1=ul,
                          fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0)

    if 'run_label' in df.columns:
        run_starts = df.groupby('run_label')['shot_time'].min().sort_values();
        for start_time in run_starts.iloc[1:]: fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="purple")
    y_axis_cap_val = 50
    if isinstance(mode_ct, (int, float)) and mode_ct > 0: y_axis_cap_val = mode_ct
    elif 'mode_ct' in df and not df['mode_ct'].empty:
        mean_mode = pd.to_numeric(df['mode_ct'], errors='coerce').mean()
        if pd.notna(mean_mode) and mean_mode > 0: y_axis_cap_val = mean_mode
    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)
    fig.update_layout(title="Cycle Time per Shot vs. Tolerance", xaxis_title="Time", yaxis_title="Cycle Time (sec)", yaxis=dict(range=[0, y_axis_cap]), bargap=0.05, xaxis=dict(showgrid=True), showlegend=True, legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_trend_chart(df, x_col, y_col, title, x_title, y_title, y_range=[0, 101], is_stability=False):
    if df is None or df.empty or y_col not in df.columns: st.info(f"Not enough data to plot {title}."); return
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df_plot = df.dropna(subset=[y_col])
    if df_plot.empty: st.info(f"No valid numeric data in '{y_col}' to plot {title}."); return
    fig = go.Figure(); marker_config = {}
    if is_stability: marker_config['color'] = [PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green'] for v in df_plot[y_col]]; marker_config['size'] = 10
    fig.add_trace(go.Scatter(x=df_plot[x_col], y=df_plot[y_col], mode="lines+markers", name=y_title, line=dict(color="black" if is_stability else "royalblue", width=2), marker=marker_config))
    if is_stability:
        for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]: fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1, fillcolor=c, opacity=0.2, line_width=0, layer="below")
    fig.update_layout(title=title, yaxis=dict(title=y_title, range=y_range), xaxis_title=x_title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig, use_container_width=True)

def plot_mttr_mtbf_chart(df, x_col, mttr_col, mtbf_col, shots_col, title):
    if df is None or df.empty or shots_col not in df.columns or df[shots_col].sum() == 0: st.info(f"Not enough data to plot {title}."); return
    df[mttr_col] = pd.to_numeric(df[mttr_col], errors='coerce')
    df[mtbf_col] = pd.to_numeric(df[mtbf_col], errors='coerce')
    df[shots_col] = pd.to_numeric(df[shots_col], errors='coerce')
    df_plot = df.dropna(subset=[mttr_col, mtbf_col, shots_col])
    if df_plot.empty: st.info(f"Missing numeric MTTR/MTBF/Shots data for {title}."); return
    mttr = df_plot[mttr_col]; mtbf = df_plot[mtbf_col]; shots = df_plot[shots_col]; x_axis = df_plot[x_col]
    max_mttr = np.nanmax(mttr[np.isfinite(mttr)]) if not mttr.empty and any(np.isfinite(mttr)) else 0
    max_mtbf = np.nanmax(mtbf[np.isfinite(mtbf)]) if not mtbf.empty and any(np.isfinite(mtbf)) else 0
    y_range_mttr = [0, max_mttr * 1.15 if max_mttr > 0 else 10]; y_range_mtbf = [0, max_mtbf * 1.15 if max_mtbf > 0 else 10]
    shots_min, shots_max = shots.min(), shots.max()
    if (shots_max - shots_min) == 0: scaled_shots = pd.Series([y_range_mtbf[1] / 2 if y_range_mtbf[1] > 0 else 0.5] * len(shots), index=shots.index)
    else: scaled_shots = (shots - shots_min) / (shots_max - shots_min) * (y_range_mtbf[1] * 0.9)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x_axis, y=mttr, name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mtbf, name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
    fig.add_trace(go.Scatter(x=x_axis, y=scaled_shots, name='Total Shots', mode='lines+markers+text', text=shots.astype(int), textposition='top center', textfont=dict(color='blue'), line=dict(color='blue', dash='dot')), secondary_y=True) # Added astype(int) for text
    fig.update_layout(title_text=title, yaxis_title="MTTR (min)", yaxis2_title="MTBF (min)", yaxis=dict(range=y_range_mttr), yaxis2=dict(range=y_range_mtbf), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig, use_container_width=True)

# --- Formatting and Calculation Helpers ---
def format_minutes_to_dhm(total_minutes):
    if pd.isna(total_minutes) or total_minutes < 0: return "N/A"
    total_minutes = int(total_minutes); days = total_minutes // (60 * 24); remaining_minutes = total_minutes % (60 * 24); hours = remaining_minutes // 60; minutes = remaining_minutes % 60; parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def format_duration(seconds):
    if pd.isna(seconds) or seconds < 0: return "N/A"
    return format_minutes_to_dhm(seconds / 60)

def calculate_daily_summaries_for_week(df_week, tolerance, downtime_gap_tolerance, analysis_mode):
    daily_results_list = []
    for date_val in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date_val]
        if not df_day.empty: calc = RunRateCalculator(df_day.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode); res = calc.results; summary = {'date': date_val, 'stability_index': res.get('stability_index', np.nan), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}; daily_results_list.append(summary)
    return pd.DataFrame(daily_results_list) if daily_results_list else pd.DataFrame()

def calculate_weekly_summaries_for_month(df_month, tolerance, downtime_gap_tolerance, analysis_mode):
    weekly_results_list = []
    for week in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week]
        if not df_week.empty: calc = RunRateCalculator(df_week.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode); res = calc.results; summary = {'week': week, 'stability_index': res.get('stability_index', np.nan), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}; weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance, downtime_gap_tolerance):
    run_summary_list = []
    if 'run_label' not in df_period.columns: return pd.DataFrame() # Need run_label
    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty: calc = RunRateCalculator(df_run.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate'); res = calc.results; summary = {'run_label': run_label, 'start_time': df_run['shot_time'].min(), 'end_time': df_run['shot_time'].max(),'total_shots': res.get('total_shots', 0), 'normal_shots': res.get('normal_shots', 0), 'stopped_shots': res.get('total_shots', 0) - res.get('normal_shots', 0), 'mode_ct': res.get('mode_ct', 0), 'lower_limit': res.get('lower_limit', 0), 'upper_limit': res.get('upper_limit', 0), 'total_runtime_sec': res.get('total_runtime_sec', 0), 'production_time_sec': res.get('production_time_sec', 0), 'downtime_sec': res.get('downtime_sec', 0), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stability_index': res.get('stability_index', np.nan), 'stops': res.get('stop_events', 0)}; run_summary_list.append(summary)
    if not run_summary_list: return pd.DataFrame()
    return pd.DataFrame(run_summary_list).sort_values('start_time').reset_index(drop=True)

# --- Analysis Engine Functions ---
def generate_detailed_analysis(analysis_df, overall_stability, overall_mttr, overall_mtbf, analysis_level):
    insights = {"overall": "N/A", "predictive": "N/A", "best_worst": "N/A", "patterns": "", "recommendation": "N/A", "error": None}
    if analysis_df is None or analysis_df.empty or 'stability' not in analysis_df.columns or analysis_df['stability'].isna().all():
        insights["error"] = "Not enough data/stability values for analysis."; return insights
    try:
        stability_class = "good (>70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (<50%)"
        insights["overall"] = f"Overall stability: <strong>{overall_stability:.1f}%</strong> ({stability_class})."
        if len(analysis_df) > 1:
            volatility_std = analysis_df['stability'].std(); volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"
            half_point = len(analysis_df) // 2; first_half_mean = analysis_df['stability'].iloc[:half_point].mean(); second_half_mean = analysis_df['stability'].iloc[half_point:].mean()
            trend_direction = "stable";
            if pd.notna(first_half_mean) and pd.notna(second_half_mean):
                if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
                elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"
            insights["predictive"] = f"Performance trend: <strong>{trend_direction}</strong>. Volatility: <strong>{volatility_level}</strong>."
        else: insights["predictive"] = "Not enough data points for trend analysis."

        if not analysis_df.empty and 'stops' in analysis_df.columns and 'period' in analysis_df.columns :
            analysis_df['stability'] = pd.to_numeric(analysis_df['stability'], errors='coerce')
            analysis_df.dropna(subset=['stability'], inplace=True)
            if not analysis_df.empty:
                best_performer = analysis_df.loc[analysis_df['stability'].idxmax()]; worst_performer = analysis_df.loc[analysis_df['stability'].idxmin()]
                def format_period(p, l):
                    if isinstance(p, (pd.Timestamp, datetime)): return pd.to_datetime(p).strftime('%A, %b %d')
                    elif isinstance(p, date): return p.strftime('%A, %b %d')
                    # Adjusted condition to handle week numbers correctly for specified levels
                    elif l in ["Monthly (by Run)", "Weekly (by Run)", "Custom Period (by Run)"] and isinstance(p, (int, np.integer)):
                        return f"Week {p}"
                    elif analysis_level == "Daily" and isinstance(p, (int, np.integer, float)): return f"{int(p)}:00" # Ensure it's treated as hour
                    return str(p) # Fallback for run labels etc.

                best_p = format_period(best_performer['period'], analysis_level); worst_p = format_period(worst_performer['period'], analysis_level)
                insights["best_worst"] = (f"Best: <strong>{best_p}</strong> (Stab: {best_performer['stability']:.1f}%, Stops: {int(best_performer.get('stops', 0))}, MTTR: {best_performer.get('mttr', 0):.1f}m). Worst: <strong>{worst_p}</strong> (Stab: {worst_performer['stability']:.1f}%, Stops: {int(worst_performer.get('stops', 0))}, MTTR: {worst_performer.get('mttr', 0):.1f}m).")

                if 'stops' in analysis_df.columns and analysis_df['stops'].sum() > 0:
                    if analysis_level == "Daily":
                        peak_stop_hour = analysis_df.loc[analysis_df['stops'].idxmax()]
                        # Ensure period is treated as integer for formatting
                        insights["patterns"] = f"Stop events peak around <strong>{int(peak_stop_hour['period'])}:00</strong> ({int(peak_stop_hour['stops'])} stops)."
                    else:
                        mean_stab = analysis_df['stability'].mean(); std_stab = analysis_df['stability'].std(); outlier_thresh = mean_stab - (1.5 * std_stab)
                        outliers = analysis_df[analysis_df['stability'] < outlier_thresh]
                        if not outliers.empty:
                            worst_outlier = outliers.loc[outliers['stability'].idxmin()]; outlier_label = format_period(worst_outlier['period'], analysis_level)
                            insights["patterns"] = f"Outlier: <strong>{outlier_label}</strong> performed significantly below average."
            else: insights["best_worst"] = "Could not determine best/worst (missing stability data)."

        overall_mttr_nn = 0 if pd.isna(overall_mttr) else overall_mttr; overall_mtbf_nn = 0 if pd.isna(overall_mtbf) else overall_mtbf
        if overall_stability >= 95: insights["recommendation"] = "Excellent performance. Monitor trends."
        elif overall_stability > 70:
            focus = "<strong>MTBF</strong> (increase uptime)" if (overall_mtbf_nn > 0 and overall_mttr_nn > 0 and overall_mtbf_nn < overall_mttr_nn * 5) else "<strong>MTTR</strong> (reduce stop duration)"
            insights["recommendation"] = f"Good performance. Focus on {focus}. MTBF: {overall_mtbf_nn:.1f}m, MTTR: {overall_mttr_nn:.1f}m."
        else:
            driver = "Low <strong>MTBF</strong> (frequent stops)" if (overall_mtbf_nn > 0 and overall_mttr_nn > 0 and overall_mtbf_nn < overall_mttr_nn) else "High <strong>MTTR</strong> (long stops)"
            insights["recommendation"] = f"Poor stability. Primary driver: {driver}. MTBF: {overall_mtbf_nn:.1f}m, MTTR: {overall_mttr_nn:.1f}m. Investigate root cause."
    except KeyError as e: insights["error"] = f"Analysis error: Missing key {e}"; insights.update({k: "Error" for k in insights if k != "error"})
    # --- Catch TypeError specifically ---
    except TypeError as e: insights["error"] = f"Analysis error: Type mismatch during calculation ({e})"; insights.update({k: "Error" for k in insights if k != "error"})
    except Exception as e: insights["error"] = f"Unexpected analysis error: {e}"; insights.update({k: "Error" for k in insights if k != "error"})
    return insights

def generate_bucket_analysis(complete_runs, bucket_labels):
    if complete_runs.empty or 'duration_min' not in complete_runs.columns: return "No completed runs to analyze."
    total_runs = len(complete_runs); long_buckets = []
    if bucket_labels and isinstance(bucket_labels, (list, tuple, pd.Series)):
        try: long_buckets = [label for label in bucket_labels if int(str(label).split('-')[0].replace('+', '')) >= 60]
        except (ValueError, IndexError, TypeError): pass
    num_long_runs = 0
    if long_buckets and 'time_bucket' in complete_runs.columns:
        if pd.api.types.is_categorical_dtype(complete_runs['time_bucket']):
            # Ensure categories exist before filtering
            valid_categories = [cat for cat in long_buckets if cat in complete_runs['time_bucket'].cat.categories]
            if valid_categories:
                num_long_runs = complete_runs[complete_runs['time_bucket'].isin(valid_categories)].shape[0]
        elif pd.api.types.is_object_dtype(complete_runs['time_bucket']): # Handle if it's object type
            try:
                # Check if long_buckets exist in the actual data values
                present_buckets = [b for b in long_buckets if b in complete_runs['time_bucket'].unique()]
                if present_buckets:
                    num_long_runs = complete_runs[complete_runs['time_bucket'].isin(present_buckets)].shape[0]
            except TypeError: pass # Ignore if isin fails due to mixed types potentially
        # Add a check for other types just in case, though less likely
        elif 'time_bucket' in complete_runs.columns:
            try:
                present_buckets = [b for b in long_buckets if b in complete_runs['time_bucket'].unique()]
                if present_buckets:
                    num_long_runs = complete_runs[complete_runs['time_bucket'].isin(present_buckets)].shape[0]
            except Exception: pass # General catch-all

    pct_long = (num_long_runs / total_runs * 100) if total_runs > 0 else 0; longest_run = "N/A"
    # Ensure duration_min is numeric before finding max
    if not complete_runs.empty and 'duration_min' in complete_runs.columns:
        numeric_durations = pd.to_numeric(complete_runs['duration_min'], errors='coerce')
        longest_run_min_val = numeric_durations.max()
        if pd.notna(longest_run_min_val):
            longest_run = format_minutes_to_dhm(longest_run_min_val)

    analysis = f"<strong>{total_runs}</strong> completed runs. <strong>{num_long_runs}</strong> ({pct_long:.1f}%) >60 min. Longest: <strong>{longest_run}</strong>. "
    if total_runs > 0:
        if pct_long < 20: analysis += "Suggests frequent interruptions."
        elif pct_long > 50: analysis += "Indicates strong sustained operation."
        else: analysis += "Shows mixed performance."
    return analysis

def generate_mttr_mtbf_analysis(analysis_df, analysis_level):
    if analysis_df is None or analysis_df.empty or 'stops' not in analysis_df or analysis_df['stops'].sum()==0 or len(analysis_df)<2 or 'stability' not in analysis_df or 'mttr' not in analysis_df: return "Not enough data for correlation."
    analysis_df['stops'] = pd.to_numeric(analysis_df['stops'], errors='coerce')
    analysis_df['stability'] = pd.to_numeric(analysis_df['stability'], errors='coerce')
    analysis_df['mttr'] = pd.to_numeric(analysis_df['mttr'], errors='coerce')
    analysis_df.dropna(subset=['stops', 'stability', 'mttr'], inplace=True)
    if len(analysis_df) < 2: return "Not enough numeric data points for correlation."

    stops_corr = analysis_df['stops'].corr(analysis_df['stability']); mttr_corr = analysis_df['mttr'].corr(analysis_df['stability'])
    corr_insight = ""; primary_driver_freq = False; primary_driver_dur = False
    if not pd.isna(stops_corr) and not pd.isna(mttr_corr):
        if abs(stops_corr) > abs(mttr_corr) * 1.5: primary_driver = "**frequency of stops**"; primary_driver_freq = True
        elif abs(mttr_corr) > abs(stops_corr) * 1.5: primary_driver = "**duration of stops**"; primary_driver_dur = True
        else: primary_driver = "**frequency and duration**"
        corr_insight = f"Analysis suggests <strong>{primary_driver}</strong> most impacts stability."

    example_insight = ""

    # --- Function definition moved inside ---
    def format_p(p, l):
        # More robust type checking
        if isinstance(p, (pd.Timestamp, datetime)): # Check for datetime objects
            return pd.to_datetime(p).strftime('%A, %b %d')
        elif isinstance(p, date): # Check for date objects
            return p.strftime('%A, %b %d')
        elif l in ["Monthly (by Run)", "Weekly (by Run)", "Custom Period (by Run)"]:
             if isinstance(p, (int, np.integer)): # Check if it's a week number
                 return f"Week {p}"
        elif l == "Daily": # Check for hourly
            # Ensure p is treated as a number before formatting
            if isinstance(p, (int, np.integer, float)):
                return f"{int(p)}:00"
        return str(p) # Fallback for Run IDs or other types

    if not analysis_df.empty:
        if primary_driver_freq:
            highest_stops = analysis_df.loc[analysis_df['stops'].idxmax()]
            p_label = format_p(highest_stops['period'], analysis_level)
            example_insight = f"E.g., <strong>{p_label}</strong> had most stops (<strong>{int(highest_stops['stops'])}</strong>). Prioritize root cause."
        elif primary_driver_dur:
            highest_mttr = analysis_df.loc[analysis_df['mttr'].idxmax()]
            p_label = format_p(highest_mttr['period'], analysis_level)
            example_insight = f"E.g., <strong>{p_label}</strong> had longest downtimes (avg <strong>{highest_mttr['mttr']:.1f} min</strong>). Investigate delays."
        else:
            highest_mttr = analysis_df.loc[analysis_df['mttr'].idxmax()]
            p_label = format_p(highest_mttr['period'], analysis_level)
            example_insight = (f"E.g., <strong>{p_label}</strong> had long downtimes (avg <strong>{highest_mttr['mttr']:.1f} min</strong>), showing duration impact.")

    return f"<div style='line-height:1.6;'><p>{corr_insight}</p><p>{example_insight}</p></div>"

# --- Excel Generation Function ---
def generate_excel_report(all_runs_data, tolerance):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format=workbook.add_format({'bold':True,'bg_color':'#002060','font_color':'white','align':'center','valign':'vcenter','border':1});sub_header_format=workbook.add_format({'bold':True,'bg_color':'#C5D9F1','border':1});label_format=workbook.add_format({'bold':True,'align':'left'});percent_format=workbook.add_format({'num_format':'0.0%','border':1});time_format=workbook.add_format({'num_format':'[h]:mm:ss','border':1});mins_format=workbook.add_format({'num_format':'0.00 "min"','border':1});secs_format=workbook.add_format({'num_format':'0.00 "sec"','border':1});data_format=workbook.add_format({'border':1});datetime_format=workbook.add_format({'num_format':'yyyy-mm-dd hh:mm:ss','border':1});error_format=workbook.add_format({'bold':True,'font_color':'red'})
        for run_id,data in all_runs_data.items():
            ws=workbook.add_worksheet(f"Run_{run_id:03d}");df_run=data['processed_df'].copy();start_row=19
            col_map={name:chr(ord('A')+i)for i,name in enumerate(df_run.columns)};shot_time_col_dyn=col_map.get('SHOT TIME');stop_col=col_map.get('STOP');stop_event_col=col_map.get('STOP EVENT');time_bucket_col=col_map.get('TIME BUCKET');first_col_for_count=shot_time_col_dyn if shot_time_col_dyn else'A';cum_count_col_dyn=col_map.get('CUMULATIVE COUNT');run_dur_col_dyn=col_map.get('RUN DURATION');bucket_col_dyn=col_map.get('TIME BUCKET');time_diff_col_dyn=col_map.get('TIME DIFF SEC')
            data_cols_count=len(df_run.columns);helper_col_letter=chr(ord('A')+data_cols_count);ws.set_column(f'{helper_col_letter}:{helper_col_letter}',None,None,{'hidden':True});analysis_start_col_idx=data_cols_count+2;analysis_col_1=chr(ord('A')+analysis_start_col_idx);analysis_col_2=chr(ord('A')+analysis_start_col_idx+1);analysis_col_3=chr(ord('A')+analysis_start_col_idx+2)
            missing_cols=[col for col,letter in[('STOP',stop_col),('STOP EVENT',stop_event_col),('TIME BUCKET',time_bucket_col),('TIME DIFF SEC',time_diff_col_dyn),('CUMULATIVE COUNT',cum_count_col_dyn),('RUN DURATION',run_dur_col_dyn),('SHOT TIME',shot_time_col_dyn)]if letter is None]
            if missing_cols:ws.write('A5',f"Error: Missing columns: {', '.join(missing_cols)}",error_format)
            table_formulas_ok=not missing_cols
            ws.merge_range('A1:B1',data['equipment_code'],header_format);ws.write('A2','Date',label_format);ws.write('B2',f"{data['start_time']:%Y-%m-%d} to {data['end_time']:%Y-%m-%d}");ws.write('A3','Method',label_format);ws.write('B3','Every Shot');ws.write('E1','Mode CT',sub_header_format);mode_ct_val=data.get('mode_ct',0);ws.write('E2',mode_ct_val if isinstance(mode_ct_val,(int,float))else 0,secs_format);ws.write('F1','Outside L1',sub_header_format);ws.write('G1','Outside L2',sub_header_format);ws.write('H1','IDLE',sub_header_format);ws.write('F2','Lower Limit',label_format);ws.write('G2','Upper Limit',label_format);ws.write('H2','Stops',label_format)
            lower_limit_val=data.get('lower_limit');upper_limit_val=data.get('upper_limit');ws.write('F3',lower_limit_val if lower_limit_val is not None else'N/A',secs_format);ws.write('G3',upper_limit_val if upper_limit_val is not None else'N/A',secs_format)
            if stop_col:ws.write_formula('H3',f"=SUM({stop_col}{start_row}:{stop_col}{start_row+len(df_run)-1})",sub_header_format)
            else:ws.write('H3','N/A',sub_header_format)
            ws.write('K1','Total Shot Count',label_format);ws.write('L1','Normal Shot Count',label_format);ws.write_formula('K2',f"=COUNTA({first_col_for_count}{start_row}:{first_col_for_count}{start_row+len(df_run)-1})",sub_header_format);ws.write_formula('L2',f"=K2-H3",sub_header_format);ws.write('K4','Efficiency',label_format);ws.write('L4','Stop Events',label_format);ws.write_formula('K5',f"=L2/K2",percent_format)
            if stop_event_col:ws.write_formula('L5',f"=SUM({stop_event_col}{start_row}:{stop_event_col}{start_row+len(df_run)-1})",sub_header_format)
            else:ws.write('L5','N/A',sub_header_format)
            ws.write('F5','Tot Run Time (Calc)',label_format);ws.write('G5','Tot Down Time',label_format);ws.write('H5','Tot Prod Time',label_format)
            ws.write('F6',data.get('total_runtime_sec',0)/86400,time_format);ws.write('G6',data.get('tot_down_time_sec',0)/86400,time_format);ws.write('H6',data.get('production_time_sec',0)/86400,time_format)
            ws.write_formula('F7',f"=IFERROR(H6/F6, 0)",percent_format); ws.write_formula('G7',f"=IFERROR(G6/F6, 0)",percent_format); ws.write('H7','',data_format) # Added IFERROR
            ws.merge_range('K8:L8','Reliability Metrics',header_format);ws.write('K9','MTTR (Avg)',label_format);ws.write('L9',data.get('mttr_min',0),mins_format);ws.write('K10','MTBF (Avg)',label_format);ws.write('L10',data.get('mtbf_min',0),mins_format);ws.write('K11','Time to First DT',label_format);ws.write('L11',data.get('time_to_first_dt_min',0),mins_format);ws.write('K12','Avg Cycle Time',label_format);ws.write('L12',data.get('avg_cycle_time_sec',0),secs_format)
            ws.merge_range(f'{analysis_col_1}14:{analysis_col_3}14','Time Bucket Analysis',header_format);ws.write(f'{analysis_col_1}15','Bucket',sub_header_format);ws.write(f'{analysis_col_2}15','Duration Range',sub_header_format);ws.write(f'{analysis_col_3}15','Events Count',sub_header_format);max_bucket=20
            for i in range(1,max_bucket+1):
                ws.write(f'{analysis_col_1}{15+i}',i,sub_header_format);ws.write(f'{analysis_col_2}{15+i}',f"{(i-1)*20} - {i*20} min",sub_header_format)
                # --- EXCEL FORMULA FIX for Time Bucket Count ---
                # Check if time_bucket_col exists before writing formula
                if time_bucket_col:
                    # Correct COUNTIF: Compare numeric bucket index (i) with the TIME BUCKET column
                    ws.write_formula(f'{analysis_col_3}{15+i}',f'=COUNTIF({bucket_col_dyn}{start_row}:{bucket_col_dyn}{start_row + len(df_run) - 1},{i})',sub_header_format)
                else:
                    ws.write(f'{analysis_col_3}{15+i}','N/A',sub_header_format)
            ws.write(f'{analysis_col_2}{16+max_bucket}','Grand Total',sub_header_format);ws.write_formula(f'{analysis_col_3}{16+max_bucket}',f"=SUM({analysis_col_3}16:{analysis_col_3}{15+max_bucket})",sub_header_format)
            ws.write_row('A18',df_run.columns,header_format)
            if'SHOT TIME'in df_run.columns:df_run['SHOT TIME']=pd.to_datetime(df_run['SHOT TIME'],errors='coerce').dt.tz_localize(None)
            df_run.fillna('',inplace=True)
            for i,row_values in enumerate(df_run.to_numpy()):
                current_row_excel_idx=start_row+i
                for c_idx,value in enumerate(row_values):
                    col_name=df_run.columns[c_idx]
                    # --- Skip writing raw values for formula columns in Excel ---
                    if col_name in['CUMULATIVE COUNT','RUN DURATION','TIME BUCKET','TIME DIFF SEC']:continue
                    cell_format=data_format
                    if isinstance(value,pd.Timestamp):
                        if pd.notna(value):ws.write_datetime(current_row_excel_idx-1,c_idx,value,datetime_format)
                        else:ws.write_string(current_row_excel_idx-1,c_idx,'',data_format)
                    elif isinstance(value,(bool,np.bool_)):ws.write_number(current_row_excel_idx-1,c_idx,int(value),data_format)
                    elif isinstance(value,(int,float,np.number)):
                        if col_name in['ACTUAL CT','adj_ct_sec']:cell_format=secs_format
                        if np.isfinite(value): ws.write_number(current_row_excel_idx-1,c_idx,value,cell_format)
                        else: ws.write_string(current_row_excel_idx-1, c_idx, '', data_format) # Handle NaN/Inf
                    else:ws.write(current_row_excel_idx-1,c_idx,value,data_format) # Write strings, etc.

            if table_formulas_ok:
                # --- Get column indices safely ---
                time_diff_col_idx = df_run.columns.get_loc('TIME DIFF SEC') if 'TIME DIFF SEC' in df_run.columns else -1
                cum_count_col_idx = df_run.columns.get_loc('CUMULATIVE COUNT') if 'CUMULATIVE COUNT' in df_run.columns else -1
                run_dur_col_idx = df_run.columns.get_loc('RUN DURATION') if 'RUN DURATION' in df_run.columns else -1
                bucket_col_idx = df_run.columns.get_loc('TIME BUCKET') if 'TIME BUCKET' in df_run.columns else -1

                # --- Check if all necessary indices were found ---
                if -1 in [time_diff_col_idx, cum_count_col_idx, run_dur_col_idx, bucket_col_idx]:
                    ws.write('A6', "Error: Could not find all columns needed for formulas.", error_format)
                else:
                    for i in range(len(df_run)):
                        row_num=start_row+i;prev_row=row_num-1

                        # Helper column formula (calculates cumulative time within a run)
                        if i==0:helper_formula=f'=IF({stop_col}{row_num}=0,{time_diff_col_dyn}{row_num},0)'
                        else:helper_formula=f'=IF({stop_event_col}{row_num}=1,0,IF({stop_col}{row_num}=0,{time_diff_col_dyn}{row_num},{helper_col_letter}{prev_row}))' # Corrected helper logic
                        ws.write_formula(f'{helper_col_letter}{row_num}',helper_formula) # Write helper formula for current row

                        # Time Diff Sec (Excel calculation)
                        if i==0:
                            # Use the pre-calculated value for the first row (or 0 if not available)
                            first_diff = data['processed_df_original']['time_diff_sec'].iloc[0] if 'processed_df_original' in data and not data['processed_df_original'].empty else 0
                            ws.write_number(row_num-1, time_diff_col_idx, first_diff, secs_format)
                        else:
                            formula=f'=IFERROR(({shot_time_col_dyn}{row_num}-{shot_time_col_dyn}{prev_row})*86400, 0)'; # Calculate diff, default to 0 on error
                            ws.write_formula(row_num-1,time_diff_col_idx,formula,secs_format)

                        # Cumulative Count / Run Duration String
                        cum_count_formula=f'=COUNTIF(${stop_event_col}${start_row}:${stop_event_col}{row_num},TRUE)&"/"&IF({stop_event_col}{row_num}=TRUE,"0 sec",TEXT({helper_col_letter}{row_num}/86400,"[h]:mm:ss"))'
                        ws.write_formula(row_num-1,cum_count_col_idx,cum_count_formula,data_format)

                        # Run Duration (Numeric, formatted as time)
                        # This should show the duration of the *completed* run ending *before* the current stop event
                        run_dur_formula = f'=IF({stop_event_col}{row_num}=TRUE, IF({row_num}>{start_row}, {helper_col_letter}{prev_row}/86400, 0), "")'
                        ws.write_formula(row_num-1,run_dur_col_idx,run_dur_formula,time_format)

                        # Time Bucket (Numeric index of the bucket)
                        time_bucket_formula = f'=IF({stop_event_col}{row_num}=TRUE, IF({row_num}>{start_row}, IFERROR(FLOOR({helper_col_letter}{prev_row}/60/20,1)+1, ""), ""), "")'
                        ws.write_formula(row_num-1,bucket_col_idx,time_bucket_formula,data_format) # Write as number, format later if needed

            else: # If essential columns for formulas were missing
                if cum_count_col_dyn: ws.write(f'{cum_count_col_dyn}{start_row}',"Formula Error",error_format)
                if time_diff_col_dyn: ws.write(f'{time_diff_col_dyn}{start_row}',"Formula Error",error_format)
                if run_dur_col_dyn: ws.write(f'{run_dur_col_dyn}{start_row}', "Formula Error", error_format)
                if bucket_col_dyn: ws.write(f'{bucket_col_dyn}{start_row}', "Formula Error", error_format)

            # Auto-adjust column widths
            for i,col_name in enumerate(df_run.columns):
                # Calculate max width needed for header and data
                try:
                    # Convert potential numeric/datetime data to string for length check, handle NaN safely
                    max_len_data = df_run[col_name].astype(str).map(len).max()
                    # Ensure max_len_data is a number, default to 0 if NaN
                    max_len_data = 0 if pd.isna(max_len_data) else max_len_data
                    # Compare with header length
                    width = max(len(str(col_name)), int(max_len_data))
                except Exception: # Fallback if error during length calculation
                    width=len(str(col_name))
                # Apply width limit and padding
                ws.set_column(i,i,min(width + 2, 40)) # Limit max width to 40

    return output.getvalue()


# --- Wrapper Function (Uses UNIFIED RunRateCalculator) ---
def generate_run_based_excel_export(df_for_export, tolerance, downtime_gap_tolerance, run_interval_hours, tool_id_selection):
    try:
        # Use main calculator for base processing AND run splitting
        base_calc = RunRateCalculator(df_for_export, tolerance, downtime_gap_tolerance, analysis_mode='aggregate') # Use aggregate initially
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())

        if df_processed.empty:
            st.error("Could not process data for Excel export. Check input data.")
            return BytesIO().getvalue()

        # --- Split into runs using time_diff_sec ---
        split_col = 'time_diff_sec'
        if split_col not in df_processed.columns:
            st.error(f"'{split_col}' column not found after initial processing. Cannot split into runs.")
            return BytesIO().getvalue()

        is_new_run = df_processed[split_col] > (run_interval_hours * 3600)
        # Ensure run_id starts from 1 correctly, handle potential empty is_new_run Series
        if not is_new_run.empty:
            # cumsum() creates groups, add 1 to start from 1.
            # The logic `+ (0 if is_new_run.iloc[0] else 1)` was potentially flawed.
            # A simpler approach: cumsum starts groups from 0, so add 1.
            df_processed['run_id'] = is_new_run.cumsum() + 1
        else: # Handle case with only one row or no time differences
            df_processed['run_id'] = 1

    except Exception as e:
        st.error(f"Error during initial data processing or run splitting for Excel: {e}")
        st.text(traceback.format_exc())
        return BytesIO().getvalue()


    all_runs_data = {}
    # Define base columns + formula placeholders
    # Ensure 'tool_id' is included if available in the original data
    desired_columns_base = ['SUPPLIER NAME', 'tool_id', 'SESSION ID', 'SHOT ID', 'shot_time','APPROVED CT', 'ACTUAL CT']
    # Columns calculated by RunRateCalculator (will be present after recalculation)
    calculated_cols = ['time_diff_sec', 'stop_flag', 'stop_event', 'run_group', 'adj_ct_sec'] # adj_ct_sec needed for metrics
    formula_columns = ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET'] # Placeholders for Excel formulas

    # Final columns in desired Excel order (match generate_excel_report function)
    final_desired_renamed = ['SUPPLIER NAME', 'EQUIPMENT CODE', 'SESSION ID', 'SHOT ID', 'SHOT TIME','APPROVED CT', 'ACTUAL CT', 'TIME DIFF SEC', 'STOP', 'STOP EVENT', 'run_group','CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET']

    # Loop through runs identified by run_id
    for run_id, df_run_raw in df_processed.groupby('run_id'):
        try:
            # --- Recalculate THIS run using the SAME unified logic ---
            # Use 'aggregate' mode here as each run is treated independently for its metrics/limits
            run_calculator = RunRateCalculator(df_run_raw.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
            run_results = run_calculator.results

            if not run_results or 'processed_df' not in run_results or run_results['processed_df'].empty:
                st.warning(f"Skipping empty/invalid Run ID {run_id} for Excel.")
                continue

            # This is the DataFrame with calculated cols (stop_flag, etc.) for *this run*
            export_df = run_results['processed_df'].copy()
            # Store original time_diff_sec for first row handling in Excel export
            run_results['processed_df_original'] = df_run_raw[['time_diff_sec']].copy()


            # --- Add general run info to the results dict (for Excel header) ---
            run_results['equipment_code'] = df_run_raw['tool_id'].iloc[0] if 'tool_id' in df_run_raw.columns and not df_run_raw['tool_id'].empty else tool_id_selection
            run_results['start_time'] = df_run_raw['shot_time'].min()
            run_results['end_time'] = df_run_raw['shot_time'].max()
            # Add scalar lower/upper limits calculated for this specific run
            run_results['lower_limit'] = run_results.get('lower_limit', 0) # From recalculation
            run_results['upper_limit'] = run_results.get('upper_limit', np.inf) # From recalculation

            # --- Prepare the DataFrame specific for this Excel sheet ---
            # Select base columns that exist in the original raw data for this run
            cols_to_keep = [col for col in desired_columns_base if col in df_run_raw.columns]
            final_export_df = df_run_raw[cols_to_keep].copy()

            # Merge calculated columns from the re-processed run_results['processed_df']
            calculated_cols_to_merge = [col for col in calculated_cols if col in export_df.columns]
            final_export_df = final_export_df.merge(export_df[calculated_cols_to_merge], left_index=True, right_index=True, how='left')


            # Add placeholders for formula columns
            for col in formula_columns:
                final_export_df[col] = '' # Initialize as empty string

            # Rename columns for Excel
            final_export_df.rename(columns={'tool_id': 'EQUIPMENT CODE', 'shot_time': 'SHOT TIME','time_diff_sec': 'TIME DIFF SEC', 'stop_flag': 'STOP', 'stop_event': 'STOP EVENT'}, inplace=True)

            # Ensure all final desired columns exist, adding blanks if necessary, and reorder
            for col in final_desired_renamed:
                if col not in final_export_df.columns: final_export_df[col] = ''
            final_export_df = final_export_df[[col for col in final_desired_renamed if col in final_export_df.columns]] # Reorder

            # Store the prepared DataFrame back into the results dict for this run
            run_results['processed_df'] = final_export_df
            all_runs_data[run_id] = run_results # Add run results to the main dict

        except Exception as e:
            st.warning(f"Could not process Run ID {run_id} for Excel: {e}")
            st.text(traceback.format_exc())
            continue # Skip to the next run

    if not all_runs_data:
        st.error("No valid runs were processed for the Excel export.")
        return BytesIO().getvalue()

    # Generate the Excel file using the dedicated reporting function
    try:
        excel_data = generate_excel_report(all_runs_data, tolerance)
        return excel_data
    except Exception as e:
        st.error(f"Error generating the final Excel file: {e}")
        st.text(traceback.format_exc())
        return BytesIO().getvalue()


# --- Dashboard Rendering Function ---
def render_dashboard(df_tool, tool_id_selection):
    st.sidebar.title("Dashboard Controls ⚙️")

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
        - **Daily**: Hourly trends (Aggregate).
        - **Weekly / Monthly / Custom (by Run)**: Tolerance based on Mode CT of each run. New run after 'Run Interval Threshold' idle time. Shows Daily/Weekly trends respectively.
        ---
        ### Sliders
        - **Tolerance Band**: Defines acceptable CT range (±% of Mode CT).
        - **Downtime Gap Tolerance**: Min idle sec between shots to count as a separate stop.
        - **Run Interval Threshold**: Max hours idle before a new Production Run starts.
        - **Remove Runs...**: (Only in 'by Run' mode) Filters out runs with fewer shots.
        """)

    # --- Analysis Level Options ---
    # Simplified options based on the second script's preference
    analysis_options = ["Daily", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"]
    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        options=analysis_options
    )

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Defines the max hours between shots before a new Production Run is identified.")

    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours, tol, gap_tol):
        try:
            # Use main calculator, pass all params for initial processing
            base_calc = RunRateCalculator(df, tol, gap_tol, analysis_mode='aggregate') # Use aggregate mode initially
            df_processed = base_calc.results.get("processed_df", pd.DataFrame())
            if not df_processed.empty:
                df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
                df_processed['date'] = df_processed['shot_time'].dt.date
                df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
                # Base run identification
                is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600)
                # Ensure correct run_id start, handle edge case of single row df
                if not is_new_run.empty:
                    # Corrected run_id assignment: cumsum() + 1
                    df_processed['run_id'] = is_new_run.cumsum() + 1
                else:
                    df_processed['run_id'] = 1 # Assign run_id 1 if only one row
            return df_processed
        except Exception as e:
            st.error(f"Error during initial data processing cache: {e}")
            st.text(traceback.format_exc())
            return pd.DataFrame()


    df_processed = get_processed_data(df_tool.copy(), run_interval_hours, tolerance, downtime_gap_tolerance)

    min_shots_filter = 1
    if 'by Run' in analysis_level: # Filter slider only appears for 'by Run' modes
        st.sidebar.markdown("---")
        if not df_processed.empty and 'run_id' in df_processed.columns:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if pd.notna(run_shot_counts.max()) else 1 # Handle potential NaN max
                max_shots = max(1, max_shots) # Ensure max_value is at least 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                # Ensure value <= max_value
                min_shots_filter = st.sidebar.slider("Remove Runs with Fewer Than X Shots", 1, max_shots, min(default_value, max_shots), 1, help="Filters out smaller production runs.")


    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)

    if df_processed.empty: st.error(f"Could not process data for {tool_id_selection}. Check format/range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")
    # Determine mode based on selection
    mode = 'by_run' if 'by Run' in analysis_level else 'aggregate'
    df_view = pd.DataFrame() # This will hold the data subset for the selected period

    # --- Period Selection Logic ---
    if analysis_level == "Daily":
        st.header("Daily Analysis")
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates: st.warning("No data available for any date."); st.stop()
        selected_date = st.selectbox("Select Date", available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"
    elif "Weekly (by Run)" in analysis_level:
        st.header("Weekly Analysis (by Production Run)")
        available_weeks = sorted(df_processed["week"].unique()); year = df_processed['shot_time'].iloc[0].year
        if not available_weeks: st.warning("No data available for any week."); st.stop()
        selected_week = st.selectbox(f"Select Week (Year {year})", available_weeks, index=len(available_weeks)-1)
        df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"
    elif "Monthly (by Run)" in analysis_level:
        st.header("Monthly Analysis (by Production Run)")
        available_months = sorted(df_processed["month"].unique())
        if not available_months: st.warning("No data available for any month."); st.stop()
        selected_month = st.selectbox("Select Month", available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'))
        df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"
    elif "Custom Period (by Run)" in analysis_level:
        st.header("Custom Period Analysis (by Production Run)")
        min_d, max_d = df_processed['date'].min(), df_processed['date'].max()
        start_date = st.date_input("Start date", min_d, min_value=min_d, max_value=max_d)
        end_date = st.date_input("End date", max_d, min_value=start_date, max_value=max_d)
        if start_date and end_date:
            df_view = df_processed[(df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    # --- Apply Run Filtering and Labeling ---
    if not df_view.empty:
        df_view = df_view.copy()
        if 'run_id' in df_view.columns:
            if 'by Run' in analysis_level: # Apply filter only in 'by Run' modes
                runs_before = df_view['run_id'].nunique()
                # Use transform to get counts per run and filter
                run_shot_counts = df_view.groupby('run_id')['run_id'].transform('count')
                df_view = df_view[run_shot_counts >= min_shots_filter].copy() # Ensure it's a copy after filtering
                runs_after = df_view['run_id'].nunique()
                if runs_before > 0: st.sidebar.metric("Runs Displayed", f"{runs_after} / {runs_before}", f"-{runs_before - runs_after} filtered", delta_color="off")
            # --- Relabel runs AFTER filtering ---
            if not df_view.empty: # Add labels if data remains
                # Get unique run_ids present *in the filtered view*, sorted by time
                unique_run_ids_in_view = df_view.sort_values('shot_time')['run_id'].unique()
                run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids_in_view)}
                # Apply the map based on the original run_id to create the label
                df_view['run_label'] = df_view['run_id'].map(run_label_map)


    if df_view.empty: st.warning(f"No data for selected period/filters."); st.stop()
    else:
        # --- Calculate Final Metrics (using unified RunRateCalculator) ---
        # Pass the potentially filtered df_view
        calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
        results = calc.results

        # Ensure calculation was successful and add run_label back if needed
        if not results or 'processed_df' not in results or results['processed_df'].empty:
            st.error("Calculation failed or resulted in empty data after filtering."); st.stop()
        else:
             # Add the run_label back to the processed df if it exists in df_view
            if 'run_label' in df_view.columns:
                results['processed_df'] = results['processed_df'].merge(
                    df_view[['run_label']].drop_duplicates(subset=['run_label'], keep='first'), # Use drop_duplicates if merging causes issues
                    left_index=True,
                    right_index=True, # Align on index if possible
                    how='left'
                )


        summary_metrics = results # Use results directly

        # Adjust subheader for combined runs if in 'by_run' mode
        if mode == 'by_run': sub_header = sub_header.replace("Summary for", "Summary (Combined Runs)")

        # --- Display Section ---
        col1, col2 = st.columns([3, 1]);
        with col1:
            st.subheader(sub_header)
        with col2:
            # --- UPDATED DOWNLOAD BUTTON ---
            st.download_button(
                label="📥 Export Run-Based Report",
                data=generate_run_based_excel_export(df_view.copy(), tolerance, downtime_gap_tolerance, run_interval_hours, tool_id_selection),
                file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            # --- END OF UPDATE ---

        # --- Trend Summary Calculation ---
        trend_summary_df = None
        run_summary_df_for_trends = None # Initialize

        # Calculate summaries based on the potentially filtered df_view
        if "Weekly (by Run)" in analysis_level:
            trend_summary_df = calculate_daily_summaries_for_week(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
        elif "Monthly (by Run)" in analysis_level:
            trend_summary_df = calculate_weekly_summaries_for_month(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
        elif "Custom Period (by Run)" in analysis_level:
            time_span_days = (df_view['date'].max() - df_view['date'].min()).days
            if time_span_days > 14: trend_summary_df = calculate_weekly_summaries_for_month(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
            else: trend_summary_df = calculate_daily_summaries_for_week(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
        elif analysis_level == "Daily":
            # Hourly summary is already in 'results' from the main calculation on df_view
            trend_summary_df = results.get('hourly_summary', pd.DataFrame())

        # Calculate run summary *once* if in a 'by Run' mode (using filtered df_view)
        if "by Run" in analysis_level:
            # Pass the filtered df_view to calculate summaries only for displayed runs
            run_summary_df_for_trends = calculate_run_summaries(df_view.copy(), tolerance, downtime_gap_tolerance)
            if run_summary_df_for_trends is not None and not run_summary_df_for_trends.empty:
                # Rename columns for plotting and table display consistency
                run_summary_df_for_trends.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True, errors='ignore')


        # --- Metric Display Containers ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0)
            prod_t = summary_metrics.get('production_time_sec', 0)
            down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            with col1: st.metric("MTTR", f"{summary_metrics.get('mttr_min', 0):.1f} min")
            with col2: st.metric("MTBF", f"{summary_metrics.get('mtbf_min', 0):.1f} min")
            with col3: st.metric("Total Run Duration", format_duration(total_d)) # Uptime + Downtime
            with col4: st.metric("Production Time", f"{format_duration(prod_t)}"); st.markdown(f'<span style="background-color:{PASTEL_COLORS["green"]};color:#0E1117;padding:3px 8px;border-radius:10px;font-size:0.8rem;font-weight:bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5: st.metric("Downtime", f"{format_duration(down_t)}"); st.markdown(f'<span style="background-color:{PASTEL_COLORS["red"]};color:#0E1117;padding:3px 8px;border-radius:10px;font-size:0.8rem;font-weight:bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_gauge(summary_metrics.get('efficiency', 0) * 100, "Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': PASTEL_COLORS['red']}, {'range': [50, 70], 'color': PASTEL_COLORS['orange']},{'range': [70, 100], 'color': PASTEL_COLORS['green']}]
            c2.plotly_chart(create_gauge(summary_metrics.get('stability_index', 0), "Stability Index (%)", steps=steps), use_container_width=True)
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0); s_s = t_s - n_s
            n_p = (n_s / t_s * 100) if t_s > 0 else 0; s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2: st.metric("Normal Shots", f"{n_s:,}"); st.markdown(f'<span style="background-color:{PASTEL_COLORS["green"]};color:#0E1117;padding:3px 8px;border-radius:10px;font-size:0.8rem;font-weight:bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3: st.metric("Stop Events", f"{summary_metrics.get('stop_events', 0)}"); st.markdown(f'<span style="background-color:{PASTEL_COLORS["red"]};color:#0E1117;padding:3px 8px;border-radius:10px;font-size:0.8rem;font-weight:bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            # Display limits based on calculated results (could be scalar or range)
            if mode == 'by_run' and 'min_lower_limit' in summary_metrics: # Check if by_run range limits exist
                min_ll=summary_metrics.get('min_lower_limit', 0); max_ll=summary_metrics.get('max_lower_limit', 0);
                min_mc=summary_metrics.get('min_mode_ct', 0); max_mc=summary_metrics.get('max_mode_ct', 0);
                min_ul=summary_metrics.get('min_upper_limit', 0); max_ul=summary_metrics.get('max_upper_limit', 0)
                c1.metric("Lower Limit (sec)", f"{min_ll:.2f} – {max_ll:.2f}" if min_ll != max_ll else f"{min_ll:.2f}")
                with c2:
                    with st.container(border=True):
                        st.metric("Mode CT (sec)", f"{min_mc:.2f} – {max_mc:.2f}" if min_mc != max_mc else f"{min_mc:.2f}")
                c3.metric("Upper Limit (sec)", f"{min_ul:.2f} – {max_ul:.2f}" if min_ul != max_ul else f"{min_ul:.2f}")
            else: # Aggregate Mode or fallback
                mode_val = summary_metrics.get('mode_ct', "N/A"); # Use N/A if not calculated
                mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else str(mode_val)
                c1.metric("Lower Limit (sec)", f"{summary_metrics.get('lower_limit', 0):.2f}") # Default to 0 if missing
                with c2:
                    with st.container(border=True):
                        st.metric("Mode CT (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{summary_metrics.get('upper_limit', np.inf):.2f}") # Default to inf if missing

        # --- Detailed Analysis HTML ---
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame() # Prepare df for analysis function

                # Determine which df to use for analysis based on mode/level
                if analysis_level == "Daily":
                    if trend_summary_df is not None and not trend_summary_df.empty:
                        analysis_df = trend_summary_df.copy()
                        analysis_df.rename(columns={'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}, inplace=True, errors='ignore')
                elif "by Run" in analysis_level:
                    # Use run summary directly for analysis in 'by Run' mode
                    if run_summary_df_for_trends is not None and not run_summary_df_for_trends.empty:
                        analysis_df = run_summary_df_for_trends.copy()
                        # Use the renamed columns from the run summary
                        analysis_df.rename(columns={'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}, inplace=True, errors='ignore')
                    # If run summary failed, try the daily/weekly trend as fallback (less ideal for run-level analysis)
                    elif trend_summary_df is not None and not trend_summary_df.empty:
                        analysis_df = trend_summary_df.copy()
                        if 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                        elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                        else: rename_map = {} # Should not happen if date/week exist
                        analysis_df.rename(columns=rename_map, inplace=True, errors='ignore')

                # Pass aggregated metrics calculated earlier
                insights = generate_detailed_analysis(analysis_df, summary_metrics.get('stability_index', 0), summary_metrics.get('mttr_min', 0), summary_metrics.get('mtbf_min', 0), analysis_level)

                if insights.get("error"):
                    st.error(insights["error"])
                else:
                    patterns_html = f'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> {insights.get("patterns", "")}</p>' if insights.get("patterns") else ''
                    html_content = f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights.get("overall", "N/A")}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights.get("predictive", "N/A")}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights.get("best_worst", "N/A")}</p>{patterns_html}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights.get("recommendation", "N/A")}</p></div>"""
                    components.html(html_content, height=400, scrolling=True)

        # --- Breakdown Tables ---
        # Show daily/weekly table *only* if trend_summary_df represents daily or weekly data
        if trend_summary_df is not None and not trend_summary_df.empty and ('date' in trend_summary_df.columns or 'week' in trend_summary_df.columns):
            table_title = "View Daily Breakdown Table" if 'date' in trend_summary_df.columns else "View Weekly Breakdown Table"
            with st.expander(table_title, expanded=False):
                d_df = trend_summary_df.copy()
                if 'date' in d_df.columns:
                    d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                    d_df.rename(columns={'date': 'Day', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops', 'total_shots': 'Total Shots'}, inplace=True)
                    cols_to_show = [col for col in ['Day', 'Stability (%)', 'MTTR (min)', 'MTBF (min)', 'Stops', 'Total Shots'] if col in d_df.columns]
                elif 'week' in d_df.columns:
                    d_df.rename(columns={'week': 'Week', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops', 'total_shots': 'Total Shots'}, inplace=True)
                    cols_to_show = [col for col in ['Week', 'Stability (%)', 'MTTR (min)', 'MTBF (min)', 'Stops', 'Total Shots'] if col in d_df.columns]
                else:
                    cols_to_show = [] # Should not happen based on outer condition

                if cols_to_show:
                    st.dataframe(d_df[cols_to_show].style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)


        # --- Run Breakdown Table ---
        # Show this table *whenever* a 'by Run' mode is selected and the run summary exists
        if "by Run" in analysis_level and run_summary_df_for_trends is not None and not run_summary_df_for_trends.empty:
            with st.expander("View Run Breakdown Table", expanded=False):
                d_df = run_summary_df_for_trends.copy() # Use the already calculated and renamed df
                # Ensure start/end times are datetime objects before formatting
                d_df['start_time'] = pd.to_datetime(d_df['start_time'], errors='coerce')
                d_df['end_time'] = pd.to_datetime(d_df['end_time'], errors='coerce')
                d_df = d_df.dropna(subset=['start_time', 'end_time'])

                if not d_df.empty:
                    # Safely convert 'Total Shots' back to numeric for calculation, handle errors
                    d_df['Total Shots Numeric'] = pd.to_numeric(d_df['Total Shots'], errors='coerce').fillna(0).astype(int)

                    d_df["Period"] = d_df.apply(lambda r: f"{r['start_time']:%Y-%m-%d %H:%M} to {r['end_time']:%Y-%m-%d %H:%M}", axis=1)
                    # --- FIX: Calculate percentage using the numeric column ---
                    d_df["Normal Shots (%)"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['Total Shots Numeric']*100:.1f}%)" if r['Total Shots Numeric']>0 else "0 (0.0%)", axis=1)
                    d_df["Stops (%)"] = d_df.apply(lambda r: f"{r['STOPS']} ({r['stopped_shots']/r['Total Shots Numeric']*100:.1f}%)" if r['Total Shots Numeric']>0 else "0 (0.0%)", axis=1)
                    # --- FIX: Format Total Shots after calculation ---
                    d_df["Total Shots Formatted"] = d_df['Total Shots Numeric'].apply(lambda x: f"{x:,}")

                    d_df["Total Duration"] = d_df['total_runtime_sec'].apply(format_duration)
                    d_df["Prod. Time (%)"] = d_df.apply(lambda r: f"{format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (%)"] = d_df.apply(lambda r: f"{format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)

                    # Adjust rename map and columns list for display
                    d_df.rename(columns={'RUN ID':'Run ID','mode_ct':'Mode CT','lower_limit':'LL','upper_limit':'UL','MTTR (min)':'MTTR','MTBF (min)':'MTBF','STABILITY %':'Stability %', 'Total Shots Formatted': 'Total Shots'}, inplace=True, errors='ignore')
                    cols = ['Run ID','Period','Total Shots','Normal Shots (%)','Stops (%)','Mode CT','LL','UL','Total Duration','Prod. Time (%)','Downtime (%)','MTTR','MTBF','Stability %']
                    cols_exist = [c for c in cols if c in d_df.columns]
                    st.dataframe(d_df[cols_exist].style.format({'Mode CT':'{:.2f}','LL':'{:.2f}','UL':'{:.2f}','MTTR':'{:.1f}','MTBF':'{:.1f}','Stability %':'{:.1f}'}), use_container_width=True)


        # --- Plots ---
        # Pass the processed dataframe from the main results for plotting shots
        plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=('hourly' if analysis_level == "Daily" else 'daily' if 'Weekly' in analysis_level else 'weekly'))

        with st.expander("View Shot Data Table", expanded=False):
            display_cols = ['shot_time', 'run_label', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event']
            # Select only columns that actually exist in the processed dataframe
            display_cols_exist = [col for col in display_cols if col in results['processed_df'].columns]
            st.dataframe(results['processed_df'][display_cols_exist])


        st.markdown("---")

        # --- Trend Plot Section ---
        if analysis_level == "Daily":
            st.header("Hourly Analysis")
            # Hourly data is in trend_summary_df
            hourly_summary = trend_summary_df

            run_durations_period=results.get("run_durations",pd.DataFrame());processed_period_df=results.get('processed_df',pd.DataFrame());stop_events_df=processed_period_df.loc[processed_period_df['stop_event']].copy();complete_runs=pd.DataFrame()
            if not stop_events_df.empty and 'run_group' in stop_events_df.columns and 'shot_time' in stop_events_df.columns and not run_durations_period.empty and 'run_group' in run_durations_period.columns:
                try:
                    stop_events_df['terminated_run_group']=stop_events_df['run_group']-1;
                    end_time_map=stop_events_df.set_index('terminated_run_group')['shot_time'];
                    run_durations_period['run_end_time']=run_durations_period['run_group'].map(end_time_map);
                    complete_runs=run_durations_period.dropna(subset=['run_end_time']).copy()
                except KeyError as e:
                    st.warning(f"Could not map run end times for bucket analysis: Missing key {e}")
                    complete_runs = pd.DataFrame() # Ensure it's empty if mapping fails
            else: complete_runs = pd.DataFrame() # Ensure it's empty if insufficient data

            c1,c2=st.columns(2)
            with c1:
                if not complete_runs.empty and "time_bucket" in complete_runs.columns and "bucket_labels" in results and results["bucket_labels"]:
                    # Ensure time_bucket is categorical with all potential labels
                    complete_runs["time_bucket"] = pd.Categorical(complete_runs["time_bucket"], categories=results["bucket_labels"], ordered=True)
                    b_counts=complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"],fill_value=0);
                    if not b_counts.empty:
                        fig_b=px.bar(b_counts,title="Time Bucket Analysis (Completed Runs)",labels={"index":"Duration (min)","value":"Occurrences"},text_auto=True,color=b_counts.index,color_discrete_map=results.get("bucket_color_map",{})).update_layout(legend_title_text='Duration');st.plotly_chart(fig_b,use_container_width=True);
                        with st.expander("View Bucket Data"):
                            st.dataframe(complete_runs)
                    else: st.info("No bucket counts to plot.")
                else:
                    st.info("No complete runs or bucket labels for bucket analysis.")
            with c2:
                if hourly_summary is not None and not hourly_summary.empty:
                    plot_trend_chart(hourly_summary,'hour','stability_index',"Hourly Stability Trend","Hour of Day","Stability (%)",is_stability=True);
                    with st.expander("View Stability Data"):
                        st.dataframe(hourly_summary[['hour','stability_index','mttr_min','mtbf_min','stops','total_shots']])
                else: st.info("No hourly stability data.")

            if not complete_runs.empty and 'run_end_time' in complete_runs and 'time_bucket' in complete_runs.columns and "bucket_labels" in results and results["bucket_labels"]:
                st.subheader("Hourly Bucket Trend");
                try:
                    complete_runs['hour']=complete_runs['run_end_time'].dt.hour;
                    # Ensure time_bucket is categorical
                    complete_runs["time_bucket"] = pd.Categorical(complete_runs["time_bucket"], categories=results["bucket_labels"], ordered=True)
                    pivot_df=pd.crosstab(index=complete_runs['hour'],columns=complete_runs['time_bucket'], dropna=False); # Use dropna=False
                    pivot_df=pivot_df.reindex(pd.Index(range(24),name='hour'),fill_value=0); # Reindex hours 0-23
                    # Ensure all expected bucket labels are columns
                    pivot_df = pivot_df.reindex(columns=results["bucket_labels"], fill_value=0)

                    if not pivot_df.empty:
                        fig_hourly_bucket=px.bar(pivot_df,x=pivot_df.index,y=pivot_df.columns,title='Hourly Distribution of Run Durations',barmode='stack',color_discrete_map=results.get("bucket_color_map",{}),labels={'hour':'Hour of Stop','value':'Number of Runs','variable':'Run Duration (min)'});st.plotly_chart(fig_hourly_bucket,use_container_width=True);
                        with st.expander("View Bucket Trend Data"):
                            st.dataframe(pivot_df)
                        if detailed_view:
                            with st.expander("🤖 View Bucket Trend Analysis"):
                                st.markdown(generate_bucket_analysis(complete_runs,results["bucket_labels"]),unsafe_allow_html=True)
                    else: st.info("Could not create pivot table for hourly bucket trend.")
                except Exception as e:
                    st.warning(f"Error generating hourly bucket trend: {e}")
            else: st.info("Not enough data for hourly bucket trend.")

            st.subheader("Hourly MTTR & MTBF Trend");
            if hourly_summary is not None and not hourly_summary.empty:
                plot_mttr_mtbf_chart(df=hourly_summary,x_col='hour',mttr_col='mttr_min',mtbf_col='mtbf_min',shots_col='total_shots',title="Hourly MTTR, MTBF & Shot Count Trend");
                with st.expander("View MTTR/MTBF Data"):
                    st.dataframe(hourly_summary)
            if detailed_view and hourly_summary is not None and not hourly_summary.empty and hourly_summary['stops'].sum()>0:
                with st.expander("🤖 View MTTR/MTBF Correlation Analysis"):
                    st.info("""**How this works:** Frequency vs Duration analysis.""");
                    analysis_df=hourly_summary.copy().rename(columns={'hour':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'});
                    st.markdown(generate_mttr_mtbf_analysis(analysis_df,analysis_level),unsafe_allow_html=True)

        elif "by Run" in analysis_level:
            # Determine trend level based on what summary was generated
            if trend_summary_df is not None and 'date' in trend_summary_df.columns: trend_level = "Daily"
            elif trend_summary_df is not None and 'week' in trend_summary_df.columns: trend_level = "Weekly"
            else: trend_level = "Run" # Default or if only run summary exists

            summary_df = trend_summary_df # This is daily/weekly summary if applicable
            # run_summary_df_for_trends was calculated earlier

            st.header(f"{trend_level} Trends for {analysis_level.split(' (')[0]}" if trend_level != "Run" else "Run-Based Analysis")

            # Bucket Analysis (Overall for the period)
            run_durations_period=results.get("run_durations",pd.DataFrame());processed_period_df=results.get('processed_df',pd.DataFrame());stop_events_df=processed_period_df.loc[processed_period_df['stop_event']].copy();complete_runs=pd.DataFrame()
            # --- Robust mapping for complete runs ---
            if not stop_events_df.empty and 'run_group' in stop_events_df.columns and 'shot_time' in stop_events_df.columns and not run_durations_period.empty and 'run_group' in run_durations_period.columns:
                try:
                    stop_events_df['terminated_run_group']=stop_events_df['run_group']-1;
                    end_time_map=stop_events_df.set_index('terminated_run_group')['shot_time'];
                    run_durations_period['run_end_time']=run_durations_period['run_group'].map(end_time_map);
                    complete_runs=run_durations_period.dropna(subset=['run_end_time']).copy()
                    # --- Add run_label to complete_runs ---
                    if 'run_label' in processed_period_df.columns and 'run_group' in processed_period_df.columns:
                        run_group_to_label = processed_period_df.drop_duplicates('run_group')[['run_group', 'run_label']].set_index('run_group')
                        complete_runs = complete_runs.merge(run_group_to_label, on='run_group', how='left')
                except KeyError as e:
                    st.warning(f"Could not map run end times or labels for bucket analysis: Missing key {e}")
                    complete_runs = pd.DataFrame() # Ensure empty on error
            else: complete_runs = pd.DataFrame() # Ensure empty if insufficient data


            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Total Bucket Analysis (Across Runs/Period)")
                if not complete_runs.empty and "time_bucket" in complete_runs.columns and "bucket_labels" in results and results["bucket_labels"]:
                    # Ensure time_bucket is categorical
                    complete_runs["time_bucket"] = pd.Categorical(complete_runs["time_bucket"], categories=results["bucket_labels"], ordered=True)
                    b_counts=complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"],fill_value=0);
                    if not b_counts.empty:
                        fig_b=px.bar(b_counts,title="Total Time Bucket Analysis",labels={"index":"Duration(min)","value":"Occurrences"},text_auto=True,color=b_counts.index,color_discrete_map=results.get("bucket_color_map",{})).update_layout(legend_title_text='Duration');st.plotly_chart(fig_b,use_container_width=True);
                        with st.expander("View Bucket Data"):
                            st.dataframe(complete_runs)
                    else: st.info("No bucket counts to plot.")
                else:
                    st.info("No complete runs or bucket data.")
            with c2: # Stability Trend
                if trend_level == "Run":
                    st.subheader("Stability per Production Run")
                    if run_summary_df_for_trends is not None and not run_summary_df_for_trends.empty:
                        plot_trend_chart(run_summary_df_for_trends,'RUN ID','STABILITY %',"Stability per Run","Run ID","Stability (%)",is_stability=True)
                        with st.expander("View Stability Data"):
                            st.dataframe(run_summary_df_for_trends)
                    else:
                        st.info(f"No runs to analyze.")
                else: # Daily or Weekly Trend
                    st.subheader(f"{trend_level} Stability Trend")
                    if summary_df is not None and not summary_df.empty:
                        x_col='date'if trend_level=="Daily"else'week';
                        plot_trend_chart(summary_df,x_col,'stability_index',f"{trend_level} Stability Trend",trend_level,"Stability (%)",is_stability=True);
                        with st.expander("View Stability Data"):
                            st.dataframe(summary_df)
                    else:
                        st.info(f"No {trend_level.lower()} stability data.")

            # Bucket Trend per Unit (Run, Daily, or Weekly)
            if not complete_runs.empty and ('run_end_time' in complete_runs.columns or trend_level == "Run") and 'time_bucket' in complete_runs.columns and "bucket_labels" in results and results["bucket_labels"]:
                st.subheader(f"{trend_level} Bucket Trend")
                pivot_df = pd.DataFrame()
                trend_df_for_shots = None # This will hold the df used for the secondary axis (shots)
                x_axis_title = trend_level

                try:
                    # Ensure time_bucket is categorical before crosstab
                    complete_runs["time_bucket"] = pd.Categorical(complete_runs["time_bucket"], categories=results["bucket_labels"], ordered=True)

                    if trend_level == "Run":
                        # Use run_label which should exist now on complete_runs
                        if 'run_label' in complete_runs.columns and run_summary_df_for_trends is not None and 'RUN ID' in run_summary_df_for_trends.columns:
                            # Use only runs present in the filtered summary
                            valid_run_labels = run_summary_df_for_trends['RUN ID'].unique()
                            complete_runs_filtered = complete_runs[complete_runs['run_label'].isin(valid_run_labels)]
                            if not complete_runs_filtered.empty:
                                pivot_df = pd.crosstab(index=complete_runs_filtered['run_label'],columns=complete_runs_filtered['time_bucket'], dropna=False).reindex(valid_run_labels, fill_value=0) # Index by Run ID
                                trend_df_for_shots = run_summary_df_for_trends.set_index('RUN ID').reindex(valid_run_labels) # Align shots data
                        x_axis_title = "Run ID"
                    else: # Daily or Weekly
                        time_col = 'date' if trend_level == "Daily" else 'week'
                        # Ensure the time column exists from the previous calculation
                        if time_col not in complete_runs.columns and 'run_end_time' in complete_runs.columns:
                            complete_runs[time_col] = complete_runs['run_end_time'].dt.date if trend_level == "Daily" else complete_runs['run_end_time'].dt.isocalendar().week

                        if time_col in complete_runs.columns and summary_df is not None and time_col in summary_df.columns:
                            all_units = sorted(summary_df[time_col].unique()) # Use units from the summary df
                            pivot_df = pd.crosstab(index=complete_runs[time_col],columns=complete_runs['time_bucket'], dropna=False).reindex(all_units, fill_value=0).sort_index()
                            trend_df_for_shots = summary_df.set_index(time_col).reindex(all_units).sort_index() # Align shots data
                        x_axis_title = trend_level

                    # Ensure all bucket labels are columns after potential filtering/reindexing
                    if not pivot_df.empty:
                        pivot_df = pivot_df.reindex(columns=results["bucket_labels"], fill_value=0)


                    if not pivot_df.empty and trend_df_for_shots is not None:
                        fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                        for col in pivot_df.columns: fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results.get("bucket_color_map",{}).get(col)), secondary_y=False)

                        shots_col_name = 'Total Shots' if trend_level=='Run' else 'total_shots'
                        if shots_col_name in trend_df_for_shots.columns:
                            shots_data = pd.to_numeric(trend_df_for_shots[shots_col_name], errors='coerce').fillna(0) # Ensure numeric
                            fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=trend_df_for_shots.index, y=shots_data, mode='lines+markers+text', text=shots_data.astype(int), textposition='top center', line=dict(color='blue')), secondary_y=True)

                        fig_bucket_trend.update_layout(barmode='stack', title_text=f'{trend_level} Distribution of Run Durations vs. Shot Count', xaxis_title=x_axis_title, yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                        st.plotly_chart(fig_bucket_trend, use_container_width=True)
                        with st.expander("View Bucket Trend Data"): st.dataframe(pivot_df)
                        if detailed_view:
                            with st.expander("🤖 View Bucket Trend Analysis"):
                                st.markdown(generate_bucket_analysis(complete_runs_filtered if trend_level=='Run' else complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
                    else: st.info(f"Not enough combined data for {trend_level} bucket trend vs shots.")
                except Exception as e:
                    st.warning(f"Error generating {trend_level} bucket trend: {e}")
                    st.text(traceback.format_exc()) # Show traceback for debugging
            else: st.info(f"Not enough data for {trend_level} bucket trend.")

            # MTTR/MTBF Trend per Unit
            st.subheader(f"{trend_level} MTTR & MTBF Trend")
            trend_df_for_mttr = run_summary_df_for_trends if trend_level == "Run" else summary_df # Use correct df based on trend level
            if trend_df_for_mttr is not None and not trend_df_for_mttr.empty:
                x_col = 'RUN ID' if trend_level == "Run" else ('date' if trend_level == "Daily" else 'week')
                mttr_col = 'MTTR (min)' if trend_level == "Run" else 'mttr_min'
                mtbf_col = 'MTBF (min)' if trend_level == "Run" else 'mtbf_min'
                shots_col = 'Total Shots' if trend_level == "Run" else 'total_shots'
                stops_col = 'STOPS' if trend_level == "Run" else 'stops' # Use STOPS for run summary, stops otherwise

                # Check if essential columns exist
                if all(c in trend_df_for_mttr.columns for c in [x_col, mttr_col, mtbf_col, shots_col, stops_col]):
                    # Ensure stops data is numeric before summing
                    stops_numeric = pd.to_numeric(trend_df_for_mttr[stops_col], errors='coerce').fillna(0)
                    if stops_numeric.sum() > 0:
                        plot_mttr_mtbf_chart(df=trend_df_for_mttr, x_col=x_col, mttr_col=mttr_col, mtbf_col=mtbf_col, shots_col=shots_col, title=f"{trend_level} MTTR, MTBF & Shot Count Trend")
                        with st.expander("View MTTR/MTBF Data"): st.dataframe(trend_df_for_mttr)
                        if detailed_view:
                            with st.expander("🤖 View MTTR/MTBF Correlation Analysis"):
                                st.info("""**How this works:** Frequency vs Duration analysis.""");
                                analysis_df=trend_df_for_mttr.copy();
                                # Define rename map based on trend level
                                if trend_level=="Run": rename_map={'RUN ID':'period','STABILITY %':'stability','STOPS':'stops','MTTR (min)':'mttr'}
                                elif trend_level=="Daily": rename_map={'date':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'}
                                else: rename_map={'week':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'} # Weekly
                                analysis_df.rename(columns=rename_map, inplace=True, errors='ignore');
                                st.markdown(generate_mttr_mtbf_analysis(analysis_df,analysis_level),unsafe_allow_html=True)
                    else: st.info(f"No stops recorded for {trend_level} MTTR/MTBF trend.")
                else: st.info(f"Missing columns needed for {trend_level} MTTR/MTBF trend.")
            else: st.info(f"Not enough data for {trend_level} MTTR/MTBF trend.")


# --- Risk Tower Functions ---
@st.cache_data(show_spinner="Analyzing tool performance for Risk Tower...")
def calculate_risk_scores(df_all_tools):
    id_col="tool_id";initial_metrics=[];default_tol,default_gap=0.05,2.0
    for tool_id,df_tool in df_all_tools.groupby(id_col):
        if df_tool.empty or len(df_tool)<10:continue
        try:
            # Prepare data first to get accurate date range
            calc_prepare=RunRateCalculator(df_tool,default_tol,default_gap, analysis_mode='aggregate');
            df_prepared=calc_prepare.results.get("processed_df");
            if df_prepared is None or df_prepared.empty:continue

            # Get the date range from the prepared data
            end_date=df_prepared['shot_time'].max();
            if pd.isna(end_date): continue # Skip if no valid end date
            start_date=end_date-timedelta(weeks=4);

            # Filter the PREPARED data for the 4-week period
            df_period=df_prepared[(df_prepared['shot_time']>=start_date)&(df_prepared['shot_time']<=end_date)];

            if df_period.empty or len(df_period)<10:continue

            # Recalculate metrics specifically for the 4-week period
            calc=RunRateCalculator(df_period.copy(),default_tol,default_gap, analysis_mode='aggregate');
            res=calc.results;
            if not res:continue

            # Calculate weekly stability within the 4-week period
            df_period_copy = df_period.copy() # Avoid SettingWithCopyWarning
            df_period_copy['week']=df_period_copy['shot_time'].dt.isocalendar().week;
            weekly_stabilities = []
            for _,df_week in df_period_copy.groupby('week'):
                if not df_week.empty:
                    week_calc = RunRateCalculator(df_week,default_tol,default_gap, analysis_mode='aggregate')
                    weekly_stabilities.append(week_calc.results.get('stability_index', 0))

            trend="Stable";
            if len(weekly_stabilities)>1:
                # Check for NaN/None before comparison
                first_stab = weekly_stabilities[0]
                last_stab = weekly_stabilities[-1]
                if pd.notna(first_stab) and pd.notna(last_stab) and last_stab < first_stab * 0.95:
                    trend="Declining"

            initial_metrics.append({'Tool ID':tool_id,'Stability':res.get('stability_index',0),'MTTR':res.get('mttr_min',0),'MTBF':res.get('mtbf_min',0),'Weekly Stability':' → '.join([f'{s:.0f}%'for s in weekly_stabilities if pd.notna(s)]),'Trend':trend,'Analysis Period':f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"})
        except Exception as e:st.warning(f"Could not calculate risk score for {tool_id}: {e}") # Keep warning

    if not initial_metrics:return pd.DataFrame()

    metrics_df=pd.DataFrame(initial_metrics);
    # Calculate means only from valid numeric values
    overall_mttr_mean=pd.to_numeric(metrics_df['MTTR'], errors='coerce').mean();
    overall_mtbf_mean=pd.to_numeric(metrics_df['MTBF'], errors='coerce').mean();
    final_risk_data=[]

    for _,row in metrics_df.iterrows():
        # Ensure stability is numeric
        stability_val = pd.to_numeric(row['Stability'], errors='coerce')
        if pd.isna(stability_val): stability_val = 0 # Default to 0 if not numeric

        risk_score=stability_val;
        details=f"Overall stability: {stability_val:.1f}%."; # Default detail
        primary_factor="Low Stability" # Default factor

        if row['Trend']=="Declining":risk_score-=20;primary_factor="Declining Trend";details="Stability shows downward trend."
        # Check MTTR/MTBF only if Stability is below threshold AND trend isn't declining
        elif stability_val < 70:
            mttr_val = pd.to_numeric(row['MTTR'], errors='coerce')
            mtbf_val = pd.to_numeric(row['MTBF'], errors='coerce')

            # Check against valid means
            if pd.notna(mttr_val) and pd.notna(overall_mttr_mean) and mttr_val>(overall_mttr_mean*1.2):
                primary_factor="High MTTR";details=f"Avg stop duration (MTTR) {mttr_val:.1f} min concern."
            elif pd.notna(mtbf_val) and pd.notna(overall_mtbf_mean) and mtbf_val<(overall_mtbf_mean*0.8):
                primary_factor="Frequent Stops";details=f"Frequent stops (MTBF {mtbf_val:.1f} min) impacting."
            # If MTTR/MTBF aren't the primary issue but stability is low, keep default
            elif stability_val < 70 and primary_factor == "Low Stability":
                details = f"Overall stability ({stability_val:.1f}%) is low, but MTTR/MTBF not flagged as primary drivers relative to average."


        final_risk_data.append({'Tool ID':row['Tool ID'],'Analysis Period':row['Analysis Period'],'Risk Score':max(0,risk_score),'Primary Risk Factor':primary_factor,'Weekly Stability':row['Weekly Stability'],'Details':details})

    if not final_risk_data:return pd.DataFrame()
    return pd.DataFrame(final_risk_data).sort_values('Risk Score',ascending=True).reset_index(drop=True)

def render_risk_tower(df_all_tools):
    st.title("Run Rate Risk Tower");st.info("Analyzes last 4 weeks. Lowest scores = highest risk.")
    with st.expander("ℹ️ How the Risk Tower Works"):st.markdown("""**Analysis Period**: 4-week range/tool. **Risk Score**: Stability Index (%), -20 for declining trend. **Primary Risk Factor**: Declining > High MTTR (>1.2x avg) > Frequent Stops (MTBF <0.8x avg) > Low Stability. **Color**: <span style='background-color:#ff6961;color:black;padding:2px 5px;border-radius:5px;'>Red (0-50)</span>,<span style='background-color:#ffb347;color:black;padding:2px 5px;border-radius:5px;'>Orange (51-70)</span>,<span style='background-color:#77dd77;color:black;padding:2px 5px;border-radius:5px;'>Green (>70)</span>.""",unsafe_allow_html=True)
    risk_df=calculate_risk_scores(df_all_tools)
    if risk_df.empty:st.warning("Not enough data for Risk Tower.");return
    def style_risk(row):
        score = pd.to_numeric(row['Risk Score'], errors='coerce') # Ensure score is numeric
        if pd.isna(score): color = '#808080' # Grey for NaN scores
        elif score > 70: color = PASTEL_COLORS['green']
        elif score > 50: color = PASTEL_COLORS['orange']
        else: color = PASTEL_COLORS['red']
        return[f'background-color: {color}'for _ in row]
    st.dataframe(risk_df.style.apply(style_risk,axis=1).format({'Risk Score':'{:.0f}'}),use_container_width=True,hide_index=True)


# --- Main App Structure ---
st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload one or more Run Rate Excel files", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploaded_files: st.info("👈 Upload one or more Excel files."); st.stop()

@st.cache_data
def load_all_data(files):
    df_list = []
    required_time_cols_alt1 = {"YEAR", "MONTH", "DAY", "TIME"}
    required_time_cols_alt2 = {"SHOT TIME"}

    for file in files:
        df = None
        try:
            df = pd.read_excel(file)
            has_id = False
            # Standardize tool ID column name robustly
            id_col_found = None
            if "TOOLING ID" in df.columns: id_col_found = "TOOLING ID"
            elif "EQUIPMENT CODE" in df.columns: id_col_found = "EQUIPMENT CODE"
            elif "tool_id" in df.columns: id_col_found = "tool_id"

            if id_col_found:
                df.rename(columns={id_col_found: "tool_id"}, inplace=True)
                has_id = True
            else:
                st.warning(f"Skipping {file.name}: Missing required Tool ID column (TOOLING ID, EQUIPMENT CODE, or tool_id).")
                continue # Skip this file

            # Check for time columns
            has_time = ("SHOT TIME" in df.columns) or (required_time_cols_alt1.issubset(df.columns))

            if has_time:
                # Ensure tool_id is string type before appending
                if 'tool_id' in df.columns:
                    df['tool_id'] = df['tool_id'].astype(str)
                df_list.append(df)
            else:
                st.warning(f"Skipping {file.name}: Missing required Time columns (SHOT TIME or YEAR/MONTH/DAY/TIME).")

        except Exception as e:
            st.warning(f"Could not load or process file: {file.name}. Error: {e}")
            st.text(traceback.format_exc()) # Show traceback for loading errors

    if not df_list: return pd.DataFrame()
    try:
        # Concatenate valid dataframes
        combined_df = pd.concat(df_list, ignore_index=True)
        # Final check for tool_id column after concat
        if "tool_id" not in combined_df.columns:
            st.error("Tool ID column ('tool_id') missing after combining files.")
            return pd.DataFrame()
        # Ensure tool_id is consistently string type after concat
        combined_df['tool_id'] = combined_df['tool_id'].astype(str)
        return combined_df
    except Exception as e:
        st.error(f"Error combining loaded files: {e}")
        st.text(traceback.format_exc())
        return pd.DataFrame()


df_all_tools = load_all_data(uploaded_files)
id_col = "tool_id" # Standardized name

if df_all_tools.empty:
    st.error("No valid data loaded from the uploaded files. Please check file contents and required columns.")
    st.stop()

# This check might be redundant due to checks in load_all_data, but kept for safety
if id_col not in df_all_tools.columns:
    st.error("Tool ID column ('tool_id') missing after combining files. This should not happen.")
    st.stop()

# Clean data: drop rows where tool_id is NaN/None *before* getting unique values
df_all_tools.dropna(subset=[id_col], inplace=True)

# Ensure tool_id is string type AGAIN after potential NaNs dropped (belt-and-suspenders)
df_all_tools[id_col] = df_all_tools[id_col].astype(str)

unique_tool_ids = sorted(df_all_tools[id_col].unique().tolist())

if not unique_tool_ids: st.error("No valid tools found after cleaning data."); st.stop()


tool_ids_options = ["All Tools (Risk Tower)"] + unique_tool_ids
dashboard_tool_id_selection = st.sidebar.selectbox("Select Tool ID for Dashboard Analysis", tool_ids_options)

df_for_dashboard = pd.DataFrame() # Initialize
tool_id_for_dashboard_display = "No Tool Selected" # Default

if dashboard_tool_id_selection == "All Tools (Risk Tower)":
    if unique_tool_ids: # Check if there are any tools
        first_tool = unique_tool_ids[0];
        # Ensure filtering uses the standardized 'tool_id' column
        df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool].copy();
        tool_id_for_dashboard_display = first_tool
    # No else needed, df_for_dashboard remains empty if no unique_tool_ids
else:
    # Ensure filtering uses the standardized 'tool_id' column
    df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection].copy();
    tool_id_for_dashboard_display = dashboard_tool_id_selection

# --- Tab Display ---
tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])
with tab1: render_risk_tower(df_all_tools.copy())
with tab2:
    if not df_for_dashboard.empty:
        render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
    else:
        # Provide more context if no data for the selected/default tool
        if dashboard_tool_id_selection == "All Tools (Risk Tower)" and not unique_tool_ids:
            st.warning("No tools found in the uploaded data to display a default dashboard.")
        elif dashboard_tool_id_selection == "All Tools (Risk Tower)":
            st.info(f"Defaulting dashboard to '{tool_id_for_dashboard_display}'. Select another tool from the sidebar if needed.")
            # Attempt to render dashboard for the default tool even if "All Tools" selected
            render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
        else:
            st.warning(f"No data found for the selected tool '{tool_id_for_dashboard_display}'. Please select another tool or check the uploaded files.")
