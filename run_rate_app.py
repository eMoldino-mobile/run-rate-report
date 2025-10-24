import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import warnings
import streamlit.components.v1 as components
import xlsxwriter
from datetime import datetime, timedelta

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
    # ... (Keep the RunRateCalculator class exactly as it was in the previous correct version) ...
    # No changes needed here as it already provides the necessary metrics.
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.analysis_mode = analysis_mode
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        if "shot_time" not in df.columns:
            if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
                datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
                df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
            elif "SHOT TIME" in df.columns:
                df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
            else: return pd.DataFrame()

        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        if df.empty: return pd.DataFrame()
        df["time_diff_sec"] = df["shot_time"].diff().dt.total_seconds()
        if not df.empty and pd.isna(df.loc[0, "time_diff_sec"]):
            if "ACTUAL CT" in df.columns: df.loc[0, "time_diff_sec"] = df.loc[0, "ACTUAL CT"]
            else: df.loc[0, "time_diff_sec"] = 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'stop_event' not in df.columns: return pd.DataFrame()
        df['hour'] = df['shot_time'].dt.hour; hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        total_downtime_sec = hourly_groups.apply(lambda x: x[x['stop_flag'] == 1]['adj_ct_sec'].sum())
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ACTUAL CT'].sum() / 60
        shots = hourly_groups.size().rename('total_shots')
        hourly_summary = pd.DataFrame(index=range(24)); hourly_summary['hour'] = hourly_summary.index
        hourly_summary = hourly_summary.join(stops.rename('stops')).join(shots).join(uptime_min.rename('uptime_min')).fillna(0)
        hourly_summary = hourly_summary.join(total_downtime_sec.rename('total_downtime_sec')).fillna(0)
        hourly_summary['mttr_min'] = (hourly_summary['total_downtime_sec'] / 60) / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])
        effective_runtime_min = hourly_summary['uptime_min'] + (hourly_summary['total_downtime_sec'] / 60)
        hourly_summary['stability_index'] = np.where(effective_runtime_min > 0,(hourly_summary['uptime_min'] / effective_runtime_min) * 100, np.where(hourly_summary['stops'] == 0, 100.0, 0.0))
        return hourly_summary.fillna(0)

    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or ("ACTUAL CT" not in df.columns and "time_diff_sec" not in df.columns): return {}

        mode_ct = 0; lower_limit = 0; upper_limit = np.inf; mode_ct_display = "N/A"
        if "ACTUAL CT" in df.columns:
            if self.analysis_mode == 'by_run' and 'run_id' in df.columns:
                run_modes = df.groupby('run_id')['ACTUAL CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
                df['mode_ct'] = df['run_id'].map(run_modes); lower_limit = df['mode_ct'] * (1 - self.tolerance); upper_limit = df['mode_ct'] * (1 + self.tolerance)
                df['lower_limit'] = lower_limit; df['upper_limit'] = upper_limit; mode_ct_display = "Varies by Run"
            else:
                df_for_mode_calc = df[df["ACTUAL CT"] < 999.9].copy(); df_for_mode_calc['rounded_ct'] = df_for_mode_calc['ACTUAL CT'].round(0)
                mode_ct = df_for_mode_calc['rounded_ct'].mode().iloc[0] if not df_for_mode_calc['rounded_ct'].mode().empty else 0
                lower_limit = mode_ct * (1 - self.tolerance); upper_limit = mode_ct * (1 + self.tolerance); mode_ct_display = mode_ct
        else: lower_limit = 0; upper_limit = np.inf

        df["stop_flag"] = 0; df["stop_event"] = False
        if "ACTUAL CT" in df.columns:
            is_abnormal_cycle = (df["ACTUAL CT"] < lower_limit) | (df["ACTUAL CT"] > upper_limit)
            prev_actual_ct = df["ACTUAL CT"].shift(1)
            is_downtime_gap = df["time_diff_sec"] > (prev_actual_ct.fillna(0) + self.downtime_gap_tolerance)
            df["stop_flag"] = np.where(is_abnormal_cycle | is_downtime_gap.fillna(False), 1, 0)
        else: df["stop_flag"] = 0

        if not df.empty: df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        if "ACTUAL CT" in df.columns: df["adj_ct_sec"] = np.where(df["stop_flag"] == 1, df["time_diff_sec"], df["ACTUAL CT"])
        else: df["adj_ct_sec"] = df["time_diff_sec"]

        total_shots = len(df); stop_events = df["stop_event"].sum()
        downtime_sec = df.loc[df['stop_flag'] == 1, 'adj_ct_sec'].sum(); mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
        production_time_sec = 0
        if "ACTUAL CT" in df.columns: production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum()
        else: production_time_sec = df.loc[df['stop_flag'] == 0, 'adj_ct_sec'].sum()
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        total_runtime_calc = production_time_sec + downtime_sec
        stability_index = (production_time_sec / total_runtime_calc * 100) if total_runtime_calc > 0 else (100.0 if stop_events == 0 else 0.0)
        normal_shots = total_shots - df["stop_flag"].sum(); efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        run_duration_col = "ACTUAL CT" if "ACTUAL CT" in df.columns else "adj_ct_sec"
        df_for_runs = df[df['adj_ct_sec'] <= (24 * 3600)].copy()
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")[run_duration_col].sum().div(60).reset_index(name="duration_min")

        avg_cycle_time_sec = production_time_sec / normal_shots if normal_shots > 0 else 0
        first_stop_index = df[df['stop_event']].index.min()
        time_to_first_dt_sec = df.loc[:first_stop_index-1, 'adj_ct_sec'].sum() if pd.notna(first_stop_index) and first_stop_index > 0 else production_time_sec
        production_run_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0

        labels = []; bucket_color_map = {}
        if not run_durations.empty:
            max_minutes = min(run_durations["duration_min"].max(), 240); upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
            edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
            labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
            if edges and len(edges) > 1: last_edge_start = edges[-2]; labels[-1] = f"{last_edge_start}+"; edges[-1] = np.inf
            run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False, include_lowest=True)
            reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
            red_labels, blue_labels, green_labels = [], [], []
            for label in labels:
                try:
                    # Attempt to parse the lower bound from the bucket label
                    lower_bound = int(label.split('-')[0].replace('+', ''))
                    # Assign to color groups based on the lower bound
                    if lower_bound < 60: red_labels.append(label)
                    elif 60 <= lower_bound < 160: blue_labels.append(label)
                    else: green_labels.append(label)
                except (ValueError, IndexError):
                    # If parsing fails (e.g., unexpected label format), skip this label
                    continue
                except: continue
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

        if self.analysis_mode == 'by_run' and isinstance(lower_limit, pd.Series) and not df.empty:
            final_results.update({"min_lower_limit": lower_limit.min(), "max_lower_limit": lower_limit.max(), "min_upper_limit": upper_limit.min(), "max_upper_limit": upper_limit.max(), "min_mode_ct": df['mode_ct'].min(), "max_mode_ct": df['mode_ct'].max()})
        elif "ACTUAL CT" in df.columns : final_results.update({"lower_limit": lower_limit, "upper_limit": upper_limit})

        return final_results


# --- UI Helper and Plotting Functions (Unchanged) ---
def create_gauge(value, title, steps=None):
    # ... (code remains the same) ...
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps: gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else: gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20)); return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct, time_agg='hourly'):
    # ... (code remains the same) ...
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
    ll = lower_limit if lower_limit is not None else 0; ul = upper_limit if upper_limit is not None else np.inf
    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        for run_id, group in df.groupby('run_id'):
            if not group.empty: fig.add_shape(type="rect", xref="x", yref="y", x0=group['shot_time'].min(), y0=group['lower_limit'].iloc[0], x1=group['shot_time'].max(), y1=group['upper_limit'].iloc[0], fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0)
    elif ll is not None and ul is not None and ul != np.inf:
        if not df.empty: fig.add_shape(type="rect", xref="x", yref="y", x0=df['shot_time'].min(), y0=ll, x1=df['shot_time'].max(), y1=ul, fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0)
    if 'run_label' in df.columns:
        run_starts = df.groupby('run_label')['shot_time'].min().sort_values();
        for start_time in run_starts.iloc[1:]: fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="purple")
    y_axis_cap_val = 50
    if isinstance(mode_ct, (int, float)) and mode_ct > 0: y_axis_cap_val = mode_ct
    elif 'mode_ct' in df: mean_mode = df['mode_ct'].mean();
    if pd.notna(mean_mode) and mean_mode > 0: y_axis_cap_val = mean_mode
    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)
    fig.update_layout(title="Cycle Time per Shot vs. Tolerance", xaxis_title="Time", yaxis_title="Cycle Time (sec)", yaxis=dict(range=[0, y_axis_cap]), bargap=0.05, xaxis=dict(showgrid=True), showlegend=True, legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_trend_chart(df, x_col, y_col, title, x_title, y_title, y_range=[0, 101], is_stability=False):
    # ... (code remains the same) ...
    if df is None or df.empty or y_col not in df.columns: st.info(f"Not enough data to plot {title}."); return
    fig = go.Figure(); marker_config = {}
    if is_stability: marker_config['color'] = [PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green'] for v in df[y_col]]; marker_config['size'] = 10
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", name=y_title, line=dict(color="black" if is_stability else "royalblue", width=2), marker=marker_config))
    if is_stability:
        for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]: fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1, fillcolor=c, opacity=0.2, line_width=0, layer="below")
    fig.update_layout(title=title, yaxis=dict(title=y_title, range=y_range), xaxis_title=x_title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig, use_container_width=True)

def plot_mttr_mtbf_chart(df, x_col, mttr_col, mtbf_col, shots_col, title):
    # ... (code remains the same) ...
    if df is None or df.empty or shots_col not in df.columns or df[shots_col].sum() == 0: st.info(f"Not enough data to plot {title}."); return
    mttr = df[mttr_col]; mtbf = df[mtbf_col]; shots = df[shots_col]; x_axis = df[x_col]
    max_mttr = np.nanmax(mttr[np.isfinite(mttr)]) if not mttr.empty and any(np.isfinite(mttr)) else 0
    max_mtbf = np.nanmax(mtbf[np.isfinite(mtbf)]) if not mtbf.empty and any(np.isfinite(mtbf)) else 0
    y_range_mttr = [0, max_mttr * 1.15 if max_mttr > 0 else 10]; y_range_mtbf = [0, max_mtbf * 1.15 if max_mtbf > 0 else 10]
    shots_min, shots_max = shots.min(), shots.max()
    if (shots_max - shots_min) == 0: scaled_shots = pd.Series([y_range_mtbf[1] / 2 if y_range_mtbf[1] > 0 else 0.5] * len(shots), index=shots.index)
    else: scaled_shots = (shots - shots_min) / (shots_max - shots_min) * (y_range_mtbf[1] * 0.9)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x_axis, y=mttr, name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mtbf, name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
    fig.add_trace(go.Scatter(x=x_axis, y=scaled_shots, name='Total Shots', mode='lines+markers+text', text=shots, textposition='top center', textfont=dict(color='blue'), line=dict(color='blue', dash='dot')), secondary_y=True)
    fig.update_layout(title_text=title, yaxis_title="MTTR (min)", yaxis2_title="MTBF (min)", yaxis=dict(range=y_range_mttr), yaxis2=dict(range=y_range_mtbf), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig, use_container_width=True)

# --- Formatting and Calculation Helpers (Unchanged) ---
def format_minutes_to_dhm(total_minutes):
    # ... (code remains the same) ...
    if pd.isna(total_minutes) or total_minutes < 0: return "N/A"
    total_minutes = int(total_minutes); days = total_minutes // (60 * 24); remaining_minutes = total_minutes % (60 * 24); hours = remaining_minutes // 60; minutes = remaining_minutes % 60; parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def format_duration(seconds):
    # ... (code remains the same) ...
    if pd.isna(seconds) or seconds < 0: return "N/A"; return format_minutes_to_dhm(seconds / 60)

def calculate_daily_summaries_for_week(df_week, tolerance, downtime_gap_tolerance, analysis_mode):
    # ... (code remains the same) ...
    daily_results_list = []
    for date in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date]
        if not df_day.empty: calc = RunRateCalculator(df_day.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode); res = calc.results; summary = {'date': date, 'stability_index': res.get('stability_index', np.nan), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}; daily_results_list.append(summary)
    return pd.DataFrame(daily_results_list) if daily_results_list else pd.DataFrame()

def calculate_weekly_summaries_for_month(df_month, tolerance, downtime_gap_tolerance, analysis_mode):
    # ... (code remains the same) ...
    weekly_results_list = []
    for week in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week]
        if not df_week.empty: calc = RunRateCalculator(df_week.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode); res = calc.results; summary = {'week': week, 'stability_index': res.get('stability_index', np.nan), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}; weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance, downtime_gap_tolerance):
    # ... (code remains the same) ...
    run_summary_list = []
    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty: calc = RunRateCalculator(df_run.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate'); res = calc.results; summary = {'run_label': run_label, 'start_time': df_run['shot_time'].min(), 'end_time': df_run['shot_time'].max(),'total_shots': res.get('total_shots', 0), 'normal_shots': res.get('normal_shots', 0), 'stopped_shots': res.get('total_shots', 0) - res.get('normal_shots', 0), 'mode_ct': res.get('mode_ct', 0), 'lower_limit': res.get('lower_limit', 0), 'upper_limit': res.get('upper_limit', 0), 'total_runtime_sec': res.get('total_runtime_sec', 0), 'production_time_sec': res.get('production_time_sec', 0), 'downtime_sec': res.get('downtime_sec', 0), 'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan), 'stability_index': res.get('stability_index', np.nan), 'stops': res.get('stop_events', 0)}; run_summary_list.append(summary)
    if not run_summary_list: return pd.DataFrame()
    return pd.DataFrame(run_summary_list).sort_values('start_time').reset_index(drop=True)

# --- Analysis Engine Functions ---
# --- [MODIFIED] generate_detailed_analysis ---
def generate_detailed_analysis(analysis_df, overall_stability, overall_mttr, overall_mtbf, analysis_level):
    # Initialize all keys with default values
    insights = {
        "overall": "N/A", "predictive": "N/A", "best_worst": "N/A",
        "patterns": "", "recommendation": "N/A", "error": None
    }
    if analysis_df is None or analysis_df.empty or 'stability' not in analysis_df.columns or analysis_df['stability'].isna().all():
        insights["error"] = "Not enough data or stability values missing for detailed analysis."
        return insights

    try:
        stability_class = "good (above 70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (below 50%)"
        insights["overall"] = f"Overall stability is <strong>{overall_stability:.1f}%</strong> ({stability_class})."

        if len(analysis_df) > 1:
            volatility_std = analysis_df['stability'].std(); volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"
            half_point = len(analysis_df) // 2; first_half_mean = analysis_df['stability'].iloc[:half_point].mean(); second_half_mean = analysis_df['stability'].iloc[half_point:].mean()
            trend_direction = "stable";
            if pd.notna(first_half_mean) and pd.notna(second_half_mean):
                 if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
                 elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"
            insights["predictive"] = f"Performance shows a <strong>{trend_direction} trend</strong> and has been <strong>{volatility_level}</strong>."
        else:
             insights["predictive"] = "Not enough data points for trend analysis."


        if not analysis_df.empty and 'stops' in analysis_df.columns: # Check stops column exists
            best_performer = analysis_df.loc[analysis_df['stability'].idxmax()]; worst_performer = analysis_df.loc[analysis_df['stability'].idxmin()]
            def format_period(p, l):
                if isinstance(p, (pd.Timestamp, pd.Period, pd.Timedelta)): return pd.to_datetime(p).strftime('%A, %b %d')
                if l == "Monthly": return f"Week {p}"
                if "Daily" in l: return f"{p}:00"
                return str(p)
            best_p = format_period(best_performer['period'], analysis_level); worst_p = format_period(worst_performer['period'], analysis_level)
            insights["best_worst"] = (f"Best performance: <strong>{best_p}</strong> (Stability: {best_performer['stability']:.1f}%, Stops: {int(best_performer.get('stops', 0))}, MTTR: {best_performer.get('mttr', 0):.1f} min). Worst: <strong>{worst_p}</strong> (Stability: {worst_performer['stability']:.1f}%, Stops: {int(worst_performer.get('stops', 0))}, MTTR: {worst_performer.get('mttr', 0):.1f} min).")

            if analysis_df['stops'].sum() > 0:
                if "Daily" in analysis_level:
                    peak_stop_hour = analysis_df.loc[analysis_df['stops'].idxmax()]
                    insights["patterns"] = f"Stop events peak around <strong>{int(peak_stop_hour['period'])}:00</strong> ({int(peak_stop_hour['stops'])} stops)."
                else:
                    mean_stab = analysis_df['stability'].mean(); std_stab = analysis_df['stability'].std(); outlier_thresh = mean_stab - (1.5 * std_stab)
                    outliers = analysis_df[analysis_df['stability'] < outlier_thresh]
                    if not outliers.empty:
                        worst_outlier = outliers.loc[outliers['stability'].idxmin()]; outlier_label = format_period(worst_outlier['period'], analysis_level)
                        insights["patterns"] = f"<strong>{outlier_label}</strong> performed significantly below average, impacting overall stability."

        # Ensure MTTR/MTBF are not NaN before using in recommendation logic
        overall_mttr_nn = 0 if pd.isna(overall_mttr) else overall_mttr
        overall_mtbf_nn = 0 if pd.isna(overall_mtbf) else overall_mtbf

        if overall_stability >= 95: insights["recommendation"] = "Excellent performance. Monitor MTBF/MTTR trends."
        elif overall_stability > 70:
            focus = "<strong>MTBF</strong>" if (overall_mtbf_nn > 0 and overall_mttr_nn > 0 and overall_mtbf_nn < overall_mttr_nn * 5) else "<strong>MTTR</strong>"
            insights["recommendation"] = f"Good performance. Focus on {focus}. Current MTBF: {overall_mtbf_nn:.1f} min, MTTR: {overall_mttr_nn:.1f} min."
        else:
            driver = "Low <strong>MTBF</strong>" if (overall_mtbf_nn > 0 and overall_mttr_nn > 0 and overall_mtbf_nn < overall_mttr_nn) else "High <strong>MTTR</strong>"
            insights["recommendation"] = f"Poor stability. Primary driver: {driver}. Current MTBF: {overall_mtbf_nn:.1f} min, MTTR: {overall_mttr_nn:.1f} min. Investigate root cause."

    except KeyError as e:
         insights["error"] = f"Calculation error in analysis: Missing key {e}"
         # Reset potentially partially filled fields
         insights.update({"overall": "Error", "predictive": "Error", "best_worst": "Error", "patterns": "", "recommendation": "Error"})
    except Exception as e:
        insights["error"] = f"Unexpected error during analysis: {e}"
        insights.update({"overall": "Error", "predictive": "Error", "best_worst": "Error", "patterns": "", "recommendation": "Error"})

    return insights
# --- (End of MODIFIED generate_detailed_analysis) ---

def generate_bucket_analysis(complete_runs, bucket_labels):
    # Default message if no data
    if complete_runs.empty or 'duration_min' not in complete_runs.columns:
        return "No completed runs to analyze for long-run trends."

    total_runs = len(complete_runs)
    long_buckets = [] # Initialize empty list

    # Ensure bucket_labels is valid before iterating
    if bucket_labels and isinstance(bucket_labels, (list, tuple, pd.Series)):
        try:
            # Correctly structured list comprehension within try block
            long_buckets = [
                label for label in bucket_labels
                # Added str() conversion for robustness against non-string labels
                if int(str(label).split('-')[0].replace('+', '')) >= 60
            ]
        except (ValueError, IndexError, TypeError):
             # If any label fails parsing (e.g., unexpected format),
             # keep long_buckets potentially partially filled or empty.
             # You could add st.warning("Could not parse all bucket labels.") here if needed.
             pass # Continue with the long_buckets found so far (or empty list)

    # Calculate number of long runs *after* defining long_buckets
    num_long_runs = 0 # Default to 0
    if long_buckets and 'time_bucket' in complete_runs.columns: # Check if time_bucket exists
        # Ensure time_bucket column type is compatible before using isin
        if pd.api.types.is_categorical_dtype(complete_runs['time_bucket']):
            # Filter categories present in long_buckets before filtering the DataFrame
            valid_categories = [cat for cat in long_buckets if cat in complete_runs['time_bucket'].cat.categories]
            if valid_categories:
                num_long_runs = complete_runs[complete_runs['time_bucket'].isin(valid_categories)].shape[0]
        else: # Handle non-categorical case if necessary (less likely with pd.cut)
             try:
                 num_long_runs = complete_runs[complete_runs['time_bucket'].isin(long_buckets)].shape[0]
             except TypeError: # Catch error if types are incompatible for isin
                  st.warning("Could not filter runs by time bucket due to type mismatch.")
                  pass # Keep num_long_runs as 0

    # Calculate percentage and longest run
    pct_long = (num_long_runs / total_runs * 100) if total_runs > 0 else 0
    longest_run = "N/A" # Default longest run
    if not complete_runs.empty:
        longest_run_min_val = complete_runs['duration_min'].max()
        if pd.notna(longest_run_min_val):
            longest_run = format_minutes_to_dhm(longest_run_min_val)


    # Build analysis text
    analysis = f"<strong>{total_runs}</strong> completed runs. <strong>{num_long_runs}</strong> ({pct_long:.1f}%) were long runs (>60 min). Longest run: <strong>{longest_run}</strong>. "
    if total_runs > 0:
        if pct_long < 20: analysis += "Suggests frequent interruptions."
        elif pct_long > 50: analysis += "Indicates strong capability for sustained operation."
        else: analysis += "Shows mixed performance."

    return analysis

def generate_mttr_mtbf_analysis(analysis_df, analysis_level):
    # ... (code remains the same) ...
    if analysis_df is None or analysis_df.empty or 'stops' not in analysis_df or analysis_df['stops'].sum()==0 or len(analysis_df)<2 or 'stability' not in analysis_df or 'mttr' not in analysis_df: return "Not enough data for correlation."
    stops_corr = analysis_df['stops'].corr(analysis_df['stability']); mttr_corr = analysis_df['mttr'].corr(analysis_df['stability'])
    corr_insight = ""; primary_driver_freq = False; primary_driver_dur = False
    if not pd.isna(stops_corr) and not pd.isna(mttr_corr):
        if abs(stops_corr) > abs(mttr_corr) * 1.5: primary_driver = "**frequency of stops**"; primary_driver_freq = True
        elif abs(mttr_corr) > abs(stops_corr) * 1.5: primary_driver = "**duration of stops**"; primary_driver_dur = True
        else: primary_driver = "**frequency and duration**"
        corr_insight = f"Analysis suggests <strong>{primary_driver}</strong> most impacts stability."
        example_insight = ""; 
        def format_p(p, l):
            if isinstance(p, (pd.Timestamp, pd.Period, pd.Timedelta)): 
                return pd.to_datetime(p).strftime('%A, %b %d')
            if l == "Monthly": 
                return f"Week {p}"
            if "Daily" in l: 
                return f"{p}:00"; 
            return str(p)
    if primary_driver_freq: highest_stops = analysis_df.loc[analysis_df['stops'].idxmax()]; p_label = format_p(highest_stops['period'], analysis_level); example_insight = f"E.g., <strong>{p_label}</strong> had most stops (<strong>{int(highest_stops['stops'])}</strong>). Prioritize root cause."
    elif primary_driver_dur: highest_mttr = analysis_df.loc[analysis_df['mttr'].idxmax()]; p_label = format_p(highest_mttr['period'], analysis_level); example_insight = f"E.g., <strong>{p_label}</strong> had longest downtimes (avg <strong>{highest_mttr['mttr']:.1f} min</strong>). Investigate delays."
    else: highest_mttr = analysis_df.loc[analysis_df['mttr'].idxmax()]; p_label = format_p(highest_mttr['period'], analysis_level); example_insight = (f"E.g., <strong>{p_label}</strong> had long downtimes (avg <strong>{highest_mttr['mttr']:.1f} min</strong>), showing duration impact.")
    return f"<div style='line-height:1.6;'><p>{corr_insight}</p><p>{example_insight}</p></div>"

# --- [MODIFIED] Excel Generation Function ---
def generate_excel_report(all_runs_data, tolerance):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        header_format = workbook.add_format({'bold': True, 'bg_color': '#002060', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter', 'border': 1})
        sub_header_format = workbook.add_format({'bold': True, 'bg_color': '#C5D9F1', 'border': 1})
        label_format = workbook.add_format({'bold': True, 'align': 'left'})
        percent_format = workbook.add_format({'num_format': '0.0%', 'border': 1})
        time_format = workbook.add_format({'num_format': '[h]:mm:ss', 'border': 1})
        mins_format = workbook.add_format({'num_format': '0.00 "min"', 'border': 1})
        secs_format = workbook.add_format({'num_format': '0.00 "sec"', 'border': 1})
        data_format = workbook.add_format({'border': 1})
        datetime_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss', 'border': 1})
        error_format = workbook.add_format({'bold': True, 'font_color': 'red'})

        for run_id, data in all_runs_data.items():
            ws = workbook.add_worksheet(f"Run_{run_id:03d}")
            df_run = data['processed_df'].copy()
            start_row = 19

            col_map = {name: chr(ord('A') + i) for i, name in enumerate(df_run.columns)}
            shot_time_col_dyn = col_map.get('SHOT TIME'); stop_col = col_map.get('STOP')
            stop_event_col = col_map.get('STOP EVENT'); time_bucket_col = col_map.get('TIME BUCKET')
            first_col_for_count = shot_time_col_dyn if shot_time_col_dyn else 'A'
            cum_count_col_dyn = col_map.get('CUMULATIVE COUNT'); run_dur_col_dyn = col_map.get('RUN DURATION')
            bucket_col_dyn = col_map.get('TIME BUCKET'); time_diff_col_dyn = col_map.get('TIME DIFF SEC')

            data_cols_count = len(df_run.columns); helper_col_letter = chr(ord('A') + data_cols_count)
            ws.set_column(f'{helper_col_letter}:{helper_col_letter}', None, None, {'hidden': True})
            analysis_start_col_idx = data_cols_count + 2
            analysis_col_1 = chr(ord('A') + analysis_start_col_idx); analysis_col_2 = chr(ord('A') + analysis_start_col_idx + 1); analysis_col_3 = chr(ord('A') + analysis_start_col_idx + 2)

            missing_cols = [col for col, letter in [('STOP', stop_col), ('STOP EVENT', stop_event_col), ('TIME BUCKET', time_bucket_col), ('TIME DIFF SEC', time_diff_col_dyn), ('CUMULATIVE COUNT', cum_count_col_dyn), ('RUN DURATION', run_dur_col_dyn), ('SHOT TIME', shot_time_col_dyn)] if letter is None]
            if missing_cols: ws.write('A5', f"Error: Missing columns: {', '.join(missing_cols)}", error_format)
            table_formulas_ok = not missing_cols

            # --- Layout ---
            ws.merge_range('A1:B1', data['equipment_code'], header_format)
            ws.write('A2', 'Date', label_format); ws.write('B2', f"{data['start_time']:%Y-%m-%d} to {data['end_time']:%Y-%m-%d}")
            ws.write('A3', 'Method', label_format); ws.write('B3', 'Every Shot')
            ws.write('E1', 'Mode CT', sub_header_format); mode_ct_val = data.get('mode_ct', 0); ws.write('E2', mode_ct_val if isinstance(mode_ct_val, (int, float)) else 0, secs_format)
            ws.write('F1', 'Outside L1', sub_header_format); ws.write('G1', 'Outside L2', sub_header_format); ws.write('H1', 'IDLE', sub_header_format)
            ws.write('F2', 'Lower Limit', label_format); ws.write('G2', 'Upper Limit', label_format); ws.write('H2', 'Stops', label_format)
            lower_limit_val = data.get('lower_limit'); upper_limit_val = data.get('upper_limit') # Get Python calculated values
            ws.write('F3', lower_limit_val if lower_limit_val is not None else 'N/A', secs_format) # Write static value
            ws.write('G3', upper_limit_val if upper_limit_val is not None else 'N/A', secs_format) # Write static value
            if stop_col: ws.write_formula('H3', f"=SUM({stop_col}{start_row}:{stop_col}{start_row + len(df_run) - 1})", sub_header_format)
            else: ws.write('H3', 'N/A', sub_header_format)

            ws.write('K1', 'Total Shot Count', label_format); ws.write('L1', 'Normal Shot Count', label_format)
            ws.write_formula('K2', f"=COUNTA({first_col_for_count}{start_row}:{first_col_for_count}{start_row + len(df_run) - 1})", sub_header_format)
            ws.write_formula('L2', f"=K2-H3", sub_header_format)
            ws.write('K4', 'Efficiency', label_format); ws.write('L4', 'Stop Events', label_format)
            ws.write_formula('K5', f"=L2/K2", percent_format) # Efficiency based on counts
            if stop_event_col: ws.write_formula('L5', f"=SUM({stop_event_col}{start_row}:{stop_event_col}{start_row + len(df_run) - 1})", sub_header_format)
            else: ws.write('L5', 'N/A', sub_header_format)

            # --- [MODIFIED] Time Section ---
            ws.write('F5', 'Tot Run Time (Calc)', label_format) # Renamed label
            ws.write('G5', 'Tot Down Time', label_format)
            ws.write('H5', 'Tot Prod Time', label_format) # Added Production Time label

            # Use Python calculated values for header times (Consistent with Dashboard)
            ws.write('F6', data.get('total_runtime_sec', 0) / 86400, time_format) # Uptime + Downtime
            ws.write('G6', data.get('tot_down_time_sec', 0) / 86400, time_format) # Sum stops
            ws.write('H6', data.get('production_time_sec', 0) / 86400, time_format) # Added Production Time value

            # Recalculate percentages based on written calculated times
            ws.write_formula('F7', f"=(H6)/F6", percent_format) # Production % of Calc Total
            ws.write_formula('G7', f"=G6/F6", percent_format) # Downtime % of Calc Total
            ws.write('H7', '', data_format) # Blank cell under Prod time value

            # --- Reliability Metrics (Static Values) ---
            ws.merge_range('K8:L8', 'Reliability Metrics', header_format)
            ws.write('K9', 'MTTR (Avg)', label_format); ws.write('L9', data.get('mttr_min', 0), mins_format)
            ws.write('K10', 'MTBF (Avg)', label_format); ws.write('L10', data.get('mtbf_min', 0), mins_format)
            ws.write('K11', 'Time to First DT', label_format); ws.write('L11', data.get('time_to_first_dt_min', 0), mins_format)
            ws.write('K12', 'Avg Cycle Time', label_format); ws.write('L12', data.get('avg_cycle_time_sec', 0), secs_format)

            # --- Time Bucket Analysis ---
            ws.merge_range(f'{analysis_col_1}14:{analysis_col_3}14', 'Time Bucket Analysis', header_format)
            ws.write(f'{analysis_col_1}15', 'Bucket', sub_header_format); ws.write(f'{analysis_col_2}15', 'Duration Range', sub_header_format); ws.write(f'{analysis_col_3}15', 'Events Count', sub_header_format)
            max_bucket = 20
            for i in range(1, max_bucket + 1):
                ws.write(f'{analysis_col_1}{15+i}', i, sub_header_format); ws.write(f'{analysis_col_2}{15+i}', f"{(i-1)*20} - {i*20} min", sub_header_format)
                if time_bucket_col: ws.write_formula(f'{analysis_col_3}{15+i}', f'=COUNTIF({time_bucket_col}:{time_bucket_col},{i})', sub_header_format)
                else: ws.write(f'{analysis_col_3}{15+i}', 'N/A', sub_header_format)
            ws.write(f'{analysis_col_2}{16+max_bucket}', 'Grand Total', sub_header_format); ws.write_formula(f'{analysis_col_3}{16+max_bucket}', f"=SUM({analysis_col_3}16:{analysis_col_3}{15+max_bucket})", sub_header_format)

            # --- Data Table ---
            ws.write_row('A18', df_run.columns, header_format)
            if 'SHOT TIME' in df_run.columns: df_run['SHOT TIME'] = pd.to_datetime(df_run['SHOT TIME'], errors='coerce').dt.tz_localize(None)
            df_run.fillna('', inplace=True)

            for i, row_values in enumerate(df_run.to_numpy()):
                current_row_excel_idx = start_row + i
                for c_idx, value in enumerate(row_values):
                    col_name = df_run.columns[c_idx]
                    if col_name in ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET', 'TIME DIFF SEC']: continue
                    cell_format = data_format
                    if isinstance(value, pd.Timestamp):
                         if pd.notna(value): ws.write_datetime(current_row_excel_idx - 1, c_idx, value, datetime_format)
                         else: ws.write_string(current_row_excel_idx - 1, c_idx, '', data_format)
                    elif isinstance(value, (bool, np.bool_)): ws.write_number(current_row_excel_idx - 1, c_idx, int(value), data_format)
                    elif isinstance(value, (int, float, np.number)):
                         if col_name in ['ACTUAL CT', 'adj_ct_sec']: cell_format = secs_format
                         ws.write_number(current_row_excel_idx - 1, c_idx, value, cell_format)
                    else: ws.write(current_row_excel_idx - 1, c_idx, value, data_format)

            if table_formulas_ok:
                time_diff_col_idx = df_run.columns.get_loc('TIME DIFF SEC'); cum_count_col_idx = df_run.columns.get_loc('CUMULATIVE COUNT')
                run_dur_col_idx = df_run.columns.get_loc('RUN DURATION'); bucket_col_idx = df_run.columns.get_loc('TIME BUCKET')
                for i in range(len(df_run)):
                    row_num = start_row + i; prev_row = row_num - 1
                    if i == 0: helper_formula = f'=IF({stop_col}{row_num}=0, {time_diff_col_dyn}{row_num}, 0)'
                    else: helper_formula = f'=IF({stop_event_col}{row_num}=1, 0, {helper_col_letter}{prev_row}) + IF({stop_col}{row_num}=0, {time_diff_col_dyn}{row_num}, 0)'
                    ws.write_formula(f'{helper_col_letter}{row_num}', helper_formula)
                    if i == 0: ws.write_number(row_num - 1, time_diff_col_idx, 0, secs_format) # First row time diff is 0
                    else: formula = f'=({shot_time_col_dyn}{row_num}-{shot_time_col_dyn}{prev_row})*86400'; ws.write_formula(row_num - 1, time_diff_col_idx, formula, secs_format)
                    cum_count_formula = f'=COUNTIF(${stop_event_col}${start_row}:${stop_event_col}{row_num},1) & "/" & IF({stop_event_col}{row_num}=1, "0 sec", TEXT({helper_col_letter}{row_num}/86400, "[h]:mm:ss"))'
                    ws.write_formula(row_num - 1, cum_count_col_idx, cum_count_formula, data_format)
                    if i == 0: run_dur_formula = f'=IF({stop_event_col}{row_num}=1, 0, "")'
                    else: run_dur_formula = f'=IF({stop_event_col}{row_num}=1, {helper_col_letter}{prev_row}/86400, "")'
                    ws.write_formula(row_num - 1, run_dur_col_idx, run_dur_formula, time_format)
                    if i == 0: time_bucket_formula = f'=IF({stop_event_col}{row_num}=1, IFERROR(FLOOR(0/60/20, 1) + 1, ""), "")'
                    else: time_bucket_formula = f'=IF({stop_event_col}{row_num}=1, IFERROR(FLOOR({helper_col_letter}{prev_row}/60/20, 1) + 1, ""), "")'
                    ws.write_formula(row_num - 1, bucket_col_idx, time_bucket_formula, data_format)
            else:
                 if cum_count_col_dyn: ws.write(f'{cum_count_col_dyn}{start_row}', "Formula Error", error_format)
                 if time_diff_col_dyn: ws.write(f'{time_diff_col_dyn}{start_row}', "Formula Error", error_format)

            for i, col_name in enumerate(df_run.columns):
                try: width = max(len(str(col_name)), df_run[col_name].astype(str).map(len).max())
                except: width = len(str(col_name))
                ws.set_column(i, i, width + 2 if width < 40 else 40)

    return output.getvalue()


# --- Wrapper Function (Uses UNIFIED RunRateCalculator) ---
def generate_run_based_excel_export(df_for_export, tolerance, downtime_gap_tolerance, run_interval_hours, tool_id_selection):
    # ... (Keep this wrapper function exactly as it was in the previous version) ...
    # It correctly uses RunRateCalculator now.
    try:
        base_calc = RunRateCalculator(df_for_export, tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
    except Exception as e: st.error(f"Error in Excel base calculation: {e}"); return BytesIO().getvalue()
    if df_processed.empty: st.error("Could not process data for Excel export..."); return BytesIO().getvalue()
    split_col = 'time_diff_sec'; is_new_run = df_processed[split_col] > (run_interval_hours * 3600)
    df_processed['run_id'] = is_new_run.cumsum() + (0 if is_new_run.iloc[0] else 1)
    all_runs_data = {}; desired_columns_base = ['SUPPLIER NAME', 'tool_id', 'SESSION ID', 'SHOT ID', 'shot_time','APPROVED CT', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event', 'run_group']
    formula_columns = ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET']
    for run_id, df_run_raw in df_processed.groupby('run_id'):
        try:
            run_calculator = RunRateCalculator(df_run_raw.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
            run_results = run_calculator.results
            if not run_results or 'processed_df' not in run_results or run_results['processed_df'].empty: continue
            run_results['equipment_code'] = df_run_raw['tool_id'].iloc[0] if 'tool_id' in df_run_raw.columns else tool_id_selection
            run_results['start_time'] = df_run_raw['shot_time'].min(); run_results['end_time'] = df_run_raw['shot_time'].max()
            export_df = run_results['processed_df'].copy()
            for col in ['SUPPLIER NAME', 'SESSION ID', 'SHOT ID', 'APPROVED CT']:
                 if col in df_run_raw.columns and col not in export_df.columns: export_df = export_df.merge(df_run_raw[[col]], left_index=True, right_index=True, how='left')
            for col in formula_columns:
                if col not in export_df: export_df[col] = ''
            columns_to_export = [col for col in desired_columns_base if col in export_df.columns] + formula_columns
            final_export_df = export_df[columns_to_export].rename(columns={'tool_id': 'EQUIPMENT CODE', 'shot_time': 'SHOT TIME','time_diff_sec': 'TIME DIFF SEC', 'stop_flag': 'STOP', 'stop_event': 'STOP EVENT'})
            final_desired_renamed = ['SUPPLIER NAME', 'EQUIPMENT CODE', 'SESSION ID', 'SHOT ID', 'SHOT TIME','APPROVED CT', 'ACTUAL CT', 'TIME DIFF SEC', 'STOP', 'STOP EVENT', 'run_group','CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET']
            for col in final_desired_renamed:
                if col not in final_export_df.columns: final_export_df[col] = ''
            final_export_df = final_export_df[[col for col in final_desired_renamed if col in final_export_df.columns]]
            run_results['processed_df'] = final_export_df; all_runs_data[run_id] = run_results
        except Exception as e: st.warning(f"Could not process Run ID {run_id} for Excel export: {e}"); import traceback; st.text(traceback.format_exc())
    if not all_runs_data: st.error("No valid runs processed for Excel export."); return BytesIO().getvalue()
    return generate_excel_report(all_runs_data, tolerance)


# --- Dashboard Rendering Function ---
def render_dashboard(df_tool, tool_id_selection):
    st.sidebar.title("Dashboard Controls ⚙️")

    with st.sidebar.expander("ℹ️ About This Dashboard", expanded=False):
        # ... (description remains the same) ...
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
        - **Daily / Weekly / Monthly (by Run)**: Tolerance based on Mode CT of each run. New run after 'Run Interval Threshold' idle time.
        ---
        ### Sliders
        - **Tolerance Band**: Defines acceptable CT range (±% of Mode CT).
        - **Downtime Gap Tolerance**: Min idle sec between shots to count as a separate stop.
        - **Run Interval Threshold**: Max hours idle before a new Production Run starts.
        - **Remove Runs...**: (Only in 'by Run' mode) Filters out runs with fewer shots.
        """)

    # --- [MODIFIED] Analysis Level Options ---
    analysis_options = ["Daily", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"]
    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        options=analysis_options
    )
    # --- End Modification ---

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Defines the max hours between shots before a new Production Run is identified.")

    # Cache function uses RunRateCalculator
    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours, tol, gap_tol):
        # ... (code remains the same) ...
        base_calc = RunRateCalculator(df, tol, gap_tol, analysis_mode='aggregate'); df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week; df_processed['date'] = df_processed['shot_time'].dt.date; df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600); df_processed['run_id'] = is_new_run.cumsum() + (0 if is_new_run.iloc[0] else 1)
        return df_processed

    df_processed = get_processed_data(df_tool, run_interval_hours, tolerance, downtime_gap_tolerance)

    min_shots_filter = 1
    if 'by Run' in analysis_level: # Filter slider only appears for 'by Run' modes
        st.sidebar.markdown("---")
        if not df_processed.empty and 'run_id' in df_processed.columns:
             run_shot_counts = df_processed.groupby('run_id').size()
             if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()); default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.sidebar.slider("Remove Runs with Fewer Than X Shots", 1, max_shots, default_value, 1, help="Filters out smaller production runs.")

    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)

    if df_processed.empty: st.error(f"Could not process data for {tool_id_selection}. Check format/range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")
    # Determine mode based on selection (implicit now with reduced options)
    mode = 'by_run' if 'by Run' in analysis_level else 'aggregate'
    df_view = pd.DataFrame()

    # --- Period Selection Logic (Adjusted for new options) ---
    if analysis_level == "Daily": # Now only the aggregate daily exists
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
                run_shot_counts = df_view.groupby('run_id')['run_id'].transform('count')
                df_view = df_view[run_shot_counts >= min_shots_filter]
                runs_after = df_view['run_id'].nunique()
                if runs_before > 0: st.sidebar.metric("Runs Displayed", f"{runs_after} / {runs_before}", f"-{runs_before - runs_after} filtered", delta_color="off")
            if not df_view.empty: # Add labels if data remains
                 unique_run_ids_in_view = df_view.sort_values('shot_time')['run_id'].unique()
                 run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids_in_view)}
                 df_view['run_label'] = df_view['run_id'].map(run_label_map)

    if df_view.empty: st.warning(f"No data for selected period/filters."); st.stop()
    else:
        # --- Calculate Final Metrics (using unified RunRateCalculator) ---
        calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
        results = calc.results
        if not results or 'processed_df' not in results or results['processed_df'].empty: st.error("Calculation failed."); st.stop()
        summary_metrics = results # Use results directly

        if mode == 'by_run': sub_header = sub_header.replace("Summary for", "Summary for (Combined Runs)")

        # --- Display Section ---
        col1, col2 = st.columns([3, 1]);
        with col1: st.subheader(sub_header)
        with col2:
            st.download_button(
                label="📥 Export Run-Based Report",
                data=generate_run_based_excel_export(df_view.copy(), tolerance, downtime_gap_tolerance, run_interval_hours, tool_id_selection), # Pass gap tol
                file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # --- Trend Summary Calculation (Adjusted) ---
        trend_summary_df = None
        # Note: Daily aggregate mode now doesn't produce a 'daily summary', it shows hourly directly.
        if "Weekly (by Run)" in analysis_level: trend_summary_df = calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode)
        elif "Monthly (by Run)" in analysis_level: trend_summary_df = calculate_weekly_summaries_for_month(df_view, tolerance, downtime_gap_tolerance, mode)
        elif "Custom Period (by Run)" in analysis_level: # Treat custom 'by run' like weekly/monthly for trends
             # Determine dominant time scale (days or weeks) for trend display
             time_span_days = (df_view['date'].max() - df_view['date'].min()).days
             if time_span_days > 14: # More than 2 weeks, show weekly trend
                  trend_summary_df = calculate_weekly_summaries_for_month(df_view, tolerance, downtime_gap_tolerance, mode) # Use weekly func
             else: # Shorter period, show daily trend
                  trend_summary_df = calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode) # Use daily func
        elif "by Run" in analysis_level: # Generic fallback for 'by Run' if needed, though covered above
             trend_summary_df = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
             if not trend_summary_df.empty: trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
        elif analysis_level == "Daily": # Daily Aggregate mode uses hourly summary directly
             trend_summary_df = results.get('hourly_summary', pd.DataFrame())


        # --- Metric Display Containers (Unchanged) ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0); prod_t = summary_metrics.get('production_time_sec', 0); down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0; down_p = (down_t / total_d * 100) if total_d > 0 else 0
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
            # Use .get() for limits as they might not exist if ACTUAL CT is missing
            if mode == 'by_run':
                min_ll = summary_metrics.get('min_lower_limit', 0); max_ll = summary_metrics.get('max_lower_limit', 0)
                c1.metric("Lower Limit (sec)", f"{min_ll:.2f} – {max_ll:.2f}" if min_ll != max_ll else f"{min_ll:.2f}")
                with c2:
                    min_mc = summary_metrics.get('min_mode_ct', 0); max_mc = summary_metrics.get('max_mode_ct', 0)
                    # Corrected indentation for the nested 'with' statement
                    with st.container(border=True):
                        st.metric("Mode CT (sec)", f"{min_mc:.2f} – {max_mc:.2f}" if min_mc != max_mc else f"{min_mc:.2f}")
                min_ul = summary_metrics.get('min_upper_limit', 0); max_ul = summary_metrics.get('max_upper_limit', 0)
                c3.metric("Upper Limit (sec)", f"{min_ul:.2f} – {max_ul:.2f}" if min_ul != max_ul else f"{min_ul:.2f}")
            else: # Aggregate Mode
                mode_val = summary_metrics.get('mode_ct', 0); mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else mode_val
                c1.metric("Lower Limit (sec)", f"{summary_metrics.get('lower_limit', 0):.2f}")
                with c2:
                    # Corrected indentation for the nested 'with'
                    with st.container(border=True):
                        st.metric("Mode CT (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{summary_metrics.get('upper_limit', 0):.2f}")

        # --- [MODIFIED] Detailed Analysis HTML Call ---
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame() # Prepare df for analysis function
                if trend_summary_df is not None and not trend_summary_df.empty:
                    analysis_df = trend_summary_df.copy(); rename_map = {}
                    # Add robustness: check if columns exist before renaming
                    if 'hour' in analysis_df.columns: rename_map = {'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'RUN ID' in analysis_df.columns: rename_map = {'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}
                    analysis_df.rename(columns=rename_map, inplace=True, errors='ignore') # Ignore errors if cols don't exist

                # Call analysis function (now returns defaults/error)
                insights = generate_detailed_analysis(
                    analysis_df,
                    summary_metrics.get('stability_index', 0),
                    summary_metrics.get('mttr_min', 0),
                    summary_metrics.get('mtbf_min', 0),
                    analysis_level
                )

                # Display error or insights using .get() for safety
                if insights.get("error"):
                    st.error(insights["error"])
                else:
                    patterns_html = f'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> {insights.get("patterns", "")}</p>' if insights.get("patterns") else ''
                    html_content = f"""
                    <div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;">
                        <h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4>
                        <p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights.get("overall", "N/A")}</p>
                        <p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights.get("predictive", "N/A")}</p>
                        <p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights.get("best_worst", "N/A")}</p>
                        {patterns_html}
                        <p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights.get("recommendation", "N/A")}</p>
                    </div>
                    """
                    components.v1.html(html_content, height=400, scrolling=True)

        # --- Breakdown Tables (Adjusted for fewer options) ---
        # Show daily/weekly table only for the relevant 'by Run' modes
        if analysis_level in ["Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"] and trend_summary_df is not None and not trend_summary_df.empty:
            with st.expander("View Daily/Weekly Breakdown Table", expanded=False):
                d_df = trend_summary_df.copy()
                if 'date' in d_df.columns: d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d'); d_df.rename(columns={'date': 'Day', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                elif 'week' in d_df.columns: d_df.rename(columns={'week': 'Week', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                st.dataframe(d_df.style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
        elif "by Run" in analysis_level: # Show Run breakdown for all 'by Run' modes
            run_summary_df = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            if run_summary_df is not None and not run_summary_df.empty:
                 with st.expander("View Run Breakdown Table", expanded=False):
                    d_df = run_summary_df.copy()
                    d_df["Period"] = d_df.apply(lambda r: f"{r['start_time']:%Y-%m-%d %H:%M} to {r['end_time']:%Y-%m-%d %H:%M}", axis=1)
                    d_df["Total Shots"] = d_df['total_shots'].apply(lambda x: f"{x:,}")
                    d_df["Normal Shots (%)"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Stops (%)"] = d_df.apply(lambda r: f"{r['stops']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Total Duration"] = d_df['total_runtime_sec'].apply(format_duration)
                    d_df["Prod. Time (%)"] = d_df.apply(lambda r: f"{format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (%)"] = d_df.apply(lambda r: f"{format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df.rename(columns={'run_label':'Run ID','mode_ct':'Mode CT','lower_limit':'LL','upper_limit':'UL','mttr_min':'MTTR','mtbf_min':'MTBF','stability_index':'Stability %'}, inplace=True)
                    cols = ['Run ID','Period','Total Shots','Normal Shots (%)','Stops (%)','Mode CT','LL','UL','Total Duration','Prod. Time (%)','Downtime (%)','MTTR','MTBF','Stability %']
                    st.dataframe(d_df[cols].style.format({'Mode CT':'{:.2f}','LL':'{:.2f}','UL':'{:.2f}','MTTR':'{:.1f}','MTBF':'{:.1f}','Stability %':'{:.1f}'}), use_container_width=True)

        # --- Plots ---
        plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=('hourly' if analysis_level == "Daily" else 'daily' if 'Weekly' in analysis_level else 'weekly'))
        with st.expander("View Shot Data Table", expanded=False): st.dataframe(results['processed_df'][['shot_time', 'run_label', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event']])

        st.markdown("---")

        # --- Trend Plot Section (Adjusted Logic) ---
        if analysis_level == "Daily": # Only Daily Aggregate mode left
            # ... (Daily/hourly plot code remains the same) ...
            st.header("Hourly Analysis")
            run_durations_period=results.get("run_durations",pd.DataFrame());processed_period_df=results.get('processed_df',pd.DataFrame());stop_events_df=processed_period_df.loc[processed_period_df['stop_event']].copy();complete_runs=pd.DataFrame()
            if not stop_events_df.empty: stop_events_df['terminated_run_group']=stop_events_df['run_group']-1;end_time_map=stop_events_df.set_index('terminated_run_group')['shot_time'];run_durations_period['run_end_time']=run_durations_period['run_group'].map(end_time_map);complete_runs=run_durations_period.dropna(subset=['run_end_time']).copy()
            else: complete_runs=run_durations_period
            c1,c2=st.columns(2)
            with c1:
                if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                    b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                    fig_b = px.bar(b_counts, title="Time Bucket Analysis (Completed Runs)", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                    st.plotly_chart(fig_b, use_container_width=True)
                    # Corrected Indentation:
                    with st.expander("View Bucket Data"):
                        st.dataframe(complete_runs)
                else:
                    st.info("No complete runs for bucket analysis.")
            with c2:plot_trend_chart(trend_summary_df,'hour','stability_index',"Hourly Stability Trend","Hour of Day","Stability (%)",is_stability=True);
            with st.expander("View Stability Data"):st.dataframe(trend_summary_df[['hour','stability_index','mttr_min','mtbf_min','stops','total_shots']])
            if not complete_runs.empty and'run_end_time'in complete_runs:
                st.subheader("Hourly Bucket Trend");complete_runs['hour']=complete_runs['run_end_time'].dt.hour;pivot_df=pd.crosstab(index=complete_runs['hour'],columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]));pivot_df=pivot_df.reindex(pd.Index(range(24),name='hour'),fill_value=0);fig_hourly_bucket=px.bar(pivot_df,x=pivot_df.index,y=pivot_df.columns,title='Hourly Distribution of Run Durations',barmode='stack',color_discrete_map=results["bucket_color_map"],labels={'hour':'Hour of Stop','value':'Number of Runs','variable':'Run Duration (min)'});st.plotly_chart(fig_hourly_bucket,use_container_width=True);
                with st.expander("View Bucket Trend Data"):st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis"):st.markdown(generate_bucket_analysis(complete_runs,results["bucket_labels"]),unsafe_allow_html=True)
            st.subheader("Hourly MTTR & MTBF Trend");hourly_summary=results.get('hourly_summary',pd.DataFrame())
            if hourly_summary is not None and not hourly_summary.empty:plot_mttr_mtbf_chart(df=hourly_summary,x_col='hour',mttr_col='mttr_min',mtbf_col='mtbf_min',shots_col='total_shots',title="Hourly MTTR, MTBF & Shot Count Trend");
            with st.expander("View MTTR/MTBF Data"):st.dataframe(hourly_summary)
            if detailed_view and hourly_summary['stops'].sum()>0:
                with st.expander("🤖 View MTTR/MTBF Correlation Analysis"):st.info("""**How this works:** Frequency vs Duration analysis.""");analysis_df=hourly_summary.copy().rename(columns={'hour':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'});st.markdown(generate_mttr_mtbf_analysis(analysis_df,analysis_level),unsafe_allow_html=True)

        # Logic for Weekly/Monthly/Custom (by Run) modes
        elif "by Run" in analysis_level:
            # Determine trend level based on time span or explicit mode
            if analysis_level == "Weekly (by Run)": trend_level = "Daily"
            elif analysis_level == "Monthly (by Run)": trend_level = "Weekly"
            elif analysis_level == "Custom Period (by Run)":
                 time_span_days = (df_view['date'].max() - df_view['date'].min()).days
                 trend_level = "Weekly" if time_span_days > 14 else "Daily"
            else: # Should not happen with new options, but fallback
                 trend_level = "Run" # Plot run-by-run directly if other logic fails

            # Use Run Summary directly if plotting run-by-run
            if trend_level == "Run":
                 st.header("Run-Based Analysis")
                 run_summary_df = trend_summary_df # Already calculated run summaries
            else: # Use Daily/Weekly summaries for trends
                 st.header(f"{trend_level} Trends for {analysis_level.split(' (')[0]}")
                 summary_df = trend_summary_df # Daily or Weekly summary

            # Bucket Analysis (remains the same logic, uses 'results' from overall calc)
            run_durations_period=results.get("run_durations",pd.DataFrame());processed_period_df=results.get('processed_df',pd.DataFrame());stop_events_df=processed_period_df.loc[processed_period_df['stop_event']].copy();complete_runs=pd.DataFrame()
            if not stop_events_df.empty: stop_events_df['terminated_run_group']=stop_events_df['run_group']-1;end_time_map=stop_events_df.set_index('terminated_run_group')['shot_time'];run_durations_period['run_end_time']=run_durations_period['run_group'].map(end_time_map);complete_runs=run_durations_period.dropna(subset=['run_end_time']).copy()
            else: complete_runs=run_durations_period

            c1, c2 = st.columns(2)
            with c1: # Bucket analysis always uses 'complete_runs' from the overall period
                st.subheader("Total Bucket Analysis (Across Runs/Period)")
                if not complete_runs.empty and "time_bucket" in complete_runs.columns: b_counts=complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"],fill_value=0);fig_b=px.bar(b_counts,title="Total Time Bucket Analysis",labels={"index":"Duration(min)","value":"Occurrences"},text_auto=True,color=b_counts.index,color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration');st.plotly_chart(fig_b,use_container_width=True);
                with st.expander("View Bucket Data"):st.dataframe(complete_runs)
                else: st.info("No complete runs.")

            with c2: # Stability Trend
                if trend_level == "Run": # Plot run stability
                     st.subheader("Stability per Production Run")
                     if run_summary_df is not None and not run_summary_df.empty: plot_trend_chart(run_summary_df,'RUN ID','STABILITY %',"Stability per Run","Run ID","Stability (%)",is_stability=True);
                     with st.expander("View Stability Data"):st.dataframe(run_summary_df)
                     else: st.info(f"No runs to analyze.")
                else: # Plot Daily/Weekly stability
                     st.subheader(f"{trend_level} Stability Trend")
                     if summary_df is not None and not summary_df.empty: x_col='date'if trend_level=="Daily"else'week';plot_trend_chart(summary_df,x_col,'stability_index',f"{trend_level} Stability Trend",trend_level,"Stability (%)",is_stability=True);
                     with st.expander("View Stability Data"):st.dataframe(summary_df)
                     else: st.info(f"No {trend_level.lower()} data.")

            # Bucket Trend per Unit (Day/Week/Run)
            if not complete_runs.empty and 'run_end_time' in complete_runs:
                st.subheader(f"{trend_level} Bucket Trend")
                pivot_df = pd.DataFrame()
                if trend_level == "Run":
                     # Use run_label from complete_runs if available
                     if 'run_label' not in complete_runs: # Add if missing (should be added earlier now)
                          run_group_to_label_map=processed_period_df.drop_duplicates('run_group')[['run_group','run_label']].set_index('run_group')['run_label']
                          complete_runs['run_label']=complete_runs['run_group'].map(run_group_to_label_map)
                     valid_run_labels = run_summary_df['RUN ID'].unique() if run_summary_df is not None else complete_runs['run_label'].unique()
                     complete_runs_filtered = complete_runs[complete_runs['run_label'].isin(valid_run_labels)]
                     if not complete_runs_filtered.empty: pivot_df = pd.crosstab(index=complete_runs_filtered['run_label'],columns=complete_runs_filtered['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"])).reindex(valid_run_labels, fill_value=0)
                     trend_df = run_summary_df.set_index('RUN ID').reindex(valid_run_labels) if run_summary_df is not None else None
                     x_axis_title = "Run ID"
                else: # Daily or Weekly
                    time_col = 'date' if trend_level == "Daily" else 'week'
                    complete_runs[time_col] = complete_runs['run_end_time'].dt.date if trend_level == "Daily" else complete_runs['run_end_time'].dt.isocalendar().week
                    all_units = summary_df[time_col].unique()
                    pivot_df = pd.crosstab(index=complete_runs[time_col],columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"])).reindex(all_units, fill_value=0).sort_index()
                    trend_df = summary_df.set_index(time_col).reindex(all_units).sort_index() if summary_df is not None else None
                    x_axis_title = trend_level

                if not pivot_df.empty and trend_df is not None:
                     fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                     for col in pivot_df.columns: fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results["bucket_color_map"].get(col)), secondary_y=False)
                     fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=trend_df.index, y=trend_df['Total Shots' if trend_level=='Run' else 'total_shots'], mode='lines+markers+text', text=trend_df['Total Shots' if trend_level=='Run' else 'total_shots'], textposition='top center', line=dict(color='blue')), secondary_y=True)
                     fig_bucket_trend.update_layout(barmode='stack', title_text=f'{trend_level} Distribution of Run Durations vs. Shot Count', xaxis_title=x_axis_title, yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                     st.plotly_chart(fig_bucket_trend, use_container_width=True)
                     with st.expander("View Bucket Trend Data"): st.dataframe(pivot_df)
                     if detailed_view: with st.expander("🤖 View Bucket Trend Analysis"): st.markdown(generate_bucket_analysis(complete_runs_filtered if trend_level=='Run' else complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
                else: st.info(f"Not enough data for {trend_level} bucket trend.")


            # MTTR/MTBF Trend per Unit
            st.subheader(f"{trend_level} MTTR & MTBF Trend")
            trend_df_for_mttr = run_summary_df if trend_level == "Run" else summary_df
            if trend_df_for_mttr is not None and not trend_df_for_mttr.empty:
                x_col = 'RUN ID' if trend_level == "Run" else ('date' if trend_level == "Daily" else 'week')
                mttr_col = 'MTTR (min)' if trend_level == "Run" else 'mttr_min'
                mtbf_col = 'MTBF (min)' if trend_level == "Run" else 'mtbf_min'
                shots_col = 'Total Shots' if trend_level == "Run" else 'total_shots'
                stops_col = 'STOPS' if trend_level == "Run" else 'stops'
                # Check if stops exist before plotting
                if stops_col in trend_df_for_mttr.columns and trend_df_for_mttr[stops_col].sum() > 0:
                    plot_mttr_mtbf_chart(df=trend_df_for_mttr, x_col=x_col, mttr_col=mttr_col, mtbf_col=mtbf_col, shots_col=shots_col, title=f"{trend_level} MTTR, MTBF & Shot Count Trend")
                    with st.expander("View MTTR/MTBF Data"): st.dataframe(trend_df_for_mttr)
                    if detailed_view:
                        with st.expander("🤖 View MTTR/MTBF Correlation Analysis"): st.info("""**How this works:** Frequency vs Duration analysis."""); analysis_df=trend_df_for_mttr.copy(); rename_map={};
                        if trend_level=="Run": rename_map={'RUN ID':'period','STABILITY %':'stability','STOPS':'stops','MTTR (min)':'mttr'}
                        elif trend_level=="Daily": rename_map={'date':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'}
                        else: rename_map={'week':'period','stability_index':'stability','stops':'stops','mttr_min':'mttr'}; analysis_df.rename(columns=rename_map,inplace=True); st.markdown(generate_mttr_mtbf_analysis(analysis_df,analysis_level),unsafe_allow_html=True)
                else: st.info(f"No stop events recorded in this period to calculate MTTR/MTBF for {trend_level} trend.")
            else: st.info(f"Not enough data for {trend_level} MTTR/MTBF trend.")


# --- Risk Tower Functions (Unchanged) ---
@st.cache_data(show_spinner="Analyzing tool performance for Risk Tower...")
def calculate_risk_scores(df_all_tools):
    # ... (code remains the same) ...
    id_col="tool_id";initial_metrics=[];default_tol,default_gap=0.05,2.0
    for tool_id,df_tool in df_all_tools.groupby(id_col):
        if df_tool.empty or len(df_tool)<10:continue
        try:
            calc_prepare=RunRateCalculator(df_tool,default_tol,default_gap);df_prepared=calc_prepare.results.get("processed_df");
            if df_prepared is None or df_prepared.empty:continue
            end_date=df_prepared['shot_time'].max();start_date=end_date-timedelta(weeks=4);df_period=df_prepared[(df_prepared['shot_time']>=start_date)&(df_prepared['shot_time']<=end_date)];
            if df_period.empty or len(df_period)<10:continue
            calc=RunRateCalculator(df_period.copy(),default_tol,default_gap);res=calc.results;
            if not res:continue
            df_period['week']=df_period['shot_time'].dt.isocalendar().week;weekly_stabilities=[RunRateCalculator(df_week,default_tol,default_gap).results.get('stability_index',0)for _,df_week in df_period.groupby('week')if not df_week.empty];trend="Stable";
            if len(weekly_stabilities)>1 and weekly_stabilities[-1]<weekly_stabilities[0]*0.95:trend="Declining"
            initial_metrics.append({'Tool ID':tool_id,'Stability':res.get('stability_index',0),'MTTR':res.get('mttr_min',0),'MTBF':res.get('mtbf_min',0),'Weekly Stability':' → '.join([f'{s:.0f}%'for s in weekly_stabilities]),'Trend':trend,'Analysis Period':f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"})
        except Exception as e:st.warning(f"Could not calculate risk score for {tool_id}: {e}")
    if not initial_metrics:return pd.DataFrame()
    metrics_df=pd.DataFrame(initial_metrics);overall_mttr_mean=metrics_df['MTTR'].mean();overall_mtbf_mean=metrics_df['MTBF'].mean();final_risk_data=[]
    for _,row in metrics_df.iterrows():
        risk_score=row['Stability'];details=f"Overall stability is {row['Stability']:.1f}%.";
        if row['Trend']=="Declining":risk_score-=20;primary_factor="Declining Trend";details="Stability shows downward trend."
        elif row['Stability']<70 and row['MTTR']>(overall_mttr_mean*1.2):primary_factor="High MTTR";details=f"Avg stop duration (MTTR) {row['MTTR']:.1f} min concern."
        elif row['Stability']<70 and row['MTBF']<(overall_mtbf_mean*0.8):primary_factor="Frequent Stops";details=f"Frequent stops (MTBF {row['MTBF']:.1f} min) impacting."
        else:primary_factor="Low Stability"
        final_risk_data.append({'Tool ID':row['Tool ID'],'Analysis Period':row['Analysis Period'],'Risk Score':max(0,risk_score),'Primary Risk Factor':primary_factor,'Weekly Stability':row['Weekly Stability'],'Details':details})
    if not final_risk_data:return pd.DataFrame()
    return pd.DataFrame(final_risk_data).sort_values('Risk Score',ascending=True).reset_index(drop=True)

def render_risk_tower(df_all_tools):
    # ... (code remains the same) ...
    st.title("Run Rate Risk Tower");st.info("Analyzes last 4 weeks. Lowest scores = highest risk.")
    with st.expander("ℹ️ How the Risk Tower Works"):st.markdown("""**Analysis Period**: 4-week range/tool. **Risk Score**: Stability Index (%), -20 for declining trend. **Primary Risk Factor**: Declining > High MTTR (>1.2x avg) > Frequent Stops (MTBF <0.8x avg) > Low Stability. **Color**: <span style='background-color:#ff6961;color:black;padding:2px 5px;border-radius:5px;'>Red (0-50)</span>,<span style='background-color:#ffb347;color:black;padding:2px 5px;border-radius:5px;'>Orange (51-70)</span>,<span style='background-color:#77dd77;color:black;padding:2px 5px;border-radius:5px;'>Green (>70)</span>.""",unsafe_allow_html=True)
    risk_df=calculate_risk_scores(df_all_tools)
    if risk_df.empty:st.warning("Not enough data for Risk Tower.");return
    def style_risk(row):score=row['Risk Score'];color=PASTEL_COLORS['green']if score>70 else PASTEL_COLORS['orange']if score>50 else PASTEL_COLORS['red'];return[f'background-color: {color}'for _ in row]
    st.dataframe(risk_df.style.apply(style_risk,axis=1).format({'Risk Score':'{:.0f}'}),use_container_width=True,hide_index=True)

# --- Main App Structure ---
st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload one or more Run Rate Excel files", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploaded_files: st.info("👈 Upload one or more Excel files."); st.stop()

@st.cache_data
def load_all_data(files):
    # ... (code remains the same) ...
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file)
            if "TOOLING ID" in df.columns: df.rename(columns={"TOOLING ID": "tool_id"}, inplace=True)
            elif "EQUIPMENT CODE" in df.columns: df.rename(columns={"EQUIPMENT CODE": "tool_id"}, inplace=True)
            if "tool_id" in df.columns and (("SHOT TIME" in df.columns) or ({"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns))): df_list.append(df)
            elif "tool_id" not in df.columns: st.warning(f"Skipping {file.name}: Missing Tool ID.")
            else: st.warning(f"Skipping {file.name}: Missing Time columns.")
        except Exception as e: st.warning(f"Load error {file.name}: {e}")
    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

df_all_tools = load_all_data(uploaded_files)
id_col = "tool_id"
if id_col not in df_all_tools.columns: st.error("No valid data loaded."); st.stop()
df_all_tools.dropna(subset=[id_col], inplace=True); df_all_tools[id_col] = df_all_tools[id_col].astype(str)
unique_tool_ids = sorted(df_all_tools[id_col].unique().tolist())
if not unique_tool_ids: st.error("No tools found after cleaning."); st.stop()

tool_ids_options = ["All Tools (Risk Tower)"] + unique_tool_ids
dashboard_tool_id_selection = st.sidebar.selectbox("Select Tool ID for Dashboard Analysis", tool_ids_options)

if dashboard_tool_id_selection == "All Tools (Risk Tower)":
    first_tool = unique_tool_ids[0]; df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool].copy(); tool_id_for_dashboard_display = first_tool
else: df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection].copy(); tool_id_for_dashboard_display = dashboard_tool_id_selection

tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])
with tab1: render_risk_tower(df_all_tools)
with tab2:
    if not df_for_dashboard.empty: render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
    else: st.info("Select Tool ID or ensure default tool has data.")