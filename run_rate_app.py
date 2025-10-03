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
        
        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        total_downtime_sec = hourly_groups.apply(lambda x: x[x['stop_flag'] == 1]['ct_diff_sec'].sum())
        uptime_min = df[(df['stop_flag'] == 0) & (df['ct_diff_sec'] <= 28800)].groupby('hour')['ct_diff_sec'].sum() / 60
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

    def _calculate_stop_durations(self, df):
        stop_durations = []
        stop_start_times = []
        stop_start_time = None
        
        for i in range(len(df)):
            if df.loc[i, "stop_event"]:
                stop_start_time = df.loc[i - 1, "shot_time"] if i > 0 else df.loc[i, "shot_time"]
            elif stop_start_time is not None and df.loc[i, "stop_flag"] == 0:
                stop_end_time = df.loc[i, "shot_time"]
                duration_sec = (stop_end_time - stop_start_time).total_seconds()
                if duration_sec <= 28800:
                    stop_durations.append(duration_sec)
                    stop_start_times.append(stop_start_time)
                stop_start_time = None
        return pd.Series(stop_durations, index=stop_start_times)

    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        if self.analysis_mode == 'by_run' and 'run_id' in df.columns:
            run_modes = df.groupby('run_id')['ACTUAL CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
            df['mode_ct'] = df['run_id'].map(run_modes)
            lower_limit = df['mode_ct'] * (1 - self.tolerance)
            upper_limit = df['mode_ct'] * (1 + self.tolerance)
            df['lower_limit'] = lower_limit
            df['upper_limit'] = upper_limit
            mode_ct_display = "Varies by Run"
        else:
            df_for_mode_calc = df[df["ct_diff_sec"] <= 28800]
            mode_ct = df_for_mode_calc["ACTUAL CT"].mode().iloc[0] if not df_for_mode_calc["ACTUAL CT"].mode().empty else 0
            lower_limit = mode_ct * (1 - self.tolerance)
            upper_limit = mode_ct * (1 + self.tolerance)
            mode_ct_display = mode_ct

        stop_condition = (((df["ct_diff_sec"] < lower_limit) | (df["ct_diff_sec"] > upper_limit)) & (df["ct_diff_sec"] <= 28800))
        df["stop_flag"] = np.where(stop_condition, 1, 0)
        if not df.empty:
            df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)
        
        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        
        downtime_sec = df.loc[df['stop_flag'] == 1, 'ct_diff_sec'].sum()
        mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0

        production_time_sec = df[(df['stop_flag'] == 0) & (df['ct_diff_sec'] <= 28800)]['ct_diff_sec'].sum()

        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        effective_runtime_sec = production_time_sec + downtime_sec
        stability_index = (production_time_sec / effective_runtime_sec * 100) if effective_runtime_sec > 0 else (100.0 if stop_events == 0 else 0.0)
        
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        df_for_runs = df[df['ct_diff_sec'] <= 28800].copy()
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")["ct_diff_sec"].sum().div(60).reset_index(name="duration_min")

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
        
        # --- MODIFIED BLOCK START ---
        # Define color palettes for gradients
        reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
        
        # Categorize bucket labels to apply gradients
        red_labels, blue_labels, green_labels = [], [], []
        for label in labels:
            try:
                lower_bound = int(label.split('-')[0].replace('+', ''))
                if lower_bound < 60: red_labels.append(label)
                elif 60 <= lower_bound < 160: blue_labels.append(label)
                else: green_labels.append(label)
            except (ValueError, IndexError): continue
            
        # Assign gradient colors to each category
        bucket_color_map = {}
        for i, label in enumerate(red_labels): bucket_color_map[label] = reds[i % len(reds)]
        for i, label in enumerate(blue_labels): bucket_color_map[label] = blues[i % len(blues)]
        for i, label in enumerate(green_labels): bucket_color_map[label] = greens[i % len(greens)]
        # --- MODIFIED BLOCK END ---
            
        hourly_summary = self._calculate_hourly_summary(df)
        
        final_results = {
            "processed_df": df, "mode_ct": mode_ct_display, "total_shots": total_shots, "efficiency": efficiency,
            "stop_events": stop_events, "normal_shots": normal_shots, "mttr_min": mttr_min,
            "mtbf_min": mtbf_min, "stability_index": stability_index, "run_durations": run_durations,
            "bucket_labels": labels, "bucket_color_map": bucket_color_map, "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_sec, "production_time_sec": production_time_sec, "downtime_sec": downtime_sec,
        }
        
        if self.analysis_mode == 'by_run' and isinstance(lower_limit, pd.Series) and not df.empty:
            final_results["min_lower_limit"] = lower_limit.min()
            final_results["max_lower_limit"] = lower_limit.max()
            final_results["min_upper_limit"] = upper_limit.min()
            final_results["max_upper_limit"] = upper_limit.max()
            final_results["min_mode_ct"] = df['mode_ct'].min()
            final_results["max_mode_ct"] = df['mode_ct'].max()
        else:
            final_results["lower_limit"] = lower_limit
            final_results["upper_limit"] = upper_limit
            
        return final_results

# ... (the rest of your code remains unchanged) ...
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

    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        for run_id, group in df.groupby('run_id'):
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=group['plot_time'].min(), y0=group['lower_limit'].iloc[0],
                x1=group['plot_time'].max(), y1=group['upper_limit'].iloc[0],
                fillcolor=PASTEL_COLORS['green'], opacity=0.2, layer="below", line_width=0
            )
    else:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=df['plot_time'].min(), y0=lower_limit,
            x1=df['plot_time'].max(), y1=upper_limit,
            fillcolor=PASTEL_COLORS['green'], opacity=0.2, layer="below", line_width=0
        )

    fig.add_trace(go.Bar(x=df['plot_time'], y=df['ct_diff_sec'], marker_color=df['color'], name='Cycle Time'))
    
    y_axis_cap_val = mode_ct if isinstance(mode_ct, (int, float)) else df['mode_ct'].mean() if 'mode_ct' in df else 50
    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)
    
    fig.update_layout(title="Cycle Time per Shot vs. Tolerance", xaxis_title="Time", yaxis_title="Cycle Time (sec)",
                        yaxis=dict(range=[0, y_axis_cap]), bargap=0.05, xaxis=dict(showgrid=True))
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

def format_minutes_to_dhm(total_minutes):
    if pd.isna(total_minutes) or total_minutes < 0: return "N/A"
    total_minutes = int(total_minutes)
    days = total_minutes // (60 * 24)
    remaining_minutes = total_minutes % (60 * 24)
    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def format_duration(seconds):
    if pd.isna(seconds) or seconds < 0: return "N/A"
    return format_minutes_to_dhm(seconds / 60)
    
def calculate_daily_summaries_for_week(df_week, tolerance, analysis_mode):
    daily_results_list = []
    for date in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date]
        if not df_day.empty:
            calc = RunRateCalculator(df_day.copy(), tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'date': date, 'stability_index': res.get('stability_index', np.nan),
                        'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                        'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}
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
                        'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}
            weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance):
    """Iterates through a period's data, calculates metrics for each run, and returns a summary DataFrame."""
    run_summary_list = []
    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty:
            calc = RunRateCalculator(df_run.copy(), tolerance, analysis_mode='aggregate')
            res = calc.results
            
            total_shots = res.get('total_shots', 0)
            normal_shots = res.get('normal_shots', 0)
            stopped_shots = total_shots - normal_shots
            total_runtime_sec = res.get('total_runtime_sec', 0)
            production_time_sec = res.get('production_time_sec', 0)
            downtime_sec = res.get('downtime_sec', 0)
            
            summary = {
                'run_label': run_label,
                'start_time': df_run['shot_time'].min(),
                'end_time': df_run['shot_time'].max(),
                'total_shots': total_shots,
                'normal_shots': normal_shots,
                'stopped_shots': stopped_shots,
                'mode_ct': res.get('mode_ct', 0),
                'lower_limit': res.get('lower_limit', 0),
                'upper_limit': res.get('upper_limit', 0),
                'total_runtime_sec': total_runtime_sec,
                'production_time_sec': production_time_sec,
                'downtime_sec': downtime_sec,
                'mttr_min': res.get('mttr_min', np.nan),
                'mtbf_min': res.get('mtbf_min', np.nan),
                'stability_index': res.get('stability_index', np.nan),
                'stops': res.get('stop_events', 0)
            }
            run_summary_list.append(summary)
            
    if not run_summary_list:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(run_summary_list).sort_values('start_time').reset_index(drop=True)
    return summary_df

# --- Analysis Engine Functions ---
def generate_detailed_analysis(analysis_df, overall_stability, overall_mttr, overall_mtbf, analysis_level):
    if analysis_df is None or analysis_df.empty:
        return {"error": "Not enough data to generate a trend analysis."}

    stability_class = "good (above 70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (below 50%)"
    overall_summary = f"The overall stability for this period is <strong>{overall_stability:.1f}%</strong>, which is considered <strong>{stability_class}</strong>."

    predictive_insight = ""
    if len(analysis_df) > 1:
        volatility_std = analysis_df['stability'].std()
        volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"
        
        half_point = len(analysis_df) // 2
        first_half_mean = analysis_df['stability'].iloc[:half_point].mean()
        second_half_mean = analysis_df['stability'].iloc[half_point:].mean()
        
        trend_direction = "stable"
        if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
        elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"

        if trend_direction == "stable":
            predictive_insight = f"Performance has been <strong>{volatility_level}</strong> with no clear long-term upward or downward trend."
        else:
            predictive_insight = f"Performance shows a <strong>{trend_direction} trend</strong>, although this has been <strong>{volatility_level}</strong>."

    best_worst_analysis = ""
    if not analysis_df.empty:
        best_performer = analysis_df.loc[analysis_df['stability'].idxmax()]
        worst_performer = analysis_df.loc[analysis_df['stability'].idxmin()]

        def format_period(period_value, level):
            if isinstance(period_value, (pd.Timestamp, pd.Period, pd.Timedelta)):
                return pd.to_datetime(period_value).strftime('%A, %b %d')
            if level == "Monthly": return f"Week {period_value}"
            if level == "Daily": return f"{period_value}:00"
            return str(period_value)

        best_period_label = format_period(best_performer['period'], analysis_level)
        worst_period_label = format_period(worst_performer['period'], analysis_level)

        best_worst_analysis = (f"The best performance was during <strong>{best_period_label}</strong> (Stability: {best_performer['stability']:.1f}%), "
                                f"while the worst was during <strong>{worst_period_label}</strong> (Stability: {worst_performer['stability']:.1f}%). "
                                f"The key difference was the impact of stoppages: the worst period had {int(worst_performer['stops'])} stops with an average duration of {worst_performer.get('mttr', 0):.1f} min, "
                                f"compared to {int(best_performer['stops'])} stops during the best period.")

    pattern_insight = ""
    if not analysis_df.empty and analysis_df['stops'].sum() > 0:
        if analysis_level == "Daily":
            peak_stop_hour = analysis_df.loc[analysis_df['stops'].idxmax()]
            pattern_insight = f"A notable pattern is the concentration of stop events around <strong>{int(peak_stop_hour['period'])}:00</strong>, which saw the highest number of interruptions ({int(peak_stop_hour['stops'])} stops)."
        else:
            mean_stability = analysis_df['stability'].mean()
            std_stability = analysis_df['stability'].std()
            outlier_threshold = mean_stability - (1.5 * std_stability)
            outliers = analysis_df[analysis_df['stability'] < outlier_threshold]
            if not outliers.empty:
                worst_outlier = outliers.loc[outliers['stability'].idxmin()]
                outlier_label = format_period(worst_outlier['period'], analysis_level)
                pattern_insight = f"A key area of concern is <strong>{outlier_label}</strong>, which performed significantly below average and disproportionately affected the overall stability."

    recommendation = ""
    if overall_stability >= 95:
        recommendation = "Overall performance is excellent. Continue monitoring for any emerging negative trends in either MTBF or MTTR to maintain this high level of stability."
    elif overall_stability > 70:
        if overall_mtbf > 0 and overall_mttr > 0 and overall_mtbf < (overall_mttr * 5):
            recommendation = f"Performance is good, but could be improved by focusing on <strong>Mean Time Between Failures (MTBF)</strong>. With an MTBF of <strong>{overall_mtbf:.1f} minutes</strong>, investigating the root causes of the more frequent, smaller stops could yield significant gains."
        else:
            recommendation = f"Performance is good, but could be improved by focusing on <strong>Mean Time To Repair (MTTR)</strong>. With an MTTR of <strong>{overall_mttr:.1f} minutes</strong>, streamlining the repair process for the infrequent but longer stops could yield significant gains."
    else:
        if overall_mtbf > 0 and overall_mttr > 0 and overall_mtbf < overall_mttr:
            recommendation = f"Stability is poor and requires attention. The primary driver is a low <strong>Mean Time Between Failures (MTBF)</strong> of <strong>{overall_mtbf:.1f} minutes</strong>. The top priority should be investigating the root cause of frequent machine stoppages."
        else:
            recommendation = f"Stability is poor and requires attention. The primary driver is a high <strong>Mean Time To Repair (MTTR)</strong> of <strong>{overall_mttr:.1f} minutes</strong>. The top priority should be investigating why stops take a long time to resolve and streamlining the repair process."

    return {"overall": overall_summary, "predictive": predictive_insight, "best_worst": best_worst_analysis, "patterns": pattern_insight, "recommendation": recommendation}

def generate_bucket_analysis(complete_runs, bucket_labels):
    if complete_runs.empty or 'duration_min' not in complete_runs.columns:
        return "No completed runs to analyze for long-run trends."
    total_completed_runs = len(complete_runs)
    try:
        long_run_buckets = [label for label in bucket_labels if int(label.split('-')[0].replace('+', '')) >= 60]
    except (ValueError, IndexError):
        long_run_buckets = []
    if not long_run_buckets:
        num_long_runs = 0
    else:
        num_long_runs = complete_runs[complete_runs['time_bucket'].isin(long_run_buckets)].shape[0]
    percent_long_runs = (num_long_runs / total_completed_runs * 100) if total_completed_runs > 0 else 0
    longest_run_min = complete_runs['duration_min'].max()
    longest_run_formatted = format_minutes_to_dhm(longest_run_min)
    analysis_text = f"Out of <strong>{total_completed_runs}</strong> completed runs, <strong>{num_long_runs}</strong> ({percent_long_runs:.1f}%) qualified as long runs (lasting over 60 minutes). "
    analysis_text += f"The single longest stable run during this period lasted for <strong>{longest_run_formatted}</strong>."
    if total_completed_runs > 0:
        if percent_long_runs < 20:
            analysis_text += " This suggests that most stoppages occur after relatively short periods of operation, indicating frequent process interruptions."
        elif percent_long_runs > 50:
            analysis_text += " This indicates a strong capability for sustained stable operation, with over half the runs achieving significant duration before a stop event."
        else:
            analysis_text += " This shows a mixed performance, with a reasonable number of long runs but also frequent shorter ones."
    return analysis_text

def generate_mttr_mtbf_analysis(analysis_df, analysis_level):
    if analysis_df is None or analysis_df.empty or analysis_df['stops'].sum() == 0 or len(analysis_df) < 2:
        return "Not enough stoppage data to generate a detailed correlation analysis."
    if not all(col in analysis_df.columns for col in ['stops', 'stability', 'mttr']):
        return "Could not perform analysis due to missing data columns."
    stops_stability_corr = analysis_df['stops'].corr(analysis_df['stability'])
    mttr_stability_corr = analysis_df['mttr'].corr(analysis_df['stability'])
    corr_insight = ""
    primary_driver_is_frequency = False
    primary_driver_is_duration = False
    if not pd.isna(stops_stability_corr) and not pd.isna(mttr_stability_corr):
        if abs(stops_stability_corr) > abs(mttr_stability_corr) * 1.5:
            primary_driver = "the **frequency of stops**"
            primary_driver_is_frequency = True
        elif abs(mttr_stability_corr) > abs(stops_stability_corr) * 1.5:
            primary_driver = "the **duration of stops**"
            primary_driver_is_duration = True
        else:
            primary_driver = "both the **frequency and duration of stops**"
        corr_insight = (f"This analysis suggests that <strong>{primary_driver}</strong> has the strongest impact on overall stability.")
    example_insight = ""
    def format_period(period_value, level):
        if isinstance(period_value, (pd.Timestamp, pd.Period, pd.Timedelta)):
            return pd.to_datetime(period_value).strftime('%A, %b %d')
        if level == "Monthly": return f"Week {period_value}"
        if level == "Daily": return f"{period_value}:00"
        return str(period_value)
    if primary_driver_is_frequency:
        highest_stops_period_row = analysis_df.loc[analysis_df['stops'].idxmax()]
        period_label = format_period(highest_stops_period_row['period'], analysis_level)
        example_insight = (f"For example, the period with the most interruptions was <strong>{period_label}</strong>, which recorded <strong>{int(highest_stops_period_row['stops'])} stops</strong>. Prioritizing the root cause of these frequent events is recommended.")
    elif primary_driver_is_duration:
        highest_mttr_period_row = analysis_df.loc[analysis_df['mttr'].idxmax()]
        period_label = format_period(highest_mttr_period_row['period'], analysis_level)
        example_insight = (f"The period with the longest downtimes was <strong>{period_label}</strong>, where the average repair time was <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>. Investigating the cause of these prolonged stops is the top priority.")
    else:
        highest_mttr_period_row = analysis_df.loc[analysis_df['mttr'].idxmax()]
        period_label = format_period(highest_mttr_period_row['period'], analysis_level)
        example_insight = (f"As an example, <strong>{period_label}</strong> experienced prolonged downtimes with an average repair time of <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>, highlighting the impact of long stops.")
    return f"<div style='line-height: 1.6;'><p>{corr_insight}</p><p>{example_insight}</p></div>"

def create_excel_export(df_view, results, tolerance, run_interval_hours, analysis_level, tool_id_selection):
    output_buffer = BytesIO()
    with xlsxwriter.Workbook(output_buffer, {'in_memory': True, 'nan_inf_to_errors': True}) as workbook:
        header_format = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter', 'border': 1})
        label_format = workbook.add_format({'bold': True, 'align': 'right'})
        output_format = workbook.add_format({'bold': True, 'bg_color': '#C5D9F1', 'border': 1})
        percent_format = workbook.add_format({'num_format': '0.0%', 'bg_color': '#C5D9F1', 'border': 1})
        decimal_format = workbook.add_format({'num_format': '0.00', 'bg_color': '#C5D9F1', 'border': 1})
        dhm_format = workbook.add_format({'num_format': '[h]"h" mm"m"', 'bg_color': '#C5D9F1', 'border': 1})
        ws_dash = workbook.add_worksheet('Dashboard')
        ws_dash.set_column('B:C', 25)
        ws_dash.merge_range('B2:C2', f'Overall Performance Metrics: {tool_id_selection}', header_format)
        metrics = {
            'Total Shots:': results.get('total_shots', 0), 'Normal Shots:': results.get('normal_shots', 0),
            'Stop Events:': results.get('stop_events', 0), 'Efficiency:': results.get('efficiency', 0),
            'Total Duration:': results.get('total_runtime_sec', 0) / 86400, 'Total Production Time:': results.get('production_time_sec', 0) / 86400,
            'Total Downtime:': results.get('downtime_sec', 0) / 86400, 'Stability Index:': results.get('stability_index', 0) / 100,
            'MTTR (min):': results.get('mttr_min', 0), 'MTBF (min):': results.get('mtbf_min', 0),
        }
        row = 3
        for label, value in metrics.items():
            ws_dash.write(f'B{row}', label, label_format)
            fmt = output_format
            if '%' in label or 'Efficiency' in label or 'Stability' in label: fmt = percent_format
            elif '(min)' in label: fmt = decimal_format
            elif 'Duration' in label or 'Time' in label: fmt = dhm_format
            ws_dash.write(f'C{row}', value, fmt)
            row += 1
        ws_raw = workbook.add_worksheet('Exported_Raw_Data')
        df_to_export = df_view.copy()
        if 'month' in df_to_export.columns:
            df_to_export['month'] = df_to_export['month'].astype(str)
        df_to_export.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_to_export = df_to_export.fillna('')
        for col_num, value in enumerate(df_to_export.columns.values):
            ws_raw.write(0, col_num, value, header_format)
        for row_num, row_data in enumerate(df_to_export.itertuples(index=False), 1):
            ws_raw.write_row(row_num, 0, row_data)
        ws_calc = workbook.add_worksheet('Calculations_Data')
        calc_df = results.get('processed_df', pd.DataFrame()).copy()
        if 'month' in calc_df.columns:
            calc_df['month'] = calc_df['month'].astype(str)
        calc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        calc_df = calc_df.fillna('')
        for col_num, value in enumerate(calc_df.columns.values):
            ws_calc.write(0, col_num, value, header_format)
        for row_num, row_data in enumerate(calc_df.itertuples(index=False), 1):
            ws_calc.write_row(row_num, 0, row_data)
    return output_buffer.getvalue()

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
        - **Daily**: Hourly trends for one day.
        - **Weekly / Monthly**: Aggregated data, with daily/weekly trend charts.
        - **Weekly / Monthly (by Run)**: A more precise analysis where the tolerance for stops is calculated from the Mode CT of each individual production run. A new run is identified after a stoppage longer than the selected 'Run Interval Threshold'.
        ---
        ### Sliders
        - **Tolerance Band**: Defines the acceptable CT range around the Mode CT.
        - **Run Interval Threshold**: Defines the max hours between shots before a new Production Run is identified.
        """)
        

    analysis_level = st.sidebar.radio("Select Analysis Level", ["Daily", "Weekly", "Monthly", "Custom Period", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"])

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Defines the max hours between shots before a new Production Run is identified.")
    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)

    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours):
        base_calc = RunRateCalculator(df, 0.01)
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
            df_processed['date'] = df_processed['shot_time'].dt.date
            df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            is_new_run = df_processed['ct_diff_sec'] > (interval_hours * 3600)
            df_processed['run_id'] = is_new_run.cumsum()
            run_start_dates = df_processed.groupby('run_id')['shot_time'].min()
            run_labels = {run_id: f"{i+1:03d} ({date.strftime('%Y-%m-%d')})" for i, (run_id, date) in enumerate(run_start_dates.items())}
            df_processed['run_label'] = df_processed['run_id'].map(run_labels)
        return df_processed

    df_processed = get_processed_data(df_tool, run_interval_hours)
    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")

    mode = 'by_run' if '(by Run)' in analysis_level else 'aggregate'
    df_view = pd.DataFrame()

    if analysis_level == "Daily":
        st.header("Daily Analysis")
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates:
            st.warning("No data available for any date.")
            st.stop()
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {selected_date.strftime('%d %b %Y')}"
    elif "Weekly" in analysis_level:
        st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_weeks = sorted(df_processed["week"].unique())
        if not available_weeks:
            st.warning("No data available for any week.")
            st.stop()
        year = df_processed['shot_time'].iloc[0].year
        selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1)
        if mode == 'by_run':
            runs_in_week = df_processed[df_processed['week'] == selected_week]['run_label'].unique()
            df_view = df_processed[df_processed['run_label'].isin(runs_in_week)]
        else:
            df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"
    elif "Monthly" in analysis_level:
        st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_months = sorted(df_processed["month"].unique())
        if not available_months:
            st.warning("No data available for any month.")
            st.stop()
        selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'))
        if mode == 'by_run':
            runs_in_month = df_processed[df_processed['month'] == selected_month]['run_label'].unique()
            df_view = df_processed[df_processed['run_label'].isin(runs_in_month)]
        else:
            df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"
    elif "Custom Period" in analysis_level:
        st.header(f"Custom Period Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        min_date = df_processed['date'].min()
        max_date = df_processed['date'].max()
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)
        if start_date and end_date:
            if mode == 'by_run':
                mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
                runs_in_period = df_processed[mask]['run_label'].unique()
                df_view = df_processed[df_processed['run_label'].isin(runs_in_period)]
            else:
                mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
                df_view = df_processed[mask]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    if df_view.empty:
        st.warning(f"No data for the selected period.")
    else:
        calc = RunRateCalculator(df_view.copy(), tolerance, analysis_mode=mode)
        results = calc.results
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(sub_header)
        with col2:
            st.download_button(
                label="📥 Export to Excel",
                data=create_excel_export(df_view, results, tolerance, run_interval_hours, analysis_level, tool_id_selection),
                file_name=f"Run_Rate_Analysis_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        trend_summary_df = None
        if analysis_level == "Weekly":
            trend_summary_df = calculate_daily_summaries_for_week(df_view, tolerance, mode)
        elif analysis_level == "Monthly":
            trend_summary_df = calculate_weekly_summaries_for_month(df_view, tolerance, mode)
        elif "by Run" in analysis_level:
            trend_summary_df = calculate_run_summaries(df_view, tolerance)
            if not trend_summary_df.empty:
                trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'total_shots': 'Total Shots'}, inplace=True)
        elif analysis_level == "Daily":
            trend_summary_df = results.get('hourly_summary', pd.DataFrame())
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = results.get('total_runtime_sec', 0); prod_t = results.get('production_time_sec', 0); down_t = results.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            with col1: st.metric("MTTR", f"{results.get('mttr_min', 0):.1f} min")
            with col2: st.metric("MTBF", f"{results.get('mtbf_min', 0):.1f} min")
            with col3: st.metric("Total Run Duration", format_duration(total_d))
            with col4:
                st.metric("Production Time", f"{format_duration(prod_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{format_duration(down_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
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
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{results.get('stop_events', 0)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            if mode == 'by_run':
                min_ll = results.get('min_lower_limit', 0); max_ll = results.get('max_lower_limit', 0)
                c1.metric("Lower Limit (sec)", f"{min_ll:.2f} – {max_ll:.2f}")
                with c2:
                    min_mc = results.get('min_mode_ct', 0); max_mc = results.get('max_mode_ct', 0)
                    with st.container(border=True): st.metric("Mode CT (sec)", f"{min_mc:.2f} – {max_mc:.2f}")
                min_ul = results.get('min_upper_limit', 0); max_ul = results.get('max_upper_limit', 0)
                c3.metric("Upper Limit (sec)", f"{min_ul:.2f} – {max_ul:.2f}")
            else:
                mode_val = results.get('mode_ct', 0)
                mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else mode_val
                c1.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}")
                with c2:
                    with st.container(border=True): st.metric("Mode CT (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}")
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame()
                if trend_summary_df is not None and not trend_summary_df.empty:
                    analysis_df = trend_summary_df.copy()
                    rename_map = {}
                    if 'hour' in analysis_df.columns: rename_map = {'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'RUN ID' in analysis_df.columns: rename_map = {'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}
                    analysis_df.rename(columns=rename_map, inplace=True)
                insights = generate_detailed_analysis(analysis_df, results.get('stability_index', 0), results.get('mttr_min', 0), results.get('mtbf_min', 0), analysis_level)
                if "error" in insights: st.error(insights["error"])
                else:
                    st.components.v1.html(f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights['overall']}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights['predictive']}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights['best_worst']}</p> {'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> ' + insights['patterns'] + '</p>' if insights['patterns'] else ''}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights['recommendation']}</p></div>""", height=400, scrolling=True)
        if analysis_level in ["Weekly", "Monthly", "Custom Period"]:
            with st.expander("View Daily Breakdown Table", expanded=False):
                if trend_summary_df is not None and not trend_summary_df.empty:
                    d_df = trend_summary_df.copy()
                    if 'date' in d_df.columns:
                        d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                        d_df.rename(columns={'date': 'Day', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                    elif 'week' in d_df.columns:
                        d_df.rename(columns={'week': 'Week', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                    st.dataframe(d_df.style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
        elif analysis_level in ["Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"]:
            run_summary_df = calculate_run_summaries(df_view, tolerance)
            with st.expander("View Run Breakdown Table", expanded=False):
                if run_summary_df is not None and not run_summary_df.empty:
                    d_df = run_summary_df.copy()
                    d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
                    d_df["Total shots"] = d_df['total_shots'].apply(lambda x: f"{x:,}")
                    d_df["Normal shots (& %)"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["STOPS (&%)"] = d_df.apply(lambda r: f"{r['stops']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(format_duration)
                    d_df["Production Time (d/h/m) (& %)"] = d_df.apply(lambda r: f"{format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (& %)"] = d_df.apply(lambda r: f"{format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df.rename(columns={'run_label':'RUN ID','mode_ct':'Mode CT (for the run)','lower_limit':'Lower limit CT (sec)','upper_limit':'Upper Limit CT (sec)','mttr_min':'MTTR (min)','mtbf_min':'MTBF (min)','stability_index':'STABILITY %','stops':'STOPS'}, inplace=True)
                    final_cols = ['RUN ID','Period (date/time from to)','Total shots','Normal shots (& %)','STOPS (&%)','Mode CT (for the run)','Lower limit CT (sec)','Upper Limit CT (sec)','Total Run duration (d/h/m)','Production Time (d/h/m) (& %)','Downtime (& %)','MTTR (min)','MTBF (min)','STABILITY %','STOPS']
                    st.dataframe(d_df[final_cols].style.format({'Mode CT (for the run)':'{:.2f}','Lower limit CT (sec)':'{:.2f}','Upper Limit CT (sec)':'{:.2f}','MTTR (min)':'{:.1f}','MTBF (min)':'{:.1f}','STABILITY %':'{:.1f}'}), use_container_width=True)
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
            complete_runs = pd.DataFrame()
            if not stop_events_df.empty:
                stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
                end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
                run_durations_day['run_end_time'] = run_durations_day['run_group'].map(end_time_map)
                complete_runs = run_durations_day.dropna(subset=['run_end_time']).copy()
            else: complete_runs = run_durations_day
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
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            st.subheader("Hourly MTTR & MTBF Trend")
            hourly_summary = results['hourly_summary']
            if not hourly_summary.empty and hourly_summary['stops'].sum() > 0:
                fig_mt = make_subplots(specs=[[{"secondary_y": True}]])
                fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
                fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
                fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['total_shots'], name='Total Shots', mode='lines+markers+text', text=hourly_summary['total_shots'], textposition='top center', line=dict(color='blue', dash='dot')), secondary_y=True)
                fig_mt.update_layout(title_text="Hourly MTTR, MTBF & Shot Count Trend", yaxis_title="MTTR (min)", yaxis2_title="MTBF (min)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_mt, use_container_width=True)
                with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(hourly_summary)
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = hourly_summary.copy().rename(columns={'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'})
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
        elif analysis_level in ["Weekly", "Monthly", "Custom Period"]:
            trend_level = "Daily" if "Weekly" in analysis_level else "Weekly" if "Monthly" in analysis_level else "Daily"
            st.header(f"{trend_level} Trends for {analysis_level.split(' ')[0]}")
            summary_df = trend_summary_df
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
                if summary_df is not None and not summary_df.empty:
                    x_col = 'date' if trend_level == "Daily" else 'week'
                    plot_trend_chart(summary_df, x_col, 'stability_index', f"{trend_level} Stability Trend", trend_level, "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): st.dataframe(summary_df)
                else: st.info(f"No {trend_level.lower()} data.")
            st.subheader(f"{trend_level} Bucket Trend")
            if not complete_runs.empty and summary_df is not None and not summary_df.empty:
                time_col = 'date' if trend_level == "Daily" else 'week'
                complete_runs[time_col] = complete_runs['run_end_time'].dt.date if trend_level == "Daily" else complete_runs['run_end_time'].dt.isocalendar().week
                pivot_df = pd.crosstab(index=complete_runs[time_col], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
                all_units = summary_df[time_col]
                pivot_df = pivot_df.reindex(all_units, fill_value=0)
                fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                for col in pivot_df.columns:
                    fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results["bucket_color_map"].get(col)), secondary_y=False)
                fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=summary_df[time_col], y=summary_df['total_shots'], mode='lines+markers+text', text=summary_df['total_shots'], textposition='top center', line=dict(color='blue')), secondary_y=True)
                fig_bucket_trend.update_layout(barmode='stack', title_text=f'{trend_level} Distribution of Run Durations vs. Shot Count', xaxis_title=trend_level, yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                st.plotly_chart(fig_bucket_trend, use_container_width=True)
                with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            st.subheader(f"{trend_level} MTTR & MTBF Trend")
            if summary_df is not None and not summary_df.empty and summary_df['stops'].sum() > 0:
                x_col = 'date' if trend_level == "Daily" else 'week'
                fig_mt = make_subplots(specs=[[{"secondary_y": True}]])
                fig_mt.add_trace(go.Scatter(x=summary_df[x_col], y=summary_df['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
                fig_mt.add_trace(go.Scatter(x=summary_df[x_col], y=summary_df['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
                fig_mt.add_trace(go.Scatter(x=summary_df[x_col], y=summary_df['total_shots'], name='Total Shots', mode='lines+markers+text', text=summary_df['total_shots'], textposition='top center', line=dict(color='blue', dash='dot')), secondary_y=True)
                fig_mt.update_layout(title_text=f"{trend_level} MTTR, MTBF & Shot Count Trend", yaxis_title="MTTR (min)", yaxis2_title="MTBF (min)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_mt, use_container_width=True)
                with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(summary_df)
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = summary_df.copy()
                        rename_map = {}
                        if 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                        elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                        analysis_df.rename(columns=rename_map, inplace=True)
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
        elif "by Run" in analysis_level:
            st.header(f"Run-Based Analysis")
            run_summary_df = calculate_run_summaries(df_view, tolerance)
            if not run_summary_df.empty:
                run_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
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
                if run_summary_df is not None and not run_summary_df.empty:
                    plot_trend_chart(run_summary_df, 'RUN ID', 'STABILITY %', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): st.dataframe(run_summary_df)
                else: st.info(f"No runs to analyze.")
            st.subheader("Bucket Trend per Production Run")
            if not complete_runs.empty and run_summary_df is not None and not run_summary_df.empty:
                run_group_to_label_map = processed_df.drop_duplicates('run_group')[['run_group', 'run_label']].set_index('run_group')['run_label']
                complete_runs['run_label'] = complete_runs['run_group'].map(run_group_to_label_map)
                pivot_df = pd.crosstab(index=complete_runs['run_label'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
                all_runs = run_summary_df['RUN ID']
                pivot_df = pivot_df.reindex(all_runs, fill_value=0)
                fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                for col in pivot_df.columns:
                    fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results["bucket_color_map"].get(col)), secondary_y=False)
                fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=run_summary_df['RUN ID'], y=run_summary_df['Total Shots'], mode='lines+markers+text', text=run_summary_df['Total Shots'], textposition='top center', line=dict(color='blue')), secondary_y=True)
                fig_bucket_trend.update_layout(barmode='stack', title_text='Distribution of Run Durations per Run vs. Shot Count', xaxis_title='Run ID', yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                st.plotly_chart(fig_bucket_trend, use_container_width=True)
                with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            st.subheader("MTTR & MTBF per Production Run")
            if run_summary_df is not None and not run_summary_df.empty and run_summary_df['STOPS'].sum() > 0:
                fig_mt = make_subplots(specs=[[{"secondary_y": True}]])
                fig_mt.add_trace(go.Scatter(x=run_summary_df['RUN ID'], y=run_summary_df['MTTR (min)'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
                fig_mt.add_trace(go.Scatter(x=run_summary_df['RUN ID'], y=run_summary_df['MTBF (min)'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
                fig_mt.add_trace(go.Scatter(x=run_summary_df['RUN ID'], y=run_summary_df['Total Shots'], name='Total Shots', mode='lines+markers+text', text=run_summary_df['Total Shots'], textposition='top center', line=dict(color='blue', dash='dot')), secondary_y=True)
                fig_mt.update_layout(title_text="MTTR, MTBF & Shot Count per Run", yaxis_title="MTTR (min)", yaxis2_title="MTBF (min)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_mt, use_container_width=True)
                with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(run_summary_df)
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = run_summary_df.copy().rename(columns={'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'})
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)

@st.cache_data(show_spinner="Analyzing tool performance for Risk Tower...")
def calculate_risk_scores(df_all_tools):
    """Analyzes data for all tools over the last 4 weeks to generate risk scores."""
    id_col = "TOOLING ID" if "TOOLING ID" in df_all_tools.columns else "EQUIPMENT CODE"
    
    # Prepare data once
    calc_prepare = RunRateCalculator(df_all_tools, tolerance=0.05)
    df_prepared = calc_prepare.results.get("processed_df")
    if df_prepared is None or df_prepared.empty:
        return pd.DataFrame()

    end_date = df_prepared['shot_time'].max()
    start_date = end_date - timedelta(weeks=4)
    df_period = df_prepared[(df_prepared['shot_time'] >= start_date) & (df_prepared['shot_time'] <= end_date)]

    if df_period.empty:
        return pd.DataFrame()

    risk_data = []
    tool_ids = df_period[id_col].unique()
    
    # Pre-calculate overall stats to avoid repeated groupby
    overall_mttr_mean = df_period.groupby(id_col).apply(lambda x: RunRateCalculator(x, 0.05).results.get('mttr_min',0)).mean()
    overall_mtbf_mean = df_period.groupby(id_col).apply(lambda x: RunRateCalculator(x, 0.05).results.get('mtbf_min',0)).mean()


    for tool_id in tool_ids:
        df_tool = df_period[df_period[id_col] == tool_id].copy()
        if len(df_tool) < 10: continue

        calc = RunRateCalculator(df_tool, tolerance=0.05)
        res = calc.results
        stability = res.get('stability_index', 0)
        mttr = res.get('mttr_min', 0)
        mtbf = res.get('mtbf_min', 0)
        
        df_tool['week'] = df_tool['shot_time'].dt.isocalendar().week
        weekly_stabilities = []
        for week in sorted(df_tool['week'].unique()):
            df_week = df_tool[df_tool['week'] == week]
            if not df_week.empty:
                week_calc = RunRateCalculator(df_week, 0.05)
                weekly_stabilities.append(week_calc.results.get('stability_index', 0))
        
        trend = "Stable"
        if len(weekly_stabilities) > 1:
            if weekly_stabilities[-1] < weekly_stabilities[0] * 0.95:
                trend = "Declining"

        risk_score = stability * 0.8
        if trend == "Declining":
            risk_score -= 20
        
        primary_factor = "Low Stability"
        details = f"Overall stability is {stability:.1f}%."
        if trend == "Declining":
            primary_factor = "Declining Trend"
            details = "Stability shows a consistent downward trend."
        elif stability < 70 and mttr > (overall_mttr_mean * 1.2):
             primary_factor = "High MTTR"
             details = f"Average stop duration (MTTR) of {mttr:.1f} min is a key concern."
        elif stability < 70 and mtbf < (overall_mtbf_mean * 0.8):
             primary_factor = "Frequent Stops"
             details = f"Frequent stops (MTBF of {mtbf:.1f} min) are impacting stability."

        risk_data.append({
            'Tool ID': tool_id, 'Risk Score': max(0, risk_score),
            'Primary Risk Factor': primary_factor,
            'Weekly Stability': ' → '.join([f'{s:.0f}%' for s in weekly_stabilities]),
            'Details': details
        })

    if not risk_data:
        return pd.DataFrame()
        
    return pd.DataFrame(risk_data).sort_values('Risk Score', ascending=True).reset_index(drop=True)

def render_risk_tower(df_all_tools):
    st.title("Run Rate Risk Tower")
    st.info("This tower analyzes performance over the last 4 weeks, identifying tools that require attention. Tools with the lowest scores are at the highest risk.")
    
    risk_df = calculate_risk_scores(df_all_tools)

    if risk_df.empty:
        st.warning("Not enough data across multiple tools in the last 4 weeks to generate a risk tower.")
        return

    def style_risk(row):
        score = row['Risk Score']
        if score > 70: color = PASTEL_COLORS['green']
        elif score > 50: color = PASTEL_COLORS['orange']
        else: color = PASTEL_COLORS['red']
        return [f'background-color: {color}' for _ in row]

    st.dataframe(risk_df.style.apply(style_risk, axis=1).format({'Risk Score': '{:.0f}'}), use_container_width=True, hide_index=True)

# --- Main App Structure ---
st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload one or more Run Rate Excel files", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploaded_files:
    st.info("👈 Upload one or more Excel files to begin.")
    st.stop()

@st.cache_data
def load_all_data(files):
    df_list = []
    for file in files:
        df = pd.read_excel(file)
        # Ensure 'shot_time' is parsed correctly across all files
        if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
            datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
            df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
        elif "SHOT TIME" in df.columns:
            df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
        df_list.append(df)
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

df_all_tools = load_all_data(uploaded_files)

id_col = "TOOLING ID" if "TOOLING ID" in df_all_tools.columns else "EQUIPMENT CODE"
if id_col not in df_all_tools.columns:
    st.error(f"Files must contain 'TOOLING ID' or 'EQUIPMENT CODE'.")
    st.stop()

# Add a selectbox for Tool ID for the main dashboard
tool_ids = ["All Tools (Aggregated)"] + sorted(df_all_tools[id_col].unique().tolist())
tool_id_selection = st.sidebar.selectbox(f"Select {id_col} for Dashboard Analysis", tool_ids)

if tool_id_selection == "All Tools (Aggregated)":
    df_for_dashboard = df_all_tools
else:
    df_for_dashboard = df_all_tools[df_all_tools[id_col] == tool_id_selection]


tab1, tab2 = st.tabs(["Run Rate Dashboard", "Risk Tower"])

with tab1:
    render_dashboard(df_for_dashboard, tool_id_selection)

with tab2:
    render_risk_tower(df_all_tools)