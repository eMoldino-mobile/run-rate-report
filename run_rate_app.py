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
from datetime import datetime, timedelta, date

# ==============================================================================
# --- 1. PAGE CONFIG, CONSTANTS, & UTILITY FUNCTIONS ---
# ==============================================================================

# --- Page and Code Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

# --- Constants ---
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- Utility Functions ---

def format_minutes_to_dhm(total_minutes):
    """Converts total minutes into a 'Xd Yh Zm' string."""
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
    """Converts total seconds into a 'Xd Yh Zm' string."""
    if pd.isna(seconds) or seconds < 0: return "N/A"
    return format_minutes_to_dhm(seconds / 60)

def get_renamed_summary_df(df_in):
    """
    Helper function to rename summary tables consistently
    AND select only the columns intended for display.
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame()
    
    df = df_in.copy()
    
    # Define all possible columns and their new names
    rename_map = {
        'hour': 'Hour',
        'date': 'Date',
        'week': 'Week',
        'RUN ID': 'RUN ID',
        'stops': 'Stops',
        'STOPS': 'Stops',
        'total_shots': 'Total Shots',
        'Total Shots': 'Total Shots',
        # 'total_downtime_sec': 'Total Downtime (sec)', # Removed - too granular
        # 'uptime_min': 'Uptime (min)',              # Removed - too granular
        'mttr_min': 'MTTR (min)',
        'MTTR (min)': 'MTTR (min)',
        'mtbf_min': 'MTBF (min)',
        'MTBF (min)': 'MTBF (min)',
        'stability_index': 'Stability Index (%)',
        'STABILITY %': 'Stability Index (%)'
    }
    
    # NEW: Filter df to only include columns that are keys in the map
    cols_to_keep = [col for col in df.columns if col in rename_map]
    df_filtered = df[cols_to_keep]
    
    # Rename only columns that exist in the dataframe
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df_filtered.columns}
    df_renamed = df_filtered.rename(columns=cols_to_rename)
    
    # NEW: Re-order columns to a logical display order
    display_order = [
        'Hour', 'Date', 'Week', 'RUN ID', 'Stops', 'Total Shots',
        # 'Uptime (min)', 'Total Downtime (sec)', # Removed
        'Stability Index (%)', 'MTTR (min)', 'MTBF (min)'
    ]
    
    # Get the intersection of columns we have and the desired order
    final_cols = [col for col in display_order if col in df_renamed.columns]
    
    # Add any columns that were renamed but not in the display_order list (as a fallback)
    for col in df_renamed.columns:
        if col not in final_cols:
            final_cols.append(col)
            
    return df_renamed[final_cols]

# ==============================================================================
# --- 2. CORE CALCULATION ENGINE ---
# ==============================================================================

class RunRateCalculator:
    """
    Handles all core metric calculations for a given DataFrame.
    """
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.analysis_mode = analysis_mode
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        """Prepares raw DataFrame by parsing time and calculating initial 'time_diff_sec'."""
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

        df["time_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        if not df.empty and pd.isna(df.loc[0, "time_diff_sec"]):
            if "ACTUAL CT" in df.columns:
                df.loc[0, "time_diff_sec"] = df.loc[0, "ACTUAL CT"]
            else:
                df.loc[0, "time_diff_sec"] = 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates an hourly summary for the 'Daily' view."""
        if df.empty or 'stop_event' not in df.columns:
            return pd.DataFrame()

        df['hour'] = df['shot_time'].dt.hour
        
        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        total_downtime_sec = hourly_groups.apply(lambda x: x[x['stop_flag'] == 1]['adj_ct_sec'].sum())
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ACTUAL CT'].sum() / 60
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
        
        hourly_summary['stability_index'] = np.where(
             hourly_summary['total_shots'] == 0,
             np.nan, # Set to NaN if no shots occurred
             hourly_summary['stability_index']
        )
        
        cols_to_fill = [col for col in hourly_summary.columns if col != 'stability_index']
        hourly_summary[cols_to_fill] = hourly_summary[cols_to_fill].fillna(0)
        
        return hourly_summary

    def _calculate_all_metrics(self) -> dict:
        """The main calculation function. Runs all metrics."""
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        # --- 1. Determine Mode CT and Tolerance Limits ---
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
            if not df_for_mode_calc.empty and not df_for_mode_calc['ACTUAL CT'].value_counts().empty:
                 mode_ct = df_for_mode_calc['ACTUAL CT'].value_counts().idxmax()
            else:
                 mode_ct = 0
            lower_limit = mode_ct * (1 - self.tolerance)
            upper_limit = mode_ct * (1 + self.tolerance)
            mode_ct_display = mode_ct

        # --- 2. Stop Detection Logic ---
        # (This section replaces the previous fix)

        is_hard_stop_code = df["ACTUAL CT"] >= 999.9
        is_abnormal_cycle = ((df["ACTUAL CT"] < lower_limit) | (df["ACTUAL CT"] > upper_limit)) & ~is_hard_stop_code
        prev_actual_ct = df["ACTUAL CT"].shift(1)
        # This is the "pure" time gap, ignoring the hard stop code for now
        is_time_gap = df["time_diff_sec"] > (prev_actual_ct + self.downtime_gap_tolerance)

        # Flag all three types of stops
        df["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code, 1, 0)
        if not df.empty:
            df.loc[0, "stop_flag"] = 0
        
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # --- NEW LOGIC for adj_ct_sec ---
        # This determines the value to be summed for downtime and plotted.
        
        # By default, the value is just the cycle time
        df['adj_ct_sec'] = df['ACTUAL CT']
        
        # If it's a "pure" time gap, the value *is* that gap's duration
        df.loc[is_time_gap, 'adj_ct_sec'] = df['time_diff_sec']
        
        # If it's a "hard stop" marker (999.9), its *own* value must be 0
        # because the time_diff_sec of the *next* shot will capture the full duration.
        df.loc[is_hard_stop_code, 'adj_ct_sec'] = 0


        # --- 3. Core Metric Calculations ---
        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        
        downtime_sec = df.loc[df['stop_flag'] == 1, 'adj_ct_sec'].sum()
        production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum()
        
        # --- START: Modified Total Run Duration Logic ---
        # This logic replaces: total_runtime_sec = production_time_sec + downtime_sec
        # We rely on df being sorted by 'shot_time' from the _prepare_data step
        
        if total_shots > 1:
            first_shot_time = df['shot_time'].iloc[0]   # First shot's time
            last_shot_time = df['shot_time'].iloc[-1]  # Last shot's time
            last_shot_ct = df['ACTUAL CT'].iloc[-1]   # Last shot's CT
            
            time_span_sec = (last_shot_time - first_shot_time).total_seconds()
            
            # New, more precise total duration
            total_runtime_sec = time_span_sec + last_shot_ct
            
        elif total_shots == 1:
            # If only one shot, duration is just its own cycle time
            total_runtime_sec = df['ACTUAL CT'].iloc[0]
        else:
            total_runtime_sec = 0 # No shots, no runtime
        
        # --- END: Modified Total Run Duration Logic ---

        mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else (100.0 if stop_events == 0 else 0.0)
        
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        # --- 4. Bucket Analysis Calculations ---
        df_for_runs = df[df['adj_ct_sec'] <= 28800].copy()
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")["ACTUAL CT"].sum().div(60).reset_index(name="duration_min")

        max_minutes = min(run_durations["duration_min"].max(), 240) if not run_durations.empty else 0
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        
        labels = [f"{edges[i]} to <{edges[i+1]}" for i in range(len(edges) - 1)]
        if labels:
            last_edge_start = edges[-2]
            labels[-1] = f"{last_edge_start}+"
        if edges and len(edges) > 1:
            edges[-1] = np.inf
        if not run_durations.empty:
            run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False, include_lowest=True)
        
        reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
        red_labels, blue_labels, green_labels = [], [], []
        for label in labels:
            try:
                lower_bound = int(label.split(' ')[0].replace('+', ''))
                if lower_bound < 60: red_labels.append(label)
                elif 60 <= lower_bound < 160: blue_labels.append(label)
                else: green_labels.append(label)
            except (ValueError, IndexError): continue
            
        bucket_color_map = {}
        for i, label in enumerate(red_labels): bucket_color_map[label] = reds[i % len(reds)]
        for i, label in enumerate(blue_labels): bucket_color_map[label] = blues[i % len(blues)]
        for i, label in enumerate(green_labels): bucket_color_map[label] = greens[i % len(greens)]
                
        # --- 5. Additional Metric Calculations (for Excel, etc.) ---
        avg_cycle_time_sec = production_time_sec / normal_shots if normal_shots > 0 else 0
        
        first_stop_event_index = df[df['stop_event'] == True].index.min()
        if pd.isna(first_stop_event_index):
            time_to_first_dt_sec = production_time_sec
        elif first_stop_event_index == 0:
             time_to_first_dt_sec = 0
        else:
             time_to_first_dt_sec = df.loc[:first_stop_event_index - 1, 'adj_ct_sec'].sum()
        
        # Note: 'production_run_sec' is the wall-clock time, which is different
        # from the 'total_runtime_sec' we just calculated.
        # We will keep this separate. The 'total_runtime_sec' is the one
        # used for Stability Index.
        production_run_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0
        
        # --- 6. Final Hourly Summary ---
        hourly_summary = self._calculate_hourly_summary(df)
        
        # --- 7. Compile Results Dictionary ---
        final_results = {
            "processed_df": df, "mode_ct": mode_ct_display, "total_shots": total_shots, "efficiency": efficiency,
            "stop_events": stop_events, "normal_shots": normal_shots, "mttr_min": mttr_min,
            "mtbf_min": mtbf_min, "stability_index": stability_index, "run_durations": run_durations,
            "bucket_labels": labels, "bucket_color_map": bucket_color_map, "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_sec, # This now uses the new calculation
            "production_time_sec": production_time_sec, 
            "downtime_sec": downtime_sec,
            "avg_cycle_time_sec": avg_cycle_time_sec,
            "time_to_first_dt_min": time_to_first_dt_sec / 60,
            "production_run_sec": production_run_sec, # This is wall-clock span, kept for reference
            "tot_down_time_sec": downtime_sec
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

# --- Calculation Helper Functions ---

def calculate_daily_summaries_for_week(df_week, tolerance, downtime_gap_tolerance, analysis_mode):
    """Rolls up daily metrics for the Weekly view."""
    daily_results_list = []
    for date in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date]
        if not df_day.empty:
            calc = RunRateCalculator(df_day.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'date': date, 'stability_index': res.get('stability_index', np.nan),
                           'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                           'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0),
                           'total_downtime_sec': res.get('downtime_sec', 0), 'uptime_min': res.get('production_time_sec', 0) / 60}
            daily_results_list.append(summary)
    return pd.DataFrame(daily_results_list) if daily_results_list else pd.DataFrame()

def calculate_weekly_summaries_for_month(df_month, tolerance, downtime_gap_tolerance, analysis_mode):
    """Rolls up weekly metrics for the Monthly view."""
    weekly_results_list = []
    for week in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week]
        if not df_week.empty:
            calc = RunRateCalculator(df_week.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'week': week, 'stability_index': res.get('stability_index', np.nan),
                           'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                           'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0),
                           'total_downtime_sec': res.get('downtime_sec', 0), 'uptime_min': res.get('production_time_sec', 0) / 60}
            weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance, downtime_gap_tolerance):
    """Iterates through a period's data, calculates metrics for each run, and returns a summary DataFrame."""
    run_summary_list = []
    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty:
            calc = RunRateCalculator(df_run.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
            res = calc.results
            
            total_shots = res.get('total_shots', 0)
            normal_shots = res.get('normal_shots', 0)
            stopped_shots = total_shots - normal_shots
            total_runtime_sec = res.get('total_runtime_sec', 0) # This now uses the new calculation
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


# ==============================================================================
# --- 3. PLOTTING FUNCTIONS ---
# ==============================================================================

def create_gauge(value, title, steps=None):
    """Creates a Plotly gauge chart."""
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps:
        gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else:
        gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct, time_agg='hourly'):
    """Creates the main Plotly bar chart of cycle times."""
    if df.empty:
        st.info("No shot data to display for this period."); return
    df = df.copy()
    df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB')
    
    # --- Start: Plot Time Jitter & Shift Logic ---
    
    # 1. Identify true downtime gaps (where bar height is time_diff_sec, not ACTUAL CT)
    #    These are the only bars we want to shift to the *previous* shot's timestamp.
    downtime_gap_indices = df[df['adj_ct_sec'] != df['ACTUAL CT']].index
    valid_downtime_gap_indices = downtime_gap_indices[downtime_gap_indices > 0]
    
    # 2. Identify all other "normal" shots (including abnormal CT shots).
    #    These are the ones that might overlap and need "jitter".
    normal_shot_indices = df.index.difference(valid_downtime_gap_indices)

    # 3. Apply jitter to "normal" shots
    if not normal_shot_indices.empty:
        # Calculate an index for each shot within its given second (0, 1, 2...)
        shot_index_in_second = df.loc[normal_shot_indices].groupby('shot_time').cumcount()
        
        # Calculate the time offset (e.g., 0s, 0.2s, 0.4s)
        # We use a small fraction of a second (0.2s) to separate them.
        time_offset = pd.to_timedelta(shot_index_in_second * 0.2, unit='s')
        
        # Apply the jittered time as the plot_time
        df.loc[normal_shot_indices, 'plot_time'] = df.loc[normal_shot_indices, 'shot_time'] + time_offset
    
    # 4. Apply time-shift to the true downtime gaps
    if not valid_downtime_gap_indices.empty:
        # Get the timestamp from the *previous* shot
        prev_shot_timestamps = df['shot_time'].shift(1).loc[valid_downtime_gap_indices]
        
        # Assign this previous timestamp as the plot_time.
        # We DON'T jitter these, as they represent a time *span*.
        df.loc[valid_downtime_gap_indices, 'plot_time'] = prev_shot_timestamps

    # 5. Handle the very first shot (index 0)
    #    It must be a "normal shot" (no previous shot to gap from)
    if 0 in normal_shot_indices:
         df.loc[0, 'plot_time'] = df.loc[0, 'shot_time']
    elif 0 in valid_downtime_gap_indices:
         # This case should be rare, but if the first shot is somehow a gap, just plot it
         df.loc[0, 'plot_time'] = df.loc[0, 'shot_time']
    else:
        # Fallback for index 0 if it's not in either (e.g., empty dataframe, though we checked)
        if 0 in df.index:
            df.loc[0, 'plot_time'] = df.loc[0, 'shot_time']
         
    # --- End: Plot Time Jitter & Shift Logic ---
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['plot_time'], y=df['adj_ct_sec'], marker_color=df['color'], name='Cycle Time', showlegend=False))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Normal Shot", marker_color='#3498DB', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Stopped Shot", marker_color=PASTEL_COLORS['red'], showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                           line=dict(width=0),
                           fill='tozeroy',
                           fillcolor='rgba(119, 221, 119, 0.3)',
                           name='Tolerance Band', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='New Run Start',
                           line=dict(color='purple', dash='dash', width=2), showlegend=True))

    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        for run_id, group in df.groupby('run_id'):
            if not group.empty:
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=group['shot_time'].min(), y0=group['lower_limit'].iloc[0],
                    x1=group['shot_time'].max(), y1=group['upper_limit'].iloc[0],
                    fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0
                )
    else:
        if not df.empty:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=df['shot_time'].min(), y0=lower_limit,
                x1=df['shot_time'].max(), y1=upper_limit,
                fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0
            )
            
    if 'run_label' in df.columns:
        run_starts = df.groupby('run_label')['shot_time'].min().sort_values()
        for start_time in run_starts.iloc[1:]:
            fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="purple")

    y_axis_cap_val = mode_ct if isinstance(mode_ct, (int, float)) else df['mode_ct'].mean() if 'mode_ct' in df else 50
    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)
    
    fig.update_layout(
        title="Run Rate Cycle Time",
        xaxis_title="Date / Time",
        yaxis_title="Cycle Time (sec)",
        yaxis=dict(range=[0, y_axis_cap]),
        bargap=0.05,
        xaxis=dict(showgrid=True),
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_trend_chart(df, x_col, y_col, title, x_title, y_title, y_range=[0, 101], is_stability=False):
    """Creates the line chart for stability, MTTR, or MTBF trends."""
    fig = go.Figure()
    marker_config = {}
    
    # Filter out NaN values from the plot
    plot_df = df.dropna(subset=[y_col])
    if plot_df.empty: # Don't plot if no valid data
        st.info(f"No valid data to plot for {title}.")
        return

    if is_stability:
        marker_config['color'] = [PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green'] for v in plot_df[y_col]]
        marker_config['size'] = 10
    
    fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y_col], mode="lines+markers", name=y_title,
                           line=dict(color="black" if is_stability else "royalblue", width=2), marker=marker_config))
    if is_stability:
        for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]:
            fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1, fillcolor=c, opacity=0.2, line_width=0, layer="below")
    
    fig.update_layout(title=title, 
                      yaxis=dict(title=y_title, range=y_range), 
                      xaxis_title=x_title.title(), # Apply title case
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_mttr_mtbf_chart(df, x_col, mttr_col, mtbf_col, shots_col, title):
    """Creates the dual-axis MTTR/MTBF chart."""
    if df is None or df.empty or df[shots_col].sum() == 0:
        return 

    mttr = df[mttr_col]
    mtbf = df[mtbf_col]
    shots = df[shots_col]
    x_axis = df[x_col]

    max_mttr = np.nanmax(mttr[np.isfinite(mttr)]) if not mttr.empty and any(np.isfinite(mttr)) else 0
    max_mtbf = np.nanmax(mtbf[np.isfinite(mtbf)]) if not mtbf.empty and any(np.isfinite(mtbf)) else 0
    y_range_mttr = [0, max_mttr * 1.15 if max_mttr > 0 else 10]
    y_range_mtbf = [0, max_mtbf * 1.15 if max_mtbf > 0 else 10]
    
    shots_min, shots_max = shots.min(), shots.max()
    
    if (shots_max - shots_min) == 0:
        scaled_shots = pd.Series([y_range_mtbf[1] / 2 if y_range_mtbf[1] > 0 else 0.5] * len(shots), index=shots.index)
    else:
        scaled_shots = (shots - shots_min) / (shots_max - shots_min) * (y_range_mtbf[1] * 0.9)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=x_axis, y=mttr, name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mtbf, name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=scaled_shots,
        name='Total Shots', 
        mode='lines+markers+text', 
        text=shots,
        textposition='top center',
        textfont=dict(color='blue'),
        line=dict(color='blue', dash='dot')), 
        secondary_y=True
    )
    
    fig.update_layout(
        title_text=title, 
        yaxis_title="MTTR (min)", 
        yaxis2_title="MTBF (min)",
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis=dict(range=y_range_mttr),
        yaxis2=dict(range=y_range_mtbf),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if x_col == 'hour':
        fig.update_layout(xaxis_title="Hour")
        
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# --- 4. TEXT ANALYSIS ENGINE ---
# ==============================================================================

def generate_detailed_analysis(analysis_df, overall_stability, overall_mttr, overall_mtbf, analysis_level):
    """Generates the main automated analysis summary."""
    if analysis_df is None or analysis_df.empty:
        return {"error": "Not enough data to generate a trend analysis."}

    stability_class = "good (above 70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (below 50%)"
    overall_summary = f"The overall stability for this period is <strong>{overall_stability:.1f}%</strong>, which is considered <strong>{stability_class}</strong>."

    predictive_insight = ""
    analysis_df_clean = analysis_df.dropna(subset=['stability'])
    if len(analysis_df_clean) > 1:
        volatility_std = analysis_df_clean['stability'].std()
        volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"
        
        half_point = len(analysis_df_clean) // 2
        first_half_mean = analysis_df_clean['stability'].iloc[:half_point].mean()
        second_half_mean = analysis_df_clean['stability'].iloc[half_point:].mean()
        
        trend_direction = "stable"
        if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
        elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"

        if trend_direction == "stable":
            predictive_insight = f"Performance has been <strong>{volatility_level}</strong> with no clear long-term upward or downward trend."
        else:
            predictive_insight = f"Performance shows a <strong>{trend_direction} trend</strong>, although this has been <strong>{volatility_level}</strong>."

    best_worst_analysis = ""
    if not analysis_df_clean.empty:
        best_performer = analysis_df_clean.loc[analysis_df_clean['stability'].idxmax()]
        worst_performer = analysis_df_clean.loc[analysis_df_clean['stability'].idxmin()]

        def format_period(period_value, level):
            if isinstance(period_value, (pd.Timestamp, pd.Period, pd.Timedelta)):
                return pd.to_datetime(period_value).strftime('%A, %b %d')
            if level == "Monthly": return f"Week {period_value}"
            if "Daily" in level: return f"{period_value}:00"
            return str(period_value)

        best_period_label = format_period(best_performer['period'], analysis_level)
        worst_period_label = format_period(worst_performer['period'], analysis_level)

        best_worst_analysis = (f"The best performance was during <strong>{best_period_label}</strong> (Stability: {best_performer['stability']:.1f}%), "
                               f"while the worst was during <strong>{worst_period_label}</strong> (Stability: {worst_performer['stability']:.1f}%). "
                               f"The key difference was the impact of stoppages: the worst period had {int(worst_performer['stops'])} stops with an average duration of {worst_performer.get('mttr', 0):.1f} min, "
                               f"compared to {int(best_performer['stops'])} stops during the best period.")

    pattern_insight = ""
    if not analysis_df_clean.empty and analysis_df_clean['stops'].sum() > 0:
        if "Daily" in analysis_level:
            peak_stop_hour = analysis_df_clean.loc[analysis_df_clean['stops'].idxmax()]
            pattern_insight = f"A notable pattern is the concentration of stop events around <strong>{int(peak_stop_hour['period'])}:00</strong>, which saw the highest number of interruptions ({int(peak_stop_hour['stops'])} stops)."
        else:
            mean_stability = analysis_df_clean['stability'].mean()
            std_stability = analysis_df_clean['stability'].std()
            outlier_threshold = mean_stability - (1.5 * std_stability)
            outliers = analysis_df_clean[analysis_df_clean['stability'] < outlier_threshold]
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
    """Generates text analysis for the bucket charts."""
    if complete_runs.empty or 'duration_min' not in complete_runs.columns:
        return "No completed runs to analyze for long-run trends."
    total_completed_runs = len(complete_runs)
    try:
        long_run_buckets = [label for label in bucket_labels if int(label.split(' ')[0].replace('+', '')) >= 60]
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
    """Generates text analysis for the MTTR/MTBF chart."""
    analysis_df_clean = analysis_df.dropna(subset=['stops', 'stability', 'mttr'])
    if analysis_df_clean is None or analysis_df_clean.empty or analysis_df_clean['stops'].sum() == 0 or len(analysis_df_clean) < 2:
        return "Not enough stoppage data to generate a detailed correlation analysis."
    if not all(col in analysis_df_clean.columns for col in ['stops', 'stability', 'mttr']):
        return "Could not perform analysis due to missing data columns."
    
    stops_stability_corr = analysis_df_clean['stops'].corr(analysis_df_clean['stability'])
    mttr_stability_corr = analysis_df_clean['mttr'].corr(analysis_df_clean['stability'])
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
        if "Daily" in level: return f"{period_value}:00"
        return str(period_value)
    if primary_driver_is_frequency:
        highest_stops_period_row = analysis_df_clean.loc[analysis_df_clean['stops'].idxmax()]
        period_label = format_period(highest_stops_period_row['period'], analysis_level)
        example_insight = (f"For example, the period with the most interruptions was <strong>{period_label}</strong>, which recorded <strong>{int(highest_stops_period_row['stops'])} stops</strong>. Prioritizing the root cause of these frequent events is recommended.")
    elif primary_driver_is_duration:
        highest_mttr_period_row = analysis_df_clean.loc[analysis_df_clean['mttr'].idxmax()]
        period_label = format_period(highest_mttr_period_row['period'], analysis_level)
        example_insight = (f"The period with the longest downtimes was <strong>{period_label}</strong>, where the average repair time was <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>. Investigating the cause of these prolonged stops is the top priority.")
    else:
        if not analysis_df_clean['mttr'].empty:
            highest_mttr_period_row = analysis_df_clean.loc[analysis_df_clean['mttr'].idxmax()]
            period_label = format_period(highest_mttr_period_row['period'], analysis_level)
            example_insight = (f"As an example, <strong>{period_label}</strong> experienced prolonged downtimes with an average repair time of <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>, highlighting the impact of long stops.")
    return f"<div style='line-height: 1.6;'><p>{corr_insight}</p><p>{example_insight}</p></div>"


# ==============================================================================
# --- 5. EXCEL EXPORT MODULE ---
# ==============================================================================

def generate_excel_report(all_runs_data, tolerance):
    """Creates the in-memory Excel file from a dictionary of run data."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # --- Define Formats ---
        header_format=workbook.add_format({'bold':True,'bg_color':'#002060','font_color':'white','align':'center','valign':'vcenter','border':1});sub_header_format=workbook.add_format({'bold':True,'bg_color':'#C5D9F1','border':1});label_format=workbook.add_format({'bold':True,'align':'left'});percent_format=workbook.add_format({'num_format':'0.0%','border':1});time_format=workbook.add_format({'num_format':'[h]:mm:ss','border':1});mins_format=workbook.add_format({'num_format':'0.00 "min"','border':1});secs_format=workbook.add_format({'num_format':'0.00 "sec"','border':1});data_format=workbook.add_format({'border':1});datetime_format=workbook.add_format({'num_format':'yyyy-mm-dd hh:mm:ss','border':1});error_format=workbook.add_format({'bold':True,'font_color':'red'})

        # --- Generate a Sheet for Each Run ---
        for run_id, data in all_runs_data.items():
            ws = workbook.add_worksheet(f"Run_{run_id:03d}")
            df_run = data['processed_df'].copy()
            start_row = 19

            col_map = {name: chr(ord('A') + i) for i, name in enumerate(df_run.columns)}

            shot_time_col_dyn = col_map.get('SHOT TIME')
            stop_col = col_map.get('STOP')
            stop_event_col = col_map.get('STOP EVENT')
            time_bucket_col = col_map.get('TIME BUCKET')
            cum_count_col_dyn = col_map.get('CUMULATIVE COUNT')
            run_dur_col_dyn = col_map.get('RUN DURATION')
            bucket_col_dyn = col_map.get('TIME BUCKET')
            time_diff_col_dyn = col_map.get('TIME DIFF SEC')
            first_col_for_count = shot_time_col_dyn if shot_time_col_dyn else 'A'

            data_cols_count = len(df_run.columns)
            helper_col_letter = chr(ord('A') + data_cols_count)
            ws.set_column(f'{helper_col_letter}:{helper_col_letter}', None, None, {'hidden': True})

            analysis_start_col_idx = data_cols_count + 2
            analysis_col_1 = chr(ord('A') + analysis_start_col_idx)
            analysis_col_2 = chr(ord('A') + analysis_start_col_idx + 1)
            analysis_col_3 = chr(ord('A') + analysis_start_col_idx + 2)

            missing_cols = []
            essential_cols = {
                'STOP': stop_col, 'STOP EVENT': stop_event_col,
                'TIME DIFF SEC': time_diff_col_dyn, 'CUMULATIVE COUNT': cum_count_col_dyn,
                'RUN DURATION': run_dur_col_dyn, 'TIME BUCKET': bucket_col_dyn,
                'SHOT TIME': shot_time_col_dyn
            }
            for name, letter in essential_cols.items():
                if not letter:
                    missing_cols.append(name)

            if missing_cols:
                ws.write('A5', f"Error: Missing columns for formulas: {', '.join(missing_cols)}", error_format)
            table_formulas_ok = not missing_cols

            # --- Layout Header ---
            ws.merge_range('A1:B1', data['equipment_code'], header_format)
            ws.write('A2', 'Date', label_format); ws.write('B2', f"{data['start_time']:%Y-%m-%d} to {data['end_time']:%Y-%m-%d}")
            ws.write('A3', 'Method', label_format); ws.write('B3', 'Every Shot')

            ws.write('E1', 'Mode CT', sub_header_format)
            mode_ct_val = data.get('mode_ct', 0)
            ws.write('E2', mode_ct_val if isinstance(mode_ct_val,(int,float)) else 0, secs_format)

            ws.write('F1', 'Outside L1', sub_header_format); ws.write('G1', 'Outside L2', sub_header_format); ws.write('H1', 'IDLE', sub_header_format)
            ws.write('F2', 'Lower Limit', label_format); ws.write('G2', 'Upper Limit', label_format); ws.write('H2', 'Stops', label_format)
            lower_limit_val = data.get('lower_limit'); upper_limit_val = data.get('upper_limit')
            ws.write('F3', lower_limit_val if lower_limit_val is not None else'N/A', secs_format)
            ws.write('G3', upper_limit_val if upper_limit_val is not None else'N/A', secs_format)

            if stop_col:
                ws.write_formula('H3', f"=SUM({stop_col}{start_row}:{stop_col}{start_row + len(df_run) - 1})", sub_header_format)
            else: ws.write('H3', 'N/A', sub_header_format)

            ws.write('K1', 'Total Shot Count', label_format); ws.write('L1', 'Normal Shot Count', label_format)
            ws.write_formula('K2', f"=COUNTA({first_col_for_count}{start_row}:{first_col_for_count}{start_row + len(df_run) - 1})", sub_header_format)
            ws.write_formula('L2', f"=K2-H3", sub_header_format)

            ws.write('K4', 'Efficiency', label_format); ws.write('L4', 'Stop Events', label_format)
            ws.write_formula('K5', f"=L2/K2", percent_format)
            if stop_event_col:
                ws.write_formula('L5', f"=SUM({stop_event_col}{start_row}:{stop_event_col}{start_row + len(df_run) - 1})", sub_header_format)
            else: ws.write('L5', 'N/A', sub_header_format)

            ws.write('F5', 'Tot Run Time (Calc)', label_format)
            ws.write('G5', 'Tot Down Time', label_format)
            ws.write('H5', 'Tot Prod Time', label_format)

            downtime_to_write = data.get('tot_down_time_sec', 0)
            if not isinstance(downtime_to_write, (int, float)):
                downtime_to_write = 0

            # Write the total_runtime_sec (which now uses the new calc)
            ws.write('F6', data.get('total_runtime_sec', 0) / 86400, time_format)
            ws.write('G6', downtime_to_write / 86400, time_format)
            ws.write('H6', data.get('production_time_sec', 0) / 86400, time_format)

            ws.write('F4', '', label_format); ws.write('G4', 'Down %', label_format); ws.write('H4', 'Prod %', label_format)
            ws.write('F7', '', data_format); ws.write_formula('G7', f"=IFERROR(G6/F6, 0)", percent_format); ws.write_formula('H7', f"=IFERROR(H6/F6, 0)", percent_format)

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
                if time_bucket_col:
                    ws.write_formula(f'{analysis_col_3}{15+i}', f'=COUNTIF({bucket_col_dyn}{start_row}:{bucket_col_dyn}{start_row + len(df_run) - 1},{i})', sub_header_format)
                else: ws.write(f'{analysis_col_3}{15+i}', 'N/A', sub_header_format)
            ws.write(f'{analysis_col_2}{16+max_bucket}', 'Grand Total', sub_header_format); ws.write_formula(f'{analysis_col_3}{16+max_bucket}', f"=SUM({analysis_col_3}16:{analysis_col_3}{15+max_bucket})", sub_header_format)

            # --- Data Table Header ---
            ws.write_row('A18', df_run.columns, header_format)

            # --- Write Static Data Values ---
            df_run_nan_filled = df_run.fillna(np.nan)
            for i, row_values in enumerate(df_run_nan_filled.itertuples(index=False)):
                current_row_excel_idx = start_row + i - 1

                for c_idx, value in enumerate(row_values):
                    col_name = df_run.columns[c_idx]

                    if col_name in ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET', 'TIME DIFF SEC']:
                        continue

                    cell_format = data_format
                    if col_name == 'STOP':
                        num_value = int(value) if pd.notna(value) else 0
                        ws.write_number(current_row_excel_idx, c_idx, num_value, cell_format)
                    elif col_name == 'STOP EVENT':
                        num_value = 1 if value == True else 0
                        ws.write_number(current_row_excel_idx, c_idx, num_value, cell_format)
                    elif isinstance(value, pd.Timestamp):
                        if pd.notna(value):
                            value_no_tz = value.tz_localize(None) if value.tzinfo is not None else value
                            ws.write_datetime(current_row_excel_idx, c_idx, value_no_tz, datetime_format)
                        else: ws.write_blank(current_row_excel_idx, c_idx, None, cell_format)
                    elif isinstance(value, (int, float, np.number)):
                        if col_name in ['ACTUAL CT', 'adj_ct_sec']: cell_format = secs_format
                        if pd.notna(value) and np.isfinite(value): ws.write_number(current_row_excel_idx, c_idx, value, cell_format)
                        else: ws.write_blank(current_row_excel_idx, c_idx, None, cell_format)
                    elif pd.isna(value): ws.write_blank(current_row_excel_idx, c_idx, None, cell_format)
                    else: ws.write_string(current_row_excel_idx, c_idx, str(value), cell_format)

            # --- Write Dynamic Table Formulas ---
            if table_formulas_ok:
                time_diff_col_idx = df_run.columns.get_loc('TIME DIFF SEC')
                cum_count_col_idx = df_run.columns.get_loc('CUMULATIVE COUNT')
                run_dur_col_idx = df_run.columns.get_loc('RUN DURATION')
                bucket_col_idx = df_run.columns.get_loc('TIME BUCKET')

                for i in range(len(df_run)):
                    row_num = start_row + i
                    prev_row = row_num - 1
                    current_row_zero_idx = start_row + i - 1

                    if i == 0:
                         first_diff_val = data.get('first_shot_time_diff', 0)
                         ws.write_number(current_row_zero_idx, time_diff_col_idx, first_diff_val, secs_format)
                    else:
                         formula = f'=IFERROR(({shot_time_col_dyn}{row_num}-{shot_time_col_dyn}{prev_row})*86400, 0)'
                         ws.write_formula(current_row_zero_idx, time_diff_col_idx, formula, secs_format)

                    if i == 0:
                        helper_formula = f'=IF({stop_col}{row_num}=0, {time_diff_col_dyn}{row_num}, 0)'
                    else:
                        helper_formula = f'=IF({stop_event_col}{row_num}=1, 0, IF({stop_col}{row_num}=0, {helper_col_letter}{prev_row}+{time_diff_col_dyn}{row_num}, {helper_col_letter}{prev_row}))'
                    ws.write_formula(current_row_zero_idx, data_cols_count, helper_formula)

                    cum_count_formula = f'=COUNTIF(${stop_event_col}${start_row}:${stop_event_col}{row_num},1)&"/"&IF({stop_event_col}{row_num}=1,"0 sec",TEXT({helper_col_letter}{row_num}/86400,"[h]:mm:ss"))'
                    ws.write_formula(current_row_zero_idx, cum_count_col_idx, cum_count_formula, data_format)

                    run_dur_formula = f'=IF({stop_event_col}{row_num}=1, IF({row_num}>{start_row}, {helper_col_letter}{prev_row}/86400, 0), "")'
                    ws.write_formula(current_row_zero_idx, run_dur_col_idx, run_dur_formula, time_format)

                    time_bucket_formula = f'=IF({stop_event_col}{row_num}=1, IF({row_num}>{start_row}, IFERROR(FLOOR({helper_col_letter}{prev_row}/60/20,1)+1, ""), ""), "")'
                    ws.write_formula(current_row_zero_idx, bucket_col_idx, time_bucket_formula, data_format)

            else:
                if cum_count_col_dyn: ws.write(f'{cum_count_col_dyn}{start_row}', "Formula Error", error_format)
                if time_diff_col_dyn: ws.write(f'{time_diff_col_dyn}{start_row}', "Formula Error", error_format)
                if run_dur_col_dyn: ws.write(f'{run_dur_col_dyn}{start_row}', "Formula Error", error_format)
                if bucket_col_dyn: ws.write(f'{bucket_col_dyn}{start_row}', "Formula Error", error_format)

            # --- Auto-adjust column widths ---
            for i, col_name in enumerate(df_run.columns):
                try:
                    max_len_data = df_run[col_name].astype(str).map(len).max()
                    max_len_data = 0 if pd.isna(max_len_data) else int(max_len_data)
                    width = max(len(str(col_name)), max_len_data)
                    ws.set_column(i, i, min(width + 2, 40))
                except Exception:
                    ws.set_column(i, i, len(str(col_name)) + 2)

    return output.getvalue()


def prepare_and_generate_run_based_excel(df_for_export, tolerance, downtime_gap_tolerance, run_interval_hours, tool_id_selection):
    """
    Wrapper function to split data into runs and prepare it for the Excel export.
    """
    try:
        base_calc = RunRateCalculator(df_for_export, tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())

        if df_processed.empty:
            st.error("Initial processing failed for Excel export.")
            return BytesIO().getvalue()

        split_col = 'time_diff_sec'
        if split_col not in df_processed.columns:
            st.error(f"Required column '{split_col}' not found. Cannot split into runs.")
            return BytesIO().getvalue()

        is_new_run = df_processed[split_col] > (run_interval_hours * 3600)
        df_processed['run_id'] = is_new_run.cumsum() + 1

        all_runs_data = {}
        desired_columns_base = [
            'SUPPLIER NAME', 'tool_id', 'SESSION ID', 'shot_time',
            'APPROVED CT', 'ACTUAL CT',
            'time_diff_sec', 'stop_flag', 'stop_event', 'run_group'
        ]
        formula_columns = ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET']

        for run_id, df_run_raw in df_processed.groupby('run_id'):
            try:
                run_calculator = RunRateCalculator(df_run_raw.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
                run_results = run_calculator.results

                if not run_results or 'processed_df' not in run_results or run_results['processed_df'].empty:
                    st.warning(f"Skipping empty/invalid Run ID {run_id} for Excel.")
                    continue

                run_results['equipment_code'] = df_run_raw['tool_id'].iloc[0] if 'tool_id' in df_run_raw.columns and not df_run_raw['tool_id'].empty else tool_id_selection
                run_results['start_time'] = df_run_raw['shot_time'].min()
                run_results['end_time'] = df_run_raw['shot_time'].max()
                run_results['mode_ct'] = run_results.get('mode_ct', 0)
                run_results['lower_limit'] = run_results.get('lower_limit', 0)
                run_results['upper_limit'] = run_results.get('upper_limit', np.inf)
                
                # This 'production_run_sec' is just wall-clock time, but the 'total_runtime_sec'
                # from the results dict now contains the *correct* calculation
                run_results['production_run_sec'] = (run_results['end_time'] - run_results['start_time']).total_seconds() if run_id > 0 else run_results.get('total_runtime_sec', 0)
                
                run_results['tot_down_time_sec'] = run_results.get('downtime_sec', 0)
                run_results['mttr_min'] = run_results.get('mttr_min', 0)
                run_results['mtbf_min'] = run_results.get('mtbf_min', 0)
                run_results['time_to_first_dt_min'] = run_results.get('time_to_first_dt_min', 0)
                run_results['avg_cycle_time_sec'] = run_results.get('avg_cycle_time_sec', 0)
                if not run_results['processed_df'].empty:
                     run_results['first_shot_time_diff'] = run_results['processed_df']['time_diff_sec'].iloc[0]
                else:
                     run_results['first_shot_time_diff'] = 0

                export_df = run_results['processed_df'].copy()
                for col in formula_columns:
                    if col not in export_df.columns:
                        export_df[col] = ''

                cols_to_keep = [col for col in desired_columns_base if col in export_df.columns]
                cols_to_keep_final = cols_to_keep + [col for col in formula_columns if col in export_df.columns]

                final_export_df = export_df[list(dict.fromkeys(cols_to_keep_final))].rename(columns={
                    'tool_id': 'EQUIPMENT CODE', 'shot_time': 'SHOT TIME',
                    'time_diff_sec': 'TIME DIFF SEC',
                    'stop_flag': 'STOP', 'stop_event': 'STOP EVENT'
                })

                final_desired_renamed = [
                    'SUPPLIER NAME', 'EQUIPMENT CODE', 'SESSION ID',
                    'Shot Sequence', # <-- Replace SHOT ID
                    'SHOT TIME',
                    'APPROVED CT', 'ACTUAL CT',
                    'TIME DIFF SEC', 'STOP', 'STOP EVENT', 'run_group',
                    'CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET'
                ]

                for col in final_desired_renamed:
                    if col not in final_export_df.columns:
                        final_export_df[col] = ''
                final_export_df = final_export_df[[col for col in final_desired_renamed if col in final_export_df.columns]]

                run_results['processed_df'] = final_export_df
                all_runs_data[run_id] = run_results

            except Exception as e:
                st.warning(f"Could not process Run ID {run_id} for Excel: {e}")
                import traceback
                st.text(traceback.format_exc())
                continue

        if not all_runs_data:
            st.error("No valid runs were processed for the Excel export.")
            return BytesIO().getvalue()

        excel_data = generate_excel_report(all_runs_data, tolerance)
        return excel_data

    except Exception as e:
        st.error(f"Error preparing data for run-based Excel export: {e}")
        import traceback
        st.text(traceback.format_exc())
        return BytesIO().getvalue()


# ==============================================================================
# --- 6. RISK TOWER MODULE ---
# ==============================================================================

@st.cache_data(show_spinner="Analyzing tool performance for Risk Tower...")
def calculate_risk_scores(df_all_tools):
    """
    Analyzes data for all tools, each within its own last 4-week window,
    using 'aggregate of runs' logic for cleaner MTTR/Stability.
    """
    id_col = "tool_id"
    initial_metrics = []
    
    RUN_INTERVAL_HOURS = 8
    RUN_INTERVAL_SEC = RUN_INTERVAL_HOURS * 3600

    # First pass: Calculate metrics for each tool
    for tool_id, df_tool in df_all_tools.groupby(id_col):
        if df_tool.empty or len(df_tool) < 10:
            continue

        calc_prepare = RunRateCalculator(df_tool, 0.05, 2.0)
        df_prepared = calc_prepare.results.get("processed_df")
        if df_prepared is None or df_prepared.empty:
            continue
            
        end_date = df_prepared['shot_time'].max()
        start_date = end_date - timedelta(weeks=4)
        df_period = df_prepared[(df_prepared['shot_time'] >= start_date) & (df_prepared['shot_time'] <= end_date)].copy() # Use .copy()

        if df_period.empty or len(df_period) < 10:
            continue

        is_new_run = df_period['time_diff_sec'] > RUN_INTERVAL_SEC
        df_period['run_id_risk'] = is_new_run.cumsum()
        df_period['run_label'] = df_period['run_id_risk'].apply(lambda x: f'Run_{x}')
        
        run_summary_df = calculate_run_summaries(df_period, 0.05, 2.0)
        
        if run_summary_df.empty:
            continue
                
        total_runtime_sec = run_summary_df['total_runtime_sec'].sum() # This now uses the new calculation
        production_time_sec = run_summary_df['production_time_sec'].sum()
        downtime_sec = run_summary_df['downtime_sec'].sum()
        stop_events = run_summary_df['stops'].sum()

        res_stability = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0
        res_mttr = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
        res_mtbf = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        
        weekly_stats = []
        df_period['week'] = df_period['shot_time'].dt.isocalendar().week
        
        for week_num, df_week_full in df_period.groupby('week'):
            df_week = df_week_full.copy()
            is_new_run_week = df_week['time_diff_sec'] > RUN_INTERVAL_SEC
            df_week['run_id_week'] = is_new_run_week.cumsum()
            df_week['run_label'] = df_week['run_id_week'].apply(lambda x: f'WeekRun_{x}')
            
            weekly_run_summary = calculate_run_summaries(df_week, 0.05, 2.0)
            
            if not weekly_run_summary.empty:
                w_tot_runtime = weekly_run_summary['total_runtime_sec'].sum() # This now uses the new calculation
                w_prod_time = weekly_run_summary['production_time_sec'].sum()
                w_stability = (w_prod_time / w_tot_runtime * 100) if w_tot_runtime > 0 else 100.0
                weekly_stats.append({'week': week_num, 'stability': w_stability})
            else:
                if not df_week.empty:
                     weekly_stats.append({'week': week_num, 'stability': 0.0})

        
        weekly_stabilities_df = pd.DataFrame(weekly_stats).sort_values('week')
        weekly_stabilities = weekly_stabilities_df['stability'].tolist()

        trend = "Stable"
        if len(weekly_stabilities) > 1 and weekly_stabilities[-1] < weekly_stabilities[0] * 0.95:
            trend = "Declining"

        initial_metrics.append({
            'Tool ID': tool_id,
            'Stability': res_stability,
            'MTTR': res_mttr,
            'MTBF': res_mtbf,
            'Weekly Stability': ' → '.join([f'{s:.0f}%' for s in weekly_stabilities]),
            'Trend': trend,
            'Analysis Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        })

    if not initial_metrics:
        return pd.DataFrame()

    # --- Second pass: Determine risk factors by comparing against GLOBAL averages ---
    metrics_df = pd.DataFrame(initial_metrics)
    
    overall_mttr_mean = metrics_df['MTTR'].mean()
    overall_mtbf_mean = metrics_df['MTBF'].mean()

    final_risk_data = []
    for _, row in metrics_df.iterrows():
        risk_score = row['Stability']
        if row['Trend'] == "Declining":
            risk_score -= 20
        
        primary_factor = "Low Stability"
        details = f"Overall stability is {row['Stability']:.1f}%."
        
        if row['Trend'] == "Declining":
            primary_factor = "Declining Trend"
            details = "Stability shows a consistent downward trend."
        elif row['Stability'] < 70 and overall_mttr_mean > 0 and row['MTTR'] > (overall_mttr_mean * 1.2):
            primary_factor = "High MTTR"
            details = f"Avg stop duration (MTTR) of {row['MTTR']:.1f} min is high (Avg: {overall_mttr_mean:.1f} min)."
        elif row['Stability'] < 70 and overall_mtbf_mean > 0 and row['MTBF'] < (overall_mtbf_mean * 0.8):
            primary_factor = "Frequent Stops"
            details = f"Frequent stops (MTBF of {row['MTBF']:.1f} min) is low (Avg: {overall_mtbf_mean:.1f} min)."

        final_risk_data.append({
            'Tool ID': row['Tool ID'],
            'Analysis Period': row['Analysis Period'],
            'Risk Score': max(0, risk_score),
            'Primary Risk Factor': primary_factor,
            'Weekly Stability': row['Weekly Stability'],
            'Details': details
        })

    if not final_risk_data:
        return pd.DataFrame()
            
    return pd.DataFrame(final_risk_data).sort_values('Risk Score', ascending=True).reset_index(drop=True)

def render_risk_tower(df_all_tools):
    """Renders the Risk Tower tab."""
    st.title("Run Rate Risk Tower")
    st.info("This tower analyzes performance over the last 4 weeks, identifying tools that require attention. Tools with the lowest scores are at the highest risk.")
    
    with st.expander("ℹ️ How the Risk Tower Works"):
        st.markdown("""
        The Risk Tower evaluates each tool based on its performance over its own most recent 4-week period of operation. Here’s how the metrics are calculated:

        - **Analysis Period**: Shows the exact 4-week date range used for each tool's analysis, based on its latest available data.
        - **Risk Score**: A performance indicator from 0-100.
            - It starts with the tool's overall **Stability Index (%)** for the period.
            - A **20-point penalty** is applied if the stability shows a declining trend.
        - **Primary Risk Factor**: Identifies the main issue affecting performance, prioritized as follows:
            1.  **Declining Trend**: If stability is worsening over time.
            2.  **High MTTR**: If the average stop duration is significantly longer than the average of all tools.
            3.  **Frequent Stops**: If the time between stops (MTBF) is significantly shorter than the average of all tools.
            4.  **Low Stability**: If none of the above are true, but overall stability is low.
        - **Color Coding**: Rows are colored based on the Risk Score:
            - <span style='background-color:#ff6961; color: black; padding: 2px 5px; border-radius: 5px;'>Red (0-50)</span>: High Risk
            - <span style='background-color:#ffb347; color: black; padding: 2px 5px; border-radius: 5px;'>Orange (51-70)</span>: Medium Risk
            - <span style='background-color:#77dd77; color: black; padding: 2px 5px; border-radius: 5px;'>Green (>70)</span>: Low Risk
        """, unsafe_allow_html=True)

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
    
    cols_order = ['Tool ID', 'Analysis Period', 'Risk Score', 'Primary Risk Factor', 'Weekly Stability', 'Details']
    display_df = risk_df[[col for col in cols_order if col in risk_df.columns]]

    st.dataframe(display_df.style.apply(style_risk, axis=1).format({'Risk Score': '{:.0f}'}), use_container_width=True, hide_index=True)


# ==============================================================================
# --- 7. MAIN DASHBOARD MODULE ---
# ==============================================================================

def render_dashboard(df_tool, tool_id_selection):
    """Renders the main Run Rate Dashboard tab."""
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
        - **Daily / Weekly / Monthly (by Run)**: A more precise analysis where the tolerance for stops is calculated from the Mode CT of each individual production run. A new run is identified after a stoppage longer than the selected 'Run Interval Threshold'.
        ---
        ### Sliders
        - **Tolerance Band**: Defines the acceptable CT range around the Mode CT.
        - **Run Interval Threshold**: Defines the max hours between shots before a new Production Run is identified.
        - **Remove Runs...**: (Only in 'by Run' mode) Filters out runs with fewer shots than the selected value.
        """)
        

    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        options=["Daily", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"]
    )

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.25, 0.01, help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Defines the max hours between shots before a new Production Run is identified.")
    
    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours):
        base_calc = RunRateCalculator(df, 0.01, 2.0)
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
            df_processed['date'] = df_processed['shot_time'].dt.date
            df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600)
            df_processed['run_id'] = is_new_run.cumsum()
        return df_processed

    df_processed = get_processed_data(df_tool, run_interval_hours)
    
    min_shots_filter = 1 
    if 'by Run' in analysis_level:
        st.sidebar.markdown("---")
        
        # Add the toggle
        enable_run_filter = st.sidebar.toggle("Filter Small Production Runs", value=False, help="Turn this on to show the slider for filtering out runs with few shots.")
        
        if enable_run_filter and not df_processed.empty:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if not run_shot_counts.empty else 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.sidebar.slider(
                    "Remove Runs with Fewer Than X Shots",
                    min_value=1,
                    max_value=max_shots,
                    value=default_value,
                    step=1,
                    help="Filters out smaller production runs to focus on more significant ones."
                )
        elif not enable_run_filter:
            min_shots_filter = 1 # Explicitly set to 1 if filter is off
    
    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)


    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")

    mode = 'by_run' if 'by Run' in analysis_level else 'aggregate'
    df_view = pd.DataFrame()

    # --- Date/Period Selection ---
    if "Daily" in analysis_level:
        st.header(f"Daily Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates:
            st.warning("No data available for any date."); st.stop()
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"

    elif "Weekly" in analysis_level:
        st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_weeks = sorted(df_processed["week"].unique())
        if not available_weeks:
            st.warning("No data available for any week."); st.stop()
        year = df_processed['shot_time'].iloc[0].year
        selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1)
        df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"

    elif "Monthly" in analysis_level:
        st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_months = sorted(df_processed["month"].unique())
        if not available_months:
            st.warning("No data available for any month."); st.stop()
        selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'))
        df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"

    elif "Custom Period" in analysis_level:
        st.header(f"Custom Period Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        min_date = df_processed['date'].min()
        max_date = df_processed['date'].max()
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)
        if start_date and end_date:
            mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
            df_view = df_processed[mask]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    # --- Run Filtering and Labeling ---
    if not df_view.empty:
        df_view = df_view.copy()
        # Only add run_label if in a 'by Run' mode
        if 'run_id' in df_view.columns and 'by Run' in analysis_level:
            # Create a consistent integer-based index for runs within the current view
            df_view['run_id_local'] = df_view.groupby('run_id').ngroup()
            unique_run_ids = df_view.sort_values('shot_time')['run_id_local'].unique()
            run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids)}
            df_view['run_label'] = df_view['run_id_local'].map(run_label_map)

    if 'by Run' in analysis_level and not df_view.empty:
        runs_before_filter = df_view['run_label'].nunique()
        run_shot_counts_in_view = df_view.groupby('run_label')['run_label'].transform('count')
        df_view = df_view[run_shot_counts_in_view >= min_shots_filter]
        runs_after_filter = df_view['run_label'].nunique()

        if runs_before_filter > 0:
            st.sidebar.metric(
                label="Runs Displayed",
                value=f"{runs_after_filter} / {runs_before_filter}",
                delta=f"-{runs_before_filter - runs_after_filter} filtered",
                delta_color="off"
            )

    if df_view.empty:
        st.warning(f"No data for the selected period (or all runs were filtered out).")
    else:
        # --- Main Calculation for Selected Period ---
        results = {}
        summary_metrics = {}

        if 'by Run' in analysis_level:
            run_summary_df_for_totals = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            
            if not run_summary_df_for_totals.empty:
                total_runtime_sec = run_summary_df_for_totals['total_runtime_sec'].sum()
                production_time_sec = run_summary_df_for_totals['production_time_sec'].sum()
                downtime_sec = run_summary_df_for_totals['downtime_sec'].sum()
                total_shots = run_summary_df_for_totals['total_shots'].sum()
                normal_shots = run_summary_df_for_totals['normal_shots'].sum()
                stop_events = run_summary_df_for_totals['stops'].sum()

                mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
                mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
                stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0
                efficiency = (normal_shots / total_shots) if total_shots > 0 else 0

                summary_metrics = {
                    'total_runtime_sec': total_runtime_sec, 'production_time_sec': production_time_sec,
                    'downtime_sec': downtime_sec, 'total_shots': total_shots,
                    'normal_shots': normal_shots, 'stop_events': stop_events,
                    'mttr_min': mttr_min, 'mtbf_min': mtbf_min,
                    'stability_index': stability_index, 'efficiency': efficiency,
                }
                sub_header = sub_header.replace("Summary for", "Summary for (Combined Runs)")

            calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
            results = calc.results
            summary_metrics.update({
                'min_lower_limit': results.get('min_lower_limit', 0), 'max_lower_limit': results.get('max_lower_limit', 0),
                'min_mode_ct': results.get('min_mode_ct', 0), 'max_mode_ct': results.get('max_mode_ct', 0),
                'min_upper_limit': results.get('min_upper_limit', 0), 'max_upper_limit': results.get('max_upper_limit', 0),
            })
        else:
            calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
            results = calc.results
            summary_metrics = results

        # --- Header & Download Button ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(sub_header)
        with col2:
            st.download_button(
                label="📥 Export Run-Based Report",
                data=prepare_and_generate_run_based_excel(
                    df_view.copy(), tolerance, downtime_gap_tolerance,
                    run_interval_hours, tool_id_selection
                ),
                file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        # --- Create Trend Summary DataFrame ---
        trend_summary_df = None
        trend_level = "" # Define trend_level
        if analysis_level == "Weekly":
            trend_level = "Daily"
            trend_summary_df = calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode)
        elif analysis_level == "Monthly":
            trend_level = "Weekly"
            trend_summary_df = calculate_weekly_summaries_for_month(df_view, tolerance, downtime_gap_tolerance, mode)
        elif "Custom Period" in analysis_level:
             trend_level = "Daily" # Default to daily for custom
             trend_summary_df = calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode)
        elif "by Run" in analysis_level:
            trend_level = "Run"
            trend_summary_df = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            if not trend_summary_df.empty:
                trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
        elif "Daily" in analysis_level:
            trend_level = "Hourly"
            trend_summary_df = results.get('hourly_summary', pd.DataFrame())
        
        # --- KPI Metrics Display ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0); prod_t = summary_metrics.get('production_time_sec', 0); down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            with col1: st.metric("Run Rate MTTR", f"{summary_metrics.get('mttr_min', 0):.1f} min")
            with col2: st.metric("Run Rate MTBF", f"{summary_metrics.get('mtbf_min', 0):.1f} min")
            with col3: st.metric("Total Run Duration", format_duration(total_d)) # This now uses the new calculation
            with col4:
                st.metric("Production Time", f"{format_duration(prod_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{format_duration(down_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_gauge(summary_metrics.get('efficiency', 0) * 100, "Run Rate Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': PASTEL_COLORS['red']}, {'range': [50, 70], 'color': PASTEL_COLORS['orange']},{'range': [70, 100], 'color': PASTEL_COLORS['green']}]
            c2.plotly_chart(create_gauge(summary_metrics.get('stability_index', 0), "Run Rate Stability Index (%)", steps=steps), use_container_width=True)
        
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0)
            s_s = t_s - n_s
            n_p = (n_s / t_s * 100) if t_s > 0 else 0
            s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{summary_metrics.get('stop_events', 0)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            if mode == 'by_run':
                min_ll = summary_metrics.get('min_lower_limit', 0); max_ll = summary_metrics.get('max_lower_limit', 0)
                c1.metric("Lower Limit Range (sec)", f"{min_ll:.2f} – {max_ll:.2f}")
                with c2:
                    min_mc = summary_metrics.get('min_mode_ct', 0); max_mc = summary_metrics.get('max_mode_ct', 0)
                    with st.container(border=True): st.metric("Mode Cycle Time Range (sec)", f"{min_mc:.2f} – {max_mc:.2f}")
                min_ul = summary_metrics.get('min_upper_limit', 0); max_ul = summary_metrics.get('max_upper_limit', 0)
                c3.metric("Upper Limit Range (sec)", f"{min_ul:.2f} – {max_ul:.2f}")
            else:
                mode_val = summary_metrics.get('mode_ct', 0)
                mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else mode_val
                c1.metric("Lower Limit (sec)", f"{summary_metrics.get('lower_limit', 0):.2f}")
                with c2:
                    with st.container(border=True): st.metric("Mode Cycle Time (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{summary_metrics.get('upper_limit', 0):.2f}")
        
        # --- Detailed Analysis Expander ---
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
                
                insights = generate_detailed_analysis(analysis_df, summary_metrics.get('stability_index', 0), summary_metrics.get('mttr_min', 0), summary_metrics.get('mtbf_min', 0), analysis_level)
                
                if "error" in insights: 
                    st.error(insights["error"])
                else:
                    st.components.v1.html(f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights['overall']}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights['predictive']}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights['best_worst']}</p> {'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> ' + insights['patterns'] + '</p>' if insights['patterns'] else ''}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights['recommendation']}</p></div>""", height=400, scrolling=True)

        # --- Breakdown Table Expander ---
        if analysis_level in ["Weekly", "Monthly", "Custom Period"] and "by Run" not in analysis_level:
            with st.expander(f"View {trend_level.title()} Breakdown Table", expanded=False):
                if trend_summary_df is not None and not trend_summary_df.empty:
                    d_df = trend_summary_df.copy()
                    if 'date' in d_df.columns:
                        d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                    d_df = get_renamed_summary_df(d_df)
                    st.dataframe(d_df.style.format({'Stability Index (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
        
        elif "by Run" in analysis_level:
            run_summary_df = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            with st.expander("View Run Breakdown Table", expanded=False):
                if run_summary_df is not None and not run_summary_df.empty:
                    d_df = run_summary_df.copy()
                    d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
                    d_df["Total shots"] = d_df['total_shots'].apply(lambda x: f"{x:,}")
                    d_df["Normal Shots"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Stop Events"] = d_df.apply(lambda r: f"{r['stops']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(format_duration)
                    d_df["Production Time (d/h/m)"] = d_df.apply(lambda r: f"{format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (d/h/m)"] = d_df.apply(lambda r: f"{format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df.rename(columns={
                        'run_label':'RUN ID', 'mode_ct':'Mode CT (for the run)',
                        'lower_limit':'Lower limit CT (sec)', 'upper_limit':'Upper Limit CT (sec)',
                        'mttr_min':'MTTR (min)', 'mtbf_min':'MTBF (min)',
                        'stability_index':'Stability (%)', 'stops':'STOPS'
                        }, inplace=True)
                    final_cols = ['RUN ID','Period (date/time from to)','Total shots','Normal Shots','Stop Events','Mode CT (for the run)','Lower limit CT (sec)','Upper Limit CT (sec)','Total Run duration (d/h/m)','Production Time (d/h/m)','Downtime (d/h/m)','MTTR (min)','MTBF (min)','Stability (%)']
                    final_cols = [col for col in final_cols if col in d_df.columns]
                    st.dataframe(d_df[final_cols].style.format({'Mode CT (for the run)':'{:.2f}','Lower limit CT (sec)':'{:.2f}','Upper Limit CT (sec)':'{:.2f}','MTTR (min)':'{:.1f}','MTBF (min)':'{:.1f}','Stability (%)':'{:.1f}'}), use_container_width=True)
        
        # --- Main Shot Bar Chart ---
        time_agg = 'hourly' if "Daily" in analysis_level else 'daily' if 'Weekly' in analysis_level else 'weekly'
        plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=time_agg)
        
        with st.expander("View Shot Data Table", expanded=False):
            # Conditionally select columns based on mode
            cols_to_show = ['shot_time', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event']
            rename_map = {
                'shot_time': 'Date / Time',
                'time_diff_sec': 'Time Difference (sec)',
                'stop_flag': 'Stop Flag',
                'stop_event': 'Stop Event'
            }
            
            # Add run_label only if it exists (which it now only will in 'by Run' modes)
            if 'run_label' in results['processed_df'].columns:
                cols_to_show.append('run_label')
                rename_map['run_label'] = 'Run ID'
                
            df_shot_data = results['processed_df'][cols_to_show].copy()
            df_shot_data.rename(columns=rename_map, inplace=True)
            st.dataframe(df_shot_data)
        
        st.markdown("---")
        
        # --- Detailed Trend Charts (Dynamic by Analysis Level) ---
        if "Daily" in analysis_level:
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
                    fig_b.update_xaxes(title_text="Duration (min)")
                    fig_b.update_yaxes(title_text="Occurrences")
                    st.plotly_chart(fig_b, use_container_width=True)
                    with st.expander("View Bucket Data", expanded=False): 
                        df_bucket_data = complete_runs.copy()
                        cols_to_show = ['run_group', 'duration_min', 'time_bucket', 'run_end_time']
                        df_bucket_data = df_bucket_data[[col for col in cols_to_show if col in df_bucket_data.columns]].rename(columns={
                            'run_group': 'Run Group', 'duration_min': 'Duration (min)',
                            'time_bucket': 'Time Bucket', 'run_end_time': 'Run End Date/ Time'
                        })
                        st.dataframe(df_bucket_data)
                else: st.info("No complete runs.")
            with c2:
                plot_trend_chart(trend_summary_df, 'hour', 'stability_index', "Hourly Stability Trend", "Hour of Day", "Stability (%)", is_stability=True)
                with st.expander("View Stability Data", expanded=False): 
                    st.dataframe(get_renamed_summary_df(trend_summary_df))
            
            st.subheader("Hourly Bucket Trend")
            if not complete_runs.empty:
                complete_runs['hour'] = complete_runs['run_end_time'].dt.hour
                pivot_df = pd.crosstab(index=complete_runs['hour'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
                pivot_df = pivot_df.reindex(pd.Index(range(24), name='hour'), fill_value=0)
                fig_hourly_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, 
                                           title='Hourly Distribution of Run Durations', barmode='stack', 
                                           color_discrete_map=results["bucket_color_map"], 
                                           labels={'hour': 'Hour', 'value': 'Number of Buckets', 'variable': 'Run Duration (min)'})
                st.plotly_chart(fig_hourly_bucket, use_container_width=True)
                with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            
            st.subheader("Hourly MTTR & MTBF Trend")
            hourly_summary = results.get('hourly_summary')
            if hourly_summary is not None and not hourly_summary.empty and hourly_summary['stops'].sum() > 0:
                plot_mttr_mtbf_chart(
                    df=hourly_summary, x_col='hour', mttr_col='mttr_min',
                    mtbf_col='mtbf_min', shots_col='total_shots',
                    title="Hourly MTTR & MTBF Trend"
                )
                with st.expander("View MTTR/MTBF Data", expanded=False): 
                    st.dataframe(get_renamed_summary_df(hourly_summary))
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = hourly_summary.copy().rename(columns={'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'})
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
        
        elif analysis_level in ["Weekly", "Monthly", "Custom Period"]:
            trend_level = "Daily" if "Weekly" in analysis_level else "Weekly" if "Monthly" in analysis_level else "Daily"
            st.header(f"{trend_level.title()} Trends for {analysis_level.split(' ')[0]}")
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
                    fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                    fig_b.update_xaxes(title_text="Duration (min)")
                    fig_b.update_yaxes(title_text="Occurrences")
                    st.plotly_chart(fig_b, use_container_width=True)
                    with st.expander("View Bucket Data", expanded=False): 
                        df_bucket_data = complete_runs.copy()
                        cols_to_show = ['run_group', 'duration_min', 'time_bucket', 'run_end_time']
                        df_bucket_data = df_bucket_data[[col for col in cols_to_show if col in df_bucket_data.columns]].rename(columns={
                            'run_group': 'Run Group', 'duration_min': 'Duration (min)',
                            'time_bucket': 'Time Bucket', 'run_end_time': 'Run End Date/ Time'
                        })
                        st.dataframe(df_bucket_data)
                else: st.info("No complete runs.")
            with c2:
                st.subheader(f"{trend_level.title()} Stability Trend")
                if summary_df is not None and not summary_df.empty:
                    x_col = 'date' if trend_level == "Daily" else 'week'
                    plot_trend_chart(summary_df, x_col, 'stability_index', f"{trend_level.title()} Stability Trend", trend_level, "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): 
                        st.dataframe(get_renamed_summary_df(summary_df))
                else: st.info(f"No {trend_level.lower()} data.")
            
            st.subheader(f"{trend_level.title()} Bucket Trend")
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
                fig_bucket_trend.update_layout(barmode='stack', title_text=f'{trend_level.title()} Distribution of Run Durations vs. Shot Count', 
                                               xaxis_title=trend_level.title(), yaxis_title='Number of Runs', 
                                               yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                st.plotly_chart(fig_bucket_trend, use_container_width=True)
                with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            
            st.subheader(f"{trend_level.title()} MTTR & MTBF Trend")
            if summary_df is not None and not summary_df.empty and summary_df['stops'].sum() > 0:
                x_col = 'date' if trend_level == "Daily" else 'week'
                plot_mttr_mtbf_chart(
                    df=summary_df, x_col=x_col, mttr_col='mttr_min',
                    mtbf_col='mtbf_min', shots_col='total_shots',
                    title=f"{trend_level.title()} MTTR, MTBF & Shot Count Trend"
                )
                with st.expander("View MTTR/MTBF Data", expanded=False): 
                    st.dataframe(get_renamed_summary_df(summary_df))
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
            run_summary_df = trend_summary_df 
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
                    fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                    fig_b.update_xaxes(title_text="Duration (min)")
                    fig_b.update_yaxes(title_text="Occurrences")
                    st.plotly_chart(fig_b, use_container_width=True)
                    with st.expander("View Bucket Data", expanded=False): 
                        df_bucket_data = complete_runs.copy()
                        cols_to_show = ['run_group', 'duration_min', 'time_bucket', 'run_end_time', 'run_label']
                        df_bucket_data = df_bucket_data[[col for col in cols_to_show if col in df_bucket_data.columns]].rename(columns={
                            'run_group': 'Run Group', 'duration_min': 'Duration (min)',
                            'time_bucket': 'Time Bucket', 'run_end_time': 'Run End Date/ Time',
                            'run_label': 'Run ID'
                        })
                        st.dataframe(df_bucket_data)
                else: st.info("No complete runs.")
            with c2:
                st.subheader("Stability per Production Run")
                if run_summary_df is not None and not run_summary_df.empty:
                    plot_trend_chart(run_summary_df, 'RUN ID', 'STABILITY %', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): 
                        st.dataframe(get_renamed_summary_df(run_summary_df))
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
                fig_bucket_trend.update_layout(barmode='stack', title_text='Distribution of Run Durations per Run vs. Shot Count', 
                                               xaxis_title='Run ID', yaxis_title='Number of Runs', 
                                               yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                st.plotly_chart(fig_bucket_trend, use_container_width=True)
                with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                if detailed_view:
                    with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                        st.markdown(generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            
            st.subheader("MTTR & MTBF per Production Run")
            if run_summary_df is not None and not run_summary_df.empty and run_summary_df['STOPS'].sum() > 0:
                plot_mttr_mtbf_chart(
                    df=run_summary_df, x_col='RUN ID', mttr_col='MTTR (min)',
                    mtbf_col='MTBF (min)', shots_col='Total Shots',
                    title="MTTR, MTBF & Shot Count per Run"
                )
                with st.expander("View MTTR/MTBF Data", expanded=False): 
                    st.dataframe(get_renamed_summary_df(run_summary_df))
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = run_summary_df.copy().rename(columns={'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'})
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)

# ==============================================================================
# --- 8. MAIN APP LOGIC (FILE UPLOAD & TAB RENDERING) ---
# ==============================================================================

st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload one or more Run Rate Excel files", type=["xlsx", "xls"], accept_multiple_files=True)

if not uploaded_files:
    st.info("👈 Upload one or more Excel files to begin.")
    st.stop()

@st.cache_data
def load_all_data(files):
    """Loads and combines all uploaded Excel files."""
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file)
            
            if "TOOLING ID" in df.columns:
                df.rename(columns={"TOOLING ID": "tool_id"}, inplace=True)
            elif "EQUIPMENT CODE" in df.columns:
                df.rename(columns={"EQUIPMENT CODE": "tool_id"}, inplace=True)

            if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
                datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
                df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
            elif "SHOT TIME" in df.columns:
                df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
            
            if "tool_id" in df.columns:
                df_list.append(df)
        except Exception as e:
            st.warning(f"Could not load file: {file.name}. Error: {e}")
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

df_all_tools = load_all_data(uploaded_files)

id_col = "tool_id"
if id_col not in df_all_tools.columns:
    st.error(f"None of the uploaded files contain a 'TOOLING ID' or 'EQUIPMENT CODE' column.")
    st.stop()

df_all_tools.dropna(subset=[id_col], inplace=True)
df_all_tools[id_col] = df_all_tools[id_col].astype(str)

# --- Sidebar Tool ID Selection ---
tool_ids = ["All Tools (Risk Tower)"] + sorted(df_all_tools[id_col].unique().tolist())
dashboard_tool_id_selection = st.sidebar.selectbox("Select Tool ID for Dashboard Analysis", tool_ids)

if dashboard_tool_id_selection == "All Tools (Risk Tower)":
    if len(tool_ids) > 1:
        first_tool = tool_ids[1]
        df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool]
        tool_id_for_dashboard_display = first_tool
    else:
        df_for_dashboard = pd.DataFrame()
        tool_id_for_dashboard_display = "No Tool Selected"
else:
    df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection]
    tool_id_for_dashboard_display = dashboard_tool_id_selection

# --- Tab Rendering ---
tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])

with tab1:
    render_risk_tower(df_all_tools)

with tab2:
    if not df_for_dashboard.empty:
        render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
    else:
        st.info("Select a specific Tool ID from the sidebar to view its dashboard.")