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
import math # Import math for isnan check

# --- Page and Code Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

# --- Constants ---
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- Core Calculation Class (for Dashboard) ---
class RunRateCalculator:
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.analysis_mode = analysis_mode
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

        # Always calculate the raw time difference between consecutive timestamps.
        df["time_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        # Handle the NaN for the first shot.
        if not df.empty and pd.isna(df.loc[0, "time_diff_sec"]):
            if "ACTUAL CT" in df.columns:
                df.loc[0, "time_diff_sec"] = df.loc[0, "ACTUAL CT"]
            else:
                df.loc[0, "time_diff_sec"] = 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
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
            df_for_mode_calc = df[df["ACTUAL CT"] < 999.9].copy()
            df_for_mode_calc['rounded_ct'] = df_for_mode_calc['ACTUAL CT'].round(0)
            mode_ct = df_for_mode_calc['rounded_ct'].mode().iloc[0] if not df_for_mode_calc['rounded_ct'].mode().empty else 0
            lower_limit = mode_ct * (1 - self.tolerance)
            upper_limit = mode_ct * (1 + self.tolerance)
            mode_ct_display = mode_ct

        # --- Two-Phase Stop Detection Logic ---
        # Phase 1: Check for abnormal cycle times.
        is_abnormal_cycle = (df["ACTUAL CT"] < lower_limit) | (df["ACTUAL CT"] > upper_limit)
        
        # Phase 2: Check for downtime gaps using the new adjustable tolerance.
        prev_actual_ct = df["ACTUAL CT"].shift(1)
        is_downtime_gap = df["time_diff_sec"] > (prev_actual_ct + self.downtime_gap_tolerance)

        # A shot is flagged as a stop if EITHER condition is true.
        df["stop_flag"] = np.where(is_abnormal_cycle | is_downtime_gap.fillna(False), 1, 0)
        
        if not df.empty:
            df.loc[0, "stop_flag"] = 0 # The first shot can never be a stop.
        
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)
        
        df["adj_ct_sec"] = np.where(df["stop_flag"] == 1, df["time_diff_sec"], df["ACTUAL CT"])

        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        
        downtime_sec = df.loc[df['stop_flag'] == 1, 'adj_ct_sec'].sum()
        mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0

        production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum()

        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        
        total_runtime_sec = production_time_sec + downtime_sec
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else (100.0 if stop_events == 0 else 0.0)
        
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        df_for_runs = df[df['adj_ct_sec'] <= 28800].copy()
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")["ACTUAL CT"].sum().div(60).reset_index(name="duration_min")

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
        
        reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
        
        red_labels, blue_labels, green_labels = [], [], []
        for label in labels:
            try:
                lower_bound = int(label.split('-')[0].replace('+', ''))
                if lower_bound < 60: red_labels.append(label)
                elif 60 <= lower_bound < 160: blue_labels.append(label)
                else: green_labels.append(label)
            except (ValueError, IndexError): continue
            
        bucket_color_map = {}
        for i, label in enumerate(red_labels): bucket_color_map[label] = reds[i % len(reds)]
        for i, label in enumerate(blue_labels): bucket_color_map[label] = blues[i % len(blues)]
        for i, label in enumerate(green_labels): bucket_color_map[label] = greens[i % len(greens)]
            
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

# --- Core Calculation Class (for Excel Export) ---
# This class is specifically for the detailed Excel export and contains the logic from the standalone app
class ExcelReportCalculator:
    def __init__(self, df: pd.DataFrame, tolerance: float):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
            datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
            df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
        elif "SHOT TIME" in df.columns:
            df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
        else:
            # Try to find 'shot_time' if it was already processed by the main app
            if "shot_time" not in df.columns:
                return pd.DataFrame()

        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        if df.empty: return pd.DataFrame()

        # --- LOGIC CHANGE (from simple app) ---
        # 1. Create the pure timestamp difference column for export
        df["time_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        if "ACTUAL CT" in df.columns:
            # Use the pure time_diff_sec for stop logic comparison
            time_diff_sec_calc = df["time_diff_sec"]
            prev_actual_ct = df["ACTUAL CT"].shift(1)
            rounding_buffer = 2.0

            # Check if prev_actual_ct is NaN before comparison
            is_a_stop_condition = (time_diff_sec_calc > (prev_actual_ct + rounding_buffer))
            is_a_stop_999 = (prev_actual_ct == 999.9)

            # Combine conditions safely, handling potential NaNs from shift()
            is_a_stop = np.where(pd.isna(prev_actual_ct), False, is_a_stop_999) | np.where(pd.isna(prev_actual_ct), False, is_a_stop_condition)

            # 2. Create a new internal column for stop/run logic
            df["logic_ct_diff"] = np.where(is_a_stop, time_diff_sec_calc, df["ACTUAL CT"])
        else:
            # If no 'ACTUAL CT', use the pure timestamp diff for logic
            df["logic_ct_diff"] = df["time_diff_sec"]

        # 3. Handle NaNs for both new columns in the first row
        if not df.empty:
            if pd.isna(df.loc[0, "time_diff_sec"]):
                df.loc[0, "time_diff_sec"] = 0 # First shot has no time diff

            if pd.isna(df.loc[0, "logic_ct_diff"]):
                # For the first row, logic_ct_diff should be ACTUAL CT if it exists, else 0
                df.loc[0, "logic_ct_diff"] = df.loc[0, "ACTUAL CT"] if "ACTUAL CT" in df.columns else 0

        # Ensure logic_ct_diff doesn't have NaNs that cause issues later
        df["logic_ct_diff"] = df["logic_ct_diff"].fillna(0)

        return df
        # --- END LOGIC CHANGE ---


    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns:
             # st.warning("Excel Exporter: Empty DataFrame or missing 'ACTUAL CT' after preparation.") # Removed Streamlit call
             print("Excel Exporter: Empty DataFrame or missing 'ACTUAL CT' after preparation.")
             return {}

        # --- LOGIC CHANGE: Use 'logic_ct_diff' for all calculations ---
        # Ensure logic_ct_diff exists and is numeric before filtering
        if 'logic_ct_diff' not in df.columns or not pd.api.types.is_numeric_dtype(df['logic_ct_diff']):
             # st.error("Excel Exporter: 'logic_ct_diff' column is missing or not numeric.") # Removed Streamlit call
             print("Excel Exporter: 'logic_ct_diff' column is missing or not numeric.")
             return {}

        # Filter out NaNs explicitly before comparison
        df_for_mode_calc = df.dropna(subset=['logic_ct_diff'])
        df_for_mode_calc = df_for_mode_calc[df_for_mode_calc["logic_ct_diff"] <= 28800]

        # Use dropna before mode calculation
        mode_ct_series = df_for_mode_calc["ACTUAL CT"].dropna().mode()
        mode_ct = mode_ct_series.iloc[0] if not mode_ct_series.empty else 0

        if not isinstance(mode_ct, (int, float, np.number)) or pd.isna(mode_ct):
             # st.warning(f"Excel Exporter: Could not determine valid Mode CT. Found: {mode_ct}. Defaulting to 0.") # Removed Streamlit call
             print(f"Excel Exporter: Could not determine valid Mode CT. Found: {mode_ct}. Defaulting to 0.")
             mode_ct = 0


        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)

        # Ensure numeric comparisons work even if limits are NaN (though mode_ct check should prevent this)
        lower_limit = lower_limit if pd.notna(lower_limit) else -np.inf
        upper_limit = upper_limit if pd.notna(upper_limit) else np.inf


        # Apply stop condition using dropna for safety
        logic_ct_diff_numeric = df['logic_ct_diff'].dropna()
        stop_condition = ((logic_ct_diff_numeric < lower_limit) | (logic_ct_diff_numeric > upper_limit)) & (logic_ct_diff_numeric <= 28800)

        # Use reindex to align the boolean series back to the original df index
        stop_condition_aligned = stop_condition.reindex(df.index, fill_value=False)

        df["stop_flag"] = np.where(stop_condition_aligned, 1, 0)

        if not df.empty:
            df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        total_shots = len(df)
        stop_events = df["stop_event"].sum()

        # Use .loc with boolean indexing which handles NaNs correctly
        downtime_sec = df.loc[df['stop_flag'] == 1, 'logic_ct_diff'].sum()
        production_time_sec = df.loc[df['stop_flag'] == 0, 'logic_ct_diff'].sum() # Summing logic_ct for non-stops too, as per original logic


        stop_durations = []
        is_in_stop = False
        current_stop_duration = 0
        # Iterate safely, checking for NaN
        for _, row in df.iterrows():
            stop_flag = row['stop_flag']
            logic_ct = row['logic_ct_diff']
            if pd.isna(stop_flag) or pd.isna(logic_ct):
                continue # Skip rows with NaN in critical columns

            if stop_flag == 1:
                is_in_stop = True
                current_stop_duration += logic_ct # Use logic_ct_diff
            elif is_in_stop and stop_flag == 0:
                stop_durations.append(current_stop_duration)
                is_in_stop = False
                current_stop_duration = 0

        # Add the last stop if the data ends during a stop
        if is_in_stop:
            stop_durations.append(current_stop_duration)

        total_downtime_from_stops = sum(stop_durations)
        mttr_sec = total_downtime_from_stops / stop_events if stop_events > 0 else 0

        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)

        # Ensure min/max don't fail on empty or all-NaT series
        shot_times_valid = df['shot_time'].dropna()
        total_runtime_sec = (shot_times_valid.max() - shot_times_valid.min()).total_seconds() if len(shot_times_valid) > 1 else 0

        normal_shots = total_shots - df["stop_flag"].sum() # sum() handles NaNs by treating them as 0
        efficiency = normal_shots / total_shots if total_shots > 0 else 0

        first_stop_index = df[df['stop_event']].index.min()
        # Ensure index selection doesn't create NaN issues
        time_to_first_dt_sec = df.loc[:first_stop_index-1, 'logic_ct_diff'].sum() if pd.notna(first_stop_index) and first_stop_index > 0 else production_time_sec
        avg_cycle_time = production_time_sec / normal_shots if normal_shots > 0 else 0

        df["run_group"] = df["stop_event"].cumsum()
        # Filter NaNs before groupby
        run_durations = df.dropna(subset=['logic_ct_diff'])[df["stop_flag"] == 0].groupby("run_group")["logic_ct_diff"].sum().div(60).reset_index(name="duration_min")
        # --- END LOGIC CHANGE ---

        # Final check for NaN in results before returning
        mttr_min_final = mttr_sec / 60
        mtbf_min_final = mtbf_min # already in minutes
        time_to_first_dt_min_final = time_to_first_dt_sec / 60
        avg_cycle_time_sec_final = avg_cycle_time

        return {
            "processed_df": df,
            "mode_ct": mode_ct if pd.notna(mode_ct) else 0,
            "lower_limit": lower_limit if pd.notna(lower_limit) else 0,
            "upper_limit": upper_limit if pd.notna(upper_limit) else 0,
            "total_shots": total_shots,
            "efficiency": efficiency if pd.notna(efficiency) else 0,
            "stop_events": stop_events,
            "normal_shots": normal_shots,
            "mttr_min": mttr_min_final if pd.notna(mttr_min_final) else 0,
            "mtbf_min": mtbf_min_final if pd.notna(mtbf_min_final) else 0,
            "production_run_sec": total_runtime_sec,
            "tot_down_time_sec": downtime_sec,
            "time_to_first_dt_min": time_to_first_dt_min_final if pd.notna(time_to_first_dt_min_final) else 0,
            "avg_cycle_time_sec": avg_cycle_time_sec_final if pd.notna(avg_cycle_time_sec_final) else 0,
            "run_durations": run_durations
        }


# --- UI Helper and Plotting Functions ---
def create_gauge(value, title, steps=None):
    # Ensure value is numeric, default to 0 if not
    value = value if pd.notna(value) and isinstance(value, (int, float, np.number)) else 0
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

    # The plot_time for stop events should be the time of the previous shot, to show the gap.
    df['plot_time'] = df['shot_time']
    stop_indices = df[df['stop_flag'] == 1].index
    if not stop_indices.empty:
        # Ensure we don't try to shift from a non-existent index -1
        valid_stop_indices = stop_indices[stop_indices > 0]
        # Use .loc for safe assignment
        shifted_times = df['shot_time'].shift(1).loc[valid_stop_indices]
        df.loc[valid_stop_indices, 'plot_time'] = shifted_times

    fig = go.Figure()

    # --- 1. Main Bar Chart Trace (without its own legend item) ---
    # Add this first to establish the axes correctly.
    # Ensure y values are numeric, replace NaNs if necessary
    y_values = pd.to_numeric(df['adj_ct_sec'], errors='coerce').fillna(0)
    fig.add_trace(go.Bar(x=df['plot_time'], y=y_values, marker_color=df['color'], name='Cycle Time', showlegend=False))


    # --- 2. Add Dummy Traces for a Custom Legend ---
    fig.add_trace(go.Bar(x=[None], y=[None], name="Normal Shot", marker_color='#3498DB', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Stopped Shot", marker_color=PASTEL_COLORS['red'], showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                           line=dict(width=0),
                           fill='tozeroy',
                           fillcolor='rgba(119, 221, 119, 0.3)',  # Pastel green with opacity
                           name='Tolerance Band', showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='New Run Start',
                           line=dict(color='purple', dash='dash', width=2), showlegend=True))

    # --- 3. Draw Correctly Scoped Tolerance Bands ---
    if 'lower_limit' in df.columns and 'run_id' in df.columns:
        # For 'by run' mode, draw a band for each run from its first to last actual shot time
        for run_id, group in df.groupby('run_id'):
             # Ensure group limits are valid numbers
             group_ll = group['lower_limit'].iloc[0] if pd.notna(group['lower_limit'].iloc[0]) else 0
             group_ul = group['upper_limit'].iloc[0] if pd.notna(group['upper_limit'].iloc[0]) else group_ll + 1 # Avoid zero height band
             if not group.empty:
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=group['shot_time'].min(), y0=group_ll,
                    x1=group['shot_time'].max(), y1=group_ul,
                    fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0
                )
    else:
         # Ensure global limits are valid numbers
         global_ll = lower_limit if pd.notna(lower_limit) else 0
         global_ul = upper_limit if pd.notna(upper_limit) else global_ll + 1
         # For aggregate mode, draw one band for the entire period's shots
         if not df.empty:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=df['shot_time'].min(), y0=global_ll,
                x1=df['shot_time'].max(), y1=global_ul,
                fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0
            )


    # --- 4. Add Vertical Lines for New Run Starts ---
    if 'run_label' in df.columns:
        run_starts = df.groupby('run_label')['shot_time'].min().sort_values()
        # Draw a line for each run start after the first one
        for start_time in run_starts.iloc[1:]:
             # Ensure start_time is a valid timestamp before adding line
             if pd.notna(start_time):
                  fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="purple")


    # Ensure mode_ct is valid before using for y-axis cap calculation
    if isinstance(mode_ct, (int, float, np.number)) and pd.notna(mode_ct):
         y_axis_cap_val = mode_ct
    elif 'mode_ct' in df and pd.api.types.is_numeric_dtype(df['mode_ct']) and not df['mode_ct'].dropna().empty:
         y_axis_cap_val = df['mode_ct'].dropna().mean()
    else:
         y_axis_cap_val = 50 # Default cap

    y_axis_cap = min(max(y_axis_cap_val * 2, 50), 500)


    fig.update_layout(
        title="Cycle Time per Shot vs. Tolerance",
        xaxis_title="Time",
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
    # Ensure the y-column exists and has numeric data
    if y_col not in df.columns or df[y_col].dropna().empty:
         st.warning(f"Not enough valid data in '{y_col}' to plot trend chart '{title}'.")
         return # Don't plot if no valid data
         
    fig = go.Figure()
    marker_config = {}
    
    # Ensure y-values are numeric for coloring logic
    y_values_numeric = pd.to_numeric(df[y_col], errors='coerce')
    
    if is_stability:
        # Handle potential NaNs when determining color
        marker_config['color'] = [PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green']
                                  for v in y_values_numeric.fillna(-1)] # Fill NaN with value outside ranges
        marker_config['size'] = 10

    # Plot using the numeric (and potentially NaN-handled) y-values
    fig.add_trace(go.Scatter(x=df[x_col], y=y_values_numeric, mode="lines+markers", name=y_title,
                           line=dict(color="black" if is_stability else "royalblue", width=2), marker=marker_config))
                           
    if is_stability:
        for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]:
            fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1, fillcolor=c, opacity=0.2, line_width=0, layer="below")
    fig.update_layout(title=title, yaxis=dict(title=y_title, range=y_range), xaxis_title=x_title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def plot_mttr_mtbf_chart(df, x_col, mttr_col, mtbf_col, shots_col, title):
    # Ensure required columns exist
    required_cols = [x_col, mttr_col, mtbf_col, shots_col]
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        st.warning(f"Cannot plot MTTR/MTBF chart '{title}' due to missing columns or empty data.")
        return

    # Check if there are any shots, handle potential NaNs
    if df[shots_col].sum(skipna=True) == 0:
        st.info(f"No shot data to display for MTTR/MTBF chart '{title}'.")
        return # Don't plot if no data


    # Get data series, convert to numeric and handle errors/NaNs
    mttr = pd.to_numeric(df[mttr_col], errors='coerce')
    mtbf = pd.to_numeric(df[mtbf_col], errors='coerce')
    shots = pd.to_numeric(df[shots_col], errors='coerce').fillna(0) # Shots default to 0 if NaN
    x_axis = df[x_col]

    # --- Scaling Logic ---
    # Define ranges for left and right Y-axes, handle potential NaN/inf values after coerce
    max_mttr = np.nanmax(mttr[np.isfinite(mttr)]) if not mttr.empty and any(np.isfinite(mttr)) else 0
    max_mtbf = np.nanmax(mtbf[np.isfinite(mtbf)]) if not mtbf.empty and any(np.isfinite(mtbf)) else 0
    y_range_mttr = [0, max_mttr * 1.15 if max_mttr > 0 else 10]
    y_range_mtbf = [0, max_mtbf * 1.15 if max_mtbf > 0 else 10]

    # Scale the 'shots' data to fit within the MTBF axis range
    shots_min, shots_max = shots.min(), shots.max()

    # Avoid division by zero if all shot counts are the same or only one point
    if (shots_max - shots_min) == 0 or len(shots) <= 1:
        # Set scaled shots to half the axis height if axis has height, else 0.5
        scaled_shots = pd.Series([y_range_mtbf[1] / 2 if y_range_mtbf[1] > 0 else 0.5] * len(shots), index=shots.index)
    else:
        # Scale to 90% of the axis height to avoid text labels going off-chart
        scaled_shots = (shots - shots_min) / (shots_max - shots_min) * (y_range_mtbf[1] * 0.9)


    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces, using the cleaned numeric series
    fig.add_trace(go.Scatter(x=x_axis, y=mttr, name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mtbf, name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4)), secondary_y=True)

    # Add the SCALED shot trace, but with ORIGINAL text labels (use original df column for text)
    # Ensure text labels are strings
    text_labels = df[shots_col].fillna(0).astype(int).astype(str)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=scaled_shots,  # Plot scaled data
        name='Total Shots',
        mode='lines+markers+text',
        text=text_labels, # Display original data as text
        textposition='top center',
        textfont=dict(color='blue'),
        line=dict(color='blue', dash='dot')),
        secondary_y=True # Plot on the right axis
    )

    fig.update_layout(
        title_text=title,
        yaxis_title="MTTR (min)",
        yaxis2_title="MTBF (min)",
        yaxis=dict(range=y_range_mttr),
        yaxis2=dict(range=y_range_mtbf), # Enforce the MTBF range
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
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

def calculate_daily_summaries_for_week(df_week, tolerance, downtime_gap_tolerance, analysis_mode):
    daily_results_list = []
    for date in sorted(df_week['date'].unique()):
        df_day = df_week[df_week['date'] == date]
        if not df_day.empty:
            calc = RunRateCalculator(df_day.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'date': date, 'stability_index': res.get('stability_index', np.nan),
                         'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                         'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}
            daily_results_list.append(summary)
    return pd.DataFrame(daily_results_list) if daily_results_list else pd.DataFrame()

def calculate_weekly_summaries_for_month(df_month, tolerance, downtime_gap_tolerance, analysis_mode):
    weekly_results_list = []
    for week in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week]
        if not df_week.empty:
            calc = RunRateCalculator(df_week.copy(), tolerance, downtime_gap_tolerance, analysis_mode=analysis_mode)
            res = calc.results
            summary = {'week': week, 'stability_index': res.get('stability_index', np.nan),
                         'mttr_min': res.get('mttr_min', np.nan), 'mtbf_min': res.get('mtbf_min', np.nan),
                         'stops': res.get('stop_events', 0), 'total_shots': res.get('total_shots', 0)}
            weekly_results_list.append(summary)
    return pd.DataFrame(weekly_results_list) if weekly_results_list else pd.DataFrame()

def calculate_run_summaries(df_period, tolerance, downtime_gap_tolerance):
    """Iterates through a period's data, calculates metrics for each run, and returns a summary DataFrame."""
    run_summary_list = []
    # Ensure 'run_label' column exists before grouping
    if 'run_label' not in df_period.columns:
        st.warning("Cannot calculate run summaries: 'run_label' column missing.")
        return pd.DataFrame()

    for run_label, df_run in df_period.groupby('run_label'):
        if not df_run.empty:
            calc = RunRateCalculator(df_run.copy(), tolerance, downtime_gap_tolerance, analysis_mode='aggregate') # Always use aggregate for individual run calcs
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

    # Ensure overall metrics are valid numbers
    overall_stability = overall_stability if pd.notna(overall_stability) else 0
    overall_mttr = overall_mttr if pd.notna(overall_mttr) else 0
    overall_mtbf = overall_mtbf if pd.notna(overall_mtbf) else 0


    stability_class = "good (above 70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (below 50%)"
    overall_summary = f"The overall stability for this period is <strong>{overall_stability:.1f}%</strong>, which is considered <strong>{stability_class}</strong>."

    predictive_insight = ""
    if len(analysis_df) > 1:
        # Use dropna before calculating std to avoid warnings/errors with NaN
        volatility_std = analysis_df['stability'].dropna().std()
        if pd.isna(volatility_std): # Handle case where all values are NaN or only one non-NaN value exists
             volatility_level = "undetermined"
        else:
            volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"

        half_point = len(analysis_df) // 2
        # Use dropna before mean calculation
        first_half_mean = analysis_df['stability'].iloc[:half_point].dropna().mean()
        second_half_mean = analysis_df['stability'].iloc[half_point:].dropna().mean()

        trend_direction = "stable"
        # Check if means are valid numbers before comparison
        if not pd.isna(first_half_mean) and not pd.isna(second_half_mean):
            if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
            elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"

        if trend_direction == "stable":
            predictive_insight = f"Performance has been <strong>{volatility_level}</strong> with no clear long-term upward or downward trend."
        else:
            predictive_insight = f"Performance shows a <strong>{trend_direction} trend</strong>, although this has been <strong>{volatility_level}</strong>."

    best_worst_analysis = ""
    # Use dropna before idxmax/idxmin
    if not analysis_df.empty and not analysis_df['stability'].dropna().empty:
        best_performer = analysis_df.loc[analysis_df['stability'].dropna().idxmax()]
        worst_performer = analysis_df.loc[analysis_df['stability'].dropna().idxmin()]

        # --- FIX: Updated format_period logic ---
        def format_period(period_value, level):
             # --- FIX: Check for By Run FIRST ---
             if "(by Run)" in level:
                 # The period value is already the run label string (e.g., "Run 001")
                 return str(period_value)
             # --- END FIX ---

             # Handle date/time like objects first
             if isinstance(period_value, (pd.Timestamp, datetime, pd.Period, pd.Timedelta)):
                 try:
                     return pd.to_datetime(period_value).strftime('%A, %b %d')
                 except Exception:
                     return "Invalid Date" # Fallback for unparseable dates

             # Handle potential float conversion for week/hour if needed, ensure it's int first
             if level == "Monthly":
                 return f"Week {int(period_value)}" if pd.notna(period_value) else "N/A"

             # Check level *exactly* for Daily (hourly) formatting - NOW ONLY IF NOT By Run
             if level == "Daily":
                 # Ensure it's a number that can be converted to int
                 if pd.notna(period_value) and isinstance(period_value, (int, float)) and not math.isnan(period_value):
                      return f"{int(period_value)}:00"
                 else:
                      return "N/A" # Was not a valid hour number

             # Fallback for other potential cases (like Custom Period if period is just a number)
             return str(period_value)
        # --- END FIX ---


        best_period_label = format_period(best_performer['period'], analysis_level)
        worst_period_label = format_period(worst_performer['period'], analysis_level)

        # Ensure stops and mttr are numeric before formatting
        worst_stops = int(worst_performer['stops']) if pd.notna(worst_performer['stops']) else 0
        worst_mttr = worst_performer.get('mttr', 0) if pd.notna(worst_performer.get('mttr', 0)) else 0
        best_stops = int(best_performer['stops']) if pd.notna(best_performer['stops']) else 0


        best_worst_analysis = (f"The best performance was during <strong>{best_period_label}</strong> (Stability: {best_performer['stability']:.1f}%), "
                                 f"while the worst was during <strong>{worst_period_label}</strong> (Stability: {worst_performer['stability']:.1f}%). "
                                 f"The key difference was the impact of stoppages: the worst period had {worst_stops} stops with an average duration of {worst_mttr:.1f} min, "
                                 f"compared to {best_stops} stops during the best period.")

    pattern_insight = ""
    # Use dropna before sum and idxmax
    if not analysis_df.empty and not analysis_df['stops'].dropna().empty and analysis_df['stops'].dropna().sum() > 0:
        # --- FIX: Only run hourly pattern logic if level is *exactly* Daily ---
        if analysis_level == "Daily": # <<< Only for hourly analysis
        # --- END FIX ---
            peak_stop_hour_row = analysis_df.loc[analysis_df['stops'].dropna().idxmax()]
            peak_period = peak_stop_hour_row['period']
            # Check if peak_period is valid before int()
            if pd.notna(peak_period) and isinstance(peak_period, (int, float)) and not math.isnan(peak_period):
                 pattern_insight = f"A notable pattern is the concentration of stop events around <strong>{int(peak_period)}:00</strong>, which saw the highest number of interruptions ({int(peak_stop_hour_row['stops'])} stops)."
            else:
                 pattern_insight = "Could not identify a peak stop hour due to data issues."
        # --- ADDED: Handle other levels including By Run ---
        else: # For Weekly, Monthly, Custom, By Run (including Daily by Run)
        # --- END ADDED ---
            # Use dropna before mean/std
            mean_stability = analysis_df['stability'].dropna().mean()
            std_stability = analysis_df['stability'].dropna().std()
            # Check if mean/std are valid
            if pd.notna(mean_stability) and pd.notna(std_stability) and std_stability > 0:
                outlier_threshold = mean_stability - (1.5 * std_stability)
                outliers = analysis_df[analysis_df['stability'] < outlier_threshold]
                if not outliers.empty and not outliers['stability'].dropna().empty:
                    worst_outlier = outliers.loc[outliers['stability'].dropna().idxmin()]
                    # Use updated format_period function
                    outlier_label = format_period(worst_outlier['period'], analysis_level)
                    pattern_insight = f"A key area of concern is <strong>{outlier_label}</strong>, which performed significantly below average and disproportionately affected the overall stability."
            elif pd.notna(mean_stability): # Handle case with std=0 or NaN std
                 pattern_insight = "Stability levels were consistent, no significant outliers detected."


    recommendation = ""
    # Ensure stability/mttr/mtbf are valid numbers before comparison
    overall_stability = overall_stability if pd.notna(overall_stability) else 0
    overall_mttr = overall_mttr if pd.notna(overall_mttr) else 0
    overall_mtbf = overall_mtbf if pd.notna(overall_mtbf) else 0

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
    # Ensure analysis_df is not None and required columns exist
    if analysis_df is None or analysis_df.empty:
        return "Not enough data for MTTR/MTBF correlation analysis."
        
    required_cols = ['stops', 'stability', 'mttr']
    if not all(col in analysis_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in analysis_df.columns]
        return f"Could not perform analysis due to missing data columns: {', '.join(missing)}"
        
    # Drop rows with NaN in critical columns before calculating correlation and sum
    analysis_df_clean = analysis_df.dropna(subset=required_cols)
    
    if analysis_df_clean.empty or analysis_df_clean['stops'].sum() == 0 or len(analysis_df_clean) < 2:
        return "Not enough valid stoppage data to generate a detailed correlation analysis."

    stops_stability_corr = analysis_df_clean['stops'].corr(analysis_df_clean['stability'])
    mttr_stability_corr = analysis_df_clean['mttr'].corr(analysis_df_clean['stability'])

    corr_insight = ""
    primary_driver_is_frequency = False
    primary_driver_is_duration = False

    # Check if correlations are valid numbers
    valid_stops_corr = pd.notna(stops_stability_corr)
    valid_mttr_corr = pd.notna(mttr_stability_corr)

    if valid_stops_corr and valid_mttr_corr:
        if abs(stops_stability_corr) > abs(mttr_stability_corr) * 1.5:
            primary_driver = "the **frequency of stops**"
            primary_driver_is_frequency = True
        elif abs(mttr_stability_corr) > abs(stops_stability_corr) * 1.5:
            primary_driver = "the **duration of stops**"
            primary_driver_is_duration = True
        else:
            primary_driver = "both the **frequency and duration of stops**"
        corr_insight = (f"This analysis suggests that <strong>{primary_driver}</strong> has the strongest impact on overall stability.")
    elif valid_stops_corr:
        corr_insight = "Analysis suggests the **frequency of stops** impacts stability, but correlation with duration could not be determined."
        primary_driver_is_frequency = True # Assume frequency if duration corr is invalid
    elif valid_mttr_corr:
        corr_insight = "Analysis suggests the **duration of stops** impacts stability, but correlation with frequency could not be determined."
        primary_driver_is_duration = True # Assume duration if frequency corr is invalid
    else:
        corr_insight = "Could not reliably determine the primary driver of stability due to data variability or insufficient stop events."

    example_insight = ""
    # Use the cleaned dataframe for idxmax
    if primary_driver_is_frequency and not analysis_df_clean.empty:
        highest_stops_period_row = analysis_df_clean.loc[analysis_df_clean['stops'].idxmax()]
        period_label = format_period(highest_stops_period_row['period'], analysis_level)
        example_insight = (f"For example, the period with the most interruptions was <strong>{period_label}</strong>, which recorded <strong>{int(highest_stops_period_row['stops'])} stops</strong>. Prioritizing the root cause of these frequent events is recommended.")
    elif primary_driver_is_duration and not analysis_df_clean.empty:
        highest_mttr_period_row = analysis_df_clean.loc[analysis_df_clean['mttr'].idxmax()]
        period_label = format_period(highest_mttr_period_row['period'], analysis_level)
        example_insight = (f"The period with the longest downtimes was <strong>{period_label}</strong>, where the average repair time was <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>. Investigating the cause of these prolonged stops is the top priority.")
    elif not primary_driver_is_frequency and not primary_driver_is_duration and not analysis_df_clean.empty: # If both influence or undetermined
         # Check if 'mttr' column has valid max before using idxmax
         if not analysis_df_clean['mttr'].dropna().empty:
             highest_mttr_period_row = analysis_df_clean.loc[analysis_df_clean['mttr'].dropna().idxmax()]
             period_label = format_period(highest_mttr_period_row['period'], analysis_level)
             example_insight = (f"As an example, <strong>{period_label}</strong> experienced prolonged downtimes with an average repair time of <strong>{highest_mttr_period_row['mttr']:.1f} minutes</strong>, highlighting the impact of long stops.")

    return f"<div style='line-height: 1.6;'><p>{corr_insight}</p><p>{example_insight}</p></div>"


# --- NEW Excel Generation Function (replaces old create_excel_export) ---
def generate_excel_report(all_runs_data, tolerance):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter', engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        workbook = writer.book

        # --- Define Formats ---
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

        # --- Generate a Sheet for Each Run ---
        for run_id, data in all_runs_data.items():
            # Sanitize sheet name if needed (xlsxwriter limitations)
            sheet_name = f"Run_{run_id:03d}"
            sheet_name = sheet_name[:31] # Max 31 chars
            sheet_name = sheet_name.replace('[', '').replace(']', '').replace(':', '').replace('*', '').replace('?', '').replace('/', '').replace('\\', '') # Remove invalid chars

            ws = workbook.add_worksheet(sheet_name)
            df_run = data.get('processed_df', pd.DataFrame()).copy() # Use get with default

            # Check if df_run is valid
            if df_run.empty:
                 ws.write('A1', f"Error: No processed data found for Run ID {run_id}", error_format)
                 continue # Skip this sheet if no data

            start_row = 19 # The row where the data table starts (1-indexed)

            # --- Dynamically find column letters ---
            col_map = {name: chr(ord('A') + i) for i, name in enumerate(df_run.columns)}

            # Columns for Table Formulas
            shot_time_col_dyn = col_map.get('SHOT TIME')

            # Columns for Header Formulas
            stop_col = col_map.get('STOP')
            stop_event_col = col_map.get('STOP EVENT')
            time_bucket_col = col_map.get('TIME BUCKET')
            first_col_for_count = shot_time_col_dyn if shot_time_col_dyn else 'A'

            cum_count_col_dyn = col_map.get('CUMULATIVE COUNT')
            run_dur_col_dyn = col_map.get('RUN DURATION')
            bucket_col_dyn = col_map.get('TIME BUCKET')
            time_diff_col_dyn = col_map.get('TIME DIFF SEC')


            # Helper column will be the one *after* the last data column
            data_cols_count = len(df_run.columns)
            helper_col_letter = chr(ord('A') + data_cols_count)
            ws.set_column(f'{helper_col_letter}:{helper_col_letter}', None, None, {'hidden': True})

            # --- Define Analysis Block Columns ---
            analysis_start_col_idx = data_cols_count + 2
            analysis_col_1 = chr(ord('A') + analysis_start_col_idx)     # Bucket #
            analysis_col_2 = chr(ord('A') + analysis_start_col_idx + 1) # Duration Range
            analysis_col_3 = chr(ord('A') + analysis_start_col_idx + 2) # Events Count

            # Check for missing essential columns for formulas
            missing_cols_formula = []
            if not stop_col: missing_cols_formula.append('STOP')
            if not stop_event_col: missing_cols_formula.append('STOP EVENT')
            if not time_bucket_col: missing_cols_formula.append('TIME BUCKET')
            if not time_diff_col_dyn: missing_cols_formula.append('TIME DIFF SEC')
            if not cum_count_col_dyn: missing_cols_formula.append('CUMULATIVE COUNT')
            if not run_dur_col_dyn: missing_cols_formula.append('RUN DURATION')
            if not shot_time_col_dyn: missing_cols_formula.append('SHOT TIME')

            if missing_cols_formula:
                ws.write('A5', f"Warning: Missing columns for formulas: {', '.join(missing_cols_formula)}. Some calculations may be static.", error_format)

            table_formulas_ok = not missing_cols_formula # True if no essential columns are missing

            # --- Layout ---
            ws.merge_range('A1:B1', data.get('equipment_code', 'N/A'), header_format)
            ws.write('A2', 'Date', label_format)
            start_t = data.get('start_time', pd.NaT)
            end_t = data.get('end_time', pd.NaT)
            date_range = "N/A"
            if pd.notna(start_t) and pd.notna(end_t):
                 date_range = f"{start_t.strftime('%Y-%m-%d')} to {end_t.strftime('%Y-%m-%d')}"
            ws.write('B2', date_range)

            ws.write('A3', 'Method', label_format)
            ws.write('B3', 'Every Shot')

            mode_ct_val = data.get('mode_ct', 0)
            ws.write('E1', 'Mode CT', sub_header_format)
            ws.write('E2', mode_ct_val, secs_format)

            ws.write('F1', 'Outside L1', sub_header_format); ws.write('G1', 'Outside L2', sub_header_format); ws.write('H1', 'IDLE', sub_header_format)
            ws.write('F2', 'Lower Limit', label_format); ws.write('G2', 'Upper Limit', label_format); ws.write('H2', 'Stops', label_format)

            ws.write_formula('F3', f'=E2*(1-{tolerance})', secs_format)
            ws.write_formula('G3', f'=E2*(1+{tolerance})', secs_format)
            if stop_col:
                ws.write_formula('H3', f"=SUM({stop_col}{start_row}:{stop_col}{start_row + len(df_run) - 1})", sub_header_format)
            else:
                ws.write('H3', data.get('total_shots', 0) - data.get('normal_shots', 0), sub_header_format) # Fallback


            ws.write('K1', 'Total Shot Count', label_format); ws.write('L1', 'Normal Shot Count', label_format)
            if first_col_for_count:
                ws.write_formula('K2', f"=COUNTA({first_col_for_count}{start_row}:{first_col_for_count}{start_row + len(df_run) - 1})", sub_header_format)
            else:
                 ws.write('K2', data.get('total_shots', 0), sub_header_format) # Fallback

            ws.write_formula('L2', f"=K2-H3", sub_header_format) # Uses K2 and H3 which have fallbacks

            ws.write('K4', 'Efficiency', label_format); ws.write('L4', 'Stop Events', label_format)
            ws.write_formula('K5', f"=L2/K2", percent_format) # Uses K2 and L2
            if stop_event_col:
                ws.write_formula('L5', f"=SUM({stop_event_col}{start_row}:{stop_event_col}{start_row + len(df_run) - 1})", sub_header_format)
            else:
                ws.write('L5', data.get('stop_events', 0), sub_header_format) # Fallback


            ws.write('F5', 'Tot Run Time', label_format); ws.write('G5', 'Tot Down Time', label_format)
            prod_run_sec = data.get('production_run_sec', 0)
            tot_down_sec = data.get('tot_down_time_sec', 0)
            ws.write('F6', prod_run_sec / 86400, time_format) # Convert seconds to Excel time fraction
            ws.write('G6', tot_down_sec / 86400, time_format)
            # Add checks for division by zero
            ws.write_formula('F7', f'=IF(F6>0,(F6-G6)/F6,0)', percent_format)
            ws.write_formula('G7', f'=IF(F6>0,G6/F6,0)', percent_format)


            ws.merge_range('K8:L8', 'Reliability Metrics', header_format)
            ws.write('K9', 'MTTR (Avg)', label_format); ws.write('L9', data.get('mttr_min', 0), mins_format)
            ws.write('K10', 'MTBF (Avg)', label_format); ws.write('L10', data.get('mtbf_min', 0), mins_format)

            ws.write('K11', 'Time to First DT', label_format)
            time_to_first_dt_min_val = data.get('time_to_first_dt_min', 0)
            if table_formulas_ok: # Only write formula if columns exist
                end_row_num = start_row + len(df_run) - 1
                match_range = f'{stop_event_col}{start_row}:{stop_event_col}{end_row_num}'
                index_range = f'{run_dur_col_dyn}:{run_dur_col_dyn}'
                # Adjusted row index (start_row - 1 = 18)
                formula = f'=IFERROR(INDEX({index_range}, {start_row - 1} + MATCH(1, {match_range}, 0)) * 1440, {time_to_first_dt_min_val})'
                ws.write_formula('L11', formula, mins_format)
            else:
                ws.write('L11', time_to_first_dt_min_val, mins_format) # Fallback

            ws.write('K12', 'Avg Cycle Time', label_format); ws.write('L12', data.get('avg_cycle_time_sec', 0), secs_format)

            # --- Time Bucket Analysis (Dynamically Placed) ---
            ws.merge_range(f'{analysis_col_1}14:{analysis_col_3}14', 'Time Bucket Analysis', header_format)
            ws.write(f'{analysis_col_1}15', 'Bucket', sub_header_format)
            ws.write(f'{analysis_col_2}15', 'Duration Range', sub_header_format)
            ws.write(f'{analysis_col_3}15', 'Events Count', sub_header_format)

            max_bucket = 20
            for i in range(1, max_bucket + 1):
                ws.write(f'{analysis_col_1}{15+i}', i, sub_header_format)
                ws.write(f'{analysis_col_2}{15+i}', f"{(i-1)*20} - {i*20} min", sub_header_format)
                if time_bucket_col: # Use the dynamically found column
                    ws.write_formula(f'{analysis_col_3}{15+i}', f'=COUNTIF({time_bucket_col}:{time_bucket_col},{i})', sub_header_format)
                else:
                    ws.write(f'{analysis_col_3}{15+i}', 'N/A', sub_header_format)

            ws.write(f'{analysis_col_2}{16+max_bucket}', 'Grand Total', sub_header_format)
            ws.write_formula(f'{analysis_col_3}{16+max_bucket}', f"=SUM({analysis_col_3}16:{analysis_col_3}{15+max_bucket})", sub_header_format)

            # --- Data Table ---
            ws.write_row('A18', df_run.columns, header_format)

            # Convert Timestamp if exists, handle NaT safely
            if 'SHOT TIME' in df_run.columns:
                 df_run['SHOT TIME'] = pd.to_datetime(df_run['SHOT TIME'], errors='coerce').dt.tz_localize(None) # Remove timezone if present

            # Replace remaining NaNs with empty string AFTER datetime conversion
            df_run.fillna('', inplace=True)

            # Write the entire DataFrame first
            for i, row in enumerate(df_run.to_numpy()):
                current_row_excel_idx = start_row + i # Excel row index (1-based)
                for c_idx, value in enumerate(row):
                    col_name = df_run.columns[c_idx]

                    # Skip columns that will be entirely replaced by formulas later
                    if col_name in ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET', 'TIME DIFF SEC']:
                        continue

                    # Attempt to write value with appropriate format
                    try:
                         if isinstance(value, (datetime, pd.Timestamp)):
                              # Check if it's NaT before writing
                              if pd.isna(value):
                                   ws.write_blank(current_row_excel_idx - 1, c_idx, None, data_format)
                              else:
                                   # Ensure naive datetime for xlsxwriter
                                   naive_dt = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
                                   if naive_dt.tzinfo is not None:
                                        naive_dt = naive_dt.replace(tzinfo=None)
                                   ws.write_datetime(current_row_excel_idx - 1, c_idx, naive_dt, datetime_format)
                         elif isinstance(value, (bool, np.bool_)):
                              ws.write_number(current_row_excel_idx - 1, c_idx, int(value), data_format)
                         elif isinstance(value, (int, float, np.number)):
                               # Check for NaN/inf specifically for numbers
                              if pd.isna(value) or np.isinf(value):
                                    ws.write_blank(current_row_excel_idx - 1, c_idx, None, data_format)
                              else:
                                    ws.write_number(current_row_excel_idx - 1, c_idx, value, data_format)
                         elif isinstance(value, str):
                                # Limit string length to avoid Excel errors
                                ws.write_string(current_row_excel_idx - 1, c_idx, value[:32767], data_format)
                         else:
                                # Fallback for other types
                                ws.write(current_row_excel_idx - 1, c_idx, str(value)[:32767], data_format)
                    except Exception as cell_error:
                         print(f"Error writing cell ({current_row_excel_idx-1}, {c_idx}), value: {value}, type: {type(value)}. Error: {cell_error}")
                         try:
                              # Try writing as blank on error
                              ws.write_blank(current_row_excel_idx - 1, c_idx, None, data_format)
                         except:
                              pass # Ignore if writing blank fails


            # --- Write Dynamic Table Formulas ---
            if table_formulas_ok:
                for i in range(len(df_run)):
                    row_num = start_row + i
                    prev_row = row_num - 1

                    # --- TIME DIFF SEC FORMULA ---
                    time_diff_cell_ref = f'{time_diff_col_dyn}{row_num}'
                    if i == 0:
                        # Write 0 for the first row's time diff
                        ws.write_number(row_num - 1, df_run.columns.get_loc('TIME DIFF SEC'), 0, secs_format)
                    else:
                        # Write the formula for subsequent rows
                        formula = f'=IFERROR(({shot_time_col_dyn}{row_num}-{shot_time_col_dyn}{prev_row})*86400, 0)'
                        ws.write_formula(time_diff_cell_ref, formula, secs_format)

                    # --- Helper column for run duration sum ---
                    helper_cell_ref = f'{helper_col_letter}{row_num}'
                    if i == 0:
                        # Reference the TIME DIFF SEC cell we just wrote/calculated
                        helper_formula = f'=IF({stop_col}{row_num}=0, {time_diff_cell_ref}, 0)'
                    else:
                        # Reference the TIME DIFF SEC cell and the previous helper cell
                        helper_formula = f'=IF({stop_event_col}{row_num}=1, 0, {helper_col_letter}{prev_row}) + IF({stop_col}{row_num}=0, {time_diff_cell_ref}, 0)'
                    ws.write_formula(helper_cell_ref, helper_formula) # No format needed for hidden helper

                    # --- CUMULATIVE COUNT FORMULA ---
                    cum_count_cell_ref = f'{cum_count_col_dyn}{row_num}'
                    cum_count_formula = f'=COUNTIF(${stop_event_col}${start_row}:${stop_event_col}{row_num},1) & "/" & IF({stop_event_col}{row_num}=1, "0 sec", TEXT({helper_cell_ref}/86400, "[h]:mm:ss"))'
                    ws.write_formula(cum_count_cell_ref, cum_count_formula, data_format)

                    # --- RUN DURATION FORMULA ---
                    run_dur_cell_ref = f'{run_dur_col_dyn}{row_num}'
                    if i == 0:
                        run_dur_formula = f'=IF({stop_event_col}{row_num}=1, 0, "")' # Special case for first row
                    else:
                        run_dur_formula = f'=IF({stop_event_col}{row_num}=1, {helper_col_letter}{prev_row}/86400, "")'
                    ws.write_formula(run_dur_cell_ref, run_dur_formula, time_format)

                    # --- TIME BUCKET FORMULA ---
                    bucket_cell_ref = f'{bucket_col_dyn}{row_num}'
                    if i == 0:
                         time_bucket_formula = f'=IF({stop_event_col}{row_num}=1, IFERROR(FLOOR(0/60/20, 1) + 1, ""), "")' # Special case for first row
                    else:
                        # Reference previous helper cell
                        time_bucket_formula = f'=IF({stop_event_col}{row_num}=1, IFERROR(FLOOR({helper_col_letter}{prev_row}/60/20, 1) + 1, ""), "")'
                    ws.write_formula(bucket_cell_ref, time_bucket_formula, data_format)


            # Auto-fit columns (adjust width calculation)
            for i, col_name in enumerate(df_run.columns):
                try:
                    # Calculate max length safely, handling potential non-string data and NaNs
                    header_len = len(str(col_name))
                    # Convert column to string, fill NA, get max length
                    max_len = df_run[col_name].astype(str).fillna('').map(len).max()
                    # Ensure max_len is a number, default to header_len if column is empty/all NaN
                    if pd.isna(max_len):
                         width = header_len
                    else:
                         width = max(header_len, int(max_len))

                except Exception as e:
                    print(f"Error calculating width for column '{col_name}': {e}")
                    width = len(str(col_name)) # Fallback

                # Apply constraints
                col_width = min(max(width + 2, 10), 50) # Set min 10, max 50 width
                ws.set_column(i, i, col_width)


    return output.getvalue()


# --- NEW Wrapper Function to prepare data for the new Excel export ---
def generate_run_based_excel_export(df_for_export, tolerance, run_interval_hours, tool_id_selection):
    """
    This function replicates the logic from the simple run_rate_app.py UI
    to prepare the 'all_runs_data' dict needed by 'generate_excel_report'.
    """
    
    # Ensure tool_id is present for grouping later if needed by ExcelReportCalculator
    if "tool_id" not in df_for_export.columns and "EQUIPMENT CODE" in df_for_export.columns:
         df_for_export = df_for_export.rename(columns={"EQUIPMENT CODE": "tool_id"})
    elif "tool_id" not in df_for_export.columns and "TOOLING ID" in df_for_export.columns:
         df_for_export = df_for_export.rename(columns={"TOOLING ID": "tool_id"})
    elif "tool_id" not in df_for_export.columns:
         # Assign the selected tool ID if the column is completely missing
         df_for_export['tool_id'] = tool_id_selection

    # 1. Base calculation to get processed df using the Excel-specific calculator
    try:
        # Use a copy to avoid modifying the original df_view passed to the dashboard
        base_calc = ExcelReportCalculator(df_for_export.copy(), tolerance)
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
    except Exception as e:
        st.error(f"Error in Excel base calculation: {e}")
        return BytesIO().getvalue()

    if df_processed.empty:
        st.error("Could not process data for Excel export. Check source data and 'ACTUAL CT' column.")
        return BytesIO().getvalue()

    # 2. Split into runs based on the interval threshold
    # Use logic_ct_diff if available, otherwise fall back gracefully
    if 'logic_ct_diff' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['logic_ct_diff']):
         split_col = 'logic_ct_diff'
    elif 'time_diff_sec' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['time_diff_sec']):
         split_col = 'time_diff_sec'
    else:
         st.error("Cannot determine run splits for Excel export: Missing required time difference columns.")
         return BytesIO().getvalue() # Cannot proceed without a split column

    # Ensure the split column has no NaNs before comparison
    df_processed[split_col] = df_processed[split_col].fillna(0)
    is_new_run = df_processed[split_col] > (run_interval_hours * 3600)
    df_processed['run_id'] = is_new_run.cumsum()


    all_runs_data = {}
    # Define desired columns, ensure 'tool_id' is included for equipment code fallback
    desired_columns_base = [
        'SUPPLIER NAME', 'tool_id', 'SESSION ID', 'SHOT ID', 'shot_time',
        'APPROVED CT', 'ACTUAL CT',
        'time_diff_sec', 'stop_flag', 'stop_event', 'run_group'
    ]
    formula_columns = ['CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET']

    # 3. Loop through each run and calculate its specific metrics
    for run_id, df_run_raw in df_processed.groupby('run_id'):
        try:
             # Ensure the loop uses a copy for calculations
            run_calculator = ExcelReportCalculator(df_run_raw.copy(), tolerance)
            run_results = run_calculator.results

            if not run_results or 'processed_df' not in run_results or run_results['processed_df'].empty:
                 st.warning(f"Excel Export: Calculation failed or produced no data for Run ID {run_id}. Skipping.")
                 continue

            # Safely get equipment code, default to selection if 'tool_id' is missing in this specific run's df
            equipment_code = df_run_raw['tool_id'].iloc[0] if 'tool_id' in df_run_raw.columns and not df_run_raw.empty else tool_id_selection
            run_results['equipment_code'] = equipment_code

            run_results['start_time'] = df_run_raw['shot_time'].min()
            run_results['end_time'] = df_run_raw['shot_time'].max()

            export_df = run_results['processed_df'].copy()

            # Add placeholder columns if they weren't created (e.g., due to errors)
            for col in formula_columns:
                if col not in export_df:
                    export_df[col] = ''

            # Filter for existing base columns + add formula columns
            columns_to_export = [col for col in desired_columns_base if col in export_df.columns]
            columns_to_export.extend(formula_columns)

            final_export_df = export_df[columns_to_export].rename(columns={
                'tool_id': 'EQUIPMENT CODE', # Rename after filtering
                'shot_time': 'SHOT TIME',
                'time_diff_sec': 'TIME DIFF SEC',
                'stop_flag': 'STOP',
                'stop_event': 'STOP EVENT'
            })


            # Define the final desired order and columns
            final_desired_renamed = [
                'SUPPLIER NAME', 'EQUIPMENT CODE', 'SESSION ID', 'SHOT ID', 'SHOT TIME',
                'APPROVED CT', 'ACTUAL CT',
                'TIME DIFF SEC', 'STOP', 'STOP EVENT', 'run_group',
                'CUMULATIVE COUNT', 'RUN DURATION', 'TIME BUCKET'
            ]

            # Add any missing final columns as blank strings
            for col in final_desired_renamed:
                if col not in final_export_df.columns:
                    final_export_df[col] = ''

            # Reorder and select final columns, only keeping those that exist
            final_export_df = final_export_df[[col for col in final_desired_renamed if col in final_export_df.columns]]


            run_results['processed_df'] = final_export_df
            all_runs_data[run_id] = run_results

        except Exception as e:
            st.warning(f"Could not process Run ID {run_id} for Excel export: {e}")
            import traceback
            st.exception(traceback.format_exc()) # Show full traceback for debugging

    if not all_runs_data:
        st.error("No valid runs were processed for the Excel export.")
        return BytesIO().getvalue()

    # 4. Generate the Excel file using the new function
    try:
        excel_bytes = generate_excel_report(all_runs_data, tolerance)
        return excel_bytes
    except Exception as e:
        st.error(f"Error generating final Excel file: {e}")
        import traceback
        st.exception(traceback.format_exc()) # Show full traceback
        return BytesIO().getvalue()



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
        - **Daily / Weekly / Monthly (by Run)**: A more precise analysis where the tolerance for stops is calculated from the Mode CT of each individual production run. A new run is identified after a stoppage longer than the selected 'Run Interval Threshold'.
        ---
        ### Sliders
        - **Tolerance Band**: Defines the acceptable CT range around the Mode CT.
        - **Run Interval Threshold**: Defines the max hours between shots before a new Production Run is identified.
        - **Remove Runs...**: (Only in 'by Run' mode) Filters out runs with fewer shots than the selected value.
        """)


    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        ["Daily", "Daily (by Run)", "Weekly", "Monthly", "Custom Period", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"]
    )

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Defines the max hours between shots before a new Production Run is identified.")

    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours):
        # Use the Dashboard's RunRateCalculator for initial processing
        base_calc = RunRateCalculator(df, 0.01, 2.0) # Use default tolerances here
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
            df_processed['date'] = df_processed['shot_time'].dt.date
            df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            # Only create the base run identifier here
            # Ensure time_diff_sec exists and handle NaNs before comparison
            if 'time_diff_sec' in df_processed.columns:
                 df_processed['time_diff_sec'] = df_processed['time_diff_sec'].fillna(0)
                 is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600)
                 df_processed['run_id'] = is_new_run.cumsum()
            else:
                 # Cannot determine runs if time_diff_sec is missing
                 df_processed['run_id'] = 0 # Assign a default run_id
        return df_processed


    df_processed = get_processed_data(df_tool.copy(), run_interval_hours) # Pass a copy

    min_shots_filter = 1
    if 'by Run' in analysis_level:
        st.sidebar.markdown("---")
        if not df_processed.empty and 'run_id' in df_processed.columns:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if not run_shot_counts.empty else 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.sidebar.slider(
                    "Remove Runs with Fewer Than X Shots",
                    min_value=1,
                    max_value=max(1, max_shots), # Ensure max_value is at least 1
                    value=default_value,
                    step=1,
                    help="Filters out smaller production runs to focus on more significant ones."
                )

    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)


    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")

    mode = 'by_run' if 'by Run' in analysis_level else 'aggregate'
    df_view = pd.DataFrame()

    # --- Date/Time Filtering ---
    if "Daily" in analysis_level:
        st.header(f"Daily Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        # Ensure 'date' column exists and handle potential errors
        if 'date' in df_processed.columns:
             available_dates = sorted(df_processed["date"].dropna().unique())
             if not available_dates:
                  st.warning("No valid dates found in the data."); st.stop()
             selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
             df_view = df_processed[df_processed["date"] == selected_date]
             sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"
        else:
             st.error("Date column missing, cannot perform Daily analysis."); st.stop()


    elif "Weekly" in analysis_level:
        st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        if 'week' in df_processed.columns and 'shot_time' in df_processed.columns:
             available_weeks = sorted(df_processed["week"].dropna().unique())
             if not available_weeks:
                  st.warning("No valid weeks found in the data."); st.stop()
             # Get year from the first valid timestamp
             first_valid_time = df_processed['shot_time'].dropna().iloc[0] if not df_processed['shot_time'].dropna().empty else datetime.now()
             year = first_valid_time.year
             selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1)
             df_view = df_processed[df_processed["week"] == selected_week]
             sub_header = f"Summary for Week {selected_week}, {year}"
        else:
             st.error("Week or shot_time column missing, cannot perform Weekly analysis."); st.stop()


    elif "Monthly" in analysis_level:
        st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        if 'month' in df_processed.columns:
             available_months = sorted(df_processed["month"].dropna().unique())
             if not available_months:
                  st.warning("No valid months found in the data."); st.stop()
             selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y') if pd.notna(p) else "Invalid Month")
             df_view = df_processed[df_processed["month"] == selected_month]
             sub_header = f"Summary for {selected_month.strftime('%B %Y')}" if pd.notna(selected_month) else "Summary for Invalid Month"
        else:
             st.error("Month column missing, cannot perform Monthly analysis."); st.stop()


    elif "Custom Period" in analysis_level:
        st.header(f"Custom Period Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        if 'date' in df_processed.columns:
             min_date = df_processed['date'].min()
             max_date = df_processed['date'].max()
             # Check if min/max dates are valid
             if pd.isna(min_date) or pd.isna(max_date):
                  st.error("Cannot determine date range for Custom Period."); st.stop()

             start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
             end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)
             if start_date and end_date:
                  mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
                  df_view = df_processed[mask]
                  sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
        else:
             st.error("Date column missing, cannot perform Custom Period analysis."); st.stop()



    if not df_view.empty:
        df_view = df_view.copy()
        if 'run_id' in df_view.columns and 'shot_time' in df_view.columns:
            # Create a consistent integer-based index for runs within the current view
            # Sort by shot_time first to ensure run labels are chronological
            df_view.sort_values('shot_time', inplace=True)
            df_view['run_id_local'] = df_view.groupby('run_id', sort=False).ngroup() # Use sort=False after sorting df
            unique_run_ids_local = df_view['run_id_local'].unique() # Get unique local IDs
            # Map local IDs to labels
            run_label_map = {run_id_local: f"Run {i+1:03d}" for i, run_id_local in enumerate(unique_run_ids_local)}
            df_view['run_label'] = df_view['run_id_local'].map(run_label_map)


    if 'by Run' in analysis_level and not df_view.empty and 'run_label' in df_view.columns:
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
                    'total_runtime_sec': total_runtime_sec,
                    'production_time_sec': production_time_sec,
                    'downtime_sec': downtime_sec,
                    'total_shots': total_shots,
                    'normal_shots': normal_shots,
                    'stop_events': stop_events,
                    'mttr_min': mttr_min,
                    'mtbf_min': mtbf_min,
                    'stability_index': stability_index,
                    'efficiency': efficiency,
                }
                sub_header = sub_header.replace("Summary for", "Summary for (Combined Runs)")
            else:
                 # Provide default values if no runs were summarized
                 summary_metrics = {
                    'total_runtime_sec': 0, 'production_time_sec': 0, 'downtime_sec': 0,
                    'total_shots': 0, 'normal_shots': 0, 'stop_events': 0,
                    'mttr_min': 0, 'mtbf_min': 0, 'stability_index': 0, 'efficiency': 0,
                 }


            # Calculate results using the dashboard calculator for 'by run' specific metrics like limits
            try:
                # Pass a copy to avoid modifying df_view further
                calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
                results = calc.results
                # Add the min/max limit info to summary_metrics
                summary_metrics.update({
                    'min_lower_limit': results.get('min_lower_limit', 0), 'max_lower_limit': results.get('max_lower_limit', 0),
                    'min_mode_ct': results.get('min_mode_ct', 0), 'max_mode_ct': results.get('max_mode_ct', 0),
                    'min_upper_limit': results.get('min_upper_limit', 0), 'max_upper_limit': results.get('max_upper_limit', 0),
                })
            except Exception as e:
                st.error(f"Error during 'by Run' mode calculation: {e}")
                results = {} # Ensure results is defined even on error


        else: # Aggregate mode
            try:
                # Pass a copy to avoid modifying df_view further
                calc = RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
                results = calc.results
                summary_metrics = results # In aggregate mode, summary IS the results
            except Exception as e:
                st.error(f"Error during aggregate mode calculation: {e}")
                results = {}
                summary_metrics = {} # Ensure summary_metrics is defined

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(sub_header)
        with col2:
            # Prepare data specifically for the export button, ensuring it's a fresh copy
            df_export_ready = df_view.copy()
            # The wrapper function now handles adding 'tool_id' if missing
            excel_bytes_data = generate_run_based_excel_export(
                 df_export_ready,
                 tolerance,
                 run_interval_hours,
                 tool_id_selection # Pass the selected tool ID
             )

            st.download_button(
                 label="📥 Export Run-Based Report",
                 data=excel_bytes_data,
                 file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 use_container_width=True,
                 # Disable button if export failed (data is empty BytesIO)
                 disabled=len(excel_bytes_data) == 0
             )

        trend_summary_df = None
        # Safely calculate trend summaries, handling potential errors
        try:
            if analysis_level == "Weekly":
                trend_summary_df = calculate_daily_summaries_for_week(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
            elif analysis_level == "Monthly":
                trend_summary_df = calculate_weekly_summaries_for_month(df_view.copy(), tolerance, downtime_gap_tolerance, mode)
            elif "by Run" in analysis_level:
                trend_summary_df = calculate_run_summaries(df_view.copy(), tolerance, downtime_gap_tolerance)
                if not trend_summary_df.empty:
                    trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
            elif "Daily" in analysis_level:
                trend_summary_df = results.get('hourly_summary', pd.DataFrame())
        except Exception as e:
            st.error(f"Error calculating trend summary: {e}")
            trend_summary_df = pd.DataFrame() # Ensure it's defined as empty on error

        # --- Display Metrics ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            # Use .get() with defaults for safety, ensure division by zero check
            total_d = summary_metrics.get('total_runtime_sec', 0)
            prod_t = summary_metrics.get('production_time_sec', 0)
            down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d else 0
            down_p = (down_t / total_d * 100) if total_d else 0

            with col1: st.metric("MTTR", f"{summary_metrics.get('mttr_min', 0):.1f} min")
            with col2: st.metric("MTBF", f"{summary_metrics.get('mtbf_min', 0):.1f} min")
            with col3: st.metric("Total Run Duration", format_duration(total_d))
            with col4:
                st.metric("Production Time", f"{format_duration(prod_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{format_duration(down_t)}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.plotly_chart(create_gauge(summary_metrics.get('efficiency', 0) * 100, "Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': PASTEL_COLORS['red']}, {'range': [50, 70], 'color': PASTEL_COLORS['orange']},{'range': [70, 100], 'color': PASTEL_COLORS['green']}]
            c2.plotly_chart(create_gauge(summary_metrics.get('stability_index', 0), "Stability Index (%)", steps=steps), use_container_width=True)

        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0)
            s_e = summary_metrics.get('stop_events', 0) # Use stop_events directly
            s_s = t_s - n_s # Stopped shots for percentage calculation
            n_p = (n_s / t_s * 100) if t_s > 0 else 0
            s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}")
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{s_e}") # Display stop event count
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True) # Show percentage based on stopped shots


        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            if mode == 'by_run':
                min_ll = summary_metrics.get('min_lower_limit', 0); max_ll = summary_metrics.get('max_lower_limit', 0)
                c1.metric("Lower Limit (sec)", f"{min_ll:.2f} – {max_ll:.2f}")
                with c2:
                    min_mc = summary_metrics.get('min_mode_ct', 0); max_mc = summary_metrics.get('max_mode_ct', 0)
                    with st.container(border=True): st.metric("Mode CT (sec)", f"{min_mc:.2f} – {max_mc:.2f}")
                min_ul = summary_metrics.get('min_upper_limit', 0); max_ul = summary_metrics.get('max_upper_limit', 0)
                c3.metric("Upper Limit (sec)", f"{min_ul:.2f} – {max_ul:.2f}")
            else:
                mode_val = summary_metrics.get('mode_ct', 0)
                # Ensure mode_val is treated correctly if it's not numeric
                mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int, float, np.number)) and pd.notna(mode_val) else str(mode_val)
                c1.metric("Lower Limit (sec)", f"{summary_metrics.get('lower_limit', 0):.2f}")
                with c2:
                    with st.container(border=True): st.metric("Mode CT (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{summary_metrics.get('upper_limit', 0):.2f}")

        # --- Detailed Analysis and Plots ---
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame()
                if trend_summary_df is not None and not trend_summary_df.empty:
                    analysis_df = trend_summary_df.copy()
                    rename_map = {}
                    # Define mappings based on columns present
                    if 'hour' in analysis_df.columns: rename_map = {'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'RUN ID' in analysis_df.columns: rename_map = {'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}
                    
                    # Only rename if mapping is found
                    if rename_map:
                         # Ensure all columns to be renamed exist before renaming
                         cols_to_rename = {k: v for k, v in rename_map.items() if k in analysis_df.columns}
                         analysis_df.rename(columns=cols_to_rename, inplace=True)
                         
                # Check if essential columns for analysis exist after renaming
                essential_analysis_cols = ['period', 'stability', 'stops', 'mttr']
                if all(col in analysis_df.columns for col in essential_analysis_cols):
                    insights = generate_detailed_analysis(analysis_df, summary_metrics.get('stability_index', 0), summary_metrics.get('mttr_min', 0), summary_metrics.get('mtbf_min', 0), analysis_level)
                    if "error" in insights: st.error(insights["error"])
                    else:
                        st.components.v1.html(f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights['overall']}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights['predictive']}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights['best_worst']}</p> {'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> ' + insights['patterns'] + '</p>' if insights['patterns'] else ''}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights['recommendation']}</p></div>""", height=400, scrolling=True)
                else:
                    st.warning("Cannot generate detailed analysis summary due to missing renamed columns.")


        if analysis_level in ["Weekly", "Monthly", "Custom Period"]:
            with st.expander("View Daily/Weekly Breakdown Table", expanded=False):
                if trend_summary_df is not None and not trend_summary_df.empty:
                    d_df = trend_summary_df.copy()
                    if 'date' in d_df.columns:
                        d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                        d_df.rename(columns={'date': 'Day', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                    elif 'week' in d_df.columns:
                        d_df.rename(columns={'week': 'Week', 'stability_index': 'Stability (%)', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stops': 'Stops'}, inplace=True)
                    # Select and format columns safely
                    display_cols = [col for col in ['Day', 'Week', 'Stability (%)', 'MTTR (min)', 'MTBF (min)', 'Stops'] if col in d_df.columns]
                    st.dataframe(d_df[display_cols].style.format({'Stability (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
                else:
                     st.info("No breakdown data available.")

        elif "by Run" in analysis_level:
            run_summary_df = calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            with st.expander("View Run Breakdown Table", expanded=False):
                if run_summary_df is not None and not run_summary_df.empty:
                    d_df = run_summary_df.copy()
                    # Apply calculations safely, checking for column existence and division by zero
                    d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}" if pd.notna(row['start_time']) and pd.notna(row['end_time']) else "N/A", axis=1)
                    d_df["Total shots"] = d_df['total_shots'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")
                    d_df["Normal shots (& %)"] = d_df.apply(lambda r: f"{int(r['normal_shots']):,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if pd.notna(r['total_shots']) and r['total_shots']>0 and pd.notna(r['normal_shots']) else "0 (0.0%)", axis=1)
                    d_df["STOPS (&%)"] = d_df.apply(lambda r: f"{int(r['stops'])} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if pd.notna(r['total_shots']) and r['total_shots']>0 and pd.notna(r['stops']) and pd.notna(r['stopped_shots']) else "0 (0.0%)", axis=1)
                    d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(format_duration)
                    d_df["Production Time (d/h/m) (& %)"] = d_df.apply(lambda r: f"{format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if pd.notna(r['total_runtime_sec']) and r['total_runtime_sec']>0 and pd.notna(r['production_time_sec']) else "0m (0.0%)", axis=1)
                    d_df["Downtime (& %)"] = d_df.apply(lambda r: f"{format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if pd.notna(r['total_runtime_sec']) and r['total_runtime_sec']>0 and pd.notna(r['downtime_sec']) else "0m (0.0%)", axis=1)

                    d_df.rename(columns={'run_label':'RUN ID','mode_ct':'Mode CT (for the run)','lower_limit':'Lower limit CT (sec)','upper_limit':'Upper Limit CT (sec)','mttr_min':'MTTR (min)','mtbf_min':'MTBF (min)','stability_index':'STABILITY %','stops':'STOPS_Count'}, inplace=True) # Renamed stops to avoid clash

                    final_cols = ['RUN ID','Period (date/time from to)','Total shots','Normal shots (& %)','STOPS (&%)','Mode CT (for the run)','Lower limit CT (sec)','Upper Limit CT (sec)','Total Run duration (d/h/m)','Production Time (d/h/m) (& %)','Downtime (& %)','MTTR (min)','MTBF (min)','STABILITY %']
                    # Select only columns that actually exist after renaming/calculations
                    display_cols_run = [col for col in final_cols if col in d_df.columns]
                    st.dataframe(d_df[display_cols_run].style.format({'Mode CT (for the run)':'{:.2f}','Lower limit CT (sec)':'{:.2f}','Upper Limit CT (sec)':'{:.2f}','MTTR (min)':'{:.1f}','MTBF (min)':'{:.1f}','STABILITY %':'{:.1f}'}), use_container_width=True)
                else:
                    st.info("No run summary data available.")


        # --- Main Plots ---
        # Ensure 'results' is not empty and contains necessary keys before plotting
        if results and 'processed_df' in results:
            plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg='hourly') # time_agg seems less relevant now

            with st.expander("View Shot Data Table", expanded=False):
                 # Define columns to display, check if they exist
                 shot_data_cols = ['shot_time', 'run_label', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event']
                 display_shot_cols = [col for col in shot_data_cols if col in results['processed_df'].columns]
                 st.dataframe(results['processed_df'][display_shot_cols])
        else:
             st.warning("Cannot display shot bar chart or data table as results are missing.")


        st.markdown("---")
        # --- Trend Plots ---
        if "Daily" in analysis_level:
            st.header("Hourly Analysis")
            # Safely get dataframes from results
            run_durations_day = results.get("run_durations", pd.DataFrame())
            processed_day_df = results.get('processed_df', pd.DataFrame())
            hourly_summary = results.get('hourly_summary', pd.DataFrame()) # Use hourly summary directly

            complete_runs = pd.DataFrame()
            # Calculate complete runs only if necessary data exists
            if not processed_day_df.empty and 'stop_event' in processed_day_df.columns and 'run_group' in processed_day_df.columns and not run_durations_day.empty:
                stop_events_df = processed_day_df.loc[processed_day_df['stop_event']].copy()
                if not stop_events_df.empty:
                    stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
                    end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
                    run_durations_day['run_end_time'] = run_durations_day['run_group'].map(end_time_map)
                    complete_runs = run_durations_day.dropna(subset=['run_end_time']).copy()

            c1,c2 = st.columns(2)
            with c1:
                # Plot Bucket Analysis if data available
                if not complete_runs.empty and "time_bucket" in complete_runs.columns and "bucket_labels" in results and "bucket_color_map" in results:
                    # Ensure time_bucket is categorical
                    complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)
                    b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                    if not b_counts.empty:
                         fig_b = px.bar(b_counts, title="Time Bucket Analysis (Completed Runs)", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                         st.plotly_chart(fig_b, use_container_width=True)
                         with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
                    else: st.info("No bucket counts for completed runs.")
                else: st.info("No complete runs with bucket data.")
            with c2:
                 # Plot Hourly Stability if data available
                 if trend_summary_df is not None and not trend_summary_df.empty and 'hour' in trend_summary_df.columns and 'stability_index' in trend_summary_df.columns:
                     plot_trend_chart(trend_summary_df, 'hour', 'stability_index', "Hourly Stability Trend", "Hour of Day", "Stability (%)", is_stability=True)
                     with st.expander("View Stability Data", expanded=False): st.dataframe(trend_summary_df)
                 else: st.info("Hourly stability data not available.")

            st.subheader("Hourly Bucket Trend")
            if not complete_runs.empty and 'hour' not in complete_runs.columns and 'run_end_time' in complete_runs.columns:
                 complete_runs['hour'] = complete_runs['run_end_time'].dt.hour # Calculate hour if missing

            if not complete_runs.empty and 'hour' in complete_runs.columns and 'time_bucket' in complete_runs.columns and "bucket_labels" in results:
                try:
                    # Ensure time_bucket is categorical
                    complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)
                    pivot_df = pd.crosstab(index=complete_runs['hour'], columns=complete_runs['time_bucket'])
                    pivot_df = pivot_df.reindex(pd.Index(range(24), name='hour'), fill_value=0) # Ensure all hours 0-23 exist
                    
                    # Check if pivot table is empty
                    if not pivot_df.empty:
                        fig_hourly_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, title='Hourly Distribution of Run Durations', barmode='stack', color_discrete_map=results.get("bucket_color_map",{}), labels={'hour': 'Hour of Stop', 'value': 'Number of Runs', 'variable': 'Run Duration (min)'})
                        st.plotly_chart(fig_hourly_bucket, use_container_width=True)
                        with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                        if detailed_view:
                            with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                                st.markdown(generate_bucket_analysis(complete_runs, results.get("bucket_labels", [])), unsafe_allow_html=True)
                    else: st.info("No data for hourly bucket trend pivot.")
                except Exception as e:
                     st.error(f"Error creating hourly bucket trend: {e}")
            else: st.info("Hourly bucket trend data not available.")


            st.subheader("Hourly MTTR & MTBF Trend")
            if not hourly_summary.empty and hourly_summary['stops'].sum() > 0:
                plot_mttr_mtbf_chart(
                    df=hourly_summary,
                    x_col='hour',
                    mttr_col='mttr_min',
                    mtbf_col='mtbf_min',
                    shots_col='total_shots',
                    title="Hourly MTTR, MTBF & Shot Count Trend"
                )
                with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(hourly_summary)
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                        st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                        analysis_df = hourly_summary.copy().rename(columns={'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'})
                        st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
            else: st.info("Not enough stop data for hourly MTTR/MTBF trend.")

        elif analysis_level in ["Weekly", "Monthly", "Custom Period"]:
             trend_level_map = {"Weekly": "Daily", "Monthly": "Weekly", "Custom Period": "Daily"}
             trend_level = trend_level_map.get(analysis_level, "Unknown")
             st.header(f"{trend_level} Trends for {analysis_level.split(' ')[0]}")

             summary_df = trend_summary_df # Already calculated earlier
             run_durations = results.get("run_durations", pd.DataFrame())
             processed_df = results.get('processed_df', pd.DataFrame())

             complete_runs = pd.DataFrame()
             # Calculate complete runs if data exists
             if not processed_df.empty and 'stop_event' in processed_df.columns and 'run_group' in processed_df.columns and not run_durations.empty:
                 stop_events_df = processed_df.loc[processed_df['stop_event']].copy()
                 if not stop_events_df.empty:
                     stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
                     end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
                     run_durations['run_end_time'] = run_durations['run_group'].map(end_time_map)
                     complete_runs = run_durations.dropna(subset=['run_end_time']).copy()

             c1, c2 = st.columns(2)
             with c1:
                 st.subheader("Total Bucket Analysis")
                 if not complete_runs.empty and "time_bucket" in complete_runs.columns and "bucket_labels" in results:
                     # Ensure time_bucket is categorical
                     complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)
                     b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                     if not b_counts.empty:
                          fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration(min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results.get("bucket_color_map",{})).update_layout(legend_title_text='Duration')
                          st.plotly_chart(fig_b, use_container_width=True)
                          with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
                     else: st.info("No bucket counts for completed runs.")
                 else: st.info("No complete runs with bucket data for this period.")

             with c2:
                 st.subheader(f"{trend_level} Stability Trend")
                 if summary_df is not None and not summary_df.empty:
                     x_col = 'date' if trend_level == "Daily" else 'week'
                     if x_col in summary_df.columns and 'stability_index' in summary_df.columns:
                         plot_trend_chart(summary_df, x_col, 'stability_index', f"{trend_level} Stability Trend", trend_level, "Stability (%)", is_stability=True)
                         with st.expander("View Stability Data", expanded=False): st.dataframe(summary_df)
                     else: st.info(f"Missing '{x_col}' or 'stability_index' column for {trend_level} trend.")
                 else: st.info(f"No {trend_level.lower()} summary data available.")

             st.subheader(f"{trend_level} Bucket Trend")
             # Plot bucket trend if relevant dataframes and columns exist
             if not complete_runs.empty and summary_df is not None and not summary_df.empty and 'run_end_time' in complete_runs.columns and 'time_bucket' in complete_runs.columns and "bucket_labels" in results:
                 try:
                     time_col = 'date' if trend_level == "Daily" else 'week'
                     # Calculate date/week column if not present
                     if time_col not in complete_runs.columns:
                          if trend_level == "Daily":
                               complete_runs[time_col] = complete_runs['run_end_time'].dt.date
                          elif trend_level == "Weekly":
                               complete_runs[time_col] = complete_runs['run_end_time'].dt.isocalendar().week

                     # Ensure time_bucket is categorical
                     complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)
                     
                     pivot_df = pd.crosstab(index=complete_runs[time_col], columns=complete_runs['time_bucket'])
                     all_units = summary_df[time_col].unique() # Get unique units from summary_df
                     pivot_df = pivot_df.reindex(all_units, fill_value=0).sort_index() # Reindex and sort

                     if not pivot_df.empty and 'total_shots' in summary_df.columns:
                          fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                          for col in pivot_df.columns:
                               fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results.get("bucket_color_map",{}).get(col)), secondary_y=False)

                          # Align summary_df for total shots plotting
                          summary_df_indexed = summary_df.set_index(time_col).reindex(pivot_df.index)

                          fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=summary_df_indexed.index, y=summary_df_indexed['total_shots'], mode='lines+markers+text', text=summary_df_indexed['total_shots'].fillna(0).astype(int), textposition='top center', line=dict(color='blue')), secondary_y=True)
                          fig_bucket_trend.update_layout(barmode='stack', title_text=f'{trend_level} Distribution of Run Durations vs. Shot Count', xaxis_title=trend_level, yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                          st.plotly_chart(fig_bucket_trend, use_container_width=True)
                          with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                          if detailed_view:
                              with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                                  st.markdown(generate_bucket_analysis(complete_runs, results.get("bucket_labels",[])), unsafe_allow_html=True)
                     else: st.info("No data for bucket trend pivot or missing total shots.")
                 except Exception as e:
                      st.error(f"Error creating {trend_level} bucket trend: {e}")
             else: st.info(f"{trend_level} bucket trend data not available.")


             st.subheader(f"{trend_level} MTTR & MTBF Trend")
             if summary_df is not None and not summary_df.empty and 'stops' in summary_df.columns and summary_df['stops'].sum(skipna=True) > 0:
                 x_col = 'date' if trend_level == "Daily" else 'week'
                 # Check if necessary columns exist before plotting
                 if all(c in summary_df.columns for c in [x_col, 'mttr_min', 'mtbf_min', 'total_shots']):
                     plot_mttr_mtbf_chart(
                         df=summary_df,
                         x_col=x_col,
                         mttr_col='mttr_min',
                         mtbf_col='mtbf_min',
                         shots_col='total_shots',
                         title=f"{trend_level} MTTR, MTBF & Shot Count Trend"
                     )
                     with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(summary_df)
                     if detailed_view:
                         with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                             st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                             analysis_df = summary_df.copy()
                             rename_map = {}
                             if 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                             elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                             # Only rename if mapping found and columns exist
                             if rename_map and all(k in analysis_df.columns for k in rename_map.keys()):
                                  analysis_df.rename(columns=rename_map, inplace=True)
                                  st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
                             else:
                                  st.warning("Cannot generate MTTR/MTBF analysis due to missing columns.")
                 else: st.info(f"Missing columns required for {trend_level} MTTR/MTBF plot.")
             else: st.info(f"Not enough stop data for {trend_level} MTTR/MTBF trend.")


        elif "by Run" in analysis_level:
            st.header(f"Run-Based Analysis")
            run_summary_df = trend_summary_df # Already calculated and renamed
            run_durations = results.get("run_durations", pd.DataFrame())
            processed_df = results.get('processed_df', pd.DataFrame())


            complete_runs = pd.DataFrame()
            if not processed_df.empty and 'stop_event' in processed_df.columns and 'run_group' in processed_df.columns and not run_durations.empty:
                stop_events_df = processed_df.loc[processed_df['stop_event']].copy()
                if not stop_events_df.empty:
                    stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
                    end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
                    run_durations['run_end_time'] = run_durations['run_group'].map(end_time_map)
                    complete_runs = run_durations.dropna(subset=['run_end_time']).copy()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Total Bucket Analysis")
                if not complete_runs.empty and "time_bucket" in complete_runs.columns and "bucket_labels" in results:
                    # Ensure time_bucket is categorical
                    complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)
                    b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                    if not b_counts.empty:
                         fig_b = px.bar(b_counts, title="Total Time Bucket Analysis (Completed Runs)", labels={"index": "Duration(min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results.get("bucket_color_map",{})).update_layout(legend_title_text='Duration')
                         st.plotly_chart(fig_b, use_container_width=True)
                         with st.expander("View Bucket Data", expanded=False): st.dataframe(complete_runs)
                    else: st.info("No bucket counts for completed runs.")
                else: st.info("No complete runs with bucket data.")
            with c2:
                st.subheader("Stability per Production Run")
                if run_summary_df is not None and not run_summary_df.empty and 'RUN ID' in run_summary_df.columns and 'STABILITY %' in run_summary_df.columns:
                    plot_trend_chart(run_summary_df, 'RUN ID', 'STABILITY %', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): st.dataframe(run_summary_df)
                else: st.info(f"No run stability data to analyze.")

            st.subheader("Bucket Trend per Production Run")
            # Ensure necessary dataframes and columns exist
            if not complete_runs.empty and run_summary_df is not None and not run_summary_df.empty and 'run_group' in complete_runs.columns and 'time_bucket' in complete_runs.columns and "bucket_labels" in results and 'processed_df' in results:
                 try:
                     # Map run_group to run_label using the processed_df from results
                     if 'run_label' in results['processed_df'].columns and 'run_group' in results['processed_df'].columns:
                          run_group_to_label_map = results['processed_df'].drop_duplicates('run_group')[['run_group', 'run_label']].set_index('run_group')['run_label']
                          complete_runs['run_label'] = complete_runs['run_group'].map(run_group_to_label_map)

                          # Proceed only if run_label mapping was successful
                          if 'run_label' in complete_runs.columns and not complete_runs['run_label'].isna().all():
                               # Ensure time_bucket is categorical
                               complete_runs['time_bucket'] = pd.Categorical(complete_runs['time_bucket'], categories=results["bucket_labels"], ordered=True)

                               pivot_df = pd.crosstab(index=complete_runs['run_label'], columns=complete_runs['time_bucket'])
                               all_runs = run_summary_df['RUN ID'].unique() # Use unique run IDs from summary
                               pivot_df = pivot_df.reindex(all_runs, fill_value=0) # Reindex based on summary run IDs

                               if not pivot_df.empty and 'Total Shots' in run_summary_df.columns:
                                    fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                                    for col in pivot_df.columns:
                                         fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results.get("bucket_color_map",{}).get(col)), secondary_y=False)

                                    # Align run_summary_df for plotting total shots
                                    run_summary_indexed = run_summary_df.set_index('RUN ID').reindex(pivot_df.index)

                                    fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=run_summary_indexed.index, y=run_summary_indexed['Total Shots'], mode='lines+markers+text', text=run_summary_indexed['Total Shots'].fillna(0).astype(int), textposition='top center', line=dict(color='blue')), secondary_y=True)
                                    fig_bucket_trend.update_layout(barmode='stack', title_text='Distribution of Run Durations per Run vs. Shot Count', xaxis_title='Run ID', yaxis_title='Number of Runs', yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                                    st.plotly_chart(fig_bucket_trend, use_container_width=True)
                                    with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                                    if detailed_view:
                                        with st.expander("🤖 View Bucket Trend Analysis", expanded=False):
                                            st.markdown(generate_bucket_analysis(complete_runs, results.get("bucket_labels",[])), unsafe_allow_html=True)
                               else: st.info("No data for run bucket trend pivot or missing total shots.")
                          else: st.info("Could not map run groups to labels for run bucket trend.")
                     else: st.info("Missing 'run_label' or 'run_group' in processed data for mapping.")
                 except Exception as e:
                      st.error(f"Error creating run bucket trend: {e}")
            else: st.info("Run bucket trend data not available.")


            st.subheader("MTTR & MTBF per Production Run")
            if run_summary_df is not None and not run_summary_df.empty and 'STOPS' in run_summary_df.columns and run_summary_df['STOPS'].sum(skipna=True) > 0:
                 # Check if necessary columns exist
                 if all(c in run_summary_df.columns for c in ['RUN ID', 'MTTR (min)', 'MTBF (min)', 'Total Shots']):
                     plot_mttr_mtbf_chart(
                         df=run_summary_df,
                         x_col='RUN ID',
                         mttr_col='MTTR (min)',
                         mtbf_col='MTBF (min)',
                         shots_col='Total Shots',
                         title="MTTR, MTBF & Shot Count per Run"
                     )
                     with st.expander("View MTTR/MTBF Data", expanded=False): st.dataframe(run_summary_df)
                     if detailed_view:
                         with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                             st.info("""**How this analysis works:** It determines if stability is more affected by many small stops (a **frequency** problem) or a few long stops (a **duration** problem). This helps prioritize engineering efforts.""")
                             analysis_df = run_summary_df.copy().rename(columns={'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'})
                             st.markdown(generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)
                 else: st.info("Missing columns required for run MTTR/MTBF plot.")
            else: st.info("Not enough stop data for run MTTR/MTBF trend.")


@st.cache_data(show_spinner="Analyzing tool performance for Risk Tower...")
def calculate_risk_scores(df_all_tools):
    """Analyzes data for all tools, each within its own last 4-week window."""
    id_col = "tool_id"
    initial_metrics = []

    # First pass: Calculate metrics for each tool based on its own 4-week window
    for tool_id, df_tool in df_all_tools.groupby(id_col):
        if df_tool.empty or len(df_tool) < 10:
            continue

        # Use the Dashboard's RunRateCalculator for consistency
        calc_prepare = RunRateCalculator(df_tool, 0.05, 2.0)
        df_prepared = calc_prepare.results.get("processed_df")
        if df_prepared is None or df_prepared.empty or 'shot_time' not in df_prepared.columns:
            continue

        end_date = df_prepared['shot_time'].max()
        start_date = end_date - timedelta(weeks=4)
        df_period = df_prepared[(df_prepared['shot_time'] >= start_date) & (df_prepared['shot_time'] <= end_date)]

        if df_period.empty or len(df_period) < 10:
            continue

        calc = RunRateCalculator(df_period.copy(), 0.05, 2.0)
        res = calc.results

        # Ensure 'week' column exists before grouping
        if 'shot_time' in df_period.columns:
             df_period['week'] = df_period['shot_time'].dt.isocalendar().week
             weekly_stabilities = [
                 RunRateCalculator(df_week.copy(), 0.05, 2.0).results.get('stability_index', 0)
                 for _, df_week in df_period.groupby('week') if not df_week.empty
             ]
        else:
             weekly_stabilities = []


        trend = "Stable"
        if len(weekly_stabilities) > 1:
             first_stability = weekly_stabilities[0] if pd.notna(weekly_stabilities[0]) else None
             last_stability = weekly_stabilities[-1] if pd.notna(weekly_stabilities[-1]) else None
             if first_stability is not None and last_stability is not None and last_stability < first_stability * 0.95:
                  trend = "Declining"

        # Safely get metrics, defaulting to 0 or NaN
        stability = res.get('stability_index', 0) if pd.notna(res.get('stability_index')) else 0
        mttr = res.get('mttr_min', 0) if pd.notna(res.get('mttr_min')) else 0
        mtbf = res.get('mtbf_min', 0) if pd.notna(res.get('mtbf_min')) else 0


        initial_metrics.append({
            'Tool ID': tool_id,
            'Stability': stability,
            'MTTR': mttr,
            'MTBF': mtbf,
            'Weekly Stability': ' → '.join([f'{s:.0f}%' for s in weekly_stabilities if pd.notna(s)]),
            'Trend': trend,
            'Analysis Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        })

    if not initial_metrics:
        return pd.DataFrame()

    # Second pass: Determine risk factors by comparing against the averages
    metrics_df = pd.DataFrame(initial_metrics)
    # Calculate means safely, skipping NaNs
    overall_mttr_mean = metrics_df['MTTR'].mean(skipna=True)
    overall_mtbf_mean = metrics_df['MTBF'].mean(skipna=True)

    # Handle cases where means might still be NaN (e.g., all input MTTR/MTBF were NaN)
    overall_mttr_mean = overall_mttr_mean if pd.notna(overall_mttr_mean) else 0
    overall_mtbf_mean = overall_mtbf_mean if pd.notna(overall_mtbf_mean) else 0


    final_risk_data = []
    for _, row in metrics_df.iterrows():
        # Ensure row values are valid numbers before calculations
        row_stability = row['Stability'] if pd.notna(row['Stability']) else 0
        row_mttr = row['MTTR'] if pd.notna(row['MTTR']) else 0
        row_mtbf = row['MTBF'] if pd.notna(row['MTBF']) else 0

        risk_score = row_stability
        if row['Trend'] == "Declining":
            risk_score -= 20

        primary_factor = "Low Stability"
        details = f"Overall stability is {row_stability:.1f}%."
        if row['Trend'] == "Declining":
            primary_factor = "Declining Trend"
            details = "Stability shows a consistent downward trend."
        elif row_stability < 70 and overall_mttr_mean > 0 and row_mttr > (overall_mttr_mean * 1.2): # Check mean > 0
            primary_factor = "High MTTR"
            details = f"Average stop duration (MTTR) of {row_mttr:.1f} min is a key concern."
        elif row_stability < 70 and overall_mtbf_mean > 0 and row_mtbf < (overall_mtbf_mean * 0.8): # Check mean > 0
            primary_factor = "Frequent Stops"
            details = f"Frequent stops (MTBF of {row_mtbf:.1f} min) are impacting stability."

        final_risk_data.append({
            'Tool ID': row['Tool ID'],
            'Analysis Period': row['Analysis Period'],
            'Risk Score': max(0, risk_score), # Ensure score doesn't go below 0
            'Primary Risk Factor': primary_factor,
            'Weekly Stability': row['Weekly Stability'],
            'Details': details
        })

    if not final_risk_data:
        return pd.DataFrame()

    return pd.DataFrame(final_risk_data).sort_values('Risk Score', ascending=True).reset_index(drop=True)


def render_risk_tower(df_all_tools):
    st.title("Run Rate Risk Tower")
    st.info("This tower analyzes performance over the last 4 weeks, identifying tools that require attention. Tools with the lowest scores are at the highest risk.")

    with st.expander("ℹ️ How the Risk Tower Works"):
        st.markdown("""
        The Risk Tower evaluates each tool based on its performance over its own most recent 4-week period of operation. Here’s how the metrics are calculated:

        - **Analysis Period**: Shows the exact 4-week date range used for each tool's analysis, based on its latest available data.
        - **Risk Score**: A performance indicator from 0-100.
          - It starts with the tool's overall **Stability Index (%)** for the period.
          - A **20-point penalty** is applied if the stability shows a declining trend from the first week to the last week of its analysis period.
        - **Primary Risk Factor**: Identifies the main issue affecting performance, prioritized as follows:
          1.  **Declining Trend**: If stability is worsening over time.
          2.  **High MTTR**: If the average stop duration is significantly longer than the average of all tools.
          3.  **Frequent Stops**: If the time between stops (MTBF) is significantly shorter than the average.
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
        # Default color
        color = PASTEL_COLORS['green'] # Default to green
        # Apply coloring based on score safely
        if pd.notna(score):
            if score <= 50:
                color = PASTEL_COLORS['red']
            elif score <= 70:
                color = PASTEL_COLORS['orange']
        # Apply the style to all columns in the row
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
    files_with_issues = []
    required_time_cols_option1 = {"YEAR", "MONTH", "DAY", "TIME"}
    required_time_cols_option2 = {"SHOT TIME"}
    required_id_cols = {"TOOLING ID", "EQUIPMENT CODE", "tool_id"} # Include standardized name

    for file in files:
        df = None # Initialize df to None for each file
        try:
            df = pd.read_excel(file)

            # --- Validation ---
            # 1. Check for ID column
            current_cols = set(df.columns)
            id_col_found = required_id_cols.intersection(current_cols)
            if not id_col_found:
                 files_with_issues.append(f"{file.name} (Missing ID column: TOOLING ID or EQUIPMENT CODE)")
                 continue # Skip this file

            # Standardize ID column *after* check
            id_col_name = id_col_found.pop() # Get the actual ID col name found
            if id_col_name != "tool_id":
                 df.rename(columns={id_col_name: "tool_id"}, inplace=True)

            # 2. Check for Time columns
            has_time_option1 = required_time_cols_option1.issubset(current_cols)
            has_time_option2 = required_time_cols_option2.intersection(current_cols)

            if not has_time_option1 and not has_time_option2:
                 files_with_issues.append(f"{file.name} (Missing Time columns: YEAR/MONTH/DAY/TIME or SHOT TIME)")
                 continue # Skip this file

            # --- Parsing ---
            if has_time_option1:
                # Attempt to combine and parse, handle potential errors
                try:
                    datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
                    df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
                except Exception as e:
                    files_with_issues.append(f"{file.name} (Error parsing YEAR/MONTH/DAY/TIME: {e})")
                    continue # Skip on parsing error
            elif has_time_option2:
                 time_col_name = has_time_option2.pop()
                 # Attempt to parse, handle potential errors
                 try:
                      df["shot_time"] = pd.to_datetime(df[time_col_name], errors="coerce")
                 except Exception as e:
                      files_with_issues.append(f"{file.name} (Error parsing {time_col_name}: {e})")
                      continue # Skip on parsing error

            # 3. Check if parsing resulted in NaT (Not a Time) for all rows
            if "shot_time" not in df.columns or df["shot_time"].isna().all():
                 files_with_issues.append(f"{file.name} (Could not parse any valid timestamps)")
                 continue # Skip if no valid times

            # --- Append if valid ---
            # Ensure 'tool_id' exists after potential rename
            if "tool_id" in df.columns:
                 df_list.append(df)
            else:
                 # This case should be rare due to earlier check, but as a safeguard
                 files_with_issues.append(f"{file.name} (ID column processing failed unexpectedly)")


        except Exception as e:
            # Catch errors during read_excel or other processing
            files_with_issues.append(f"{file.name} (General Load Error: {e})")

    # Display warnings for files with issues
    if files_with_issues:
         st.sidebar.warning("Some files had issues and were skipped:")
         for issue in files_with_issues:
              st.sidebar.markdown(f"- `{issue}`")


    if not df_list:
        return pd.DataFrame()

    # Concatenate valid dataframes
    try:
        df_combined = pd.concat(df_list, ignore_index=True)
        # Final check for essential columns after concat
        if "tool_id" not in df_combined.columns or "shot_time" not in df_combined.columns:
             st.error("Critical columns ('tool_id' or 'shot_time') are missing after combining files.")
             return pd.DataFrame()
        return df_combined
    except Exception as e:
        st.error(f"Error combining loaded dataframes: {e}")
        return pd.DataFrame()


df_all_tools = load_all_data(uploaded_files)

# Check if df_all_tools is empty after loading
if df_all_tools.empty:
    st.error("No valid data could be loaded from the uploaded files. Please check the file contents and formats.")
    st.stop()


# The column is now standardized to 'tool_id'
id_col = "tool_id"
# This check should ideally not be needed due to load_all_data, but kept as safety
if id_col not in df_all_tools.columns:
    st.error(f"Critical error: 'tool_id' column not found after loading data.")
    st.stop()

# Data cleaning - Drop rows where 'tool_id' or 'shot_time' is NaN AFTER loading
df_all_tools.dropna(subset=[id_col, 'shot_time'], inplace=True)
df_all_tools[id_col] = df_all_tools[id_col].astype(str)

# Re-check if empty after dropping NaNs
if df_all_tools.empty:
    st.error("No valid rows remaining after removing entries with missing ID or timestamp.")
    st.stop()


# Add a selectbox for Tool ID for the main dashboard
tool_ids = ["All Tools (Risk Tower)"] + sorted(df_all_tools[id_col].unique().tolist())
# Map "All Tools (Risk Tower)" to a specific tool for the dashboard view, e.g., the first one
dashboard_tool_id_selection = st.sidebar.selectbox("Select Tool ID for Dashboard Analysis", tool_ids)

if dashboard_tool_id_selection == "All Tools (Risk Tower)":
    # Default to showing the first tool in the list for the dashboard if 'All' is selected
    if len(tool_ids) > 1:
        first_tool = tool_ids[1]
        df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool].copy() # Use copy
        tool_id_for_dashboard_display = first_tool
    else: # Handle case where there are no tools or only 'All Tools' option
        df_for_dashboard = pd.DataFrame()
        tool_id_for_dashboard_display = "No Tool Selected"
else:
    df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection].copy() # Use copy
    tool_id_for_dashboard_display = dashboard_tool_id_selection


tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])

with tab1:
    render_risk_tower(df_all_tools) # Pass the full dataframe for risk tower

with tab2:
    if not df_for_dashboard.empty:
        # Pass the filtered dataframe for the specific tool to the dashboard
        render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
    else:
        st.info("Select a specific Tool ID from the sidebar to view its dashboard, or ensure data was loaded correctly.")

