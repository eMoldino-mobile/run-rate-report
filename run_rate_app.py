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
    # ... (rest of the class remains the same) ...
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
            # Calculate mode per run safely
            run_modes = df.groupby('run_id')['ACTUAL CT'].apply(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan # Use NaN if mode fails
            ).fillna(0) # Fill NaN modes with 0 after calculation
            df['mode_ct'] = df['run_id'].map(run_modes)

            # Ensure tolerance calculations work even if mode_ct is 0
            df['mode_ct_safe'] = df['mode_ct'].replace(0, 1) # Use 1 temporarily if mode is 0
            lower_limit = df['mode_ct_safe'] * (1 - self.tolerance)
            upper_limit = df['mode_ct_safe'] * (1 + self.tolerance)
            # If original mode was 0, set limits appropriately (e.g., 0 to tolerance)
            lower_limit = np.where(df['mode_ct'] == 0, 0, lower_limit)
            upper_limit = np.where(df['mode_ct'] == 0, self.tolerance, upper_limit) # Or some small value

            df['lower_limit'] = lower_limit
            df['upper_limit'] = upper_limit
            mode_ct_display = "Varies by Run"
            df.drop(columns=['mode_ct_safe'], inplace=True) # Remove temporary column
        else:
            df_for_mode_calc = df[df["ACTUAL CT"] < 999.9].copy()
            # Ensure 'rounded_ct' doesn't contain NaN before mode calculation
            df_for_mode_calc['rounded_ct'] = df_for_mode_calc['ACTUAL CT'].round(0)
            mode_ct_series = df_for_mode_calc['rounded_ct'].dropna().mode()
            mode_ct = mode_ct_series.iloc[0] if not mode_ct_series.empty else 0

            # Ensure mode_ct is numeric before calculating limits
            if not isinstance(mode_ct, (int, float, np.number)) or pd.isna(mode_ct):
                mode_ct = 0

            if mode_ct == 0:
                lower_limit = 0
                upper_limit = self.tolerance # Or some small default if mode is 0
            else:
                 lower_limit = mode_ct * (1 - self.tolerance)
                 upper_limit = mode_ct * (1 + self.tolerance)
            mode_ct_display = mode_ct


        # --- Two-Phase Stop Detection Logic ---
        # Ensure 'ACTUAL CT' is numeric before comparison
        actual_ct_numeric = pd.to_numeric(df["ACTUAL CT"], errors='coerce')

        # Phase 1: Check for abnormal cycle times. Handle NaNs in limits.
        ll = lower_limit if isinstance(lower_limit, (int, float, np.number)) and pd.notna(lower_limit) else -np.inf
        ul = upper_limit if isinstance(upper_limit, (int, float, np.number)) and pd.notna(upper_limit) else np.inf
        # Use .loc with boolean indexing to handle potential NaNs in actual_ct_numeric
        is_abnormal_cycle_series = (actual_ct_numeric < ll) | (actual_ct_numeric > ul)
        is_abnormal_cycle = is_abnormal_cycle_series.fillna(False) # Treat NaN cycle times as not abnormal

        # Phase 2: Check for downtime gaps using the new adjustable tolerance.
        prev_actual_ct = actual_ct_numeric.shift(1)
        # Ensure time_diff_sec is numeric
        time_diff_numeric = pd.to_numeric(df["time_diff_sec"], errors='coerce')
        # Handle potential NaNs in prev_actual_ct and time_diff_numeric during comparison
        is_downtime_gap_series = time_diff_numeric > (prev_actual_ct + self.downtime_gap_tolerance)
        is_downtime_gap = is_downtime_gap_series.fillna(False) # Treat NaN comparisons as False

        # A shot is flagged as a stop if EITHER condition is true.
        df["stop_flag"] = np.where(is_abnormal_cycle | is_downtime_gap, 1, 0)

        if not df.empty:
            df.loc[0, "stop_flag"] = 0 # The first shot can never be a stop.

        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # Ensure 'adj_ct_sec' handles NaNs from time_diff_sec or actual_ct_numeric
        df["adj_ct_sec"] = np.where(df["stop_flag"] == 1, time_diff_numeric, actual_ct_numeric)
        df["adj_ct_sec"] = df["adj_ct_sec"].fillna(0) # Fill any resulting NaNs with 0


        total_shots = len(df)
        stop_events = df["stop_event"].sum() # sum treats NaN as 0

        # Calculate sums safely using .loc which ignores NaNs implicitly in boolean indexing
        downtime_sec = df.loc[df['stop_flag'] == 1, 'adj_ct_sec'].sum()
        production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum() # Use original ACTUAL CT for production time sum


        mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60) # Default to total prod time if no stops

        total_runtime_sec = production_time_sec + downtime_sec
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else (100.0 if stop_events == 0 else 0.0)

        normal_shots = total_shots - df["stop_flag"].sum() # sum treats NaN as 0
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        df["run_group"] = df["stop_event"].cumsum()

        # Filter out large adj_ct_sec values AND NaNs before grouping for run durations
        df_for_runs = df.dropna(subset=['adj_ct_sec'])
        df_for_runs = df_for_runs[df_for_runs['adj_ct_sec'] <= 28800].copy()
        # Ensure ACTUAL CT used for duration sum is numeric and handle NaNs
        run_durations = df_for_runs[df_for_runs["stop_flag"] == 0].groupby("run_group")["ACTUAL CT"].apply(lambda x: pd.to_numeric(x, errors='coerce').sum()).div(60).reset_index(name="duration_min")


        # Safely calculate max minutes and edges
        max_minutes = 0
        if not run_durations.empty and 'duration_min' in run_durations.columns:
             duration_numeric = pd.to_numeric(run_durations["duration_min"], errors='coerce').dropna()
             if not duration_numeric.empty:
                  max_minutes = min(duration_numeric.max(), 240)


        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
        if edges and len(edges) > 1:
            last_edge_start = edges[-2]
            labels[-1] = f"{last_edge_start}+"
            edges[-1] = np.inf # Use infinity for the last bin edge

        # Apply pd.cut only if run_durations is not empty and has the column
        if not run_durations.empty and 'duration_min' in run_durations.columns:
            # Convert duration_min to numeric, coercing errors, before cutting
            duration_numeric = pd.to_numeric(run_durations["duration_min"], errors='coerce')
            # Only cut if there are valid numeric durations
            if not duration_numeric.dropna().empty:
                 run_durations["time_bucket"] = pd.cut(duration_numeric, bins=edges, labels=labels, right=False, include_lowest=True)
            else:
                 run_durations["time_bucket"] = pd.NA # Assign NA if no valid durations

        # --- Bucket Colors ---
        reds, blues, greens = px.colors.sequential.Reds[3:7], px.colors.sequential.Blues[3:8], px.colors.sequential.Greens[3:8]
        red_labels, blue_labels, green_labels = [], [], []
        # Ensure labels exist before iterating
        if labels:
             for label in labels:
                 try:
                     # Handle '+' in the last label
                     lower_bound_str = label.split('-')[0].replace('+', '')
                     lower_bound = int(lower_bound_str)
                     if lower_bound < 60: red_labels.append(label)
                     elif 60 <= lower_bound < 160: blue_labels.append(label)
                     else: green_labels.append(label)
                 except (ValueError, IndexError): continue # Skip if label format is unexpected

        bucket_color_map = {}
        for i, label in enumerate(red_labels): bucket_color_map[label] = reds[i % len(reds)]
        for i, label in enumerate(blue_labels): bucket_color_map[label] = blues[i % len(blues)]
        for i, label in enumerate(green_labels): bucket_color_map[label] = greens[i % len(greens)]

        hourly_summary = self._calculate_hourly_summary(df)

        # Final check for NaNs in key results
        final_results = {
            "processed_df": df,
            "mode_ct": mode_ct_display,
            "total_shots": total_shots,
            "efficiency": efficiency if pd.notna(efficiency) else 0,
            "stop_events": stop_events,
            "normal_shots": normal_shots,
            "mttr_min": mttr_min if pd.notna(mttr_min) else 0,
            "mtbf_min": mtbf_min if pd.notna(mtbf_min) else 0,
            "stability_index": stability_index if pd.notna(stability_index) else 0,
            "run_durations": run_durations,
            "bucket_labels": labels if labels else [], # Ensure list even if empty
            "bucket_color_map": bucket_color_map,
            "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_sec,
            "production_time_sec": production_time_sec,
            "downtime_sec": downtime_sec,
        }

        # Handle limits specifically based on mode
        if self.analysis_mode == 'by_run':
             # Extract min/max safely from the dataframe columns
             final_results["min_lower_limit"] = df['lower_limit'].min() if 'lower_limit' in df.columns and not df['lower_limit'].dropna().empty else 0
             final_results["max_lower_limit"] = df['lower_limit'].max() if 'lower_limit' in df.columns and not df['lower_limit'].dropna().empty else 0
             final_results["min_upper_limit"] = df['upper_limit'].min() if 'upper_limit' in df.columns and not df['upper_limit'].dropna().empty else 0
             final_results["max_upper_limit"] = df['upper_limit'].max() if 'upper_limit' in df.columns and not df['upper_limit'].dropna().empty else 0
             final_results["min_mode_ct"] = df['mode_ct'].min() if 'mode_ct' in df.columns and not df['mode_ct'].dropna().empty else 0
             final_results["max_mode_ct"] = df['mode_ct'].max() if 'mode_ct' in df.columns and not df['mode_ct'].dropna().empty else 0
        else:
             final_results["lower_limit"] = lower_limit if pd.notna(lower_limit) else 0
             final_results["upper_limit"] = upper_limit if pd.notna(upper_limit) else 0

        return final_results


# --- Core Calculation Class (for Excel Export) ---
# ... (ExcelReportCalculator class remains the same) ...
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
        run_durations = df.dropna(subset=['logic_ct_diff'])
        # Further filter based on stop_flag == 0 before grouping
        run_durations = run_durations[run_durations["stop_flag"] == 0].groupby("run_group")["logic_ct_diff"].sum().div(60).reset_index(name="duration_min")


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
    # ... (function remains the same) ...
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps:
        gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else:
        gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ... (plot_shot_bar_chart remains the same) ...
# ... (plot_trend_chart remains the same) ...
# ... (plot_mttr_mtbf_chart remains the same) ...

# --- REMOVED format_minutes_to_dhm and format_duration from global scope ---

# --- FIX: Moved format_period function to global scope ---
def format_period(period_value, level):
    # ... (function remains the same) ...
    """Formats a period value (hour, date, week, run_label) for display."""
    # Check for By Run FIRST
    if "(by Run)" in level:
        # The period value is already the run label string (e.g., "Run 001")
        return str(period_value)

    # Handle date/time like objects first
    if isinstance(period_value, (pd.Timestamp, datetime, pd.Period, pd.Timedelta)):
        try:
            # Check for NaT explicitly
            if pd.isna(period_value):
                return "Invalid Date"
            return pd.to_datetime(period_value).strftime('%A, %b %d')
        except Exception:
            return "Invalid Date" # Fallback for unparseable dates

    # Handle potential float conversion for week/hour if needed, ensure it's int first
    if level == "Monthly":
        # Check for NaN and convertability before int()
        if pd.notna(period_value) and isinstance(period_value, (int, float)) and not math.isnan(period_value):
             return f"Week {int(period_value)}"
        else:
             return "N/A"

    # Check level *exactly* for Daily (hourly) formatting - NOW ONLY IF NOT By Run
    if level == "Daily":
        # Ensure it's a number that can be converted to int
        if pd.notna(period_value) and isinstance(period_value, (int, float)) and not math.isnan(period_value):
             return f"{int(period_value)}:00"
        else:
             return "N/A" # Was not a valid hour number

    # Fallback for other potential cases (like Custom Period if period is just a number)
    # Also handles cases where period_value might be unexpected type
    if pd.isna(period_value):
         return "N/A"
    return str(period_value)

# ... (calculate_daily_summaries_for_week remains the same) ...
# ... (calculate_weekly_summaries_for_month remains the same) ...
# ... (calculate_run_summaries remains the same) ...
# ... (generate_detailed_analysis remains the same) ...
# ... (generate_bucket_analysis remains the same) ...
# ... (generate_mttr_mtbf_analysis remains the same) ...
# ... (generate_excel_report remains the same) ...
# ... (generate_run_based_excel_export remains the same) ...


def render_dashboard(df_tool, tool_id_selection):
    st.sidebar.title("Dashboard Controls ⚙️")

    # ... (sidebar setup remains the same) ...
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
        # ... (function remains the same) ...
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

    # ... (rest of sidebar setup remains the same) ...
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

    # ... (Date/Time Filtering remains the same) ...
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
        # ... (run filtering remains the same) ...
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

        # ... (Calculation logic for summary_metrics and results remains the same) ...
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
            # ... (Excel download button remains the same) ...
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
        # ... (Trend summary calculation remains the same) ...
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
                # Ensure results and hourly_summary exist
                if results and 'hourly_summary' in results:
                     trend_summary_df = results['hourly_summary']
                else:
                     trend_summary_df = pd.DataFrame() # Default empty
        except Exception as e:
            st.error(f"Error calculating trend summary: {e}")
            trend_summary_df = pd.DataFrame() # Ensure it's defined as empty on error


        # --- FIX: Define format functions INSIDE render_dashboard ---
        def format_minutes_to_dhm(total_minutes):
            if pd.isna(total_minutes) or total_minutes < 0: return "N/A"
            try:
                total_minutes = int(total_minutes)
            except (ValueError, TypeError):
                return "Invalid Input" # Handle non-integer convertible input
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
            # Calls format_minutes_to_dhm after converting seconds to minutes
            try:
                 # Ensure seconds is numeric before division
                 if not isinstance(seconds, (int, float, np.number)):
                      return "Invalid Input"
                 return format_minutes_to_dhm(seconds / 60)
            except Exception:
                 return "Calculation Error" # Catch potential division errors etc.

        # --- END FIX ---


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
            with col3: st.metric("Total Run Duration", format_duration(total_d)) # Now calls locally defined function
            with col4:
                st.metric("Production Time", f"{format_duration(prod_t)}") # Now calls locally defined function
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{format_duration(down_t)}") # Now calls locally defined function
                st.markdown(f'<span style="background-color: {PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)

        # ... (rest of metric display and plotting logic remains the same) ...
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
            # ... (detailed analysis expander remains the same) ...
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


        # ... (Breakdown tables remain the same) ...
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
        # ... (plotting calls remain the same) ...
        # Ensure 'results' is not empty and contains necessary keys before plotting
        if results and 'processed_df' in results:
            plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg='hourly') # time_agg seems less relevant now

            with st.expander("View Shot Data Table", expanded=False):
                 # Define columns to display, check if they exist
                 shot_data_cols = ['shot_time', 'run_label', 'ACTUAL CT', 'time_diff_sec', 'stop_flag', 'stop_event']
                 display_shot_cols = [col for col in shot_data_cols if col in results['processed_df'].columns]
                 # Ensure df exists before displaying
                 if not results['processed_df'].empty:
                      st.dataframe(results['processed_df'][display_shot_cols])
                 else:
                      st.info("Processed shot data is empty.")
        else:
             st.warning("Cannot display shot bar chart or data table as results are missing.")


        st.markdown("---")
        # --- Trend Plots ---
        # ... (rest of trend plotting logic remains the same) ...
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
             if not hourly_summary.empty and 'stops' in hourly_summary.columns and hourly_summary['stops'].sum(skipna=True) > 0:
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
    # ... (function remains the same) ...
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
    # ... (function remains the same) ...
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
    # ... (function remains the same) ...
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

