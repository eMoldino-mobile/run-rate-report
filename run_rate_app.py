import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
# v7.17: Fix All-Time Summary calc. Use global min/max time.
__version__ = "7.17 (Fix All-Time Summary Run Time calc)"
# ==================================================================

# ==================================================================
#                            HELPER FUNCTIONS
# ==================================================================

def format_seconds_to_dhm(total_seconds):
    """Converts total seconds into a 'Xd Yh Zm' string."""
    if pd.isna(total_seconds) or total_seconds < 0: return "N/A"
    total_minutes = int(total_seconds / 60)
    days = total_minutes // (60 * 24)
    remaining_minutes = total_minutes % (60 * 24)
    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

# --- v6.89: Define all_result_columns globally to fix NameError ---
ALL_RESULT_COLUMNS = [
    'Date', 'Filtered Run Time (sec)', 'Optimal Output (parts)',
    'Capacity Loss (downtime) (sec)',
    'Capacity Loss (downtime) (parts)',
    'Actual Output (parts)', 'Actual Cycle Time Total (sec)',
    'Capacity Gain (fast cycle time) (sec)', 'Capacity Loss (slow cycle time) (sec)',
    'Capacity Loss (slow cycle time) (parts)', 'Capacity Gain (fast cycle time) (parts)',
    'Total Capacity Loss (parts)', 'Total Capacity Loss (sec)',
    'Target Output (parts)', 'Gap to Target (parts)',
    'Capacity Loss (vs Target) (parts)', 'Capacity Loss (vs Target) (sec)',
    'Total Shots (all)', 'Production Shots', 'Downtime Shots'
]

# ==================================================================
#                           DATA CALCULATION
# ==================================================================

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            uploaded_file.seek(0) # Reset file pointer for reading
            df = pd.read_csv(uploaded_file, header=0)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            uploaded_file.seek(0) # Reset file pointer for reading
            df = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Error: Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Caching is REMOVED from the core calculation function.
# --- v6.64: This function is now ONLY run vs Optimal (100%) ---
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc_slider, mode_ct_tolerance, rr_downtime_gap, run_interval_hours):
    """
    Core function to process the raw DataFrame and calculate all Capacity Risk fields
    using the new hybrid RR (downtime) + CR (inefficiency) logic.
    
    This function ALWAYS calculates vs Optimal (Approved CT).
    """

    # --- 1. Standardize and Prepare Data ---
    df = _df_raw.copy()

    # --- Flexible Column Name Mapping ---
    column_variations = {
        'SHOT TIME': ['shot time', 'shot_time', 'timestamp', 'datetime'],
        'Approved CT': ['approved ct', 'approved_ct', 'approved cycle time', 'std ct', 'standard ct'],
        'Actual CT': ['actual ct', 'actual_ct', 'actual cycle time', 'cycle time', 'ct'],
        'Working Cavities': ['working cavities', 'working_cavities', 'cavities', 'cavity'],
        'Plant Area': ['plant area', 'plant_area', 'area']
    }

    rename_dict = {}
    found_cols = {}

    for standard_name, variations in column_variations.items():
        found = False
        for col in df.columns:
            col_str_lower = str(col).strip().lower()
            if col_str_lower in variations:
                rename_dict[col] = standard_name
                found_cols[standard_name] = True
                found = True
                break
        if not found:
            found_cols[standard_name] = False

    df.rename(columns=rename_dict, inplace=True)

    # --- 2. Check for Required Columns ---
    required_cols = ['SHOT TIME', 'Approved CT', 'Actual CT']
    missing_cols = [col for col in required_cols if not found_cols.get(col)]

    if missing_cols:
        st.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return None, None

    # --- 3. Handle Optional Columns and Data Types ---
    if not found_cols.get('Working Cavities'):
        st.info(f"'Working Cavities' column not found. Using default value: {default_cavities}")
        df['Working Cavities'] = default_cavities
    else:
        df['Working Cavities'] = pd.to_numeric(df['Working Cavities'], errors='coerce')
        df['Working Cavities'].fillna(1, inplace=True)

    if not found_cols.get('Plant Area'):
        if toggle_filter:
            st.warning("'Plant Area' column not found. Cannot apply Maintenance/Warehouse filter.")
            toggle_filter = False
        df['Plant Area'] = 'Production'
    else:
        df['Plant Area'].fillna('Production', inplace=True)

    try:
        df['SHOT TIME'] = pd.to_datetime(df['SHOT TIME'])
        df['Actual CT'] = pd.to_numeric(df['Actual CT'], errors='coerce')
        df['Approved CT'] = pd.to_numeric(df['Approved CT'], errors='coerce')
        
        # Drop rows where essential data could not be parsed
        df.dropna(subset=['SHOT TIME', 'Actual CT', 'Approved CT'], inplace=True)
        
    except Exception as e:
        st.error(f"Error converting data types: {e}. Check for non-numeric values in CT or Cavities columns.")
        return None, None


    # --- 4. Apply Filters (The Toggle) ---

    if df.empty or len(df) < 2:
        st.error("Error: Not enough data in the file to calculate run time.")
        return None, None

    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty:
        st.error("Error: No 'Production' data found after filtering.")
        return None, None

    # --- v6.88: NEW LOGIC - Process *entire* dataframe first ---
    
    # 1. Sort all shots by time
    df_rr = df_production_only.sort_values("SHOT TIME").reset_index(drop=True)

    # --- v7.06: FIX for Run Breaks vs RR Stoppages ---
    is_hard_stop_code = df_rr["Actual CT"] >= 999.9
    
    # 2. Calculate `run_break_time_diff` (for Run Interval)
    # This finds the true gaps between *production runs*
    df_production_gaps = df_rr[~is_hard_stop_code]["SHOT TIME"].diff().dt.total_seconds()
    df_rr["run_break_time_diff"] = df_production_gaps
    df_rr["run_break_time_diff"].fillna(0.0, inplace=True)
    df_rr.loc[0, "run_break_time_diff"] = 0.0 # First shot has no gap
    
    # 3. Calculate `rr_time_diff` (for RR Downtime)
    # This finds gaps between *all shots* (for RR stoppage logic)
    df_rr["rr_time_diff"] = df_rr["SHOT TIME"].diff().dt.total_seconds()
    df_rr["rr_time_diff"].fillna(0.0, inplace=True)
    df_rr.loc[0, "rr_time_diff"] = 0.0 # First shot has no gap
    # --- End v7.06 Fix ---

    # 4. Identify global "Run Breaks"
    run_break_threshold_sec = run_interval_hours * 3600
    # Use the production-only gap calculation for this
    is_run_break = df_rr["run_break_time_diff"] > run_break_threshold_sec
    df_rr['is_run_break'] = is_run_break # Store this for later
    
    # 5. Assign a *global* run_id
    df_rr['run_id'] = is_run_break.cumsum()

    # ==================================================================
    # --- v7.11: KEY BUG FIX ---
    # Initialize *all* computed columns on df_rr first.
    # This ensures that all_shots_df (which is made from df_rr)
    # has these columns, even if logic fails or edge cases occur.
    # This fixes KeyErrors in 'by Run' mode and the Shot Chart.
    # ==================================================================
    df_rr['mode_ct'] = 0.0
    df_rr['mode_lower_limit'] = 0.0
    df_rr['mode_upper_limit'] = 0.0
    df_rr['approved_ct_for_run'] = 0.0
    df_rr['reference_ct'] = 0.0
    df_rr['stop_flag'] = 0
    df_rr['adj_ct_sec'] = 0.0
    df_rr['parts_gain'] = 0.0
    df_rr['parts_loss'] = 0.0
    df_rr['time_gain_sec'] = 0.0
    df_rr['time_loss_sec'] = 0.0
    df_rr['Shot Type'] = 'N/A'
    df_rr['Mode CT Lower'] = 0.0
    df_rr['Mode CT Upper'] = 0.0
    # ==================================================================
    
    # 6. Calculate Mode CT *per global run*
    df_for_mode = df_rr[df_rr["Actual CT"] < 999.9]
    # --- v7.13: Reverted .median() back to .mode() to match run_rate_app ---
    run_modes = df_for_mode.groupby('run_id')['Actual CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['mode_ct'] = df_rr['run_id'].map(run_modes)
    df_rr['mode_lower_limit'] = df_rr['mode_ct'] * (1 - mode_ct_tolerance)
    df_rr['mode_upper_limit'] = df_rr['mode_ct'] * (1 + mode_ct_tolerance)

    # 7. Calculate Approved CT *per global run*
    run_approved_cts = df_rr.groupby('run_id')['Approved CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['approved_ct_for_run'] = df_rr['run_id'].map(run_approved_cts)
    
    # 8. Set REFERENCE_CT (always Approved CT in this function)
    df_rr['reference_ct'] = df_rr['approved_ct_for_run']

    # 9. Run Stop Detection on *all shots*
    prev_actual_ct = df_rr["Actual CT"].shift(1).fillna(0)
    
    in_mode_band = (df_rr["Actual CT"] >= df_rr['mode_lower_limit']) & (df_rr["Actual CT"] <= df_rr['mode_upper_limit'])
    
    # --- v7.08: Remove '& ~in_mode_band' to fix downtime calc ---
    # Use the 'rr_time_diff' for this logic
    is_time_gap = (df_rr["rr_time_diff"] > (prev_actual_ct + rr_downtime_gap)) & ~is_run_break
    
    # --- v7.15: FINAL FIX - Remove '& ~is_run_break' ---
    # This correctly flags the first shot of a run as downtime
    # if it is an abnormal cycle.
    is_abnormal_cycle = ~in_mode_band & ~is_hard_stop_code
    
    # --- v7.07: Remove 'is_run_break' from stop_flag ---
    df_rr["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code, 1, 0)
    
    df_rr['adj_ct_sec'] = df_rr['Actual CT']
    # Use 'rr_time_diff' for the stoppage time
    df_rr.loc[is_time_gap, 'adj_ct_sec'] = df_rr['rr_time_diff']
    df_rr.loc[is_hard_stop_code, 'adj_ct_sec'] = 0 
    df_rr.loc[is_run_break, 'adj_ct_sec'] = 0 # Run break gaps are NOT downtime

    # 10. Separate all shots into Production and Downtime
    df_production = df_rr[df_rr['stop_flag'] == 0].copy()
    df_downtime   = df_rr[df_rr['stop_flag'] == 1].copy()

    # 11. Calculate per-shot losses/gains
    df_production['parts_gain'] = np.where(
        df_production['Actual CT'] < df_production['reference_ct'],
        ((df_production['reference_ct'] - df_production['Actual CT']) / df_production['reference_ct']) * df_production['Working Cavities'],
        0
    )
    df_production['parts_loss'] = np.where(
        df_production['Actual CT'] > df_production['reference_ct'],
        ((df_production['Actual CT'] - df_production['reference_ct']) / df_production['reference_ct']) * df_production['Working Cavities'],
        0
    )
    df_production['time_gain_sec'] = np.where(
        df_production['Actual CT'] < df_production['reference_ct'],
        (df_production['reference_ct'] - df_production['Actual CT']),
        0
    )
    df_production['time_loss_sec'] = np.where(
        df_production['Actual CT'] > df_production['reference_ct'],
        (df_production['Actual CT'] - df_production['reference_ct']),
        0
    )
    
    # Update df_rr with the values from df_production
    df_rr.update(df_production[['parts_gain', 'parts_loss', 'time_gain_sec', 'time_loss_sec']])

    # 12. Add Shot Type and date
    conditions = [
        (df_production['Actual CT'] > df_production['reference_ct']),
        (df_production['Actual CT'] < df_production['reference_ct']),
        (df_production['Actual CT'] == df_production['reference_ct'])
    ]
    choices = ['Slow', 'Fast', 'On Target']
    df_production['Shot Type'] = np.select(conditions, choices, default='N/A')
    
    # Update df_rr with the new 'Shot Type'
    df_rr['Shot Type'] = df_production['Shot Type'] 
    df_rr.loc[is_run_break, 'Shot Type'] = 'Run Break (Excluded)'
    df_rr['Shot Type'].fillna('RR Downtime (Stop)', inplace=True)
    
    df_rr['date'] = df_rr['SHOT TIME'].dt.date
    df_production['date'] = df_production['SHOT TIME'].dt.date
    df_downtime['date'] = df_downtime['SHOT TIME'].dt.date
    
    # 13. Add Mode CT band columns for the chart
    df_rr['Mode CT Lower'] = df_rr['mode_lower_limit']
    df_rr['Mode CT Upper'] = df_rr['mode_upper_limit']

    all_shots_list = [df_rr] # Store the processed df
    
    # --- End v6.88 Global Processing ---

    # --- v6.88: NEW - Group by Day *after* all logic is applied ---
    daily_results_list = []
    
    if df_rr.empty:
        st.warning("No data found to process.")
        return None, None

    for date, daily_df in df_rr.groupby('date'):

        results = {col: 0 for col in ALL_RESULT_COLUMNS} # Pre-fill all with 0
        results['Date'] = date
        
        # Get the day's subsets from the pre-processed dataframes
        daily_prod = df_production[df_production['date'] == date]
        daily_down = df_downtime[df_downtime['date'] == date]

        # --- 6. Get Wall Clock Time (Basis for Segment 4) ---
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        # --- v6.90: BUG FIX ---
        results['Filtered Run Time (sec)'] = base_run_time_sec
        # --- End v6.90 Bug Fix ---

        # --- 9. Get Config (Max Cavities & Avg Reference CT) ---
        max_cavities = daily_df['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = daily_df['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct):
            avg_reference_ct = 1
            
        avg_approved_ct = daily_df['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct):
            avg_approved_ct = 1


        # --- 10. Calculate The 4 Segments (in Parts) ---

        # SEGMENT 4: Optimal Production (Benchmark)
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities

        # SEGMENT 3: RR Downtime Loss
        results['Capacity Loss (downtime) (sec)'] = daily_down['adj_ct_sec'].sum()

        # SEGMENT 1: Actual Production
        results['Actual Output (parts)'] = daily_prod['Working Cavities'].sum()
        
        # --- v7.16: RECONCILIATION ---
        # Force Actual Prod Time to be Run Time - Downtime to match RR app
        results['Actual Cycle Time Total (sec)'] = results['Filtered Run Time (sec)'] - results['Capacity Loss (downtime) (sec)']
        # --- END v7.16 ---

        # SEGMENT 2: Inefficiency (CT Slow/Fast) Loss
        results['Capacity Gain (fast cycle time) (sec)'] = daily_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = daily_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = daily_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = daily_prod['parts_gain'].sum()
        
        # --- v6.56: RECONCILIATION LOGIC ---
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        # --- END v.6.56 RECONCILIATION ---

        # --- 11. Final Aggregations ---
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        net_cycle_loss_sec = results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # --- v6.64: This function now ALSO calculates the Target values ---
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities


        # New Shot Counts
        results['Total Shots (all)'] = len(daily_df)
        results['Production Shots'] = len(daily_prod)
        results['Downtime Shots'] = len(daily_down)

        daily_results_list.append(results)

    # --- 12. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list).replace([np.inf, -np.inf], np.nan).fillna(0)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')

    if not all_shots_list:
        return final_df, pd.DataFrame()

    all_shots_df = pd.concat(all_shots_list, ignore_index=True)
    
    # --- v7.03: 'run_id' increment moved to main app scope ---
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date
    
    return final_df, all_shots_df

# ==================================================================
# --- v6.89: NEW HELPER FUNCTION FOR 'BY RUN' AGGREGATION ---
# ==================================================================

def calculate_run_summaries(all_shots_df, target_output_perc_slider):
    """
    Takes the full, processed all_shots_df and aggregates it by run_id
    instead of by date.
    """
    
    # --- v6.97: Fix KeyError ---
    if all_shots_df.empty or 'run_id' not in all_shots_df.columns:
        return pd.DataFrame()
    # --- End Fix ---
    
    run_summary_list = []
    
    # Group by the global run_id
    for run_id, df_run in all_shots_df.groupby('run_id'):
        
        results = {col: 0 for col in ALL_RESULT_COLUMNS}
        results['run_id'] = run_id
        
        run_prod = df_run[df_run['stop_flag'] == 0]
        run_down = df_run[df_run['stop_flag'] == 1]

        # 1. Get Wall Clock Time for the run
        first_shot_time = df_run['SHOT TIME'].min()
        last_shot_time = df_run['SHOT TIME'].max()
        last_shot_ct = df_run.iloc[-1]['Actual CT'] if not df_run.empty else 0
        
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        # --- v6.90: BUG FIX ---
        results['Filtered Run Time (sec)'] = base_run_time_sec
        # --- End v6.90 Bug Fix ---
        
        # 2. Get Config (Max Cavities & Avg Reference CT)
        max_cavities = df_run['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = df_run['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct):
            avg_reference_ct = 1
            
        avg_approved_ct = df_run['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct):
            avg_approved_ct = 1
            
        # --- v7.11: Add Mode CT for the run ---
        df_run_prod_for_mode = df_run[df_run["Actual CT"] < 999.9]
        if not df_run_prod_for_mode.empty:
            # --- v7.13: Reverted .median() back to .mode() ---
            results['Mode CT'] = df_run_prod_for_mode['Actual CT'].mode().iloc[0] if not df_run_prod_for_mode['Actual CT'].mode().empty else 0.0
        else:
            results['Mode CT'] = 0.0

        # 3. Calculate Segments
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities
        results['Capacity Loss (downtime) (sec)'] = run_down['adj_ct_sec'].sum()
        results['Actual Output (parts)'] = run_prod['Working Cavities'].sum()
        
        # --- v7.16: RECONCILIATION ---
        # Force Actual Prod Time to be Run Time - Downtime to match RR app
        results['Actual Cycle Time Total (sec)'] = results['Filtered Run Time (sec)'] - results['Capacity Loss (downtime) (sec)']
        # --- END v7.16 ---


        # --- v6.95: Fix KeyError ---
        results['Capacity Gain (fast cycle time) (sec)'] = run_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = run_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = run_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = run_prod['parts_gain'].sum()
        # --- End v6.95 Fix ---

        # 4. Reconciliation
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        
        # 5. Final Aggregations
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + net_cycle_loss_parts
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']

        # 6. Target Calcs
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities

        # 7. Shot Counts
        results['Total Shots (all)'] = len(df_run)
        results['Production Shots'] = len(run_prod)
        results['Downtime Shots'] = len(run_down)
        
        # 8. Add start time for charting
        results['Start Time'] = first_shot_time

        run_summary_list.append(results)

    if not run_summary_list:
        return pd.DataFrame()
        
    run_summary_df = pd.DataFrame(run_summary_list).replace([np.inf, -np.inf], np.nan).fillna(0)
    run_summary_df = run_summary_df.set_index('run_id')
    
    return run_summary_df


# ==================================================================
#                       CACHING WRAPPER
# ==================================================================

@st.cache_data
# --- v6.64: Renamed target_perc to target_perc_slider ---
# --- v6.61: Removed approved_tol ---
def run_capacity_calculation(raw_data_df, toggle, cavities, target_perc_slider, mode_tol, rr_gap, run_interval, _cache_version=None):
    """Cached wrapper for the main calculation function."""
    return calculate_capacity_risk(
        raw_data_df,
        toggle,
        cavities,
        target_perc_slider,
        mode_tol,      
        rr_gap,        
        run_interval    
    )

# ==================================================================
#                       STREAMLIT APP LAYOUT
# ==================================================================

# --- Page Config ---
st.set_page_config(
    page_title=f"Capacity Risk Calculator (v{__version__})",
    layout="wide"
)

st.title("Capacity Risk Report")
st.markdown(f"**App Version:** `{__version__}` (RR-Downtime + CR-Inefficiency)")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Raw Data File (CSV or Excel)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("Run Rate Logic (for Downtime)")
st.sidebar.info("These settings define 'Downtime'.")

# --- v6.57: Restored Mode CT Tolerance slider ---
mode_ct_tolerance = st.sidebar.slider(
    "Mode CT Tolerance (%)", 0.01, 0.50, 0.25, 0.01,  
    help="Tolerance band (±) around the **Actual Mode CT**. Shots outside this band are flagged as 'Abnormal Cycle' (Downtime)."
)

# --- v6.61: Removed Approved CT Tolerance slider ---

rr_downtime_gap = st.sidebar.slider(
    "RR Downtime Gap (sec)", 0.0, 10.0, 2.0, 0.5, 
    help="Minimum idle time between shots to be considered a stop."
)

# --- v6.27: Add Run Interval Threshold ---
run_interval_hours = st.sidebar.slider(
    "Run Interval Threshold (hours)", 1.0, 24.0, 8.0, 0.5,
    help="Gaps between shots *longer* than this will be excluded from all calculations (e.g., weekends)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("CR Logic (for Inefficiency)")
st.sidebar.info("These settings define 'Inefficiency' during Uptime.")

data_frequency = st.sidebar.radio(
    "Select Graph Frequency",
    ['Daily', 'Weekly', 'Monthly', 'by Run'], # <-- v6.89: Added 'by Run'
    index=0,
    horizontal=True
)

toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots",
    value=False, # Default OFF
    help="If ON, all calculations will exclude shots where 'Plant Area' is 'Maintenance' or 'Warehouse'."
)

default_cavities = st.sidebar.number_input(
    "Default Working Cavities",
    min_value=1,
    value=2,
    help="This value will be used if the 'Working Cavities' column is not found in the file."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Target & View")

benchmark_view = st.sidebar.radio(
    "Select Report Benchmark",
    ['Optimal Output', 'Target Output'],
    index=0, # Default to Optimal
    horizontal=False,
    help="Select the benchmark to compare against (e.g., 'Total Capacity Loss' vs 'Optimal' or 'Target')."
)

if benchmark_view == "Target Output":
    target_output_perc = st.sidebar.slider(
        "Target Output % (of Optimal)",
        min_value=0.0, max_value=100.0,
        value=90.0, # Default 90%
        step=1.0,
        format="%.0f%%",
        help="Sets the 'Target Output (parts)' goal as a percentage of 'Optimal Output (parts)'."
    )
else:
    # --- v6.64: Set to 100.0 for the function ---
    target_output_perc = 100.0 
    
st.sidebar.caption(f"App Version: **{__version__}**")


# --- Main Page Display ---
if uploaded_file is not None:

    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")

        # --- v6.5: Removed CSS ---

        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk... (Using new hybrid logic)"):
            
            # --- v6.64: Single Calculation Logic ---
            
            # 1. Always calculate vs. Optimal (100%)
            cache_key = f"{__version__}_{uploaded_file.name}_{target_output_perc}_{mode_ct_tolerance}_{rr_downtime_gap}_{run_interval_hours}"
            results_df, all_shots_df = run_capacity_calculation(
                df_raw,
                toggle_filter,
                default_cavities,
                target_output_perc, # Pass the slider value
                mode_ct_tolerance,  
                rr_downtime_gap,        
                run_interval_hours,      
                _cache_version=cache_key
            )

            # --- v7.11: REDUNDANT FIX BLOCK ---
            # This block ensures all_shots_df has all required columns
            # *after* being loaded from cache.
            if not all_shots_df.empty:
                if 'run_id' not in all_shots_df.columns:
                    st.warning("Cache issue: 'run_id' missing. Recalculating...")
                    is_run_break = all_shots_df["run_break_time_diff"] > (run_interval_hours * 3600)
                    all_shots_df['run_id'] = is_run_break.cumsum()
                
                # --- v7.03: Increment run_id to be 1-based ---
                all_shots_df['run_id'] = all_shots_df['run_id'] + 1
                    
                # --- v7.02: Add missing columns if they don't exist ---
                if 'reference_ct' not in all_shots_df.columns:
                    st.warning("Cache issue: 'reference_ct' missing. Recalculating...")
                    run_approved_cts = all_shots_df.groupby('run_id')['Approved CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
                    all_shots_df['approved_ct_for_run'] = all_shots_df['run_id'].map(run_approved_cts)
                    all_shots_df['reference_ct'] = all_shots_df['approved_ct_for_run']

                if 'Mode CT Lower' not in all_shots_df.columns:
                    st.warning("Cache issue: 'Mode CT' columns missing. Recalculating...")
                    df_for_mode = all_shots_df[all_shots_df["Actual CT"] < 999.9]
                    # --- v7.13: Reverted .median() back to .mode() ---
                    run_modes = df_for_mode.groupby('run_id')['Actual CT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
                    all_shots_df['mode_ct'] = all_shots_df['run_id'].map(run_modes)
                    all_shots_df['mode_lower_limit'] = all_shots_df['mode_ct'] * (1 - mode_ct_tolerance)
                    all_shots_df['mode_upper_limit'] = all_shots_df['mode_ct'] * (1 + mode_ct_tolerance)
                    all_shots_df['Mode CT Lower'] = all_shots_df['mode_lower_limit']
                    all_shots_df['Mode CT Upper'] = all_shots_df['mode_upper_limit']
                    
                # --- v7.04: Add gain/loss columns ---
                if 'time_gain_sec' not in all_shots_df.columns:
                    st.warning("Cache issue: 'gain/loss' columns missing. Adding...")
                    all_shots_df['time_gain_sec'] = 0.0
                    all_shots_df['time_loss_sec'] = 0.0
                    all_shots_df['parts_gain'] = 0.0
                    all_shots_df['parts_loss'] = 0.0
                    
                # --- v7.11: Add all other missing columns ---
                if 'mode_ct' not in all_shots_df.columns:
                     all_shots_df['mode_ct'] = all_shots_df['run_id'].map(all_shots_df.groupby('run_id')['mode_ct'].first())
                if 'rr_time_diff' not in all_shots_df.columns:
                    all_shots_df["rr_time_diff"] = all_shots_df["SHOT TIME"].diff().dt.total_seconds().fillna(0.0)
                if 'adj_ct_sec' not in all_shots_df.columns:
                    all_shots_df['adj_ct_sec'] = 0.0 # This is ok to leave at 0, only used in aggregates
                if 'Shot Type' not in all_shots_df.columns:
                    all_shots_df['Shot Type'] = 'N/A'
                if 'stop_flag' not in all_shots_df.columns:
                    all_shots_df['stop_flag'] = 0


            # --- END REDUNDANT FIX BLOCK ---

            if results_df is None or results_df.empty or all_shots_df.empty:
                st.error("No valid data found in file. Cannot proceed.")
            else:
                # --- End v6.64 ---

                # --- 1. All-Time Summary Dashboard Calculations ---
                st.header("All-Time Summary")

                # 1. Calculate totals (based on primary calculation)
                total_produced = results_df['Actual Output (parts)'].sum()
                total_downtime_loss_parts = results_df['Capacity Loss (downtime) (parts)'].sum()
                total_slow_loss_parts = results_df['Capacity Loss (slow cycle time) (parts)'].sum()
                total_fast_gain_parts = results_df['Capacity Gain (fast cycle time) (parts)'].sum()
                total_net_cycle_loss_parts = total_slow_loss_parts - total_fast_gain_parts
                
                # These are always based on the 100% (Optimal) run
                total_optimal_100 = results_df['Optimal Output (parts)'].sum()
                total_target = results_df['Target Output (parts)'].sum()
                
                # Calculate corresponding time values
                total_downtime_loss_sec = results_df['Capacity Loss (downtime) (sec)'].sum()
                total_slow_loss_sec = results_df['Capacity Loss (slow cycle time) (sec)'].sum()
                total_fast_gain_sec = results_df['Capacity Gain (fast cycle time) (sec)'].sum()
                total_net_cycle_loss_sec = total_slow_loss_sec - total_fast_gain_sec

                
                # --- v7.17: FIX for All-Time Summary ---
                # Calculate Run Time from the *full* dataset, not summing daily.
                
                # 1. Get Wall Clock Time for *all data*
                global_first_shot_time = all_shots_df['SHOT TIME'].min()
                global_last_shot_time = all_shots_df['SHOT TIME'].max()
                global_last_shot_ct = all_shots_df.iloc[-1]['Actual CT'] if not all_shots_df.empty else 0
                
                global_time_span_sec = (global_last_shot_time - global_first_shot_time).total_seconds()
                run_time_sec_total = global_time_span_sec + global_last_shot_ct
                run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)
                
                # 2. Reconcile Actual Prod Time
                # Use the *summed* downtime, but the *global* run time
                total_actual_ct_sec = run_time_sec_total - total_downtime_loss_sec
                total_actual_ct_dhm = format_seconds_to_dhm(total_actual_ct_sec)
                # --- End v7.17 Fix ---
                

                run_time_label = "Overall Run Time" if not toggle_filter else "Filtered Run Time"
                actual_output_perc_val = (total_produced / total_optimal_100) if total_optimal_100 > 0 else 0

                total_calculated_net_loss_parts = total_downtime_loss_parts + total_net_cycle_loss_parts
                total_calculated_net_loss_sec = total_downtime_loss_sec + total_net_cycle_loss_sec
                
                # --- v6.56: Calculate True Loss (based on current benchmark) ---
                total_true_net_loss_parts = total_optimal_100 - total_produced
                # --- v6.80: Fix time discrepancy ---
                total_true_net_loss_sec = total_calculated_net_loss_sec

                
                # --- NEW LAYOUT (Replaces old 4-column layout) ---
                
                # --- v6.64: Title is always vs Optimal, since that's the calc ---
                benchmark_title = "Optimal Output"

                # --- Box 1: Overall Summary ---
                st.subheader(f"Overall Summary")
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.metric(run_time_label, run_time_dhm_total)
                    
                    with c2:
                        st.metric("Optimal Output (100%)", f"{total_optimal_100:,.0f}")
                        if benchmark_view == "Target Output":
                             st.caption(f"Target Output: {total_target:,.0f}")
                            
                    with c3:
                        st.metric(f"Actual Output ({actual_output_perc_val:.1%})", f"{total_produced:,.0f} parts")
                        st.caption(f"Actual Production Time: {total_actual_ct_dhm}")
                        
                    with c4:
                        # --- v6.82: Make color consistent ---
                        st.markdown(f"**Total Capacity Loss (True)**")
                        st.markdown(f"<h3><span style='color:red;'>{total_true_net_loss_parts:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                        st.caption(f"Total Time Lost: {format_seconds_to_dhm(total_true_net_loss_sec)}")

                        if benchmark_view == "Target Output":
                            gap_to_target = total_produced - total_target
                            gap_perc = (gap_to_target / total_target) if total_target > 0 else 0
                            gap_color = "green" if gap_to_target > 0 else "red"
                            st.caption(f"Gap to Target: <span style='color:{gap_color};'>{gap_to_target:+,.0f} parts ({gap_perc:+.1%})</span>", unsafe_allow_html=True)
                            
                # --- v6.84: Waterfall Chart Layout ---
                st.subheader(f"Capacity Loss Breakdown (vs {benchmark_title})")
                st.info(f"These values are calculated based on the *time-based* logic (Downtime + Slow/Fast Cycles) using **{benchmark_title}** as the benchmark.")
                
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown("<h6 style='text-align: center;'>Overall Performance Breakdown</h6>", unsafe_allow_html=True)
                    
                    # --- Waterfall Chart ---
                    # --- v6.85: Dynamic Benchmark Label ---
                    waterfall_x = [f"<b>Optimal Output (100%)</b>", "Loss (RR Downtime)"]
                    waterfall_y = [total_optimal_100, -total_downtime_loss_parts]
                    waterfall_measure = ["absolute", "relative"]
                    waterfall_text = [f"{total_optimal_100:,.0f}", f"{-total_downtime_loss_parts:,.0f}"]

                    if total_net_cycle_loss_parts >= 0:
                        # It's a net loss
                        waterfall_x.append("Net Loss (Cycle Time)")
                        waterfall_y.append(-total_net_cycle_loss_parts)
                        waterfall_measure.append("relative")
                        waterfall_text.append(f"{-total_net_cycle_loss_parts:,.0f}")
                    else:
                        # It's a net gain
                        waterfall_x.append("Net Gain (Cycle Time)")
                        waterfall_y.append(abs(total_net_cycle_loss_parts)) # Add it back
                        waterfall_measure.append("relative")
                        waterfall_text.append(f"{abs(total_net_cycle_loss_parts):+,.0f}")
                    
                    # Add the final total
                    waterfall_x.append("<b>Actual Output</b>")
                    waterfall_y.append(total_produced)
                    waterfall_measure.append("total")
                    waterfall_text.append(f"{total_produced:,.0f}")
                    

                    fig_waterfall = go.Figure(go.Waterfall(
                        name = "Breakdown",
                        orientation = "v",
                        measure = waterfall_measure,
                        x = waterfall_x,
                        y = waterfall_y,
                        text = waterfall_text,
                        textposition = "outside",
                        connector = {"line":{"color":"rgb(63, 63, 63)"}},
                        increasing = {"marker":{"color":"#2ca02c"}}, # Green for gains
                        decreasing = {"marker":{"color":"#ff6961"}},  # Red for losses
                        totals = {"marker":{"color":"#1f77b4"}} # Blue for totals (Benchmark & Actual)
                    ))
                    
                    fig_waterfall.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0), # --- v6.84: Removed title, set margin-top to 0 ---
                        height=400,
                        yaxis_title='Parts'
                    )
                    
                    # --- v6.85: Add Target Line ---
                    if benchmark_view == "Target Output":
                        fig_waterfall.add_shape(
                            type='line',
                            x0=-0.5, x1=len(waterfall_x)-0.5, # Span all columns
                            y0=total_target, y1=total_target,
                            line=dict(color='deepskyblue', dash='dash', width=2)
                        )
                        fig_waterfall.add_annotation(
                            x=0, y=total_target,
                            text=f"Target: {total_target:,.0f}",
                            showarrow=True, arrowhead=1, ax=-40, ay=-20
                        )
                        
                        # Optimal line is now the main benchmark, so it's already there.
                        # We'll just add the annotation
                        fig_waterfall.add_annotation(
                            x=len(waterfall_x)-0.5, y=total_optimal_100,
                            text=f"Optimal (100%): {total_optimal_100:,.0f}",
                            showarrow=True, arrowhead=1, ax=40, ay=-20
                        )
                    
                    st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
                    

                with c2:
                    # --- v6.79: New compact table layout with color ---
                    
                    # --- Helper function for color ---
                    def get_color_css(val):
                        if val > 0: return "color: red;"
                        if val < 0: return "color: green;"
                        return "color: black;"

                    # --- Color-code Total Net Loss ---
                    net_loss_val = total_calculated_net_loss_parts
                    net_loss_color = get_color_css(net_loss_val)
                    with st.container(border=True):
                        # --- v6.80: Rename to "Total Net Impact" ---
                        st.markdown(f"**Total Net Impact**")
                        st.markdown(f"<h3><span style='{net_loss_color}'>{net_loss_val:,.0f} parts</span></h3>", unsafe_allow_html=True)
                        st.caption(f"Net Time Lost: {format_seconds_to_dhm(total_calculated_net_loss_sec)}")
                    
                    # --- Create Data for the table ---
                    table_data = {
                        "Metric": [
                            "Loss (RR Downtime)", 
                            "Net Loss (Cycle Time)", 
                            # --- v6.80: Fix &nbsp; formatting ---
                            "\u00A0\u00A0\u00A0 └ Loss (Slow Cycles)", 
                            "\u00A0\u00A0\u00A0 └ Gain (Fast Cycles)"
                        ],
                        "Parts": [
                            total_downtime_loss_parts,
                            total_net_cycle_loss_parts,
                            total_slow_loss_parts,
                            total_fast_gain_parts
                        ],
                        "Time": [
                            format_seconds_to_dhm(total_downtime_loss_sec),
                            format_seconds_to_dhm(total_net_cycle_loss_sec),
                            format_seconds_to_dhm(total_slow_loss_sec),
                            format_seconds_to_dhm(total_fast_gain_sec)
                        ]
                    }
                    df_table = pd.DataFrame(table_data)

                    # --- Function to apply color styling to the "Parts" column ---
                    def style_parts_col(val, row_index):
                        # Get the correct color based on the metric
                        if row_index == 0: # Loss (RR Downtime)
                            color_style = get_color_css(val)
                        elif row_index == 1: # Net Loss (Cycle Time)
                            color_style = get_color_css(val)
                        elif row_index == 2: # Loss (Slow Cycles)
                            color_style = get_color_css(val)
                        elif row_index == 3: # Gain (Fast Cycles)
                            color_style = get_color_css(val * -1) # Invert gain for color
                        else:
                            color_style = "color: black;"
                        
                        return color_style

                    # --- Apply styling to the DataFrame ---
                    styled_df = df_table.style.apply(
                        lambda row: [style_parts_col(row['Parts'], row.name) if col == 'Parts' else '' for col in row.index],
                        axis=1
                    ).format(
                        {"Parts": "{:,.0f}"} # Apply comma formatting
                    ).set_properties(
                        **{'text-align': 'left'}, subset=['Metric', 'Time']
                    ).set_properties(
                        **{'text-align': 'right'}, subset=['Parts']
                    ).hide(axis='index') # Hide the 0,1,2,3 index
                    
                    # --- Display the styled table ---
                    st.dataframe(
                        styled_df,
                        use_container_width=True
                    )

                # --- End v6.79 Layout ---


                # --- Collapsible Daily Summary Table ---
                with st.expander("View Daily Summary Data"):

                    # --- v6.44: Use primary results_df ---
                    daily_summary_df = results_df.copy()

                    # Calculate all % and formatted columns needed for the table
                    daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    # --- v6.56: Actual Output % is always vs 100% Optimal ---
                    daily_summary_df['Actual Output (parts %)'] = np.where( results_df['Optimal Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / results_df['Optimal Output (parts)'], 0 )
                    
                    # --- v6.64: Perc base is always Optimal ---
                    perc_base_parts = daily_summary_df['Optimal Output (parts)']
                    perc_base_sec = daily_summary_df['Filtered Run Time (sec)']

                    
                    daily_summary_df['Total Capacity Loss (time %)'] = np.where( perc_base_sec > 0, daily_summary_df['Total Capacity Loss (sec)'] / perc_base_sec, 0 )
                    daily_summary_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, daily_summary_df['Total Capacity Loss (parts)'] / perc_base_parts, 0 )
                    
                    daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)

                    daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( daily_summary_df['Target Output (parts)'] > 0, daily_summary_df['Capacity Loss (vs Target) (parts)'] / daily_summary_df['Target Output (parts)'], 0 )
                    # --- v6.53: Bug Fix ---
                    daily_summary_df['Capacity Loss (vs Target) (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Capacity Loss (vs Target) (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = daily_summary_df['Capacity Loss (vs Target) (sec)'].apply(format_seconds_to_dhm)

                    daily_summary_df['Filtered Run Time (d/h/m)'] = daily_summary_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                    daily_summary_df['Actual Cycle Time Total (d/h/m)'] = daily_summary_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                    daily_kpi_table = pd.DataFrame(index=daily_summary_df.index)
                    daily_kpi_table[run_time_label] = daily_summary_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                    daily_kpi_table['Actual Production Time'] = daily_summary_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                    
                    daily_kpi_table['Actual Output (parts)'] = daily_summary_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (parts %)']:.1%})", axis=1)

                    # --- v6.1.1: Conditional Styling ---
                    # --- v6.3.2: Fixed IndentationError ---
                    if benchmark_view == "Optimal Output":
                        daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                        daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                        
                        st.dataframe(daily_kpi_table, use_container_width=True)

                    else: # Target Output
                        # --- v6.3.2: FIX for ValueError ---
                        # Force the column to numeric to handle any non-numeric values (like inf) before formatting
                        daily_summary_df['Gap to Target (parts)'] = pd.to_numeric(daily_summary_df['Gap to Target (parts)'], errors='coerce').fillna(0)
                        
                        # --- v6.22 FIX: Corrected format string (space removed) ---
                        daily_kpi_table['Gap to Target (parts)'] = daily_summary_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                        
                        # --- v6.64: Restored this column ---
                        daily_kpi_table['Capacity Loss (vs Target) (Time)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (d/h/m)']} ({r['Capacity Loss (vs Target) (time %)']:.1%})", axis=1)


                        st.dataframe(daily_kpi_table.style.applymap(
                            lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                            subset=['Gap to Target (parts)']
                        ), use_container_width=True)

                st.divider()

                # --- 2. WATERFALL CHART (REMOVED) ---
                # ... (Waterfall code remains commented out) ...
                # st.divider() # <-- Also commenting out this divider

                # --- 3. AGGREGATED REPORT (Chart & Table) ---
                
                # --- v6.64: Helper function for processing dataframes ---
                def process_aggregated_dataframe(daily_df, all_shots_df, target_perc):
                    if data_frequency == 'by Run':
                        # --- v6.89: Use new 'by Run' aggregation ---
                        agg_df = calculate_run_summaries(all_shots_df, target_perc)
                        chart_title_prefix = "Run-by-Run"
                    elif data_frequency == 'Weekly':
                        agg_df = daily_df.resample('W').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Weekly"
                    elif data_frequency == 'Monthly':
                        agg_df = daily_df.resample('ME').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Monthly"
                    else: # Daily
                        agg_df = daily_df.copy()
                        chart_title_prefix = "Daily"
                        
                    # --- v6.98: Fix KeyError by checking for empty agg_df ---
                    if agg_df.empty:
                        st.warning(f"No data to display for the '{data_frequency}' frequency.")
                        return pd.DataFrame(), "No Data"

                    # --- Calculate Percentage Columns AFTER aggregation ---
                    # --- v6.64: All calcs are vs Optimal ---
                    perc_base_parts = agg_df['Optimal Output (parts)']
                    chart_title = f"{chart_title_prefix} Capacity Report (vs Optimal)"
                    optimal_100_base = agg_df['Optimal Output (parts)']


                    agg_df['Actual Output (%)'] = np.where( optimal_100_base > 0, agg_df['Actual Output (parts)'] / optimal_100_base, 0)
                    agg_df['Production Shots (%)'] = np.where( agg_df['Total Shots (all)'] > 0, agg_df['Production Shots'] / agg_df['Total Shots (all)'], 0)
                    agg_df['Actual Cycle Time Total (time %)'] = np.where( agg_df['Filtered Run Time (sec)'] > 0, agg_df['Actual Cycle Time Total (sec)'] / agg_df['Filtered Run Time (sec)'], 0)
                    
                    agg_df['Capacity Loss (downtime) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (downtime) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (slow cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Gain (fast cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, agg_df['Total Capacity Loss (parts)'] / perc_base_parts, 0)

                    agg_df['Capacity Loss (vs Target) (parts %)'] = np.where( agg_df['Target Output (parts)'] > 0, agg_df['Capacity Loss (vs Target) (parts)'] / agg_df['Target Output (parts)'], 0)
                    agg_df['Total Capacity Loss (cycle time) (parts)'] = agg_df['Capacity Loss (slow cycle time) (parts)'] - agg_df['Capacity Gain (fast cycle time) (parts)']
                    
                    # --- v6.64: Allocation Logic for Gap Allocation Table ---
                    agg_df['(Ref) Net Loss (RR)'] = agg_df['Capacity Loss (downtime) (parts)']
                    agg_df['(Ref) Net Loss (Cycle)'] = agg_df['Total Capacity Loss (cycle time) (parts)']
                    
                    # --- v6.67: Use absolute values for ratio base ---
                    total_ref_loss_abs = agg_df['(Ref) Net Loss (RR)'].abs() + agg_df['(Ref) Net Loss (Cycle)'].abs()
                    
                    agg_df['loss_downtime_ratio'] = np.where(
                        total_ref_loss_abs > 0,
                        agg_df['(Ref) Net Loss (RR)'].abs() / total_ref_loss_abs,
                        0
                    )
                    agg_df['loss_cycletime_ratio'] = np.where(
                        total_ref_loss_abs > 0,
                        agg_df['(Ref) Net Loss (Cycle)'].abs() / total_ref_loss_abs,
                        0
                    )
                    
                    # --- v6.67: Allocate the 'Gap to Target' (positive or negative) ---
                    agg_df['Allocated Impact (RR Downtime)'] = agg_df['Gap to Target (parts)'] * agg_df['loss_downtime_ratio']
                    agg_df['Allocated Impact (Net Cycle)'] = agg_df['Gap to Target (parts)'] * agg_df['loss_cycletime_ratio']
                    # --- End v6.67 ---
                    
                    
                    agg_df['Filtered Run Time (d/h/m)'] = agg_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                    agg_df['Actual Cycle Time Total (d/h/m)'] = agg_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                    
                    return agg_df, chart_title
                # --- End v6.64 Helper ---
                
                # --- v6.64: Process the main dataframe for the chart ---
                display_df, chart_title = process_aggregated_dataframe(results_df, all_shots_df, target_output_perc)
                
                # --- v6.98: Check if processing failed ---
                if not display_df.empty:
                    if data_frequency == 'Weekly':
                        xaxis_title = "Week"
                    elif data_frequency == 'Monthly':
                        xaxis_title = "Month"
                    elif data_frequency == 'by Run':
                        xaxis_title = "Run ID" # <-- v6.89: New X-axis title
                    else: # Daily
                        xaxis_title = "Date"
                    
                    # --- v6.89: Handle different indexes ---
                    if data_frequency == 'by Run':
                        chart_df = display_df.reset_index().rename(columns={'run_id': 'X-Axis'})
                        chart_df['X-Axis'] = 'Run ' + chart_df['X-Axis'].astype(str) # Label as "Run 1", "Run 2"
                    else:
                        chart_df = display_df.reset_index().rename(columns={'Date': 'X-Axis'})
                    

                    # --- NEW: Unified Performance Breakdown Chart (Time Series) ---
                    st.header(f"{data_frequency} Performance Breakdown (vs {benchmark_title})")
                    fig_ts = go.Figure()

                    fig_ts.add_trace(go.Bar(
                        x=chart_df['X-Axis'],
                        y=chart_df['Actual Output (parts)'],
                        name='Actual Output',
                        marker_color='green',
                        customdata=chart_df['Actual Output (%)'],
                        hovertemplate='Actual Output: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                    ))
                    
                    chart_df['Net Cycle Time Loss (parts)'] = chart_df['Total Capacity Loss (cycle time) (parts)']
                    chart_df['Net Cycle Time Loss (positive)'] = np.maximum(0, chart_df['Net Cycle Time Loss (parts)'])

                    fig_ts.add_trace(go.Bar(
                        x=chart_df['X-Axis'],
                        y=chart_df['Net Cycle Time Loss (positive)'],
                        name='Capacity Loss (cycle time)',
                        marker_color='#ffb347', # Pastel Orange
                        customdata=np.stack((
                            chart_df['Net Cycle Time Loss (parts)'],
                            chart_df['Capacity Loss (slow cycle time) (parts)'],
                            chart_df['Capacity Gain (fast cycle time) (parts)']
                        ), axis=-1),
                        # --- v6.86: Fix Tooltip ---
                        hovertemplate=
                            '<b>Net Cycle Time Loss: %{customdata[0]:,.0f}</b><br>' +
                            'Slow Cycle Loss: %{customdata[1]:,.0f}<br>' +
                            'Fast Cycle Gain: -%{customdata[2]:,.0f}<br>' + 
                            '<extra></extra>'
                    ))
                    
                    fig_ts.add_trace(go.Bar(
                        x=chart_df['X-Axis'],
                        y=chart_df['Capacity Loss (downtime) (parts)'],
                        name='Run Rate Downtime (Stops)',
                        marker_color='#ff6961', # Pastel Red
                        customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                        hovertemplate='Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                    ))
                    
                    fig_ts.update_layout(barmode='stack')

                    if benchmark_view == "Target Output":
                        fig_ts.add_trace(go.Scatter(
                            x=chart_df['X-Axis'],
                            y=chart_df['Target Output (parts)'],
                            name=f'Target Output ({target_output_perc:.0f}%)',
                            mode='lines',
                            line=dict(color='deepskyblue', dash='dash'),
                            hovertemplate=f'<b>Target Output ({target_output_perc:.0f}%)</b>: %{{y:,.0f}}<extra></extra>'
                        ))
                        
                    fig_ts.add_trace(go.Scatter(
                        x=chart_df['X-Axis'],
                        y=chart_df['Optimal Output (parts)'],
                        name='Optimal Output (100%)',
                        mode='lines',
                        line=dict(color='darkblue', dash='dot'),
                        hovertemplate='Optimal Output (100%): %{y:,.0f}<extra></extra>'
                    ))

                    fig_ts.update_layout(
                        title=chart_title,
                        xaxis_title=xaxis_title,
                        yaxis_title='Parts (Output & Loss)',
                        legend_title='Metric',
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                    # --- Full Data Table (Open by Default) ---
                    
                    # --- v6.89: Use the already processed display_df ---
                    display_df_totals = display_df
                    
                    st.header(f"Production Totals Report ({data_frequency})")
                    # --- v6.89: Reset index for Run ID table ---
                    if data_frequency == 'by Run':
                        report_table_1_df = display_df_totals.reset_index().rename(columns={'run_id': 'Run ID'})
                        report_table_1 = pd.DataFrame(index=report_table_1_df.index)
                        report_table_1['Run ID'] = report_table_1_df['Run ID']
                        # --- v7.11: Add Mode CT to 'by Run' table ---
                        report_table_1['Mode CT'] = report_table_1_df['Mode CT'].map('{:.2f}s'.format)
                    else:
                        report_table_1 = pd.DataFrame(index=display_df_totals.index)
                        report_table_1_df = display_df_totals # Use the original df for applying data
                        

                    report_table_1['Total Shots (all)'] = report_table_1_df['Total Shots (all)'].map('{:,.0f}'.format)
                    report_table_1['Production Shots'] = report_table_1_df.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                    report_table_1['Downtime Shots'] = report_table_1_df['Downtime Shots'].map('{:,.0f}'.format)
                    report_table_1[run_time_label] = report_table_1_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                    report_table_1['Actual Production Time'] = report_table_1_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                    st.dataframe(report_table_1, use_container_width=True)

                    # --- v6.64: Conditional Tables ---
                    
                    # --- TABLE 1: vs Optimal ---
                    st.header(f"Capacity Loss & Gain Report (vs Optimal) ({data_frequency})")
                    
                    # --- v6.89: Use the already processed display_df ---
                    display_df_optimal = display_df

                    # --- v6.89: Reset index for Run ID table ---
                    if data_frequency == 'by Run':
                        report_table_optimal_df = display_df_optimal.reset_index().rename(columns={'run_id': 'Run ID'})
                        report_table_optimal = pd.DataFrame(index=report_table_optimal_df.index)
                        report_table_optimal['Run ID'] = report_table_optimal_df['Run ID']
                    else:
                        report_table_optimal = pd.DataFrame(index=display_df_optimal.index)
                        report_table_optimal_df = display_df_optimal

                    report_table_optimal['Optimal Output (parts)'] = report_table_optimal_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                    report_table_optimal['Actual Output (parts)'] = report_table_optimal_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                    report_table_optimal['Loss (RR Downtime)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                    report_table_optimal['Loss (Slow Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                    report_table_optimal['Gain (Fast Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                    report_table_optimal['Total Net Loss'] = report_table_optimal_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                    st.dataframe(report_table_optimal, use_container_width=True)
                    
                    
                    if benchmark_view == "Target Output": 
                        # --- TABLE 2: vs Target (Gap Allocation) ---
                        st.header(f"Gap Allocation Report (vs Target {target_output_perc:.0f}%) ({data_frequency})")
                        st.info("This table allocates the 'Gap to Target' based on the *ratio* of *real* Net Losses (Downtime vs. Net Cycle Time).")
                        
                        # --- v6.64: Use same processed df ---
                        display_df_target = display_df
                        
                        # --- v6.89: Reset index for Run ID table ---
                        if data_frequency == 'by Run':
                            report_table_target_df = display_df_target.reset_index().rename(columns={'run_id': 'Run ID'})
                            report_table_target = pd.DataFrame(index=report_table_target_df.index)
                            report_table_target['Run ID'] = report_table_target_df['Run ID']
                        else:
                            report_table_target = pd.DataFrame(index=display_df_target.index)
                            report_table_target_df = display_df_target

                        report_table_target['Target Output (parts)'] = report_table_target_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f}", axis=1)
                        report_table_target['Actual Output (parts)'] = report_table_target_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                        
                        report_table_target['Gap to Target (parts)'] = report_table_target_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                        report_table_target['Gap % (vs Target)'] = report_table_target_df.apply(lambda r: r['Gap to Target (parts)'] / r['Target Output (parts)'] if r['Target Output (parts)'] > 0 else 0, axis=1).apply(lambda x: "{:+.1%}".format(x) if pd.notna(x) else "N/A")
                        
                        
                        # --- v6.67: Renamed columns and logic ---
                        report_table_target['Allocated Impact (RR Downtime)'] = report_table_target_df.apply(
                            lambda r: f"{r['Allocated Impact (RR Downtime)']:,.2f} ({r['loss_downtime_ratio']:.1%})", 
                            axis=1
                        )
                        report_table_target['Allocated Impact (Net Cycle)'] = report_table_target_df.apply(
                            lambda r: f"{r['Allocated Impact (Net Cycle)']:,.2f} ({r['loss_cycletime_ratio']:.1%})", 
                            axis=1
                        )
                        
                        # --- v6.64: Add Reference Loss Columns ---
                        report_table_target['(Ref) Net Loss (RR)'] = report_table_target_df['(Ref) Net Loss (RR)'].apply(lambda x: "{:,.2f}".format(x))
                        report_table_target['(Ref) Net Loss (Cycle)'] = report_table_target_df['(Ref) Net Loss (Cycle)'].apply(lambda x: "{:,.2f}".format(x))

                        
                        st.dataframe(report_table_target.style.applymap(
                            lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                            # --- v6.67: Add new columns to style ---
                            subset=['Gap to Target (parts)', 'Gap % (vs Target)', 'Allocated Impact (RR Downtime)', 'Allocated Impact (Net Cycle)']
                        ), use_container_width=True)
                    # --- End v6.64 ---


                # --- 4. SHOT-BY-SHOT ANALYSIS ---
                st.divider()
                st.header("Shot-by-Shot Analysis (All Shots)")
                
                # --- v6.64: Benchmark is always Optimal ---
                st.info(f"This chart shows all shots. 'Production' shots are color-coded based on the **Optimal Output (Approved CT)** benchmark. 'RR Downtime (Stop)' shots are grey.")

                if all_shots_df.empty:
                    st.warning("No shots were found in the file to analyze.")
                else:
                    # --- v7.04: Remove "All Dates" option ---
                    available_dates_list = sorted(all_shots_df['date'].unique(), reverse=True)
                    
                    if not available_dates_list:
                        st.warning("No valid dates found in shot data.")
                    else:
                        selected_date = st.selectbox(
                            "Select a Date to Analyze",
                            options=available_dates_list,
                            format_func=lambda d: d.strftime('%Y-%m-%d') # Format for display
                        )

                        # --- v7.04: Filter to selected date ---
                        df_day_shots = all_shots_df[all_shots_df['date'] == selected_date]
                        chart_title = f"All Shots for {selected_date}"
                        
                        st.subheader("Chart Controls")
                        # --- v6.27: Filter out huge run breaks from the slider max calculation ---
                        non_break_df = df_day_shots[df_day_shots['Shot Type'] != 'Run Break (Excluded)']
                        max_ct_for_day = 100 # Default
                        if not non_break_df.empty:
                            max_ct_for_day = non_break_df['Actual CT'].max()

                        slider_max = int(np.ceil(max_ct_for_day / 10.0)) * 10
                        slider_max = max(slider_max, 50)
                        slider_max = min(slider_max, 1000)

                        y_axis_max = st.slider(
                            "Zoom Y-Axis (sec)",
                            min_value=10,
                            max_value=1000, # Max to see all outliers
                            value=min(slider_max, 50), # Default to a "zoomed in" view
                            step=10,
                            help="Adjust the max Y-axis to zoom in on the cluster. (Set to 1000 to see all outliers)."
                        )

                        # --- v7.02: Check for all required columns ---
                        required_shot_cols = ['reference_ct', 'Mode CT Lower', 'Mode CT Upper', 'run_id', 'mode_ct', 'rr_time_diff', 'adj_ct_sec']
                        missing_shot_cols = [col for col in required_shot_cols if col not in df_day_shots.columns]
                        
                        if missing_shot_cols:
                            st.error(f"Error: Shot data is missing required columns. {', '.join(missing_shot_cols)}")
                        elif df_day_shots.empty:
                            st.warning(f"No shots found for {selected_date}.")
                        else:
                            # --- v6.64: Use Reference CT (which is Approved CT) ---
                            reference_ct_for_day = df_day_shots['reference_ct'].iloc[0] 
                            reference_ct_label = "Approved CT"
                            
                            fig_ct = go.Figure()
                            # --- v6.27: Add new color for run breaks ---
                            color_map = {
                                'Slow': '#ff6961', 
                                'Fast': '#ffb347', 
                                'On Target': '#3498DB', 
                                'RR Downtime (Stop)': '#808080',
                                'Run Break (Excluded)': '#d3d3d3' # Light grey
                            }


                            for shot_type, color in color_map.items():
                                df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                                if not df_subset.empty:
                                    fig_ct.add_bar(
                                        x=df_subset['SHOT TIME'], y=df_subset['Actual CT'],
                                        name=shot_type, marker_color=color,
                                        # --- v6.89: Add run_id to hover text (now 1-based) ---
                                        customdata=df_subset['run_id'],
                                        hovertemplate='<b>%{x|%H:%M:%S}</b><br>Run ID: %{customdata}<br>Shot Type: %{fullData.name}<br>Actual CT: %{y:.2f}s<extra></extra>'
                                    )
                            
                            # --- v6.96: Add dynamic, per-run Mode CT bands ---
                            for run_id, df_run in df_day_shots.groupby('run_id'):
                                if not df_run.empty:
                                    mode_ct_lower_for_run = df_run['Mode CT Lower'].iloc[0]
                                    mode_ct_upper_for_run = df_run['Mode CT Upper'].iloc[0]
                                    run_start_time = df_run['SHOT TIME'].min()
                                    run_end_time = df_run['SHOT TIME'].max()
                                    
                                    fig_ct.add_hrect(
                                        x0=run_start_time, x1=run_end_time,
                                        y0=mode_ct_lower_for_run, y1=mode_ct_upper_for_run,
                                        fillcolor="grey", opacity=0.20,
                                        line_width=0,
                                        name=f"Run {run_id} Mode Band" if len(df_day_shots['run_id'].unique()) > 1 else "Mode CT Band"
                                    )
                            
                            # --- Hide duplicate legend entries for the bands ---
                            legend_names_seen = set()
                            for trace in fig_ct.data:
                                if "Mode Band" in trace.name:
                                    if trace.name in legend_names_seen:
                                        trace.showlegend = False
                                    else:
                                        legend_names_seen.add(trace.name)
                            # --- End v6.96 ---
                            
                            # --- v6.54: Use Reference CT for line ---
                            fig_ct.add_shape(
                                type='line',
                                x0=df_day_shots['SHOT TIME'].min(), x1=df_day_shots['SHOT TIME'].max(),
                                y0=reference_ct_for_day, y1=reference_ct_for_day,
                                line=dict(color='green', dash='dash'), name=f'{reference_ct_label} ({reference_ct_for_day:.2f}s)'
                            )

                            fig_ct.add_annotation(
                                x=df_day_shots['SHOT TIME'].max(), y=reference_ct_for_day,
                                text=f"{reference_ct_label}: {reference_ct_for_day:.2f}s", showarrow=True, arrowhead=1
                            )
                            # --- vs6.54 End ---
                            
                            # --- v6.91: Add vertical lines for new runs ---
                            if 'run_id' in df_day_shots.columns:
                                run_starts = df_day_shots.groupby('run_id')['SHOT TIME'].min().sort_values()
                                for start_time in run_starts.iloc[1:]: # Skip the very first run
                                    run_id_val = df_day_shots[df_day_shots['SHOT TIME'] == start_time]['run_id'].iloc[0]
                                    # --- v6.94: Fix TypeError by separating vline and annotation ---
                                    fig_ct.add_vline(
                                        x=start_time, 
                                        line_width=2, 
                                        line_dash="dash", 
                                        line_color="purple"
                                    )
                                    fig_ct.add_annotation(
                                        x=start_time,
                                        y=y_axis_max * 0.95, # Position annotation near the top
                                        text=f"Run {run_id_val} Start",
                                        showarrow=False,
                                        yshift=10,
                                        textangle=-90
                                    )

                            fig_ct.update_layout(
                                title=chart_title, # --- v6.91: Use dynamic title ---
                                xaxis_title='Time of Day',
                                yaxis_title='Actual Cycle Time (sec)',
                                hovermode="closest",
                                # --- v6.31: Fix typo y_aws_max -> y_axis_max ---
                                yaxis_range=[0, y_axis_max], # Apply the zoom
                                # --- v6.25: REMOVED barmode='overlay' ---
                            )
                            st.plotly_chart(fig_ct, use_container_width=True)

                            # --- v6.91: Handle "All Dates" in table ---
                            st.subheader(f"Data for all {len(df_day_shots)} shots ({selected_date.strftime('%Y-%m-%d')})")
                            if len(df_day_shots) > 1000:
                                st.info(f"Displaying first 1,000 shots of {len(df_day_shots)} total.")
                                df_to_display = df_day_shots.head(1000)
                            else:
                                df_to_display = df_day_shots
                                
                            # --- v7.11: Add new columns to table ---
                            st.dataframe(
                                df_to_display[[
                                    'SHOT TIME', 'Actual CT', 'Approved CT',
                                    'Working Cavities', 'run_id', 'mode_ct', 
                                    'Shot Type', 'stop_flag',
                                    'rr_time_diff', 'adj_ct_sec',
                                    'reference_ct', 'Mode CT Lower', 'Mode CT Upper'
                                ]].style.format({
                                    'Actual CT': '{:.2f}',
                                    'Approved CT': '{:.1f}',
                                    'reference_ct': '{:.2f}', 
                                    'Mode CT Lower': '{:.2f}',
                                    'Mode CT Upper': '{:.2f}',
                                    'mode_ct': '{:.2f}',
                                    'rr_time_diff': '{:.1f}s',
                                    'adj_ct_sec': '{:.1f}s',
                                    'SHOT TIME': lambda t: t.strftime('%H:%M:%S')
                                }),
                                use_container_width=True
                            )

else:
    st.info("👈 Please upload a data file to begin.")