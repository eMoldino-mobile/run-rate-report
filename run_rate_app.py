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
BASE_BUCKET_COLORS = [
    "#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#abd9e9",
    "#74add1", "#4575b4", "#313695"
]


# --- Core Calculation Class ---
class RunRateCalculator:
    """Encapsulates all logic for calculating run rate and stability metrics."""
    def __init__(self, df: pd.DataFrame, tolerance: float):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.results = self._calculate_all_metrics()

    def _prepare_data(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
            df["shot_time"] = pd.to_datetime(
                df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" +
                df["DAY"].astype(str) + " " + df["TIME"].astype(str),
                errors="coerce"
            )
        elif "SHOT TIME" in df.columns:
            df["shot_time"] = pd.to_datetime(df["SHOT TIME"], errors="coerce")
        else:
            return pd.DataFrame()
            
        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        if df.empty: return pd.DataFrame()

        df["ct_diff_sec"] = df["shot_time"].diff().dt.total_seconds()
        
        if "ACTUAL CT" in df.columns:
            ct_from_col = df["ACTUAL CT"].shift(1)
            df["ct_diff_sec"] = np.where(ct_from_col == 999.9, df["ct_diff_sec"], ct_from_col)
        
        if not df.empty and pd.isna(df.loc[0, "ct_diff_sec"]):
             df.loc[0, "ct_diff_sec"] = df.loc[0, "ACTUAL CT"] if "ACTUAL CT" in df.columns else 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'stop_event' not in df.columns: return pd.DataFrame()

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
        hourly_summary['stability_index'] = (hourly_summary['mtbf_min'] / (hourly_summary['mtbf_min'] + hourly_summary['mttr_min'].fillna(0))) * 100
        hourly_summary.loc[hourly_summary['stops'] == 0, 'stability_index'] = 100.0
        return hourly_summary

    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns: return {}

        mode_ct = df["ACTUAL CT"].mode().iloc[0] if not df["ACTUAL CT"].mode().empty else 0
        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)
        
        stop_condition = ((df["ct_diff_sec"] < lower_limit) | (df["ct_diff_sec"] > upper_limit)) & (df["ct_diff_sec"] <= 28800)
        
        df["stop_flag"] = np.where(stop_condition, 1, 0)
        df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0
        production_time_sec = total_runtime_sec - downtime_sec
        
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        
        stability_index = (mtbf_min / (mtbf_min + mttr_min) * 100) if (mtbf_min + mttr_min) > 0 else (100.0 if stop_events == 0 else 0.0)

        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0

        df["run_group"] = df["stop_event"].cumsum()
        run_durations = df[df['stop_flag'] == 0].groupby("run_group")["ct_diff_sec"].sum().div(60).reset_index(name="duration_min")
        
        max_minutes = min(run_durations["duration_min"].max() if not run_durations.empty else 0, 240)
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
        run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False)
        bucket_color_map = {label: BASE_BUCKET_COLORS[i % len(BASE_BUCKET_COLORS)] for i, label in enumerate(labels)}

        hourly_summary = self._calculate_hourly_summary(df)

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "efficiency": efficiency, "stop_events": stop_events,
            "downtime_min": downtime_sec / 60, "mttr_min": mttr_min, "mtbf_min": mtbf_min,
            "stability_index": stability_index, "run_durations": run_durations, "bucket_labels": labels,
            "hourly_summary": hourly_summary, "bucket_color_map": bucket_color_map, "normal_shots": normal_shots
        }

# --- UI Helper and Plotting Functions ---

def display_stability_index_explanation():
    # ... (function content as before)
    pass

def display_main_dashboard(results: dict):
    # ... (function content as before)
    pass

@st.cache_data
def create_gauge(value, title, color):
    # ... (function content as before)
    pass

def plot_time_bucket_analysis(run_durations, bucket_labels, color_map, title="Time Bucket Analysis"):
    # ... (function content as before)
    pass

def plot_mt_trend(df, time_col, mttr_col, mtbf_col, title="MTTR & MTBF Trend"):
    # ... (function content as before)
    pass

def plot_stability_trend(df, time_col, stability_col, title="Stability Index Trend"):
    # ... (function content as before)
    pass

# --- NEW: Enhanced Excel Export Function ---
@st.cache_data
def export_to_excel(calculator: RunRateCalculator):
    """Creates a multi-sheet Excel report with all key analyses."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # --- Sheet 1: Dashboard ---
        results = calculator.results
        summary_kpis = {
            "Metric": ["Total Shots", "Normal Shots", "Stop Events", "Efficiency (%)", 
                       "Stability Index (%)", "MTTR (min)", "MTBF (min)", "Downtime (min)",
                       "Mode CT (sec)", "Lower Limit (sec)", "Upper Limit (sec)"],
            "Value": [results['total_shots'], results['normal_shots'], results['stop_events'],
                      f"{results['efficiency']*100:.2f}", f"{results['stability_index']:.2f}",
                      f"{results['mttr_min']:.2f}", f"{results['mtbf_min']:.2f}", f"{results['downtime_min']:.2f}",
                      f"{results['mode_ct']:.2f}", f"{results['lower_limit']:.2f}", f"{results['upper_limit']:.2f}"]
        }
        df_dashboard = pd.DataFrame(summary_kpis)
        df_dashboard.to_excel(writer, sheet_name="Dashboard", index=False)

        # --- Sheet 2 & 3: Daily and Weekly Summaries ---
        df_processed = results['processed_df'].copy()
        if not df_processed.empty:
            df_processed['date'] = df_processed['shot_time'].dt.date
            daily_summary_data = [RunRateCalculator(df_day, calculator.tolerance).results | {'date': date} for date, df_day in df_processed.groupby('date')]
            df_daily = pd.DataFrame(daily_summary_data)
            
            df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
            weekly_summary_data = [RunRateCalculator(df_week, calculator.tolerance).results | {'week_start': week} for week, df_week in df_processed.groupby('week_start')]
            df_weekly = pd.DataFrame(weekly_summary_data)

            for df_summary, name in [(df_daily, "Daily"), (df_weekly, "Weekly")]:
                df_summary['bad_shots'] = df_summary['total_shots'] - df_summary['normal_shots']
                cols = [c for c in [df_summary.columns[0], 'total_shots', 'normal_shots', 'bad_shots', 'stop_events', 'mttr_min', 'mtbf_min', 'stability_index', 'efficiency'] if c in df_summary.columns]
                df_summary[cols].to_excel(writer, sheet_name=f"{name} Summary", index=False)

        # --- Sheet 4: Time Bucket Analysis ---
        bucket_counts = results["run_durations"]["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
        df_buckets = bucket_counts.reset_index().rename(columns={'index': 'Run Duration (min)', 'time_bucket': 'Occurrences'})
        df_buckets.to_excel(writer, sheet_name="Time Bucket Analysis", index=False)
        
        # --- Sheet 5: Processed Data ---
        df_export = results['processed_df'].copy()
        export_cols = ['shot_time', 'ACTUAL CT', 'ct_diff_sec', 'stop_flag', 'stop_event', 'run_group']
        df_export[export_cols].to_excel(writer, sheet_name="Processed Shot Data", index=False)

    return output.getvalue()


# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("👈 Upload an Excel file to begin.")
    st.stop()

@st.cache_data
def load_data(file): return pd.read_excel(file)

df_raw = load_data(uploaded_file)
id_col = "TOOLING ID" if "TOOLING ID" in df_raw.columns else "EQUIPMENT CODE"
if id_col not in df_raw.columns:
    st.error(f"File must contain 'TOOLING ID' or 'EQUIPMENT CODE'.")
    st.stop()
    
tool_id = st.sidebar.selectbox(f"Select {id_col}", df_raw[id_col].unique())
df_tool = df_raw.loc[df_raw[id_col] == tool_id].copy()

if df_tool.empty:
    st.warning(f"No data for: {tool_id}")
    st.stop()

st.sidebar.markdown("---")
tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")

@st.cache_data
def get_calculator(df, tol): return RunRateCalculator(df, tol)

calculator_full = get_calculator(df_tool, tolerance)

if not calculator_full.results:
    st.error(f"Could not process data for {tool_id}. Please ensure it contains valid time and 'ACTUAL CT' columns.")
    st.stop()

st.title(f"Run Rate Dashboard: {tool_id}")
st.markdown("---")
page = st.radio("Select Analysis View", ["📊 Daily Deep-Dive", "🗓️ Weekly Trends", "📂 View Processed Data"], horizontal=True, label_visibility="collapsed")

# The Daily Deep-Dive and Weekly Trends pages remain the same as the previous version
# ...

elif page == "📂 View Processed Data":
    st.header("Processed Cycle Data")
    
    results = calculator_full.results
    st.subheader("Calculation Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}", help="Cycles below this are flagged as stops.")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}", help="Cycles above this are flagged as stops.")

    st.markdown("---")
    st.subheader("Shot-by-Shot Data")
    
    df_display = results["processed_df"].copy()
    
    df_display["Stop"] = np.where(df_display["stop_flag"] == 1, "🔴", "🟢")
    df_display["Stop Event Start"] = np.where(df_display["stop_event"], "🛑", "")
    
    display_cols = ["shot_time", "ACTUAL CT", "ct_diff_sec", "Stop", "Stop Event Start", "run_group"]
    display_subset = df_display[display_cols].rename(columns={
        "shot_time": "Shot Time", "ACTUAL CT": "Actual CT (sec)", 
        "ct_diff_sec": "Time Since Last Shot (sec)", "run_group": "Run Group ID"
    })
    
    st.dataframe(display_subset.style.format({
        "Actual CT (sec)": "{:.1f}", "Time Since Last Shot (sec)": "{:.2f}"
    }), use_container_width=True)
    
    # --- New Download Button Logic ---
    excel_data = export_to_excel(calculator_full)
    st.download_button(
        label="📥 Download Full Excel Report",
        data=excel_data,
        file_name=f"{tool_id}_full_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )