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
                f"{df['YEAR']}-{df['MONTH']}-{df['DAY']} " + df['TIME'].astype(str),
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
        
        # --- NEW COLOR LOGIC ---
        reds = px.colors.sequential.Reds[2::2]   # Start from a lighter red
        blues = px.colors.sequential.Blues[2:]
        greens = px.colors.sequential.Greens[2:]
        bucket_color_map = {}
        red_idx, blue_idx, green_idx = 0, 0, 0
        for label in labels:
            lower_bound = int(label.split('-')[0])
            if lower_bound < 60:
                bucket_color_map[label] = reds[red_idx % len(reds)]
                red_idx += 1
            elif 60 <= lower_bound < 160:
                bucket_color_map[label] = blues[blue_idx % len(blues)]
                blue_idx += 1
            else:
                bucket_color_map[label] = greens[green_idx % len(greens)]
                green_idx += 1
        
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
    fig = px.bar(
        run_durations["time_bucket"].value_counts().reindex(bucket_labels, fill_value=0),
        title=title, labels={"index": "Continuous Run Duration (min)", "value": "Number of Occurrences"},
        text_auto=True, color=bucket_labels, color_discrete_map=color_map
    )
    # --- LEGEND ENABLED ---
    fig.update_layout(legend_title_text='Run Duration (min)')
    st.plotly_chart(fig, use_container_width=True)

def plot_mt_trend(df, time_col, mttr_col, mtbf_col, title="MTTR & MTBF Trend"):
    # ... (function content as before)
    pass

def plot_stability_trend(df, time_col, stability_col, title="Stability Index Trend"):
    # ... (function content as before)
    pass

@st.cache_data
def export_to_excel(results: dict, tolerance: float):
    # ... (function content as before)
    pass

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

if page == "📊 Daily Deep-Dive":
    st.header("Daily Analysis")
    df_processed = calculator_full.results["processed_df"]
    available_dates = df_processed["shot_time"].dt.date.unique()
    
    if len(available_dates) == 0:
        st.warning("No date data available in the uploaded file.")
    else:
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
        df_day = df_processed[df_processed["shot_time"].dt.date == selected_date]
        
        if df_day.empty:
            st.warning(f"No data for {selected_date.strftime('%d %b %Y')}.")
        else:
            calc_day = RunRateCalculator(df_day, tolerance)
            
            st.subheader(f"Dashboard for {selected_date.strftime('%d %b %Y')}")
            with st.container(border=True):
                display_main_dashboard(calc_day.results)
            
            st.markdown("---")
            st.subheader("Daily Charts")
            
            plot_time_bucket_analysis(calc_day.results["run_durations"], calc_day.results["bucket_labels"], calc_day.results["bucket_color_map"], f"Time Bucket Analysis")
            st.caption("This chart groups continuous production runs by their duration. Shorter red bars indicate frequent stops, while longer blue bars show periods of stable production.")
            with st.expander("View Data Table"):
                bucket_counts = calc_day.results["run_durations"]["time_bucket"].value_counts().reindex(calc_day.results["bucket_labels"], fill_value=0)
                st.dataframe(bucket_counts.reset_index().rename(columns={'index': 'Run Duration (min)', 'time_bucket': 'Number of Occurrences'}), use_container_width=True)

            st.markdown("---")
            
            st.subheader("Hourly Breakdown of Continuous Runs")
            results_day = calc_day.results
            run_durations_day = results_day['run_durations']
            if not run_durations_day.empty:
                processed_day_df = results_day['processed_df']
                run_start_times = processed_day_df[['run_group', 'shot_time']].drop_duplicates(subset=['run_group'], keep='first')
                run_times = run_durations_day.merge(run_start_times, on='run_group', how='left')
                run_times['hour'] = run_times['shot_time'].dt.hour
                bucket_hourly = run_times.groupby(['hour', 'time_bucket'], observed=False).size().reset_index(name='count')
                
                if not bucket_hourly.empty:
                    fig_hourly_bucket = px.bar(
                        bucket_hourly, x='hour', y='count', color='time_bucket', title=f'Hourly Distribution of Run Durations',
                        barmode='stack', category_orders={"time_bucket": results_day["bucket_labels"]},
                        color_discrete_map=results_day["bucket_color_map"],
                        labels={'hour': 'Hour of Day', 'count': 'Number of Runs', 'time_bucket': 'Run Duration (min)'}
                    )
                    st.plotly_chart(fig_hourly_bucket, use_container_width=True)
                    st.caption("This chart breaks down the continuous runs by the hour in which they started, showing when your most stable (or unstable) periods occurred.")
                    with st.expander("View Data Table"):
                        st.dataframe(bucket_hourly.rename(columns={'hour': 'Hour', 'time_bucket': 'Run Duration (min)', 'count': 'Occurrences'}), use_container_width=True)

            st.markdown("---")
            st.subheader("Hourly Trends for Selected Day")
            hourly_df = calc_day.results['hourly_summary']
            if not hourly_df.empty and hourly_df['stops'].sum() > 0:
                plot_mt_trend(hourly_df, 'hour', 'mttr_min', 'mtbf_min')
                st.caption("This chart tracks the average stop duration (MTTR - red) and the average uptime between stops (MTBF - green) for each hour. Ideally, the green line should be high and the red line low.")
                with st.expander("View Data Table"):
                    hourly_display = hourly_df[['hour', 'stops', 'mttr_min', 'mtbf_min', 'stability_index']].rename(columns={
                        'hour': 'Hour of Day', 'stops': 'Stop Events', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stability_index': 'Stability Index (%)'
                    })
                    st.dataframe(hourly_display.style.format({'MTTR (min)': '{:.2f}', 'MTBF (min)': '{:.2f}', 'Stability Index (%)': '{:.2f}%'}), use_container_width=True)
                
                plot_stability_trend(hourly_df, 'hour', 'stability_index')
                display_stability_index_explanation()
                
            st.markdown("---")
            st.subheader("🚨 Stoppage Alerts")
            df_day_processed = calc_day.results['processed_df']
            stoppage_alerts = df_day_processed[df_day_processed['stop_event']].copy()
            
            if stoppage_alerts.empty:
                st.info("✅ No new stop events were recorded on this day.")
            else:
                stop_event_indices = stoppage_alerts.index.to_series()
                shots_since_last = stop_event_indices.diff().fillna(stop_event_indices.iloc[0] + 1).astype(int) - 1
                
                stoppage_alerts['Shots Since Last Stop'] = shots_since_last.values
                stoppage_alerts["Duration (min)"] = (stoppage_alerts["ct_diff_sec"] / 60)
                
                display_table = stoppage_alerts[['shot_time', 'Duration (min)', 'Shots Since Last Stop']].rename(columns={
                    "shot_time": "Event Time",
                })
                st.dataframe(display_table.style.format({
                    'Duration (min)': '{:.1f}'
                }), use_container_width=True)


elif page == "🗓️ Weekly Trends":
    # This page logic remains the same
    pass

elif page == "📂 View Processed Data":
    # This page logic remains the same
    pass