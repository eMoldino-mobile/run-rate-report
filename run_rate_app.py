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
        """Standardizes the datetime column and calculates cycle time differences."""
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
            st.error("Dataframe must contain 'SHOT TIME' or 'YEAR', 'MONTH', 'DAY', 'TIME' columns.")
            st.stop()
            
        df = df.dropna(subset=["shot_time"]).sort_values("shot_time").reset_index(drop=True)
        
        if df.empty:
            return pd.DataFrame()

        df["ct_diff_sec"] = df["shot_time"].diff().dt.total_seconds()
        
        if "ACTUAL CT" in df.columns:
            ct_from_col = df["ACTUAL CT"].shift(1)
            df["ct_diff_sec"] = np.where(ct_from_col == 999.9, df["ct_diff_sec"], ct_from_col)
        
        if pd.isna(df.loc[0, "ct_diff_sec"]):
             df.loc[0, "ct_diff_sec"] = df.loc[0, "ACTUAL CT"] if "ACTUAL CT" in df.columns else 0
                 
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates hourly MTTR, MTBF, and Stability."""
        if df.empty or 'stop_event' not in df.columns:
            return pd.DataFrame(columns=['hour', 'mttr_min', 'mtbf_min', 'stability_index'])

        df['hour'] = df['shot_time'].dt.hour
        df['downtime_min_event'] = np.where(df['stop_event'], df['ct_diff_sec'] / 60, np.nan)
        
        # Group data by hour
        hourly_groups = df.groupby('hour')

        # Calculate metrics per hour
        stops = hourly_groups['stop_event'].sum()
        total_downtime = hourly_groups['downtime_min_event'].sum()
        
        # Calculate uptime: sum of all normal cycle times in that hour
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ct_diff_sec'].sum() / 60
        
        # Combine into a summary dataframe
        hourly_summary = pd.DataFrame({
            'stops': stops,
            'total_downtime_min': total_downtime
        }).join(uptime_min.rename('uptime_min')).fillna(0).reset_index()

        hourly_summary['mttr_min'] = hourly_summary['total_downtime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        
        # If no stops in an hour, MTBF is the total uptime for that hour.
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])

        hourly_summary['stability_index'] = (hourly_summary['mtbf_min'] / 
                                             (hourly_summary['mtbf_min'] + hourly_summary['mttr_min'].fillna(0))) * 100
        
        return hourly_summary.fillna(0)


    def _calculate_all_metrics(self) -> dict:
        """Executes the full analysis pipeline."""
        df = self._prepare_data()

        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        mode_ct = df["ACTUAL CT"].mode().iloc[0] if not df["ACTUAL CT"].mode().empty else 0
        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)

        df["stop_flag"] = np.where((df["ct_diff_sec"] > upper_limit) & (df["ct_diff_sec"] <= 28800), 1, 0)
        df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds()
        production_time_sec = total_runtime_sec - downtime_sec
        
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        stability_index = (mtbf_min / (mtbf_min + mttr_min) * 100) if (mtbf_min + mttr_min) > 0 else 100
        
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0

        df["run_group"] = df["stop_event"].cumsum()
        run_durations = df[df['stop_flag'] == 0].groupby("run_group")["ct_diff_sec"].sum().div(60).reset_index(name="duration_min")
        
        max_minutes = min(run_durations["duration_min"].max() if not run_durations.empty else 0, 240)
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
        run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False)

        hourly_summary = self._calculate_hourly_summary(df)

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "efficiency": efficiency, "stop_events": stop_events,
            "downtime_min": downtime_sec / 60, "mttr_min": mttr_min, "mtbf_min": mtbf_min,
            "stability_index": stability_index, "run_durations": run_durations, "bucket_labels": labels,
            "hourly_summary": hourly_summary,
        }

# --- UI and Plotting Functions ---

@st.cache_data
def create_gauge(value, title, color):
    """Creates a cleaner, simplified Plotly gauge indicator."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def display_main_dashboard(results: dict):
    """Renders the main KPI dashboard."""
    st.subheader("📈 Key Performance Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gauge(results.get('efficiency', 0) * 100, "Efficiency (%)", "cornflowerblue"), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge(results.get('stability_index', 0), "Stability Index (%)", "lightseagreen"), use_container_width=True)
    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MTTR (min)", f"{results.get('mttr_min', 0):.2f}", help="Mean Time To Recover: The average duration of a stop.")
    col2.metric("MTBF (min)", f"{results.get('mtbf_min', 0):.2f}", help="Mean Time Between Failures: The average uptime between stops.")
    col3.metric("Total Stops", f"{results.get('stop_events', 0):,}", help="The total number of detected stop events.")
    col4.metric("Downtime (hrs)", f"{results.get('downtime_min', 0) / 60:.2f}", help="Total time the process was stopped.")
    col5.metric("Total Shots", f"{results.get('total_shots', 0):,}", help="Total cycles in the period.")
    
    st.markdown("---")
    st.subheader("⏱️ Cycle Time Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}", help="The most frequent cycle time, used as the baseline.")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}")

@st.cache_data
def plot_time_bucket_analysis(run_durations, bucket_labels, title="Time Bucket Analysis"):
    if run_durations.empty or 'time_bucket' not in run_durations.columns:
        st.info("No valid run duration data to plot.")
        return
    bucket_counts = run_durations["time_bucket"].value_counts().reindex(bucket_labels, fill_value=0)
    fig = px.bar(bucket_counts, x=bucket_counts.index, y=bucket_counts.values, title=title,
                 labels={"x": "Continuous Run Duration (min)", "y": "Number of Occurrences"}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View Data Table"):
        st.dataframe(bucket_counts.reset_index().rename(columns={'index': 'Bucket', 'time_bucket': 'Count'}))

@st.cache_data
def plot_mt_trend(df, time_col, mttr_col, mtbf_col, title="MTTR & MTBF Trend"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[mttr_col], name='MTTR (min)', mode='lines+markers', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[bf_col], name='MTBF (min)', mode='lines+markers', line=dict(color='green'), yaxis='y2'))
    fig.update_layout(title=title, yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View Data Table"):
        st.dataframe(df)

@st.cache_data
def plot_stability_trend(df, time_col, stability_col, title="Stability Index Trend"):
    fig = px.line(df, x=time_col, y=stability_col, title=title, markers=True, labels={stability_col: "Stability Index (%)"})
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View Data Table"):
        st.dataframe(df)

# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("👈 Upload an Excel file to begin.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

df_raw = load_data(uploaded_file)
id_col = "TOOLING ID" if "TOOLING ID" in df_raw.columns else "EQUIPMENT CODE"
if id_col not in df_raw.columns:
    st.error("File must contain 'TOOLING ID' or 'EQUIPMENT CODE'.")
    st.stop()
    
tool_id = st.sidebar.selectbox(f"Select {id_col}", df_raw[id_col].unique())
df_tool = df_raw.loc[df_raw[id_col] == tool_id].copy()

if df_tool.empty:
    st.warning(f"No data for: {tool_id}")
    st.stop()

st.sidebar.markdown("---")
tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01, help="Defines the ±% around Mode CT.")

@st.cache_data
def get_calculator(df, tol):
    return RunRateCalculator(df, tol)

calculator_full = get_calculator(df_tool, tolerance)

st.title(f"Run Rate Dashboard: {tool_id}")
with st.container(border=True):
    display_main_dashboard(calculator_full.results)

st.markdown("---")
page = st.radio("Select Analysis View", ["📊 Daily Deep-Dive", "🗓️ Weekly Trends", "📂 View Processed Data"], horizontal=True)

if page == "📊 Daily Deep-Dive":
    st.header("Daily Analysis")
    df_processed = calculator_full.results["processed_df"]
    available_dates = df_processed["shot_time"].dt.date.unique()
    selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %B %Y'))

    df_day = df_processed[df_processed["shot_time"].dt.date == selected_date]
    
    if df_day.empty:
        st.warning(f"No data for {selected_date.strftime('%d %B %Y')}.")
    else:
        calc_day = RunRateCalculator(df_day, tolerance)
        plot_time_bucket_analysis(calc_day.results["run_durations"], calc_day.results["bucket_labels"], title=f"Time Bucket Analysis for {selected_date.strftime('%d %B %Y')}")
        
        st.markdown("---")
        st.subheader("Hourly Trends")
        hourly_df = calc_day.results['hourly_summary']
        if not hourly_df.empty:
            plot_mt_trend(hourly_df, 'hour', 'mttr_min', 'mtbf_min', title="Hourly MTTR & MTBF Trend")
            plot_stability_trend(hourly_df, 'hour', 'stability_index', title="Hourly Stability Index Trend")
        else:
            st.info("Not enough data to generate hourly trends for this day.")


elif page == "🗓️ Weekly Trends":
    st.header("Weekly Trend Analysis")
    df_processed = calculator_full.results["processed_df"]
    df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
    
    weekly_summary_data = [RunRateCalculator(df_week, tolerance).results | {'week_start': week} for week, df_week in df_processed.groupby('week_start')]
    summary_df = pd.DataFrame(weekly_summary_data)
    
    if not summary_df.empty:
        plot_mt_trend(summary_df, 'week_start', 'mttr_min', 'mtbf_min', title='Weekly MTTR & MTBF Trend')
        plot_stability_trend(summary_df, 'week_start', 'stability_index', title='Weekly Stability Index Trend')
        
        # Weekly Time Bucket Trend Chart
        all_run_durations = calculator_full.results['run_durations']
        if not all_run_durations.empty:
            df_processed_with_groups = calculator_full.results['processed_df'][['shot_time', 'run_group']].drop_duplicates()
            run_times = all_run_durations.merge(df_processed_with_groups, on='run_group')
            run_times['week_start'] = run_times['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
            bucket_weekly = run_times.groupby(['week_start', 'time_bucket']).size().reset_index(name='count')
            
            fig_bucket = px.bar(bucket_weekly, x='week_start', y='count', color='time_bucket', title='Weekly Time Bucket Trend',
                                category_orders={"time_bucket": calculator_full.results["bucket_labels"]})
            st.plotly_chart(fig_bucket, use_container_width=True)
            with st.expander("View Weekly Bucket Data"):
                st.dataframe(bucket_weekly)

elif page == "📂 View Processed Data":
    st.header("Processed Cycle Data")
    df_display = calculator_full.results["processed_df"].copy()
    df_display["Stop"] = np.where(df_display["stop_flag"] == 1, "🔴", "🟢")
    df_display["Stop Event Start"] = np.where(df_display["stop_event"], "🛑", "")
    st.dataframe(df_display[["shot_time", "ACTUAL CT", "ct_diff_sec", "Stop", "Stop Event Start"]].rename(columns={
        "shot_time": "Shot Time", "ACTUAL CT": "Actual CT (sec)", "ct_diff_sec": "Time Since Last Shot (sec)"
    }), use_container_width=True)
    
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Processed Data (CSV)", csv, f"{tool_id}_processed_data.csv", "text/csv")