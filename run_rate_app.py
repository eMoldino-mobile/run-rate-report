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
    "#d73027", "#fc8d59", "#fee090", "#c6dbef", "#9ecae1",
    "#6baed6", "#4292c6", "#2171b5", "#084594"
]

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
        
        # Calculate the time difference between consecutive shots
        df["ct_diff_sec"] = df["shot_time"].diff().dt.total_seconds()
        
        # If 'ACTUAL CT' exists, use it to handle sensor max-out values (e.g., 999.9)
        if "ACTUAL CT" in df.columns:
            # Use the previous shot's 'ACTUAL CT' for the current time difference.
            ct_from_col = df["ACTUAL CT"].shift(1)
            # A common pattern is for a maxed-out sensor value (e.g., 999.9) to indicate a long stop.
            # In these cases, we prefer the actual calculated time difference.
            df["ct_diff_sec"] = np.where(
                ct_from_col == 999.9,
                df["ct_diff_sec"],
                ct_from_col
            )
        
        # Ensure the first row has a valid time difference if possible
        if not df.empty and pd.isna(df.loc[0, "ct_diff_sec"]):
            if "ACTUAL CT" in df.columns:
                 df.loc[0, "ct_diff_sec"] = df.loc[0, "ACTUAL CT"]
            else: # Fallback if ACTUAL CT isn't present, fill with 0 to avoid errors
                 df.loc[0, "ct_diff_sec"] = 0
                 
        return df

    def _calculate_all_metrics(self) -> dict:
        """Executes the full analysis pipeline."""
        df = self._prepare_data()

        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        mode_ct = df["ACTUAL CT"].mode().iloc[0] if not df["ACTUAL CT"].mode().empty else 0
        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)

        # Flag cycles that are outside the normal tolerance band as potential stops.
        # We cap this at 8 hours (28800s) to exclude major shutdowns.
        df["stop_flag"] = np.where(
            (df["ct_diff_sec"] > upper_limit) & (df["ct_diff_sec"] <= 28800), 1, 0
        )
        df.loc[0, "stop_flag"] = 0 # The first shot cannot be a stop

        # Identify the START of a stop event, not every cycle within it.
        # This prevents double-counting. An event is where the flag turns from 0 to 1.
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # --- Calculate Key Metrics ---
        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds()
        production_time_sec = total_runtime_sec - downtime_sec
        
        # MTTR: Mean Time To Repair/Recover (average duration of a stop event)
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0
        
        # MTBF: Mean Time Between Failures (average duration of uptime)
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        
        stability_index = (mtbf_min / (mtbf_min + mttr_min) * 100) if (mtbf_min + mttr_min) > 0 else 100
        
        # Efficiency is based on shots within cycle time tolerance
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0

        # --- Time Bucket Analysis ---
        # Group consecutive running periods to analyze their duration
        df["run_group"] = df["stop_event"].cumsum()
        run_durations = df[df['stop_flag'] == 0].groupby("run_group")["ct_diff_sec"].sum().div(60).reset_index(name="duration_min")
        
        # Build bins for the bar chart
        max_minutes = min(run_durations["duration_min"].max() if not run_durations.empty else 0, 240)
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
        run_durations["time_bucket"] = pd.cut(run_durations["duration_min"], bins=edges, labels=labels, right=False)

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "normal_shots": normal_shots, "efficiency": efficiency,
            "stop_events": stop_events, "downtime_min": downtime_sec / 60, "production_time_min": production_time_sec / 60,
            "total_runtime_min": total_runtime_sec / 60, "mttr_min": mttr_min, "mtbf_min": mtbf_min,
            "stability_index": stability_index, "run_durations": run_durations, "bucket_labels": labels,
        }

# --- UI and Plotting Functions ---

@st.cache_data
def create_gauge(value, title, reference_value, color):
    """Creates a Plotly bullet gauge indicator."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': '#d73027'},
                {'range': [60, 85], 'color': '#fee090'},
                {'range': [85, 100], 'color': 'lightgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 'value': reference_value
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def display_main_dashboard(results: dict):
    """Renders the main KPI dashboard."""
    st.subheader("📈 Key Performance Indicators")
    
    # Row 1: Gauges
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_gauge(results.get('efficiency', 0) * 100, "Efficiency (%)", 95, "royalblue"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_gauge(results.get('stability_index', 0), "Stability Index (%)", 90, "darkorange"),
            use_container_width=True
        )
    
    st.markdown("---")

    # Row 2: Core Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MTTR (min)", f"{results.get('mttr_min', 0):.2f}", help="Mean Time To Recover: The average duration of a stop.")
    col2.metric("MTBF (min)", f"{results.get('mtbf_min', 0):.2f}", help="Mean Time Between Failures: The average uptime between stops.")
    col3.metric("Total Stops", f"{results.get('stop_events', 0):,}", help="The total number of detected stop events.")
    col4.metric("Downtime (hrs)", f"{results.get('downtime_min', 0) / 60:.2f}", help="Total time the process was stopped.")
    col5.metric("Total Shots", f"{results.get('total_shots', 0):,}", help="Total number of cycles recorded in the period.")
    
    st.markdown("---")

    # Row 3: Cycle Time Details
    st.subheader("⏱️ Cycle Time Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}", help="The most frequent cycle time, used as the baseline.")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}", help="Cycles below this are flagged (Mode CT - Tolerance).")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}", help="Cycles above this are flagged (Mode CT + Tolerance).")

@st.cache_data
def plot_time_bucket_analysis(run_durations, bucket_labels, title="Time Bucket Analysis"):
    """Generates and displays the time bucket bar chart."""
    if run_durations.empty or 'time_bucket' not in run_durations.columns:
        st.info("No valid run duration data to plot for Time Bucket Analysis.")
        return
        
    bucket_counts = run_durations["time_bucket"].value_counts().reindex(bucket_labels, fill_value=0)
    
    fig = px.bar(
        bucket_counts,
        x=bucket_counts.index, y=bucket_counts.values,
        title=title,
        labels={"x": "Continuous Run Duration (min)", "y": "Number of Occurrences"},
        text_auto=True,
        color=bucket_counts.index,
        color_discrete_sequence=px.colors.sequential.Viridis_r
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")
uploaded_file = st.sidebar.file_uploader("Upload Run Rate Excel", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("👈 Upload a cleaned run rate Excel file to begin.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

df_raw = load_data(uploaded_file)

# Identify the column for tool/equipment selection
id_col = "TOOLING ID" if "TOOLING ID" in df_raw.columns else "EQUIPMENT CODE"
if id_col not in df_raw.columns:
    st.error("File must contain either 'TOOLING ID' or 'EQUIPMENT CODE'.")
    st.stop()
    
tool_id = st.sidebar.selectbox(f"Select {id_col}", df_raw[id_col].unique())
df_tool = df_raw.loc[df_raw[id_col] == tool_id].copy()

if df_tool.empty:
    st.warning(f"No data found for: {tool_id}")
    st.stop()

st.sidebar.markdown("---")
tolerance = st.sidebar.slider(
    "Tolerance Band (% of Mode CT)", 0.01, 0.20, 0.05, 0.01,
    help="Defines the ±% around the Mode CT to classify normal cycles vs. stops."
)

# --- Instantiate Calculator and Display Global Dashboard ---
@st.cache_data
def get_calculator(df, tol):
    return RunRateCalculator(df, tol)

calculator_full = get_calculator(df_tool, tolerance)

st.title(f"Run Rate Dashboard: {tool_id}")
with st.container(border=True):
    display_main_dashboard(calculator_full.results)

st.markdown("---")

# --- Page Navigation and Content ---
page = st.radio(
    "Select Analysis View",
    ["📊 Daily Deep-Dive", "🗓️ Weekly Trends", "📂 View Processed Data"],
    horizontal=True
)

if page == "📊 Daily Deep-Dive":
    st.header("Daily Analysis")
    df_processed = calculator_full.results["processed_df"]
    
    available_dates = df_processed["shot_time"].dt.date.unique()
    selected_date = st.selectbox(
        "Select Date",
        options=available_dates,
        index=len(available_dates)-1, # Default to the most recent date
        format_func=lambda date: pd.to_datetime(date).strftime('%d %B %Y')
    )

    df_day = df_processed[df_processed["shot_time"].dt.date == selected_date]
    
    if df_day.empty:
        st.warning(f"No data for {selected_date.strftime('%d %B %Y')}.")
    else:
        # Create a new calculator instance specifically for the selected day's data
        calc_day = RunRateCalculator(df_day, tolerance)
        with st.container(border=True):
            display_main_dashboard(calc_day.results)
        
        plot_time_bucket_analysis(
            calc_day.results["run_durations"],
            calc_day.results["bucket_labels"],
            title=f"Time Bucket Analysis for {selected_date.strftime('%d %B %Y')}"
        )


elif page == "🗓️ Weekly Trends":
    st.header("Weekly Trend Analysis")
    df_processed = calculator_full.results["processed_df"]
    df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
    
    # Calculate weekly summaries
    weekly_summary_data = []
    for week in sorted(df_processed['week_start'].unique()):
        df_week = df_processed[df_processed['week_start'] == week]
        calc_week = RunRateCalculator(df_week, tolerance)
        res = calc_week.results
        res['week_start'] = week
        weekly_summary_data.append(res)
        
    summary_df = pd.DataFrame(weekly_summary_data)
    
    if not summary_df.empty:
        # Plot MTTR/MTBF Trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary_df['week_start'], y=summary_df['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=summary_df['week_start'], y=summary_df['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green'), yaxis='y2'))
        fig.update_layout(
            title='Weekly MTTR & MTBF Trend',
            yaxis=dict(title='MTTR (min)'),
            yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot Stability/Efficiency Trend
        fig2 = px.line(summary_df, x='week_start', y=['stability_index', 'efficiency'], title='Weekly Stability & Efficiency Trend', markers=True)
        # Convert 'efficiency' to percentage for the plot
        fig2.data[1].y = fig2.data[1].y * 100
        fig2.update_layout(
            yaxis_title='Percentage (%)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("View Weekly Summary Data"):
            display_cols = ['week_start', 'total_shots', 'mttr_min', 'mtbf_min', 'stability_index', 'efficiency', 'stop_events']
            st.dataframe(summary_df[display_cols].style.format({
                'mttr_min': '{:.2f}', 'mtbf_min': '{:.2f}', 'stability_index': '{:.1f}%', 'efficiency': '{:.2%}'
            }))


elif page == "📂 View Processed Data":
    st.header("Processed Cycle Data")
    df_display = calculator_full.results["processed_df"].copy()
    
    # Format for display
    df_display["Stop"] = np.where(df_display["stop_flag"] == 1, "🔴", "🟢")
    df_display["Stop Event Start"] = np.where(df_display["stop_event"], "🛑", "")
    
    st.dataframe(df_display[[
        "shot_time", "ACTUAL CT", "ct_diff_sec", "Stop", "Stop Event Start"
    ]].rename(columns={
        "shot_time": "Shot Time",
        "ACTUAL CT": "Actual CT (sec)",
        "ct_diff_sec": "Time Since Last Shot (sec)"
    }), use_container_width=True)
    
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Processed Data (CSV)",
        data=csv,
        file_name=f"{tool_id}_processed_data.csv",
        mime="text/csv"
    )