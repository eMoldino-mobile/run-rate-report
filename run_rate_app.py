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
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- Core Calculation Class ---
class RunRateCalculator:
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
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0
        
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0
        
        total_runtime_sec = (df["shot_time"].max() - df["shot_time"].min()).total_seconds() if total_shots > 1 else 0
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        production_time_sec = total_runtime_sec - downtime_sec
        mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        stability_index = (mtbf_min / (mtbf_min + mttr_min) * 100) if (mtbf_min + mttr_min) > 0 else (100.0 if stop_events == 0 else 0.0)
        
        hourly_summary = self._calculate_hourly_summary(df)

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "efficiency": efficiency, "stop_events": stop_events, "normal_shots": normal_shots,
            "mttr_min": mttr_min, "mtbf_min": mtbf_min, "stability_index": stability_index,
            "hourly_summary": hourly_summary
        }

# --- UI Helper and Plotting Functions ---

def create_gauge(value, title, steps=None):
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps:
        gauge_config['steps'] = steps
        gauge_config['bar'] = {'color': '#262730'} 
    else:
        gauge_config['bar'] = {'color': "darkblue"}
        gauge_config['bgcolor'] = "lightgray"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title},
        gauge=gauge_config
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct):
    df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB') # Red for stops, Blue for normal
    fig = go.Figure()
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=lower_limit, x1=1, y1=upper_limit,
                  fillcolor=PASTEL_COLORS['green'], opacity=0.2, layer="below", line_width=0)
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['ct_diff_sec'],
        marker_color=df['color'],
        name='Shots'
    ))
    fig.update_layout(
        title="Cycle Time per Shot vs. Daily Tolerance",
        xaxis_title="Shot Sequence", yaxis_title="Cycle Time (sec)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_stability_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['hour'], y=df['stability_index'], mode="lines+markers", name="Stability Index (%)",
        line=dict(color="black", width=2),
        marker=dict(color=[PASTEL_COLORS['red'] if v <= 50 else PASTEL_COLORS['orange'] if v <= 70 else PASTEL_COLORS['green'] for v in df['stability_index']], size=10)
    ))
    for y0, y1, c in [(0, 50, PASTEL_COLORS['red']), (50, 70, PASTEL_COLORS['orange']), (70, 100, PASTEL_COLORS['green'])]:
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=c, opacity=0.2, line_width=0, layer="below")
    fig.update_layout(
        title="Hourly Stability Index Trend", yaxis=dict(title="Stability Index (%)", range=[0, 101]),
        xaxis_title="Hour of Day",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_stability_index_explanation():
    with st.expander("What is the Stability Index?"):
        st.markdown(""" ... """) # Content omitted for brevity

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

# --- Main Page Content ---
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
        results_day = calc_day.results
        
        # --- SECTION 1: Summary ---
        st.header(f"Daily Analysis for {selected_date.strftime('%d %b %Y')}")
        
        with st.container(border=True):
            st.subheader("Performance Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge(results_day.get('efficiency', 0) * 100, "Efficiency (%)"), use_container_width=True)
                st.metric("MTTR (min)", f"{results_day.get('mttr_min', 0):.2f}")
            with col2:
                stability_steps = [
                    {'range': [0, 50], 'color': PASTEL_COLORS['red']}, 
                    {'range': [50, 70], 'color': PASTEL_COLORS['orange']}, 
                    {'range': [70, 100], 'color': PASTEL_COLORS['green']}
                ]
                st.plotly_chart(create_gauge(results_day.get('stability_index', 0), "Stability Index (%)", steps=stability_steps), use_container_width=True)
                st.metric("MTBF (min)", f"{results_day.get('mtbf_min', 0):.2f}")
        
        st.subheader("Daily Cycle Time Parameters")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Limit (sec)", f"{results_day.get('lower_limit', 0):.2f}")
            with col2:
                st.markdown(f"""<div style="background-color: #e0f3ff; border-radius: 5px; padding: 0.1em 0.5em 0.5em 0.5em; text-align: center; margin-top: 1.2em;">
                                <label style="font-size: 0.8rem;">Mode CT (sec)</label>
                                <p style="font-size: 1.75rem; font-weight: bold; margin-bottom: 0;">{results_day.get('mode_ct', 0):.2f}</p>
                                </div>""", unsafe_allow_html=True)
            col3.metric("Upper Limit (sec)", f"{results_day.get('upper_limit', 0):.2f}")

        # --- SECTION 2: Main CT Graph ---
        st.markdown("---")
        plot_shot_bar_chart(results_day['processed_df'], results_day['lower_limit'], results_day['upper_limit'], results_day['mode_ct'])
        with st.expander("View Shot Data"):
            st.dataframe(results_day['processed_df'])

        # --- SECTION 3: Graph Section ---
        st.markdown("---")
        st.header("Hourly Analysis")
        
        plot_stability_trend(results_day['hourly_summary'])
        display_stability_index_explanation()