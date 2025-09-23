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
    # ... (This class remains unchanged from the previous version)
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
        
        reds = px.colors.sequential.Reds[4:8]
        blues = px.colors.sequential.Blues[3:9]
        greens = px.colors.sequential.Greens[4:9]
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

def create_kpi_card(title, value, unit=""):
    """Creates a single KPI card figure."""
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = value,
        number = {'suffix': unit, 'font': {'size': 40}},
        title = {"text": title, "font": {"size": 20}},
    ))
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_control_chart_example(mode_ct, lower_limit, upper_limit):
    """Creates an example control chart to explain the tolerance bands."""
    fig = go.Figure()

    # In-spec green zone
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=lower_limit, x1=1, y1=upper_limit,
                  fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
    
    # Lines for limits and mode
    fig.add_hline(y=upper_limit, line_dash="solid", line_color="red", annotation_text="Upper Limit", annotation_position="bottom right")
    fig.add_hline(y=mode_ct, line_dash="dash", line_color="blue", annotation_text="Mode CT", annotation_position="bottom right")
    fig.add_hline(y=lower_limit, line_dash="solid", line_color="red", annotation_text="Lower Limit", annotation_position="bottom right")
    
    # Example data points
    example_x = [1, 2, 3, 4, 5, 6, 7, 8]
    example_y = [mode_ct - 2, upper_limit + 5, mode_ct + 1, lower_limit - 4, mode_ct, mode_ct - 3, upper_limit - 1, mode_ct + 2]
    colors = ['red' if y > upper_limit or y < lower_limit else 'royalblue' for y in example_y]
    
    fig.add_trace(go.Scatter(x=example_x, y=example_y, mode='lines+markers', name='Example Shots',
                             marker=dict(color=colors, size=10, symbol='diamond'), line=dict(color='grey')))
    
    fig.update_layout(
        title="Visualizing Cycle Time Tolerance",
        xaxis_title="Example Shot Sequence",
        yaxis_title="Cycle Time (sec)",
        showlegend=False
    )
    return fig

def display_main_dashboard(results: dict):
    # Top Gauges
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_gauge(results.get('efficiency', 0) * 100, "Efficiency (%)", "cornflowerblue"), use_container_width=True)
    with col2: st.plotly_chart(create_gauge(results.get('stability_index', 0), "Stability Index (%)", "lightseagreen"), use_container_width=True)
    
    st.markdown("---")
    
    # KPI Grid
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1: st.plotly_chart(create_kpi_card("MTTR", round(results.get('mttr_min', 0), 2), " min"), use_container_width=True)
    with col2: st.plotly_chart(create_kpi_card("MTBF", round(results.get('mtbf_min', 0), 2), " min"), use_container_width=True)
    with col3: st.plotly_chart(create_kpi_card("Total Stops", results.get('stop_events', 0)), use_container_width=True)
    
    col1, col2 = st.columns([2,1]) # Make Downtime wider
    with col1: st.plotly_chart(create_kpi_card("Total Downtime", round(results.get('downtime_min', 0) / 60, 2), " hrs"), use_container_width=True)
    with col2: st.plotly_chart(create_kpi_card("Total Shots", f"{results.get('total_shots', 0):,}", ""), use_container_width=True)

    st.markdown("---")

    # Control Chart Explanation
    st.subheader("Cycle Time Analysis")
    st.plotly_chart(plot_control_chart_example(
        results.get('mode_ct', 0),
        results.get('lower_limit', 0),
        results.get('upper_limit', 0)
    ), use_container_width=True)
    st.caption("This chart explains the tolerance band. Shots falling within the green zone are 'Normal,' while shots outside this zone are flagged as 'Stoppages.'")

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
            
            # The rest of the page logic (Daily Charts, Hourly Breakdown, etc.) remains the same
            # ...

elif page == "🗓️ Weekly Trends":
    # This page logic remains the same
    pass

elif page == "📂 View Processed Data":
    # This page logic remains the same
    pass