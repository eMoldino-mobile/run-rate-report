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

# --- CSS for Custom Styling ---
def load_css():
    st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background-color: #F0F2F6;
        }
    </style>
    """, unsafe_allow_html=True)

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
        
        # Hourly summary for the new chart
        df['hour'] = df['shot_time'].dt.hour
        hourly_summary = df.groupby('hour').agg(
            hourly_mode_ct=('ACTUAL CT', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            average_ct=('ct_diff_sec', 'mean')
        ).reset_index()
        hourly_summary.dropna(subset=['hourly_mode_ct'], inplace=True)
        hourly_summary['lower_band'] = hourly_summary['hourly_mode_ct'] * (1 - self.tolerance)
        hourly_summary['upper_band'] = hourly_summary['hourly_mode_ct'] * (1 + self.tolerance)

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "efficiency": efficiency, "stop_events": stop_events,
            "normal_shots": normal_shots, "hourly_ct_summary": hourly_summary
        }

# --- UI Helper and Plotting Functions ---

def styled_metric(label, value, color):
    st.markdown(f"""
    <div style="border-left: 6px solid {color}; padding: 1rem; background-color: #FFFFFF; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);">
        <div style="font-size: 1rem; color: #555555;">{label}</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: #111111;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def plot_hourly_ct_trend(df_hourly):
    fig = go.Figure()

    # Tolerance Band
    fig.add_trace(go.Scatter(
        x=df_hourly['hour'], y=df_hourly['upper_band'],
        mode='lines', line=dict(color='lightgray', width=1),
        name='Upper Tolerance Band'
    ))
    fig.add_trace(go.Scatter(
        x=df_hourly['hour'], y=df_hourly['lower_band'],
        mode='lines', line=dict(color='lightgray', width=1),
        fill='tonexty', fillcolor='rgba(211,211,211,0.2)',
        name='Lower Tolerance Band'
    ))

    # Actual Data Lines
    fig.add_trace(go.Scatter(
        x=df_hourly['hour'], y=df_hourly['hourly_mode_ct'],
        mode='lines+markers', line=dict(color='#00829B', width=3),
        name='Hourly Mode CT'
    ))
    fig.add_trace(go.Scatter(
        x=df_hourly['hour'], y=df_hourly['average_ct'],
        mode='lines', line=dict(color='#E74C3C', width=2, dash='dot'),
        name='Hourly Average CT'
    ))

    fig.update_layout(
        title="Hourly Cycle Time Performance vs. Tolerance",
        xaxis_title="Hour of Day",
        yaxis_title="Cycle Time (sec)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main Application Logic ---
load_css()
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

# Page selection is simplified as we are focusing on the main dashboard for now.
# You can add back the other pages ('Weekly Trends', 'Processed Data') if needed.
page = "📊 Daily Deep-Dive"

if page == "📊 Daily Deep-Dive":
    st.header("Daily Analysis")
    df_processed = calculator_full.results["processed_df"]
    available_dates = df_processed["shot_time"].dt.date.unique()
    
    if len(available_dates) == 0:
        st.warning("No date data available in the uploaded file.")
    else:
        selected_date_str = pd.to_datetime(available_dates[-1]).strftime('%d %b %Y')
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'))
        
        # Create a calculator instance specifically for the selected day
        df_day = df_processed[df_processed["shot_time"].dt.date == selected_date]
        if df_day.empty:
            st.warning(f"No data for {selected_date.strftime('%d %b %Y')}.")
        else:
            calc_day = RunRateCalculator(df_day, tolerance)
            results_day = calc_day.results
            
            st.subheader(f"Dashboard for {selected_date.strftime('%d %b %Y')}")
            
            # --- KPI Cards ---
            cols = st.columns(4)
            with cols[0]:
                eff = results_day.get('efficiency', 0) * 100
                styled_metric("Efficiency", f"{eff:.1f}%", "#2ECC71" if eff >= 90 else "#F39C12")
            with cols[1]:
                stops = results_day.get('stop_events', 0)
                styled_metric("Total Stops", f"{stops}", "#E74C3C" if stops > 0 else "#2ECC71")
            with cols[2]:
                total_shots = results_day.get('total_shots', 0)
                styled_metric("Total Shots", f"{total_shots:,}", "#3498DB")
            with cols[3]:
                mode_ct = results_day.get('mode_ct', 0)
                styled_metric("Mode CT", f"{mode_ct:.2f}s", "#9B59B6")

            st.markdown("---")

            # --- Main Chart ---
            df_hourly_ct = results_day.get('hourly_ct_summary')
            if df_hourly_ct is not None and not df_hourly_ct.empty:
                st.plotly_chart(plot_hourly_ct_trend(df_hourly_ct), use_container_width=True)
            else:
                st.info("Not enough data to generate the hourly cycle time trend for this day.")