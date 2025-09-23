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
        
        # --- STRONGER COLOR PALETTES ---
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
    with st.expander("What is the Stability Index?"):
        st.markdown("""
        #### 🔹 Stability Index Calculation
        The Stability Index (SI) is derived from two key reliability metrics:
        - **MTBF (Mean Time Between Failures)** → average uptime between stoppages
        - **MTTR (Mean Time To Repair/Recover)** → average downtime per stoppage

        **Formula:**
        $$\\text{Stability Index (\\%)} = \\frac{\\text{MTBF}}{\\text{MTBF} + \\text{MTTR}} \\times 100$$

        *Special cases:* If no stops occur, SI is 100% (perfect stability).

        ---
        #### 🔹 Stability Index Meaning
        It gives a single number representing production consistency:
        - 🟩 **70–100% (Low Risk / Stable):** Long runs and short recoveries.
        - 🟨 **50–70% (Medium Risk / Watch):** Inconsistent flow that needs monitoring.
        - 🟥 **0–50% (High Risk / Unstable):** Frequent stops with long recovery.
        
        👉 In short, the **Stability Index is a risk-oriented health score of your production flow.**
        """)

def display_main_dashboard(results: dict):
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_gauge(results.get('efficiency', 0) * 100, "Efficiency (%)", "cornflowerblue"), use_container_width=True)
    with col2: st.plotly_chart(create_gauge(results.get('stability_index', 0), "Stability Index (%)", "lightseagreen"), use_container_width=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MTTR (min)", f"{results.get('mttr_min', 0):.2f}")
    col2.metric("MTBF (min)", f"{results.get('mtbf_min', 0):.2f}")
    col3.metric("Total Stops", f"{results.get('stop_events', 0):,}")
    col4.metric("Downtime (hrs)", f"{results.get('downtime_min', 0) / 60:.2f}")
    col5.metric("Total Shots", f"{results.get('total_shots', 0):,}")
    
    st.markdown("---",)
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}")

@st.cache_data
def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 20}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_time_bucket_analysis(run_durations, bucket_labels, color_map, title="Time Bucket Analysis"):
    fig = px.bar(
        run_durations["time_bucket"].value_counts().reindex(bucket_labels, fill_value=0),
        title=title, labels={"index": "Continuous Run Duration (min)", "value": "Number of Occurrences"},
        text_auto=True, color=bucket_labels, color_discrete_map=color_map
    )
    fig.update_layout(legend_title_text='Run Duration (min)')
    st.plotly_chart(fig, use_container_width=True)

def plot_mt_trend(df, time_col, mttr_col, mtbf_col, title="MTTR & MTBF Trend"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[mttr_col], name='MTTR (min)', mode='lines+markers', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[mtbf_col], name='MTBF (min)', mode='lines+markers', line=dict(color='green'), yaxis='y2'))
    fig.update_layout(title=title, yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_stability_trend(df, time_col, stability_col, title="Stability Index Trend"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[stability_col], mode="lines+markers",
        name="Stability Index (%)", line=dict(color="blue", width=2),
        marker=dict(color=["red" if v <= 50 else "orange" if v <= 70 else "green" for v in df[stability_col]], size=8)
    ))
    for y0, y1, c in [(0, 50, "red"), (50, 70, "orange"), (70, 100, "green")]:
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=c, opacity=0.1, line_width=0, layer="below")
    fig.update_layout(
        title=title, xaxis_title=time_col.replace('_', ' ').title(),
        yaxis=dict(title="Stability Index (%)", range=[0, 101]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def export_to_excel(results: dict, tolerance: float):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_kpis = {
            "Metric": ["Total Shots", "Normal Shots", "Stop Events", "Efficiency (%)", 
                       "Stability Index (%)", "MTTR (min)", "MTBF (min)", "Downtime (min)",
                       "Mode CT (sec)", "Lower Limit (sec)", "Upper Limit (sec)"],
            "Value": [results.get(k, 0) for k in ['total_shots', 'normal_shots', 'stop_events']] +
                     [f"{results.get('efficiency', 0)*100:.2f}", f"{results.get('stability_index', 0):.2f}",
                      f"{results.get('mttr_min', 0):.2f}", f"{results.get('mtbf_min', 0):.2f}", f"{results.get('downtime_min', 0):.2f}",
                      f"{results.get('mode_ct', 0):.2f}", f"{results.get('lower_limit', 0):.2f}", f"{results.get('upper_limit', 0):.2f}"]
        }
        pd.DataFrame(summary_kpis).to_excel(writer, sheet_name="Dashboard", index=False)

        df_processed = results['processed_df'].copy()
        if not df_processed.empty:
            df_processed['date'] = df_processed['shot_time'].dt.date
            daily_summary_data = [RunRateCalculator(df_day, tolerance).results | {'date': date} for date, df_day in df_processed.groupby('date')]
            df_daily = pd.DataFrame(daily_summary_data)
            
            df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
            weekly_summary_data = [RunRateCalculator(df_week, tolerance).results | {'week_start': week} for week, df_week in df_processed.groupby('week_start')]
            df_weekly = pd.DataFrame(weekly_summary_data)

            for df_summary, name in [(df_daily, "Daily"), (df_weekly, "Weekly")]:
                df_summary['bad_shots'] = df_summary['total_shots'] - df_summary['normal_shots']
                cols = [c for c in [df_summary.columns[0], 'total_shots', 'normal_shots', 'bad_shots', 'stop_events', 'mttr_min', 'mtbf_min', 'stability_index', 'efficiency'] if c in df_summary.columns]
                df_summary[cols].to_excel(writer, sheet_name=f"{name} Summary", index=False)

        bucket_counts = results["run_durations"]["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
        df_buckets = bucket_counts.reset_index().rename(columns={'index': 'Run Duration (min)', 'time_bucket': 'Occurrences'})
        df_buckets.to_excel(writer, sheet_name="Time Bucket Analysis", index=False)
        
        df_export = results['processed_df'].copy()
        export_cols = ['shot_time', 'ACTUAL CT', 'ct_diff_sec', 'stop_flag', 'stop_event', 'run_group']
        df_export_final = df_export[export_cols]
        df_export_final.to_excel(writer, sheet_name="Processed Shot Data", index=False, startrow=5)
        
        ws = writer.sheets["Processed Shot Data"]
        ws['A1'] = "Calculation Parameters"
        ws['A2'] = "Mode CT (sec)"
        ws['B2'] = results['mode_ct']
        ws['A3'] = "Lower Limit (sec)"
        ws['B3'] = results['lower_limit']
        ws['A4'] = "Upper Limit (sec)"
        ws['B4'] = results['upper_limit']

    return output.getvalue()

# --- Main Application Logic ---
st.sidebar.title("Run Rate Report Generator ⚙️")
uploaded_file = st.sidebar.file