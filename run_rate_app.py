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
    fig.update_layout(showlegend=False)
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
    """Creates a multi-sheet Excel report with all key analyses and parameters."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Dashboard
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

        # Sheets 2 & 3: Daily and Weekly Summaries
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

        # Sheet 4: Time Bucket Analysis
        bucket_counts = results["run_durations"]["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
        df_buckets = bucket_counts.reset_index().rename(columns={'index': 'Run Duration (min)', 'time_bucket': 'Occurrences'})
        df_buckets.to_excel(writer, sheet_name="Time Bucket Analysis", index=False)
        
        # Sheet 5: Processed Data with Parameters
        df_export = results['processed_df'].copy()
        export_cols = ['shot_time', 'ACTUAL CT', 'ct_diff_sec', 'stop_flag', 'stop_event', 'run_group']
        df_export_final = df_export[export_cols]
        df_export_final.to_excel(writer, sheet_name="Processed Shot Data", index=False, startrow=5)
        
        # Add parameters to the top of the sheet
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

elif page == "🗓️ Weekly Trends":
    st.header("Weekly Trend Analysis")
    df_processed = calculator_full.results["processed_df"]
    df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
    
    weekly_summary_data = [RunRateCalculator(df_week, tolerance).results | {'week_start': week} for week, df_week in df_processed.groupby('week_start')]
    summary_df = pd.DataFrame(weekly_summary_data)
    
    if not summary_df.empty:
        summary_df['bad_shots'] = summary_df['total_shots'] - summary_df['normal_shots']
        display_cols = ['week_start', 'total_shots', 'normal_shots', 'bad_shots', 'stop_events', 'mttr_min', 'mtbf_min', 'stability_index']
        
        summary_display = summary_df[display_cols].copy()
        summary_display.rename(columns={
            'week_start': 'Week Starting', 'total_shots': 'Total Shots', 'normal_shots': 'Normal Shots', 'bad_shots': 'Bad Shots',
            'stop_events': 'Stop Events', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'stability_index': 'Stability Index (%)'
        }, inplace=True)
        summary_display['Week Starting'] = pd.to_datetime(summary_display['Week Starting']).dt.strftime('%d %b %Y')
        
        st.subheader("Weekly Summary Table")
        def highlight_stability(val):
            if pd.isna(val) or val > 70: return ""
            elif val <= 50: return "background-color: rgba(255, 77, 77, 0.3);"
            else: return "background-color: rgba(255, 191, 0, 0.3);"
        st.dataframe(summary_display.style
            .applymap(highlight_stability, subset=["Stability Index (%)"])
            .format({'MTTR (min)': '{:.2f}', 'MTBF (min)': '{:.2f}', 'Stability Index (%)': '{:.2f}%'}), use_container_width=True)

        st.markdown("---")
        st.subheader("Weekly Trend Charts")
        
        plot_mt_trend(summary_df, 'week_start', 'mttr_min', 'mtbf_min')
        st.caption("This chart tracks the weekly trend of average stop duration (MTTR - red) and average uptime (MTBF - green). A rising green line and falling red line indicate improving reliability.")
        with st.expander("View Data Table"):
            st.dataframe(summary_display, use_container_width=True)

        plot_stability_trend(summary_df, 'week_start', 'stability_index')
        display_stability_index_explanation()

        all_run_durations = calculator_full.results['run_durations']
        if not all_run_durations.empty and 'run_group' in all_run_durations.columns:
            df_proc_groups = calculator_full.results['processed_df'][['shot_time', 'run_group']].drop_duplicates()
            run_times = all_run_durations.merge(df_proc_groups, on='run_group', how='left').dropna(subset=['shot_time'])
            run_times['week_start'] = run_times['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
            bucket_weekly = run_times.groupby(['week_start', 'time_bucket'], observed=False).size().reset_index(name='count')
            
            fig_bucket = px.bar(bucket_weekly, x='week_start', y='count', color='time_bucket', title='Weekly Time Bucket Trend',
                                category_orders={"time_bucket": calculator_full.results["bucket_labels"]},
                                color_discrete_map=calculator_full.results["bucket_color_map"],
                                labels={'week_start': 'Week Starting', 'count': 'Number of Occurrences', 'time_bucket': 'Run Duration (min)'})
            st.plotly_chart(fig_bucket, use_container_width=True)
            st.caption("This chart shows the weekly evolution of continuous run durations. Look for a shift from shorter red/orange bars to longer blue bars over time as stability improves.")
            with st.expander("View Data Table"):
                bucket_weekly_display = bucket_weekly.copy()
                bucket_weekly_display['week_start'] = pd.to_datetime(bucket_weekly_display['week_start']).dt.strftime('%d %b %Y')
                st.dataframe(bucket_weekly_display.rename(columns={'week_start': 'Week Starting', 'time_bucket': 'Run Duration (min)', 'count': 'Occurrences'}), use_container_width=True)


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
        "shot_time": "Shot Time", "ACTUAL CT": "Actual CT (sec)", "ct_diff_sec": "Time Since Last Shot (sec)", "run_group": "Run Group ID"
    })
    
    st.dataframe(display_subset.style.format({
        "Actual CT (sec)": "{:.1f}", "Time Since Last Shot (sec)": "{:.2f}"
    }), use_container_width=True)
    
    excel_data = export_to_excel(calculator_full.results, calculator_full.tolerance)
    st.download_button(
        label="📥 Download Full Excel Report", data=excel_data,
        file_name=f"{tool_id}_full_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )