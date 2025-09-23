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
        
        df['hour'] = df['shot_time'].dt.hour
        hourly_ct_summary = df.groupby('hour').agg(
            average_ct=('ct_diff_sec', 'mean')
        ).reset_index()

        return {
            "processed_df": df, "mode_ct": mode_ct, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "total_shots": total_shots, "efficiency": efficiency, "stop_events": stop_events,
            "downtime_min": downtime_sec / 60, "mttr_min": mttr_min, "mtbf_min": mtbf_min,
            "stability_index": stability_index, "run_durations": run_durations, "bucket_labels": labels,
            "bucket_color_map": bucket_color_map, "normal_shots": normal_shots,
            "hourly_ct_summary": hourly_ct_summary
        }

# --- UI Helper and Plotting Functions ---

def styled_metric(label, value, color):
    st.markdown(f"""
    <div style="border-left: 6px solid {color}; padding: 1rem; background-color: #FFFFFF; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);">
        <div style="font-size: 1rem; color: #555555;">{label}</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: #111111;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def plot_hourly_ct_trend(df_hourly, daily_mode_ct, daily_lower_limit, daily_upper_limit):
    fig = go.Figure()
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=daily_lower_limit, x1=1, y1=daily_upper_limit,
                  fillcolor="rgba(211,211,211,0.2)", layer="below", line_width=0)
    fig.add_hline(y=daily_upper_limit, line_dash="dot", line_color="red", annotation_text="Daily Upper Limit")
    fig.add_hline(y=daily_mode_ct, line_dash="dash", line_color="#00829B", annotation_text="Daily Mode CT")
    fig.add_hline(y=daily_lower_limit, line_dash="dot", line_color="red", annotation_text="Daily Lower Limit")
    fig.add_trace(go.Scatter(
        x=df_hourly['hour'], y=df_hourly['average_ct'],
        mode='lines+markers', line=dict(color='#E74C3C', width=3),
        name='Hourly Average CT'
    ))
    fig.update_layout(
        title="Hourly Average Cycle Time vs. Daily Tolerance", xaxis_title="Hour of Day",
        yaxis_title="Cycle Time (sec)", template="plotly_white", showlegend=False
    )
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
            results_day = calc_day.results
            
            st.subheader(f"Dashboard for {selected_date.strftime('%d %b %Y')}")
            
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

            df_hourly_ct = results_day.get('hourly_ct_summary')
            if df_hourly_ct is not None and not df_hourly_ct.empty:
                st.plotly_chart(plot_hourly_ct_trend(
                    df_hourly_ct,
                    results_day['mode_ct'],
                    results_day['lower_limit'],
                    results_day['upper_limit']
                ), use_container_width=True)
                st.caption("This chart shows the average cycle time for each hour (red line) compared to the acceptable tolerance band (grey area) for the entire day.")
            else:
                st.info("Not enough data to generate the hourly cycle time trend for this day.")
            
            st.markdown("---")
            st.subheader("Additional Daily Charts")
            plot_time_bucket_analysis(results_day["run_durations"], results_day["bucket_labels"], results_day["bucket_color_map"], f"Time Bucket Analysis")
            st.caption("This chart groups continuous production runs by their duration. Shorter red bars indicate frequent stops, while longer blue bars show periods of stable production.")

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
                display_table = stoppage_alerts[['shot_time', 'Duration (min)', 'Shots Since Last Stop']].rename(columns={"shot_time": "Event Time"})
                st.dataframe(display_table.style.format({'Duration (min)': '{:.1f}'}), use_container_width=True)

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
        
        plot_mt_trend(summary_df, 'week_start', 'mttr_min', 'mtbf_min', "Weekly MTTR & MTBF Trend")
        
        plot_stability_trend(summary_df, 'week_start', 'stability_index', "Weekly Stability Index Trend")

elif page == "📂 View Processed Data":
    st.header("Processed Cycle Data")
    results = calculator_full.results
    st.subheader("Calculation Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}")

    st.markdown("---")
    st.subheader("Shot-by-Shot Data")
    
    df_display = results["processed_df"].copy()
    
    df_display["Stop Cycle"] = np.where(df_display["stop_flag"] == 1, "⚪", "")
    df_display["Stop Event Start"] = np.where(df_display["stop_event"], "🛑", "")
    
    downtime_event = np.where(df_display['stop_event'], df_display['ct_diff_sec'], np.nan)
    downtime_propagated = pd.Series(downtime_event).groupby(df_display['run_group']).transform('ffill')
    df_display['Downtime (sec)'] = np.where(df_display['stop_flag'] == 1, downtime_propagated, np.nan)
    
    conditions = [
        df_display['ct_diff_sec'] > results['upper_limit'],
        df_display['ct_diff_sec'] < results['lower_limit']
    ]
    outcomes = [
        df_display['ct_diff_sec'] - results['upper_limit'],
        df_display['ct_diff_sec'] - results['lower_limit']
    ]
    df_display['Excess Downtime (sec)'] = np.select(conditions, outcomes, default=np.nan)
    
    display_cols = ["shot_time", "ACTUAL CT", "ct_diff_sec", "Excess Downtime (sec)",
                    "Stop Cycle", "Stop Event Start", "Downtime (sec)", "run_group"]
                    
    display_subset = df_display[display_cols].rename(columns={
        "shot_time": "Shot Time", "ACTUAL CT": "Actual CT (sec)", 
        "ct_diff_sec": "Time Since Last Shot (sec)", "run_group": "Run Group ID"
    })
    
    st.dataframe(display_subset.style.format({
        "Actual CT (sec)": "{:.1f}", 
        "Time Since Last Shot (sec)": "{:.2f}",
        "Excess Downtime (sec)": "{:+.2f}",
        "Downtime (sec)": "{:.1f}"
    }), use_container_width=True)
    
    excel_data = export_to_excel(calculator_full.results, calculator_full.tolerance)
    st.download_button(
        label="📥 Download Full Excel Report", data=excel_data,
        file_name=f"{tool_id}_full_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )