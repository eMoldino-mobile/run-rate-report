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

        if "ACTUAL CT" in df.columns:
            # Calculate both potential sources of cycle time for each shot
            time_diff_sec = df["shot_time"].diff().dt.total_seconds()
            prev_actual_ct = df["ACTUAL CT"].shift(1)

            # --- New Logic to handle timestamp rounding ---
            # Define a small buffer in seconds. This prevents the timestamp, which might
            # be slightly higher due to rounding, from incorrectly overriding a valid Actual CT.
            rounding_buffer = 2.0 # seconds

            # A true stop is flagged if:
            # 1. The machine's CT reading is invalid (999.9).
            # OR
            # 2. The real-world time gap is significantly larger than the machine's
            #    reported cycle time (i.e., it exceeds the CT by more than the buffer).
            #    This catches long pauses between otherwise normal shots.
            use_timestamp_diff = (prev_actual_ct == 999.9) | \
                                 (time_diff_sec > (prev_actual_ct + rounding_buffer))

            # If the conditions above are met, we use the real-world time difference.
            # Otherwise, we trust the machine's more precise 'ACTUAL CT' value.
            df["ct_diff_sec"] = np.where(
                use_timestamp_diff,
                time_diff_sec,
                prev_actual_ct
            )
        else:
            # If there's no ACTUAL CT, we can only rely on the timestamp difference.
            df["ct_diff_sec"] = df["shot_time"].diff().dt.total_seconds()

        if not df.empty and pd.isna(df.loc[0, "ct_diff_sec"]):
                df.loc[0, "ct_diff_sec"] = df.loc[0, "ACTUAL CT"] if "ACTUAL CT" in df.columns else 0
        return df

    def _calculate_hourly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'stop_event' not in df.columns:
            return pd.DataFrame()

        df['hour'] = df['shot_time'].dt.hour
        df['downtime_min_event'] = np.where(df['stop_event'], df['ct_diff_sec'] / 60, np.nan)

        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        total_downtime = hourly_groups['downtime_min_event'].sum()
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['ct_diff_sec'].sum() / 60

        hourly_summary = pd.DataFrame({
            'stops': stops,
            'total_downtime_min': total_downtime
        })
        hourly_summary = hourly_summary.join(uptime_min.rename('uptime_min')).fillna(0).reset_index()

        # --- MTTR & MTBF ---
        hourly_summary['mttr_min'] = hourly_summary['total_downtime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])

        # --- Stability Index ---
        total_runtime = hourly_summary['uptime_min'] + hourly_summary['total_downtime_min']
        hourly_summary['stability_index'] = np.where(
            total_runtime > 0,
            (hourly_summary['uptime_min'] / total_runtime) * 100,
            np.where(hourly_summary['stops'] == 0, 100.0, 0.0)
        )

        return hourly_summary


    def _calculate_all_metrics(self) -> dict:
        df = self._prepare_data()
        if df.empty or "ACTUAL CT" not in df.columns:
            return {}

        # --- Mode CT and Tolerance Limits ---
        mode_ct = df["ACTUAL CT"].mode().iloc[0] if not df["ACTUAL CT"].mode().empty else 0
        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)

        # --- Stop Detection ---
        stop_condition = (
            ((df["ct_diff_sec"] < lower_limit) | (df["ct_diff_sec"] > upper_limit))
            & (df["ct_diff_sec"] <= 28800)
        )
        df["stop_flag"] = np.where(stop_condition, 1, 0)
        df.loc[0, "stop_flag"] = 0
        df["stop_event"] = (df["stop_flag"] == 1) & (
            df["stop_flag"].shift(1, fill_value=0) == 0
        )

        # --- Basic Counts ---
        total_shots = len(df)
        stop_events = df["stop_event"].sum()
        normal_shots = total_shots - df["stop_flag"].sum()
        efficiency = normal_shots / total_shots if total_shots > 0 else 0

        # --- MTTR & MTBF ---
        downtime_per_event_sec = df.loc[df["stop_event"], "ct_diff_sec"]
        mttr_min = (downtime_per_event_sec.mean() / 60) if stop_events > 0 else 0

        total_runtime_sec = (
            (df["shot_time"].max() - df["shot_time"].min()).total_seconds()
            if total_shots > 1
            else 0
        )
        downtime_sec = df.loc[df["stop_flag"] == 1, "ct_diff_sec"].sum()
        production_time_sec = total_runtime_sec - downtime_sec

        mtbf_min = (
            (production_time_sec / 60 / stop_events)
            if stop_events > 0
            else (production_time_sec / 60)
        )

        # --- Stability Index ---
        stability_index = (
            (production_time_sec / total_runtime_sec * 100)
            if total_runtime_sec > 0
            else (100.0 if stop_events == 0 else 0.0)
        )

        # --- Run Duration Buckets ---
        df["run_group"] = df["stop_event"].cumsum()
        run_durations = (
            df[df["stop_flag"] == 0]
            .groupby("run_group")["ct_diff_sec"]
            .sum()
            .div(60)
            .reset_index(name="duration_min")
        )

        # --- Bucket Binning ---
        max_minutes = (
            min(run_durations["duration_min"].max(), 240)
            if not run_durations.empty
            else 0
        )
        upper_bound = int(np.ceil(max_minutes / 20.0) * 20)
        edges = list(range(0, upper_bound + 20, 20)) if upper_bound > 0 else [0, 20]
        
        # --- FIX: Handle runs longer than the max bucket ---
        labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
        if edges and len(edges) > 1:
            last_edge_start = edges[-2]
            labels[-1] = f"{last_edge_start}+"
            edges[-1] = np.inf

        if not run_durations.empty:
            run_durations["time_bucket"] = pd.cut(
                run_durations["duration_min"], bins=edges, labels=labels, right=False,
                include_lowest=True
            )

        # --- Bucket Colors ---
        reds = px.colors.sequential.Reds[4:8]
        blues = px.colors.sequential.Blues[3:9]
        greens = px.colors.sequential.Greens[4:9]
        bucket_color_map = {}
        red_idx, blue_idx, green_idx = 0, 0, 0
        for label in labels:
            try:
                lower_bound_str = label.split("-")[0].replace('+', '')
                lower_bound = int(lower_bound_str)
            except (ValueError, IndexError):
                continue
            
            if lower_bound < 60:
                bucket_color_map[label] = reds[red_idx % len(reds)]
                red_idx += 1
            elif 60 <= lower_bound < 160:
                bucket_color_map[label] = blues[blue_idx % len(blues)]
                blue_idx += 1
            else:
                bucket_color_map[label] = greens[green_idx % len(greens)]
                green_idx += 1
        
        # --- Hourly Summary ---
        hourly_summary = self._calculate_hourly_summary(df)
        
        return {
            "processed_df": df,
            "mode_ct": mode_ct,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "total_shots": total_shots,
            "efficiency": efficiency,
            "stop_events": stop_events,
            "normal_shots": normal_shots,
            "mttr_min": mttr_min,
            "mtbf_min": mtbf_min,
            "stability_index": stability_index,
            "run_durations": run_durations,
            "bucket_labels": labels,
            "bucket_color_map": bucket_color_map,
            "hourly_summary": hourly_summary,
            "total_runtime_sec": total_runtime_sec,
            "production_time_sec": production_time_sec,
            "downtime_sec": downtime_sec,
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
    df = df.copy()
    df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB')

    # --- FIX: Shift stop bar's x-position to the start of the stop ---
    # The ct_diff_sec for a stop is on the row AFTER the stop.
    # We create a 'plot_time' to associate the stop duration with the timestamp
    # of the last shot BEFORE the stop, which is more intuitive.
    df['plot_time'] = df['shot_time']
    stop_indices = df[df['stop_flag'] == 1].index
    if not stop_indices.empty:
        # Get the timestamp from the previous row for each stop
        df.loc[stop_indices, 'plot_time'] = df['shot_time'].shift(1).loc[stop_indices]


    fig = go.Figure()

    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=df['plot_time'].min(), y0=lower_limit,
        x1=df['plot_time'].max(), y1=upper_limit,
        fillcolor=PASTEL_COLORS['green'], opacity=0.2,
        layer="below", line_width=0
    )

    fig.add_trace(go.Bar(
        x=df['plot_time'],
        y=df['ct_diff_sec'],
        marker_color=df['color'],
        name='Cycle Time',
    ))

    # --- New: Set a dynamic y-axis range to improve visualization ---
    # This prevents very large stop times from flattening the rest of the chart.
    y_axis_cap = min(max(mode_ct * 2, 50), 500)

    fig.update_layout(
        title="Cycle Time per Shot vs. Daily Tolerance",
        xaxis_title="Time",
        yaxis_title="Cycle Time (sec)",
        yaxis=dict(range=[0, y_axis_cap]), # Apply the calculated y-axis cap
        bargap=0.05,
        xaxis=dict(
            tickformat="%H:%M",
            showgrid=True
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_stability_trend(df, title="Hourly Stability Index Trend"):
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
        title=title, yaxis=dict(title="Stability Index (%)", range=[0, 101]),
        xaxis_title="Hour of Day",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def format_duration(seconds):
    """Converts seconds into a human-readable Hh Mm format."""
    if pd.isna(seconds) or seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

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

# --- SINGLE PAGE LAYOUT ---
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
        calc_day = RunRateCalculator(df_day.copy(), tolerance)
        results_day = calc_day.results
        
        # --- Explainer Section ---
        with st.expander("ℹ️ About This Dashboard", expanded=False):
            st.markdown("""
            ### Run Rate Analysis
            
            - **Real-time Capture:** Tracks the live run of the tooling in the press.  
            - **MTTR & MTBF:** Identifies stoppages and inefficiencies during a run, or aggregated over time.  
            - **Bucket Analysis:** Groups run durations into intervals to reveal patterns of short vs. long runs.  
            - **Stoppage Analysis:** Calculates MTTR and MTBF from stoppages, analyzing both duration and frequency.  
            - **Runtime Breakdown:** Separates total run time vs. downtime.  
            - **Cycle Insights:** Captures efficiency, stop counts, and cycle deviations at the shot level.  
            
            ---
            
            ### Calculation Methods
            
            - **Efficiency (%)** = Normal Shots ÷ Total Shots × 100  
            - **MTTR (min)** = Average downtime per stop event (downtime ÷ stop events)  
            - **MTBF (min)** = Average uptime between failures (uptime ÷ stop events)  
            - **Stability Index (%)** = Uptime ÷ (Uptime + Downtime) × 100  
            - **Bucket Analysis** = Groups each continuous run duration into 20-minute intervals (e.g., 0–20, 20–40, etc.)  
            
            ---
            
            ### Tolerance Slider
            
            The tolerance slider defines the **acceptable cycle time range** around the mode CT:  
            
            - **Lower Limit** = Mode CT × (1 − Tolerance)  
            - **Upper Limit** = Mode CT × (1 + Tolerance)  
            
            Any cycle time outside this range (but below 8 hours) is flagged as a **stop event**.  
            A smaller tolerance makes the dashboard more sensitive to deviations; a larger tolerance makes it less sensitive.  
            """)

        # --- SECTION 1: Summary ---
        st.header(f"Daily Analysis for {selected_date.strftime('%d %b %Y')}")

        # --- New Downtime Summary ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)

            mttr = results_day.get('mttr_min', 0)
            mtbf = results_day.get('mtbf_min', 0)
            total_duration = results_day.get('total_runtime_sec', 0)
            prod_time = results_day.get('production_time_sec', 0)
            down_time = results_day.get('downtime_sec', 0)

            prod_percent = (prod_time / total_duration * 100) if total_duration > 0 else 0
            down_percent = (down_time / total_duration * 100) if total_duration > 0 else 0

            col1.metric("MTTR", f"{mttr:.1f} min")
            col2.metric("MTBF", f"{mtbf:.1f} min")
            col3.metric("Total Run Duration", format_duration(total_duration))
            col4.metric("Production Time", format_duration(prod_time), f"{prod_percent:.1f}% of Total")
            col5.metric("Downtime", format_duration(down_time), f"{down_percent:.1f}% of Total", delta_color="inverse")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge(results_day.get('efficiency', 0) * 100, "Efficiency (%)"), use_container_width=True)
            with col2:
                stability_steps = [
                    {'range': [0, 50], 'color': PASTEL_COLORS['red']},
                    {'range': [50, 70], 'color': PASTEL_COLORS['orange']},
                    {'range': [70, 100], 'color': PASTEL_COLORS['green']}
                ]
                st.plotly_chart(create_gauge(results_day.get('stability_index', 0), "Stability Index (%)", steps=stability_steps), use_container_width=True)
                
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            
            total_shots = results_day.get('total_shots', 0)
            normal_shots = results_day.get('normal_shots', 0)
            stop_events_val = results_day.get('stop_events', 0)

            stopped_shots = total_shots - normal_shots
            
            normal_percent = (normal_shots / total_shots * 100) if total_shots > 0 else 0
            stopped_percent = (stopped_shots / total_shots * 100) if total_shots > 0 else 0

            col1.metric("Total Shots", f"{total_shots:,}")
            col2.metric("Normal Shots", f"{normal_shots:,}", f"{normal_percent:.1f}% of Total")
            col3.metric("Stop Count", f"{stop_events_val}", f"{stopped_percent:.1f}% Stopped Shots", delta_color="inverse")

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Limit (sec)", f"{results_day.get('lower_limit', 0):.2f}")
            with col2:
                with st.container(border=True):
                    st.metric("Mode CT (sec)", f"{results_day.get('mode_ct', 0):.2f}")
            col3.metric("Upper Limit (sec)", f"{results_day.get('upper_limit', 0):.2f}")

        # --- SECTION 2: Main CT Graph ---
        plot_shot_bar_chart(results_day['processed_df'], results_day['lower_limit'], results_day['upper_limit'], results_day['mode_ct'])

        # --- SECTION 3: Graph Section ---
        st.markdown("---")
        st.header("Hourly Analysis")

        # --- FIX: Centralize the logic for identifying complete runs ---
        run_durations_day = results_day["run_durations"]
        processed_day_df = results_day['processed_df']

        # Get all stop events for the day
        stop_events_df = processed_day_df.loc[processed_day_df['stop_event']].copy()

        complete_runs = pd.DataFrame()
        incomplete_run = pd.DataFrame()

        # A run is "complete" if it is terminated by a stop event.
        if not stop_events_df.empty:
            # A stop in run_group 'N' terminates the previous run, which is 'N-1'
            stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
            
            # Create a map from the terminated run_group to the time of the stop
            end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
            
            # Add the 'run_end_time' to the original run_durations dataframe using the map
            run_durations_day['run_end_time'] = run_durations_day['run_group'].map(end_time_map)
            
            # Complete runs are those that have a non-null end time
            complete_runs = run_durations_day.dropna(subset=['run_end_time']).copy()
            # Incomplete runs are the rest (typically just the final run of the day)
            incomplete_run = run_durations_day[run_durations_day['run_end_time'].isna()]
        else:
            # If there are no stops on this day, the entire day's production is one single, incomplete run.
            incomplete_run = run_durations_day


        col1, col2 = st.columns(2)
        with col1:
            # --- FIX: Time Bucket Analysis now only shows complete runs ---
            if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                bucket_counts = (
                    complete_runs["time_bucket"]
                    .value_counts()
                    .reindex(results_day["bucket_labels"], fill_value=0)
                )
        
                fig_bucket = px.bar(
                    bucket_counts,
                    title="Time Bucket Analysis (Completed Runs)",
                    labels={"index": "Run Duration (min)", "value": "Occurrences"},
                    text_auto=True,
                    color=bucket_counts.index,
                    color_discrete_map=results_day["bucket_color_map"]
                ).update_layout(legend_title_text='Run Duration')
        
                st.plotly_chart(fig_bucket, use_container_width=True)
                
                # Add a notification if a trailing run was excluded
                if not incomplete_run.empty:
                    duration = incomplete_run['duration_min'].iloc[0]
                    st.info(f"ℹ️ An incomplete trailing run of {duration:.1f} min was excluded from this chart.")

                with st.expander("View Complete Run Data"):
                    st.dataframe(complete_runs)
            else:
                st.info("No complete run durations were recorded for this day.")
        with col2:
            plot_stability_trend(results_day['hourly_summary'])
            with st.expander("View Stability Data"):
                st.dataframe(results_day['hourly_summary'])

        st.subheader("Hourly Bucket Trend")
        # --- FIX: Hourly Trend also uses the same `complete_runs` dataframe ---
        if not complete_runs.empty:
            complete_runs['hour'] = complete_runs['run_end_time'].dt.hour
            
            # Create a pivot table to explicitly count occurrences per hour
            pivot_df = pd.crosstab(
                index=complete_runs['hour'],
                columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results_day["bucket_labels"])
            )
            
            # Ensure all hours from 0-23 are present for a consistent x-axis
            all_hours_index = pd.Index(range(24), name='hour')
            pivot_df = pivot_df.reindex(all_hours_index, fill_value=0)

            # Plot the pivoted data
            fig_hourly_bucket = px.bar(
                pivot_df,
                x=pivot_df.index,
                y=pivot_df.columns,
                title='Hourly Distribution of Run Durations (anchored to stop hour)',
                barmode='stack',
                color_discrete_map=results_day["bucket_color_map"],
                labels={'hour': 'Hour of Stop', 'value': 'Number of Runs', 'variable': 'Run Duration (min)'}
            )
            fig_hourly_bucket.update_layout(
                height=400,
                margin=dict(l=40, r=40, t=80, b=40),
                xaxis=dict(range=[-0.5, 23.5], tickvals=list(range(24)))
            )
            st.plotly_chart(fig_hourly_bucket, use_container_width=True)
    
            with st.expander("View Bucket Trend Data", expanded=False):
                st.dataframe(pivot_df[pivot_df.sum(axis=1) > 0]) # Show only hours with stops
        else:
            st.info("No complete runs to display in the hourly trend.")


        st.subheader("Hourly MTTR & MTBF Trend")
        hourly_summary = results_day['hourly_summary']
        if not hourly_summary.empty and hourly_summary['stops'].sum() > 0:
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mttr_min'], name='MTTR (min)', mode='lines+markers', line=dict(color='red', width=4)))
            fig_mt.add_trace(go.Scatter(x=hourly_summary['hour'], y=hourly_summary['mtbf_min'], name='MTBF (min)', mode='lines+markers', line=dict(color='green', width=4), yaxis='y2'))
            fig_mt.update_layout(title="Hourly MTTR & MTBF Trend", yaxis=dict(title='MTTR (min)'), yaxis2=dict(title='MTBF (min)', overlaying='y', side='right'),
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mt, use_container_width=True)
            with st.expander("View MTTR/MTBF Data"):
                st.dataframe(hourly_summary)
        else:
            st.info("No stops on this day to generate MTTR/MTBF trend.")

        st.subheader("🚨 Stoppage Alerts")
        stoppage_alerts = results_day['processed_df'][results_day['processed_df']['stop_event']].copy()
        if stoppage_alerts.empty:
            st.info("✅ No new stop events were recorded on this day.")
        else:
            stop_event_indices = stoppage_alerts.index.to_series()
            shots_since_last = stop_event_indices.diff().fillna(stop_event_indices.iloc[0] + 1).astype(int) - 1
            stoppage_alerts['Shots Since Last Stop'] = shots_since_last.values
            stoppage_alerts["Duration (min)"] = (stoppage_alerts["ct_diff_sec"] / 60)
            display_table = stoppage_alerts[['shot_time', 'Duration (min)', 'Shots Since Last Stop']].rename(columns={"shot_time": "Event Time"})
            st.dataframe(display_table.style.format({'Duration (min)': '{:.1f}'}), use_container_width=True)

