import streamlit as st
import pandas as pd
import numpy as np
import warnings
import streamlit.components.v1 as components
from datetime import datetime

# Import all logic functions from the new utils file
import run_rate_utils as rr_utils

# ==============================================================================
# --- 1. UI RENDERING FUNCTIONS ---
# (These functions render the UI but call rr_utils for logic)
# ==============================================================================

def render_risk_tower(df_all_tools):
    """Renders the Risk Tower tab."""
    st.title("Run Rate Risk Tower")
    st.info("This tower analyzes performance over the last 4 weeks, identifying tools that require attention. Tools with the lowest scores are at the highest risk.")
    
    with st.expander("ℹ️ How the Risk Tower Works"):
        st.markdown("""
        The Risk Tower evaluates each tool based on its performance over its own most recent 4-week period of operation. Here’s how the metrics are calculated:
        - **Analysis Period**: Shows the exact 4-week date range used for each tool's analysis, based on its latest available data.
        - **Risk Score**: A performance indicator from 0-100.
        - **Primary Risk Factor**: Identifies the main issue affecting performance.
        - **Color Coding**: Rows are colored based on the Risk Score (Red, Orange, Green).
        """, unsafe_allow_html=True)

    # Call the logic function from the utils file
    risk_df = rr_utils.calculate_risk_scores(df_all_tools)

    if risk_df.empty:
        st.warning("Not enough data across multiple tools in the last 4 weeks to generate a risk tower.")
        return

    def style_risk(row):
        score = row['Risk Score']
        if score > 70: color = rr_utils.PASTEL_COLORS['green']
        elif score > 50: color = rr_utils.PASTEL_COLORS['orange']
        else: color = rr_utils.PASTEL_COLORS['red']
        return [f'background-color: {color}' for _ in row]
    
    cols_order = ['Tool ID', 'Analysis Period', 'Risk Score', 'Primary Risk Factor', 'Weekly Stability', 'Details']
    display_df = risk_df[[col for col in cols_order if col in risk_df.columns]]

    st.dataframe(display_df.style.apply(style_risk, axis=1).format({'Risk Score': '{:.0f}'}), use_container_width=True, hide_index=True)


def render_dashboard(df_tool, tool_id_selection):
    """Renders the main Run Rate Dashboard tab."""
    st.sidebar.title("Dashboard Controls ⚙️")

    with st.sidebar.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        ### Run Rate Analysis
        - **Efficiency (%)**: Normal Shots ÷ Total Shots
        - **MTTR (min)**: Average downtime per stop.
        - **MTBF (min)**: Average uptime between stops.
        - **Stability Index (%)**: Uptime ÷ (Uptime + Downtime)
        """)
        
    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        options=["Daily", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"],
        key="rr_analysis_level" # Added key
    )

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.25, 0.01, key="rr_tolerance", help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, key="rr_downtime_gap", help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, key="rr_run_interval", help="Defines the max hours between shots before a new Production Run is identified.")
    
    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours):
        # Call the logic function from the utils file
        base_calc = rr_utils.RunRateCalculator(df, 0.01, 2.0)
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
            df_processed['date'] = df_processed['shot_time'].dt.date
            df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600)
            df_processed['run_id'] = is_new_run.cumsum()
        return df_processed

    df_processed = get_processed_data(df_tool, run_interval_hours)
    
    min_shots_filter = 1 
    if 'by Run' in analysis_level:
        st.sidebar.markdown("---")
        
        enable_run_filter = st.sidebar.toggle("Filter Small Production Runs", value=False, key="rr_enable_run_filter", help="Turn this on to show the slider for filtering out runs with few shots.")
        
        if enable_run_filter and not df_processed.empty:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if not run_shot_counts.empty else 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.sidebar.slider(
                    "Remove Runs with Fewer Than X Shots",
                    min_value=1,
                    max_value=max_shots,
                    value=default_value,
                    step=1,
                    key="rr_min_shots_filter",
                    help="Filters out smaller production runs to focus on more significant ones."
                )
        elif not enable_run_filter:
            min_shots_filter = 1
    
    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True, key="rr_detailed_view")


    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")

    mode = 'by_run' if 'by Run' in analysis_level else 'aggregate'
    df_view = pd.DataFrame()

    # --- Date/Period Selection ---
    if "Daily" in analysis_level:
        st.header(f"Daily Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates:
            st.warning("No data available for any date."); st.stop()
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'), key="rr_daily_select")
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"

    elif "Weekly" in analysis_level:
        st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_weeks = sorted(df_processed["week"].unique())
        if not available_weeks:
            st.warning("No data available for any week."); st.stop()
        year = df_processed['shot_time'].iloc[0].year
        selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1, key="rr_weekly_select")
        df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"

    elif "Monthly" in analysis_level:
        st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_months = sorted(df_processed["month"].unique())
        if not available_months:
            st.warning("No data available for any month."); st.stop()
        selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'), key="rr_monthly_select")
        df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"

    elif "Custom Period" in analysis_level:
        st.header(f"Custom Period Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        min_date = df_processed['date'].min()
        max_date = df_processed['date'].max()
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date, key="rr_custom_start")
        end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date, key="rr_custom_end")
        if start_date and end_date:
            mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
            df_view = df_processed[mask]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    # --- Run Filtering and Labeling ---
    if not df_view.empty:
        df_view = df_view.copy()
        if 'run_id' in df_view.columns and 'by Run' in analysis_level:
            df_view['run_id_local'] = df_view.groupby('run_id').ngroup()
            unique_run_ids = df_view.sort_values('shot_time')['run_id_local'].unique()
            run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids)}
            df_view['run_label'] = df_view['run_id_local'].map(run_label_map)

    if 'by Run' in analysis_level and not df_view.empty:
        runs_before_filter = df_view['run_label'].nunique()
        run_shot_counts_in_view = df_view.groupby('run_label')['run_label'].transform('count')
        df_view = df_view[run_shot_counts_in_view >= min_shots_filter]
        runs_after_filter = df_view['run_label'].nunique()

        if runs_before_filter > 0:
            st.sidebar.metric(
                label="Runs Displayed",
                value=f"{runs_after_filter} / {runs_before_filter}",
                delta=f"-{runs_before_filter - runs_after_filter} filtered",
                delta_color="off"
            )

    if df_view.empty:
        st.warning(f"No data for the selected period (or all runs were filtered out).")
    else:
        # --- Main Calculation for Selected Period ---
        results = {}
        summary_metrics = {}

        if 'by Run' in analysis_level:
            # Call logic function
            run_summary_df_for_totals = rr_utils.calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            
            if not run_summary_df_for_totals.empty:
                total_runtime_sec = run_summary_df_for_totals['total_runtime_sec'].sum()
                production_time_sec = run_summary_df_for_totals['production_time_sec'].sum()
                downtime_sec = run_summary_df_for_totals['downtime_sec'].sum()
                total_shots = run_summary_df_for_totals['total_shots'].sum()
                normal_shots = run_summary_df_for_totals['normal_shots'].sum()
                stop_events = run_summary_df_for_totals['stops'].sum()

                mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
                mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
                stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0
                efficiency = (normal_shots / total_shots) if total_shots > 0 else 0

                summary_metrics = {
                    'total_runtime_sec': total_runtime_sec, 'production_time_sec': production_time_sec,
                    'downtime_sec': downtime_sec, 'total_shots': total_shots,
                    'normal_shots': normal_shots, 'stop_events': stop_events,
                    'mttr_min': mttr_min, 'mtbf_min': mtbf_min,
                    'stability_index': stability_index, 'efficiency': efficiency,
                }
                sub_header = sub_header.replace("Summary for", "Summary for (Combined Runs)")

            # Call logic function
            calc = rr_utils.RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
            results = calc.results
            summary_metrics.update({
                'min_lower_limit': results.get('min_lower_limit', 0), 'max_lower_limit': results.get('max_lower_limit', 0),
                'min_mode_ct': results.get('min_mode_ct', 0), 'max_mode_ct': results.get('max_mode_ct', 0),
                'min_upper_limit': results.get('min_upper_limit', 0), 'max_upper_limit': results.get('max_upper_limit', 0),
            })
        else:
            # Call logic function
            calc = rr_utils.RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
            results = calc.results
            summary_metrics = results

        # --- Header & Download Button ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(sub_header)
        with col2:
            st.download_button(
                label="📥 Export Run-Based Report",
                # Call logic function
                data=rr_utils.prepare_and_generate_run_based_excel(
                    df_view.copy(), tolerance, downtime_gap_tolerance,
                    run_interval_hours, tool_id_selection
                ),
                file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        # --- Create Trend Summary DataFrame ---
        trend_summary_df = None
        trend_level = "" 

        if "Daily" in analysis_level:
            trend_level = "Hourly"
            trend_summary_df = results.get('hourly_summary', pd.DataFrame())
        
        elif "by Run" in analysis_level: 
            trend_level = "Run"
            # Call logic function
            trend_summary_df = rr_utils.calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            if trend_summary_df is not None and not trend_summary_df.empty:
                trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
        
        elif "Weekly" in analysis_level:
            trend_level = "Daily"
            # Call logic function
            trend_summary_df = rr_utils.calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode)
        
        elif "Monthly" in analysis_level:
            trend_level = "Weekly"
            # Call logic function
            trend_summary_df = rr_utils.calculate_weekly_summaries_for_month(df_view, tolerance, downtime_gap_tolerance, mode)
        
        elif "Custom Period" in analysis_level:
             trend_level = "Daily"
             # Call logic function
             trend_summary_df = rr_utils.calculate_daily_summaries_for_week(df_view, tolerance, downtime_gap_tolerance, mode)

        # --- KPI Metrics Display ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0); prod_t = summary_metrics.get('production_time_sec', 0); down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            
            mttr_val_min = summary_metrics.get('mttr_min', 0)
            mtbf_val_min = summary_metrics.get('mtbf_min', 0)
            
            # Call logic function
            mttr_display = rr_utils.format_minutes_to_dhm(mttr_val_min)
            mtbf_display = rr_utils.format_minutes_to_dhm(mtbf_val_min)

            with col1: st.metric("Run Rate MTTR", mttr_display)
            with col2: st.metric("Run Rate MTBF", mtbf_display)
            
            # Call logic function
            with col3: st.metric("Total Run Duration", rr_utils.format_duration(total_d))
            with col4:
                st.metric("Production Time", f"{rr_utils.format_duration(prod_t)}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{rr_utils.format_duration(down_t)}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns(2)
            # Call logic function
            c1.plotly_chart(rr_utils.create_gauge(summary_metrics.get('efficiency', 0) * 100, "Run Rate Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': rr_utils.PASTEL_COLORS['red']}, {'range': [50, 70], 'color': rr_utils.PASTEL_COLORS['orange']},{'range': [70, 100], 'color': rr_utils.PASTEL_COLORS['green']}]
            # Call logic function
            c2.plotly_chart(rr_utils.create_gauge(summary_metrics.get('stability_index', 0), "Run Rate Stability Index (%)", steps=steps), use_container_width=True)
        
        # ... (rest of the KPI boxes) ...
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0)
            s_s = t_s - n_s
            n_p = (n_s / t_s * 100) if t_s > 0 else 0
            s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{summary_metrics.get('stop_events', 0)}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            if mode == 'by_run':
                min_ll = summary_metrics.get('min_lower_limit', 0); max_ll = summary_metrics.get('max_lower_limit', 0)
                c1.metric("Lower Limit Range (sec)", f"{min_ll:.2f} – {max_ll:.2f}")
                with c2:
                    min_mc = summary_metrics.get('min_mode_ct', 0); max_mc = summary_metrics.get('max_mode_ct', 0)
                    with st.container(border=True): st.metric("Mode Cycle Time Range (sec)", f"{min_mc:.2f} – {max_mc:.2f}")
                min_ul = summary_metrics.get('min_upper_limit', 0); max_ul = summary_metrics.get('max_upper_limit', 0)
                c3.metric("Upper Limit Range (sec)", f"{min_ul:.2f} – {max_ul:.2f}")
            else:
                mode_val = summary_metrics.get('mode_ct', 0)
                mode_disp = f"{mode_val:.2f}" if isinstance(mode_val, (int,float)) else mode_val
                c1.metric("Lower Limit (sec)", f"{summary_metrics.get('lower_limit', 0):.2f}")
                with c2:
                    with st.container(border=True): st.metric("Mode Cycle Time (sec)", mode_disp)
                c3.metric("Upper Limit (sec)", f"{summary_metrics.get('upper_limit', 0):.2f}")

        # --- Detailed Analysis Expander ---
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame()
                if trend_summary_df is not None and not trend_summary_df.empty:
                    analysis_df = trend_summary_df.copy()
                    rename_map = {}
                    if 'hour' in analysis_df.columns: rename_map = {'hour': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'date' in analysis_df.columns: rename_map = {'date': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'week' in analysis_df.columns: rename_map = {'week': 'period', 'stability_index': 'stability', 'stops': 'stops', 'mttr_min': 'mttr'}
                    elif 'RUN ID' in analysis_df.columns: rename_map = {'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}
                    analysis_df.rename(columns=rename_map, inplace=True)
                
                # Call logic function
                insights = rr_utils.generate_detailed_analysis(analysis_df, summary_metrics.get('stability_index', 0), summary_metrics.get('mttr_min', 0), summary_metrics.get('mtbf_min', 0), analysis_level)
                
                if "error" in insights: 
                    st.error(insights["error"])
                else:
                    components.v1.html(f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights['overall']}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights['predictive']}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights['best_worst']}</p> {'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> ' + insights['patterns'] + '</p>' if insights['patterns'] else ''}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights['recommendation']}</p></div>""", height=400, scrolling=True)

        # --- Breakdown Table Expander ---
        if analysis_level in ["Weekly", "Monthly", "Custom Period"] and "by Run" not in analysis_level:
            with st.expander(f"View {trend_level.title()} Breakdown Table", expanded=False):
                if trend_summary_df is not None and not trend_summary_df.empty:
                    d_df = trend_summary_df.copy()
                    if 'date' in d_df.columns:
                        d_df['date'] = pd.to_datetime(d_df['date']).dt.strftime('%A, %b %d')
                    # Call logic function
                    d_df = rr_utils.get_renamed_summary_df(d_df)
                    st.dataframe(d_df.style.format({'Stability Index (%)': '{:.1f}', 'MTTR (min)': '{:.1f}', 'MTBF (min)': '{:.1f}'}), use_container_width=True)
        
        elif "by Run" in analysis_level:
            # Call logic function
            run_summary_df = rr_utils.calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            with st.expander("View Run Breakdown Table", expanded=False):
                if run_summary_df is not None and not run_summary_df.empty:
                    # ... (This section is mostly UI formatting, so it stays) ...
                    d_df = run_summary_df.copy()
                    d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
                    d_df["Total shots"] = d_df['total_shots'].apply(lambda x: f"{x:,}")
                    d_df["Normal Shots"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Stop Events"] = d_df.apply(lambda r: f"{r['stops']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
                    d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(rr_utils.format_duration)
                    d_df["Production Time (d/h/m)"] = d_df.apply(lambda r: f"{rr_utils.format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (d/h/m)"] = d_df.apply(lambda r: f"{rr_utils.format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df.rename(columns={
                        'run_label':'RUN ID', 'mode_ct':'Mode CT (for the run)',
                        'lower_limit':'Lower limit CT (sec)', 'upper_limit':'Upper Limit CT (sec)',
                        'mttr_min':'MTTR (min)', 'mtbf_min':'MTBF (min)',
                        'stability_index':'Stability (%)', 'stops':'STOPS'
                        }, inplace=True)
                    final_cols = ['RUN ID','Period (date/time from to)','Total shots','Normal Shots','Stop Events','Mode CT (for the run)','Lower limit CT (sec)','Upper Limit CT (sec)','Total Run duration (d/h/m)','Production Time (d/h/m)','Downtime (d/h/m)','MTTR (min)','MTBF (min)','Stability (%)']
                    final_cols = [col for col in final_cols if col in d_df.columns]
                    st.dataframe(d_df[final_cols].style.format({'Mode CT (for the run)':'{:.2f}','Lower limit CT (sec)':'{:.2f}','Upper Limit CT (sec)':'{:.2f}','MTTR (min)':'{:.1f}','MTBF (min)':'{:.1f}','Stability (%)':'{:.1f}'}), use_container_width=True)
        
        # --- Main Shot Bar Chart ---
        time_agg = 'hourly' if "Daily" in analysis_level else 'daily' if 'Weekly' in analysis_level else 'weekly'
        # Call logic function
        rr_utils.plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=time_agg)
        
        # ... (rest of the UI logic) ...
        
        # --- Detailed Trend Charts (Dynamic by Analysis Level) ---
        if "Daily" in analysis_level:
            # ... (UI logic) ...
            # Call logic function for plot
            rr_utils.plot_trend_chart(trend_summary_df, 'hour', 'stability_index', "Hourly Stability Trend", "Hour of Day", "Stability (%)", is_stability=True)
            # ...
            # Call logic function for plot
            rr_utils.plot_mttr_mtbf_chart(
                    df=hourly_summary, x_col='hour', mttr_col='mttr_min',
                    mtbf_col='mtbf_min', shots_col='total_shots',
                    title="Hourly MTTR & MTBF Trend"
                )
            # ...
        
        elif analysis_level in ["Weekly", "Monthly", "Custom Period"]:
            # ... (UI logic) ...
            # Call logic function for plot
            rr_utils.plot_trend_chart(summary_df, x_col, 'stability_index', f"{trend_level.title()} Stability Trend", trend_level, "Stability (%)", is_stability=True)
            # ...
            # Call logic function for plot
            rr_utils.plot_mttr_mtbf_chart(
                    df=summary_df, x_col=x_col, mttr_col='mttr_min',
                    mtbf_col='mtbf_min', shots_col='total_shots',
                    title=f"{trend_level.title()} MTTR, MTBF & Shot Count Trend"
                )
            # ...

        elif "by Run" in analysis_level:
            # ... (UI logic) ...
            # Call logic function for plot
            rr_utils.plot_trend_chart(run_summary_df, 'RUN ID', 'STABILITY %', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
            # ...
            # Call logic function for plot
            rr_utils.plot_mttr_mtbf_chart(
                    df=run_summary_df, x_col='RUN ID', mttr_col='MTTR (min)',
                    mtbf_col='MTBF (min)', shots_col='Total Shots',
                    title="MTTR, MTBF & Shot Count per Run"
                )
            # ...


# ==============================================================================
# --- 8. MAIN APP LOGIC (FILE UPLOAD & TAB RENDERING) ---
# --- This is the main function that will be called by the combined app ---
# ==============================================================================

def run_run_rate_ui():
    """
    This function contains the main UI logic for the Run Rate app.
    It will be imported and called by combined_report.py.
    """
    
    st.sidebar.title("File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more Run Rate Excel files", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True,
        key="rr_file_uploader" # Add a unique key
    )

    if not uploaded_files:
        st.info("👈 Upload one or more Excel files to see the Run Rate reports.")
        st.stop()

    # Call logic function from utils
    df_all_tools = rr_utils.load_all_data(uploaded_files)

    id_col = "tool_id"
    if id_col not in df_all_tools.columns:
        st.error(f"None of the uploaded files contain a 'TOOLING ID' or 'EQUIPMENT CODE' column.")
        st.stop()

    df_all_tools.dropna(subset=[id_col], inplace=True)
    df_all_tools[id_col] = df_all_tools[id_col].astype(str)

    # --- Sidebar Tool ID Selection ---
    tool_ids = ["All Tools (Risk Tower)"] + sorted(df_all_tools[id_col].unique().tolist())
    dashboard_tool_id_selection = st.sidebar.selectbox(
        "Select Tool ID for Dashboard Analysis", 
        tool_ids,
        key="rr_tool_select" # Add a unique key
    )

    if dashboard_tool_id_selection == "All Tools (Risk Tower)":
        if len(tool_ids) > 1:
            first_tool = tool_ids[1]
            df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool]
            tool_id_for_dashboard_display = first_tool
        else:
            df_for_dashboard = pd.DataFrame()
            tool_id_for_dashboard_display = "No Tool Selected"
    else:
        df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection]
        tool_id_for_dashboard_display = dashboard_tool_id_selection

    # --- Tab Rendering ---
    tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])

    with tab1:
        render_risk_tower(df_all_tools)

    with tab2:
        if not df_for_dashboard.empty:
            render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
        else:
            st.info("Select a specific Tool ID from the sidebar to view its dashboard.")

# --- Add this block at the very end ---
# This allows the file to be run standalone for testing
if __name__ == "__main__":
    # --- Page Config (Copied from combined_report.py) ---
    st.set_page_config(
        page_title="Run Rate Analysis (Standalone)",
        layout="wide"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # This is the main function we defined above
    run_run_rate_ui()