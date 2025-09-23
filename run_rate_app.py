@ -209,7 +209,6 @@ def plot_stability_trend(df, time_col, stability_col, title="Stability Index Tre

@st.cache_data
def export_to_excel(results: dict, tolerance: float):
    """Creates a multi-sheet Excel report with all key analyses and parameters."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_kpis = {
@ -364,9 +363,32 @@ if page == "📊 Daily Deep-Dive":
                    st.dataframe(hourly_display.style.format({'MTTR (min)': '{:.2f}', 'MTBF (min)': '{:.2f}', 'Stability Index (%)': '{:.2f}%'}), use_container_width=True)
                
                plot_stability_trend(hourly_df, 'hour', 'stability_index')
                display_stability_index_explanation() 
                display_stability_index_explanation()
                
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
                
                display_table = stoppage_alerts[['shot_time', 'Duration (min)', 'Shots Since Last Stop']].rename(columns={
                    "shot_time": "Event Time",
                })
                st.dataframe(display_table.style.format({
                    'Duration (min)': '{:.1f}'
                }), use_container_width=True)


elif page == "🗓️ Weekly Trends":
    # This page logic remains the same
    st.header("Weekly Trend Analysis")
    df_processed = calculator_full.results["processed_df"]
    df_processed['week_start'] = df_processed['shot_time'].dt.to_period('W-MON').apply(lambda r: r.start_time).dt.date
@ -423,7 +445,47 @@ elif page == "🗓️ Weekly Trends":
                bucket_weekly_display['week_start'] = pd.to_datetime(bucket_weekly_display['week_start']).dt.strftime('%d %b %Y')
                st.dataframe(bucket_weekly_display.rename(columns={'week_start': 'Week Starting', 'time_bucket': 'Run Duration (min)', 'count': 'Occurrences'}), use_container_width=True)


elif page == "📂 View Processed Data":
    st.header("Processed Cycle Data")
    results = calculator_full.results
    st.
    st.subheader("Calculation Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode CT (sec)", f"{results.get('mode_ct', 0):.2f}")
    col2.metric("Lower Limit (sec)", f"{results.get('lower_limit', 0):.2f}", help="Cycles below this are flagged as stops.")
    col3.metric("Upper Limit (sec)", f"{results.get('upper_limit', 0):.2f}", help="Cycles above this are flagged as stops.")

    st.markdown("---")
    st.subheader("Shot-by-Shot Data")
    
    df_display = results["processed_df"].copy()
    
    # --- UPDATED ICON LOGIC ---
    df_display["Stop Cycle"] = np.where(df_display["stop_flag"] == 1, "⚫️", "") # Grey for stop, blank for normal
    df_display["Stop Event Start"] = np.where(df_display["stop_event"], "🛑", "") # Red for event start
    
    # --- NEW INFORMATIVE COLUMNS ---
    df_display['Downtime (sec)'] = np.where(df_display['stop_event'], df_display['ct_diff_sec'], np.nan)
    df_display['Deviation from Mode (sec)'] = df_display['ct_diff_sec'] - results['mode_ct']
    
    display_cols = ["shot_time", "ACTUAL CT", "ct_diff_sec", "Deviation from Mode (sec)",
                    "Stop Cycle", "Stop Event Start", "Downtime (sec)", "run_group"]
                    
    display_subset = df_display[display_cols].rename(columns={
        "shot_time": "Shot Time", "ACTUAL CT": "Actual CT (sec)", 
        "ct_diff_sec": "Time Since Last Shot (sec)", "run_group": "Run Group ID"
    })
    
    st.dataframe(display_subset.style.format({
        "Actual CT (sec)": "{:.1f}", 
        "Time Since Last Shot (sec)": "{:.2f}",
        "Deviation from Mode (sec)": "{:+.2f}",
        "Downtime (sec)": "{:.1f}"
    }), use_container_width=True)
    
    excel_data = export_to_excel(calculator_full.results, calculator_full.tolerance)
    st.download_button(
        label="📥 Download Full Excel Report", data=excel_data,
        file_name=f"{tool_id}_full_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )