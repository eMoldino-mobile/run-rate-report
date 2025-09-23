

        # --- SECTION 2: Main CT Graph ---
        st.markdown("---")
        plot_shot_bar_chart(results_day['processed_df'], results_day['lower_limit'], results_day['upper_limit'], results_day['mode_ct'])
        with st.expander("View Shot Data"):
            st.dataframe(results_day['processed_df'])
        
        # --- SECTION 3: Graph Section ---
        st.markdown("---")
        st.header("Hourly Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(
                results_day["run_durations"]["time_bucket"].value_counts().reindex(results_day["bucket_labels"], fill_value=0),
                title="Time Bucket Analysis", labels={"index": "Run Duration (min)", "value": "Occurrences"},
                text_auto=True, color=results_day["bucket_labels"], color_discrete_map=results_day["bucket_color_map"]
            ).update_layout(legend_title_text='Run Duration'), use_container_width=True)
            with st.expander("View Bucket Data"):
                st.dataframe(results_day["run_durations"])
        with col2:
            plot_stability_trend(results_day['hourly_summary'])
            with st.expander("View Stability Data"):
                st.dataframe(results_day['hourly_summary'])

        st.subheader("Hourly Bucket Trend")
        run_durations_day = results_day['run_durations']
        if not run_durations_day.empty:
            processed_day_df = results_day['processed_df']
            run_start_times = processed_day_df[['run_group', 'shot_time']].drop_duplicates(subset=['run_group'], keep='first')
            run_times = run_durations_day.merge(run_start_times, on='run_group', how='left')
            run_times['hour'] = run_times['shot_time'].dt.hour
            bucket_hourly = run_times.groupby(['hour', 'time_bucket'], observed=False).size().reset_index(name='count')
            if not bucket_hourly.empty:
                fig_hourly_bucket = px.bar(
                    bucket_hourly, x='hour', y='count', color='time_bucket', title='Hourly Distribution of Run Durations',
                    barmode='stack', category_orders={"time_bucket": results_day["bucket_labels"]},
                    color_discrete_map=results_day["bucket_color_map"],
                    labels={'hour': 'Hour of Day', 'count': 'Number of Runs', 'time_bucket': 'Run Duration (min)'}
                )
                fig_hourly_bucket.update_xaxes(range=[-0.5, 23.5], tickvals=list(range(24)))
                st.plotly_chart(fig_hourly_bucket, use_container_width=True)
                with st.expander("View Bucket Trend Data"):
                    st.dataframe(bucket_hourly)

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

        st.markdown("---")
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