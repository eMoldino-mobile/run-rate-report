

            # --- Time Bucket Analysis ---
            st.markdown("### Time Bucket Analysis (Table)")
            st.table(results['bucket_counts'].reset_index().rename(columns={"index": "Time Bucket", 0: "Occurrences"}))

            bucket_order = ["<1","1-2","2-3","3-5","5-10","10-20","20-30","30-60","60-120",">120"]
            df_vis = results["df"].copy()
            bucket_counts = (
                df_vis["TIME_BUCKET"]
                .value_counts()
                .reindex(bucket_order)
                .fillna(0)
                .astype(int)
            )
            bucket_df = bucket_counts.reset_index()
            bucket_df.columns = ["Time Bucket", "Occurrences"]

            fig_bucket = px.bar(
                bucket_df[bucket_df["Time Bucket"].notna()],
                x="Occurrences", y="Time Bucket",
                orientation="h", text="Occurrences",
                title="Time Bucket Analysis"
            )
            fig_bucket.update_traces(textposition="outside")
            st.plotly_chart(fig_bucket, use_container_width=True)

            # --- Time Bucket Trend by Hour ---
            src = df_vis.loc[df_vis["STOP_EVENT"] & df_vis["TIME_BUCKET"].notna(), ["HOUR", "TIME_BUCKET"]]
            if src.empty:
                st.info("No stop events with valid TIME_BUCKET for the selected tool/date.")
            else:
                hours = list(range(24))
                grid = pd.MultiIndex.from_product([hours, bucket_order], names=["HOUR","TIME_BUCKET"]).to_frame(index=False)
                counts = src.groupby(["HOUR", "TIME_BUCKET"]).size().reset_index(name="count")
                trend = grid.merge(counts, on=["HOUR","TIME_BUCKET"], how="left").fillna({"count":0})

                fig_tb_trend = px.bar(
                    trend, x="HOUR", y="count", color="TIME_BUCKET",
                    category_orders={"HOUR": hours, "TIME_BUCKET": bucket_order},
                    title="Time Bucket Trend by Hour (0–23)"
                )
                fig_tb_trend.update_layout(barmode="stack")
                st.plotly_chart(fig_tb_trend, use_container_width=True)

            # --- MTTR & MTBF Trend ---
            hourly = results["hourly"].copy()
            all_hours = pd.DataFrame({"HOUR": list(range(24))})
            hourly = all_hours.merge(hourly, on="HOUR", how="left")

            fig_mt = go.Figure()
            fig_mt.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["mttr"],
                mode="lines+markers", name="MTTR (min)",
                line=dict(color="red", width=2), yaxis="y"
            ))
            fig_mt.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["mtbf"],
                mode="lines+markers", name="MTBF (min)",
                line=dict(color="green", width=2, dash="dot"), yaxis="y2"
            ))
            fig_mt.update_layout(
                title="MTTR & MTBF Trend by Hour",
                xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", dtick=1, range=[-0.5, 23.5]),
                yaxis=dict(title="MTTR (min)", tickfont=dict(color="red"), side="left"),
                yaxis2=dict(title="MTBF (min)", tickfont=dict(color="green"), overlaying="y", side="right"),
                margin=dict(l=60, r=60, t=60, b=40),
                legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center")
            )
            st.plotly_chart(fig_mt, use_container_width=True)

            # --- Stability Index ---
            hourly["stability_index"] = np.where(
                (hourly["stops"] == 0) & (hourly["mtbf"].isna()),
                100,
                hourly["stability_index"]
            )
            hourly["stability_change_%"] = hourly["stability_index"].pct_change() * 100

            colors = []
            for v in hourly["stability_index"]:
                if pd.isna(v):
                    colors.append("gray")
                elif v <= 50:
                    colors.append("red")
                elif v <= 70:
                    colors.append("yellow")
                else:
                    colors.append("green")

            fig_stability = go.Figure()
            fig_stability.add_trace(go.Scatter(
                x=hourly["HOUR"], y=hourly["stability_index"],
                mode="lines+markers", name="Stability Index (%)",
                line=dict(color="blue", width=2), marker=dict(color=colors, size=8)
            ))
            st.plotly_chart(fig_stability, use_container_width=True)

            # --- Stoppage Alerts ---
            threshold = results["mode_ct"] * 2
            stoppage_alerts = df_vis[df_vis["CT_diff_sec"] >= threshold].copy()
            st.markdown("### 🚨 Stoppage Alert Reporting (≥ Mode CT × 2)")
            if stoppage_alerts.empty:
                st.info("✅ No stoppage alerts found.")
            else:
                stoppage_alerts["Gap (min)"] = (stoppage_alerts["CT_diff_sec"] / 60).round(2)
                stoppage_alerts["Alert"] = "🔴"
                reasons_list = ["⚙️ Equipment Failure","🔄 Changeover Delay","🧹 Cleaning / Setup","📦 Material Shortage","❓ Other"]
                table = stoppage_alerts.rename(columns={"SHOT TIME": "Event Time","CT_diff_sec":"Gap (sec)","HOUR":"Hour"})
                table = table.assign(
                    Reason=[reasons_list[0]] * len(table),
                    Details="(input soon…)"
                )
                st.data_editor(
                    table[["Event Time","Gap (sec)","Hour","Gap (min)","Alert","Reason","Details"]],
                    use_container_width=True,
                    column_config={
                        "Reason": st.column_config.SelectboxColumn("Reason", options=reasons_list),
                        "Details": st.column_config.TextColumn("Details")
                    },
                    disabled=["Reason","Details"]
                )
