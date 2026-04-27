import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
from insight_engine import generate_company_insight
from copilot_engine import answer_query

# App setup
st.set_page_config(
    page_title="AI Finance Copilot Dashboard",
    layout="wide"
)

st.title("AI Finance Copilot Dashboard")

# Paths
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# Existing parquet paths
latest_path = PROCESSED / "latest_company_snapshot.parquet"
risk_path = PROCESSED / "high_risk_companies.parquet"
cred_path = PROCESSED / "high_credibility_companies.parquet"
screen_path = PROCESSED / "signal_screener.parquet"
master_features_path = PROCESSED / "master_panel_features.parquet"

# New CSV paths
latest_csv_path = PROCESSED / "latest_company_snapshot.csv"
metric_summary_csv_path = PROCESSED / "metric_summary_dashboard.csv"
high_risk_csv_path = PROCESSED / "high_risk_companies.csv"

# Prediction CSV paths
baseline_predictions_path = PROCESSED / "baseline_predictions.csv"
feature_importance_path = PROCESSED / "feature_importance.csv"
top_prediction_errors_path = PROCESSED / "top_prediction_errors.csv"


@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# Load existing parquet data
latest_snapshot = load_parquet(latest_path)
high_risk = load_parquet(risk_path)
high_credibility = load_parquet(cred_path)
signal_screener = load_parquet(screen_path)
master_features = load_parquet(master_features_path)

# Load new summary CSVs if available
latest_company_snapshot_csv = load_csv(latest_csv_path) if latest_csv_path.exists() else None
metric_summary_dashboard = load_csv(metric_summary_csv_path) if metric_summary_csv_path.exists() else None
high_risk_companies_csv = load_csv(high_risk_csv_path) if high_risk_csv_path.exists() else None

# Load prediction CSVs if available
baseline_predictions = load_csv(baseline_predictions_path) if baseline_predictions_path.exists() else None
feature_importance = load_csv(feature_importance_path) if feature_importance_path.exists() else None
top_prediction_errors = load_csv(top_prediction_errors_path) if top_prediction_errors_path.exists() else None

# Basic cleaning
for df in [latest_snapshot, high_risk, high_credibility, signal_screener, master_features]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sidebar navigation
page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Company Explorer",
        "Signal Screener",
        "Compare Companies",
        "Summary Tables",
        "Prediction Intelligence",
        "Copilot"
    ]
)

# OVERVIEW PAGE
if page == "Overview":

    st.header("Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Companies", latest_snapshot["ticker"].nunique())
    col2.metric("Average Risk Score", round(latest_snapshot["risk_score"].mean(), 3))
    col3.metric("Average Credibility", round(latest_snapshot["credibility_score"].mean(), 3))
    col4.metric("Avg 20d Return", round(latest_snapshot["ret_20d"].mean(), 4))

    st.divider()

    st.subheader("Risk vs Credibility")

    fig = px.scatter(
        latest_snapshot,
        x="credibility_score",
        y="risk_score",
        hover_name="ticker",
        title="Company Positioning: Risk vs Credibility"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Top 10 High Risk Companies")
    st.dataframe(high_risk.head(10), use_container_width=True)

    st.divider()

    st.subheader("Top 10 High Credibility Companies")
    st.dataframe(high_credibility.head(10), use_container_width=True)


# COMPANY EXPLORER PAGE
elif page == "Company Explorer":

    st.header("Company Explorer")

    tickers = sorted(master_features["ticker"].dropna().unique())
    selected_ticker = st.selectbox("Select a company", tickers)

    company_df = master_features[master_features["ticker"] == selected_ticker].copy()
    company_df = company_df.sort_values("date")

    latest_row = company_df.tail(1)

    if latest_row.empty:
        st.warning("No data found for this company.")
    else:
        latest_row = latest_row.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Latest Close", round(float(latest_row["dlyclose"]), 2))
        col2.metric("20d Return", round(float(latest_row["ret_20d"]), 4))
        col3.metric("Risk Score", round(float(latest_row["risk_score"]), 4))
        col4.metric(
            "Credibility Score",
            round(float(latest_row["credibility_score"]), 4)
            if pd.notna(latest_row["credibility_score"]) else "N/A"
        )

        st.divider()

        st.subheader("Company Insight Summary")

        insight_lines = []

        if pd.notna(latest_row.get("ret_20d")):
            if latest_row["ret_20d"] > 0:
                insight_lines.append(f"- {selected_ticker} has shown positive 20-day momentum.")
            else:
                insight_lines.append(f"- {selected_ticker} has shown negative 20-day momentum.")

        if pd.notna(latest_row.get("risk_score")):
            avg_risk = company_df["risk_score"].mean()
            if latest_row["risk_score"] > avg_risk:
                insight_lines.append("- Current risk score is above the company’s historical average.")
            else:
                insight_lines.append("- Current risk score is below the company’s historical average.")

        if pd.notna(latest_row.get("credibility_score")):
            avg_cred = company_df["credibility_score"].mean()
            if latest_row["credibility_score"] > avg_cred:
                insight_lines.append("- Credibility score is stronger than the company’s historical average.")
            else:
                insight_lines.append("- Credibility score is weaker than the company’s historical average.")

        if pd.notna(latest_row.get("misalignment_score")):
            if latest_row["misalignment_score"] > 0:
                insight_lines.append("- Communication appears more optimistic than recent market performance.")
            else:
                insight_lines.append("- Market performance is not lagging behind communication tone.")

        if not insight_lines:
            insight_lines.append("- Not enough data available for a full insight summary.")

        for line in insight_lines:
            st.write(line)

        st.divider()

        st.subheader("AI Insight")
        insight_text = generate_company_insight(company_df)
        st.info(insight_text)

        st.divider()

        st.subheader("Latest Snapshot")

        snapshot_cols = [
            "ticker",
            "date",
            "dlyclose",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "vol_20d",
            "relative_volume",
            "net_positivity",
            "numeric_transparency",
            "language_complexity",
            "analyst_selectivity_ratio",
            "credibility_score",
            "risk_score",
            "misalignment_score",
        ]

        snapshot_cols = [c for c in snapshot_cols if c in company_df.columns]
        st.dataframe(company_df[snapshot_cols].tail(1), use_container_width=True)

        st.divider()

        st.subheader("Price Trend")

        fig_price = px.line(
            company_df,
            x="date",
            y="dlyclose",
            title=f"{selected_ticker} Closing Price"
        )

        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Transcript Metrics Over Time")

        metric_cols = [
            "net_positivity",
            "numeric_transparency",
            "language_complexity",
            "analyst_selectivity_ratio"
        ]

        available_metrics = [c for c in metric_cols if c in company_df.columns]

        if available_metrics:
            metric_plot_df = company_df[["date"] + available_metrics].copy()

            fig_metrics = px.line(
                metric_plot_df,
                x="date",
                y=available_metrics,
                title=f"{selected_ticker} Transcript Metrics"
            )

            st.plotly_chart(fig_metrics, use_container_width=True)

        st.subheader("Dashboard Scores Over Time")

        score_cols = ["credibility_score", "risk_score", "misalignment_score"]
        available_scores = [c for c in score_cols if c in company_df.columns]

        if available_scores:
            fig_scores = px.line(
                company_df,
                x="date",
                y=available_scores,
                title=f"{selected_ticker} Scores"
            )

            st.plotly_chart(fig_scores, use_container_width=True)


# SIGNAL SCREENER PAGE
elif page == "Signal Screener":

    st.header("Signal Screener")

    screener_df = signal_screener.copy()

    st.subheader("Filters")

    col1, col2 = st.columns(2)

    min_risk = float(screener_df["risk_score"].min())
    max_risk = float(screener_df["risk_score"].max())

    cred_nonnull = screener_df["credibility_score"].dropna()
    min_cred = float(cred_nonnull.min()) if not cred_nonnull.empty else 0.0
    max_cred = float(cred_nonnull.max()) if not cred_nonnull.empty else 1.0

    ret_nonnull = screener_df["ret_20d"].dropna()
    min_ret = float(ret_nonnull.min()) if not ret_nonnull.empty else -1.0
    max_ret = float(ret_nonnull.max()) if not ret_nonnull.empty else 1.0

    with col1:
        risk_range = st.slider(
            "Risk Score Range",
            min_value=min_risk,
            max_value=max_risk,
            value=(min_risk, max_risk),
        )

        cred_range = st.slider(
            "Credibility Score Range",
            min_value=min_cred,
            max_value=max_cred,
            value=(min_cred, max_cred),
        )

    with col2:
        ret_range = st.slider(
            "20-Day Return Range",
            min_value=min_ret,
            max_value=max_ret,
            value=(min_ret, max_ret),
        )

        ticker_search = st.text_input("Search ticker", "").strip().upper()

    filtered = screener_df[
        (screener_df["risk_score"] >= risk_range[0]) &
        (screener_df["risk_score"] <= risk_range[1]) &
        (
            screener_df["credibility_score"].isna() |
            (
                (screener_df["credibility_score"] >= cred_range[0]) &
                (screener_df["credibility_score"] <= cred_range[1])
            )
        ) &
        (screener_df["ret_20d"] >= ret_range[0]) &
        (screener_df["ret_20d"] <= ret_range[1])
    ].copy()

    if ticker_search:
        filtered = filtered[filtered["ticker"].str.contains(ticker_search, na=False)]

    sort_col = st.selectbox(
        "Sort by",
        ["risk_score", "credibility_score", "ret_20d", "relative_volume", "misalignment_score"]
    )

    sort_ascending = st.checkbox("Sort ascending", value=False)

    filtered = filtered.sort_values(sort_col, ascending=sort_ascending).reset_index(drop=True)

    st.write(f"Rows shown: {len(filtered)}")

    m1, m2, m3 = st.columns(3)

    m1.metric("Average Risk", round(filtered["risk_score"].mean(), 4) if len(filtered) else 0)
    m2.metric("Average Credibility", round(filtered["credibility_score"].mean(), 4) if len(filtered) else 0)
    m3.metric("Average 20d Return", round(filtered["ret_20d"].mean(), 4) if len(filtered) else 0)

    st.divider()

    st.subheader("Filtered Company Map")

    plot_df = filtered.dropna(subset=["credibility_score", "risk_score"]).copy()

    if not plot_df.empty:
        fig_screen = px.scatter(
            plot_df,
            x="credibility_score",
            y="risk_score",
            hover_name="ticker",
            size="relative_volume" if "relative_volume" in plot_df.columns else None,
            title="Filtered Companies: Risk vs Credibility"
        )

        st.plotly_chart(fig_screen, use_container_width=True)
    else:
        st.info("No companies available for the chart with the current filters.")

    st.divider()

    st.subheader("Filtered Companies Table")
    st.dataframe(filtered, use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered results as CSV",
        data=csv_data,
        file_name="filtered_signal_screener.csv",
        mime="text/csv"
    )


# COMPARE COMPANIES PAGE
elif page == "Compare Companies":

    st.header("Compare Companies")

    tickers = sorted(master_features["ticker"].dropna().unique())

    col_a, col_b = st.columns(2)

    with col_a:
        ticker_1 = st.selectbox("Select first company", tickers, index=0)

    with col_b:
        default_index = 1 if len(tickers) > 1 else 0
        ticker_2 = st.selectbox("Select second company", tickers, index=default_index)

    df1 = master_features[master_features["ticker"] == ticker_1].copy().sort_values("date")
    df2 = master_features[master_features["ticker"] == ticker_2].copy().sort_values("date")

    latest_1 = df1.tail(1)
    latest_2 = df2.tail(1)

    st.subheader("Latest Snapshot Comparison")

    c1, c2 = st.columns(2)

    compare_cols = [
        "ticker",
        "date",
        "dlyclose",
        "ret_20d",
        "vol_20d",
        "net_positivity",
        "numeric_transparency",
        "language_complexity",
        "credibility_score",
        "risk_score",
        "misalignment_score",
    ]

    with c1:
        st.markdown(f"### {ticker_1}")
        if not latest_1.empty:
            cols_1 = [c for c in compare_cols if c in latest_1.columns]
            st.dataframe(latest_1[cols_1], use_container_width=True)

    with c2:
        st.markdown(f"### {ticker_2}")
        if not latest_2.empty:
            cols_2 = [c for c in compare_cols if c in latest_2.columns]
            st.dataframe(latest_2[cols_2], use_container_width=True)

    st.divider()

    st.subheader("Price Trend Comparison")

    price_df = pd.concat([
        df1[["date", "dlyclose"]].assign(ticker=ticker_1),
        df2[["date", "dlyclose"]].assign(ticker=ticker_2)
    ])

    fig_price_compare = px.line(
        price_df,
        x="date",
        y="dlyclose",
        color="ticker",
        title=f"{ticker_1} vs {ticker_2} Closing Price"
    )

    st.plotly_chart(fig_price_compare, use_container_width=True)

    st.divider()

    st.subheader("Transcript Metrics Comparison")

    metric_choice = st.selectbox(
        "Select transcript metric",
        [
            "net_positivity",
            "numeric_transparency",
            "language_complexity",
            "analyst_selectivity_ratio"
        ]
    )

    metric_df = pd.concat([
        df1[["date", metric_choice]].rename(columns={metric_choice: "value"}).assign(ticker=ticker_1),
        df2[["date", metric_choice]].rename(columns={metric_choice: "value"}).assign(ticker=ticker_2)
    ])

    fig_metric_compare = px.line(
        metric_df,
        x="date",
        y="value",
        color="ticker",
        title=f"{ticker_1} vs {ticker_2}: {metric_choice}"
    )

    st.plotly_chart(fig_metric_compare, use_container_width=True)

    st.divider()

    st.subheader("Score Comparison")

    score_choice = st.selectbox(
        "Select score",
        ["credibility_score", "risk_score", "misalignment_score"]
    )

    score_df = pd.concat([
        df1[["date", score_choice]].rename(columns={score_choice: "value"}).assign(ticker=ticker_1),
        df2[["date", score_choice]].rename(columns={score_choice: "value"}).assign(ticker=ticker_2)
    ])

    fig_score_compare = px.line(
        score_df,
        x="date",
        y="value",
        color="ticker",
        title=f"{ticker_1} vs {ticker_2}: {score_choice}"
    )

    st.plotly_chart(fig_score_compare, use_container_width=True)


# SUMMARY TABLES PAGE
elif page == "Summary Tables":

    st.header("Summary Tables from CRSP + Transcript Merge")

    if latest_company_snapshot_csv is not None:
        st.subheader("Latest Company Snapshot")
        st.dataframe(latest_company_snapshot_csv, use_container_width=True)
    else:
        st.warning("latest_company_snapshot.csv not found.")

    st.divider()

    if metric_summary_dashboard is not None:
        st.subheader("Metric Summary Dashboard")
        st.dataframe(metric_summary_dashboard, use_container_width=True)
    else:
        st.warning("metric_summary_dashboard.csv not found.")

    st.divider()

    if high_risk_companies_csv is not None:
        st.subheader("High-Risk Companies from New Pipeline")
        st.dataframe(high_risk_companies_csv, use_container_width=True)
    else:
        st.warning("high_risk_companies.csv not found.")


# PREDICTION INTELLIGENCE PAGE
elif page == "Prediction Intelligence":

    st.header("Prediction Intelligence")

    if baseline_predictions is None:
        st.warning("baseline_predictions.csv not found. Run modeling_baseline.ipynb first.")
    else:
        st.subheader("Prediction Results")

        mae = baseline_predictions["error"].abs().mean()
        rmse = (baseline_predictions["error"] ** 2).mean() ** 0.5

        c1, c2, c3 = st.columns(3)

        c1.metric("MAE", round(mae, 4))
        c2.metric("RMSE", round(rmse, 4))
        c3.metric("Prediction Rows", len(baseline_predictions))

        st.dataframe(baseline_predictions.head(50), use_container_width=True)

        st.divider()

        st.subheader("Actual vs Predicted")

        plot_df = baseline_predictions.dropna(
            subset=["target_next_q", "prediction_next_q"]
        ).copy()

        fig_pred = px.scatter(
            plot_df,
            x="target_next_q",
            y="prediction_next_q",
            color="Metrics",
            hover_name="Symbol",
            title="Actual vs Predicted Next-Quarter Transcript Value"
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()

    if feature_importance is not None:
        st.subheader("Feature Importance")

        fig_importance = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(fig_importance, use_container_width=True)
        st.dataframe(feature_importance, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found.")

    st.divider()

    if top_prediction_errors is not None:
        st.subheader("Top Prediction Errors")
        st.dataframe(top_prediction_errors, use_container_width=True)
    else:
        st.warning("top_prediction_errors.csv not found.")


# COPILOT PAGE
elif page == "Copilot":

    st.header("Finance Copilot")

    st.write("Ask things like:")
    st.write("- show top risky companies")
    st.write("- show high credibility companies")
    st.write("- compare AAPL and MSFT")
    st.write("- summarize NVDA")

    query = st.text_input("Ask the copilot")

    if query:
        response = answer_query(
            query=query,
            latest_snapshot=latest_snapshot,
            high_risk=high_risk,
            high_credibility=high_credibility,
            signal_screener=signal_screener,
            master_features=master_features
        )

        st.subheader("Copilot Response")
        st.text(response)