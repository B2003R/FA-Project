import pandas as pd


def answer_query(query, latest_snapshot, high_risk, high_credibility, signal_screener, master_features):
    q = query.lower().strip()

    if not q:
        return "Please type a question."

    # ------------------------------------------
    # Top risky companies
    # ------------------------------------------
    if "top risky" in q or "high risk" in q:
        top = high_risk.head(5)[["ticker", "risk_score", "ret_20d", "credibility_score"]]
        return "Top high-risk companies:\n\n" + top.to_string(index=False)

    # ------------------------------------------
    # Top credibility companies
    # ------------------------------------------
    if "high credibility" in q or "top credibility" in q:
        top = high_credibility.head(5)[["ticker", "credibility_score", "ret_20d", "risk_score"]]
        return "Top high-credibility companies:\n\n" + top.to_string(index=False)

    # ------------------------------------------
    # Compare two companies
    # ------------------------------------------
    if "compare" in q:
        tickers = [t for t in latest_snapshot["ticker"].dropna().unique() if t.lower() in q]
        if len(tickers) >= 2:
            t1, t2 = tickers[0], tickers[1]

            row1 = latest_snapshot[latest_snapshot["ticker"] == t1].tail(1)
            row2 = latest_snapshot[latest_snapshot["ticker"] == t2].tail(1)

            if not row1.empty and not row2.empty:
                r1 = row1.iloc[0]
                r2 = row2.iloc[0]

                return (
                    f"Comparison: {t1} vs {t2}\n\n"
                    f"{t1}: risk={r1['risk_score']:.4f}, credibility={r1['credibility_score']:.4f}, ret_20d={r1['ret_20d']:.4f}\n"
                    f"{t2}: risk={r2['risk_score']:.4f}, credibility={r2['credibility_score']:.4f}, ret_20d={r2['ret_20d']:.4f}"
                )

        return "I could not identify two valid tickers to compare."

    # ------------------------------------------
    # Summarize one company
    # ------------------------------------------
    possible_tickers = [t for t in latest_snapshot["ticker"].dropna().unique() if t.lower() in q]

    if possible_tickers:
        ticker = possible_tickers[0]
        company_df = master_features[master_features["ticker"] == ticker].sort_values("date")

        if company_df.empty:
            return f"No data found for {ticker}."

        latest = company_df.tail(1).iloc[0]

        lines = [f"Summary for {ticker}:"]
        lines.append(f"- Latest close: {latest['dlyclose']:.2f}")

        if pd.notna(latest.get("ret_20d")):
            lines.append(f"- 20-day return: {latest['ret_20d']:.4f}")

        if pd.notna(latest.get("risk_score")):
            lines.append(f"- Risk score: {latest['risk_score']:.4f}")

        if pd.notna(latest.get("credibility_score")):
            lines.append(f"- Credibility score: {latest['credibility_score']:.4f}")

        if pd.notna(latest.get("misalignment_score")):
            lines.append(f"- Misalignment score: {latest['misalignment_score']:.4f}")

        if pd.notna(latest.get("net_positivity")):
            lines.append(f"- Net positivity: {latest['net_positivity']:.4f}")

        if pd.notna(latest.get("numeric_transparency")):
            lines.append(f"- Numeric transparency: {latest['numeric_transparency']:.4f}")

        return "\n".join(lines)

    # ------------------------------------------
    # Fallback
    # ------------------------------------------
    return (
        "I can help with:\n"
        "- show top risky companies\n"
        "- show high credibility companies\n"
        "- compare AAPL and MSFT\n"
        "- summarize NVDA"
    )