import pandas as pd


def generate_company_insight(company_df):
    """
    Generate a simple AI-style explanation for a company
    using latest dashboard features.
    """

    if company_df.empty:
        return "No data available for this company."

    latest = company_df.tail(1).iloc[0]

    insights = []

    # Momentum
    if pd.notna(latest.get("ret_20d")):
        if latest["ret_20d"] > 0:
            insights.append("Recent 20-day momentum is positive.")
        else:
            insights.append("Recent 20-day momentum is negative.")

    # Risk
    if pd.notna(latest.get("risk_score")):
        if latest["risk_score"] > 1:
            insights.append("Risk score is elevated relative to peers.")
        else:
            insights.append("Risk score appears moderate.")

    # Credibility
    if pd.notna(latest.get("credibility_score")):
        if latest["credibility_score"] > 0:
            insights.append("Management communication appears relatively credible.")
        else:
            insights.append("Communication signals appear weaker than usual.")

    # Misalignment
    if pd.notna(latest.get("misalignment_score")):
        if latest["misalignment_score"] > 0:
            insights.append("Communication tone is more optimistic than recent market performance.")
        else:
            insights.append("Market performance aligns reasonably with communication tone.")

    return " ".join(insights)