"""
Systemic Issue Detection

Classifies themes as SYSTEMIC / MODERATE / ISOLATED based on frequency
and rating impact. Labels each review with a risk cluster.
"""

import pandas as pd
from src.config import THEMES, THEME_LABELS, SYSTEMIC_THRESHOLD, ISOLATED_THRESHOLD


def classify_issues(impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each theme based on frequency AND negative rating impact.
    - SYSTEMIC: frequent (>12%) and hurts ratings
    - MODERATE: moderately frequent (>3%)
    - ISOLATED: rare (<3%)
    """
    impact_df = impact_df.copy()

    def _classify(row):
        freq = row["frequency_pct"] / 100
        has_negative_impact = row["rating_impact"] < 0

        if freq >= SYSTEMIC_THRESHOLD and has_negative_impact:
            return "SYSTEMIC"
        elif freq >= ISOLATED_THRESHOLD:
            return "MODERATE"
        else:
            return "ISOLATED"

    impact_df["issue_class"] = impact_df.apply(_classify, axis=1)

    # Escalation score reuses severity_score (frequency × |rating_impact|)
    impact_df["escalation_score"] = impact_df["severity_score"]

    return impact_df


def cluster_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Label each review as High Risk / Moderate Risk / Positive based on rating."""
    df = df.copy()
    df["risk_cluster"] = df["rating"].apply(_rating_to_risk)
    return df


def _rating_to_risk(rating):
    if pd.isna(rating):
        return "Unknown"
    if rating <= 2:
        return "High Risk"
    if rating <= 3:
        return "Moderate Risk"
    return "Positive Experience"


def detect_trends(df: pd.DataFrame) -> dict:
    """Trend detection requires date data. Our dataset has no dates."""
    return {"available": False, "trends": {}, "rising_themes": []}


def get_systemic_summary(impact_df: pd.DataFrame) -> dict:
    """Summarize systemic vs isolated issues from an already-classified impact table."""
    if "issue_class" not in impact_df.columns:
        impact_df = classify_issues(impact_df)

    systemic = impact_df[impact_df["issue_class"] == "SYSTEMIC"]["theme_label"].tolist()
    moderate = impact_df[impact_df["issue_class"] == "MODERATE"]["theme_label"].tolist()
    isolated = impact_df[impact_df["issue_class"] == "ISOLATED"]["theme_label"].tolist()

    top_escalation = impact_df.nlargest(3, "escalation_score")[
        ["theme_label", "escalation_score", "issue_class"]
    ].to_dict("records")

    return {
        "systemic_issues": systemic,
        "moderate_issues": moderate,
        "isolated_issues": isolated,
        "top_escalation_risks": top_escalation,
        "systemic_count": len(systemic),
    }


if __name__ == "__main__":
    from src.data_loader import load_dataset, preprocess
    from src.theme_extractor import run_theme_extraction, get_theme_summary
    from src.impact_quantifier import build_impact_table

    df = load_dataset("data/hospital.csv")
    df = preprocess(df)
    df = run_theme_extraction(df)
    summary = get_theme_summary(df)
    impact = build_impact_table(df, summary)
    impact = classify_issues(impact)
    df = cluster_reviews(df)

    print("\nClassified Issues:")
    print(impact[["theme_label", "issue_class", "escalation_score"]].to_string(index=False))

    print(f"\nRisk Clusters:")
    print(df["risk_cluster"].value_counts().to_string())

    sys_summary = get_systemic_summary(impact)
    print(f"\nSystemic: {sys_summary['systemic_issues']}")
    print(f"Moderate: {sys_summary['moderate_issues']}")
    print(f"Isolated: {sys_summary['isolated_issues']}")
