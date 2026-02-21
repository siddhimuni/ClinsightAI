"""
Systemic Issue Detection

Classifies themes as SYSTEMIC / MODERATE / ISOLATED using variance-based
consistency analysis on cosine similarity scores, combined with frequency
and rating impact.
"""

import numpy as np
import pandas as pd
from src.config import THEMES, THEME_LABELS


def _compute_consistency(df: pd.DataFrame, theme: str) -> float:
    """
    Measure detection consistency via coefficient of variation (CV) of
    similarity scores among reviews where the theme was detected.
    Low CV = theme appears with uniform strength = systemic pattern.
    Returns a value in [0, 1] where 1 = perfectly consistent.
    """
    sim_col = f"sim_{theme}"
    theme_col = f"theme_{theme}"

    if sim_col not in df.columns or theme_col not in df.columns:
        return 0.5

    detected_sims = df.loc[df[theme_col] == True, sim_col]

    if len(detected_sims) < 3:
        return 0.0

    mean_sim = detected_sims.mean()
    if mean_sim == 0:
        return 0.0

    cv = detected_sims.std() / mean_sim
    return round(max(0.0, 1.0 - cv), 4)


def classify_issues(impact_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each theme using a composite systemic score built from three
    normalized signals:
      - Consistency (0.4): low variance in similarity scores among detected reviews
      - Frequency  (0.3): proportion of reviews mentioning the theme
      - Impact     (0.3): absolute magnitude of rating shift

    SYSTEMIC  = score >= 0.5 AND negative rating impact
    ISOLATED  = score < 0.25
    MODERATE  = everything else
    """
    impact_df = impact_df.copy()

    impact_df["consistency"] = [
        _compute_consistency(df, row["theme"])
        for _, row in impact_df.iterrows()
    ]

    max_freq = impact_df["frequency_pct"].max()
    max_impact = impact_df["rating_impact"].abs().max()
    norm_freq = impact_df["frequency_pct"] / max_freq if max_freq > 0 else 0
    norm_impact = impact_df["rating_impact"].abs() / max_impact if max_impact > 0 else 0

    W_CONSISTENCY, W_FREQUENCY, W_IMPACT = 0.4, 0.3, 0.3
    impact_df["systemic_score"] = (
        W_CONSISTENCY * impact_df["consistency"]
        + W_FREQUENCY * norm_freq
        + W_IMPACT * norm_impact
    ).round(3)

    def _classify(row):
        if row["systemic_score"] >= 0.5 and row["rating_impact"] < 0:
            return "SYSTEMIC"
        if row["systemic_score"] < 0.25:
            return "ISOLATED"
        return "MODERATE"

    impact_df["issue_class"] = impact_df.apply(_classify, axis=1)
    impact_df["escalation_score"] = impact_df["systemic_score"]

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
        raise ValueError("impact_df must be classified first — call classify_issues(impact_df, df)")

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
    impact = classify_issues(impact, df)
    df = cluster_reviews(df)

    print("\nClassified Issues:")
    print(impact[["theme_label", "issue_class", "escalation_score"]].to_string(index=False))

    print(f"\nRisk Clusters:")
    print(df["risk_cluster"].value_counts().to_string())

    sys_summary = get_systemic_summary(impact)
    print(f"\nSystemic: {sys_summary['systemic_issues']}")
    print(f"Moderate: {sys_summary['moderate_issues']}")
    print(f"Isolated: {sys_summary['isolated_issues']}")
