"""
Phase 5: Systemic Issue Detection
- Distinguishes systemic (widespread) vs isolated (one-off) issues
- Detects trend patterns over time
- Assigns escalation risk scores
- Clusters reviews by complaint profile
"""

import pandas as pd
import numpy as np
from src.config import THEMES, THEME_LABELS, SYSTEMIC_THRESHOLD, ISOLATED_THRESHOLD


def classify_issues(impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each theme as SYSTEMIC, MODERATE, or ISOLATED
    based on frequency and severity.
    """
    def _classify(row):
        freq = row["frequency_pct"] / 100
        if freq >= SYSTEMIC_THRESHOLD:
            return "SYSTEMIC"
        elif freq >= ISOLATED_THRESHOLD:
            return "MODERATE"
        else:
            return "ISOLATED"

    impact_df = impact_df.copy()
    impact_df["issue_class"] = impact_df.apply(_classify, axis=1)

    # Escalation score: high frequency + high severity + negative impact
    impact_df["escalation_score"] = impact_df.apply(
        lambda r: round(
            (r["frequency_pct"] / 100) * 0.4 +
            r["avg_severity"] * 0.3 +
            (abs(r["rating_impact"]) / 2.0) * 0.3,
            3
        ),
        axis=1
    )

    return impact_df


def detect_trends(df: pd.DataFrame) -> dict:
    """
    If date data is available, detect themes that are growing or declining over time.
    Returns trend direction and slope per theme.
    """
    if "date" not in df.columns or df["date"].isna().all():
        return {"available": False, "trends": {}}

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].dropna().unique())

    if len(months) < 3:
        return {"available": False, "trends": {}}

    trends = {}
    for theme in THEMES:
        col = f"theme_{theme}"
        if col not in df.columns:
            continue

        monthly_freq = []
        for month in months:
            month_df = df[df["month"] == month]
            if len(month_df) == 0:
                continue
            freq = month_df[col].mean() * 100
            monthly_freq.append((str(month), round(freq, 1)))

        if len(monthly_freq) < 3:
            continue

        # Linear regression on frequency over time
        x = np.arange(len(monthly_freq))
        y = np.array([v for _, v in monthly_freq])
        slope = float(np.polyfit(x, y, 1)[0])

        if slope > 0.5:
            direction = "RISING"
        elif slope < -0.5:
            direction = "DECLINING"
        else:
            direction = "STABLE"

        trends[theme] = {
            "direction": direction,
            "slope": round(slope, 3),
            "monthly_data": monthly_freq,
            "label": THEME_LABELS[theme],
        }

    rising = {t: v for t, v in trends.items() if v["direction"] == "RISING"}

    return {
        "available": True,
        "trends": trends,
        "rising_themes": sorted(rising.keys(), key=lambda t: trends[t]["slope"], reverse=True),
        "months": [str(m) for m in months],
    }


def cluster_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster reviews into risk groups using K-Means on theme features.
    Labels clusters as: High Risk / Moderate Risk / Positive Experience
    """
    theme_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in df.columns]
    if not theme_cols or len(df) < 20:
        df["risk_cluster"] = "Unknown"
        return df

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = df[theme_cols].fillna(0).astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(3, len(df) // 5)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df = df.copy()
        df["cluster_id"] = km.fit_predict(X_scaled)

        # Label clusters by average rating
        if "rating" in df.columns:
            cluster_ratings = df.groupby("cluster_id")["rating"].mean()
            sorted_clusters = cluster_ratings.sort_values()

            labels = {}
            cluster_list = sorted_clusters.index.tolist()
            if len(cluster_list) >= 3:
                labels[cluster_list[0]] = "High Risk"
                labels[cluster_list[1]] = "Moderate Risk"
                labels[cluster_list[2]] = "Positive Experience"
            elif len(cluster_list) == 2:
                labels[cluster_list[0]] = "High Risk"
                labels[cluster_list[1]] = "Positive Experience"
            else:
                labels[cluster_list[0]] = "Mixed"

            df["risk_cluster"] = df["cluster_id"].map(labels)
        else:
            df["risk_cluster"] = "Cluster " + df["cluster_id"].astype(str)

    except ImportError:
        df = df.copy()
        # Fallback: label by rating
        if "rating" in df.columns:
            df["risk_cluster"] = pd.cut(
                df["rating"],
                bins=[0, 2, 3, 5],
                labels=["High Risk", "Moderate Risk", "Positive Experience"]
            )
        else:
            df["risk_cluster"] = "Unknown"

    return df


def get_systemic_summary(impact_df: pd.DataFrame) -> dict:
    """Return counts and lists of systemic vs isolated issues."""
    classified = classify_issues(impact_df)

    systemic = classified[classified["issue_class"] == "SYSTEMIC"]["theme_label"].tolist()
    moderate = classified[classified["issue_class"] == "MODERATE"]["theme_label"].tolist()
    isolated = classified[classified["issue_class"] == "ISOLATED"]["theme_label"].tolist()

    top_escalation = classified.nlargest(3, "escalation_score")[
        ["theme_label", "escalation_score", "issue_class"]
    ].to_dict("records")

    return {
        "systemic_issues": systemic,
        "moderate_issues": moderate,
        "isolated_issues": isolated,
        "top_escalation_risks": top_escalation,
        "systemic_count": len(systemic),
        "classified_df": classified,
    }
