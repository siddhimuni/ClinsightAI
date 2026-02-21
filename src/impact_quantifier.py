"""
Impact Quantification

Uses Ridge regression to estimate how much each theme affects the rating.
Per-theme confidence is computed via bootstrap resampling.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from src.config import THEMES, THEME_LABELS


def run_regression(df: pd.DataFrame) -> dict:
    """
    Ridge regression: rating ~ theme booleans.
    Returns coefficients, R², and per-theme bootstrap confidence intervals.
    """
    theme_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in df.columns]
    X = df[theme_cols].fillna(0).astype(float)
    y = df["rating"].dropna()
    X = X.loc[y.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    r2 = round(float(np.mean(cv_scores)), 3)

    coefficients = {}
    for col, coef in zip(theme_cols, model.coef_):
        theme = col.replace("theme_", "")
        coefficients[theme] = round(float(coef), 4)

    # Bootstrap confidence: resample 100 times, measure coefficient stability
    n_bootstrap = 100
    bootstrap_coefs = {col.replace("theme_", ""): [] for col in theme_cols}
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
        m = Ridge(alpha=1.0)
        m.fit(X_scaled[idx], y.iloc[idx])
        for col, coef in zip(theme_cols, m.coef_):
            bootstrap_coefs[col.replace("theme_", "")].append(coef)

    # Confidence = proportion of bootstrap samples where sign matches the main estimate
    confidence_per_theme = {}
    for theme, samples in bootstrap_coefs.items():
        main_sign = np.sign(coefficients[theme])
        if main_sign == 0:
            confidence_per_theme[theme] = 0.5
        else:
            agreement = np.mean(np.sign(samples) == main_sign)
            confidence_per_theme[theme] = round(float(agreement), 2)

    return {
        "coefficients": coefficients,
        "r2_score": r2,
        "confidence": confidence_per_theme,
    }


def build_impact_table(df: pd.DataFrame, theme_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ranked impact table per theme.
    Severity score is normalized to 0-1.
    Confidence is per-theme (bootstrap sign stability).
    """
    regression = run_regression(df)
    coefs = regression.get("coefficients", {})
    conf = regression.get("confidence", {})

    rows = []
    for _, row in theme_summary.iterrows():
        theme = row["theme"]
        freq_pct = row["frequency_pct"]
        rating_delta = row.get("rating_delta", 0) or 0

        rating_impact = round(coefs.get(theme, 0), 3)
        raw_severity = (freq_pct / 100) * abs(rating_impact)

        rows.append({
            "theme": theme,
            "theme_label": THEME_LABELS[theme],
            "frequency_pct": freq_pct,
            "frequency_count": row["frequency_count"],
            "rating_impact": rating_impact,
            "rating_delta": rating_delta,
            "raw_severity": raw_severity,
            "confidence": conf.get(theme, 0.5),
            "evidence_samples": row.get("evidence_samples", []),
        })

    impact_df = pd.DataFrame(rows)

    # Normalize severity to 0-1
    max_sev = impact_df["raw_severity"].max()
    if max_sev > 0:
        impact_df["severity_score"] = round(impact_df["raw_severity"] / max_sev, 2)
    else:
        impact_df["severity_score"] = 0.0
    impact_df = impact_df.drop(columns=["raw_severity"])

    impact_df = impact_df.sort_values("severity_score", ascending=False).reset_index(drop=True)
    impact_df["rank"] = range(1, len(impact_df) + 1)

    return impact_df


def get_rating_segments(df: pd.DataFrame) -> dict:
    """Which themes dominate 1-2 star vs 4-5 star reviews."""
    low_df = df[df["rating"] <= 2]
    high_df = df[df["rating"] >= 4]

    low_themes = {}
    high_themes = {}

    for theme in THEMES:
        col = f"theme_{theme}"
        if col not in df.columns:
            continue
        if len(low_df) > 0:
            low_themes[THEME_LABELS[theme]] = round(low_df[col].mean() * 100, 1)
        if len(high_df) > 0:
            high_themes[THEME_LABELS[theme]] = round(high_df[col].mean() * 100, 1)

    return {
        "low_rating_themes": dict(sorted(low_themes.items(), key=lambda x: x[1], reverse=True)),
        "high_rating_themes": dict(sorted(high_themes.items(), key=lambda x: x[1], reverse=True)),
        "n_low": len(low_df),
        "n_high": len(high_df),
    }


if __name__ == "__main__":
    from src.data_loader import load_dataset, preprocess
    from src.theme_extractor import run_theme_extraction, get_theme_summary

    df = load_dataset("data/hospital.csv")
    df = preprocess(df)
    df = run_theme_extraction(df)
    summary = get_theme_summary(df)
    impact = build_impact_table(df, summary)

    print("\nImpact Table:")
    print(impact[["rank", "theme_label", "frequency_pct", "rating_impact", "severity_score", "confidence"]].to_string(index=False))

    segments = get_rating_segments(df)
    print(f"\nLow-rating themes (n={segments['n_low']}):", segments["low_rating_themes"])
    print(f"High-rating themes (n={segments['n_high']}):", segments["high_rating_themes"])
