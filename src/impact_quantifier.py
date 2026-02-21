"""
Impact Quantification

Uses Ridge regression to estimate how much each theme affects the rating.
Outputs a ranked impact table used by the financial simulator and roadmap.
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
    Returns each theme's coefficient (how much it shifts the rating).
    """
    theme_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in df.columns]
    X = df[theme_cols].fillna(0).astype(float)
    y = df["rating"].dropna()
    X = X.loc[y.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    r2 = round(float(np.mean(cv_score)), 3)

    coefficients = {}
    for col, coef in zip(theme_cols, model.coef_):
        theme = col.replace("theme_", "")
        coefficients[theme] = round(float(coef), 4)

    return {"coefficients": coefficients, "r2_score": r2}


def build_impact_table(df: pd.DataFrame, theme_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ranked impact table per theme:
    - rating_impact: regression coefficient (how much the theme shifts rating)
    - severity_score: frequency × |rating_impact| (how widespread AND impactful)
    """
    regression = run_regression(df)
    coefs = regression.get("coefficients", {})

    rows = []
    for _, row in theme_summary.iterrows():
        theme = row["theme"]
        freq_pct = row["frequency_pct"]
        rating_delta = row.get("rating_delta", 0) or 0
        coef = coefs.get(theme, 0)

        rating_impact = round(coef, 3)
        severity_score = round((freq_pct / 100) * abs(rating_impact), 3)

        rows.append({
            "theme": theme,
            "theme_label": THEME_LABELS[theme],
            "frequency_pct": freq_pct,
            "frequency_count": row["frequency_count"],
            "rating_impact": rating_impact,
            "rating_delta": rating_delta,
            "severity_score": severity_score,
            "regression_coef": coef,
            "confidence": regression["r2_score"],
            "evidence_samples": row.get("evidence_samples", []),
        })

    impact_df = pd.DataFrame(rows)
    impact_df = impact_df.sort_values("severity_score", ascending=False).reset_index(drop=True)
    impact_df["rank"] = range(1, len(impact_df) + 1)

    return impact_df


def get_rating_segments(df: pd.DataFrame) -> dict:
    """Show which themes dominate low-rating vs high-rating reviews."""
    low_df = df[df["rating"] <= 2]
    high_df = df[df["rating"] >= 4]

    low_themes = {}
    high_themes = {}

    for theme in THEMES:
        col = f"theme_{theme}"
        if col not in df.columns:
            continue
        if len(low_df) > 0:
            low_themes[theme] = round(low_df[col].mean() * 100, 1)
        if len(high_df) > 0:
            high_themes[theme] = round(high_df[col].mean() * 100, 1)

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
    print(impact[["rank", "theme_label", "frequency_pct", "rating_impact", "severity_score"]].to_string(index=False))

    segments = get_rating_segments(df)
    print(f"\nLow-rating themes (n={segments['n_low']}):", segments["low_rating_themes"])
    print(f"High-rating themes (n={segments['n_high']}):", segments["high_rating_themes"])
