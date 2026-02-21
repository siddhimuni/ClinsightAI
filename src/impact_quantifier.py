"""
Phase 4: Impact Quantification
- Ridge regression: rating impact per theme
- Random Forest + SHAP: feature importance and explainability
- Outputs ranked theme impact table
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.config import THEMES, THEME_LABELS


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (theme binary features) and y (ratings) for modeling.
    Also includes interaction features and review metadata.
    """
    feature_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in df.columns]
    severity_cols = [f"severity_{t}" for t in THEMES if f"severity_{t}" in df.columns]

    X = df[feature_cols + severity_cols].copy().fillna(0).astype(float)

    # Add review length as a control variable
    if "word_count" in df.columns:
        X["word_count_norm"] = (df["word_count"] - df["word_count"].mean()) / (df["word_count"].std() + 1e-6)

    y = df["rating"].fillna(df["rating"].median())
    return X, y


def run_regression(df: pd.DataFrame) -> dict:
    """
    Ridge regression to estimate each theme's contribution to ratings.
    Returns coefficients as a dict: {theme: impact_score}
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        X, y = build_feature_matrix(df)
        if len(X) < 10:
            return {}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // 10), scoring="r2")
        r2 = round(float(np.mean(cv_scores)), 3)

        coef_dict = {}
        for col, coef in zip(X.columns, model.coef_):
            if col.startswith("theme_"):
                theme = col.replace("theme_", "")
                coef_dict[theme] = round(float(coef), 4)

        return {"coefficients": coef_dict, "r2_score": r2}

    except ImportError:
        print("[WARN] scikit-learn not available. Falling back to correlation analysis.")
        return _correlation_fallback(df)


def run_shap_analysis(df: pd.DataFrame) -> Optional[dict]:
    """
    Random Forest + SHAP for feature importance.
    Returns mean absolute SHAP values per theme.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap

        X, y = build_feature_matrix(df)
        if len(X) < 20:
            return None

        # Use only binary theme columns for interpretability
        theme_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in X.columns]
        X_themes = X[theme_cols]

        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_themes, y)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_themes)

        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_dict = {}
        for col, val in zip(theme_cols, mean_abs_shap):
            theme = col.replace("theme_", "")
            shap_dict[theme] = round(float(val), 4)

        feature_importance = dict(zip(
            [c.replace("theme_", "") for c in theme_cols],
            rf.feature_importances_.tolist()
        ))

        return {
            "shap_values": shap_dict,
            "feature_importance": feature_importance,
            "shap_matrix": shap_values.tolist(),
            "feature_names": [c.replace("theme_", "") for c in theme_cols],
        }

    except ImportError:
        print("[WARN] SHAP not available. Using feature importance only.")
        return _rf_importance_fallback(df)
    except Exception as e:
        print(f"[WARN] SHAP analysis failed: {e}")
        return _rf_importance_fallback(df)


def _rf_importance_fallback(df: pd.DataFrame) -> Optional[dict]:
    """Random Forest feature importance without SHAP."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        X, y = build_feature_matrix(df)
        theme_cols = [f"theme_{t}" for t in THEMES if f"theme_{t}" in X.columns]
        X_themes = X[theme_cols].fillna(0)

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_themes, y)

        return {
            "shap_values": {},
            "feature_importance": dict(zip(
                [c.replace("theme_", "") for c in theme_cols],
                rf.feature_importances_.tolist()
            )),
            "shap_matrix": [],
            "feature_names": [c.replace("theme_", "") for c in theme_cols],
        }
    except Exception:
        return None


def _correlation_fallback(df: pd.DataFrame) -> dict:
    """Simple correlation-based impact when sklearn unavailable."""
    coef_dict = {}
    for theme in THEMES:
        col = f"theme_{theme}"
        if col in df.columns and df["rating"].notna().any():
            corr = df[col].astype(float).corr(df["rating"])
            coef_dict[theme] = round(float(corr) if not np.isnan(corr) else 0.0, 4)
    return {"coefficients": coef_dict, "r2_score": None}


def build_impact_table(df: pd.DataFrame, theme_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build the master impact table combining:
    - Theme frequency
    - Rating delta
    - Regression coefficient
    - SHAP importance
    - Confidence score
    """
    regression = run_regression(df)
    shap_result = run_shap_analysis(df)

    coefs = regression.get("coefficients", {})
    shap_vals = shap_result.get("shap_values", {}) if shap_result else {}
    feat_imp = shap_result.get("feature_importance", {}) if shap_result else {}

    rows = []
    for _, row in theme_summary.iterrows():
        theme = row["theme"]
        freq_pct = row["frequency_pct"]
        rating_delta = row.get("rating_delta", 0) or 0
        avg_severity = row.get("avg_severity", 0) or 0

        coef = coefs.get(theme, 0)
        shap_imp = shap_vals.get(theme, 0)
        rf_imp = feat_imp.get(theme, 0)

        # Composite rating impact: blend regression coef + rating delta
        rating_impact = round((coef * 0.6 + rating_delta * 0.4), 3)

        # Severity score: weighted combination
        severity_score = round(
            (freq_pct / 100) * 0.35 +
            abs(rating_impact) * 0.35 +
            avg_severity * 0.30,
            3
        )

        # Confidence: based on sample size and model agreement
        n = row["frequency_count"]
        confidence = min(0.5 + (n / len(df)) * 0.3 + (rf_imp * 0.2), 0.97)

        rows.append({
            "theme": theme,
            "theme_label": THEME_LABELS[theme],
            "frequency_pct": freq_pct,
            "frequency_count": row["frequency_count"],
            "rating_impact": rating_impact,
            "avg_severity": avg_severity,
            "severity_score": round(severity_score, 3),
            "regression_coef": round(coef, 3),
            "shap_importance": round(shap_imp, 4),
            "rf_importance": round(rf_imp, 4),
            "confidence": round(confidence, 2),
            "evidence_samples": row.get("evidence_samples", []),
        })

    impact_df = pd.DataFrame(rows)
    # Rank by absolute rating impact (most impactful issues first)
    impact_df = impact_df.sort_values("severity_score", ascending=False).reset_index(drop=True)
    impact_df["rank"] = range(1, len(impact_df) + 1)

    return impact_df


def get_rating_segments(df: pd.DataFrame) -> dict:
    """
    Show which themes dominate 1-star vs 5-star reviews.
    Helps identify what causes best and worst experiences.
    """
    if "rating" not in df.columns:
        return {}

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
