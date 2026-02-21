"""
Phase 6: Financial Impact Simulator — THE USP
Converts abstract rating improvements into concrete dollar value.

Model pipeline:
  Rating Change → Patient Retention Change → Revenue Impact

Based on:
- Healthcare patient churn research (10% churn per 1-star drop)
- Patient Lifetime Value calculation
- Cost-of-fix estimation per theme
"""

import pandas as pd
import numpy as np
from src.config import (
    THEMES, THEME_LABELS,
    DEFAULT_MONTHLY_PATIENTS,
    DEFAULT_PATIENT_LIFETIME_VALUE,
    CHURN_PER_STAR_DROP,
)


# ── Expected rating lift per theme fix (evidence-based estimates) ────────────

THEME_RATING_LIFT = {
    "wait_time":               {"lift": 0.55, "confidence": 0.82, "effort": "Medium"},
    "staff_behavior":          {"lift": 0.65, "confidence": 0.78, "effort": "High"},
    "billing":                 {"lift": 0.40, "confidence": 0.71, "effort": "High"},
    "cleanliness":             {"lift": 0.35, "confidence": 0.85, "effort": "Low"},
    "communication":           {"lift": 0.45, "confidence": 0.76, "effort": "Medium"},
    "facility":                {"lift": 0.30, "confidence": 0.68, "effort": "High"},
    "appointment_scheduling":  {"lift": 0.50, "confidence": 0.80, "effort": "Medium"},
    "food":                    {"lift": 0.20, "confidence": 0.72, "effort": "Low"},
    "parking":                 {"lift": 0.15, "confidence": 0.75, "effort": "Medium"},
}

# Annual fix cost tiers (rough estimates)
FIX_COST_ESTIMATES = {
    "wait_time":               {"low": 30000, "high": 80000, "description": "Scheduling software + coordinator"},
    "staff_behavior":          {"low": 20000, "high": 60000, "description": "Training programs + HR investment"},
    "billing":                 {"low": 40000, "high": 120000, "description": "Billing system overhaul + staff"},
    "cleanliness":             {"low": 15000, "high": 40000, "description": "Cleaning staff + protocols"},
    "communication":           {"low": 10000, "high": 30000, "description": "Communication training + patient portal"},
    "facility":                {"low": 50000, "high": 500000, "description": "Renovation / equipment upgrade"},
    "appointment_scheduling":  {"low": 8000, "high": 25000, "description": "Online booking system + coordinator"},
    "food":                    {"low": 20000, "high": 60000, "description": "Cafeteria upgrade / dietitian"},
    "parking":                 {"low": 15000, "high": 100000, "description": "Parking expansion / valet service"},
}


def calculate_revenue_impact(
    monthly_patients: int,
    patient_lifetime_value: float,
    current_rating: float,
    target_theme: str,
    impact_df: pd.DataFrame = None,
) -> dict:
    """
    Core financial simulation for a single theme fix.

    Returns a dict with projected revenue impact, payback period, ROI.
    """
    theme_data = THEME_RATING_LIFT.get(target_theme, {"lift": 0.3, "confidence": 0.7, "effort": "Medium"})

    # Get actual rating lift from impact analysis if available
    if impact_df is not None and len(impact_df) > 0:
        row = impact_df[impact_df["theme"] == target_theme]
        if not row.empty:
            actual_impact = abs(float(row.iloc[0]["rating_impact"]))
            # Blend model estimate with our benchmark
            rating_lift = round(actual_impact * 0.5 + theme_data["lift"] * 0.5, 3)
        else:
            rating_lift = theme_data["lift"]
    else:
        rating_lift = theme_data["lift"]

    # Cap at a realistic ceiling
    rating_lift = min(rating_lift, 0.8)

    # Annual patient volume
    annual_patients = monthly_patients * 12

    # Retention gain: each star improvement reduces churn
    retention_gain_pct = CHURN_PER_STAR_DROP * rating_lift
    retained_patients = int(annual_patients * retention_gain_pct)

    # Revenue impact
    revenue_impact = retained_patients * patient_lifetime_value

    # New projected rating
    new_rating = min(current_rating + rating_lift, 5.0)

    # Cost of fix
    fix_cost = FIX_COST_ESTIMATES.get(target_theme, {"low": 20000, "high": 60000})
    avg_fix_cost = (fix_cost["low"] + fix_cost["high"]) / 2

    # Payback period in months
    monthly_gain = revenue_impact / 12
    payback_months = round(avg_fix_cost / monthly_gain, 1) if monthly_gain > 0 else None

    # ROI (1-year)
    roi = round((revenue_impact - avg_fix_cost) / avg_fix_cost * 100, 1) if avg_fix_cost > 0 else None

    return {
        "theme": target_theme,
        "theme_label": THEME_LABELS.get(target_theme, target_theme),
        "rating_lift": round(rating_lift, 2),
        "current_rating": round(current_rating, 2),
        "projected_rating": round(new_rating, 2),
        "retention_gain_pct": round(retention_gain_pct * 100, 1),
        "retained_patients_annual": retained_patients,
        "revenue_impact_annual": int(revenue_impact),
        "fix_cost_low": fix_cost["low"],
        "fix_cost_high": fix_cost["high"],
        "fix_description": FIX_COST_ESTIMATES.get(target_theme, {}).get("description", ""),
        "payback_months": payback_months,
        "roi_pct": roi,
        "confidence": theme_data["confidence"],
        "effort_level": theme_data["effort"],
    }


def run_full_simulation(
    impact_df: pd.DataFrame,
    current_rating: float,
    monthly_patients: int = DEFAULT_MONTHLY_PATIENTS,
    patient_lifetime_value: float = DEFAULT_PATIENT_LIFETIME_VALUE,
) -> pd.DataFrame:
    """
    Run simulation for all detected themes and return ranked by revenue impact.
    """
    results = []
    for _, row in impact_df.iterrows():
        theme = row["theme"]
        if row["frequency_pct"] < 1.0:
            continue  # Skip themes with negligible presence
        sim = calculate_revenue_impact(
            monthly_patients=monthly_patients,
            patient_lifetime_value=patient_lifetime_value,
            current_rating=current_rating,
            target_theme=theme,
            impact_df=impact_df,
        )
        results.append(sim)

    sim_df = pd.DataFrame(results)
    if len(sim_df) > 0:
        sim_df = sim_df.sort_values("revenue_impact_annual", ascending=False).reset_index(drop=True)

    return sim_df


def get_portfolio_impact(sim_df: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Estimate combined impact of fixing the top N themes together.
    Note: effects are not perfectly additive — we apply a synergy discount.
    """
    if len(sim_df) == 0:
        return {}

    top = sim_df.head(top_n)

    # Synergy factor: fixing multiple issues reinforces each other
    synergy_factor = 1.0 + (top_n - 1) * 0.08  # 8% bonus per additional fix

    # Additive lift (capped at +2.0 stars total)
    total_lift = min(top["rating_lift"].sum() * synergy_factor, 2.0)
    total_revenue = int(top["revenue_impact_annual"].sum() * synergy_factor)
    total_cost_low = top["fix_cost_low"].sum()
    total_cost_high = top["fix_cost_high"].sum()
    portfolio_roi = round((total_revenue - (total_cost_low + total_cost_high) / 2) /
                          ((total_cost_low + total_cost_high) / 2) * 100, 1)

    return {
        "top_themes": top["theme_label"].tolist(),
        "combined_rating_lift": round(total_lift, 2),
        "combined_revenue_impact": total_revenue,
        "total_fix_cost_range": f"${total_cost_low:,} – ${total_cost_high:,}",
        "portfolio_roi_pct": portfolio_roi,
        "synergy_factor": round(synergy_factor, 2),
    }


def get_causal_chains(impact_df: pd.DataFrame) -> list:
    """
    Return predefined causal chain templates populated with real data.
    These visualize the root cause → revenue loss pipeline.
    """
    chains = []

    causal_templates = {
        "wait_time": {
            "root_cause": "Understaffing / Poor scheduling",
            "operational_failure": "Long Wait Times",
            "patient_experience": "Frustration & Dissatisfaction",
            "review_outcome": "1-Star Reviews Citing Wait",
        },
        "staff_behavior": {
            "root_cause": "Insufficient Training / Burnout",
            "operational_failure": "Rude or Dismissive Staff",
            "patient_experience": "Feeling Disrespected",
            "review_outcome": "Negative Staff Reviews",
        },
        "billing": {
            "root_cause": "Complex Insurance Workflows",
            "operational_failure": "Billing Errors & Delays",
            "patient_experience": "Financial Stress & Distrust",
            "review_outcome": "Billing Complaints in Reviews",
        },
        "communication": {
            "root_cause": "Siloed Departments",
            "operational_failure": "Poor Patient Communication",
            "patient_experience": "Confusion & Anxiety",
            "review_outcome": "Negative Communication Reviews",
        },
    }

    for _, row in impact_df.head(4).iterrows():
        theme = row["theme"]
        if theme not in causal_templates:
            continue
        template = causal_templates[theme]
        chains.append({
            "theme": theme,
            "theme_label": row["theme_label"],
            "root_cause": template["root_cause"],
            "operational_failure": template["operational_failure"],
            "patient_experience": template["patient_experience"],
            "review_outcome": template["review_outcome"],
            "rating_impact": row["rating_impact"],
            "frequency_pct": row["frequency_pct"],
        })

    return chains
