"""
Phase 7: Claude-Powered Business Roadmap Generator
Converts analytics into a structured, prioritized improvement plan.
"""

import json
import re
import pandas as pd
from src.config import GROQ_API_KEY, GROQ_MODEL, THEME_LABELS


def generate_roadmap_groq(impact_df: pd.DataFrame, sim_df: pd.DataFrame, eda_stats: dict) -> dict:
    """
    Use Groq (LLaMA 3.3 70B) to generate a tailored roadmap.
    """
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        # Prepare compact summary for Groq
        top_issues = impact_df.head(5)[
            ["theme_label", "frequency_pct", "rating_impact", "severity_score"]
        ].to_dict("records")

        top_sims = []
        if sim_df is not None and len(sim_df) > 0:
            top_sims = sim_df.head(3)[
                ["theme_label", "rating_lift", "revenue_impact_annual", "effort_level"]
            ].to_dict("records")

        prompt = f"""You are a healthcare operations strategy consultant.

Hospital Performance Summary:
- Total Reviews Analyzed: {eda_stats.get('total_reviews', 'N/A')}
- Average Rating: {eda_stats.get('avg_rating', 'N/A')} / 5.0
- Negative Review Rate: {eda_stats.get('pct_negative', 'N/A')}%

Top Operational Issues (ranked by impact):
{json.dumps(top_issues, indent=2)}

Top Financial Opportunities:
{json.dumps(top_sims, indent=2)}

Generate a prioritized improvement roadmap. Return ONLY valid JSON:
{{
  "quick_wins": [
    {{
      "priority": 1,
      "theme_label": "string",
      "category": "Quick Win",
      "action": "specific actionable recommendation",
      "effort": "Low|Medium|High",
      "cost_tier": "string",
      "timeline": "string",
      "kpis": ["kpi1", "kpi2"],
      "expected_lift": "string",
      "confidence": 0.0-1.0
    }}
  ],
  "strategic": [ ... same structure ... ],
  "long_term": [ ... same structure ... ]
}}

Include 3-4 quick wins, 3-4 strategic improvements, and 2-3 long-term initiatives.
Focus on specificity, business value, and measurability. Return JSON only."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        # Robust JSON extraction — handle extra text around the JSON block
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start:i+1])
                        except json.JSONDecodeError:
                            break

    except Exception as e:
        print(f"[WARN] Groq roadmap generation failed: {e}. Using rule-based fallback.")


# Aliases so existing callers (dashboard, main.py) don't need to change
def generate_roadmap_claude(impact_df, sim_df, eda_stats):
    return generate_roadmap_groq(impact_df, sim_df, eda_stats)

def generate_roadmap_gemini(impact_df, sim_df, eda_stats):
    return generate_roadmap_groq(impact_df, sim_df, eda_stats)


def format_roadmap_for_display(roadmap: dict) -> list:
    """Flatten roadmap into a list of items for dashboard display."""
    items = []
    for category, entries in roadmap.items():
        for item in entries:
            items.append(item)
    return items


def get_executive_summary(
    eda_stats: dict,
    impact_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    systemic_summary: dict,
) -> dict:
    """
    Generate a concise executive summary for the dashboard header.
    """
    avg_rating = eda_stats.get("avg_rating", 3.0)
    top_issues = impact_df.head(3)["theme_label"].tolist() if len(impact_df) > 0 else []

    total_revenue_opportunity = 0
    if sim_df is not None and len(sim_df) > 0:
        total_revenue_opportunity = int(sim_df.head(3)["revenue_impact_annual"].sum())

    # Health score: 0–100 based on rating
    health_score = int((avg_rating / 5.0) * 100) if avg_rating else 50

    if health_score >= 75:
        health_label = "Good"
        health_color = "green"
    elif health_score >= 50:
        health_label = "Needs Improvement"
        health_color = "orange"
    else:
        health_label = "Critical"
        health_color = "red"

    return {
        "avg_rating": avg_rating,
        "health_score": health_score,
        "health_label": health_label,
        "health_color": health_color,
        "total_reviews": eda_stats.get("total_reviews", 0),
        "pct_negative": eda_stats.get("pct_negative", 0),
        "pct_positive": eda_stats.get("pct_positive", 0),
        "top_risk_themes": top_issues,
        "systemic_count": systemic_summary.get("systemic_count", 0),
        "revenue_opportunity": total_revenue_opportunity,
        "unique_hospitals": eda_stats.get("unique_hospitals", 1),
    }