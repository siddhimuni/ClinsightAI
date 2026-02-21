"""
Roadmap Generator

Uses Groq LLM to generate improvement recommendations from the impact analysis.
Also computes an executive summary with health score.
"""

import json
import pandas as pd
from src.config import MOCK_MODE, GROQ_API_KEY, GROQ_MODEL


def generate_roadmap(impact_df: pd.DataFrame, eda_stats: dict) -> list:
    """Generate a prioritized list of recommendations using Groq LLM."""
    if MOCK_MODE:
        print("[INFO] No GROQ_API_KEY — skipping roadmap generation.")
        return []

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        top_issues = impact_df.head(5)[
            ["theme_label", "frequency_pct", "rating_impact", "severity_score"]
        ].to_dict("records")

        prompt = f"""You are a healthcare operations consultant.

Hospital Summary:
- Reviews Analyzed: {eda_stats.get('total_reviews', 'N/A')}
- Average Rating: {eda_stats.get('avg_rating', 'N/A')} / 5.0
- Negative Review Rate: {eda_stats.get('pct_negative', 'N/A')}%

Top Issues (ranked by impact):
{json.dumps(top_issues, indent=2)}

Generate 5-7 prioritized recommendations. Return ONLY a JSON array:
[
  {{
    "priority": 1,
    "recommendation": "specific actionable recommendation",
    "expected_rating_lift": "+0.X",
    "confidence": 0.0-1.0
  }}
]

Order by priority. Be specific and actionable. Return JSON only."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        parsed = _parse_json(raw)
        if isinstance(parsed, list):
            return parsed

    except Exception as e:
        print(f"[WARN] Groq roadmap generation failed: {e}")

    return []


def _parse_json(raw: str):
    """Extract JSON from LLM response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = raw.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def get_executive_summary(
    eda_stats: dict,
    impact_df: pd.DataFrame,
    systemic_summary: dict,
) -> dict:
    """Compute health score and top-level stats for the dashboard."""
    avg_rating = eda_stats.get("avg_rating", 3.0)
    top_issues = impact_df.head(3)["theme_label"].tolist() if len(impact_df) > 0 else []

    health_score = int((avg_rating / 5.0) * 100) if avg_rating else 50

    if health_score >= 75:
        health_label = "Good"
    elif health_score >= 50:
        health_label = "Needs Improvement"
    else:
        health_label = "Critical"

    return {
        "avg_rating": avg_rating,
        "health_score": health_score,
        "health_label": health_label,
        "total_reviews": eda_stats.get("total_reviews", 0),
        "pct_negative": eda_stats.get("pct_negative", 0),
        "pct_positive": eda_stats.get("pct_positive", 0),
        "top_risk_themes": top_issues,
        "systemic_count": systemic_summary.get("systemic_count", 0),
    }
