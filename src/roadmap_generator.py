"""
Phase 7: Claude-Powered Business Roadmap Generator
Converts analytics into a structured, prioritized improvement plan.
Falls back to rule-based recommendations when in mock mode.
"""

import json
import re
import pandas as pd
from src.config import MOCK_MODE, GROQ_API_KEY, GROQ_MODEL, THEME_LABELS


# ── Rule-based recommendation templates (mock mode fallback) ─────────────────

RECOMMENDATIONS = {
    "wait_time": {
        "quick_win": {
            "action": "Deploy SMS/email appointment reminders to reduce no-shows and gaps",
            "effort": "Low",
            "cost_tier": "$500–$2,000",
            "timeline": "2–4 weeks",
            "kpis": ["No-show rate", "Average wait time", "Patient satisfaction score"],
            "expected_lift": "+0.25",
        },
        "strategic": {
            "action": "Redesign appointment scheduling workflow and hire a dedicated scheduling coordinator",
            "effort": "Medium",
            "cost_tier": "$40,000–$60,000/yr",
            "timeline": "60–90 days",
            "kpis": ["Average wait time", "Throughput per hour", "1-star review rate"],
            "expected_lift": "+0.55",
        },
    },
    "staff_behavior": {
        "quick_win": {
            "action": "Implement monthly patient satisfaction debrief sessions with frontline staff",
            "effort": "Low",
            "cost_tier": "$0–$500",
            "timeline": "2 weeks",
            "kpis": ["Staff satisfaction score", "Patient-reported staff rating"],
            "expected_lift": "+0.20",
        },
        "strategic": {
            "action": "Launch structured customer service and empathy training program for all patient-facing staff",
            "effort": "High",
            "cost_tier": "$20,000–$50,000",
            "timeline": "90 days",
            "kpis": ["Staff training completion rate", "Patient complaint rate", "Staff empathy score"],
            "expected_lift": "+0.65",
        },
    },
    "billing": {
        "quick_win": {
            "action": "Create a dedicated billing inquiry hotline with guaranteed 24-hour callback",
            "effort": "Low",
            "cost_tier": "$5,000–$15,000/yr",
            "timeline": "3–4 weeks",
            "kpis": ["Billing inquiry resolution time", "Billing complaint rate"],
            "expected_lift": "+0.15",
        },
        "strategic": {
            "action": "Implement transparent pre-service cost estimation tools and streamline insurance verification",
            "effort": "High",
            "cost_tier": "$40,000–$120,000",
            "timeline": "4–6 months",
            "kpis": ["Billing error rate", "Days in A/R", "Patient financial satisfaction"],
            "expected_lift": "+0.40",
        },
    },
    "cleanliness": {
        "quick_win": {
            "action": "Increase cleaning frequency in high-traffic areas and install visible cleanliness checklists",
            "effort": "Low",
            "cost_tier": "$3,000–$8,000/yr",
            "timeline": "1–2 weeks",
            "kpis": ["Cleanliness audit scores", "Patient-reported cleanliness rating"],
            "expected_lift": "+0.30",
        },
        "strategic": {
            "action": "Hire dedicated EVS coordinator and implement environmental hygiene monitoring system",
            "effort": "Medium",
            "cost_tier": "$15,000–$40,000/yr",
            "timeline": "60 days",
            "kpis": ["HAI rate", "Cleanliness audit score", "Patient satisfaction on environment"],
            "expected_lift": "+0.35",
        },
    },
    "communication": {
        "quick_win": {
            "action": "Implement post-visit follow-up calls or texts for all patients within 48 hours",
            "effort": "Low",
            "cost_tier": "$2,000–$8,000/yr",
            "timeline": "2–3 weeks",
            "kpis": ["Follow-up completion rate", "Patient-reported communication score"],
            "expected_lift": "+0.20",
        },
        "strategic": {
            "action": "Deploy a patient-facing portal with real-time care updates, test results, and direct messaging",
            "effort": "High",
            "cost_tier": "$20,000–$60,000",
            "timeline": "90–120 days",
            "kpis": ["Portal adoption rate", "Care coordination score", "Communication rating"],
            "expected_lift": "+0.45",
        },
    },
    "appointment_scheduling": {
        "quick_win": {
            "action": "Add online self-scheduling and automated confirmation system",
            "effort": "Low",
            "cost_tier": "$2,000–$8,000/yr",
            "timeline": "3–4 weeks",
            "kpis": ["Online booking rate", "Schedule fill rate", "Cancellation rate"],
            "expected_lift": "+0.25",
        },
        "strategic": {
            "action": "Implement AI-assisted scheduling optimization to balance provider capacity and patient demand",
            "effort": "Medium",
            "cost_tier": "$15,000–$40,000",
            "timeline": "60–90 days",
            "kpis": ["Appointment availability", "No-show rate", "Schedule utilization"],
            "expected_lift": "+0.50",
        },
    },
    "facility": {
        "quick_win": {
            "action": "Refresh waiting areas with comfortable seating, better signage, and improved lighting",
            "effort": "Low",
            "cost_tier": "$5,000–$15,000",
            "timeline": "2–4 weeks",
            "kpis": ["Patient comfort score", "Facility rating in reviews"],
            "expected_lift": "+0.15",
        },
        "strategic": {
            "action": "Conduct comprehensive facility assessment and develop a phased modernization roadmap",
            "effort": "High",
            "cost_tier": "$50,000–$500,000+",
            "timeline": "6–18 months",
            "kpis": ["Facility condition index", "Patient satisfaction with environment"],
            "expected_lift": "+0.30",
        },
    },
    "food": {
        "quick_win": {
            "action": "Introduce fresh grab-and-go options and improve menu variety in cafeteria",
            "effort": "Low",
            "cost_tier": "$3,000–$10,000",
            "timeline": "3–4 weeks",
            "kpis": ["Cafeteria satisfaction score", "Meal complaint rate"],
            "expected_lift": "+0.15",
        },
        "strategic": {
            "action": "Partner with a professional food service management company and hire a registered dietitian",
            "effort": "High",
            "cost_tier": "$30,000–$80,000/yr",
            "timeline": "90 days",
            "kpis": ["Patient nutrition satisfaction", "Meal quality score"],
            "expected_lift": "+0.20",
        },
    },
    "parking": {
        "quick_win": {
            "action": "Implement clear parking signage, reserve front spots for mobility-impaired patients",
            "effort": "Low",
            "cost_tier": "$500–$2,000",
            "timeline": "1 week",
            "kpis": ["Parking accessibility score", "Patient arrival experience"],
            "expected_lift": "+0.10",
        },
        "strategic": {
            "action": "Add overflow parking area or negotiate with nearby lot; consider shuttle or valet service",
            "effort": "High",
            "cost_tier": "$20,000–$100,000",
            "timeline": "3–6 months",
            "kpis": ["Patient reported parking satisfaction", "On-time arrival rate"],
            "expected_lift": "+0.15",
        },
    },
}


def generate_roadmap_mock(impact_df: pd.DataFrame, sim_df: pd.DataFrame, eda_stats: dict) -> dict:
    """
    Rule-based roadmap generator (runs without API key).
    Produces identical structure to the Claude-generated roadmap.
    """
    roadmap = {"quick_wins": [], "strategic": [], "long_term": []}
    priority = 1

    # Sort themes by severity
    sorted_themes = impact_df.sort_values("severity_score", ascending=False)

    for _, row in sorted_themes.iterrows():
        theme = row["theme"]
        if theme not in RECOMMENDATIONS:
            continue

        recs = RECOMMENDATIONS[theme]
        freq = row["frequency_pct"]
        severity = row["severity_score"]
        issue_class = row.get("issue_class", "MODERATE")

        # Get financial data if available
        revenue = None
        if sim_df is not None and len(sim_df) > 0:
            sim_row = sim_df[sim_df["theme"] == theme]
            if not sim_row.empty:
                revenue = int(sim_row.iloc[0]["revenue_impact_annual"])

        # Quick win
        qw = recs["quick_win"].copy()
        qw.update({
            "priority": priority,
            "theme": theme,
            "theme_label": THEME_LABELS.get(theme, theme),
            "category": "Quick Win",
            "issue_class": issue_class,
            "frequency_pct": freq,
            "severity_score": severity,
            "revenue_impact": revenue,
            "confidence": round(row.get("confidence", 0.75), 2),
        })
        roadmap["quick_wins"].append(qw)

        # Strategic improvement
        st = recs["strategic"].copy()
        st.update({
            "priority": priority,
            "theme": theme,
            "theme_label": THEME_LABELS.get(theme, theme),
            "category": "Strategic",
            "issue_class": issue_class,
            "frequency_pct": freq,
            "severity_score": severity,
            "revenue_impact": revenue,
            "confidence": round(row.get("confidence", 0.75), 2),
        })
        roadmap["strategic"].append(st)
        priority += 1

    # Long-term: culture and systemic transformation
    roadmap["long_term"] = [
        {
            "priority": 1,
            "category": "Long-term",
            "action": "Build a Patient Experience Committee with quarterly review cycles",
            "effort": "Medium",
            "timeline": "6–12 months",
            "kpis": ["Patient satisfaction index", "Net Promoter Score (NPS)", "Review star average"],
            "expected_lift": "+0.3 (cumulative)",
            "confidence": 0.72,
        },
        {
            "priority": 2,
            "category": "Long-term",
            "action": "Implement real-time review monitoring with automated alert system for negative patterns",
            "effort": "Medium",
            "timeline": "3–6 months",
            "kpis": ["Response rate to reviews", "Issue resolution time", "Trend detection latency"],
            "expected_lift": "Prevents future -0.5 to -1.0 star drops",
            "confidence": 0.80,
        },
        {
            "priority": 3,
            "category": "Long-term",
            "action": "Pursue healthcare accreditation that mandates patient experience standards",
            "effort": "High",
            "timeline": "12–24 months",
            "kpis": ["Accreditation status", "Compliance score", "Benchmark ranking"],
            "expected_lift": "+0.4–0.6 (reputational)",
            "confidence": 0.68,
        },
    ]

    return roadmap


def generate_roadmap_groq(impact_df: pd.DataFrame, sim_df: pd.DataFrame, eda_stats: dict) -> dict:
    """
    Use Groq (LLaMA 3.3 70B) to generate a tailored roadmap.
    Falls back to rule-based mock if unavailable.
    """
    if MOCK_MODE:
        return generate_roadmap_mock(impact_df, sim_df, eda_stats)

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

    return generate_roadmap_mock(impact_df, sim_df, eda_stats)


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
