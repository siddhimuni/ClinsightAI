"""
ClinsightAI — Main Analysis Pipeline
Run this to generate a JSON results file without the dashboard.

Usage:
    python main.py
    python main.py --data data/hospital_reviews.csv
    python main.py --output outputs/results.json
"""

import argparse
import json
import os
import sys

from src.data_loader import load_dataset, preprocess, get_eda_stats
from src.theme_extractor import run_theme_extraction, get_theme_summary
from src.impact_quantifier import build_impact_table, get_rating_segments
from src.systemic_detector import classify_issues, detect_trends, cluster_reviews, get_systemic_summary
# from src.financial_simulator import run_full_simulation, get_portfolio_impact, get_causal_chains
from src.roadmap_generator import generate_roadmap_claude, get_executive_summary
from src.config import MOCK_MODE


def run_pipeline(data_path: str, output_path: str):
    print("\n" + "="*60)
    print("  ClinsightAI — Healthcare Review Intelligence")
    print("="*60)
    if MOCK_MODE:
        print("  Mode: DEMO (no API key — using rule-based AI simulation)")
    else:
        print("  Mode: LIVE (Claude AI active)")
    print("="*60 + "\n")

    # Phase 1 & 2: Load and preprocess
    print("[1/6] Loading and preprocessing data...")
    df = load_dataset(data_path)
    df = preprocess(df)
    eda_stats = get_eda_stats(df)
    print(f"      Reviews: {eda_stats['total_reviews']:,} | Avg Rating: {eda_stats.get('avg_rating', 'N/A')}")

    # Phase 3: Theme extraction
    print("\n[2/6] Extracting themes...")
    df = run_theme_extraction(df, use_claude=True)
    theme_summary = get_theme_summary(df)
    print(f"      Themes detected: {len(theme_summary)}")

    # Phase 4: Impact quantification
    print("\n[3/6] Quantifying rating impact...")
    impact_df = build_impact_table(df, theme_summary)
    print(f"      Top theme by impact: {impact_df.iloc[0]['theme_label']} ({impact_df.iloc[0]['rating_impact']:+.2f} stars)")

    # Phase 5: Systemic detection
    print("\n[4/6] Detecting systemic issues...")
    impact_df = classify_issues(impact_df)
    df = cluster_reviews(df)
    trend_data = detect_trends(df)
    systemic_summary = get_systemic_summary(impact_df)
    print(f"      Systemic issues: {systemic_summary['systemic_count']} | "
          f"Moderate: {len(systemic_summary['moderate_issues'])}")

    # Phase 6: Roadmap generation
    print("\n[5/6] Generating action roadmap...")
    sim_df = None
    roadmap = generate_roadmap_claude(impact_df, sim_df, eda_stats)
    exec_summary = get_executive_summary(eda_stats, impact_df, sim_df, systemic_summary)

    # Phase 7: Compile output
    print("\n[6/6] Compiling structured output...")

    output = {
        "clinic_summary": {
            "overall_rating_mean": eda_stats.get("avg_rating"),
            "health_score": exec_summary["health_score"],
            "health_label": exec_summary["health_label"],
            "total_reviews": eda_stats["total_reviews"],
            "pct_negative": eda_stats.get("pct_negative"),
            "pct_positive": eda_stats.get("pct_positive"),
            "primary_risk_themes": exec_summary["top_risk_themes"],
            "systemic_issues": systemic_summary["systemic_issues"],
        },
        "theme_analysis": [
            {
                "theme": row["theme"],
                "theme_label": row["theme_label"],
                "frequency_percentage": row["frequency_pct"],
                "rating_impact": row["rating_impact"],
                "severity_score": row["severity_score"],
                "issue_classification": row.get("issue_class", "MODERATE"),
                "confidence": row["confidence"],
                "evidence_samples": row.get("evidence_samples", []),
            }
            for _, row in impact_df.iterrows()
        ],
        "improvement_roadmap": {
            "quick_wins": roadmap.get("quick_wins", [])[:4],
            "strategic": roadmap.get("strategic", [])[:4],
            "long_term": roadmap.get("long_term", []),
        },
        "trend_analysis": {
            "available": trend_data.get("available", False),
            "rising_themes": trend_data.get("rising_themes", []),
        },
        "rating_segments": {
            "top_themes_in_1star_reviews": dict(
                list(segments.items())[:5]
            ) if (segments := get_rating_segments(df)) else {},
        },
    }

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[OK] Results saved to: {output_path}")
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Overall Rating:     {eda_stats.get('avg_rating', 'N/A')} / 5.0")
    print(f"  Health Score:       {exec_summary['health_score']}/100 ({exec_summary['health_label']})")
    print(f"  Systemic Issues:    {systemic_summary['systemic_count']}")
    print(f"  Top Risk:           {exec_summary['top_risk_themes'][0] if exec_summary['top_risk_themes'] else 'N/A'}")
    print("="*60)
    print("\n  Run the dashboard: streamlit run dashboard/app.py\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClinsightAI Analysis Pipeline")
    parser.add_argument("--data", default="data/hospital.csv", help="Path to reviews CSV")
    parser.add_argument("--output", default="outputs/results.json", help="Path to output JSON")
    args = parser.parse_args()

    run_pipeline(args.data, args.output)
