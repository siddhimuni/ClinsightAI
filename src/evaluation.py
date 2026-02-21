"""
Evaluation & Metrics Module

Validates theme extraction accuracy, regression quality, and end-to-end output.
Produces evaluation visualizations for the presentation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import THEMES, THEME_LABELS
from src.theme_extractor import THEME_DESCRIPTIONS, SIMILARITY_THRESHOLD


# ── Test Cases ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "review": "I waited over 3 hours past my appointment. The delay was unacceptable.",
        "expected_themes": ["wait_time", "appointment_scheduling"],
        "expected_sentiment": "negative",
    },
    {
        "review": "The nurses were incredibly kind and the doctor explained everything clearly.",
        "expected_themes": ["staff_behavior", "communication"],
        "expected_sentiment": "positive",
    },
    {
        "review": "They charged me twice for the same procedure. Insurance claim was denied due to hospital error.",
        "expected_themes": ["billing"],
        "expected_sentiment": "negative",
    },
    {
        "review": "The hospital was spotless. Every room was well maintained and the equipment looked modern.",
        "expected_themes": ["cleanliness", "facility"],
        "expected_sentiment": "positive",
    },
    {
        "review": "Parking was impossible to find. I was late for my appointment because of it.",
        "expected_themes": ["parking", "appointment_scheduling"],
        "expected_sentiment": "negative",
    },
    {
        "review": "The food served during my stay was bland and cold. No dietary options available.",
        "expected_themes": ["food"],
        "expected_sentiment": "negative",
    },
    {
        "review": "Excellent hospital. Staff was professional, facility was clean, and my appointment was on time.",
        "expected_themes": ["staff_behavior", "cleanliness", "appointment_scheduling"],
        "expected_sentiment": "positive",
    },
    {
        "review": "The doctor was rude and dismissive. Did not listen to my concerns at all.",
        "expected_themes": ["staff_behavior", "communication"],
        "expected_sentiment": "negative",
    },
]


def run_theme_test_cases() -> pd.DataFrame:
    """
    Run test cases through the theme extractor and evaluate accuracy.
    Returns a dataframe with per-test results.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    theme_names = list(THEME_DESCRIPTIONS.keys())
    theme_texts = list(THEME_DESCRIPTIONS.values())
    theme_embeddings = model.encode(theme_texts)

    results = []
    for tc in TEST_CASES:
        review_emb = model.encode([tc["review"]])
        similarities = cosine_similarity(review_emb, theme_embeddings)[0]

        detected = [theme_names[i] for i, s in enumerate(similarities) if s >= SIMILARITY_THRESHOLD]
        expected = tc["expected_themes"]

        true_positives = set(detected) & set(expected)
        false_positives = set(detected) - set(expected)
        false_negatives = set(expected) - set(detected)

        precision = len(true_positives) / len(detected) if detected else 0
        recall = len(true_positives) / len(expected) if expected else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "review": tc["review"][:80] + "...",
            "expected": ", ".join([THEME_LABELS[t] for t in expected]),
            "detected": ", ".join([THEME_LABELS[t] for t in detected]),
            "correct": ", ".join([THEME_LABELS[t] for t in true_positives]),
            "missed": ", ".join([THEME_LABELS[t] for t in false_negatives]),
            "extra": ", ".join([THEME_LABELS[t] for t in false_positives]),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        })

    return pd.DataFrame(results)


def compute_overall_metrics(test_results: pd.DataFrame) -> dict:
    """Compute aggregate precision, recall, F1 across all test cases."""
    return {
        "avg_precision": round(test_results["precision"].mean(), 3),
        "avg_recall": round(test_results["recall"].mean(), 3),
        "avg_f1": round(test_results["f1"].mean(), 3),
        "n_test_cases": len(test_results),
        "perfect_match_rate": round((test_results["f1"] == 1.0).mean(), 3),
    }


def plot_similarity_distribution(df: pd.DataFrame, save_path: str = "outputs/similarity_distribution.png"):
    """
    Plot the cosine similarity distribution for all reviews across all themes.
    Shows how well themes separate from non-themes.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    theme_texts = list(THEME_DESCRIPTIONS.values())
    theme_embeddings = model.encode(theme_texts)

    review_col = "review_clean" if "review_clean" in df.columns else "review_text"
    sample = df.sample(min(200, len(df)), random_state=42)
    review_embeddings = model.encode(sample[review_col].tolist())

    sim_matrix = cosine_similarity(review_embeddings, theme_embeddings)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    theme_names = list(THEME_DESCRIPTIONS.keys())

    for idx, (ax, theme) in enumerate(zip(axes.flat, theme_names)):
        scores = sim_matrix[:, idx]
        ax.hist(scores, bins=30, alpha=0.7, color="#4361ee", edgecolor="white")
        ax.axvline(x=SIMILARITY_THRESHOLD, color="red", linestyle="--", label=f"Threshold={SIMILARITY_THRESHOLD}")
        ax.set_title(THEME_LABELS[theme], fontsize=10)
        ax.set_xlabel("Cosine Similarity", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        pct_above = (scores >= SIMILARITY_THRESHOLD).mean() * 100
        ax.text(0.95, 0.95, f"{pct_above:.0f}% match", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="red")

    plt.suptitle("Cosine Similarity Distribution per Theme (200 review sample)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved similarity distribution plot to {save_path}")


def plot_regression_diagnostics(df: pd.DataFrame, save_path: str = "outputs/regression_diagnostics.png"):
    """Plot regression coefficient comparison with rating deltas and confidence intervals."""
    from src.impact_quantifier import run_regression

    regression = run_regression(df)
    coefs = regression["coefficients"]
    conf = regression["confidence"]

    themes = list(coefs.keys())
    values = [coefs[t] for t in themes]
    confidences = [conf.get(t, 0.5) for t in themes]
    labels = [THEME_LABELS[t] for t in themes]

    colors = ["#e63946" if v < 0 else "#2a9d8f" for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Coefficient bar chart
    y_pos = range(len(labels))
    ax1.barh(y_pos, values, color=colors, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Ridge Regression Coefficient (Rating Impact)")
    ax1.set_title("Theme Impact on Rating", fontweight="bold")

    # Confidence bar chart
    ax2.barh(y_pos, confidences, color="#4361ee", edgecolor="white", alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Bootstrap Confidence (sign stability)")
    ax2.set_title("Per-Theme Confidence", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved regression diagnostics to {save_path}")


def run_full_evaluation(df: pd.DataFrame):
    """Run all evaluations and print results."""
    import os
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "=" * 60)
    print("  EVALUATION & METRICS")
    print("=" * 60)

    # 1. Theme extraction test cases
    print("\n[1/3] Running theme extraction test cases...")
    test_results = run_theme_test_cases()
    metrics = compute_overall_metrics(test_results)

    print(f"\n  Test Cases: {metrics['n_test_cases']}")
    print(f"  Avg Precision: {metrics['avg_precision']}")
    print(f"  Avg Recall:    {metrics['avg_recall']}")
    print(f"  Avg F1:        {metrics['avg_f1']}")
    print(f"  Perfect Match: {metrics['perfect_match_rate']:.0%}")

    print("\n  Per-test results:")
    for _, row in test_results.iterrows():
        status = "✓" if row["f1"] >= 0.5 else "✗"
        print(f"  {status} F1={row['f1']:.2f} | Expected: {row['expected']} | Got: {row['detected']}")

    # 2. Similarity distribution
    print("\n[2/3] Generating similarity distribution plot...")
    plot_similarity_distribution(df)

    # 3. Regression diagnostics
    print("\n[3/3] Generating regression diagnostics plot...")
    plot_regression_diagnostics(df)

    print("\n" + "=" * 60)
    print("  Evaluation complete. Check outputs/ for visualizations.")
    print("=" * 60)

    return {
        "test_results": test_results,
        "metrics": metrics,
    }


if __name__ == "__main__":
    from src.data_loader import load_dataset, preprocess
    from src.theme_extractor import run_theme_extraction

    df = load_dataset("data/hospital.csv")
    df = preprocess(df)
    df = run_theme_extraction(df)
    run_full_evaluation(df)
