"""
ClinsightAI — Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_dataset, preprocess, get_eda_stats
from src.theme_extractor import run_theme_extraction, get_theme_summary
from src.impact_quantifier import build_impact_table, get_rating_segments, run_regression
from src.systemic_detector import classify_issues, cluster_reviews, get_systemic_summary
from src.roadmap_generator import generate_roadmap, get_executive_summary
from src.evaluation import run_theme_test_cases, compute_overall_metrics
from src.config import THEMES, THEME_LABELS, MOCK_MODE


st.set_page_config(
    page_title="ClinsightAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; }
    .subtitle   { font-size: 1rem; color: #666; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_full_analysis(data_path: str):
    """Run the full pipeline and cache results."""
    df = load_dataset(data_path)
    df = preprocess(df)
    eda_stats = get_eda_stats(df)

    df = run_theme_extraction(df)
    theme_summary = get_theme_summary(df)
    impact_df = build_impact_table(df, theme_summary)
    impact_df = classify_issues(impact_df, df)
    df = cluster_reviews(df)
    systemic_summary = get_systemic_summary(impact_df)

    roadmap = generate_roadmap(impact_df, eda_stats)
    exec_summary = get_executive_summary(eda_stats, impact_df, systemic_summary)

    rating_segments = get_rating_segments(df)

    test_results = run_theme_test_cases()
    eval_metrics = compute_overall_metrics(test_results)

    regression = run_regression(df)

    return {
        "df": df,
        "eda_stats": eda_stats,
        "theme_summary": theme_summary,
        "impact_df": impact_df,
        "roadmap": roadmap,
        "exec_summary": exec_summary,
        "systemic_summary": systemic_summary,
        "rating_segments": rating_segments,
        "test_results": test_results,
        "eval_metrics": eval_metrics,
        "regression": regression,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinsightAI")
    st.markdown("*AI-Driven Healthcare Review Intelligence*")
    st.divider()

    data_path = "data/hospital.csv"

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Executive Summary", "EDA Overview", "Theme Analysis", "Impact Matrix",
         "Systemic Issues", "Action Roadmap", "Evaluation", "Impact Validation",
         "Review Explorer", "Raw Data"],
    )

    st.divider()
    if MOCK_MODE:
        st.warning("Running in **Demo Mode**\n\nAdd `GROQ_API_KEY` to `.env` for LLM features.")
    else:
        st.success("Groq LLM Active")


# ── Run analysis ─────────────────────────────────────────────────────────────
with st.spinner("Running AI analysis pipeline..."):
    results = run_full_analysis(data_path)

df          = results["df"]
eda_stats   = results["eda_stats"]
impact_df   = results["impact_df"]
roadmap     = results["roadmap"]
exec_sum    = results["exec_summary"]
sys_summary = results["systemic_summary"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown('<div class="main-title">🏥 ClinsightAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Driven Healthcare Review Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{exec_sum['total_reviews']:,}")
    with col2:
        st.metric("Avg Rating", f"{exec_sum['avg_rating']} ⭐")
    with col3:
        st.metric("Health Score", f"{exec_sum['health_score']}/100",
                  delta=exec_sum['health_label'])
    with col4:
        st.metric("Negative Reviews", f"{exec_sum['pct_negative']}%",
                  delta=f"-{exec_sum['pct_negative']}%", delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Rating Distribution")
        rating_dist = eda_stats.get("rating_distribution", {})
        if rating_dist:
            fig = px.bar(
                x=list(rating_dist.keys()),
                y=list(rating_dist.values()),
                labels={"x": "Star Rating", "y": "Number of Reviews"},
                color=list(rating_dist.values()),
                color_continuous_scale=["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#264653"],
            )
            fig.update_layout(height=280, showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top Risk Themes")
        for theme_label in exec_sum["top_risk_themes"]:
            row = impact_df[impact_df["theme_label"] == theme_label]
            if not row.empty:
                r = row.iloc[0]
                severity = r["severity_score"]
                color = "🔴" if severity > 0.6 else "🟡" if severity > 0.3 else "🟢"
                st.markdown(f"{color} **{theme_label}** — {r['frequency_pct']}% of reviews, "
                             f"impact: {r['rating_impact']:+.2f} stars")

        if exec_sum["systemic_count"] > 0:
            st.error(f"⚠️ {exec_sum['systemic_count']} systemic issue(s) detected.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Overview":
    st.header("Exploratory Data Analysis")
    st.caption("Understanding the dataset before diving into theme analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", f"{eda_stats['total_reviews']:,}")
    with col2:
        st.metric("Avg Review Length", f"{eda_stats['avg_review_length']} chars")
    with col3:
        st.metric("Avg Rating", f"{eda_stats.get('avg_rating', 'N/A')} / 5.0")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rating Distribution")
        rating_dist = eda_stats.get("rating_distribution", {})
        if rating_dist:
            fig = px.bar(
                x=list(rating_dist.keys()),
                y=list(rating_dist.values()),
                labels={"x": "Star Rating", "y": "Count"},
                color_discrete_sequence=["#4361ee"],
            )
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Sentiment Breakdown")
        sentiment_counts = df["overall_sentiment"].value_counts()
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={"positive": "#2a9d8f", "negative": "#e63946", "neutral": "#f4a261"},
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Review Length Distribution")
    fig = px.histogram(
        df, x="review_length", nbins=50,
        labels={"review_length": "Review Length (characters)", "count": "Number of Reviews"},
        color_discrete_sequence=["#4361ee"],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating vs Review Length")
    fig = px.box(
        df, x="rating", y="review_length",
        labels={"rating": "Star Rating", "review_length": "Review Length (chars)"},
        color_discrete_sequence=["#4361ee"],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # 1-star vs 5-star comparison
    st.subheader("What Separates 1-Star from 5-Star Reviews?")
    segments = results["rating_segments"]
    if segments:
        low = segments.get("low_rating_themes", {})
        high = segments.get("high_rating_themes", {})

        seg_data = []
        for theme_label in set(list(low.keys()) + list(high.keys())):
            seg_data.append({"Theme": theme_label, "Type": "⭐ 1-2 Star", "% Reviews": low.get(theme_label, 0)})
            seg_data.append({"Theme": theme_label, "Type": "⭐⭐⭐⭐⭐ 4-5 Star", "% Reviews": high.get(theme_label, 0)})

        fig = px.bar(
            pd.DataFrame(seg_data), x="Theme", y="% Reviews", color="Type",
            barmode="group",
            color_discrete_map={"⭐ 1-2 Star": "#e63946", "⭐⭐⭐⭐⭐ 4-5 Star": "#2a9d8f"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THEME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Theme Analysis":
    st.header("Theme Analysis")
    st.caption("Frequency and severity of operational themes across all reviews")

    theme_summary = results["theme_summary"]
    if len(theme_summary) == 0:
        st.warning("No themes detected.")
        st.stop()

    fig = px.bar(
        theme_summary.head(9),
        x="frequency_pct",
        y="theme_label",
        orientation="h",
        color="avg_severity",
        color_continuous_scale="RdYlGn_r",
        labels={"frequency_pct": "% of Reviews", "theme_label": "Theme",
                "avg_severity": "Avg Severity"},
        title="Theme Frequency & Severity",
    )
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Theme Deep Dive")
    selected_theme_label = st.selectbox("Select a theme", theme_summary["theme_label"].tolist())

    selected_row = theme_summary[theme_summary["theme_label"] == selected_theme_label].iloc[0]
    impact_row = impact_df[impact_df["theme_label"] == selected_theme_label]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frequency", f"{selected_row['frequency_pct']}%")
    with col2:
        st.metric("Avg Severity", f"{selected_row['avg_severity']:.2f}")
    with col3:
        if not impact_row.empty:
            st.metric("Rating Impact", f"{impact_row.iloc[0]['rating_impact']:+.2f} ⭐")

    st.markdown("**Evidence Samples:**")
    for sample in selected_row.get("evidence_samples", []):
        st.markdown(f"> *\"{sample}\"*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMPACT MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Matrix":
    st.header("Impact Matrix")
    st.caption("X = frequency, Y = rating impact, Size = severity score")

    if len(impact_df) == 0:
        st.warning("No impact data available.")
        st.stop()

    plot_df = impact_df.copy()
    plot_df["bubble_size"] = (plot_df["severity_score"] * 60 + 10).clip(10, 80)

    fig = px.scatter(
        plot_df,
        x="frequency_pct",
        y="rating_impact",
        size="bubble_size",
        color="rating_impact",
        color_continuous_scale="RdYlGn",
        text="theme_label",
        hover_data={"frequency_pct": True, "rating_impact": True,
                    "severity_score": True, "bubble_size": False},
        labels={
            "frequency_pct": "Theme Frequency (% of Reviews)",
            "rating_impact": "Rating Impact (stars)",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranked Theme Impact Table")
    display_cols = ["rank", "theme_label", "frequency_pct", "rating_impact",
                    "severity_score", "issue_class", "confidence"]
    available_cols = [c for c in display_cols if c in impact_df.columns]
    st.dataframe(
        impact_df[available_cols].rename(columns={
            "rank": "#", "theme_label": "Theme", "frequency_pct": "Frequency %",
            "rating_impact": "Rating Impact", "severity_score": "Severity",
            "issue_class": "Classification", "confidence": "Confidence",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEMIC ISSUES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Systemic Issues":
    st.header("Systemic Issue Detection")
    st.caption("Distinguishing widespread problems from isolated incidents")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🔴 Systemic: **{len(sys_summary['systemic_issues'])}**")
        for item in sys_summary["systemic_issues"]:
            st.markdown(f"- {item}")
        if not sys_summary["systemic_issues"]:
            st.markdown("*None detected*")
    with col2:
        st.warning(f"🟡 Moderate: **{len(sys_summary['moderate_issues'])}**")
        for item in sys_summary["moderate_issues"]:
            st.markdown(f"- {item}")
    with col3:
        st.success(f"🟢 Isolated: **{len(sys_summary['isolated_issues'])}**")
        for item in sys_summary["isolated_issues"]:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.subheader("Severity & Escalation Scores")
    if len(impact_df) > 0:
        display_cols = ["theme_label", "severity_score", "escalation_score", "rating_impact", "issue_class"]
        available = [c for c in display_cols if c in impact_df.columns]
        st.dataframe(
            impact_df.sort_values("severity_score", ascending=False)[available].rename(columns={
                "theme_label": "Theme", "severity_score": "Severity Score",
                "escalation_score": "Escalation Score", "rating_impact": "Rating Impact",
                "issue_class": "Classification",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.subheader("Escalation Risk Ranking")
    if len(impact_df) > 0:
        fig = px.bar(
            impact_df.sort_values("escalation_score", ascending=True),
            x="escalation_score",
            y="theme_label",
            orientation="h",
            color="issue_class",
            color_discrete_map={"SYSTEMIC": "#e63946", "MODERATE": "#f4a261", "ISOLATED": "#2a9d8f"},
            labels={"escalation_score": "Escalation Score", "theme_label": "Theme"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ACTION ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Action Roadmap":
    st.header("Improvement Roadmap")
    st.caption("Prioritized recommendations from AI analysis")

    if not roadmap:
        if MOCK_MODE:
            st.info("Roadmap generation requires a `GROQ_API_KEY` in `.env`.")
        else:
            st.warning("No recommendations generated.")
        st.stop()

    for item in roadmap:
        with st.expander(
            f"#{item.get('priority', '?')} — {item.get('recommendation', 'N/A')[:120].rsplit(' ', 1)[0]}",
            expanded=(item.get("priority", 99) == 1),
        ):
            st.markdown(f"**Recommendation:** {item.get('recommendation', '')}")
            st.markdown(f"**Confidence:** {item.get('confidence', 'N/A')}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.header("Evaluation & Metrics")
    st.caption("Validating theme extraction accuracy with hand-labeled test cases")

    eval_metrics = results["eval_metrics"]
    test_results = results["test_results"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Precision", f"{eval_metrics['avg_precision']:.3f}")
    with col2:
        st.metric("Avg Recall", f"{eval_metrics['avg_recall']:.3f}")
    with col3:
        st.metric("Avg F1 Score", f"{eval_metrics['avg_f1']:.3f}")

    st.markdown("---")

    st.subheader("Per-Test-Case Results")
    display_test = test_results.copy()
    display_test.columns = ["Review", "Expected", "Detected", "Correct", "Missed",
                            "Extra", "Precision", "Recall", "F1"]
    st.dataframe(display_test, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Per-Test F1 Scores")
        fig = px.bar(
            x=[f"Test {i+1}" for i in range(len(test_results))],
            y=test_results["f1"].tolist(),
            color=test_results["f1"].tolist(),
            color_continuous_scale="RdYlGn",
            labels={"x": "Test Case", "y": "F1 Score", "color": "F1"},
        )
        fig.add_hline(y=eval_metrics["avg_f1"], line_dash="dash", line_color="gray",
                      annotation_text=f"Avg F1 = {eval_metrics['avg_f1']:.3f}")
        fig.update_layout(height=350, showlegend=False, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Precision vs Recall")
        fig = px.scatter(
            test_results,
            x="recall",
            y="precision",
            text=[f"T{i+1}" for i in range(len(test_results))],
            labels={"recall": "Recall", "precision": "Precision"},
            color="f1",
            color_continuous_scale="RdYlGn",
            size=[20] * len(test_results),
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=350, xaxis=dict(range=[-0.05, 1.1]),
                          yaxis=dict(range=[-0.05, 1.1]), margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Evaluation Visualizations")
    viz_col1, viz_col2 = st.columns(2)

    sim_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "similarity_distribution.png")
    reg_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "regression_diagnostics.png")

    with viz_col1:
        st.markdown("**Cosine Similarity Distribution per Theme**")
        if os.path.exists(sim_path):
            st.image(sim_path, use_container_width=True)
        else:
            st.info("Run `python main.py` first to generate this plot.")

    with viz_col2:
        st.markdown("**Ridge Regression Coefficients & Bootstrap Confidence**")
        if os.path.exists(reg_path):
            st.image(reg_path, use_container_width=True)
        else:
            st.info("Run `python main.py` first to generate this plot.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMPACT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Validation":
    st.header("Theme Impact Score Validation")
    st.caption("Validating Ridge regression coefficients via bootstrap resampling and cross-validation")

    regression = results["regression"]
    coefs = regression["coefficients"]
    conf = regression["confidence"]
    r2 = regression["r2_score"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cross-Validated R²", f"{r2:.3f}",
                  help="Proportion of rating variance explained by theme presence (5-fold CV)")
    with col2:
        avg_conf = np.mean(list(conf.values()))
        st.metric("Avg Bootstrap Confidence", f"{avg_conf:.1%}",
                  help="Mean coefficient sign stability across 100 bootstrap rounds")
    with col3:
        n_significant = sum(1 for c in conf.values() if c >= 0.9)
        st.metric("High-Confidence Themes", f"{n_significant} / {len(conf)}",
                  help="Themes with ≥90% bootstrap sign stability")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Ridge Coefficients (Rating Impact)")
        theme_labels = [THEME_LABELS.get(t, t) for t in coefs.keys()]
        coef_values = list(coefs.values())
        colors = ["#e63946" if v < 0 else "#2a9d8f" for v in coef_values]

        fig = go.Figure(go.Bar(
            x=coef_values,
            y=theme_labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in coef_values],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
        fig.update_layout(height=400, xaxis_title="Ridge Coefficient (stars)",
                          yaxis={"categoryorder": "total ascending"},
                          margin=dict(t=20, b=20, l=10, r=60))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Bootstrap Confidence (Sign Stability)")
        conf_values = [conf.get(t, 0.5) for t in coefs.keys()]
        conf_colors = ["#2a9d8f" if c >= 0.9 else "#f4a261" if c >= 0.7 else "#e63946"
                       for c in conf_values]

        fig = go.Figure(go.Bar(
            x=conf_values,
            y=theme_labels,
            orientation="h",
            marker_color=conf_colors,
            text=[f"{v:.0%}" for v in conf_values],
            textposition="outside",
        ))
        fig.add_vline(x=0.9, line_dash="dash", line_color="gray", line_width=1,
                      annotation_text="90% threshold")
        fig.update_layout(height=400, xaxis_title="Confidence",
                          xaxis=dict(range=[0, 1.1]),
                          yaxis={"categoryorder": "total ascending"},
                          margin=dict(t=20, b=20, l=10, r=60))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Coefficient vs Rating Delta Comparison")
    st.caption("Ridge coefficient (controls for confounding) vs raw rating delta (naive average)")
    theme_summary = results["theme_summary"]

    comparison_data = []
    for _, row in theme_summary.iterrows():
        theme = row["theme"]
        comparison_data.append({
            "Theme": THEME_LABELS.get(theme, theme),
            "Ridge Coefficient": coefs.get(theme, 0),
            "Raw Rating Delta": row.get("rating_delta", 0),
        })
    comp_df = pd.DataFrame(comparison_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Ridge Coefficient", x=comp_df["Theme"], y=comp_df["Ridge Coefficient"],
        marker_color="#4361ee",
    ))
    fig.add_trace(go.Bar(
        name="Raw Rating Delta", x=comp_df["Theme"], y=comp_df["Raw Rating Delta"],
        marker_color="#f4a261",
    ))
    fig.update_layout(barmode="group", height=380,
                      yaxis_title="Stars", legend=dict(orientation="h", y=1.12))
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
> **Why do these differ?** The Ridge coefficient isolates each theme's *independent* effect on rating,
> controlling for all other themes. The raw delta is the naive difference between the average rating
> of reviews with this theme vs the overall average — it does not account for confounding
> (e.g., a review mentioning both Wait Time and Billing).
""")

    st.markdown("---")

    st.subheader("Validation Summary Table")
    val_rows = []
    for theme in coefs.keys():
        val_rows.append({
            "Theme": THEME_LABELS.get(theme, theme),
            "Coefficient": f"{coefs[theme]:+.4f}",
            "Direction": "Negative ↓" if coefs[theme] < 0 else "Positive ↑",
            "Confidence": f"{conf.get(theme, 0.5):.0%}",
            "Validated": "✅" if conf.get(theme, 0.5) >= 0.9 else "⚠️",
        })
    st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVIEW EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Review Explorer":
    st.header("Review Explorer — Explainability")
    st.caption("See exactly how the pipeline processes a single review: layered logic from text to insight")

    review_options = df["review_text"].tolist()
    short_options = [f"[{'⭐' * int(r)}] {txt[:80]}..." for r, txt in
                     zip(df["rating"].tolist(), review_options)]

    selected_idx = st.selectbox("Select a review", range(len(short_options)),
                                format_func=lambda i: short_options[i])
    selected_review = df.iloc[selected_idx]

    st.markdown("---")

    st.subheader("Step 1: Raw Review")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"> *\"{selected_review['review_text']}\"*")
    with col2:
        st.metric("Rating", f"{selected_review['rating']:.0f} ⭐")
    with col3:
        sentiment = selected_review.get("overall_sentiment", "N/A")
        color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        st.metric("Sentiment", f"{color_map.get(sentiment, '')} {sentiment}")

    st.markdown("---")

    st.subheader("Step 2: Theme Detection (Cosine Similarity)")
    sim_data = []
    for theme in THEMES:
        sim_col = f"sim_{theme}"
        theme_col = f"theme_{theme}"
        if sim_col in df.columns:
            sim_score = float(selected_review[sim_col])
            detected = bool(selected_review[theme_col])
            sim_data.append({
                "Theme": THEME_LABELS[theme],
                "Similarity": sim_score,
                "Detected": detected,
                "Status": "✅ Detected" if detected else "❌ Below threshold",
            })

    sim_df = pd.DataFrame(sim_data).sort_values("Similarity", ascending=False)

    fig = go.Figure(go.Bar(
        x=sim_df["Similarity"],
        y=sim_df["Theme"],
        orientation="h",
        marker_color=["#2a9d8f" if d else "#d3d3d3" for d in sim_df["Detected"]],
        text=[f"{s:.3f}" for s in sim_df["Similarity"]],
        textposition="outside",
    ))
    fig.add_vline(x=0.3, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Threshold = 0.3", annotation_position="top right")
    fig.update_layout(height=380, xaxis_title="Cosine Similarity",
                      xaxis=dict(range=[0, max(0.6, sim_df["Similarity"].max() + 0.05)]),
                      yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=20, b=20, l=10, r=60))
    st.plotly_chart(fig, use_container_width=True)

    detected_themes = sim_df[sim_df["Detected"]]["Theme"].tolist()
    if detected_themes:
        st.success(f"**Detected themes:** {', '.join(detected_themes)}")
    else:
        st.warning("No themes detected above the 0.3 similarity threshold.")

    st.markdown("---")

    st.subheader("Step 3: Impact Attribution")
    st.caption("How each detected theme contributes to the predicted rating shift")

    if detected_themes:
        attribution_data = []
        for _, row in impact_df.iterrows():
            if row["theme_label"] in detected_themes:
                attribution_data.append({
                    "Theme": row["theme_label"],
                    "Rating Impact": row["rating_impact"],
                    "Confidence": row["confidence"],
                    "Classification": row.get("issue_class", "MODERATE"),
                })

        if attribution_data:
            attr_df = pd.DataFrame(attribution_data)

            fig = go.Figure(go.Bar(
                x=attr_df["Rating Impact"],
                y=attr_df["Theme"],
                orientation="h",
                marker_color=["#e63946" if v < 0 else "#2a9d8f" for v in attr_df["Rating Impact"]],
                text=[f"{v:+.3f} stars" for v in attr_df["Rating Impact"]],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
            fig.update_layout(height=max(200, len(attr_df) * 50 + 80),
                              xaxis_title="Rating Impact (Ridge coefficient)",
                              margin=dict(t=20, b=20, l=10, r=80))
            st.plotly_chart(fig, use_container_width=True)

            net_impact = attr_df["Rating Impact"].sum()
            st.info(f"**Net predicted impact from detected themes:** {net_impact:+.3f} stars")

            st.dataframe(
                attr_df.rename(columns={
                    "Rating Impact": "Impact (stars)",
                    "Confidence": "Bootstrap Confidence",
                }),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("No themes detected — no impact attribution to show.")

    st.markdown("---")

    st.subheader("Step 4: Risk Classification")
    risk = selected_review.get("risk_cluster", "Unknown")
    risk_colors = {"High Risk": "🔴", "Moderate Risk": "🟡", "Positive Experience": "🟢"}
    st.markdown(f"**Review risk level:** {risk_colors.get(risk, '⚪')} **{risk}**")

    if detected_themes:
        classifications = impact_df[impact_df["theme_label"].isin(detected_themes)][
            ["theme_label", "issue_class", "escalation_score"]
        ].rename(columns={
            "theme_label": "Theme",
            "issue_class": "Classification",
            "escalation_score": "Escalation Score",
        })
        if not classifications.empty:
            st.dataframe(classifications, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Raw Data":
    st.header("Raw Review Data")
    st.caption("Browse and filter reviews with AI-detected theme tags")

    col1, col2 = st.columns(2)
    with col1:
        rating_filter = st.multiselect(
            "Filter by rating",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
        )
    with col2:
        theme_filter = st.multiselect(
            "Filter by theme",
            options=[THEME_LABELS[t] for t in THEMES if f"theme_{t}" in df.columns],
        )

    filtered = df[df["rating"].isin(rating_filter)] if rating_filter else df

    if theme_filter:
        theme_keys = [t for t, l in THEME_LABELS.items() if l in theme_filter]
        mask = pd.Series([False] * len(filtered), index=filtered.index)
        for t in theme_keys:
            col = f"theme_{t}"
            if col in filtered.columns:
                mask = mask | filtered[col].astype(bool)
        filtered = filtered[mask]

    st.write(f"Showing **{len(filtered):,}** reviews")

    display_cols = ["review_text", "rating", "overall_sentiment", "risk_cluster"]
    available = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available].copy()
    display_df.columns = ["Review", "Rating", "Sentiment", "Risk"][:len(available)]
    st.dataframe(display_df, use_container_width=True, height=500)
