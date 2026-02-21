"""
ClinsightAI — Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_dataset, preprocess, get_eda_stats
from src.theme_extractor import run_theme_extraction, get_theme_summary
from src.impact_quantifier import build_impact_table, get_rating_segments
from src.systemic_detector import classify_issues, cluster_reviews, get_systemic_summary
from src.roadmap_generator import generate_roadmap, get_executive_summary
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
    impact_df = classify_issues(impact_df)
    df = cluster_reviews(df)
    systemic_summary = get_systemic_summary(impact_df)

    roadmap = generate_roadmap(impact_df, eda_stats)
    exec_summary = get_executive_summary(eda_stats, impact_df, systemic_summary)

    rating_segments = get_rating_segments(df)

    return {
        "df": df,
        "eda_stats": eda_stats,
        "theme_summary": theme_summary,
        "impact_df": impact_df,
        "roadmap": roadmap,
        "exec_summary": exec_summary,
        "systemic_summary": systemic_summary,
        "rating_segments": rating_segments,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinsightAI")
    st.markdown("*AI-Driven Healthcare Review Intelligence*")
    st.divider()

    st.markdown("### Data Source")
    data_option = st.radio(
        "Choose data source",
        ["Use default dataset", "Upload CSV file"],
        index=0,
    )

    data_path = "data/hospital.csv"
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload hospital reviews CSV", type=["csv"])
        if uploaded_file:
            save_path = "data/uploaded_reviews.csv"
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            data_path = save_path

    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Executive Summary", "EDA Overview", "Theme Analysis", "Impact Matrix",
         "Systemic Issues", "Action Roadmap", "Raw Data"],
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
            f"#{item.get('priority', '?')} — {item.get('recommendation', 'N/A')[:80]}",
            expanded=(item.get("priority", 99) == 1),
        ):
            st.markdown(f"**Recommendation:** {item.get('recommendation', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Expected Rating Lift:** {item.get('expected_rating_lift', 'N/A')}")
            with col2:
                st.markdown(f"**Confidence:** {item.get('confidence', 'N/A')}")


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
