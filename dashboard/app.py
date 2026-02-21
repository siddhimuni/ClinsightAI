"""
ClinsightAI — Streamlit Dashboard
AI-Driven Healthcare Review Intelligence System

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
from plotly.subplots import make_subplots

from src.data_loader import load_dataset, preprocess, get_eda_stats
from src.theme_extractor import run_theme_extraction, get_theme_summary
from src.impact_quantifier import build_impact_table, get_rating_segments
from src.systemic_detector import classify_issues, detect_trends, cluster_reviews, get_systemic_summary
from src.financial_simulator import run_full_simulation, get_portfolio_impact, get_causal_chains, calculate_revenue_impact
from src.roadmap_generator import generate_roadmap_claude, get_executive_summary, format_roadmap_for_display
from src.config import THEMES, THEME_LABELS, DEFAULT_MONTHLY_PATIENTS, DEFAULT_PATIENT_LIFETIME_VALUE, MOCK_MODE


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinsightAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; }
    .subtitle   { font-size: 1rem; color: #666; margin-top: -10px; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 16px;
        border-left: 4px solid #4361ee; margin-bottom: 8px;
    }
    .risk-high   { border-left-color: #e63946 !important; }
    .risk-medium { border-left-color: #f4a261 !important; }
    .risk-low    { border-left-color: #2a9d8f !important; }
    .usp-banner {
        background: linear-gradient(135deg, #4361ee, #3a0ca3);
        color: white; padding: 20px; border-radius: 12px; margin: 10px 0;
    }
    .mock-banner {
        background: #fff3cd; border: 1px solid #ffc107;
        padding: 10px 16px; border-radius: 8px; margin-bottom: 16px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state for caching analysis results ───────────────────────────────
@st.cache_data(show_spinner=False)
def run_full_analysis(data_path: str):
    """Run the full pipeline and cache results."""
    df = load_dataset(data_path)
    df = preprocess(df)
    eda_stats = get_eda_stats(df)

    df = run_theme_extraction(df, use_claude=True)
    theme_summary = get_theme_summary(df)
    impact_df = build_impact_table(df, theme_summary)
    impact_df = classify_issues(impact_df)

    df = cluster_reviews(df)
    trend_data = detect_trends(df)
    systemic_summary = get_systemic_summary(impact_df)

    current_rating = eda_stats.get("avg_rating", 3.0) or 3.0
    sim_df = run_full_simulation(impact_df, current_rating)

    roadmap = generate_roadmap_claude(impact_df, sim_df, eda_stats)
    exec_summary = get_executive_summary(eda_stats, impact_df, sim_df, systemic_summary)
    rating_segments = get_rating_segments(df)
    causal_chains = get_causal_chains(impact_df)
    portfolio = get_portfolio_impact(sim_df)

    return {
        "df": df,
        "eda_stats": eda_stats,
        "theme_summary": theme_summary,
        "impact_df": impact_df,
        "sim_df": sim_df,
        "roadmap": roadmap,
        "exec_summary": exec_summary,
        "trend_data": trend_data,
        "systemic_summary": systemic_summary,
        "rating_segments": rating_segments,
        "causal_chains": causal_chains,
        "portfolio": portfolio,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinsightAI")
    st.markdown("*AI-Driven Healthcare Review Intelligence*")
    st.divider()

    # Data source
    st.markdown("### Data Source")
    data_option = st.radio(
        "Choose data source",
        ["Use sample data (demo)", "Upload CSV file"],
        index=0,
    )

    data_path = "data/uploaded_reviews.csv"
    uploaded_file = None
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload hospital reviews CSV", type=["csv"])
        if uploaded_file:
            save_path = "data/uploaded_reviews.csv"
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            data_path = save_path

    st.divider()

    # Financial simulator controls
    st.markdown("### Financial Parameters")
    monthly_patients = st.slider(
        "Monthly Patients", 100, 5000, DEFAULT_MONTHLY_PATIENTS, 50
    )
    patient_ltv = st.slider(
        "Patient Lifetime Value ($)", 500, 10000, DEFAULT_PATIENT_LIFETIME_VALUE, 100
    )

    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Executive Summary", "Theme Analysis", "Impact Matrix",
         "Systemic Issues", "Financial Simulator", "Action Roadmap", "Raw Data"],
    )

    st.divider()
    if MOCK_MODE:
        st.warning("Running in **Demo Mode**\n\nAdd your Anthropic API key to `.env` to enable Claude AI.")
    else:
        st.success("Claude AI Active")


# ── Run analysis ─────────────────────────────────────────────────────────────
with st.spinner("Running AI analysis pipeline..."):
    results = run_full_analysis(data_path)

df          = results["df"]
eda_stats   = results["eda_stats"]
impact_df   = results["impact_df"]
sim_df      = results["sim_df"]
roadmap     = results["roadmap"]
exec_sum    = results["exec_summary"]
trend_data  = results["trend_data"]
sys_summary = results["systemic_summary"]
segments    = results["rating_segments"]
chains      = results["causal_chains"]
portfolio   = results["portfolio"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown('<div class="main-title">🏥 ClinsightAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Driven Healthcare Review Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("---")

    if MOCK_MODE:
        st.markdown(
            '<div class="mock-banner">⚠️ <b>Demo Mode:</b> Running with simulated AI. '
            'Add <code>ANTHROPIC_API_KEY</code> to <code>.env</code> for real Claude AI analysis.</div>',
            unsafe_allow_html=True,
        )

    # Top KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Reviews", f"{exec_sum['total_reviews']:,}")
    with col2:
        st.metric("Avg Rating", f"{exec_sum['avg_rating']} ⭐", delta=None)
    with col3:
        st.metric("Health Score", f"{exec_sum['health_score']}/100",
                  delta=exec_sum['health_label'])
    with col4:
        st.metric("Negative Reviews", f"{exec_sum['pct_negative']}%",
                  delta=f"-{exec_sum['pct_negative']}%", delta_color="inverse")
    with col5:
        st.metric("Revenue Opportunity",
                  f"${exec_sum['revenue_opportunity']:,}",
                  delta="Annual potential")

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
        for i, theme_label in enumerate(exec_sum["top_risk_themes"]):
            row = impact_df[impact_df["theme_label"] == theme_label]
            if not row.empty:
                r = row.iloc[0]
                severity = r["severity_score"]
                color = "🔴" if severity > 0.6 else "🟡" if severity > 0.3 else "🟢"
                st.markdown(f"{color} **{theme_label}** — {r['frequency_pct']}% of reviews, "
                             f"impact: {r['rating_impact']:+.2f} stars")

        if exec_sum["systemic_count"] > 0:
            st.error(f"⚠️ {exec_sum['systemic_count']} systemic issue(s) detected requiring immediate attention.")

    # Revenue opportunity banner
    if portfolio:
        st.markdown(f"""
        <div class="usp-banner">
            <h3>💰 Top Revenue Opportunity</h3>
            <p>Fixing your top 3 issues could deliver a
            <b>+{portfolio.get('combined_rating_lift', 0):.1f} star improvement</b>
            and generate up to <b>${portfolio.get('combined_revenue_impact', 0):,}/year</b>
            in retained patient revenue.</p>
            <p>Portfolio ROI: <b>{portfolio.get('portfolio_roi_pct', 0):.0f}%</b></p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THEME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Theme Analysis":
    st.header("Theme Analysis")
    st.caption("Frequency and severity of operational themes across all reviews")

    theme_summary = results["theme_summary"]
    if len(theme_summary) == 0:
        st.warning("No themes detected yet.")
        st.stop()

    # Top bar chart
    fig = px.bar(
        theme_summary.head(9),
        x="frequency_pct",
        y="theme_label",
        orientation="h",
        color="avg_severity",
        color_continuous_scale="RdYlGn_r",
        labels={"frequency_pct": "% of Reviews Mentioning Theme", "theme_label": "Theme",
                "avg_severity": "Avg Severity"},
        title="Theme Frequency & Severity",
    )
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Theme detail drill-down
    st.subheader("Theme Deep Dive")
    selected_theme_label = st.selectbox(
        "Select a theme to explore",
        options=theme_summary["theme_label"].tolist(),
    )

    selected_row = theme_summary[theme_summary["theme_label"] == selected_theme_label].iloc[0]
    impact_row = impact_df[impact_df["theme_label"] == selected_theme_label]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frequency", f"{selected_row['frequency_pct']}%")
    with col2:
        st.metric("Avg Severity", f"{selected_row['avg_severity']:.2f}/1.0")
    with col3:
        if not impact_row.empty:
            st.metric("Rating Impact", f"{impact_row.iloc[0]['rating_impact']:+.2f} ⭐")
    with col4:
        if not impact_row.empty:
            st.metric("Confidence", f"{impact_row.iloc[0]['confidence']:.0%}")

    # Evidence samples
    st.markdown("**Evidence Samples (real review quotes):**")
    for sample in selected_row.get("evidence_samples", []):
        st.markdown(f"> *\"{sample}\"*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMPACT MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Matrix":
    st.header("Impact Matrix")
    st.caption("Bubble chart: X = frequency, Y = rating impact, Size = severity score")

    if len(impact_df) == 0:
        st.warning("No impact data available.")
        st.stop()

    plot_df = impact_df.copy()
    plot_df["bubble_size"] = (plot_df["severity_score"] * 60 + 10).clip(10, 80)
    plot_df["color_val"] = plot_df["rating_impact"]

    fig = px.scatter(
        plot_df,
        x="frequency_pct",
        y="rating_impact",
        size="bubble_size",
        color="rating_impact",
        color_continuous_scale="RdYlGn",
        text="theme_label",
        hover_data={"frequency_pct": True, "rating_impact": True,
                    "severity_score": True, "confidence": True, "bubble_size": False},
        title="Operational Theme Impact Matrix",
        labels={
            "frequency_pct": "Theme Frequency (% of Reviews)",
            "rating_impact": "Rating Impact (stars)",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 1-star vs 5-star theme comparison
    st.subheader("What Separates 1-Star from 5-Star Reviews?")
    if segments:
        low = segments.get("low_rating_themes", {})
        high = segments.get("high_rating_themes", {})
        themes_union = list(set(list(low.keys()) + list(high.keys())))

        seg_data = []
        for t in themes_union:
            seg_data.append({"Theme": THEME_LABELS.get(t, t), "Type": "⭐ 1-2 Star Reviews", "Frequency %": low.get(t, 0)})
            seg_data.append({"Theme": THEME_LABELS.get(t, t), "Type": "⭐⭐⭐⭐⭐ 4-5 Star Reviews", "Frequency %": high.get(t, 0)})

        seg_df = pd.DataFrame(seg_data)
        fig2 = px.bar(
            seg_df, x="Theme", y="Frequency %", color="Type",
            barmode="group",
            color_discrete_map={"⭐ 1-2 Star Reviews": "#e63946", "⭐⭐⭐⭐⭐ 4-5 Star Reviews": "#2a9d8f"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Ranked table
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
    st.caption("Distinguishing widespread systemic problems from isolated incidents")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🔴 Systemic Issues: **{len(sys_summary['systemic_issues'])}**")
        for item in sys_summary["systemic_issues"]:
            st.markdown(f"- {item}")
    with col2:
        st.warning(f"🟡 Moderate Issues: **{len(sys_summary['moderate_issues'])}**")
        for item in sys_summary["moderate_issues"]:
            st.markdown(f"- {item}")
    with col3:
        st.success(f"🟢 Isolated Issues: **{len(sys_summary['isolated_issues'])}**")
        for item in sys_summary["isolated_issues"]:
            st.markdown(f"- {item}")

    st.markdown("---")

    # Escalation risk scores
    st.subheader("Escalation Risk Ranking")
    classified = sys_summary["classified_df"]
    if len(classified) > 0:
        fig = px.bar(
            classified.sort_values("escalation_score", ascending=True).tail(8),
            x="escalation_score",
            y="theme_label",
            orientation="h",
            color="issue_class",
            color_discrete_map={"SYSTEMIC": "#e63946", "MODERATE": "#f4a261", "ISOLATED": "#2a9d8f"},
            labels={"escalation_score": "Escalation Risk Score", "theme_label": "Theme"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Trend analysis
    st.subheader("Trend Analysis Over Time")
    if trend_data.get("available"):
        rising = trend_data.get("rising_themes", [])
        if rising:
            st.warning(f"📈 Rising complaint trends detected: **{', '.join([THEME_LABELS.get(t, t) for t in rising])}**")

        # Plot top theme trends
        trends = trend_data.get("trends", {})
        trend_plot_themes = sorted(trends.keys(), key=lambda t: abs(trends[t]["slope"]), reverse=True)[:4]

        if trend_plot_themes:
            fig_trend = go.Figure()
            for theme in trend_plot_themes:
                t_data = trends[theme]["monthly_data"]
                months = [d[0] for d in t_data]
                freqs = [d[1] for d in t_data]
                fig_trend.add_trace(go.Scatter(
                    x=months, y=freqs,
                    mode="lines+markers",
                    name=THEME_LABELS.get(theme, theme),
                ))
            fig_trend.update_layout(
                title="Theme Frequency Over Time",
                xaxis_title="Month", yaxis_title="% of Reviews",
                height=350,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Time-series trend analysis requires a 'date' column in the dataset.")

    # Causal chains
    st.subheader("Causal Chain Analysis")
    st.caption("How operational failures cascade into revenue loss")
    for chain in chains:
        with st.expander(f"🔗 {chain['theme_label']} — Rating Impact: {chain['rating_impact']:+.2f} ⭐"):
            st.markdown(f"""
            ```
            {chain['root_cause']}
                └──→ {chain['operational_failure']}
                        └──→ {chain['patient_experience']}
                                └──→ {chain['review_outcome']}
                                        └──→ Rating Impact: {chain['rating_impact']:+.2f} stars
                                                └──→ Appears in {chain['frequency_pct']:.1f}% of reviews
            ```
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL SIMULATOR (USP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Financial Simulator":
    st.header("💰 Financial Impact Simulator")
    st.markdown("""
    <div class="usp-banner">
        <b>Rating-to-Revenue Bridge</b> — Convert abstract star ratings into concrete dollar impact.
        See exactly how much revenue each operational fix is worth to your hospital.
    </div>
    """, unsafe_allow_html=True)

    current_rating = eda_stats.get("avg_rating", 3.0) or 3.0

    st.markdown("### Configure Your Hospital")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_patients = st.number_input("Monthly Patients", 100, 50000, monthly_patients)
    with col2:
        sim_ltv = st.number_input("Patient Lifetime Value ($)", 100, 20000, patient_ltv)
    with col3:
        st.metric("Current Average Rating", f"{current_rating:.2f} ⭐")

    # Run simulation with user parameters
    sim_results = run_full_simulation(impact_df, current_rating, sim_patients, sim_ltv)

    if len(sim_results) == 0:
        st.warning("No simulation data available.")
        st.stop()

    # Revenue opportunity chart
    st.markdown("### Revenue Opportunity by Theme Fix")
    fig = px.bar(
        sim_results.head(8),
        x="theme_label",
        y="revenue_impact_annual",
        color="effort_level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#e63946"},
        labels={"revenue_impact_annual": "Annual Revenue Impact ($)", "theme_label": "Theme"},
        text="revenue_impact_annual",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # What-If calculator
    st.markdown("### What-If Calculator")
    col_select, col_result = st.columns([1, 2])

    with col_select:
        available_themes = [THEME_LABELS[t] for t in THEMES if f"theme_{t}" in df.columns]
        selected_label = st.selectbox("Select theme to fix", available_themes)
        selected_theme = [t for t, l in THEME_LABELS.items() if l == selected_label][0]

    scenario = calculate_revenue_impact(
        monthly_patients=sim_patients,
        patient_lifetime_value=sim_ltv,
        current_rating=current_rating,
        target_theme=selected_theme,
        impact_df=impact_df,
    )

    with col_result:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rating Improvement",
                      f"+{scenario['rating_lift']:.2f} ⭐",
                      f"{scenario['current_rating']:.1f} → {scenario['projected_rating']:.1f}")
        with c2:
            st.metric("Patient Retention Gain",
                      f"+{scenario['retention_gain_pct']:.1f}%",
                      f"+{scenario['retained_patients_annual']:,} patients/yr")
        with c3:
            st.metric("Annual Revenue Impact",
                      f"${scenario['revenue_impact_annual']:,}",
                      f"ROI: {scenario['roi_pct']:.0f}%" if scenario['roi_pct'] else "")

    # Cost vs. return breakdown
    st.markdown("**Cost vs. Return Analysis**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        | Item | Value |
        |------|-------|
        | Estimated Fix Cost | ${scenario['fix_cost_low']:,} – ${scenario['fix_cost_high']:,}/yr |
        | Fix Description | {scenario['fix_description']} |
        | Payback Period | {scenario['payback_months']:.1f} months |
        | Effort Level | {scenario['effort_level']} |
        | Confidence | {scenario['confidence']:.0%} |
        """)
    with col_b:
        # Gauge chart for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scenario['confidence'] * 100,
            title={"text": "Confidence Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4361ee"},
                "steps": [
                    {"range": [0, 50], "color": "#ffcdd2"},
                    {"range": [50, 75], "color": "#fff9c4"},
                    {"range": [75, 100], "color": "#c8e6c9"},
                ],
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Portfolio view
    portfolio_live = get_portfolio_impact(sim_results)
    if portfolio_live:
        st.markdown("### Combined Portfolio Impact (Top 3 Fixes)")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Rating Lift", f"+{portfolio_live['combined_rating_lift']:.1f} ⭐")
        with p2:
            st.metric("Revenue Impact", f"${portfolio_live['combined_revenue_impact']:,}/yr")
        with p3:
            st.metric("Total Fix Cost", portfolio_live['total_fix_cost_range'])
        with p4:
            st.metric("Portfolio ROI", f"{portfolio_live['portfolio_roi_pct']:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ACTION ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Action Roadmap":
    st.header("Action Roadmap")
    st.caption("Prioritized improvement plan generated by AI analysis")

    if MOCK_MODE:
        st.info("In demo mode, roadmap is generated by rule-based logic. "
                "Add Claude API key for AI-tailored recommendations.")

    tabs = st.tabs(["⚡ Quick Wins", "🎯 Strategic", "🔭 Long-term"])

    category_map = {
        "⚡ Quick Wins": roadmap.get("quick_wins", []),
        "🎯 Strategic": roadmap.get("strategic", []),
        "🔭 Long-term": roadmap.get("long_term", []),
    }

    effort_colors = {"Low": "green", "Medium": "orange", "High": "red"}

    for tab, (tab_label, items) in zip(tabs, category_map.items()):
        with tab:
            if not items:
                st.info("No items in this category.")
                continue
            for item in items[:5]:  # Show top 5 per category
                with st.expander(
                    f"#{item.get('priority', '?')} — {item.get('theme_label', item.get('category', 'Action'))} "
                    f"| Effort: {item.get('effort', 'N/A')} | Lift: {item.get('expected_lift', 'TBD')}",
                    expanded=(item.get("priority", 99) == 1),
                ):
                    st.markdown(f"**Action:** {item.get('action', '')}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Timeline:** {item.get('timeline', 'N/A')}")
                        st.markdown(f"**Cost:** {item.get('cost_tier', 'N/A')}")
                    with col2:
                        st.markdown(f"**Expected Lift:** {item.get('expected_lift', 'N/A')}")
                        if item.get("revenue_impact"):
                            st.markdown(f"**Revenue Impact:** ${item['revenue_impact']:,}/yr")
                    with col3:
                        kpis = item.get("kpis", [])
                        if kpis:
                            st.markdown("**KPIs to Track:**")
                            for kpi in kpis:
                                st.markdown(f"- {kpi}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Raw Data":
    st.header("Raw Review Data")
    st.caption("Browse and filter reviews with AI-detected theme tags")

    # Filters
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

    display_df = filtered[["review_text", "rating", "hospital_name", "overall_sentiment", "risk_cluster"]].copy()
    display_df.columns = ["Review", "Rating", "Hospital", "Sentiment", "Risk Cluster"]
    st.dataframe(display_df, use_container_width=True, height=500)
