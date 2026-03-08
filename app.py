"""
AI-Powered Carbon Emission Analyzer — Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.carbon_calculator import calculate_emissions, get_carbon_rating, EMISSION_FACTORS
from backend.model import CarbonEmissionModel
from backend.recommendations import get_recommendations, simulate_reduction
from backend.report_generator import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Emission Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .rating-box {
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            color: white;
            margin-bottom: 10px;
        }
        .metric-card {
            background: #F1F8E9;
            border-left: 4px solid #388E3C;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0;
        }
        .section-header {
            color: #2E7D32;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .rec-card {
            background: #FAFAFA;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 8px 0;
        }
        .priority-HIGH   { color: #D32F2F; font-weight: bold; }
        .priority-MEDIUM { color: #F57C00; font-weight: bold; }
        .priority-LOW    { color: #388E3C; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached model initialisation ───────────────────────────────────────────────
@st.cache_resource
def load_model(model_type: str) -> CarbonEmissionModel:
    m = CarbonEmissionModel(model_type=model_type)
    m.train()
    return m


# ── Chatbot responses ─────────────────────────────────────────────────────────
CHATBOT_RESPONSES = {
    "solar": (
        "☀️ Installing solar panels can reduce your electricity-related emissions by 25–40 %. "
        "Consider starting with a roof survey to estimate generation potential."
    ),
    "electric vehicle": (
        "🚗 Switching to an EV fleet can cut fuel-related CO₂ by up to 70 %. "
        "Look for government incentives and workplace charging infrastructure grants."
    ),
    "ev": (
        "🚗 Switching to an EV fleet can cut fuel-related CO₂ by up to 70 %. "
        "Look for government incentives and workplace charging infrastructure grants."
    ),
    "remote": (
        "💻 Encouraging remote work and virtual meetings can reduce business-travel "
        "emissions by 40–70 %. Tools like video conferencing can replace most short-haul trips."
    ),
    "cloud": (
        "☁️ Migrating to green cloud providers (e.g. Google Cloud, Azure with renewables) "
        "can halve your data-centre carbon footprint. Also consider right-sizing instances."
    ),
    "waste": (
        "♻️ Implementing recycling, composting, and paperless policies can reduce "
        "office-waste emissions by 30–50 %. Start with a waste audit to identify the biggest sources."
    ),
    "rating": (
        "📊 Carbon ratings:\n"
        "  • A — Sustainable  (< 5 tons CO₂/month)\n"
        "  • B — Moderate     (5–15 tons)\n"
        "  • C — High         (15–30 tons)\n"
        "  • D — Critical     (≥ 30 tons)"
    ),
    "reduce": (
        "🌱 Top strategies to reduce emissions:\n"
        "  1. Switch to renewable energy (solar, wind)\n"
        "  2. Electrify the company vehicle fleet\n"
        "  3. Replace air travel with virtual meetings\n"
        "  4. Migrate to green cloud providers\n"
        "  5. Implement waste-reduction programmes"
    ),
    "offset": (
        "🌳 Carbon offsets let you compensate for unavoidable emissions by funding projects "
        "such as reforestation or renewable energy in developing regions. "
        "Look for Gold Standard or Verified Carbon Standard (VCS) certified credits."
    ),
    "electricity": (
        "⚡ Electricity is often the largest emission source. Actions:\n"
        "  • Purchase renewable energy certificates (RECs)\n"
        "  • Install on-site solar\n"
        "  • Deploy smart energy-management systems"
    ),
    "travel": (
        "✈️ Business travel is a significant source of CO₂. "
        "Set a per-employee carbon travel budget, prefer trains over planes for trips < 500 km, "
        "and use video conferencing for routine meetings."
    ),
    "scope": (
        "📋 The GHG Protocol classifies emissions into three scopes:\n"
        "  • Scope 1 — Direct emissions (fuel burned on-site, company vehicles)\n"
        "  • Scope 2 — Indirect from purchased electricity/heat\n"
        "  • Scope 3 — All other indirect (supply chain, employee commuting, business travel)"
    ),
    "target": (
        "🎯 Setting science-based targets (SBTi) means aligning your emission reductions "
        "with the Paris Agreement goal of limiting warming to 1.5 °C. "
        "Visit sciencebasedtargets.org to get started."
    ),
}

DEFAULT_CHATBOT_REPLY = (
    "🤖 I can help with questions about carbon emissions, reduction strategies, "
    "ratings, cloud usage, EVs, solar energy, offsets, and more. "
    "Try asking: 'How can we reduce our carbon emissions?' or 'What is a carbon rating?'"
)


def chatbot_reply(user_input: str) -> str:
    query = user_input.lower()
    for keyword, response in CHATBOT_RESPONSES.items():
        if keyword in query:
            return response
    return DEFAULT_CHATBOT_REPLY


# ── Sidebar — Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/co2.png",
        width=64,
    )
    st.title("🌍 Carbon Analyzer")
    st.markdown("**Enter your organisation's monthly data:**")

    org_name = st.text_input("Organisation Name", value="My Organisation")

    st.markdown("---")
    electricity_kwh = st.number_input(
        "⚡ Electricity (kWh/month)", min_value=0.0, value=5_000.0, step=100.0,
        help="Total monthly electricity consumption in kilowatt-hours.",
    )
    fuel_liters = st.number_input(
        "⛽ Fuel Consumption (litres/month)", min_value=0.0, value=800.0, step=50.0,
        help="Total monthly diesel/petrol consumed by company vehicles.",
    )
    travel_km = st.number_input(
        "✈️ Business Travel (km/month)", min_value=0.0, value=12_000.0, step=500.0,
        help="Total monthly distance flown / driven for business travel.",
    )
    cloud_hours = st.number_input(
        "☁️ Cloud / Server Hours (hrs/month)", min_value=0.0, value=300.0, step=10.0,
        help="Total monthly cloud or on-premise server compute hours.",
    )
    waste_kg = st.number_input(
        "🗑️ Office Waste (kg/month)", min_value=0.0, value=200.0, step=10.0,
        help="Total monthly office waste generated in kilograms.",
    )

    st.markdown("---")
    model_type = st.selectbox(
        "🤖 AI Model",
        options=["random_forest", "gradient_boosting", "linear_regression"],
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "linear_regression": "Linear Regression",
        }[x],
    )

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyse Emissions", type="primary", use_container_width=True)

# ── Compute on button press or first load ─────────────────────────────────────
if "results" not in st.session_state or analyze_btn:
    total_co2, breakdown = calculate_emissions(
        electricity_kwh, fuel_liters, travel_km, cloud_hours, waste_kg
    )
    rating = get_carbon_rating(total_co2)
    recommendations = get_recommendations(
        electricity_kwh, fuel_liters, travel_km, cloud_hours, waste_kg, breakdown
    )
    model = load_model(model_type)
    ml_prediction = model.predict(electricity_kwh, fuel_liters, travel_km, cloud_hours, waste_kg)
    feature_importance = model.get_feature_importance()
    model_metrics = model.metrics

    st.session_state["results"] = {
        "total_co2": total_co2,
        "breakdown": breakdown,
        "rating": rating,
        "recommendations": recommendations,
        "ml_prediction": ml_prediction,
        "feature_importance": feature_importance,
        "model_metrics": model_metrics,
        "inputs": {
            "electricity_kwh": electricity_kwh,
            "fuel_liters": fuel_liters,
            "travel_km": travel_km,
            "cloud_hours": cloud_hours,
            "waste_kg": waste_kg,
        },
        "org_name": org_name,
        "model_type": model_type,
    }

res = st.session_state["results"]
total_co2 = res["total_co2"]
breakdown = res["breakdown"]
rating_letter, rating_label, rating_color = res["rating"]
recommendations = res["recommendations"]
ml_prediction = res["ml_prediction"]
feature_importance = res["feature_importance"]
model_metrics = res["model_metrics"]
inputs = res["inputs"]

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("# 🌍 AI-Powered Carbon Emission Analyzer")
st.markdown(
    "Analyse your organisation's carbon footprint, get AI-driven recommendations, "
    "simulate reduction scenarios, and export a professional PDF report."
)

# ── Summary KPI row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Monthly CO₂", f"{total_co2:.2f} t", help="Total monthly CO₂ in tons")
col2.metric("📆 Annual CO₂", f"{total_co2 * 12:.1f} t/yr")
col3.metric("🤖 ML Prediction", f"{ml_prediction:.2f} t", f"{ml_prediction - total_co2:+.2f} t vs calculated")
col4.markdown(
    f'<div class="rating-box" style="background:{rating_color}">'
    f"Grade {rating_letter}<br><span style='font-size:0.9rem'>{rating_label}</span></div>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_ai, tab_sim, tab_rec, tab_chat, tab_report = st.tabs(
    [
        "📊 Dashboard",
        "🤖 AI Analysis",
        "🌱 Simulator",
        "💡 Recommendations",
        "🤖 Chatbot",
        "📄 Report",
    ]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("📊 Emission Breakdown")

    c1, c2 = st.columns(2)

    # Pie chart
    with c1:
        fig_pie = px.pie(
            names=list(breakdown.keys()),
            values=list(breakdown.values()),
            title="CO₂ Emission by Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.35,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart
    with c2:
        fig_bar = px.bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            labels={"x": "Category", "y": "CO₂ (tons/month)"},
            title="Monthly CO₂ per Category",
            color=list(breakdown.values()),
            color_continuous_scale="RdYlGn_r",
            text_auto=".3f",
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Monthly trend simulation (12-month synthetic trend)
    st.subheader("📈 Simulated Monthly Emission Trend")
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    rng = np.random.default_rng(42)
    seasonal = np.array([1.1, 1.05, 1.0, 0.95, 0.92, 0.88, 0.9, 0.93, 1.0, 1.05, 1.1, 1.12])
    trend_vals = total_co2 * seasonal * (1 + rng.normal(0, 0.03, 12))

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=months,
            y=trend_vals,
            mode="lines+markers",
            name="CO₂ (tons)",
            line=dict(color="#388E3C", width=2.5),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(56,142,60,0.1)",
        )
    )
    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="CO₂ (tons)",
        title="12-Month Emission Trend",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Breakdown table
    st.subheader("📋 Detailed Breakdown Table")
    df_bd = pd.DataFrame(
        {
            "Category": list(breakdown.keys()),
            "CO₂ (tons/month)": [f"{v:.4f}" for v in breakdown.values()],
            "Share (%)": [f"{v / (total_co2 or 1) * 100:.1f}%" for v in breakdown.values()],
            "Annual CO₂ (tons)": [f"{v * 12:.2f}" for v in breakdown.values()],
        }
    )
    st.dataframe(df_bd, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🤖 Machine Learning Prediction")

    m1, m2, m3 = st.columns(3)
    m1.metric("Calculated Emissions", f"{total_co2:.3f} t CO₂/month")
    m2.metric("ML Predicted Emissions", f"{ml_prediction:.3f} t CO₂/month")
    m3.metric(
        "Model R² Score",
        f"{model_metrics.get('r2_score', 0):.4f}",
        help="1.0 = perfect fit",
    )

    st.info(
        f"**Model:** {model_metrics.get('model_type', model_type).replace('_', ' ').title()}  |  "
        f"**RMSE:** {model_metrics.get('rmse', 0):.4f} tons  |  "
        f"**R²:** {model_metrics.get('r2_score', 0):.4f}"
    )

    # Feature importance
    if feature_importance:
        st.subheader("🔍 Feature Importance")
        fi_df = pd.DataFrame(
            {
                "Feature": list(feature_importance.keys()),
                "Importance": list(feature_importance.values()),
            }
        ).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance (how much each input drives the prediction)",
            color="Importance",
            color_continuous_scale="Greens",
            text_auto=".3f",
        )
        fig_fi.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Calculated vs predicted gauge
    st.subheader("⚖️ Calculated vs ML Prediction")
    fig_gauge = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
    )
    fig_gauge.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=total_co2,
            title={"text": "Calculated"},
            gauge={"axis": {"range": [0, max(total_co2, ml_prediction) * 1.5]}, "bar": {"color": "#388E3C"}},
        ),
        row=1, col=1,
    )
    fig_gauge.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=ml_prediction,
            title={"text": "ML Predicted"},
            gauge={"axis": {"range": [0, max(total_co2, ml_prediction) * 1.5]}, "bar": {"color": "#1565C0"}},
        ),
        row=1, col=2,
    )
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Model comparison
    st.subheader("🔬 Model Comparison")
    all_models = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Linear Regression": "linear_regression",
    }
    comparison_data = []
    with st.spinner("Training all models for comparison…"):
        for display_name, mt in all_models.items():
            m_obj = load_model(mt)
            comparison_data.append(
                {
                    "Model": display_name,
                    "R² Score": f"{m_obj.metrics.get('r2_score', 0):.4f}",
                    "RMSE (tons)": f"{m_obj.metrics.get('rmse', 0):.4f}",
                    "Prediction": f"{m_obj.predict(**inputs):.3f} t",
                }
            )
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Carbon Reduction Simulator
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("🌱 Carbon Reduction Simulator")
    st.markdown(
        "Adjust the sliders below to model how different sustainability interventions "
        "would reduce your organisation's carbon footprint."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        solar_pct = st.slider("☀️ Renewable Energy Adoption (%)", 0, 100, 0, 5,
                              help="Share of electricity replaced by solar / wind.")
        ev_pct = st.slider("🚗 EV Fleet Transition (%)", 0, 100, 0, 5,
                           help="Share of fuel-based vehicles replaced by EVs.")
        remote_work_pct = st.slider("💻 Remote Work / Virtual Meetings (%)", 0, 100, 0, 5,
                                    help="Share of business travel replaced by remote work.")
    with sc2:
        green_cloud_pct = st.slider("☁️ Green Cloud Migration (%)", 0, 100, 0, 5,
                                    help="Share of servers migrated to renewable-energy cloud providers.")
        waste_reduction_pct = st.slider("♻️ Waste Reduction Programme (%)", 0, 100, 0, 5,
                                        help="Share of office waste eliminated.")

    new_total, new_breakdown = simulate_reduction(
        **inputs,
        solar_pct=solar_pct,
        ev_pct=ev_pct,
        remote_work_pct=remote_work_pct,
        green_cloud_pct=green_cloud_pct,
        waste_reduction_pct=waste_reduction_pct,
    )
    saved = total_co2 - new_total
    saved_pct = (saved / total_co2 * 100) if total_co2 > 0 else 0

    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Original Emissions", f"{total_co2:.3f} t CO₂/month")
    sm2.metric("After Interventions", f"{new_total:.3f} t CO₂/month", f"{-saved:.3f} t saved")
    sm3.metric("CO₂ Reduction", f"{saved_pct:.1f}%", help="Percentage reduction vs baseline")

    # Before / after bar comparison
    cat_labels = list(breakdown.keys())
    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Bar(name="Before", x=cat_labels, y=list(breakdown.values()), marker_color="#EF5350")
    )
    fig_sim.add_trace(
        go.Bar(name="After", x=cat_labels, y=list(new_breakdown.values()), marker_color="#66BB6A")
    )
    fig_sim.update_layout(
        barmode="group",
        title="Emission Comparison Before vs After Interventions",
        xaxis_title="Category",
        yaxis_title="CO₂ (tons/month)",
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # Annual saving context
    if saved > 0:
        trees = saved * 12 * 45  # ~45 trees absorb 1 ton CO2/year
        st.success(
            f"🌳 Your interventions would save **{saved * 12:.1f} tons CO₂/year** — "
            f"equivalent to planting approximately **{trees:,.0f} trees**!"
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Recommendations
# ════════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.subheader("💡 AI-Driven Sustainability Recommendations")

    if not recommendations:
        st.info("No specific recommendations generated. Enter your data and click 'Analyse Emissions'.")
    else:
        for rec in recommendations:
            priority_class = f"priority-{rec['priority']}"
            with st.expander(f"{rec['issue']}", expanded=True):
                st.markdown(f"**Status:** {rec['value']}")
                st.markdown(
                    f"**Priority:** <span class='{priority_class}'>{rec['priority']}</span> &nbsp;|&nbsp; "
                    f"**Potential Reduction:** {rec['potential_reduction']}",
                    unsafe_allow_html=True,
                )
                st.markdown("**Suggested Actions:**")
                for s in rec["suggestions"]:
                    st.markdown(f"- {s}")

    # Rating explanation
    st.markdown("---")
    st.subheader("🏷️ Carbon Score Rating Guide")
    rating_data = {
        "Grade": ["A", "B", "C", "D"],
        "Label": ["🌱 Sustainable", "🌿 Moderate", "⚠️ High Emissions", "🚨 Critical"],
        "Monthly CO₂ Range": ["< 5 tons", "5 – 15 tons", "15 – 30 tons", "≥ 30 tons"],
        "Action Required": [
            "Maintain & improve", "Set reduction targets", "Urgent action needed", "Immediate overhaul"
        ],
    }
    st.dataframe(pd.DataFrame(rating_data), use_container_width=True, hide_index=True)

    current_rating, current_label, current_color = res["rating"]
    st.markdown(
        f'<div class="rating-box" style="background:{current_color}; display:inline-block; '
        f'padding:12px 32px; border-radius:12px;">'
        f"Your Grade: <strong>{current_rating}</strong> — {current_label}</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI Chatbot
# ════════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("🤖 AI Sustainability Chatbot")
    st.markdown(
        "Ask questions about carbon emissions, reduction strategies, carbon ratings, or sustainability best practices."
    )

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm your AI Sustainability Advisor. "
                    "Ask me anything about carbon emissions, EVs, solar energy, offsets, "
                    "carbon ratings, remote work, or cloud sustainability."
                ),
            }
        ]

    # Display chat messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_msg := st.chat_input("Ask a sustainability question…"):
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        reply = chatbot_reply(user_msg)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Quick-question buttons
    st.markdown("**Quick questions:**")
    quick_cols = st.columns(3)
    quick_qs = [
        "How can we reduce carbon emissions?",
        "What is a carbon rating?",
        "Tell me about solar panels",
        "How do EVs help?",
        "What are carbon offsets?",
        "Explain Scope 1, 2, 3",
    ]
    for i, q in enumerate(quick_qs):
        if quick_cols[i % 3].button(q, key=f"quick_{i}"):
            st.session_state["chat_history"].append({"role": "user", "content": q})
            reply = chatbot_reply(q)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Report Export
# ════════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.subheader("📄 Export PDF Report")
    st.markdown(
        "Generate and download a professional PDF report summarising your organisation's "
        "carbon footprint analysis and AI recommendations."
    )

    report_org = st.text_input("Organisation name for the report", value=res.get("org_name", "My Organisation"))

    if st.button("📥 Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating PDF report…"):
            pdf_buffer = generate_pdf_report(
                org_name=report_org,
                inputs=inputs,
                total_co2=total_co2,
                breakdown=breakdown,
                rating=res["rating"],
                recommendations=recommendations,
            )
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"carbon_report_{report_org.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
        st.success("✅ Report generated successfully! Click the button above to download.")

    # Report preview as table
    st.markdown("---")
    st.subheader("📋 Report Preview")

    st.markdown("**Summary**")
    summary_df = pd.DataFrame(
        {
            "Metric": ["Monthly CO₂", "Annual CO₂ Estimate", "Carbon Rating"],
            "Value": [
                f"{total_co2:.3f} tons CO₂",
                f"{total_co2 * 12:.2f} tons CO₂/year",
                f"{rating_letter} — {rating_label}",
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("**Emission Breakdown**")
    total_for_pct = total_co2 or 1
    bd_preview = pd.DataFrame(
        {
            "Category": list(breakdown.keys()),
            "CO₂ (tons/month)": [f"{v:.4f}" for v in breakdown.values()],
            "Share": [f"{v / total_for_pct * 100:.1f}%" for v in breakdown.values()],
        }
    )
    st.dataframe(bd_preview, use_container_width=True, hide_index=True)
