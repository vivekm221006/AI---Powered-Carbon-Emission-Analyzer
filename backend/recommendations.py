"""
AI recommendation engine and carbon reduction simulator.
"""

from .carbon_calculator import calculate_emissions


def get_recommendations(
    electricity_kwh: float,
    fuel_liters: float,
    travel_km: float,
    cloud_hours: float,
    waste_kg: float,
    breakdown: dict,
) -> list[dict]:
    """
    Generate AI recommendations based on emission category breakdown.

    Each recommendation dict contains:
        issue, value, suggestions (list[str]), potential_reduction, priority
    """
    total = sum(breakdown.values())
    if total == 0:
        return []

    # For organisations with very low absolute emissions return positive feedback directly.
    if total < 2.0:
        return [
            {
                "issue": "✅ Strong Sustainability Performance",
                "value": (
                    f"Your organisation produces only {total:.3f} tons CO₂/month — "
                    "an excellent result."
                ),
                "suggestions": [
                    "Continue monthly monitoring and publish annual sustainability reports.",
                    "Set science-based reduction targets (SBTi) for the next 5 years.",
                    "Engage suppliers in your supply-chain decarbonisation journey.",
                    "Consider verified carbon offset programmes for remaining emissions.",
                ],
                "potential_reduction": "10–20%",
                "priority": "LOW",
            }
        ]

    pct = {k: v / total * 100 for k, v in breakdown.items()}
    recommendations = []

    if pct.get("Electricity", 0) > 25:
        recommendations.append(
            {
                "issue": "⚡ High Electricity Usage",
                "value": (
                    f"{electricity_kwh:,.0f} kWh/month "
                    f"({pct['Electricity']:.1f}% of emissions)"
                ),
                "suggestions": [
                    "Install rooftop solar panels or purchase renewable energy certificates (RECs).",
                    "Replace all lighting with LED fixtures and add occupancy sensors.",
                    "Deploy smart building automation (HVAC scheduling, smart thermostats).",
                    "Enforce power-management policies on all workstations and servers.",
                ],
                "potential_reduction": "25–40%",
                "priority": "HIGH" if pct["Electricity"] > 40 else "MEDIUM",
            }
        )

    if pct.get("Fuel", 0) > 15:
        recommendations.append(
            {
                "issue": "⛽ High Fuel Consumption",
                "value": (
                    f"{fuel_liters:,.0f} litres/month "
                    f"({pct['Fuel']:.1f}% of emissions)"
                ),
                "suggestions": [
                    "Transition the company fleet to electric or hybrid vehicles.",
                    "Implement eco-driving training programmes for drivers.",
                    "Optimise delivery and logistics routes using route-planning software.",
                    "Introduce carpooling and micro-mobility incentive schemes.",
                ],
                "potential_reduction": "30–60%",
                "priority": "HIGH" if pct["Fuel"] > 25 else "MEDIUM",
            }
        )

    if pct.get("Travel", 0) > 20:
        recommendations.append(
            {
                "issue": "✈️ Excessive Business Travel",
                "value": (
                    f"{travel_km:,.0f} km/month "
                    f"({pct['Travel']:.1f}% of emissions)"
                ),
                "suggestions": [
                    "Replace short-haul flights with high-speed rail or video conferences.",
                    "Set a per-employee annual carbon travel budget.",
                    "Implement a travel approval process with sustainability criteria.",
                    "Offset unavoidable flights through verified carbon credit programmes.",
                ],
                "potential_reduction": "40–70%",
                "priority": "HIGH" if pct["Travel"] > 35 else "MEDIUM",
            }
        )

    if pct.get("Cloud/Data Centers", 0) > 10:
        recommendations.append(
            {
                "issue": "☁️ High Cloud / Data-Center Usage",
                "value": (
                    f"{cloud_hours:,.0f} server-hours/month "
                    f"({pct['Cloud/Data Centers']:.1f}% of emissions)"
                ),
                "suggestions": [
                    "Migrate workloads to cloud providers committed to renewable energy (availability varies by region; verify current commitments).",
                    "Right-size instances and use auto-scaling to eliminate idle compute.",
                    "Adopt serverless / container architectures to improve utilisation.",
                    "Implement data lifecycle policies to purge unused storage.",
                ],
                "potential_reduction": "20–50%",
                "priority": "HIGH" if pct["Cloud/Data Centers"] > 20 else "MEDIUM",
            }
        )

    if pct.get("Office Waste", 0) > 10:
        recommendations.append(
            {
                "issue": "🗑️ High Office Waste",
                "value": (
                    f"{waste_kg:,.0f} kg/month "
                    f"({pct['Office Waste']:.1f}% of emissions)"
                ),
                "suggestions": [
                    "Roll out a comprehensive recycling and composting programme.",
                    "Go paperless — digitalise forms, invoices, and communications.",
                    "Adopt a zero-single-use-plastic policy for the office.",
                    "Partner with upcycling organisations to redirect surplus materials.",
                ],
                "potential_reduction": "30–50%",
                "priority": "MEDIUM",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "issue": "✅ Strong Sustainability Performance",
                "value": "Your organisation already has relatively low carbon emissions.",
                "suggestions": [
                    "Continue monthly monitoring and publish annual sustainability reports.",
                    "Set science-based reduction targets (SBTi) for the next 5 years.",
                    "Engage suppliers in your supply-chain decarbonisation journey.",
                    "Consider verified carbon offset programmes for remaining emissions.",
                ],
                "potential_reduction": "10–20%",
                "priority": "LOW",
            }
        )

    return recommendations


def simulate_reduction(
    electricity_kwh: float,
    fuel_liters: float,
    travel_km: float,
    cloud_hours: float,
    waste_kg: float,
    solar_pct: float = 0,
    ev_pct: float = 0,
    remote_work_pct: float = 0,
    green_cloud_pct: float = 0,
    waste_reduction_pct: float = 0,
) -> tuple[float, dict]:
    """
    Simulate the effect of sustainability interventions on monthly CO2 emissions.

    Parameters (all percentages 0–100):
        solar_pct          — share of electricity covered by solar / renewables
        ev_pct             — share of fuel replaced by EV / zero-emission vehicles
        remote_work_pct    — share of business travel avoided via remote work
        green_cloud_pct    — share of servers migrated to 100 % renewable cloud
                             (green cloud halves the emission factor)
        waste_reduction_pct — share of office waste eliminated

    Returns:
        (new_total_co2_tons, new_breakdown_dict)
    """
    new_electricity = electricity_kwh * (1 - solar_pct / 100)
    new_fuel = fuel_liters * (1 - ev_pct / 100)
    new_travel = travel_km * (1 - remote_work_pct / 100)
    # Green cloud providers typically use ~50 % of the emissions of conventional ones
    green_fraction = green_cloud_pct / 100
    effective_cloud = cloud_hours * (1 - green_fraction * 0.5)
    new_waste = waste_kg * (1 - waste_reduction_pct / 100)

    return calculate_emissions(new_electricity, new_fuel, new_travel, effective_cloud, new_waste)
