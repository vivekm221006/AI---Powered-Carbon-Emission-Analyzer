"""
Carbon emission factor calculations and scoring.
"""

# Emission factors (tons CO2 per unit)
EMISSION_FACTORS = {
    "electricity": 0.000233,   # tons CO2 per kWh  (US grid average ~0.233 kg CO2/kWh)
    "fuel": 0.00264,           # tons CO2 per liter (diesel: ~2.64 kg CO2/liter)
    "travel": 0.000089,        # tons CO2 per km    (average air travel ~0.089 kg CO2/km)
    "cloud": 0.0004,           # tons CO2 per hour  (cloud server ~0.4 kg CO2/hr)
    "waste": 0.0005,           # tons CO2 per kg    (office waste ~0.5 kg CO2/kg)
}

CATEGORY_LABELS = {
    "electricity": "Electricity",
    "fuel": "Fuel",
    "travel": "Travel",
    "cloud": "Cloud/Data Centers",
    "waste": "Office Waste",
}

# Carbon rating thresholds (monthly tons CO2)
RATING_THRESHOLDS = [
    ("A", "🌱 Sustainable", "#00C851", 5.0),
    ("B", "🌿 Moderate", "#FFB300", 15.0),
    ("C", "⚠️ High Emissions", "#FF8800", 30.0),
    ("D", "🚨 Critical", "#FF4444", float("inf")),
]


def calculate_emissions(
    electricity_kwh: float,
    fuel_liters: float,
    travel_km: float,
    cloud_hours: float,
    waste_kg: float,
) -> tuple[float, dict]:
    """
    Calculate monthly CO2 emissions (in tons) for each category.

    Returns:
        (total_co2_tons, breakdown_dict)
    """
    breakdown = {
        CATEGORY_LABELS["electricity"]: max(0.0, electricity_kwh) * EMISSION_FACTORS["electricity"],
        CATEGORY_LABELS["fuel"]: max(0.0, fuel_liters) * EMISSION_FACTORS["fuel"],
        CATEGORY_LABELS["travel"]: max(0.0, travel_km) * EMISSION_FACTORS["travel"],
        CATEGORY_LABELS["cloud"]: max(0.0, cloud_hours) * EMISSION_FACTORS["cloud"],
        CATEGORY_LABELS["waste"]: max(0.0, waste_kg) * EMISSION_FACTORS["waste"],
    }
    total = sum(breakdown.values())
    return total, breakdown


def get_carbon_rating(monthly_co2_tons: float) -> tuple[str, str, str]:
    """
    Return (letter, label, hex_color) based on monthly CO2 tons.

    Thresholds (monthly):
        A  < 5 tons   — Sustainable
        B  5–15 tons  — Moderate
        C  15–30 tons — High Emissions
        D  ≥ 30 tons  — Critical
    """
    for letter, label, color, threshold in RATING_THRESHOLDS:
        if monthly_co2_tons < threshold:
            return letter, label, color
    return "D", "🚨 Critical", "#FF4444"
