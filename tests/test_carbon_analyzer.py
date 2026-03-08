"""
Unit tests for the AI-Powered Carbon Emission Analyzer backend.
"""

import pytest
import numpy as np
import pandas as pd

from backend.carbon_calculator import (
    calculate_emissions,
    get_carbon_rating,
    EMISSION_FACTORS,
    CATEGORY_LABELS,
)
from backend.model import CarbonEmissionModel, generate_training_data
from backend.recommendations import get_recommendations, simulate_reduction


# ─────────────────────────────────────────────────────────────────────────────
# carbon_calculator
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculateEmissions:
    def test_all_zeros_returns_zero(self):
        total, breakdown = calculate_emissions(0, 0, 0, 0, 0)
        assert total == 0.0
        for v in breakdown.values():
            assert v == 0.0

    def test_positive_values_return_positive_total(self):
        total, breakdown = calculate_emissions(5000, 800, 12000, 300, 200)
        assert total > 0
        assert sum(breakdown.values()) == pytest.approx(total, rel=1e-6)

    def test_electricity_only(self):
        electricity_kwh = 1000.0
        expected = electricity_kwh * EMISSION_FACTORS["electricity"]
        total, breakdown = calculate_emissions(electricity_kwh, 0, 0, 0, 0)
        assert total == pytest.approx(expected, rel=1e-6)
        assert breakdown[CATEGORY_LABELS["electricity"]] == pytest.approx(expected, rel=1e-6)

    def test_all_categories_present(self):
        _, breakdown = calculate_emissions(1, 1, 1, 1, 1)
        for label in CATEGORY_LABELS.values():
            assert label in breakdown

    def test_negative_inputs_clamped_to_zero(self):
        total, breakdown = calculate_emissions(-100, -50, -200, -10, -5)
        assert total == 0.0

    def test_breakdown_sums_to_total(self):
        total, breakdown = calculate_emissions(3000, 600, 8000, 200, 150)
        assert sum(breakdown.values()) == pytest.approx(total, rel=1e-6)

    def test_scaling_linearity(self):
        total1, _ = calculate_emissions(1000, 100, 1000, 100, 100)
        total2, _ = calculate_emissions(2000, 200, 2000, 200, 200)
        assert total2 == pytest.approx(total1 * 2, rel=1e-6)


class TestGetCarbonRating:
    def test_rating_a(self):
        letter, label, color = get_carbon_rating(2.0)
        assert letter == "A"
        assert "Sustainable" in label

    def test_rating_b(self):
        letter, label, color = get_carbon_rating(10.0)
        assert letter == "B"

    def test_rating_c(self):
        letter, label, color = get_carbon_rating(20.0)
        assert letter == "C"

    def test_rating_d(self):
        letter, label, color = get_carbon_rating(50.0)
        assert letter == "D"
        assert "Critical" in label

    def test_boundary_a_b(self):
        # Exactly 5 — should be B (>= 5 is not A)
        letter, _, _ = get_carbon_rating(5.0)
        assert letter == "B"

    def test_zero_emissions_is_a(self):
        letter, _, _ = get_carbon_rating(0.0)
        assert letter == "A"

    def test_color_is_hex(self):
        _, _, color = get_carbon_rating(1.0)
        assert color.startswith("#")


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateTrainingData:
    def test_shape(self):
        df = generate_training_data(n_samples=100)
        assert df.shape == (100, 6)

    def test_no_negative_co2(self):
        df = generate_training_data(n_samples=500)
        assert (df["total_co2_tons"] >= 0).all()

    def test_columns(self):
        df = generate_training_data(n_samples=10)
        expected_cols = {
            "electricity_kwh", "fuel_liters", "travel_km",
            "cloud_hours", "waste_kg", "total_co2_tons",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_reproducibility(self):
        df1 = generate_training_data(n_samples=50, random_state=7)
        df2 = generate_training_data(n_samples=50, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)


class TestCarbonEmissionModel:
    @pytest.fixture(scope="class")
    def trained_rf(self):
        m = CarbonEmissionModel(model_type="random_forest")
        m.train()
        return m

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            CarbonEmissionModel(model_type="nonexistent_model")

    def test_train_returns_metrics(self, trained_rf):
        metrics = trained_rf.metrics
        assert "r2_score" in metrics
        assert "rmse" in metrics
        assert metrics["r2_score"] > 0.90  # Model should fit synthetic data well

    def test_predict_returns_non_negative(self, trained_rf):
        pred = trained_rf.predict(5000, 800, 12000, 300, 200)
        assert pred >= 0.0

    def test_predict_zero_inputs(self, trained_rf):
        pred = trained_rf.predict(0, 0, 0, 0, 0)
        assert pred >= 0.0

    def test_higher_inputs_produce_higher_prediction(self, trained_rf):
        low = trained_rf.predict(500, 100, 1000, 50, 50)
        high = trained_rf.predict(20000, 3000, 40000, 1000, 500)
        assert high > low

    def test_feature_importance_keys(self, trained_rf):
        fi = trained_rf.get_feature_importance()
        assert set(fi.keys()) == {"Electricity", "Fuel", "Travel", "Cloud", "Waste"}

    def test_feature_importance_sums_to_one(self, trained_rf):
        fi = trained_rf.get_feature_importance()
        total = sum(fi.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_gradient_boosting_model(self):
        m = CarbonEmissionModel(model_type="gradient_boosting")
        metrics = m.train()
        assert metrics["r2_score"] > 0.90

    def test_linear_regression_model(self):
        m = CarbonEmissionModel(model_type="linear_regression")
        metrics = m.train()
        # Linear regression also fits well on this synthetic data
        assert metrics["r2_score"] > 0.95

    def test_auto_train_on_first_predict(self):
        m = CarbonEmissionModel(model_type="linear_regression")
        assert not m.is_trained
        pred = m.predict(1000, 100, 1000, 100, 100)
        assert m.is_trained
        assert pred >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# recommendations
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRecommendations:
    def _breakdown(self, elec=5000, fuel=800, travel=12000, cloud=300, waste=200):
        _, bd = calculate_emissions(elec, fuel, travel, cloud, waste)
        return bd

    def test_returns_list(self):
        bd = self._breakdown()
        recs = get_recommendations(5000, 800, 12000, 300, 200, bd)
        assert isinstance(recs, list)

    def test_has_required_keys(self):
        bd = self._breakdown()
        recs = get_recommendations(5000, 800, 12000, 300, 200, bd)
        for rec in recs:
            assert "issue" in rec
            assert "suggestions" in rec
            assert "priority" in rec
            assert "potential_reduction" in rec

    def test_empty_breakdown_returns_empty(self):
        recs = get_recommendations(0, 0, 0, 0, 0, {})
        assert recs == []

    def test_high_electricity_triggers_recommendation(self):
        # Electricity dominates
        bd = self._breakdown(elec=50000, fuel=0, travel=0, cloud=0, waste=0)
        recs = get_recommendations(50000, 0, 0, 0, 0, bd)
        issues = [r["issue"] for r in recs]
        assert any("Electricity" in i for i in issues)

    def test_high_travel_triggers_recommendation(self):
        bd = self._breakdown(elec=0, fuel=0, travel=100000, cloud=0, waste=0)
        recs = get_recommendations(0, 0, 100000, 0, 0, bd)
        issues = [r["issue"] for r in recs]
        assert any("Travel" in i for i in issues)

    def test_low_emissions_returns_positive_feedback(self):
        bd = self._breakdown(elec=50, fuel=5, travel=50, cloud=5, waste=5)
        recs = get_recommendations(50, 5, 50, 5, 5, bd)
        assert len(recs) >= 1
        assert any("Sustainable" in r["issue"] or "Strong" in r["issue"] for r in recs)


class TestSimulateReduction:
    def test_no_changes_equals_baseline(self):
        elec, fuel, travel, cloud, waste = 5000, 800, 12000, 300, 200
        baseline, _ = calculate_emissions(elec, fuel, travel, cloud, waste)
        new_total, _ = simulate_reduction(elec, fuel, travel, cloud, waste)
        assert new_total == pytest.approx(baseline, rel=1e-6)

    def test_100_percent_solar_eliminates_electricity(self):
        _, bd_new = simulate_reduction(5000, 800, 12000, 300, 200, solar_pct=100)
        assert bd_new[CATEGORY_LABELS["electricity"]] == pytest.approx(0.0, abs=1e-9)

    def test_100_percent_ev_eliminates_fuel(self):
        _, bd_new = simulate_reduction(5000, 800, 12000, 300, 200, ev_pct=100)
        assert bd_new[CATEGORY_LABELS["fuel"]] == pytest.approx(0.0, abs=1e-9)

    def test_100_percent_remote_work_eliminates_travel(self):
        _, bd_new = simulate_reduction(5000, 800, 12000, 300, 200, remote_work_pct=100)
        assert bd_new[CATEGORY_LABELS["travel"]] == pytest.approx(0.0, abs=1e-9)

    def test_all_interventions_reduce_total(self):
        baseline, _ = calculate_emissions(5000, 800, 12000, 300, 200)
        new_total, _ = simulate_reduction(
            5000, 800, 12000, 300, 200,
            solar_pct=50, ev_pct=50, remote_work_pct=50,
            green_cloud_pct=50, waste_reduction_pct=50,
        )
        assert new_total < baseline

    def test_returns_non_negative(self):
        new_total, _ = simulate_reduction(
            5000, 800, 12000, 300, 200,
            solar_pct=100, ev_pct=100, remote_work_pct=100,
            green_cloud_pct=100, waste_reduction_pct=100,
        )
        assert new_total >= 0.0
