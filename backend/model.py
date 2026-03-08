"""
ML model for carbon emission prediction using Random Forest / Gradient Boosting / Linear Regression.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .carbon_calculator import EMISSION_FACTORS

FEATURE_COLUMNS = ["electricity_kwh", "fuel_liters", "travel_km", "cloud_hours", "waste_kg"]
FEATURE_DISPLAY_NAMES = ["Electricity", "Fuel", "Travel", "Cloud", "Waste"]


def generate_training_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data representing a variety of organization sizes.
    Noise (~5 %) is added to simulate real-world measurement variance.
    """
    rng = np.random.default_rng(random_state)

    electricity = rng.uniform(500, 50_000, n_samples)
    fuel = rng.uniform(50, 5_000, n_samples)
    travel = rng.uniform(100, 50_000, n_samples)
    cloud = rng.uniform(10, 2_000, n_samples)
    waste = rng.uniform(10, 1_000, n_samples)

    co2 = (
        electricity * EMISSION_FACTORS["electricity"]
        + fuel * EMISSION_FACTORS["fuel"]
        + travel * EMISSION_FACTORS["travel"]
        + cloud * EMISSION_FACTORS["cloud"]
        + waste * EMISSION_FACTORS["waste"]
    )
    noise = rng.normal(1.0, 0.05, n_samples)
    co2 = np.maximum(co2 * noise, 0.0)

    return pd.DataFrame(
        {
            "electricity_kwh": electricity,
            "fuel_liters": fuel,
            "travel_km": travel,
            "cloud_hours": cloud,
            "waste_kg": waste,
            "total_co2_tons": co2,
        }
    )


class CarbonEmissionModel:
    """
    Wraps sklearn regression models for carbon emission prediction.

    Supported model types: 'random_forest', 'gradient_boosting', 'linear_regression'.
    """

    _MODELS = {
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
        "linear_regression": lambda: LinearRegression(),
    }

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in self._MODELS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(self._MODELS)}")
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics: dict = {}

    def train(self, df: pd.DataFrame | None = None) -> dict:
        """Train on the provided DataFrame (or synthetic data if None)."""
        if df is None:
            df = generate_training_data()

        X = df[FEATURE_COLUMNS]
        y = df["total_co2_tons"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        self.model = self._MODELS[self.model_type]()
        self.model.fit(X_train_sc, y_train)

        y_pred = self.model.predict(X_test_sc)
        self.metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "model_type": self.model_type,
        }
        self.is_trained = True
        return self.metrics

    def predict(
        self,
        electricity_kwh: float,
        fuel_liters: float,
        travel_km: float,
        cloud_hours: float,
        waste_kg: float,
    ) -> float:
        """Return predicted monthly CO2 tons for the given inputs."""
        if not self.is_trained:
            self.train()

        X = pd.DataFrame(
            [[electricity_kwh, fuel_liters, travel_km, cloud_hours, waste_kg]],
            columns=FEATURE_COLUMNS,
        )
        X_sc = self.scaler.transform(X)
        return float(max(0.0, self.model.predict(X_sc)[0]))

    def get_feature_importance(self) -> dict:
        """Return feature importance dict (only for tree-based models)."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(FEATURE_DISPLAY_NAMES, self.model.feature_importances_))
