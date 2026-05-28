import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

FEATURES = ["DC_POWER", "perf_ratio", "power_deviation", "efficiency"]


class ZScoreModel:
    """Z-score baseline: anomalie si |z| > seuil dans le groupe (SOURCE_KEY, hour)."""

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._stats: pd.DataFrame | None = None
        self._threshold: float = norm.ppf(1 - contamination / 2)

    def fit(self, df: pd.DataFrame) -> "ZScoreModel":
        self._stats = (
            df.groupby(["SOURCE_KEY", "hour"])["DC_POWER"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        return self

    def _z_abs(self, df: pd.DataFrame) -> np.ndarray:
        merged = df[["SOURCE_KEY", "hour", "DC_POWER"]].merge(
            self._stats, on=["SOURCE_KEY", "hour"], how="left"
        )
        std = merged["std"].replace(0, np.nan).fillna(1)
        return ((merged["DC_POWER"] - merged["mean"]) / std).abs().values

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        return -self._z_abs(df)  # convention sklearn : plus bas = plus anormal

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(self._z_abs(df) > self._threshold, -1, 1)


def train(
    df: pd.DataFrame,
    contamination: float = 0.05,
    model_type: str = "isolation_forest",
) -> tuple:
    if model_type == "zscore":
        model = ZScoreModel(contamination=contamination)
        model.fit(df)
        return None, model

    X = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "lof":
        model = LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=True, n_jobs=-1
        )
    else:
        model = IsolationForest(
            contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1
        )

    model.fit(X_scaled)
    return scaler, model


def predict(model_tuple: tuple, df: pd.DataFrame) -> pd.DataFrame:
    scaler, model = model_tuple
    df = df.copy()

    X = df[FEATURES].dropna()
    idx = X.index
    input_data = df.loc[idx] if scaler is None else scaler.transform(X)

    df.loc[idx, "anomaly_score"] = -model.score_samples(input_data)
    df.loc[idx, "anomaly"] = (model.predict(input_data) == -1).astype(int)
    return df
