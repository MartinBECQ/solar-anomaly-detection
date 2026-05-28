import pandas as pd

IRRADIATION_MIN = 0.01  # kW/m² — filtre nuit


def compute_features(generation: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = generation.merge(weather, on="DATE_TIME", how="left")
    df = df[df["IRRADIATION"] > IRRADIATION_MIN].copy()

    df["efficiency"] = df["AC_POWER"] / df["DC_POWER"]
    df["perf_ratio"] = df["DC_POWER"] / df["IRRADIATION"]

    median_by_ts = df.groupby("DATE_TIME")["DC_POWER"].transform("median")
    df["power_deviation"] = df["DC_POWER"] - median_by_ts

    df["hour"] = df["DATE_TIME"].dt.hour
    df["date"] = df["DATE_TIME"].dt.date

    return df
