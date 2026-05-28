import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def available_plants() -> list[int]:
    files = RAW_DIR.glob("Plant_*_Generation_Data.csv")
    return sorted(
        int(re.search(r"Plant_(\d+)_", f.name).group(1)) for f in files
    )


def load_plant(plant_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation = pd.read_csv(
        RAW_DIR / f"Plant_{plant_id}_Generation_Data.csv",
        parse_dates=["DATE_TIME"],
        dayfirst=True,
    )
    weather = pd.read_csv(
        RAW_DIR / f"Plant_{plant_id}_Weather_Sensor_Data.csv",
        parse_dates=["DATE_TIME"],
    )
    weather = weather[
        ["DATE_TIME", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    ].copy()
    return generation, weather
