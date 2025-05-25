import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np
from src.config.settings import Settings


def load_data(filepath: str | Path, sheet_name: str) -> pd.DataFrame:
    data = pd.read_excel(
        filepath,
        engine='openpyxl',
        sheet_name=sheet_name,
        usecols=Settings().TEXT_COLS + Settings().NUMERIC_COLS
    )

    # Обработка числовых колонок
    data[Settings().NUMERIC_COLS] = data[Settings().NUMERIC_COLS].fillna(0).astype(float)

    # Обработка текстовых колонок
    data[Settings().TEXT_COLS] = data[Settings().TEXT_COLS].fillna('')
    data["text"] = data["project_name"] + " " + data["task_description"]

    return data


def prepare_features(data: pd.DataFrame) -> Tuple[list, np.ndarray]:
    descriptions = data["text"].tolist()
    targets = data[Settings().target_cols].values
    return descriptions, targets
