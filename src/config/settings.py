from pathlib import Path
from typing import List


class Settings:
    # Пути
    DATA_PATH = Path("src/data/raw/file_for_ai.xlsx")
    SHEET_NAME = "for_training"
    EMBEDDINGS_PATH = Path("src/data/embeddings/embeddings.npy")
    MODELS_DIR = Path("models/trained")

    # Параметры модели
    MODEL_PARAMS = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mae"
    }

    # Настройки
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # Колонки
    NUMERIC_COLS = [
        'plan_demetra_developer',
        'plan_b2b_developer',
        'plan_analyst',
        'fact_demetra_developer',
        'fact_b2b_developer',
        'fact_analyst'
    ]

    TEXT_COLS = ['project_name', 'task_description']

    @property
    def target_cols(self) -> List[str]:
        return self.NUMERIC_COLS
