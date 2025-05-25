from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np
from typing import Dict
from src.config.settings import Settings
from src.models.embedding_model import EmbeddingGenerator
from src.utils.metrics import calculate_metrics


class TimeEstimator:
    def __init__(self):
        self.settings = Settings()
        self.embedder = EmbeddingGenerator()
        self.models: Dict[str, XGBRegressor] = {}

    def train(self, descriptions: list, targets: np.ndarray) -> None:
        """Обучение моделей для всех целевых переменных"""
        embeddings = self.embedder.generate_embeddings(descriptions)
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings, targets,
            test_size=self.settings.TEST_SIZE,
            random_state=self.settings.RANDOM_STATE
        )

        for i, target_name in enumerate(self.settings.target_cols):
            model = self._train_single_model(
                x_train, y_train[:, i],
                x_test, y_test[:, i],
                target_name
            )
            self.models[target_name] = model

    def _train_single_model(self, x_train, y_train, x_test, y_test, target_name):
        """Обучение одной модели"""
        model_name = f"xgb_{target_name}"
        model_path = self.settings.MODELS_DIR / f"{model_name}.joblib"

        model = XGBRegressor(**self.settings.MODEL_PARAMS)
        model.fit(x_train, y_train)

        metrics = calculate_metrics(model, x_test, y_test)
        print(f"{model_name}")
        for name, value in metrics.items():
            print(f"{name}: {value: .4f}")

        self._save_model(model, model_path)
        return model

    def _save_model(self, model, path: Path) -> None:
        """Сохранение модели на диск"""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
