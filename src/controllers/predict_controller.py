from fastapi import HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path
from typing import Dict

from src.config.settings import Settings


class TaskRequest(BaseModel):
    project_name: str
    task_description: str


class PredictionResponse(BaseModel):
    prediction: Dict[str, float]


class PredictController:
    def __init__(self):
        self.embedder = None
        self.prediction_models = {}
        self.models_dir = Path("models/trained/")
        self.embedding_model = Settings().EMBEDDING_MODEL

    async def load_models(self):
        """Загрузка моделей при старте приложения"""
        try:
            # 1. Загрузка модели для эмбеддингов
            self.embedder = SentenceTransformer(self.embedding_model)

            # 2. Загрузка всех предсказывающих моделей
            target_columns = [
                'plan_demetra_developer',
                'plan_b2b_developer',
                'plan_analyst',
                'fact_demetra_developer',
                'fact_b2b_developer',
                'fact_analyst'
            ]

            for col in target_columns:
                model_path = self.models_dir / f"xgb_{col}.joblib"
                if model_path.exists():
                    self.prediction_models[col] = joblib.load(model_path)

            return True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading models: {str(e)}"
            )

    async def predict(self, task: TaskRequest) -> PredictionResponse:
        """Основной метод предсказания"""
        if not self.embedder or not self.prediction_models:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded"
            )

        try:
            full_text = f"{task.project_name} {task.task_description}"
            embedding = self.embedder.encode([full_text])
            predictions = {
                col: round(float(model.predict(embedding)[0]))
                for col, model in self.prediction_models.items()
            }

            return PredictionResponse(prediction=predictions)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
