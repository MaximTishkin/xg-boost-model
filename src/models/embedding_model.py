import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.settings import Settings


class EmbeddingGenerator:
    def __init__(self, model_path=Settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_path)
        self.settings = Settings()

    def generate_embeddings(self, texts):
        """Генерация и сохранение эмбеддингов"""
        if not self.settings.EMBEDDINGS_PATH.exists():
            embeddings = self.model.encode(texts, show_progress_bar=True)
            self.settings.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.settings.EMBEDDINGS_PATH, embeddings)
            return embeddings
        return np.load(self.settings.EMBEDDINGS_PATH)
