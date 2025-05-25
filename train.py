from src.utils.data_loader import load_data, prepare_features
from src.models.estimator import TimeEstimator
from src.config.settings import Settings


def main():
    # Загрузка данных
    data = load_data(Settings().DATA_PATH, Settings().SHEET_NAME)
    descriptions, targets = prepare_features(data)

    # Обучение моделей
    estimator = TimeEstimator()
    estimator.train(descriptions, targets)


if __name__ == "__main__":
    main()
