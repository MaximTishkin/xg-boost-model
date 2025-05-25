# XGBoost model with api

Сервис предсказания временных затрат на разработку задач на основе текстовых описаний.

## 🛠 Техническая реализация

### Модели
| Компонент               | Источник                          | Локация в проекте                       |
|-------------------------|-----------------------------------|-----------------------------------------|
| Текстовые эмбеддинги    | SentenceTransformer               | `paraphrase-multilingual-MiniLM-L12-v2` |
| Регрессионные модели    | Обученные XGBoost                 | `models/trained/`                       |

## Установка
pip install -r requirements.txt

## Перед запуском обучите модель
Скрипт обучения и сохранения модели  
python train.py

## Запуск API
uvicorn app:app --reload

## Пример запроса
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"project_name":"Разработка API","task_description":"Создать REST API"}'

Ответ:
{
  "prediction": {
    "plan_demetra_developer": 16.0,
    "plan_b2b_developer": 8.0,
    "plan_analyst": 24.0,
    "fact_demetra_developer": 18.0,
    "fact_b2b_developer": 10.0, 
    "fact_analyst": 22.0
  }
}