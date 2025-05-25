from fastapi import FastAPI
from src.controllers.predict_controller import PredictController, TaskRequest, PredictionResponse

app = FastAPI()
predictor = PredictController()


@app.on_event("startup")
async def startup_event():
    await predictor.load_models()


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(task: TaskRequest):
    return await predictor.predict(task)


@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "models_loaded": len(predictor.prediction_models) > 0
    }


# Для локального тестирования
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
