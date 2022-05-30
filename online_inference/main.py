from fastapi import FastAPI

from ml_project.models import load_model
from ml_project import online_predict_pipeline
from .model import Item

MODEL_PATH = 'models/model.pkl'
app = FastAPI()
model = load_model(MODEL_PATH)


@app.get("/health", status_code=200)
async def check_model():
    return {"message": "Ok"}


@app.post("/predict", status_code=201)
async def get_prediction(data: Item):
    predicts = online_predict_pipeline(data.dict(), model)
    return {"predict": int(predicts[0])}
