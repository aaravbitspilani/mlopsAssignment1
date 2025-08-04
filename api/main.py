from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[v for v in data.dict().values()]])
    prediction = model.predict(input_array)
    return {"prediction": float(prediction[0])}
