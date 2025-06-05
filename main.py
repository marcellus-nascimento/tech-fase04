from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from fastapi.responses import HTMLResponse

app = FastAPI()

model = load_model("models/lstm_model.h5")
scaler = joblib.load("models/scaler.pkl")

WINDOW_SIZE = model.input_shape[1]

class InputData(BaseModel):
    values: list[float]

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>API LSTM rodando! Vá para /docs para testar.</h2>"

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array(data.values).reshape(1, WINDOW_SIZE, 1)
    prediction = model.predict(input_data)
    result = scaler.inverse_transform(prediction)[0][0]
    return {"previsao": float(result)}

import os

if __name__ == "__main__":
    import uvicorn
    port = (os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)