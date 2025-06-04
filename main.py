from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Carregar modelo e scaler
model = load_model('./models/lstm_model.h5')
scaler = joblib.load('./models/scaler.pkl')

# Parâmetros da janela
WINDOW_SIZE = model.input_shape[1]

app = FastAPI(title="API de Previsão de Ações com LSTM")

# Definir schema da entrada
class InputData(BaseModel):
    values: list[float]  # lista com os últimos N preços normalizados (janela)

@app.post("/predict")
def predict(data: InputData):
    input_values = np.array(data.values)

    if len(input_values) != WINDOW_SIZE:
        return {"erro": f"A sequência deve conter {WINDOW_SIZE} valores."}

    # Pré-processar dados
    input_scaled = np.array(input_values).reshape(1, WINDOW_SIZE, 1)

    # Prever
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    return {
        "previsao": float(prediction)
    }