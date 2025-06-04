import yfinance as yf
import joblib
import numpy as np

# Configurações
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-12-31'
best_window = 60

# Carregar dados e scaler
df = yf.download(symbol, start=start_date, end=end_date)
scaler = joblib.load("./models/scaler.pkl")

# Pegar os últimos N preços
data = df['Close'].values.reshape(-1, 1)
scaled = scaler.transform(data)
last_window = scaled[-best_window:].reshape(-1).tolist()

# Exibir JSON para testar a API
print("\nJSON para testar no /predict:")
print({
    "values": last_window
})
