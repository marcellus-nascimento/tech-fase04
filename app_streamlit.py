import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configurações da página
st.set_page_config(page_title="Previsão de Ações com LSTM", layout="centered")
st.title("📈 Previsão de Preço de Ações com LSTM")

# URL da API
API_URL = "http://localhost:8000/predict"  # Substitua pela URL do Render se estiver em produção

# Carrega o scaler para normalizar os dados manualmente antes de enviar à API
scaler = joblib.load("models/scaler.pkl")

# Opção de entrada
st.write("Escolha a forma de entrada dos dados:")
input_method = st.radio("Selecione:", ["📂 Upload de arquivo (.xlsx)", "⌨️ Digitar manualmente os 60 valores"])

if input_method == "📂 Upload de arquivo (.xlsx)":
    uploaded_file = st.file_uploader("Envie seu arquivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                st.error("Nenhuma coluna numérica encontrada no arquivo.")
            else:
                preco_col = numeric_cols[0]
                prices = df[preco_col].dropna().values

                if len(prices) < 60:
                    st.warning("O arquivo precisa conter pelo menos 60 valores.")
                else:
                    prices = prices[-60:]
                    st.line_chart(prices, height=200)

                    if st.button("🔮 Prever preço do próximo dia"):
                        norm_values = scaler.transform(prices.reshape(-1, 1)).flatten().tolist()
                        response = requests.post(API_URL, json={"values": norm_values})

                        if response.status_code == 200 and "previsao" in response.json():
                            predicted_price = response.json()["previsao"]

                            real_price = prices[-1]
                            mae = abs(real_price - predicted_price)
                            rmse = np.sqrt((real_price - predicted_price) ** 2)
                            mape = np.mean(np.abs((real_price - predicted_price) / real_price)) * 100

                            st.success(f"📌 Preço previsto para o próximo dia: **${predicted_price:.2f}**")

                            st.subheader("📊 Indicadores de Eficiência")
                            st.write(f"**MAE**: {mae:.2f}")
                            st.write(f"**RMSE**: {rmse:.2f}")
                            st.write(f"**MAPE**: {mape:.2f}%")

                            st.subheader("📉 Comparação: Real vs. Previsto")
                            fig, ax = plt.subplots()
                            ax.plot([59], [real_price], 'bo', label='Real')
                            ax.plot([60], [predicted_price], 'ro', label='Previsto')
                            ax.legend()
                            st.pyplot(fig)
                        else:
                            st.error("Erro ao obter a previsão da API.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

elif input_method == "⌨️ Digitar manualmente os 60 valores":
    st.write("Insira 60 valores de preços (um por linha ou separados por vírgula):")
    manual_input = st.text_area("Valores:")

    if st.button("🔮 Prever com valores digitados"):
        try:
            values = [float(v.strip().replace(",", ".")) for v in manual_input.replace("\n", ",").split(",") if v.strip()]
            if len(values) != 60:
                st.error("Você precisa fornecer exatamente 60 valores.")
            else:
                norm_values = scaler.transform(np.array(values).reshape(-1, 1)).flatten().tolist()
                response = requests.post(API_URL, json={"values": norm_values})

                if response.status_code == 200 and "previsao" in response.json():
                    predicted_price = response.json()["previsao"]

                    real_price = values[-1]
                    mae = abs(real_price - predicted_price)
                    rmse = np.sqrt((real_price - predicted_price) ** 2)
                    mape = np.mean(np.abs((real_price - predicted_price) / real_price)) * 100

                    st.success(f"📌 Preço previsto para o próximo dia: **${predicted_price:.2f}**")

                    st.subheader("📊 Indicadores de Eficiência")
                    st.write(f"**MAE**: {mae:.2f}")
                    st.write(f"**RMSE**: {rmse:.2f}")
                    st.write(f"**MAPE**: {mape:.2f}%")

                    st.subheader("📉 Comparação: Real vs. Previsto")
                    fig, ax = plt.subplots()
                    ax.plot([59], [real_price], 'bo', label='Real')
                    ax.plot([60], [predicted_price], 'ro', label='Previsto')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.error("Erro ao obter a previsão da API.")
        except Exception as e:
            st.error(f"Erro ao processar os valores: {e}")

