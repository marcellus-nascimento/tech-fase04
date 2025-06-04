FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copia arquivos necessários
COPY . /app

# Instala dependências
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expondo porta
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
