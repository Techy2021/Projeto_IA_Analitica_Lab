FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8501

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false

CMD ["sh", "-c", "streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
