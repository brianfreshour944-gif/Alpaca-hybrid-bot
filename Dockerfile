FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crypto_mean_reversion_alpaca.py .

# State files persist via Coolify volume mount at /app
COPY alpaca_hybrid_bot.py .
CMD ["python", "-u", "alpaca_hybrid_bot.py"]
