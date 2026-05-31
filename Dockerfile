FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crypto_mean_reversion_alpaca.py .

# State files persist via Coolify volume mount at /app
CMD ["python", "-u", "crypto_mean_reversion_alpaca.py"]
