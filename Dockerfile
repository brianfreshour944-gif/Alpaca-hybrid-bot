FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alpaca_hybrid_bot.py .

CMD ["python", "-u", "alpaca_hybrid_bot.py"]
