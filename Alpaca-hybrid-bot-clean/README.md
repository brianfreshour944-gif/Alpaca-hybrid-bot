# Alpaca Hybrid Trading Bot

Mean reversion + RSI + trend filter strategy. Backtest: 74.6% win rate, 7.19% return.

## Local Testing

1. Copy `.env.example` to `.env` and fill in your Alpaca paper API keys.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python alpaca_hybrid_bot.py`

## Deploy on Oracle Cloud + Coolify

1. Push this repository to GitHub.
2. On your Oracle Cloud VM, install Coolify.
3. Create a new resource from this GitHub repo.
4. Add environment variables (from `.env.example`) in Coolify's UI.
5. Deploy – Coolify will build the Docker container and run the bot 24/7.

## Self‑Learning Roadmap

- Add HMM market regime detection
- Replace fixed thresholds with a DQN agent (FinRL)
- Weekly backtest on 15,000 candles + auto parameter tuning

## License

MIT
