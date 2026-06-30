import json
import aiohttp
import numpy as np
import pandas as pd

class AIClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def _call_llm(self, prompt: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            ) as resp:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]

        try:
            return json.loads(text)
        except Exception:
            return {"error": "parse_error", "raw": text}

    @staticmethod
    def compute_realized_vol(df: pd.DataFrame) -> float:
        """Simple realized volatility (percent) from log returns."""
        closes = df["close"].astype(float).values
        if len(closes) < 30:
            return 0.0
        rets = np.diff(np.log(closes))
        vol = np.std(rets) * np.sqrt(len(rets)) * 100.0
        return float(vol)

    @staticmethod
    def build_correlation_matrix(price_series_map: dict) -> dict:
        """
        price_series_map: {symbol: [prices]}
        Returns a dict: {symbol: {other_symbol: corr}}
        """
        if not price_series_map:
            return {}

        df = pd.DataFrame(price_series_map)
        corr = df.corr().fillna(0.0)

        result = {}
        for sym in corr.columns:
            result[sym] = {other: float(corr.loc[sym, other]) for other in corr.columns}
        return result

    @staticmethod
    def build_portfolio_context(positions, equity, regime_map, vol_map, corr_matrix):
        return {
            "equity": equity,
            "positions": [
                {
                    "symbol": sym,
                    "qty": pos["qty"],
                    "market_value": pos["market_value"],
                    "price": pos["price"],
                    "regime": regime_map.get(sym, "unknown"),
                    "volatility_pct": vol_map.get(sym, 0.0),
                }
                for sym, pos in positions.items()
            ],
            "total_exposure": sum(pos["market_value"] for pos in positions.values()),
            "num_positions": len(positions),
            "max_positions": 6,  # Will be configured
            "max_portfolio_value": 500.0,  # Will be configured
            "correlation_matrix": corr_matrix,
        }

    async def evaluate_trade(self, context: dict) -> dict:
        prompt = f"""
        You are an expert quantitative trading assistant for a live crypto bot.

        Evaluate this context and respond ONLY with a JSON object.

        Context:
        {json.dumps(context, indent=2)}

        Requirements:
        - "action": "BUY", "SELL", or "HOLD".
        - "confidence": 0–1.
        - "reason": short explanation.
        - "position_size_factor": 0–1.
        - "risk_score": 0–1.
        - "regime_override": null or a short string.

        Output MUST be valid JSON only.
        """

        result = await self._call_llm(prompt)

        if "error" in result:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "AI parse error",
                "position_size_factor": 0.0,
                "risk_score": 1.0,
                "regime_override": None,
            }

        return {
            "action": result.get("action", "HOLD"),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", "No reason"),
            "position_size_factor": float(result.get("position_size_factor", 0.0)),
            "risk_score": float(result.get("risk_score", 1.0)),
            "regime_override": result.get("regime_override"),
        }

    async def evaluate_portfolio(self, context: dict) -> dict:
        prompt = f"""
        You are an expert portfolio manager for a crypto trading bot.
        Analyze this portfolio and provide optimization recommendations.

        Context:
        {json.dumps(context, indent=2)}

        Requirements:
        - "overall_risk_score": 0–1 (0=safe, 1=high risk)
        - "suggestions": list of specific recommendations
        - "regime_balance": dict of {regime: count}
        - "volatility_balance": dict of {volatility_range: count}
        - "correlation_warnings": list of highly correlated pairs
        - "rebalancing_recommendations": list of specific trade ideas

        Output MUST be valid JSON only.
        """

        result = await self._call_llm(prompt)

        if "error" in result:
            return {
                "overall_risk_score": 0.5,
                "suggestions": ["AI evaluation error - using default portfolio settings"],
                "regime_balance": {},
                "volatility_balance": {},
                "correlation_warnings": [],
                "rebalancing_recommendations": [],
            }

        return {
            "overall_risk_score": float(result.get("overall_risk_score", 0.5)),
            "suggestions": result.get("suggestions", []),
            "regime_balance": result.get("regime_balance", {}),
            "volatility_balance": result.get("volatility_balance", {}),
            "correlation_warnings": result.get("correlation_warnings", []),
            "rebalancing_recommendations": result.get("rebalancing_recommendations", []),
        }
