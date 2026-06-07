#!/usr/bin/env python3
import asyncio
import os
import logging
import psycopg2
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ... (Keep your imports and logging setup as they are)

# --- DATABASE HELPERS ---
# (Keep your existing log_error_to_db, write_trade_to_db, and check_status)

def register_order_in_db(bot_name, order_id, symbol, side, price):
    """Registers an open order in the database."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO bot_orders (order_id, bot_name, symbol, side, price, status)
                    VALUES (%s, %s, %s, %s, %s, 'OPEN')
                ''', (str(order_id), bot_name, symbol, side, float(price)))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to register order in DB: {e}")

# --- BOT CLASS ---
class MeanReversionBot:
    def __init__(self):
        self.bot_name = os.getenv('BOT_NAME', 'Static-Repo-okx-bot')
        self.trading = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
        check_status(self.bot_name)

    def place_order_tracked(self, symbol, side, qty):
        """Helper to place order and immediately tag it in DB."""
        order = self.trading.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC))
        register_order_in_db(self.bot_name, order.id, symbol, side.value, 0.0)
        return order

    async def sync_orders(self):
        """Check for filled orders and update their status in DB."""
        db_url = os.getenv('DATABASE_URL')
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (self.bot_name,))
                for (oid, symbol) in cur.fetchall():
                    alpaca_order = self.trading.get_order_by_id(oid)
                    if alpaca_order.status == 'filled':
                        cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (oid,))
                        write_trade_to_db(self.bot_name, symbol, alpaca_order.side.value, alpaca_order.filled_avg_price, alpaca_order.filled_qty, 0.0, oid)
                conn.commit()

    async def run(self):
        while True:
            try:
                check_status(self.bot_name)
                await self.sync_orders()
                
                # ... (Existing logic for entry/exit)
                
                await asyncio.sleep(300)
            except Exception as e:
                error_msg = f"Main loop error: {str(e)}"
                logger.error(error_msg)
                log_error_to_db(self.bot_name, error_msg)
                await asyncio.sleep(60)
