@echo off
:: Batch script to enable live Coolify API integration
:: Configure these variables with your actual Coolify credentials

echo Configuring Coolify MCP for LIVE integration...
echo.

:: Set your Coolify API endpoint (replace with your actual Coolify URL)
set COOLIFY_API_URL=https://your-coolify-instance.com

:: Set your Coolify API key (replace with your actual API key)
set COOLIFY_API_KEY=your-coolify-api-key-here

:: Enable real API mode
set ENABLE_REAL_API=true

echo Environment variables configured:
echo Coolify API URL: %COOLIFY_API_URL%
echo Real API Mode: %ENABLE_REAL_API%
echo.

echo Starting MCP Server in LIVE mode...
echo (Make sure to replace the API credentials above with your actual Coolify credentials)
echo.

python coolify-mcp-server.py

:: Keep the environment variables set for subsequent commands
:: You can now run: python create_trading_bot.py