@echo off
:: Custom MCP Configuration for YOUR Coolify Instance
:: URL: https://64.181.195.30.sslip.io

set COOLIFY_API_URL=https://64.181.195.30.sslip.io
set COOLIFY_API_KEY=your-real-api-key-here
set ENABLE_REAL_API=true

echo Configuring MCP for YOUR Coolify instance...
echo Coolify URL: %COOLIFY_API_URL%
echo.

python coolify-mcp-server.py