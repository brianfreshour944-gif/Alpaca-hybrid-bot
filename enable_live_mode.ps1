<#
.PowerShell
# Script to enable live Coolify API integration
# Configure these variables with your actual Coolify credentials
#>

Write-Host "Configuring Coolify MCP for LIVE integration..." -ForegroundColor Green
Write-Host ""

# Set your Coolify API endpoint (replace with your actual Coolify URL)
$env:COOLIFY_API_URL = "https://your-coolify-instance.com"

# Set your Coolify API key (replace with your actual API key)
$env:COOLIFY_API_KEY = "your-coolify-api-key-here"

# Enable real API mode
$env:ENABLE_REAL_API = "true"

Write-Host "Environment variables configured:"
Write-Host "Coolify API URL: $env:COOLIFY_API_URL"
Write-Host "Real API Mode: $env:ENABLE_REAL_API"
Write-Host ""

Write-Host "Starting MCP Server in LIVE mode..." -ForegroundColor Cyan
Write-Host "(Make sure to replace the API credentials above with your actual Coolify credentials)"
Write-Host ""

# Start the MCP server
python coolify-mcp-server.py

# The environment variables will remain set for subsequent commands in this session
# You can now run: python create_trading_bot.py