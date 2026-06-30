# Coolify Deployment Guide for Apex Oracle Trading Bot

This guide explains how to deploy the Apex Oracle Trading Bot using Coolify.

## Prerequisites

- Coolify account and access
- Docker installed on your deployment target
- All required API keys (OKX, Groq AI)

## Deployment Options

### Option 1: Quick Start (SQLite)

```yaml
# Use the provided coolify-bot-config.yaml
# No additional database setup required
```

### Option 2: Production (PostgreSQL)

1. Uncomment the PostgreSQL service in `coolify-bot-config.yaml`
2. Set PostgreSQL environment variables:
   ```bash
   export POSTGRES_USER=trading
   export POSTGRES_PASSWORD=trading123
   export POSTGRES_DB=trading_db
   export DATABASE_URL=postgresql://trading:trading123@postgres:5432/trading_db
   ```

## Deployment Steps

### 1. Prepare Environment Variables

Create a `.env` file:

```bash
# API Keys (REQUIRED)
export AI_KEY="your_groq_api_key"
export OKX_API_KEY="your_okx_api_key"
export OKX_SECRET_KEY="your_okx_secret_key"
export OKX_PASSPHRASE="your_okx_passphrase"

# Database (Optional - defaults to SQLite)
export DATABASE_URL="postgresql://trading:trading123@postgres:5432/trading_db"

# Bot Configuration (Optional)
export BOT_NAME="apex_oracle_bot"
export OKX_USE_DEMO="false"  # Set to "false" for live trading
```

### 2. Deploy with Coolify

#### Using Coolify CLI:

```bash
# Install Coolify CLI if needed
npm install -g coolify-cli

# Deploy the bot
coolify deploy \
  --config coolify-bot-config.yaml \
  --env .env \
  --name apex-oracle-bot
```

#### Using Coolify Dashboard:

1. Log in to your Coolify dashboard
2. Create a new project
3. Upload the `coolify-bot-config.yaml` file
4. Add the environment variables from `.env`
5. Start the deployment

### 3. Verify Deployment

```bash
# Check service status
coolify services list

# View logs
coolify logs apex-oracle-bot

# Check bot status endpoint
curl http://localhost:8080/status
```

## Configuration Details

### Main Service (`trading-bot`)

- **Image**: `python:3.11-slim`
- **Ports**: `8080` (status endpoint)
- **Resources**: 1 CPU, 1GB RAM
- **Healthcheck**: `/status` endpoint checked every 30s

### Database Options

#### SQLite (Default)
- No setup required
- Data stored in `./data/trades.db`
- Good for testing and development

#### PostgreSQL (Recommended for Production)
- Uncomment the `postgres` service in the config
- Persistent volume for data storage
- Better performance for high-frequency trading

## Monitoring and Maintenance

### Health Checks

The bot includes comprehensive health monitoring:

- **Database connectivity**
- **API availability**
- **Trade execution status**
- **Portfolio risk levels**

### Logging

All logs are available through Coolify:

```bash
coolify logs apex-oracle-bot --follow
```

### Updates

To update the bot:

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
coolify redeploy apex-oracle-bot
```

## Troubleshooting

### Common Issues

**Database connection failed**:
- Verify `DATABASE_URL` environment variable
- Check database service is running
- Test connection manually

**API errors**:
- Verify all API keys are correct
- Check API rate limits
- Test APIs with curl/postman

**Bot not starting**:
- Check logs for specific errors
- Verify Python dependencies: `pip install -r requirements.txt`
- Test locally first

### Support

For additional help:
- Check Coolify documentation: https://coolify.io/docs
- Review bot logs for detailed error information
- Verify all environment variables are set correctly

## Production Recommendations

1. **Use PostgreSQL** for better performance and reliability
2. **Enable monitoring** in Coolify dashboard
3. **Set up alerts** for critical healthcheck failures
4. **Use separate environments** for testing vs production
5. **Regular backups** of database and configuration
6. **Monitor resource usage** and adjust as needed