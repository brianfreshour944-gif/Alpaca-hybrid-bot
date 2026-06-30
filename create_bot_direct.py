#!/usr/bin/env python3
"""
Direct Coolify API script to create trading bot
Bypasses MCP server for direct API calls
"""

import requests
import uuid
import sys

def create_trading_bot_direct(coolify_url, api_key):
    """Create trading bot directly via Coolify API"""
    try:
        # Prepare the payload for Coolify applications/deploy endpoint
        resource_id = str(uuid.uuid4())
        payload = {
            "name": f"apex_oracle_bot-{resource_id[:8]}",
            "source": {
                "type": "docker",
                "config": "coolify-bot-config.yaml"
            },
            "environment": "production",
            "buildPack": "docker",
            "destination": {
                "type": "coolify"
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Use the correct Coolify API endpoint
        url = f"{coolify_url}/v1/applications/deploy"
        print(f"Creating trading bot via Coolify API...")
        print(f"URL: {url}")
        print(f"Bot Name: apex_oracle_bot-{resource_id[:8]}")

        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"Response Status: {response.status_code}")

        if response.status_code == 201:
            result = response.json()
            print("SUCCESS: Trading bot created!")
            print(f"Coolify ID: {result.get('id')}")
            print(f"Resource ID: {resource_id}")
            return True
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"NETWORK ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Get parameters from command line or use defaults
    coolify_url = sys.argv[1] if len(sys.argv) > 1 else "https://64.181.195.30.sslip.io"
    api_key = sys.argv[2] if len(sys.argv) > 2 else ""

    if not api_key:
        print("Please provide your Coolify API key!")
        print("Usage: python create_bot_direct.py <coolify_url> <api_key>")
    else:
        print(f"Connecting to: {coolify_url}")
        print(f"Using API key: {api_key[:10]}...{api_key[-10:]}")
        create_trading_bot_direct(coolify_url, api_key)
        print("\nCheck your Coolify dashboard to see the new trading bot!")