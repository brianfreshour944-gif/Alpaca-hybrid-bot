#!/usr/bin/env python3
"""
Script to create a trading bot resource using the Coolify MCP Client
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the CoolifyMCPClient class directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("coolify_mcp_client", "coolify-mcp-client.py")
coolify_mcp_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coolify_mcp_client)
CoolifyMCPClient = coolify_mcp_client.CoolifyMCPClient

# Allow direct API calls without going through MCP server
def create_trading_bot_direct(coolify_url, api_key):
    """Create trading bot directly via Coolify API"""
    import requests

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

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Use the correct Coolify API endpoint
        url = f"{coolify_url}/v1/applications/deploy"
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 201:
            return {
                "status": "success",
                "message": "Trading bot created directly via Coolify API",
                "coolify_id": response.json().get("id"),
                "resource_id": resource_id
            }
        else:
            return {
                "status": "error",
                "message": f"Coolify API error: {response.status_code} - {response.text}",
                "details": response.json()
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Network error: {str(e)}"
        }

def main():
    print("Creating Trading Bot Resource via MCP...")

    # Initialize MCP Client
    client = CoolifyMCPClient()

    # Create trading bot resource
    print("\nCreating trading bot resource...")
    result = client.create_resource(
        resource_type="trading-bot",
        config={
            "config_file": "coolify-bot-config.yaml",
            "bot_name": "apex_oracle_bot",
            "environment": "production",
            "replicas": 1
        }
    )

    print(f"SUCCESS: {result['message']}")
    print(f"Resource ID: {result['resource_id']}")

    # List all resources to verify
    print("\nListing all MCP resources...")
    list_result = client.list_resources()
    print(f"Found {list_result['count']} resources:")

    for resource in list_result['resources']:
        print(f"  - {resource['resource_id']}: {resource['type']} ({resource['status']})")
        print(f"    Created: {resource['created_at']}")

    print("\nTrading bot resource created successfully!")
    print("\nNext steps:")
    print("1. Start the MCP server: python coolify-mcp-server.py")
    print("2. Deploy using Coolify CLI or web interface")
    print("3. Monitor resource status with: python coolify-mcp-client.py")

if __name__ == "__main__":
    main()