"""
Coolify MCP Client - Model Context Protocol Client
This client connects to the MCP server to manage Coolify resources

Usage:
1. Run the MCP server first: python coolify-mcp-server.py
2. Run this client: python coolify-mcp-client.py
3. Follow the interactive prompts to manage resources
"""

import json
import uuid
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Dict, Any, Optional

# MCP Client Configuration
MCP_SERVER_URL = "http://localhost:8765"  # Update if server runs elsewhere
SECRET_KEY = " 1|pw4uvV1q0k1ihnZMMdsY1VY4vPjzmFXeeZAQ0nrDda5a2c3a"  # Must match server's secret key
API_TIMEOUT = 30  # Timeout for API requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for failed requests
ENABLE_LOGGING = True  # Enable detailed logging

class CoolifyMCPClient:
    def __init__(self):
        self.api_key = None
        self.session_id = str(uuid.uuid4())
        self.settings = {
            'server_url': MCP_SERVER_URL,
            'secret_key': SECRET_KEY,
            'timeout': API_TIMEOUT,
            'max_retries': MAX_RETRIES,
            'enable_logging': ENABLE_LOGGING
        }

    def generate_signature(self, data: str) -> str:
        """Generate signature for MCP requests"""
        if not SECRET_KEY:
            raise ValueError("Client secret key not configured")

        return hmac.new(
            SECRET_KEY.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def make_request(self, endpoint: str, method: str = "POST", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate MCP request to server"""
        # In a real implementation, this would make an HTTP request
        # For this example, we'll simulate the response

        request_data = json.dumps(data or {})
        signature = self.generate_signature(request_data)

        print(f"\n[MCP Client] {method} {endpoint}")
        print(f"Request: {request_data}")
        print(f"Signature: {signature}")

        # Simulate server response based on endpoint
        if endpoint == "create":
            return {
                "status": "success",
                "resource_id": str(uuid.uuid4()),
                "message": f"Created {data['resource_type']} resource"
            }
        elif endpoint == "update":
            return {
                "status": "success",
                "resource_id": data['resource_id'],
                "message": "Updated resource"
            }
        elif endpoint == "delete":
            return {
                "status": "success",
                "resource_id": data['resource_id'],
                "message": "Deleted resource"
            }
        elif endpoint == "list":
            return {
                "status": "success",
                "resources": [
                    {
                        "resource_id": str(uuid.uuid4()),
                        "type": "trading-bot",
                        "status": "running",
                        "created_at": datetime.utcnow().isoformat()
                    }
                ],
                "count": 1
            }
        else:
            return {"status": "error", "message": "Unknown endpoint"}

    def create_resource(self, resource_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Coolify resource via MCP"""
        data = {
            "resource_type": resource_type,
            "config": config,
            "session_id": self.session_id
        }
        return self.make_request("create", "POST", data)

    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing Coolify resource via MCP"""
        data = {
            "resource_id": resource_id,
            "updates": updates,
            "session_id": self.session_id
        }
        return self.make_request("update", "POST", data)

    def delete_resource(self, resource_id: str) -> Dict[str, Any]:
        """Delete a Coolify resource via MCP"""
        data = {
            "resource_id": resource_id,
            "session_id": self.session_id
        }
        return self.make_request("delete", "POST", data)

    def list_resources(self) -> Dict[str, Any]:
        """List all MCP-managed resources"""
        return self.make_request("list", "GET")

    def get_settings(self) -> Dict[str, Any]:
        """Get current MCP client settings"""
        return {
            "status": "success",
            "settings": self.settings
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update MCP client settings"""
        # Validate and update settings
        for key, value in new_settings.items():
            if key in self.settings:
                self.settings[key] = value

        # Update global variables to reflect changes
        global MCP_SERVER_URL, SECRET_KEY, API_TIMEOUT, MAX_RETRIES, ENABLE_LOGGING
        MCP_SERVER_URL = self.settings['server_url']
        SECRET_KEY = self.settings['secret_key']
        API_TIMEOUT = self.settings['timeout']
        MAX_RETRIES = self.settings['max_retries']
        ENABLE_LOGGING = self.settings['enable_logging']

        return {
            "status": "success",
            "message": "Settings updated successfully",
            "settings": self.settings
        }

    def interactive_mode(self):
        """Run interactive MCP client"""
        print("Coolify MCP Client - Interactive Mode")
        print("Enter commands to manage your Coolify resources")
        print("Commands: create, update, delete, list, settings, exit")

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == "exit":
                    print("Goodbye!")
                    break

                elif command == "create":
                    resource_type = input("Resource type (e.g., trading-bot): ").strip()
                    config = input("Config file (e.g., coolify-bot-config.yaml): ").strip()
                    result = self.create_resource(resource_type, {"config": config})
                    print(f"✅ {result['message']}")

                elif command == "update":
                    resource_id = input("Resource ID: ").strip()
                    updates = input("Updates (JSON): ").strip()
                    try:
                        updates_dict = json.loads(updates)
                        result = self.update_resource(resource_id, updates_dict)
                        print(f"✅ {result['message']}")
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON format for updates")

                elif command == "delete":
                    resource_id = input("Resource ID: ").strip()
                    result = self.delete_resource(resource_id)
                    print(f"✅ {result['message']}")

                elif command == "list":
                    result = self.list_resources()
                    print(f"Found {result['count']} resources:")
                    for resource in result['resources']:
                        print(f"  - {resource['resource_id']}: {resource['type']} ({resource['status']})")

                elif command == "settings":
                    self._handle_settings_command()

                else:
                    print("❌ Unknown command. Try: create, update, delete, list, settings, exit")

            except Exception as e:
                print(f"❌ Error: {e}")

    def _handle_settings_command(self):
        """Handle the settings command with interactive menu"""
        while True:
            print("\n=== MCP Settings ===")
            print("1. View current settings")
            print("2. Edit settings")
            print("3. Back to main menu")
            print("4. Save settings to file")
            print("5. Load settings from file")

            choice = input("\nSettings> ").strip()

            if choice == "1":
                self._view_settings()
            elif choice == "2":
                self._edit_settings()
            elif choice == "3":
                print("Returning to main menu...")
                break
            elif choice == "4":
                self._save_settings_to_file()
            elif choice == "5":
                self._load_settings_from_file()
            else:
                print("ERROR: Invalid choice. Please try again.")

    def _view_settings(self):
        """Display current MCP settings"""
        settings = self.get_settings()
        print("\n=== Current MCP Settings ===")
        for key, value in settings['settings'].items():
            # Mask sensitive information
            if 'secret' in key.lower() or 'key' in key.lower():
                masked_value = value[:4] + '*' * max(0, len(str(value)) - 8) + value[-4:] if len(str(value)) > 8 else '********'
                print(f"{key}: {masked_value}")
            else:
                print(f"{key}: {value}")

    def _edit_settings(self):
        """Interactively edit MCP settings"""
        print("\n=== Edit MCP Settings ===")
        current_settings = self.get_settings()['settings']

        new_settings = {}
        for key, value in current_settings.items():
            # Mask sensitive values for display
            display_value = value
            if 'secret' in key.lower() or 'key' in key.lower():
                display_value = value[:4] + '*' * max(0, len(str(value)) - 8) + value[-4:] if len(str(value)) > 8 else '********'

            new_value = input(f"{key} [{display_value}]: ").strip()
            if new_value:  # Only update if user provides input
                new_settings[key] = new_value

        if new_settings:
            result = self.update_settings(new_settings)
            print(f"✅ {result['message']}")
        else:
            print("No settings were changed.")

    def _save_settings_to_file(self):
        """Save current settings to a configuration file"""
        filename = input("Enter filename to save settings (e.g., mcp-settings.json): ").strip()
        if not filename:
            print("❌ No filename provided.")
            return

        try:
            with open(filename, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"✅ Settings saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving settings: {e}")

    def _load_settings_from_file(self):
        """Load settings from a configuration file"""
        filename = input("Enter filename to load settings from: ").strip()
        if not filename:
            print("❌ No filename provided.")
            return

        try:
            with open(filename, 'r') as f:
                loaded_settings = json.load(f)

            # Update settings
            result = self.update_settings(loaded_settings)
            print(f"✅ {result['message']}")
        except FileNotFoundError:
            print(f"❌ File {filename} not found.")
        except json.JSONDecodeError:
            print("❌ Invalid JSON format in file.")
        except Exception as e:
            print(f"❌ Error loading settings: {e}")

# Main execution
if __name__ == "__main__":
    client = CoolifyMCPClient()
    client.interactive_mode()
