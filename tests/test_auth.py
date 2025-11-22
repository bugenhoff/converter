#!/usr/bin/env python3
"""Quick test for authorization settings."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings
from src.bot.auth import check_user_access

def test_auth():
    """Test authorization functionality."""
    print("🔐 Testing Authorization Settings")
    print(f"ALLOWED_USERS_ONLY: {settings.allowed_users_only}")
    print(f"ALLOWED_USER_IDS: {settings.allowed_user_ids}")
    print()
    
    # Test cases
    test_cases = [
        (123456789, "First allowed user"),
        (987654321, "Second allowed user"),  
        (111111111, "Unauthorized user"),
        (999999999, "Another unauthorized user")
    ]
    
    for user_id, description in test_cases:
        has_access = check_user_access(user_id)
        status = "✅ ALLOWED" if has_access else "❌ DENIED"
        print(f"{status} User {user_id} ({description})")
    
    print()
    if settings.allowed_users_only:
        print("🔒 Bot is in RESTRICTED mode - only authorized users can access")
    else:
        print("🌍 Bot is in PUBLIC mode - all users can access")

if __name__ == "__main__":
    test_auth()