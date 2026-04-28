#!/usr/bin/env python3
"""Test MiniMax API connectivity."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from src.libs.llm.minimax_llm import MiniMaxLLM
from src.libs.llm.base_llm import Message
from types import SimpleNamespace

def test_minimax():
    """Test MiniMax API."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    base_url = os.environ.get("MINIMAX_API_URL") or os.environ.get("MINIMAX_AI_URL") or "https://api.minimaxi.com/anthropic/v1"
    
    if not api_key:
        print("❌ MINIMAX_API_KEY not found in environment")
        return False
    
    print(f"Testing MiniMax API...")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:20]}...")
    print()
    
    try:
        settings = SimpleNamespace(
            llm=SimpleNamespace(
                model="MiniMax-M2.7-highspeed",
                temperature=0.7,
                max_tokens=128,
                api_key=api_key,
                base_url=base_url,
            )
        )
        llm = MiniMaxLLM(settings=settings, base_url=base_url)
        response = llm.chat([Message(role="user", content="请只回复：测试成功")])
        print("\n✓ MiniMax API test successful!")
        print(f"  Model: {response.model}")
        print(f"  Response: {response.content}")
        return True
    
    except Exception as e:
        print(f"❌ MiniMax API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimax()
    sys.exit(0 if success else 1)
