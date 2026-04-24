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

from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.base_llm import Message
from types import SimpleNamespace

def test_minimax():
    """Test MiniMax API."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    base_url = os.environ.get("MINIMAX_API_URL", "https://api.minimax.io/v1")
    
    if not api_key:
        print("❌ MINIMAX_API_KEY not found in environment")
        return False
    
    print(f"Testing MiniMax API...")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:20]}...")
    print()
    
    # Try direct HTTP request first to debug
    import httpx
    
    try:
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "abab6.5s-chat",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        print(f"Making request to: {url}")
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
            if response.status_code == 200:
                print("\n✓ MiniMax API test successful!")
                data = response.json()
                if "choices" in data:
                    print(f"  Response: {data['choices'][0]['message']['content']}")
                return True
            else:
                print(f"\n❌ API returned error: {response.status_code}")
                return False
    
    except Exception as e:
        print(f"❌ MiniMax API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimax()
    sys.exit(0 if success else 1)
