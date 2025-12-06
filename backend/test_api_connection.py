"""Test DeepSeek API connection without proxy."""
import asyncio
import os
import sys
import httpx
from openai import AsyncOpenAI

# Manually load .env file
def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✓ Loaded environment from: {env_path}\n")
    else:
        print(f"❌ .env file not found at: {env_path}\n")

load_env()

async def test_connection():
    """Test DeepSeek API connection."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in .env file")
        return
    
    print(f"✓ API Key loaded: {api_key[:10]}...")
    print("\n🔍 Testing connection to DeepSeek API...")
    
    # Clear proxy settings
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    for var in proxy_vars:
        if var in os.environ:
            print(f"   Clearing proxy: {var}={os.environ[var]}")
            del os.environ[var]
    
    try:
        # Test with httpx directly
        print("\n📡 Attempting connection...")
        
        async with httpx.AsyncClient(
            timeout=30.0,
            trust_env=False,
            proxies=None,
        ) as client:
            # Initialize OpenAI client
            openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                http_client=client,
            )
            
            # Make a simple test request
            response = await openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello in one word."}
                ],
                max_tokens=10,
            )
            
            answer = response.choices[0].message.content
            print(f"\n✅ SUCCESS! DeepSeek API is reachable")
            print(f"   Response: {answer}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            
    except httpx.ConnectError as e:
        print(f"\n❌ Connection Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify https://api.deepseek.com is accessible in your browser")
        print("   3. Try disabling VPN or firewall temporarily")
        print("   4. Check Windows proxy settings (Settings > Network & Internet > Proxy)")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_connection())

