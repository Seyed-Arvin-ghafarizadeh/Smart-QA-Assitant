"""Test script for DeepSeek API using OpenAI SDK."""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.services.llm_service import LLMService
from app.utils.logger import logger


async def test_deepseek_api():
    """Test DeepSeek API connection and response."""
    print("=" * 60)
    print("Testing DeepSeek API with OpenAI SDK")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n❌ ERROR: DEEPSEEK_API_KEY environment variable not set!")
        print("Please set it using: export DEEPSEEK_API_KEY='your-api-key'")
        return False
    
    print(f"\n✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize LLM service
        print("\n📡 Initializing LLM Service...")
        llm_service = LLMService(
            api_key=api_key,
            api_url="https://api.deepseek.com/v1/chat/completions",
            model="deepseek-chat"
        )
        print("✓ LLM Service initialized successfully")
        
        # Test with a simple question and mock chunks
        print("\n🧪 Testing API call with sample data...")
        test_question = "What is artificial intelligence?"
        test_chunks = [
            {
                "text": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
                "page_number": 1
            },
            {
                "text": "AI systems can learn, reason, and make decisions based on data and patterns. Machine learning is a subset of AI that enables systems to improve from experience.",
                "page_number": 2
            }
        ]
        
        print(f"Question: {test_question}")
        print(f"Context chunks: {len(test_chunks)} chunks provided")
        
        # Make API call
        result = await llm_service.generate_answer(
            question=test_question,
            chunks=test_chunks
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("✅ API Test Successful!")
        print("=" * 60)
        print(f"\n📝 Answer:")
        print(f"{result['answer']}")
        print(f"\n📊 Token Usage:")
        print(f"  - Prompt tokens: {result['token_usage'].get('prompt_tokens', 'N/A')}")
        print(f"  - Completion tokens: {result['token_usage'].get('completion_tokens', 'N/A')}")
        print(f"  - Total tokens: {result['token_usage'].get('total_tokens', 'N/A')}")
        print(f"\n⏱️  Response Time: {result['response_time_ms']:.2f} ms")
        
        # Cleanup
        await llm_service.close()
        print("\n✓ Test completed successfully!")
        return True
        
    except ValueError as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_deepseek_api())
    sys.exit(0 if success else 1)

