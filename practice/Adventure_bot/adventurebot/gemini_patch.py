"""Simple monkey patch for OpenAI SDK to use LiteLLM with Gemini"""

import os
import asyncio
from typing import Any, Dict, List
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global flag to track if patching is done
_patched = False

def patch_openai_for_gemini():
    """Monkey patch OpenAI SDK to use LiteLLM with Gemini"""
    global _patched
    
    if _patched:
        return
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is required")
    
    # Set the Google API key for LiteLLM
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Set a dummy OpenAI key to satisfy the SDK
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-gemini-via-litellm"
    
    # Import OpenAI after setting the environment
    from openai import AsyncOpenAI
    
    # Store the original create method
    original_create = AsyncOpenAI.chat.completions.create
    
    async def gemini_create(self, **kwargs):
        """Replacement method that uses LiteLLM with Gemini"""
        
        # Convert model name to Gemini format
        model = kwargs.get("model", "gemini-2.0-flash-exp")
        if not model.startswith("gemini/"):
            if "gemini" in model:
                model = f"gemini/{model}"
            else:
                model = "gemini/gemini-1.5-flash"
        
        # Prepare LiteLLM arguments
        litellm_kwargs = {
            "model": model,
            "messages": kwargs.get("messages", []),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "stream": kwargs.get("stream", False),
        }
        
        # Remove None values
        litellm_kwargs = {k: v for k, v in litellm_kwargs.items() if v is not None}
        
        try:
            # Call LiteLLM in a thread to make it async
            result = await asyncio.to_thread(litellm.completion, **litellm_kwargs)
            return result
        except Exception as e:
            print(f"Error calling Gemini via LiteLLM: {e}")
            raise
    
    # Apply the monkey patch
    AsyncOpenAI.chat.completions.create = gemini_create
    
    _patched = True
    print("🔧 OpenAI SDK patched to use Gemini via LiteLLM")
    
    return True
