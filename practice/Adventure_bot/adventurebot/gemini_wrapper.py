"""Custom OpenAI client wrapper that uses LiteLLM for Gemini"""

import os
import asyncio
from typing import Any, Dict, List, Optional
import litellm
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

class GeminiOpenAIWrapper:
    """Wrapper that makes LiteLLM's Gemini calls look like OpenAI API calls"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Set the API key for LiteLLM
        os.environ["GOOGLE_API_KEY"] = self.api_key
    
    async def create_chat_completion(self, **kwargs) -> ChatCompletion:
        """Create a chat completion using LiteLLM's Gemini integration"""
        
        # Extract model name and convert it to LiteLLM format
        model = kwargs.get("model", "gemini/gemini-1.5-flash")
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        
        # Prepare the request for LiteLLM
        litellm_kwargs = {
            "model": model,
            "messages": kwargs.get("messages", []),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", None),
            "stream": False,  # We'll handle streaming separately if needed
        }
        
        try:
            # Make the LiteLLM call
            response = await asyncio.to_thread(litellm.completion, **litellm_kwargs)
            
            # Convert LiteLLM response to OpenAI format
            return self._convert_litellm_response(response)
            
        except Exception as e:
            print(f"Error calling Gemini via LiteLLM: {e}")
            raise
    
    def _convert_litellm_response(self, litellm_response) -> ChatCompletion:
        """Convert LiteLLM response to OpenAI ChatCompletion format"""
        
        # Extract the content from LiteLLM response
        content = ""
        if hasattr(litellm_response, 'choices') and litellm_response.choices:
            choice = litellm_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
        
        # Create OpenAI-compatible response
        return ChatCompletion(
            id=getattr(litellm_response, 'id', 'chatcmpl-gemini'),
            choices=[
                Choice(
                    finish_reason='stop',
                    index=0,
                    message=ChatCompletionMessage(
                        content=content,
                        role='assistant'
                    )
                )
            ],
            created=getattr(litellm_response, 'created', 0),
            model=getattr(litellm_response, 'model', 'gemini-1.5-flash'),
            object='chat.completion',
            usage=CompletionUsage(
                completion_tokens=getattr(litellm_response.usage, 'completion_tokens', 0) if hasattr(litellm_response, 'usage') else 0,
                prompt_tokens=getattr(litellm_response.usage, 'prompt_tokens', 0) if hasattr(litellm_response, 'usage') else 0,
                total_tokens=getattr(litellm_response.usage, 'total_tokens', 0) if hasattr(litellm_response, 'usage') else 0
            )
        )

def patch_openai_with_gemini():
    """Monkey patch the OpenAI client to use Gemini via LiteLLM"""
    
    # Create our wrapper
    wrapper = GeminiOpenAIWrapper()
    
    # Store original methods
    original_create = AsyncOpenAI.chat.completions.create if hasattr(AsyncOpenAI, 'chat') else None
    
    # Replace with our wrapper
    async def gemini_create(*args, **kwargs):
        return await wrapper.create_chat_completion(**kwargs)
    
    # Monkey patch the method
    if original_create:
        AsyncOpenAI.chat.completions.create = gemini_create
    
    print("🔧 OpenAI client patched to use Gemini via LiteLLM")
    return True
