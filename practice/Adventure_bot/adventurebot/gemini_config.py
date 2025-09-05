"""
Configuration for using Google Gemini API with OpenAI Agents SDK
"""
import os
import google.generativeai as genai
from agents.models import ModelProvider
from agents.models.gemini_provider import GeminiProvider

def configure_gemini():
    """Configure Gemini API with the API key from environment variables"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    return api_key

def get_gemini_model_provider() -> ModelProvider:
    """Get a ModelProvider configured for Gemini"""
    configure_gemini()
    return GeminiProvider()
