"""Configuration for Adventure Bot with Gemini support via LiteLLM"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

"""Configuration for Adventure Bot with Gemini support via LiteLLM"""

import os
import sys
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StderrFilter:
    """Filter to suppress specific error messages from stderr"""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, data):
        # Filter out tracing client errors and non-fatal messages
        if not any(phrase in data for phrase in [
            "Tracing client error 401",
            "[non-fatal]",
            "Incorrect API key provided: dummy-ke"
        ]):
            self.original_stderr.write(data)
            
    def flush(self):
        self.original_stderr.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

def setup_gemini_config():
    """Setup environment for Gemini usage via LiteLLM"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required. Please set it in your .env file.\n"
            "Get your key from: https://makersuite.google.com/app/apikey"
        )
    
    # Set environment variables for LiteLLM to work as OpenAI replacement
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Set a dummy OpenAI key and configure LiteLLM to handle the requests
    os.environ["OPENAI_API_KEY"] = "dummy-key-litellm-handles-this"
    
    # Disable OpenAI tracing to avoid 401 errors
    os.environ["OPENAI_LOG_LEVEL"] = "ERROR"
    os.environ["OPENAI_DISABLE_LOGGING"] = "1"
    
    # Tell LiteLLM to handle all OpenAI requests
    os.environ["LITELLM_LOG"] = "ERROR"  # Reduce LiteLLM logging too
    
    # Apply stderr filtering to suppress tracing errors
    if not hasattr(sys.stderr, '_original_stderr'):
        sys.stderr._original_stderr = sys.stderr
        sys.stderr = StderrFilter(sys.stderr._original_stderr)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    print(f"✅ Google API Key configured")
    print(f"🤖 Using model: {get_gemini_model_name()}")
    print(f"🔧 Environment configured for Gemini via LiteLLM")
    print(f"🤫 Error messages filtered")
    
    return True

def get_gemini_model_name() -> str:
    """Get the Gemini model name to use with LiteLLM prefix"""
    return os.getenv("LITELLM_MODEL", "litellm/gemini/gemini-2.0-flash-exp")
