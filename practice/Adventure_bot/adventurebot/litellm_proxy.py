"""LiteLLM proxy server for Gemini"""

import subprocess
import time
import os
import asyncio
import signal
from pathlib import Path

def start_litellm_proxy(port=8000):
    """Start LiteLLM proxy server for Gemini"""
    
    # Create a simple config for LiteLLM
    config_content = """
model_list:
  - model_name: gemini-2.0-flash-exp
    litellm_params:
      model: gemini/gemini-2.0-flash-exp
      api_key: os.environ/GOOGLE_API_KEY
"""
    
    # Write config to a temporary file
    config_path = Path("litellm_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Start the proxy server
    cmd = [
        "python", "-m", "litellm", "--config", str(config_path), 
        "--port", str(port), "--host", "0.0.0.0"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ LiteLLM proxy server started on port {port}")
            print(f"🔗 OpenAI compatible endpoint: http://localhost:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start LiteLLM proxy")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting LiteLLM proxy: {e}")
        return None

def setup_openai_for_proxy(port=8000):
    """Configure OpenAI to use the LiteLLM proxy"""
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-proxy"
    os.environ["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
    print(f"🔧 OpenAI configured to use LiteLLM proxy at http://localhost:{port}")

async def test_proxy_connection(port=8000):
    """Test if the proxy is working"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/health") as resp:
                if resp.status == 200:
                    print("✅ LiteLLM proxy is healthy")
                    return True
                else:
                    print(f"❌ LiteLLM proxy health check failed: {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to LiteLLM proxy: {e}")
        return False
