import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY in your .env file")
    exit(1)

# Define a simple tool for our agent
@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    # This is a mock function - in real apps, you'd call a weather API
    return f"The weather in {city} is sunny with a temperature of 22°C"

@function_tool  
def calculate(expression: str) -> str:
    """Safely calculate mathematical expressions."""
    try:
        # Safe evaluation of basic math expressions
        result = eval(expression.replace('^', '**'))
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"

def main():
    print("🤖 Starting your first OpenAI Agent!")
    print("=" * 50)
    
    # Create an agent with tools
    agent = Agent(
        name="Assistant",
        instructions="""
        You are a helpful assistant that can:
        1. Check weather for cities
        2. Perform calculations
        3. Have friendly conversations
        
        Always be helpful and concise in your responses.
        """,
        tools=[get_weather, calculate]
    )
    
    examples = [
        "Hello! What can you do?",
        "What's the weather like in Tokyo?", 
        "Calculate 15 * 7 + 23",
        "Tell me a fun fact about Python programming"
    ]
    
    for i, user_input in enumerate(examples, 1):
        print(f"\n🗣️  User: {user_input}")
        
        # Run the agent synchronously
        try:
            result = Runner.run_sync(agent, user_input)
            print(f"🤖 Agent: {result.final_output}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Add separator between conversations
        if i < len(examples):
            print("-" * 30)

async def async_example():
    """Demonstrate async usage of the agent"""
    print("\n🔄 Async Example:")
    print("=" * 50)
    
    agent = Agent(
        name="AsyncAssistant", 
        instructions="You are a helpful assistant. Keep responses concise.",
        tools=[get_weather, calculate]
    )
    
    # Run multiple queries concurrently
    tasks = [
        Runner.run(agent, "What's 100 / 5?"),
        Runner.run(agent, "Weather in Paris?"),
        Runner.run(agent, "What's the capital of Japan?")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"🤖 Response {i}: {result.final_output}")

if __name__ == "__main__":
    # Run the main synchronous examples
    main()
    
    # Run the async example
    print("\n" + "=" * 60)
    asyncio.run(async_example())
