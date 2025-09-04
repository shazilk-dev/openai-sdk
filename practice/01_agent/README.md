# 🤖 My First OpenAI Agent

This project demonstrates the fundamentals of the OpenAI Agents SDK using the DACA (Dapr Agentic Cloud Ascent) learning approach.

## 🚀 Quick Start

1. **Set up your OpenAI API key** in `.env`:

   ```bash
   OPENAI_API_KEY=your_actual_api_key_here
   ```

2. **Run the agent**:
   ```bash
   python main.py
   ```

## 🧠 What You'll Learn

### Core OpenAI Agents SDK Concepts

1. **Agent Creation**: How to define an agent with instructions and tools
2. **Function Tools**: Converting Python functions into agent tools using `@function_tool`
3. **Runner Patterns**: Both sync (`Runner.run_sync`) and async (`Runner.run`) execution
4. **Agent Loop**: Understanding how the agent processes requests and tool calls

### Key Code Patterns

#### Basic Agent

```python
from agents import Agent, Runner, function_tool

agent = Agent(
    name="Assistant",
    instructions="You are helpful",
    tools=[my_tool]
)

result = Runner.run_sync(agent, "Hello!")
```

#### Function Tools

```python
@function_tool
def my_tool(param: str) -> str:
    """Tool description for the agent."""
    return f"Processed: {param}"
```

## 🎯 Understanding the Code

### Agent Design Decisions

1. **Why dataclass for Agent?**

   - Immutable configuration
   - Easy serialization and cloning
   - Type safety

2. **Why instructions as string/callable?**

   - Static instructions: Simple string
   - Dynamic instructions: Function that returns string based on context

3. **Why tools as list?**
   - Composable capabilities
   - Agent can choose which tool to use
   - Easy to add/remove tools

### Runner Class Design

1. **Why classmethod for run()?**

   - Stateless execution
   - Can be customized with AgentRunner
   - Separates agent definition from execution

2. **Why user prompt in run() not Agent?**
   - Agent defines "what" (capabilities)
   - Runner defines "how" (conversation flow)
   - Enables conversation state management

## 🔄 The Agent Loop

When you call `Runner.run()`, this happens:

1. **LLM Call**: Agent processes input with current context
2. **Response Analysis**: Check for final output, tool calls, or handoffs
3. **Tool Execution**: If tools called, execute and append results
4. **Loop**: Repeat until final output or max_turns reached

## 🛠️ Extending This Example

Try these modifications:

1. **Add more tools**:

   ```python
   @function_tool
   def search_web(query: str) -> str:
       # Implement web search
       return f"Search results for: {query}"
   ```

2. **Use structured output**:

   ```python
   from pydantic import BaseModel

   class Response(BaseModel):
       answer: str
       confidence: float

   agent.output_type = Response
   ```

3. **Add context**:
   ```python
   result = Runner.run_sync(
       agent,
       "Hello",
       context={"user_id": "123", "session": "abc"}
   )
   ```

## 🌍 DACA Integration Ready

This basic agent is ready for DACA deployment:

- ✅ **Containerizable**: Works in Docker
- ✅ **Stateless**: No local state dependencies
- ✅ **Cloud-Native**: Environment-based configuration
- ✅ **Observable**: Built-in tracing support

Next steps in your DACA journey:

1. **Containerize** this agent
2. **Add FastAPI** for HTTP endpoints
3. **Deploy** to Azure Container Apps
4. **Scale** with Kubernetes

## 📚 Learning Resources

- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
- [DACA Design Pattern Guide](../learn-agentic-ai/comprehensive_guide_daca.md)
- [Agent Examples](../learn-agentic-ai/01_ai_agents_first/)

## ❓ Common Questions

**Q: Why not just use OpenAI API directly?**
A: The Agents SDK provides the agent loop, tool calling, handoffs, and built-in patterns that you'd otherwise implement manually.

**Q: How does this scale?**
A: With DACA principles - containerize, use stateless design, deploy on Kubernetes with Dapr for distributed coordination.

**Q: Can I use different LLM providers?**
A: Yes! The SDK supports 100+ LLM providers through its model abstraction.
