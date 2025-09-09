# Practice Agent Implementation Checklist (OpenAI Agents SDK)

## Stubbed Project File Tree with Boilerplate Code

```
practice-agent-sdk/
│
├── .env.example
│   # OPENAI_API_KEY=your_api_key_here
│
├── README.md
│   # This checklist & instructions live here.
│
├── requirements.txt
│   openai
│   python-dotenv
│   pytest
│
├── scripts/
│   ├── smoke.py
│   """Quick SDK import test."""
│   import openai
│   print("OpenAI SDK imported successfully.")
│
│   ├── run_once.py
│   """Run one agent request (non-streamed)."""
│   from src.agent.runner import runner
│   from src.agent.main_agent import main_agent
│
│   if __name__ == "__main__":
│       response = runner.run(agent=main_agent, input="Hello Agent!")
│       print(response)
│
│   └── run_streamed.py
│   """Run agent with streaming output."""
│   from src.agent.runner import runner
│   from src.agent.main_agent import main_agent
│
│   if __name__ == "__main__":
│       for event in runner.run_streamed(agent=main_agent, input="Stream please"):
│           print(event)
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   """Runner singleton with tracing enabled."""
│   │   from openai import Runner
│   │   runner = Runner()
│   │
│   │   ├── instructions.py
│   │   """Dynamic instruction builder."""
│   │   def build_instructions(user_context, run_context):
│   │       return f"You are a helpful agent for {user_context.get('name','user')}."
│   │
│   │   ├── context.py
│   │   """Context models for agent runs."""
│   │   class UserContext:
│   │       def __init__(self, user_id, name, locale="en-US", goals=None):
│   │           self.user_id = user_id
│   │           self.name = name
│   │           self.locale = locale
│   │           self.goals = goals or []
│   │
│   │   class RunContext:
│   │       def __init__(self, run_id, prefer_smart_model=False):
│   │           self.run_id = run_id
│   │           self.prefer_smart_model = prefer_smart_model
│   │
│   │   ├── main_agent.py
│   │   """Main agent definition."""
│   │   from src.agent.instructions import build_instructions
│   │   from src.agent.context import UserContext, RunContext
│   │   from src.tools import search_docs, get_weather, save_note, calc
│   │   from src.guardrails import input_guards, output_guards
│   │
│   │   class MainAgent:
│   │       name = "main-agent"
│   │       tools = [search_docs.tool, get_weather.tool, save_note.tool, calc.tool]
│   │
│   │       def get_instructions(self, user_context, run_context):
│   │           return build_instructions(user_context.__dict__, run_context.__dict__)
│   │
│   │   main_agent = MainAgent()
│   │
│   │   ├── research_agent.py
│   │   """Specialist agent for research."""
│   │   class ResearchAgent:
│   │       name = "research-agent"
│   │   research_agent = ResearchAgent()
│   │
│   │   └── task_agent.py
│   │   """Specialist agent for fast tasks."""
│   │   class TaskAgent:
│   │       name = "task-agent"
│   │   task_agent = TaskAgent()
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   from . import search_docs, get_weather, save_note, calc
│   │
│   │   ├── search_docs.py
│   │   def tool(query: str):
│   │       """Search local docs (mock)."""
│   │       return {"results": [f"Result for {query}"]}
│   │
│   │   ├── get_weather.py
│   │   def tool(city: str):
│   │       """Mock weather API."""
│   │       return {"city": city, "forecast": "sunny"}
│   │
│   │   ├── save_note.py
│   │   def tool(text: str):
│   │       """Save text to file."""
│   │       with open("notes.txt", "a") as f:
│   │           f.write(text + "\n")
│   │       return {"status": "saved"}
│   │
│   │   └── calc.py
│   │   def tool(expression: str):
│   │       """Evaluate math expression safely."""
│   │       try:
│   │           result = eval(expression, {"__builtins__": {}})
│   │           return {"result": result}
│   │       except Exception as e:
│   │           return {"error": str(e)}
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   from . import input_guards, output_guards, approval
│   │
│   │   ├── input_guards.py
│   │   def check_input(user_input: str):
│   │       if "forbidden" in user_input:
│   │           raise ValueError("Input not allowed.")
│   │       return True
│   │
│   │   ├── output_guards.py
│   │   def validate_output(output: dict):
│   │       if not isinstance(output, dict):
│   │           raise ValueError("Output must be dict.")
│   │       return True
│   │
│   │   └── approval.py
│   │   def require_approval(tool_name: str):
│   │       if tool_name in ["save_note"]:
│   │           print(f"Approval required for {tool_name}.")
│   │           return False
│   │       return True
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   from . import lifecycle, event_bus
│   │
│   │   ├── lifecycle.py
│   │   def on_agent_start(agent_name):
│   │       print(f"[HOOK] Agent {agent_name} started.")
│   │   def on_agent_end(agent_name):
│   │       print(f"[HOOK] Agent {agent_name} ended.")
│   │
│   │   └── event_bus.py
│   │   subscribers = {}
│   │   def subscribe(event, fn):
│   │       subscribers.setdefault(event, []).append(fn)
│   │   def emit(event, data):
│   │       for fn in subscribers.get(event, []):
│   │           fn(data)
│   │
│   └── utils/
│       ├── tracing.py
│       def start_trace(run_id):
│           print(f"Trace started for {run_id}")
│       def end_trace(run_id):
│           print(f"Trace ended for {run_id}")
│
│       └── model_policy.py
│       def choose_model(prefer_smart: bool):
│           return "gpt-4" if prefer_smart else "gpt-4o-mini"
│
└── tests/
    ├── test_tools.py
    def test_calc():
        from src.tools import calc
        assert "result" in calc.tool("2+2")

    ├── test_guardrails.py
    def test_input_guard():
        from src.guardrails import input_guards
        assert input_guards.check_input("ok")

    ├── test_agents.py
    def test_main_agent_instructions():
        from src.agent.main_agent import main_agent
        from src.agent.context import UserContext, RunContext
        uc, rc = UserContext("u1","Test"), RunContext("r1")
        assert "Test" in main_agent.get_instructions(uc, rc)

    └── test_streaming.py
    def test_stream_events():
        from src.agent.runner import runner
        from src.agent.main_agent import main_agent
        events = list(runner.run_streamed(agent=main_agent, input="hello"))
        assert isinstance(events, list)
```

---

Each file now has **boilerplate + docstrings + TODO-ready placeholders**. 
Would you like me to expand **streaming + lifecycle hooks** into a more detailed event-handling demo (with custom event types)?

