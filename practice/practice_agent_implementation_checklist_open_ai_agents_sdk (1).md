# Practice Agent Implementation Checklist (OpenAI Agents SDK)

## Stubbed Project File Tree

```
practice-agent-sdk/
│
├── .env.example               # API keys and secrets (copy to .env)
├── README.md                  # Project overview and checklist (this doc)
├── requirements.txt           # Python deps (openai, dotenv, etc.)
├── scripts/
│   ├── smoke.py                # Verify SDK import + env setup
│   ├── run_once.py             # Run one agent request (non-streamed)
│   └── run_streamed.py         # Run agent with streaming output
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── runner.py           # Runner singleton w/ tracing enabled (TODO)
│   │   ├── instructions.py     # Dynamic instruction builder (TODO)
│   │   ├── context.py          # UserContext + RunContext models (TODO)
│   │   ├── main_agent.py       # Main agent definition (TODO)
│   │   ├── research_agent.py   # Specialist: research/handoff (TODO)
│   │   └── task_agent.py       # Specialist: fast tasks (TODO)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_docs.py      # Mock doc search tool (TODO)
│   │   ├── get_weather.py      # Mock weather tool (TODO)
│   │   ├── save_note.py        # Save to file/db tool (TODO)
│   │   └── calc.py             # Safe eval calculator tool (TODO)
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guards.py     # Validate input, block disallowed intents (TODO)
│   │   ├── output_guards.py    # Validate output JSON/schema, scrub PII (TODO)
│   │   └── approval.py         # Human-in-the-loop approval logic (TODO)
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── lifecycle.py        # Hook handlers: on_agent_start, on_tool_start, etc. (TODO)
│   │   └── event_bus.py        # Subscribe/listen to events (TODO)
│   │
│   └── utils/
│       ├── tracing.py          # Trace helpers, span wrappers (TODO)
│       └── model_policy.py     # Smart vs fast model selector (TODO)
│
└── tests/
    ├── test_tools.py           # Unit tests for tools (TODO)
    ├── test_guardrails.py      # Unit tests for guardrails (TODO)
    ├── test_agents.py          # Integration tests for main + sub agents (TODO)
    └── test_streaming.py       # Verify streaming events + lifecycle hooks (TODO)
```

## Next Step
Each `TODO` file should start with minimal imports + placeholder functions/classes. Would you like me to auto‑generate **boilerplate code with docstrings + TODO comments** for each file in this tree?

