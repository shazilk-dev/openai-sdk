# OpenAI Agents SDK Comprehensive Template Library

This index links to every deep-dive template and foundational guide included in this repository so you can jump directly to the domain you want to master.

> Each Python template is a **single, self‑contained, heavily commented reference file** demonstrating patterns, architecture, and best practices. Skim top-level sections first; then run or extract the parts you need.

## 🗂 Core Agent Building Blocks

| Domain                 | File                                                          | What You Get                                                                     |
| ---------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Agent Fundamentals     | `src/agents/complete-agent-guide.py`                          | Core agent anatomy (config, planning, tool use, error handling, streaming stubs) |
| Runner Orchestration   | `src/runner/complete-runner-guide.py`                         | Execution lifecycle, streaming integration, cancellation, tracing hooks          |
| Tools & Function Calls | `src/tools/complete-tools-guide.py`                           | Tool registry, dependency injection, tool patterns, error surfaces, validation   |
| Session Patterns       | `src/sessions/complete-sessions-guide.py`                     | Conversation/session state lifecycles, memory layers, persistence strategies     |
| Structured Outputs     | `src/structured-outputs/complete-structured-outputs-guide.py` | Pydantic / typed outputs, schema evolution, validation + repair loops            |

## 🔀 Advanced Interaction & Control

| Domain                 | File                                                    | What You Get                                                                  |
| ---------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Handoffs & Multi-Agent | `src/handoffs/complete-handoffs-guide.py`               | Role transfer, delegation trees, arbitration, agent-as-tool wiring            |
| Lifecycle Hooks        | `src/lifecycle_hooks/complete-lifecycle-hooks-guide.py` | Hook registry, domains/phases, composite hooks, plugins, audit & metrics      |
| Guardrails & Policy    | `src/guardrails/complete-guardrails-guide.py`           | Content filters, semantic policies, redaction, structured violation reporting |
| Context Management     | `src/context/complete-context-guide.py`                 | Layered context (static/dynamic/user/business/session), middleware pipeline   |
| Streaming Patterns     | `src/streaming/complete-streaming-guide.py`             | Sync/async streaming, event envelopes, progress tracking, UI adapters         |

## ⚙️ Model Configuration Layer

| Scope                | File                               | Description                                              |
| -------------------- | ---------------------------------- | -------------------------------------------------------- |
| Global Defaults      | `src/model-config/Global-level.py` | Org / platform baseline config & safety posture          |
| Agent Overrides      | `src/model-config/Agent-level.py`  | Per-agent specialization, targeted capability tuning     |
| Runner-Level Merging | `src/model-config/Runner-level.py` | Resolution order, dynamic model swapping, fallback logic |

## 📘 Foundational Knowledge (Read First If New)

| Guide                | File                                    | Description                                                                   |
| -------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| Python Prerequisites | `../PYTHON_PREREQUISITES_FOR_AGENTS.md` | Modern Python & ecosystem concepts required to leverage templates effectively |

## 🧪 Suggested Learning Path

1. Read: Python Prerequisites (skim & fill gaps).
2. Build a minimal agent: use `complete-agent-guide.py` (Sections: Config → Basic Run → Tool Execution).
3. Add tools with validation & error surfaces.
4. Introduce structured outputs (Pydantic schemas) for reliability.
5. Add sessions & persistence.
6. Layer in streaming for UX responsiveness.
7. Introduce context layering & middleware for dynamic personalization.
8. Add guardrails (policy + redaction) for safety.
9. Instrument lifecycle hooks for observability & plugins.
10. Scale to multi-agent handoffs when orchestration complexity grows.

## 🛠 Running the Templates

Each file is runnable as a script:

```
python path/to/file.py
```

If a template defines a `main()` or `if __name__ == "__main__"` block, executing the file will print a demo scenario. Many templates separate concerns into `*_examples()` functions—adapt these into tests or notebooks.

## 🔍 How To Extract Patterns

| Need                            | Where to Look                                  | Notes                                                             |
| ------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| Add a new tool quickly          | Tools guide → "@tool decorator patterns"       | Copy minimal function tool scaffold                               |
| Stream incremental output to UI | Streaming guide → `StreamingUIAdapter`         | Replace mock UI with your framework (FastAPI / WebSocket / React) |
| Enforce per-call policy         | Guardrails guide → Policy evaluation pipeline  | Combine semantic + pattern filters                                |
| Add custom metrics              | Lifecycle hooks guide → Metrics aggregation    | Register a post-run hook emitting counters                        |
| Merge dynamic business context  | Context guide → `CompositeContextManager`      | Inject user/session + inventory/pricing layers                    |
| Validate JSON output            | Structured outputs guide → Repair loop section | Shows retry with schema diff commentary                           |
| Escalate to specialist agent    | Handoffs guide → Delegation patterns           | Use agent-as-tool with reason + trace                             |

## 🧩 Composability Checklist

- Idempotent tools (pure or side-effect isolated)
- Deterministic structured output schemas (versioned)
- Context layers clearly ordered with precedence
- Guardrails execute before tool dispatch & before final emission
- Hooks instrument PRE, POST, ERROR domains (run + tool at minimum)
- Streaming events separated (text vs progress vs error) for front-end flexibility
- Model config resolved in predictable (Global → Agent → Runner override) order

## 🔐 Security & Governance Quick Pointers

| Concern                     | Pattern                                   | Location                                  |
| --------------------------- | ----------------------------------------- | ----------------------------------------- |
| Sensitive data masking      | Mask plugin & guardrail redaction         | Guardrails + Lifecycle hooks plugin       |
| Tool abuse prevention       | Capability allowlist + role-based filters | Tools + Guardrails                        |
| Prompt injection mitigation | Context sanitization middleware           | Context guide (SecurityContextMiddleware) |
| Auditability                | Structured `HookContext` + audit log list | Lifecycle hooks guide                     |
| Latency control             | Budget plugin + progress tracking         | Lifecycle + Streaming                     |

## 🧭 Extending Further

| Extension                    | Start From               | Idea                                                  |
| ---------------------------- | ------------------------ | ----------------------------------------------------- |
| Vector retrieval layer       | Context + Tools guides   | Add retrieval tool; insert pre-run context enrichment |
| Multi-modal inputs           | Agent + Streaming guides | Extend event schema w/ image/audio chunks             |
| Persistent memory store      | Sessions guide           | Replace in-memory store w/ Redis or Postgres          |
| Reversible tool transactions | Tools + Handoffs         | Add compensating actions & saga pattern               |
| SLA monitoring               | Lifecycle hooks          | Emit metrics to Prometheus, add alert thresholds      |

## ✅ Verification Strategy (When Adopting)

1. Unit test each tool (pure logic) with edge cases.
2. Contract test structured outputs (golden schema fixtures).
3. Property test context merging (precedence invariants).
4. Chaos test tool failures (timeouts, exceptions) – ensure hooks fire.
5. Load test streaming under concurrency (progress events remain ordered).
6. Security test injection attempts against context sanitization.
7. Regression test handoff chains (no orphaned agent states).

## 📦 Suggested Project Layout (Adapting These)

```
project/
  src/
    agents/
    tools/
    context/
    hooks/
    guardrails/
    sessions/
    streaming/
    config/
  tests/
    unit/
    integration/
    contracts/
  infra/
    docker/
    k8s/
```

## 🏁 Next Optional Enhancements (Not Yet Included)

| Idea                                                               | Value                               |
| ------------------------------------------------------------------ | ----------------------------------- |
| Central test harness invoking all example `*_examples()` functions | Quick regression & smoke validation |
| Benchmark harness for latency budgets                              | Capacity planning                   |
| Real WebSocket streaming demo (FastAPI + front-end)                | Production integration reference    |
| Vector retrieval augmentation example                              | Demonstrates RAG + tool synergy     |
| Deployment recipes (Docker, K8s, autoscaling)                      | Infra acceleration                  |

---

If you’d like an automated script to validate all templates or generate a docs site index, ask and we can scaffold it.
