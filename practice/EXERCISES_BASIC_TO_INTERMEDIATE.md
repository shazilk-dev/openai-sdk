# OpenAI Agents SDK Practice Track (Basic → Intermediate)

A structured, hands-on exercise path to internalize the core patterns from the template library (`agents`, `runner`, `tools`, `sessions`, `structured outputs`, `streaming`, `context`, `guardrails`, `lifecycle hooks`, `handoffs`).

> Focus on **incremental construction**: build only what each exercise demands. Avoid copying full templates—extract the minimal slice.

---

## Legend

- Goal: What you should achieve.
- Concepts: Core ideas reinforced.
- Reference: Template sections to glance at (don’t reread entire file).
- Steps: Minimal implementation sequence.
- Hints: Tactical nudges; ignore at first attempt.
- Validation: How to confirm success (add lightweight asserts/logs).
- Stretch: Optional expansion.
- Reflection: Questions to deepen understanding.
- Automation: (Optional) how you could test it.

---

## Shared Utility (Optional)

Create a tiny helper in `practice/utils/validation.py`:

```python
def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)
```

Use `check()` in exercises instead of prints.

---

# 🔰 BASIC FOUNDATIONS

### 1. Hello Agent Run

Goal: Minimal agent responding to a prompt.
Concepts: Agent instantiation, instructions, basic run.
Reference: `complete-agent-guide.py` → Minimal agent config.
Steps:

1. File: `practice/basic_01_hello_agent.py`.
2. Instantiate agent with concise instructions (tone + purpose).
3. Call `run({"prompt": "Say hello and state your purpose."})`.
4. Print response text.
   Hints: Keep config minimal; no tools; avoid streaming first.
   Validation: Response non-empty; contains purpose phrase.
   Stretch: Override temperature for creative flavor.
   Reflection: What defaults did you implicitly rely on?
   Automation: Assert substring in output.

### 2. Add Your First Tool

Goal: Tool wiring + invocation.
Concepts: @tool decorator, tool discovery, tool execution.
Reference: Tools guide → Basic function tools.
Steps:

1. Define `add_numbers(a: int, b: int) -> int`.
2. Register with agent.
3. Prompt: "Use the tool to add 7 and 15 then explain the result."
4. Inspect tool call evidence (log or trace structure if available).
   Hints: Make tool pure; annotate types clearly.
   Validation: Output includes `22` and an explanation sentence.
   Stretch: Reject negatives (raise), re-run with a negative to see failure.
   Reflection: How would tool exceptions surface?
   Automation: Parse output; confirm integer present.

### 3. Basic Structured Output

Goal: Constrain model output to schema.
Concepts: Pydantic model, validation, repair loop.
Reference: Structured outputs guide → Intro schema enforcement.
Steps:

1. Define `Summary` model: `title: str`, `bullet_points: list[str]` (<=5).
2. Prompt: "Summarize this paragraph...".
3. Attempt parse; if >5 bullets, issue corrective follow-up.
   Hints: Use a simple retry max=1.
   Validation: len(bullet_points) ≤ 5 and non-empty title.
   Stretch: Enforce bullet uniqueness.
   Reflection: Tradeoff: strict schema vs creativity.
   Automation: Assert invariants.

### 4. Session Memory (Short-Lived)

Goal: Preserve conversational state across turns.
Concepts: Session object, message history storage.
Reference: Sessions guide → In-memory session pattern.
Steps:

1. Create a session container (list of {role, content}).
2. Turn 1: greeting.
3. Turn 2: ask favorite framework; user answers.
4. Turn 3: Ask model to recall user preference.
   Hints: Feed prior turns each run.
   Validation: Third response references framework name.
   Stretch: Add session id printed with each turn.
   Reflection: What’s the minimal structure needed?
   Automation: Regex search for framework token.

### 5. Prompt Context Enrichment (Static)

Goal: Add brand voice automatically.
Concepts: Static context injection.
Reference: Context guide → Static layer.
Steps:

1. Build small function returning `{"brand_voice": "friendly analytical"}`.
2. Merge into instructions pre-run.
3. Ask for an onboarding greeting.
   Hints: Keep merge deterministic.
   Validation: Output contains an adjective from brand voice.
   Stretch: Toggle injection with ENV var.
   Reflection: Centralizing style vs per-prompt adjustments.
   Automation: Assert brand keyword present when enabled.

---

# 🧱 CORE SKILLS

### 6. Multi-Tool Selection

Goal: Conditional tool invocation.
Concepts: Tool choice, instruction priming.
Reference: Tools guide → Tool selection heuristics.
Steps:

1. Tools: `add(a,b)`, `slugify(text)`.
2. Ambiguous prompt: convert phrase to slug + add numbers.
3. Provide usage examples in system instructions.
   Hints: Show explicit few-shot pattern.
   Validation: Output shows slug + sum.
   Stretch: Capture ordered list of tool invocations.
   Reflection: How does clarity in instructions affect tool order?
   Automation: Parse slug pattern.

### 7. Runner Orchestration & Cancellation

Goal: Stop a long run mid-execution.
Concepts: Runner control, cancellation.
Reference: Runner guide → Cancellation pattern.
Steps:

1. Long tool: loops 5 times (sleep 0.5s each), yields progress.
2. Start run in a way you can cancel (method or flag).
3. Cancel after 1s; record partial state.
   Hints: Simulate cancellation boolean.
   Validation: Report fewer than 5 cycles.
   Stretch: Log total elapsed ms.
   Reflection: Why early progress emission matters.
   Automation: Assert cycle count < 5.

### 8. Progress Streaming Basics

Goal: Emit incremental text + progress events.
Concepts: Streaming generator, event typing.
Reference: Streaming guide → Event classes.
Steps:

1. Wrap model generation; yield partial text every sentence.
2. Also yield progress at 25/50/75/100.
   Hints: Simple generator composition.
   Validation: Events appear in chronological order.
   Stretch: Add ANSI color tags for progress lines.
   Reflection: Transport independence (websocket, SSE) implications.
   Automation: Count progress milestones.

### 9. Layered Context (User + Dynamic)

Goal: Merge multiple context layers.
Concepts: Composition & precedence.
Reference: Context guide → CompositeContextManager.
Steps:

1. Layers: brand → user profile → dynamic request metadata.
2. Ensure latest overrides duplicate keys.
3. Provide final merged dict to instructions.
   Hints: Keep ordering explicit list.
   Validation: Output references user tier + brand tone.
   Stretch: Print diff when dynamic layer added.
   Reflection: Precedence mistakes risk.
   Automation: Assert merged dict shape.

### 10. Guardrail: Simple Redaction

Goal: Mask sensitive email.
Concepts: Post-generation guardrail, regex.
Reference: Guardrails guide → Redaction pattern.
Steps:

1. Regex match emails; replace middle chars.
2. Prompt includes `alice@example.com`.
3. Apply redaction just before display.
   Hints: Keep reversible?—No (privacy tradeoff).
   Validation: Output hides user local-part partially.
   Stretch: Count redactions metric.
   Reflection: Where in pipeline should redaction live?
   Automation: Assert no plain email appears.

### 11. Lifecycle Hooks (PRE + ERROR)

Goal: Instrument inputs & failures.
Concepts: Hook registry, phases.
Reference: Lifecycle hooks guide → Basic registration.
Steps:

1. PRE-RUN: log prompt.
2. ERROR-TOOL: capture exception into audit list.
3. Tool that raises on `arg == "fail"`.
   Hints: Keep hook payload narrow.
   Validation: Audit list length increments on failure.
   Stretch: Add latency measurement.
   Reflection: Hooks vs decorators tradeoffs.
   Automation: Assert audit entry schema.

### 12. Session Persistence (Disk)

Goal: Resume chat across executions.
Concepts: Persistence boundary.
Reference: Sessions guide → Persistence patterns.
Steps:

1. After run, dump session messages to JSON.
2. On startup, load if exists.
3. Continue conversation referencing earlier fact.
   Hints: Version field for schema.
   Validation: Second invocation recalls prior detail.
   Stretch: Add `last_updated` timestamp.
   Reflection: Schema evolution risk.
   Automation: Assert file created & contains messages.

### 13. Simple Handoff (Escalation)

Goal: Delegate pricing queries.
Concepts: Agent-as-tool, conditional routing.
Reference: Handoffs guide → Delegation.
Steps:

1. Primary agent inspects prompt.
2. If contains pricing keyword, call specialist agent tool.
3. Fuse explanation + specialist answer.
   Hints: Provide reasoning rubric to primary.
   Validation: Pricing prompt triggers specialist only.
   Stretch: Return JSON with `handled_by` field.
   Reflection: Early vs late delegation.
   Automation: Assert path taken.

---

# 🔗 INTERMEDIATE INTEGRATION

### 14. Structured Output + Validation Retry

Goal: Enforce complex invariants.
Concepts: Nested models, repair loop.
Reference: Structured outputs guide → Repair sequence.
Steps:

1. `Plan { steps: list[Step], total:int }`, `Step { order:int, action:str }`.
2. Model generates plan; validate order sequence & count.
3. Retry if mismatch (max 2).
   Hints: Provide explicit correction instruction.
   Validation: Orders strictly 1..n & n == total.
   Stretch: Add `schema_version`.
   Reflection: When to abort vs retry.
   Automation: Assert invariants.

### 15. Context Middleware (Security + Performance)

Goal: Transform & annotate context.
Concepts: Middleware chain, timing.
Reference: Context guide → Middleware pipeline.
Steps:

1. Security middleware: replace banned terms.
2. Performance: measure build ms, attach metric.
3. Run with term triggering sanitization.
   Hints: Compose sequentially.
   Validation: Output sanitized & metric present.
   Stretch: Warn if > threshold ms.
   Reflection: Where to persist metrics.
   Automation: Assert metric key exists.

### 16. Streaming + Progress + Hooks Combined

Goal: Unified observability.
Concepts: Event interception, hook counters.
Reference: Streaming + Lifecycle hooks.
Steps:

1. PRE-RUN: init counter.
2. Wrap stream; increment on each text event.
3. POST-RUN: append summary chunk count.
   Hints: Use wrapper generator.
   Validation: Reported chunk count accurate.
   Stretch: Add progress events at 25/50/75.
   Reflection: Coupling risk between layers.
   Automation: Assert counter == emitted events.

### 17. Guardrails + Tool Fallback

Goal: Policy-based fallback path.
Concepts: Policy detection, alternate execution.
Reference: Guardrails guide → Policy violation handling.
Steps:

1. Tool returns numeric list.
2. Guardrail flags if any value > 1000.
3. If flagged → call summarizer tool.
   Hints: Keep violation object structured.
   Validation: Fallback triggered only on violation.
   Stretch: Log `violations.json`.
   Reflection: False positive risk.
   Automation: Run with safe & unsafe inputs.

### 18. Latency Budget Plugin

Goal: Enforce latency awareness.
Concepts: Plugin hook, timing, soft violation.
Reference: Lifecycle hooks guide → LatencyBudgetPlugin pattern.
Steps:

1. Register plugin with 500ms budget.
2. Tool sleeps 0.7s.
3. Plugin records violation.
   Hints: Store violation in shared list.
   Validation: Violation recorded; run completes.
   Stretch: Append warning to user response.
   Reflection: Soft vs hard limits.
   Automation: Assert violation length == 1.

### 19. Composite Context + Structured Output

Goal: Return answer + context trace.
Concepts: Context introspection + typed output.
Reference: Context + Structured outputs guides.
Steps:

1. Collect ordered layer keys & raw dicts.
2. Output model with `answer` + `context_layers`.
3. Hash each layer dict (sha256) for integrity.
   Hints: Serialize layer before hash.
   Validation: Each layer has a hash; order preserved.
   Stretch: Add signature placeholder field.
   Reflection: Reproducibility benefits.
   Automation: Assert number of layers == expected.

### 20. Mini Capability Matrix

Goal: Intent → capability routing.
Concepts: Routing table, fallback message.
Reference: Tools + Handoffs guides.
Steps:

1. Map: `math` → tool, `pricing` → specialist agent, `format` → tool.
2. Simple classifier (keyword scan) chooses path.
3. Return structured result `{capability, output}`.
   Hints: Keep classifier deterministic.
   Validation: 3 prompts route differently.
   Stretch: Add `confidence` score.
   Reflection: When to adopt semantic classifier.
   Automation: Assert routing correctness.

### 21. Resilient Tool Retry

Goal: Retry flaky tool safely.
Concepts: Backoff, exception handling.
Reference: Tools guide → Retry patterns.
Steps:

1. Tool raises randomly (e.g., `random.random()<0.5`).
2. Retry up to 3 with exponential backoff (0.1, 0.2, 0.4s).
3. On final failure, return graceful fallback message.
   Hints: Isolate retry logic in helper.
   Validation: Either success or clear fallback path.
   Stretch: Collect attempt metrics.
   Reflection: Distinguish transient vs fatal.
   Automation: Seed RNG for determinism.

### 22. Session + Guardrail Interaction

Goal: Cumulative policy enforcement.
Concepts: Cross-turn state + threshold blocking.
Reference: Sessions + Guardrails guides.
Steps:

1. Track `sensitive_count` in session.
2. Increment per redaction.
3. On count >3, block and return advisory message.
   Hints: Keep threshold configurable.
   Validation: 4th sensitive attempt blocked.
   Stretch: Implement cooldown timestamp.
   Reflection: Abuse mitigation patterns.
   Automation: Simulate 5 violations.

### 23. Audit Trail Assembly

Goal: Persist lifecycle events chronologically.
Concepts: Structured logging, trace correlation.
Reference: Lifecycle hooks guide.
Steps:

1. Hooks push dict entries: prompt, tool_start, tool_error, run_end.
2. At end, write `audit_<timestamp>.json`.
3. Include `correlation_id`.
   Hints: Use `uuid4()` once per run.
   Validation: File exists; JSON array ordered.
   Stretch: Add hashing for each record.
   Reflection: Forensic value.
   Automation: Parse JSON; assert keys.

### 24. Basic Performance Benchmark

Goal: Compare latency with context layers.
Concepts: Micro-benchmarking, variance awareness.
Reference: Runner + Context guides.
Steps:

1. Baseline: run simple prompt 5 times (record times).
2. Add 3 context layers; run 5 times.
3. Compute mean + delta.
   Hints: Use `time.perf_counter()`.
   Validation: Report table with both means.
   Stretch: Add std deviation.
   Reflection: When to optimize context build.
   Automation: Assert sample sizes.

---

## Suggested Execution Order

1 → 10 (foundations) → 14,15,16 (integration) → remaining based on interest (security vs observability vs resilience).

---

## Progress Tracking Sheet (Copy & Mark)

| #   | Exercise             | Date | Complete | Notes |
| --- | -------------------- | ---- | -------- | ----- |
| 1   | Hello Agent Run      |      |          |       |
| 2   | First Tool           |      |          |       |
| 3   | Structured Output    |      |          |       |
| 4   | Session Memory       |      |          |       |
| 5   | Static Context       |      |          |       |
| 6   | Multi-Tool Selection |      |          |       |
| 7   | Cancellation         |      |          |       |
| 8   | Streaming Basics     |      |          |       |
| 9   | Layered Context      |      |          |       |
| 10  | Redaction Guardrail  |      |          |       |
| 11  | Hooks PRE + ERROR    |      |          |       |
| 12  | Session Persistence  |      |          |       |
| 13  | Simple Handoff       |      |          |       |
| 14  | Validation Retry     |      |          |       |
| 15  | Context Middleware   |      |          |       |
| 16  | Streaming + Hooks    |      |          |       |
| 17  | Guardrail Fallback   |      |          |       |
| 18  | Latency Budget       |      |          |       |
| 19  | Context + Structured |      |          |       |
| 20  | Capability Matrix    |      |          |       |
| 21  | Retry Logic          |      |          |       |
| 22  | Session Threshold    |      |          |       |
| 23  | Audit Trail          |      |          |       |
| 24  | Perf Benchmark       |      |          |       |

---

## After Finishing

Request the **Advanced Series** (multi-agent planning graph, retrieval augmentation, sagas, vector cache, circuit breakers, semantic guardrails, streaming over WS, production observability).

Let me know if you’d like: (a) an automated harness to verify these, (b) solution outlines, or (c) an advanced roadmap.
