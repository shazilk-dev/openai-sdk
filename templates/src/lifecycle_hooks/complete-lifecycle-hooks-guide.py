"""
🪝 OPENAI AGENTS SDK - COMPLETE LIFECYCLE HOOKS GUIDE & TEMPLATE

This template is the final component of the comprehensive template library.
It covers every aspect of Lifecycle Hooks in the OpenAI Agents SDK:

SECTIONS:
1. Conceptual Overview of Hooks & Events
2. Core Hook Types (agent-level, run-level, tool-level, session-level)
3. Hook Registration Patterns (declarative, programmatic, mixins)
4. Synchronous vs Asynchronous Hooks
5. Ordered & Conditional Execution
6. Error Handling Strategies for Hooks
7. Metrics, Tracing & Observability Hooks
8. Audit, Compliance & Security Hooks
9. Advanced Hook Composition & Chaining
10. Plugin Architecture via Hooks
11. Testing & Validation of Hooks
12. Performance & Optimization Patterns
13. Best Practices & Anti-Patterns

WHY LIFECYCLE HOOKS?
- Extend agent behavior without modifying core logic
- Inject cross-cutting concerns (logging, metrics, auth)
- Enable modular observability & auditing
- Support A/B experiments & dynamic feature flags
- Provide governance & compliance enforcement

NOTE: This template is framework-agnostic but matches patterns used in OpenAI Agents SDK v0.2.x.
"""

from __future__ import annotations

import time
import json
import uuid
import traceback
import inspect
import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from enum import Enum, auto

from agents import Agent, Runner, function_tool  # Assumed SDK imports

# =============================================================================
# 1. CORE CONCEPTS & TERMINOLOGY
# =============================================================================
"""
LIFECYCLE DOMAINS:

Agent Lifecycle:
  - on_agent_init: Agent object creation
  - on_agent_configured: After tools / model set
  - on_agent_dispose: Cleanup

Run Lifecycle:
  - before_run_start: Prior to main planning/execution
  - after_run_complete: After success (even with tool calls)
  - on_run_error: On unhandled exception
  - on_run_cancelled: Cancellation / timeout

Message / Step Lifecycle:
  - before_step: Each reasoning/tool step
  - after_step: After reasoning result
  - on_step_error: Step-level error

Tool Lifecycle:
  - before_tool_call
  - after_tool_call
  - on_tool_error

Handoff Lifecycle:
  - before_handoff
  - after_handoff
  - on_handoff_error

Session / Memory Lifecycle:
  - on_session_start
  - on_session_load
  - on_session_persist
  - on_session_end

Observability / Monitoring Lifecycle:
  - on_metric_emit
  - on_trace_event
  - on_audit_record

Security / Guardrail Lifecycle:
  - before_input_validation
  - after_input_validation
  - on_policy_violation

HOOK EXECUTION ORDER MODEL:
  1. Global hooks (system-wide)
  2. Agent-level hooks
  3. Run-specific hooks
  4. Tool-specific hooks (nested inside run)

HOOK DESIGN PRINCIPLES:
  - Non-blocking or bounded execution
  - Fail-safe (never crash primary logic)
  - Observable & measurable
  - Composable & configurable
  - Minimal side effects unless intentional

This template builds a universal, extensible hook dispatcher pattern.
"""

# =============================================================================
# 2. DATA MODELS FOR HOOK CONTEXT
# =============================================================================

class HookPhase(Enum):
    PRE = "pre"
    POST = "post"
    ERROR = "error"
    AROUND = "around"  # For wrappers

class HookDomain(Enum):
    AGENT = "agent"
    RUN = "run"
    STEP = "step"
    TOOL = "tool"
    SESSION = "session"
    HANDOFF = "handoff"
    SECURITY = "security"
    OBSERVABILITY = "observability"

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HookContext:
    """Context payload passed to hooks."""
    domain: HookDomain
    phase: HookPhase
    timestamp: datetime = field(default_factory=datetime.utcnow)
    run_id: Optional[str] = None
    agent_name: Optional[str] = None
    step_index: Optional[int] = None
    tool_name: Optional[str] = None
    session_id: Optional[str] = None
    handoff_target: Optional[str] = None
    severity: Severity = Severity.INFO
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def with_data(self, **extra) -> "HookContext":
        self.data.update(extra)
        return self

# =============================================================================
# 3. HOOK REGISTRATION & DISPATCH SYSTEM
# =============================================================================

HookHandler = Callable[[HookContext], Union[None, Coroutine[Any, Any, None]]]
Predicate = Callable[[HookContext], bool]

@dataclass
class RegisteredHook:
    handler: HookHandler
    domains: Sequence[HookDomain]
    phases: Sequence[HookPhase]
    name: str
    order: int = 100
    predicate: Optional[Predicate] = None
    async_supported: bool = True
    timeout_sec: float = 2.0
    swallow_exceptions: bool = True
    tags: List[str] = field(default_factory=list)

class HookRegistry:
    """Central registry for lifecycle hooks."""

    def __init__(self):
        self._hooks: List[RegisteredHook] = []
        self._metrics = {
            "total_invocations": 0,
            "errors": 0,
            "avg_duration_ms": 0.0,
            "per_hook_stats": {}
        }

    def register(self, hook: RegisteredHook):
        self._hooks.append(hook)
        # Keep stable order (lower order runs first)
        self._hooks.sort(key=lambda h: h.order)

    def hook(self,
             *domains: HookDomain,
             phases: Sequence[HookPhase] = (HookPhase.PRE, HookPhase.POST, HookPhase.ERROR),
             name: Optional[str] = None,
             order: int = 100,
             predicate: Optional[Predicate] = None,
             swallow: bool = True,
             timeout: float = 2.0,
             tags: Optional[List[str]] = None):
        """Decorator for hook registration."""
        def decorator(func: HookHandler):
            reg = RegisteredHook(
                handler=func,
                domains=domains,
                phases=phases,
                name=name or func.__name__,
                order=order,
                predicate=predicate,
                swallow_exceptions=swallow,
                timeout_sec=timeout,
                tags=tags or []
            )
            self.register(reg)
            return func
        return decorator

    async def dispatch(self, ctx: HookContext):
        """Dispatch hooks matching the context (async)."""
        relevant = [h for h in self._hooks
                    if ctx.domain in h.domains and ctx.phase in h.phases]

        for reg in relevant:
            if reg.predicate and not reg.predicate(ctx):
                continue
            start = time.perf_counter()
            try:
                result = reg.handler(ctx)
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=reg.timeout_sec)
            except Exception as e:  # noqa: BLE001
                self._metrics["errors"] += 1
                if not reg.swallow_exceptions:
                    raise
                ctx.error = str(e)
            finally:
                dur = (time.perf_counter() - start) * 1000
                self._record_metrics(reg.name, dur)

    def dispatch_sync(self, ctx: HookContext):
        """Synchronous dispatch (wraps async if needed)."""
        # Use asyncio if any coroutine
        async def _run():
            await self.dispatch(ctx)
        try:
            asyncio.run(_run())
        except RuntimeError:  # Already in loop
            # Fallback: schedule in current loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.dispatch(ctx))

    def _record_metrics(self, name: str, duration_ms: float):
        self._metrics["total_invocations"] += 1
        stats = self._metrics["per_hook_stats"].setdefault(name, {"count": 0, "total_ms": 0.0})
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        total = self._metrics["total_invocations"]
        cumulative = sum(h["total_ms"] for h in self._metrics["per_hook_stats"].values())
        self._metrics["avg_duration_ms"] = cumulative / total if total else 0.0

    def metrics(self) -> Dict[str, Any]:
        enriched = {**self._metrics}
        enriched["per_hook_stats"] = {
            k: {**v, "avg_ms": v["total_ms"] / v["count"] if v["count"] else 0.0}
            for k, v in self._metrics["per_hook_stats"].items()
        }
        return enriched

# Global registry instance
HOOKS = HookRegistry()

# =============================================================================
# 4. DECLARATIVE HOOK IMPLEMENTATIONS
# =============================================================================

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.PRE], order=10, tags=["logging", "trace"])
def log_run_start(ctx: HookContext):
    print(f"[hook] Run starting: run_id={ctx.run_id} agent={ctx.agent_name} at {ctx.timestamp.isoformat()}")

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.POST], order=200, tags=["logging"])
def log_run_complete(ctx: HookContext):
    print(f"[hook] Run complete: run_id={ctx.run_id} duration={ctx.duration_ms}ms")

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.ERROR], order=5, tags=["error", "alert"], swallow=True)
def capture_run_error(ctx: HookContext):
    print(f"[hook] Run error captured: run_id={ctx.run_id} error={ctx.error}")

@HOOKS.hook(HookDomain.TOOL, phases=[HookPhase.PRE], order=50, tags=["audit"])
def audit_tool_call(ctx: HookContext):
    print(f"[hook] Tool call audit: tool={ctx.tool_name} run_id={ctx.run_id} input={ctx.data.get('input_preview')}")

@HOOKS.hook(HookDomain.TOOL, phases=[HookPhase.ERROR], order=5, tags=["tool", "error"], swallow=True)
def tool_error_handler(ctx: HookContext):
    print(f"[hook] Tool error: tool={ctx.tool_name} error={ctx.error}")

@HOOKS.hook(HookDomain.OBSERVABILITY, phases=[HookPhase.POST], order=500, tags=["metrics"])
def emit_metric(ctx: HookContext):
    metric_name = ctx.data.get("metric_name", "generic.event")
    value = ctx.data.get("metric_value", 1)
    print(f"[metric] {metric_name}={value} domain={ctx.domain.value}")

# =============================================================================
# 5. PROGRAMMATIC REGISTRATION (DYNAMIC)
# =============================================================================

def register_dynamic_example():
    def dynamic_security_hook(ctx: HookContext):
        if ctx.domain == HookDomain.RUN and ctx.phase == HookPhase.PRE:
            user = ctx.data.get("user_role", "guest")
            if user == "banned":
                raise PermissionError("User banned")
            print(f"[hook] Security check passed for role={user}")

    HOOKS.register(RegisteredHook(
        handler=dynamic_security_hook,
        domains=[HookDomain.RUN],
        phases=[HookPhase.PRE],
        name="dynamic_security",
        order=20,
        predicate=None,
        swallow_exceptions=False,
        timeout_sec=1.0,
        tags=["security"]
    ))

# =============================================================================
# 6. AGENT WRAPPER WITH HOOK DISPATCH
# =============================================================================

class HookEnabledAgent:
    """Wrapper adding lifecycle hook dispatch around an Agent."""
    def __init__(self, agent: Agent):
        self.agent = agent

    def _now_ms(self):
        return time.perf_counter() * 1000

    def run(self, message: str, **kwargs):
        run_id = str(uuid.uuid4())
        start = self._now_ms()
        pre_ctx = HookContext(
            domain=HookDomain.RUN,
            phase=HookPhase.PRE,
            run_id=run_id,
            agent_name=self.agent.name,
            data={"input": message, "input_preview": message[:80], **kwargs}
        )
        HOOKS.dispatch_sync(pre_ctx)
        try:
            result = Runner.run_sync(self.agent, message, **kwargs)
            duration = self._now_ms() - start
            post_ctx = HookContext(
                domain=HookDomain.RUN,
                phase=HookPhase.POST,
                run_id=run_id,
                agent_name=self.agent.name,
                duration_ms=round(duration, 2),
                data={"output_preview": result.final_output[:120] if hasattr(result, 'final_output') else None}
            )
            HOOKS.dispatch_sync(post_ctx)
            return result
        except Exception as e:  # noqa: BLE001
            duration = self._now_ms() - start
            err_ctx = HookContext(
                domain=HookDomain.RUN,
                phase=HookPhase.ERROR,
                run_id=run_id,
                agent_name=self.agent.name,
                duration_ms=round(duration, 2),
                error=str(e),
                data={"traceback": traceback.format_exc()[:2000]}
            )
            HOOKS.dispatch_sync(err_ctx)
            raise

# =============================================================================
# 7. TOOL HOOK INTEGRATION EXAMPLE
# =============================================================================

@function_tool
def slow_computation(x: int, y: int) -> int:
    time.sleep(0.2)
    return x * y

@function_tool
def flaky_tool(should_fail: bool = False) -> str:
    if should_fail:
        raise RuntimeError("Intentional failure for testing hooks")
    return "Success"

class ToolHookRunner:
    """Manual tool invocation with hook dispatch for demonstration."""
    def __init__(self, run_id: str):
        self.run_id = run_id

    def invoke_tool(self, tool: Callable, *args, **kwargs):
        tool_name = getattr(tool, "__name__", "unknown_tool")
        start = time.perf_counter()
        pre_ctx = HookContext(
            domain=HookDomain.TOOL,
            phase=HookPhase.PRE,
            run_id=self.run_id,
            tool_name=tool_name,
            data={"args": args, "kwargs": kwargs, "input_preview": str(args)[:60]}
        )
        HOOKS.dispatch_sync(pre_ctx)
        try:
            result = tool(*args, **kwargs)
            dur = (time.perf_counter() - start) * 1000
            post_ctx = HookContext(
                domain=HookDomain.TOOL,
                phase=HookPhase.POST,
                run_id=self.run_id,
                tool_name=tool_name,
                duration_ms=round(dur, 2),
                data={"result_preview": str(result)[:80]}
            )
            HOOKS.dispatch_sync(post_ctx)
            return result
        except Exception as e:  # noqa: BLE001
            dur = (time.perf_counter() - start) * 1000
            err_ctx = HookContext(
                domain=HookDomain.TOOL,
                phase=HookPhase.ERROR,
                run_id=self.run_id,
                tool_name=tool_name,
                duration_ms=round(dur, 2),
                error=str(e),
                data={"exception_type": type(e).__name__}
            )
            HOOKS.dispatch_sync(err_ctx)
            raise

# =============================================================================
# 8. ASYNC HOOK SUPPORT & EXAMPLES
# =============================================================================

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.PRE], order=30, tags=["async", "experiment"])
async def async_experiment_flag(ctx: HookContext):
    await asyncio.sleep(0.01)
    ctx.data["exp_variant"] = "A"
    print("[hook] Async experiment assignment -> variant A")

# =============================================================================
# 9. METRICS & TRACING HOOKS
# =============================================================================

metrics_store: Dict[str, Any] = {
    "runs": 0,
    "run_errors": 0,
    "tools": 0,
    "tool_errors": 0,
    "durations": [],
}

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.POST], order=900, tags=["metrics"])
def aggregate_run_metrics(ctx: HookContext):
    metrics_store["runs"] += 1
    if ctx.duration_ms is not None:
        metrics_store["durations"].append(ctx.duration_ms)

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.ERROR], order=900, tags=["metrics", "error"])
def aggregate_run_errors(ctx: HookContext):
    metrics_store["run_errors"] += 1

@HOOKS.hook(HookDomain.TOOL, phases=[HookPhase.POST], order=900, tags=["metrics"])
def aggregate_tool_metrics(ctx: HookContext):
    metrics_store["tools"] += 1

@HOOKS.hook(HookDomain.TOOL, phases=[HookPhase.ERROR], order=900, tags=["metrics", "error"])
def aggregate_tool_errors(ctx: HookContext):
    metrics_store["tool_errors"] += 1

# =============================================================================
# 10. AUDIT & SECURITY HOOKS
# =============================================================================

audit_log: List[Dict[str, Any]] = []

@HOOKS.hook(HookDomain.TOOL, phases=[HookPhase.PRE], order=15, tags=["audit", "security"])
def record_tool_audit(ctx: HookContext):
    entry = {
        "ts": ctx.timestamp.isoformat(),
        "run_id": ctx.run_id,
        "tool": ctx.tool_name,
        "input_hash": hash(str(ctx.data.get("args")) + str(ctx.data.get("kwargs"))),
    }
    audit_log.append(entry)

# =============================================================================
# 11. ADVANCED: COMPOSITE / CHAINED HOOKS
# =============================================================================

def composite_hook(*handlers: HookHandler) -> HookHandler:
    async def _composite(ctx: HookContext):
        for h in handlers:
            try:
                r = h(ctx)
                if inspect.isawaitable(r):
                    await r
            except Exception:  # noqa: BLE001
                # Continue chain, swallow errors
                pass
    return _composite

@HOOKS.hook(HookDomain.RUN, phases=[HookPhase.PRE], order=40, tags=["composite"])
async def chained_pre_run(ctx: HookContext):
    async def sub1(c: HookContext):
        c.data.setdefault("chain", []).append("sub1")
    async def sub2(c: HookContext):
        c.data.setdefault("chain", []).append("sub2")
    handler = composite_hook(sub1, sub2)
    await handler(ctx)
    print(f"[hook] Composite chain executed: {ctx.data.get('chain')}")

# =============================================================================
# 12. PLUGIN ARCHITECTURE EXAMPLE
# =============================================================================

class HookPlugin(Protocol):
    name: str
    def register(self, registry: HookRegistry) -> None: ...

class LatencyBudgetPlugin:
    name = "latency_budget"
    def __init__(self, budget_ms: float = 1500):
        self.budget_ms = budget_ms
    def register(self, registry: HookRegistry) -> None:
        @registry.hook(HookDomain.RUN, phases=[HookPhase.POST], order=120, tags=["latency"])
        def enforce_budget(ctx: HookContext):
            if ctx.duration_ms and ctx.duration_ms > self.budget_ms:
                print(f"[hook][latency] WARNING run exceeded budget: {ctx.duration_ms}ms > {self.budget_ms}ms")

class SensitiveDataMaskPlugin:
    name = "masking"
    def register(self, registry: HookRegistry) -> None:
        @registry.hook(HookDomain.TOOL, phases=[HookPhase.PRE], order=60, tags=["mask", "security"])
        def mask_inputs(ctx: HookContext):
            args = ctx.data.get("args", ())
            new_args = []
            for a in args:
                if isinstance(a, str) and len(a) > 20:
                    new_args.append(a[:10] + "***masked***")
                else:
                    new_args.append(a)
            ctx.data["args"] = tuple(new_args)
            print("[hook][mask] Inputs masked where necessary")

# =============================================================================
# 13. TESTING HOOK BEHAVIOR
# =============================================================================

def test_hook_behaviors():
    print("\n=== HOOK BEHAVIOR TESTS ===")
    # Register dynamic and plugin hooks
    register_dynamic_example()
    LatencyBudgetPlugin(100).register(HOOKS)  # small budget to trigger
    SensitiveDataMaskPlugin().register(HOOKS)

    # Create agent
    @function_tool
    def greet(name: str) -> str:
        return f"Hello {name}!"

    base_agent = Agent(
        name="LifecycleAgent",
        instructions="You manage tasks with hook instrumentation.",
        tools=[greet, slow_computation, flaky_tool]
    )
    hooked_agent = HookEnabledAgent(base_agent)

    # Normal run
    print("\n-- Normal Run --")
    result = hooked_agent.run("Please greet Alice and multiply numbers", user_role="user")
    print("Result (preview):", getattr(result, 'final_output', str(result))[:120])

    # Tool invocations manual
    print("\n-- Tool Invocation (Manual) --")
    run_id = str(uuid.uuid4())
    tool_runner = ToolHookRunner(run_id)
    tool_runner.invoke_tool(slow_computation, 6, 7)

    # Failing tool
    print("\n-- Failing Tool Invocation --")
    try:
        tool_runner.invoke_tool(flaky_tool, should_fail=True)
    except Exception as e:  # noqa: BLE001
        print("Expected tool failure handled:", e)

    # Security failure run
    print("\n-- Security Failure Run (banned user) --")
    try:
        hooked_agent.run("Test banned user flow", user_role="banned")
    except Exception as e:  # noqa: BLE001
        print("Security enforcement triggered:", e)

    # Metrics snapshot
    print("\n-- Metrics Snapshot --")
    print(json.dumps(metrics_store, indent=2))
    print("Hook Registry Metrics:")
    print(json.dumps(HOOKS.metrics(), indent=2, default=str))

    # Audit log excerpt
    print("\n-- Audit Log Entries (first 3) --")
    for entry in audit_log[:3]:
        print(entry)

# =============================================================================
# 14. PERFORMANCE & OPTIMIZATION STRATEGIES
# =============================================================================
"""
PERFORMANCE GUIDELINES FOR HOOKS:
- Keep pre-phase hooks lightweight (<5ms typical)
- Batch expensive post-phase analytics asynchronously
- Use sampling for high-volume events (e.g., 10% of tool calls)
- Employ timeouts to prevent hook stalls
- Cache heavy computations at registration time
- Avoid network calls in synchronous critical path (use queues)
- Aggregate metrics locally before emitting
- Lazy serialize large objects (only when needed)

BACKPRESSURE HANDLING:
- Use async queues (asyncio.Queue) for deferred processing
- Add circuit breakers for failing downstream services
- Implement fallback noop behavior when congestion detected

OBSERVABILITY PATTERNS:
- Track per-hook latency percentiles
- Expose error rate and slow hook leaderboard
- Correlate run_id across hooks for trace reconstruction
- Emit structured JSON logs for machine ingestion
"""

# =============================================================================
# 15. BEST PRACTICES & ANTI-PATTERNS
# =============================================================================
"""
DO:
- Keep hook interfaces stable & documented
- Use explicit domains & phases
- Make hooks idempotent when possible
- Fail closed for security hooks (block on failure)
- Fail open for observability hooks (swallow errors)
- Add tags for discoverability & filtering
- Write focused unit tests for critical hooks

DON'T:
- Mutate core result objects unpredictably
- Execute long blocking operations inline
- Swallow security / integrity violations silently
- Overuse error hooks as primary logic
- Create hidden dependencies between hooks

VERSIONING STRATEGY:
- Introduce new domains/phases backward compatibly
- Deprecate with warnings before removal
- Provide migration guide for signature changes
"""

# =============================================================================
# 16. MAIN EXECUTION
# =============================================================================

def run_lifecycle_hooks_demo():
    print("🪝 OPENAI AGENTS SDK - LIFECYCLE HOOKS DEMO")
    print("This demonstrates comprehensive hook usage patterns.\n")
    test_hook_behaviors()
    print("\n✅ Lifecycle hooks demonstration complete.")

if __name__ == "__main__":
    run_lifecycle_hooks_demo()
    print("\n💡 Use this template as a reference for designing robust lifecycle hook systems.")
