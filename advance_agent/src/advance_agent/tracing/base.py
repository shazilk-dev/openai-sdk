"""Tracing system for the Advanced Agent System."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager
from enum import Enum

from pydantic import BaseModel

from ..core.exceptions import TracingError
from ..core.logging import get_logger
from ..core.security import SecurityContext


class TraceLevel(str, Enum):
    """Trace level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SpanType(str, Enum):
    """Types of trace spans."""
    REQUEST = "request"
    AGENT_OPERATION = "agent_operation"
    TOOL_CALL = "tool_call"
    GUARDRAIL_CHECK = "guardrail_check"
    HOOK_EXECUTION = "hook_execution"
    DATABASE_QUERY = "database_query"
    EXTERNAL_API = "external_api"
    WORKFLOW_STEP = "workflow_step"


class SpanStatus(str, Enum):
    """Status of a trace span."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TraceEvent(BaseModel):
    """A single event within a trace span."""
    timestamp: datetime
    level: TraceLevel
    message: str
    attributes: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True


class TraceSpan(BaseModel):
    """A trace span representing a unit of work."""
    span_id: str
    parent_span_id: Optional[str] = None
    trace_id: str
    operation_name: str
    span_type: SpanType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.STARTED
    attributes: Dict[str, Any] = {}
    events: List[TraceEvent] = []
    tags: Dict[str, str] = {}
    error: Optional[str] = None
    
    def add_event(
        self,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Add an event to this span."""
        event = TraceEvent(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on this span."""
        self.attributes[key] = value
    
    def set_tag(self, key: str, value: str):
        """Set a tag on this span."""
        self.tags[key] = value
    
    def finish(self, status: SpanStatus = SpanStatus.COMPLETED, error: Optional[str] = None):
        """Mark the span as finished."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if error:
            self.error = error
    
    class Config:
        use_enum_values = True


class Trace(BaseModel):
    """A complete trace containing multiple spans."""
    trace_id: str
    root_span_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    spans: Dict[str, TraceSpan] = {}
    metadata: Dict[str, Any] = {}
    user_id: Optional[str] = None
    
    def add_span(self, span: TraceSpan):
        """Add a span to this trace."""
        self.spans[span.span_id] = span
    
    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Get a span by ID."""
        return self.spans.get(span_id)
    
    def get_root_span(self) -> Optional[TraceSpan]:
        """Get the root span of this trace."""
        return self.spans.get(self.root_span_id)
    
    def get_child_spans(self, parent_span_id: str) -> List[TraceSpan]:
        """Get all child spans of a parent span."""
        return [
            span for span in self.spans.values()
            if span.parent_span_id == parent_span_id
        ]
    
    def finish(self):
        """Mark the trace as finished."""
        self.end_time = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for storage/export."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "spans": [span.dict() for span in self.spans.values()]
        }


class TracingContext:
    """Context for managing trace state across async operations."""
    
    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.current_span: Optional[TraceSpan] = None
        self.span_stack: List[str] = []
    
    def set_trace(self, trace: Trace):
        """Set the current trace."""
        self.current_trace = trace
    
    def push_span(self, span: TraceSpan):
        """Push a new span onto the stack."""
        if self.current_span:
            self.span_stack.append(self.current_span.span_id)
        self.current_span = span
        
        if self.current_trace:
            self.current_trace.add_span(span)
    
    def pop_span(self) -> Optional[TraceSpan]:
        """Pop the current span from the stack."""
        current = self.current_span
        
        if self.span_stack:
            parent_span_id = self.span_stack.pop()
            self.current_span = self.current_trace.get_span(parent_span_id) if self.current_trace else None
        else:
            self.current_span = None
        
        return current
    
    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID."""
        return self.current_span.span_id if self.current_span else None


# Thread-local storage for tracing context
import contextvars
tracing_context: contextvars.ContextVar[TracingContext] = contextvars.ContextVar(
    'tracing_context',
    default=TracingContext()
)


class TracingProcessor:
    """Processes and stores trace data."""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage_backend = storage_backend
        self.logger = get_logger("tracing_processor")
        self.traces: Dict[str, Trace] = {}  # In-memory storage
        self.max_traces = 10000
        
        # Metrics
        self.traces_created = 0
        self.spans_created = 0
        self.events_recorded = 0
    
    async def start_trace(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.REQUEST,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        root_span_id = str(uuid.uuid4())
        
        # Create root span
        root_span = TraceSpan(
            span_id=root_span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=datetime.now(timezone.utc)
        )
        
        # Create trace
        trace = Trace(
            trace_id=trace_id,
            root_span_id=root_span_id,
            start_time=datetime.now(timezone.utc),
            metadata=metadata or {},
            user_id=user_id
        )
        
        trace.add_span(root_span)
        
        # Store trace
        self.traces[trace_id] = trace
        
        # Maintain trace limit
        if len(self.traces) > self.max_traces:
            # Remove oldest traces
            oldest_trace_id = min(self.traces.keys(), key=lambda tid: self.traces[tid].start_time)
            del self.traces[oldest_trace_id]
        
        # Update metrics
        self.traces_created += 1
        self.spans_created += 1
        
        # Set in context
        context = tracing_context.get()
        context.set_trace(trace)
        context.push_span(root_span)
        
        self.logger.debug(f"Started trace: {trace_id} for operation: {operation_name}")
        
        return trace
    
    async def start_span(
        self,
        operation_name: str,
        span_type: SpanType,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TraceSpan:
        """Start a new span."""
        context = tracing_context.get()
        
        if not context.current_trace:
            raise TracingError("No active trace found")
        
        span_id = str(uuid.uuid4())
        
        # Use current span as parent if not specified
        if parent_span_id is None and context.current_span:
            parent_span_id = context.current_span.span_id
        
        span = TraceSpan(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=context.current_trace.trace_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=datetime.now(timezone.utc),
            attributes=attributes or {},
            tags=tags or {}
        )
        
        # Add to context
        context.push_span(span)
        
        # Update metrics
        self.spans_created += 1
        
        self.logger.debug(f"Started span: {span_id} for operation: {operation_name}")
        
        return span
    
    async def finish_span(
        self,
        span_id: Optional[str] = None,
        status: SpanStatus = SpanStatus.COMPLETED,
        error: Optional[str] = None
    ):
        """Finish a span."""
        context = tracing_context.get()
        
        if span_id:
            # Find and finish specific span
            if context.current_trace:
                span = context.current_trace.get_span(span_id)
                if span:
                    span.finish(status, error)
        else:
            # Finish current span
            if context.current_span:
                context.current_span.finish(status, error)
                finished_span = context.pop_span()
                
                if finished_span:
                    self.logger.debug(f"Finished span: {finished_span.span_id}")
    
    async def add_event(
        self,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        attributes: Optional[Dict[str, Any]] = None,
        span_id: Optional[str] = None
    ):
        """Add an event to a span."""
        context = tracing_context.get()
        
        target_span = None
        if span_id and context.current_trace:
            target_span = context.current_trace.get_span(span_id)
        elif context.current_span:
            target_span = context.current_span
        
        if target_span:
            target_span.add_event(message, level, attributes)
            self.events_recorded += 1
    
    async def set_span_attribute(
        self,
        key: str,
        value: Any,
        span_id: Optional[str] = None
    ):
        """Set an attribute on a span."""
        context = tracing_context.get()
        
        target_span = None
        if span_id and context.current_trace:
            target_span = context.current_trace.get_span(span_id)
        elif context.current_span:
            target_span = context.current_span
        
        if target_span:
            target_span.set_attribute(key, value)
    
    async def finish_trace(self, trace_id: Optional[str] = None):
        """Finish a trace."""
        context = tracing_context.get()
        
        if trace_id:
            trace = self.traces.get(trace_id)
        else:
            trace = context.current_trace
        
        if trace:
            trace.finish()
            
            # Finish any remaining open spans
            for span in trace.spans.values():
                if span.status == SpanStatus.STARTED:
                    span.finish(SpanStatus.COMPLETED)
            
            # Store to backend if available
            if self.storage_backend:
                await self._store_trace(trace)
            
            self.logger.debug(f"Finished trace: {trace.trace_id}")
    
    async def _store_trace(self, trace: Trace):
        """Store trace to backend storage."""
        try:
            if hasattr(self.storage_backend, 'store_trace'):
                await self.storage_backend.store_trace(trace.to_dict())
        except Exception as e:
            self.logger.error(f"Failed to store trace {trace.trace_id}: {str(e)}")
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 50) -> List[Trace]:
        """Get recent traces."""
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True
        )
        return sorted_traces[:limit]
    
    def get_traces_by_user(self, user_id: str, limit: int = 50) -> List[Trace]:
        """Get traces for a specific user."""
        user_traces = [
            trace for trace in self.traces.values()
            if trace.user_id == user_id
        ]
        return sorted(user_traces, key=lambda t: t.start_time, reverse=True)[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tracing metrics."""
        active_traces = len([t for t in self.traces.values() if t.end_time is None])
        completed_traces = len([t for t in self.traces.values() if t.end_time is not None])
        
        # Calculate average span count per trace
        total_spans = sum(len(trace.spans) for trace in self.traces.values())
        avg_spans_per_trace = total_spans / len(self.traces) if self.traces else 0
        
        # Calculate average trace duration
        completed_trace_durations = [
            (trace.end_time - trace.start_time).total_seconds()
            for trace in self.traces.values()
            if trace.end_time
        ]
        avg_trace_duration = (
            sum(completed_trace_durations) / len(completed_trace_durations)
            if completed_trace_durations else 0
        )
        
        return {
            "traces_created": self.traces_created,
            "spans_created": self.spans_created,
            "events_recorded": self.events_recorded,
            "active_traces": active_traces,
            "completed_traces": completed_traces,
            "total_traces_in_memory": len(self.traces),
            "average_spans_per_trace": avg_spans_per_trace,
            "average_trace_duration_seconds": avg_trace_duration
        }


@asynccontextmanager
async def trace_operation(
    operation_name: str,
    span_type: SpanType,
    processor: TracingProcessor,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
):
    """Context manager for tracing an operation."""
    span = await processor.start_span(
        operation_name=operation_name,
        span_type=span_type,
        attributes=attributes,
        tags=tags
    )
    
    try:
        yield span
        await processor.finish_span(status=SpanStatus.COMPLETED)
    except Exception as e:
        await processor.finish_span(status=SpanStatus.FAILED, error=str(e))
        raise


class TracingIntegration:
    """Integration for adding tracing to existing systems."""
    
    def __init__(self, processor: TracingProcessor):
        self.processor = processor
        self.logger = get_logger("tracing_integration")
    
    def trace_function(
        self,
        operation_name: Optional[str] = None,
        span_type: SpanType = SpanType.AGENT_OPERATION,
        attributes: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Decorator for tracing function calls."""
        def decorator(func: Callable):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with trace_operation(
                        operation_name=operation_name,
                        span_type=span_type,
                        processor=self.processor,
                        attributes=attributes,
                        tags=tags
                    ) as span:
                        # Add function arguments as attributes
                        span.set_attribute("function.args", str(args))
                        span.set_attribute("function.kwargs", str(kwargs))
                        
                        result = await func(*args, **kwargs)
                        
                        # Add result info
                        span.set_attribute("function.result_type", type(result).__name__)
                        
                        return result
                
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    # For sync functions, we can't use async context manager
                    # So we manually handle the span
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    span = loop.run_until_complete(
                        self.processor.start_span(
                            operation_name=operation_name,
                            span_type=span_type,
                            attributes=attributes,
                            tags=tags
                        )
                    )
                    
                    try:
                        span.set_attribute("function.args", str(args))
                        span.set_attribute("function.kwargs", str(kwargs))
                        
                        result = func(*args, **kwargs)
                        
                        span.set_attribute("function.result_type", type(result).__name__)
                        
                        loop.run_until_complete(
                            self.processor.finish_span(status=SpanStatus.COMPLETED)
                        )
                        
                        return result
                    
                    except Exception as e:
                        loop.run_until_complete(
                            self.processor.finish_span(status=SpanStatus.FAILED, error=str(e))
                        )
                        raise
                
                return sync_wrapper
        
        return decorator


# Global tracing processor instance
global_tracing_processor = TracingProcessor()


def get_tracing_processor() -> TracingProcessor:
    """Get the global tracing processor."""
    return global_tracing_processor


def create_tracing_integration() -> TracingIntegration:
    """Create a tracing integration with the global processor."""
    return TracingIntegration(global_tracing_processor)
