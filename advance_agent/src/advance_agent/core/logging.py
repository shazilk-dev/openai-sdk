"""Structured logging configuration for the Advanced Agent System."""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import structlog
from structlog.processors import TimeStamper, add_log_level, JSONRenderer
from structlog.stdlib import filter_by_level, add_logger_name

from .config import get_settings


class CorrelationIdProcessor:
    """Add correlation ID to log entries for request tracing."""
    
    def __init__(self):
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self.correlation_id = correlation_id
    
    def get_correlation_id(self) -> str:
        """Get or generate correlation ID."""
        if not self.correlation_id:
            self.correlation_id = str(uuid4())
        return self.correlation_id
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to log entry."""
        event_dict["correlation_id"] = self.get_correlation_id()
        return event_dict


class AgentContextProcessor:
    """Add agent context information to log entries."""
    
    def __init__(self):
        self.agent_context = {}
    
    def set_agent_context(
        self,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None
    ) -> None:
        """Set agent context for logging."""
        self.agent_context.update({
            k: v for k, v in {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "workflow_id": workflow_id,
                "trace_id": trace_id,
                "span_id": span_id
            }.items() if v is not None
        })
    
    def clear_agent_context(self) -> None:
        """Clear agent context."""
        self.agent_context.clear()
    
    def __call__(self, logger, method_name, event_dict):
        """Add agent context to log entry."""
        if self.agent_context:
            event_dict["agent"] = self.agent_context.copy()
        return event_dict


class SecurityEventProcessor:
    """Process and flag security-related events."""
    
    SECURITY_KEYWORDS = [
        "security", "violation", "unauthorized", "forbidden",
        "breach", "attack", "malicious", "suspicious",
        "authentication", "authorization", "permission"
    ]
    
    def __call__(self, logger, method_name, event_dict):
        """Flag security-related events."""
        event_text = str(event_dict.get("event", "")).lower()
        message_text = str(event_dict.get("message", "")).lower()
        
        is_security_event = any(
            keyword in event_text or keyword in message_text
            for keyword in self.SECURITY_KEYWORDS
        )
        
        if is_security_event or method_name in ["warning", "error", "critical"]:
            event_dict["security_relevant"] = is_security_event
            event_dict["requires_attention"] = True
        
        return event_dict


class PerformanceProcessor:
    """Add performance metrics to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance context to relevant log entries."""
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            event_dict["performance"] = {
                "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent()
            }
        except ImportError:
            pass  # psutil not available
        
        return event_dict


class ErrorContextProcessor:
    """Add detailed error context for exceptions."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add error context for exception log entries."""
        if "exception" in event_dict or method_name in ["error", "critical"]:
            # Add traceback if exception is present
            exc_info = event_dict.get("exc_info")
            if exc_info:
                event_dict["traceback"] = traceback.format_exception(*exc_info)
            
            # Add error classification
            error = event_dict.get("exception")
            if error:
                event_dict["error_classification"] = {
                    "type": type(error).__name__,
                    "module": getattr(error, "__module__", "unknown"),
                    "recoverable": getattr(error, "recoverable", None),
                    "error_code": getattr(error, "error_code", None)
                }
        
        return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Create processors
    correlation_processor = CorrelationIdProcessor()
    agent_processor = AgentContextProcessor()
    security_processor = SecurityEventProcessor()
    performance_processor = PerformanceProcessor()
    error_processor = ErrorContextProcessor()
    
    # Configure structlog processors
    processors = [
        filter_by_level,
        add_log_level,
        add_logger_name,
        TimeStamper(fmt="iso"),
        correlation_processor,
        agent_processor,
        security_processor,
        error_processor,
    ]
    
    # Add performance processor in development
    if settings.environment == "development":
        processors.append(performance_processor)
    
    # Configure output format
    if settings.log_format.lower() == "json":
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Store processors for global access
    setup_logging.correlation_processor = correlation_processor
    setup_logging.agent_processor = agent_processor


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current request context."""
    if hasattr(setup_logging, "correlation_processor"):
        setup_logging.correlation_processor.set_correlation_id(correlation_id)


def set_agent_context(
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None
) -> None:
    """Set agent context for logging."""
    if hasattr(setup_logging, "agent_processor"):
        setup_logging.agent_processor.set_agent_context(
            agent_id=agent_id,
            agent_name=agent_name,
            workflow_id=workflow_id,
            trace_id=trace_id,
            span_id=span_id
        )


def clear_agent_context() -> None:
    """Clear agent context."""
    if hasattr(setup_logging, "agent_processor"):
        setup_logging.agent_processor.clear_agent_context()


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
    
    def log_operation_start(
        self, 
        operation: str, 
        **context: Any
    ) -> None:
        """Log the start of an operation."""
        self.logger.info(
            "Operation started",
            operation=operation,
            **context
        )
    
    def log_operation_success(
        self, 
        operation: str, 
        duration_ms: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log successful completion of an operation."""
        log_data = {
            "operation": operation,
            **context
        }
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        
        self.logger.info("Operation completed successfully", **log_data)
    
    def log_operation_error(
        self, 
        operation: str, 
        error: Exception,
        duration_ms: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log operation failure."""
        log_data = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        
        self.logger.error("Operation failed", **log_data, exc_info=True)
    
    def log_security_event(
        self, 
        event_type: str, 
        severity: str = "medium",
        **context: Any
    ) -> None:
        """Log security-related events."""
        self.logger.warning(
            "Security event detected",
            event_type=event_type,
            severity=severity,
            security_relevant=True,
            **context
        )
    
    def log_performance_metric(
        self, 
        metric_name: str, 
        value: Union[int, float],
        unit: str = "",
        **context: Any
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **context
        )


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        **context: Any
    ) -> None:
        """Log user actions for audit trail."""
        self.logger.info(
            "User action",
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            audit_event=True,
            **context
        )
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        details: Dict[str, Any],
        **context: Any
    ) -> None:
        """Log system events for audit trail."""
        self.logger.info(
            "System event",
            event_type=event_type,
            component=component,
            details=details,
            audit_event=True,
            **context
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        severity: str,
        details: Dict[str, Any],
        **context: Any
    ) -> None:
        """Log security violations."""
        self.logger.warning(
            "Security violation",
            violation_type=violation_type,
            severity=severity,
            details=details,
            security_relevant=True,
            audit_event=True,
            **context
        )


# Initialize logging on module import
setup_logging()

# Create module-level logger
logger = get_logger(__name__)

# Create audit logger instance
audit_logger = AuditLogger()
