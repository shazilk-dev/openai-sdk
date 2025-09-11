"""Hooks system for the Advanced Agent System."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

from pydantic import BaseModel

from ..core.exceptions import HookError, HookExecutionError
from ..core.logging import get_logger
from ..core.security import SecurityContext
from ..models.schemas import Message, Task, AgentStatus


class HookType(str, Enum):
    """Types of hooks available in the system."""
    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    BEFORE_AGENT_INIT = "before_agent_init"
    AFTER_AGENT_INIT = "after_agent_init"
    BEFORE_MESSAGE_PROCESS = "before_message_process"
    AFTER_MESSAGE_PROCESS = "after_message_process"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_ERROR = "on_error"
    ON_GUARDRAIL_VIOLATION = "on_guardrail_violation"
    ON_AGENT_STATUS_CHANGE = "on_agent_status_change"
    ON_TASK_COMPLETE = "on_task_complete"
    ON_WORKFLOW_START = "on_workflow_start"
    ON_WORKFLOW_COMPLETE = "on_workflow_complete"


class HookPriority(int, Enum):
    """Priority levels for hook execution."""
    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 100


class HookResult(BaseModel):
    """Result of hook execution."""
    success: bool
    continue_execution: bool = True
    modified_data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class BaseHook(ABC):
    """Abstract base class for all hooks."""
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
        async_execution: bool = True
    ):
        self.name = name
        self.hook_type = hook_type
        self.priority = priority
        self.enabled = enabled
        self.async_execution = async_execution
        self.logger = get_logger(f"hook.{name}")
        
        # Metrics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_execution = None
    
    @abstractmethod
    async def execute(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> HookResult:
        """Execute the hook with provided data."""
        pass
    
    async def __call__(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> HookResult:
        """Execute the hook with metrics tracking."""
        if not self.enabled:
            return HookResult(
                success=True,
                continue_execution=True,
                message="Hook disabled"
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            self.execution_count += 1
            self.last_execution = start_time
            
            result = await self.execute(data, context, security_context)
            
            # Update metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            self.total_execution_time += execution_time
            
            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.logger.debug(
                f"Hook {self.name} executed",
                extra={
                    "success": result.success,
                    "execution_time": execution_time,
                    "continue_execution": result.continue_execution
                }
            )
            
            return result
        
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.failure_count += 1
            self.total_execution_time += execution_time
            
            self.logger.error(f"Hook {self.name} execution failed: {str(e)}")
            
            return HookResult(
                success=False,
                continue_execution=True,  # Don't break execution chain by default
                error=str(e),
                execution_time=execution_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hook execution metrics."""
        avg_execution_time = (
            self.total_execution_time / self.execution_count
            if self.execution_count > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.execution_count
            if self.execution_count > 0 else 0.0
        )
        
        return {
            "name": self.name,
            "hook_type": self.hook_type.value,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }


class LoggingHook(BaseHook):
    """Hook for comprehensive logging of system events."""
    
    def __init__(
        self,
        hook_type: HookType,
        log_level: str = "INFO",
        include_data: bool = True,
        **kwargs
    ):
        name = f"logging_{hook_type.value}"
        super().__init__(name=name, hook_type=hook_type, **kwargs)
        self.log_level = log_level.upper()
        self.include_data = include_data
    
    async def execute(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> HookResult:
        """Log the event with relevant data."""
        log_data = {
            "hook_type": self.hook_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": security_context.user_id if security_context else None,
            "agent_id": data.get("agent_id"),
            "context": context
        }
        
        if self.include_data:
            # Sanitize sensitive data before logging
            sanitized_data = self._sanitize_data(data)
            log_data["data"] = sanitized_data
        
        if self.log_level == "DEBUG":
            self.logger.debug(f"Event: {self.hook_type.value}", extra=log_data)
        elif self.log_level == "INFO":
            self.logger.info(f"Event: {self.hook_type.value}", extra=log_data)
        elif self.log_level == "WARNING":
            self.logger.warning(f"Event: {self.hook_type.value}", extra=log_data)
        elif self.log_level == "ERROR":
            self.logger.error(f"Event: {self.hook_type.value}", extra=log_data)
        
        return HookResult(
            success=True,
            continue_execution=True,
            message=f"Event logged: {self.hook_type.value}"
        )
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data before logging."""
        sanitized = {}
        
        for key, value in data.items():
            if key.lower() in ["password", "token", "secret", "key", "api_key"]:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:100] + "...[truncated]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value
        
        return sanitized


class MetricsHook(BaseHook):
    """Hook for collecting system metrics."""
    
    def __init__(self, hook_type: HookType, **kwargs):
        name = f"metrics_{hook_type.value}"
        super().__init__(name=name, hook_type=hook_type, **kwargs)
        self.metrics_store = {}
    
    async def execute(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> HookResult:
        """Collect and store metrics."""
        timestamp = datetime.now(timezone.utc)
        
        # Extract relevant metrics based on hook type
        if self.hook_type == HookType.AFTER_MESSAGE_PROCESS:
            self._record_message_metrics(data, timestamp)
        elif self.hook_type == HookType.AFTER_TOOL_CALL:
            self._record_tool_metrics(data, timestamp)
        elif self.hook_type == HookType.ON_ERROR:
            self._record_error_metrics(data, timestamp)
        elif self.hook_type == HookType.ON_GUARDRAIL_VIOLATION:
            self._record_guardrail_metrics(data, timestamp)
        
        return HookResult(
            success=True,
            continue_execution=True,
            message="Metrics recorded"
        )
    
    def _record_message_metrics(self, data: Dict[str, Any], timestamp: datetime):
        """Record message processing metrics."""
        agent_id = data.get("agent_id", "unknown")
        response_time = data.get("response_time", 0.0)
        tokens_used = data.get("tokens_used", 0)
        
        if agent_id not in self.metrics_store:
            self.metrics_store[agent_id] = {
                "message_count": 0,
                "total_response_time": 0.0,
                "total_tokens": 0,
                "last_activity": None
            }
        
        metrics = self.metrics_store[agent_id]
        metrics["message_count"] += 1
        metrics["total_response_time"] += response_time
        metrics["total_tokens"] += tokens_used
        metrics["last_activity"] = timestamp.isoformat()
    
    def _record_tool_metrics(self, data: Dict[str, Any], timestamp: datetime):
        """Record tool usage metrics."""
        tool_name = data.get("tool_name", "unknown")
        execution_time = data.get("execution_time", 0.0)
        success = data.get("success", False)
        
        if "tools" not in self.metrics_store:
            self.metrics_store["tools"] = {}
        
        if tool_name not in self.metrics_store["tools"]:
            self.metrics_store["tools"][tool_name] = {
                "call_count": 0,
                "success_count": 0,
                "total_execution_time": 0.0,
                "last_used": None
            }
        
        tool_metrics = self.metrics_store["tools"][tool_name]
        tool_metrics["call_count"] += 1
        if success:
            tool_metrics["success_count"] += 1
        tool_metrics["total_execution_time"] += execution_time
        tool_metrics["last_used"] = timestamp.isoformat()
    
    def _record_error_metrics(self, data: Dict[str, Any], timestamp: datetime):
        """Record error metrics."""
        error_type = data.get("error_type", "unknown")
        agent_id = data.get("agent_id", "unknown")
        
        if "errors" not in self.metrics_store:
            self.metrics_store["errors"] = {}
        
        if error_type not in self.metrics_store["errors"]:
            self.metrics_store["errors"][error_type] = {
                "count": 0,
                "agents_affected": set(),
                "last_occurrence": None
            }
        
        error_metrics = self.metrics_store["errors"][error_type]
        error_metrics["count"] += 1
        error_metrics["agents_affected"].add(agent_id)
        error_metrics["last_occurrence"] = timestamp.isoformat()
    
    def _record_guardrail_metrics(self, data: Dict[str, Any], timestamp: datetime):
        """Record guardrail violation metrics."""
        guardrail_name = data.get("guardrail_name", "unknown")
        severity = data.get("severity", "unknown")
        action = data.get("action", "unknown")
        
        if "guardrails" not in self.metrics_store:
            self.metrics_store["guardrails"] = {}
        
        if guardrail_name not in self.metrics_store["guardrails"]:
            self.metrics_store["guardrails"][guardrail_name] = {
                "violation_count": 0,
                "severity_counts": {},
                "action_counts": {},
                "last_violation": None
            }
        
        gr_metrics = self.metrics_store["guardrails"][guardrail_name]
        gr_metrics["violation_count"] += 1
        gr_metrics["severity_counts"][severity] = gr_metrics["severity_counts"].get(severity, 0) + 1
        gr_metrics["action_counts"][action] = gr_metrics["action_counts"].get(action, 0) + 1
        gr_metrics["last_violation"] = timestamp.isoformat()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        # Convert sets to lists for JSON serialization
        metrics_copy = {}
        for key, value in self.metrics_store.items():
            if isinstance(value, dict):
                metrics_copy[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        metrics_copy[key][sub_key] = {}
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            if isinstance(sub_sub_value, set):
                                metrics_copy[key][sub_key][sub_sub_key] = list(sub_sub_value)
                            else:
                                metrics_copy[key][sub_key][sub_sub_key] = sub_sub_value
                    else:
                        metrics_copy[key][sub_key] = sub_value
            else:
                metrics_copy[key] = value
        
        return metrics_copy


class SecurityHook(BaseHook):
    """Hook for security monitoring and enforcement."""
    
    def __init__(
        self,
        hook_type: HookType,
        security_level: str = "standard",
        **kwargs
    ):
        name = f"security_{hook_type.value}"
        super().__init__(name=name, hook_type=hook_type, **kwargs)
        self.security_level = security_level
        self.security_events = []
        self.max_events = 1000
    
    async def execute(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> HookResult:
        """Perform security monitoring and checks."""
        timestamp = datetime.now(timezone.utc)
        
        # Record security event
        security_event = {
            "timestamp": timestamp.isoformat(),
            "hook_type": self.hook_type.value,
            "user_id": security_context.user_id if security_context else None,
            "agent_id": data.get("agent_id"),
            "data_keys": list(data.keys()),
            "security_level": self.security_level
        }
        
        # Check for security issues based on hook type
        security_issues = []
        
        if self.hook_type == HookType.BEFORE_MESSAGE_PROCESS:
            security_issues.extend(self._check_message_security(data, security_context))
        elif self.hook_type == HookType.BEFORE_TOOL_CALL:
            security_issues.extend(self._check_tool_security(data, security_context))
        elif self.hook_type == HookType.ON_GUARDRAIL_VIOLATION:
            security_issues.extend(self._check_guardrail_security(data, security_context))
        
        if security_issues:
            security_event["issues"] = security_issues
            self.logger.warning(
                f"Security issues detected in {self.hook_type.value}",
                extra={"issues": security_issues, "user_id": security_context.user_id if security_context else None}
            )
        
        # Store security event
        self.security_events.append(security_event)
        
        # Maintain event log size
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        # Determine if execution should continue
        continue_execution = len(security_issues) == 0 or self.security_level != "strict"
        
        return HookResult(
            success=True,
            continue_execution=continue_execution,
            message=f"Security check completed: {len(security_issues)} issues found",
            modified_data={"security_issues": security_issues} if security_issues else None
        )
    
    def _check_message_security(
        self,
        data: Dict[str, Any],
        security_context: Optional[SecurityContext]
    ) -> List[str]:
        """Check message for security issues."""
        issues = []
        
        message_content = data.get("message", "")
        
        # Check for potential injection attempts
        if any(pattern in message_content.lower() for pattern in [
            "<script", "javascript:", "onclick=", "onerror=", "union select", "drop table"
        ]):
            issues.append("Potential injection attempt detected")
        
        # Check for excessive length
        if len(message_content) > 100000:
            issues.append("Message exceeds maximum length")
        
        # Check security context
        if not security_context:
            issues.append("No security context provided")
        elif security_context.is_expired():
            issues.append("Security context expired")
        
        return issues
    
    def _check_tool_security(
        self,
        data: Dict[str, Any],
        security_context: Optional[SecurityContext]
    ) -> List[str]:
        """Check tool call for security issues."""
        issues = []
        
        tool_name = data.get("tool_name", "")
        tool_args = data.get("arguments", {})
        
        # Check for dangerous tool usage
        dangerous_tools = ["file_operation", "system_command", "network_request"]
        if tool_name in dangerous_tools:
            if not security_context or security_context.clearance_level == "public":
                issues.append(f"Insufficient clearance for dangerous tool: {tool_name}")
        
        # Check tool arguments for suspicious patterns
        if isinstance(tool_args, dict):
            for key, value in tool_args.items():
                if isinstance(value, str):
                    if "../" in value or "\\..\\" in value:
                        issues.append("Path traversal attempt in tool arguments")
                    if any(pattern in value.lower() for pattern in ["rm -rf", "del /s", "format c:"]):
                        issues.append("Destructive command detected in tool arguments")
        
        return issues
    
    def _check_guardrail_security(
        self,
        data: Dict[str, Any],
        security_context: Optional[SecurityContext]
    ) -> List[str]:
        """Check guardrail violation for security implications."""
        issues = []
        
        violation_type = data.get("guardrail_name", "")
        severity = data.get("severity", "")
        
        # Critical violations are security issues
        if severity == "critical":
            issues.append(f"Critical guardrail violation: {violation_type}")
        
        # Multiple violations from same user
        user_id = security_context.user_id if security_context else None
        if user_id:
            recent_violations = sum(
                1 for event in self.security_events[-50:]  # Check last 50 events
                if event.get("user_id") == user_id and "issues" in event
            )
            if recent_violations > 5:
                issues.append(f"User {user_id} has {recent_violations} recent security violations")
        
        return issues
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        total_events = len(self.security_events)
        events_with_issues = len([e for e in self.security_events if "issues" in e])
        
        # Count issue types
        issue_counts = {}
        for event in self.security_events:
            if "issues" in event:
                for issue in event["issues"]:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            "total_events": total_events,
            "events_with_issues": events_with_issues,
            "security_violation_rate": events_with_issues / total_events if total_events > 0 else 0.0,
            "issue_types": issue_counts,
            "recent_events": self.security_events[-10:]  # Last 10 events
        }


class HookManager:
    """Manages and executes hooks throughout the system."""
    
    def __init__(self):
        self.hooks: Dict[HookType, List[BaseHook]] = {hook_type: [] for hook_type in HookType}
        self.logger = get_logger("hook_manager")
        self.global_enabled = True
    
    def register_hook(self, hook: BaseHook):
        """Register a hook for a specific hook type."""
        hook_type = hook.hook_type
        self.hooks[hook_type].append(hook)
        
        # Sort by priority (lower number = higher priority)
        self.hooks[hook_type].sort(key=lambda h: h.priority.value)
        
        self.logger.info(f"Registered hook: {hook.name} for {hook_type.value}")
    
    def unregister_hook(self, hook_name: str, hook_type: HookType) -> bool:
        """Unregister a hook by name and type."""
        hooks_list = self.hooks[hook_type]
        for i, hook in enumerate(hooks_list):
            if hook.name == hook_name:
                del hooks_list[i]
                self.logger.info(f"Unregistered hook: {hook_name} from {hook_type.value}")
                return True
        return False
    
    def get_hooks(self, hook_type: HookType) -> List[BaseHook]:
        """Get all hooks for a specific type."""
        return self.hooks[hook_type].copy()
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None,
        stop_on_failure: bool = False
    ) -> List[HookResult]:
        """Execute all hooks of a specific type."""
        if not self.global_enabled:
            return []
        
        hooks = self.hooks[hook_type]
        if not hooks:
            return []
        
        results = []
        
        self.logger.debug(f"Executing {len(hooks)} hooks for {hook_type.value}")
        
        for hook in hooks:
            try:
                result = await hook(data, context, security_context)
                results.append(result)
                
                # Update data with modifications
                if result.modified_data:
                    data.update(result.modified_data)
                
                # Stop execution if requested
                if not result.continue_execution:
                    self.logger.info(f"Hook {hook.name} requested execution stop")
                    break
                
                # Stop on failure if requested
                if stop_on_failure and not result.success:
                    self.logger.warning(f"Hook {hook.name} failed, stopping execution")
                    break
            
            except Exception as e:
                self.logger.error(f"Hook {hook.name} execution failed: {str(e)}")
                
                if stop_on_failure:
                    break
        
        return results
    
    def enable_hooks(self):
        """Enable hook execution globally."""
        self.global_enabled = True
        self.logger.info("Hook execution enabled globally")
    
    def disable_hooks(self):
        """Disable hook execution globally."""
        self.global_enabled = False
        self.logger.info("Hook execution disabled globally")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all registered hooks."""
        metrics = {}
        
        for hook_type, hooks_list in self.hooks.items():
            metrics[hook_type.value] = [hook.get_metrics() for hook in hooks_list]
        
        return metrics


# Create default hook manager with common hooks
def create_default_hook_manager() -> HookManager:
    """Create a hook manager with default hooks."""
    manager = HookManager()
    
    # Add logging hooks for all major events
    for hook_type in [
        HookType.BEFORE_MESSAGE_PROCESS,
        HookType.AFTER_MESSAGE_PROCESS,
        HookType.BEFORE_TOOL_CALL,
        HookType.AFTER_TOOL_CALL,
        HookType.ON_ERROR,
        HookType.ON_GUARDRAIL_VIOLATION
    ]:
        manager.register_hook(LoggingHook(hook_type=hook_type))
    
    # Add metrics hooks
    for hook_type in [
        HookType.AFTER_MESSAGE_PROCESS,
        HookType.AFTER_TOOL_CALL,
        HookType.ON_ERROR,
        HookType.ON_GUARDRAIL_VIOLATION
    ]:
        manager.register_hook(MetricsHook(hook_type=hook_type))
    
    # Add security hooks
    for hook_type in [
        HookType.BEFORE_MESSAGE_PROCESS,
        HookType.BEFORE_TOOL_CALL,
        HookType.ON_GUARDRAIL_VIOLATION
    ]:
        manager.register_hook(SecurityHook(hook_type=hook_type))
    
    return manager
