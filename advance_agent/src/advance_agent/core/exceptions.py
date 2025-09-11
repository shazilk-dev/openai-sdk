"""Custom exception definitions for the Advanced Agent System."""

from typing import Optional, Dict, Any, List

# Attempt to import shared/base exceptions from the external SDK or package.
# If those imports are not available (or would cause a circular import),
# provide lightweight fallback definitions so this module can be imported
# safely during package initialization.
try:
    # Prefer external SDK exceptions if available
    from agents.exceptions import (
        AgentError,
        MaxTurnsExceeded,
        ModelBehaviorError,
        ToolExecutionError,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
        HandoffError,
    )
except Exception:
    # Fallback minimal definitions to avoid circular imports or missing deps
    class AgentError(Exception):
        """Base agent exception fallback."""
        pass

    class MaxTurnsExceeded(AgentError):
        pass

    class ModelBehaviorError(AgentError):
        pass

    class ToolExecutionError(AgentError):
        pass

    class InputGuardrailTripwireTriggered(AgentError):
        pass

    class OutputGuardrailTripwireTriggered(AgentError):
        pass

    class HandoffError(AgentError):
        pass


class AdvancedAgentError(AgentError):
    """Base exception for Advanced Agent System."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []


class ConfigurationError(AdvancedAgentError):
    """Raised when there are configuration issues."""
    
    def __init__(self, message: str, missing_config: Optional[List[str]] = None):
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context={"missing_config": missing_config or []},
            suggestions=[
                "Check your environment variables",
                "Verify configuration file syntax",
                "Ensure all required settings are provided"
            ]
        )


class SecurityViolationError(AdvancedAgentError):
    """Raised when security violations are detected."""
    
    def __init__(
        self, 
        message: str, 
        violation_type: str,
        severity: str = "medium",
        detected_patterns: Optional[List[str]] = None
    ):
        super().__init__(
            message,
            error_code="SECURITY_VIOLATION",
            context={
                "violation_type": violation_type,
                "severity": severity,
                "detected_patterns": detected_patterns or []
            },
            suggestions=[
                "Review input data for sensitive information",
                "Check content against security policies",
                "Consider using data sanitization"
            ]
        )


class WorkflowExecutionError(AdvancedAgentError):
    """Raised when workflow execution fails."""
    
    def __init__(
        self, 
        message: str,
        workflow_id: str,
        failed_stage: Optional[str] = None,
        recovery_possible: bool = True
    ):
        super().__init__(
            message,
            error_code="WORKFLOW_EXECUTION_ERROR",
            context={
                "workflow_id": workflow_id,
                "failed_stage": failed_stage,
                "recovery_possible": recovery_possible
            },
            suggestions=[
                "Check workflow configuration",
                "Verify agent availability",
                "Review input data validity",
                "Consider retry with different parameters"
            ] if recovery_possible else [
                "Manual intervention required",
                "Contact system administrator"
            ]
        )


class AgentCommunicationError(AdvancedAgentError):
    """Raised when agents fail to communicate properly."""
    
    def __init__(
        self, 
        message: str,
        source_agent: str,
        target_agent: str,
        handoff_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="AGENT_COMMUNICATION_ERROR",
            context={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "handoff_data": handoff_data
            },
            suggestions=[
                "Verify agent handoff configuration",
                "Check data format compatibility",
                "Ensure target agent is available",
                "Review handoff parameters"
            ]
        )


class ToolValidationError(AdvancedAgentError):
    """Raised when tool validation fails."""
    
    def __init__(
        self, 
        message: str,
        tool_name: str,
        validation_errors: List[str],
        input_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="TOOL_VALIDATION_ERROR",
            context={
                "tool_name": tool_name,
                "validation_errors": validation_errors,
                "input_data": input_data
            },
            suggestions=[
                "Check tool input parameters",
                "Verify data types and formats",
                "Review tool documentation",
                "Validate required fields"
            ]
        )


class ResourceExhaustionError(AdvancedAgentError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self, 
        message: str,
        resource_type: str,
        current_usage: Optional[Dict[str, Any]] = None,
        limits: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="RESOURCE_EXHAUSTION",
            context={
                "resource_type": resource_type,
                "current_usage": current_usage or {},
                "limits": limits or {}
            },
            suggestions=[
                "Wait for resources to become available",
                "Consider reducing concurrent operations",
                "Review resource allocation settings",
                "Contact administrator if limits need adjustment"
            ]
        )


class DataProcessingError(AdvancedAgentError):
    """Raised when data processing fails."""
    
    def __init__(
        self, 
        message: str,
        processing_stage: str,
        data_type: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="DATA_PROCESSING_ERROR",
            context={
                "processing_stage": processing_stage,
                "data_type": data_type,
                "error_details": error_details or {}
            },
            suggestions=[
                "Verify input data format",
                "Check data integrity",
                "Review processing parameters",
                "Consider alternative processing methods"
            ]
        )


class ExternalServiceError(AdvancedAgentError):
    """Raised when external service calls fail."""
    
    def __init__(
        self, 
        message: str,
        service_name: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="EXTERNAL_SERVICE_ERROR",
            context={
                "service_name": service_name,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_data": response_data
            },
            suggestions=[
                "Check service availability",
                "Verify API credentials",
                "Review request parameters",
                "Consider fallback options",
                "Implement retry with backoff"
            ]
        )


class BusinessLogicError(AdvancedAgentError):
    """Raised when business logic validation fails."""
    
    def __init__(
        self, 
        message: str,
        rule_name: str,
        violated_constraints: List[str],
        business_context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code="BUSINESS_LOGIC_ERROR",
            context={
                "rule_name": rule_name,
                "violated_constraints": violated_constraints,
                "business_context": business_context or {}
            },
            suggestions=[
                "Review business rule configuration",
                "Check input data against business constraints",
                "Consider rule exceptions or overrides",
                "Contact business analyst for clarification"
            ]
        )


# Enhanced SDK Exception Wrappers
class EnhancedMaxTurnsExceeded(MaxTurnsExceeded, AdvancedAgentError):
    """Enhanced version of MaxTurnsExceeded with additional context."""
    
    def __init__(
        self, 
        message: str, 
        agent_name: str,
        current_turns: int,
        max_turns: int,
        conversation_context: Optional[Dict[str, Any]] = None
    ):
        MaxTurnsExceeded.__init__(self, message)
        AdvancedAgentError.__init__(
            self,
            message,
            error_code="MAX_TURNS_EXCEEDED",
            context={
                "agent_name": agent_name,
                "current_turns": current_turns,
                "max_turns": max_turns,
                "conversation_context": conversation_context or {}
            },
            suggestions=[
                "Consider increasing max_turns limit",
                "Review conversation complexity",
                "Implement conversation summarization",
                "Break down complex tasks into smaller steps"
            ]
        )


class EnhancedModelBehaviorError(ModelBehaviorError, AdvancedAgentError):
    """Enhanced version of ModelBehaviorError with additional context."""
    
    def __init__(
        self, 
        message: str,
        model_name: str,
        expected_behavior: str,
        actual_behavior: str,
        model_response: Optional[str] = None
    ):
        ModelBehaviorError.__init__(self, message)
        AdvancedAgentError.__init__(
            self,
            message,
            error_code="MODEL_BEHAVIOR_ERROR",
            context={
                "model_name": model_name,
                "expected_behavior": expected_behavior,
                "actual_behavior": actual_behavior,
                "model_response": model_response
            },
            suggestions=[
                "Review model instructions and prompts",
                "Consider adjusting temperature settings",
                "Implement stricter output validation",
                "Try alternative model configuration"
            ]
        )


class EnhancedToolExecutionError(ToolExecutionError, AdvancedAgentError):
    """Enhanced version of ToolExecutionError with additional context."""
    
    def __init__(
        self, 
        message: str,
        tool_name: str,
        execution_context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 3
    ):
        ToolExecutionError.__init__(self, message)
        AdvancedAgentError.__init__(
            self,
            message,
            error_code="TOOL_EXECUTION_ERROR",
            context={
                "tool_name": tool_name,
                "execution_context": execution_context or {},
                "retry_count": retry_count,
                "max_retries": max_retries
            },
            suggestions=[
                "Check tool configuration and parameters",
                "Verify tool dependencies are available",
                "Review input data format",
                "Consider implementing fallback tools"
            ] if retry_count >= max_retries else [
                "Tool will be retried automatically",
                "Check tool availability",
                "Verify input parameters"
            ]
        )


# Exception utility functions
def create_error_context(
    error: Exception,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error context for logging and monitoring."""
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_module": getattr(error, "__module__", "unknown"),
    }
    
    if isinstance(error, AdvancedAgentError):
        context.update({
            "error_code": error.error_code,
            "error_context": error.context,
            "suggestions": error.suggestions
        })
    
    if additional_context:
        context.update(additional_context)
    
    return context


def format_error_message(error: Exception, include_suggestions: bool = True) -> str:
    """Format error message with suggestions for user-friendly display."""
    message = str(error)
    
    if isinstance(error, AdvancedAgentError) and include_suggestions and error.suggestions:
        suggestions = "\n".join(f"  • {suggestion}" for suggestion in error.suggestions)
        message += f"\n\nSuggestions:\n{suggestions}"
    
    return message
