"""Package initialization for the Advanced Agent System."""

__version__ = "1.0.0"
__author__ = "Advanced Agent System Team"
__description__ = "Enterprise-grade AI agent system with OpenAI SDK integration"

# Core exports
from .core.config import Settings, get_settings
from .core.exceptions import (
    AgentError,
    AgentConfigurationError,
    AgentExecutionError,
    AgentStateError,
    ToolError,
    ToolConfigurationError,
    ToolExecutionError,
    GuardrailViolationError,
    GuardrailConfigurationError,
    HookError,
    HookExecutionError,
    TracingError,
    WorkflowError,
    WorkflowExecutionError,
    SecurityViolationError
)
from .core.logging import setup_logging, get_logger
from .core.security import (
    SecurityContext,
    create_security_context,
    comprehensive_security_analysis,
    generate_secure_token,
    hash_password,
    verify_password
)

# Agent system exports
from .agents.base import (
    BaseAgent,
    SimpleAgent,
    ToolAgent,
    AgentCapabilities,
    AgentState
)

# Model exports
from .models.schemas import (
    AgentStatus,
    MessageRole,
    SecurityClearance,
    TaskPriority,
    TaskStatus,
    Message,
    Task,
    ConversationContext,
    AgentConfiguration,
    SystemHealth,
    APIResponse
)

# Tool system exports
from .tools.base import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    WebSearchTool,
    CalculatorTool,
    EmailTool,
    FileOperationTool,
    default_tool_registry
)

# Guardrail system exports
from .guardrails.base import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailSeverity,
    GuardrailResult,
    ContentLengthGuardrail,
    ProfanityGuardrail,
    PIIGuardrail,
    RateLimitGuardrail,
    TopicGuardrail,
    GuardrailEngine,
    create_default_guardrail_engine
)

# Hook system exports
from .hooks.base import (
    BaseHook,
    HookType,
    HookPriority,
    HookResult,
    LoggingHook,
    MetricsHook,
    SecurityHook,
    HookManager,
    create_default_hook_manager
)

# Tracing system exports
from .tracing.base import (
    TraceLevel,
    SpanType,
    SpanStatus,
    TraceEvent,
    TraceSpan,
    Trace,
    TracingProcessor,
    trace_operation,
    global_tracing_processor,
    get_tracing_processor,
    create_tracing_integration
)

# Workflow system exports
from .workflows.orchestrator import (
    WorkflowStepType,
    WorkflowExecutionStatus,
    WorkflowExecution,
    StepExecutionResult,
    WorkflowEngine,
    create_customer_support_workflow
)

# Main system exports
from .main import AgentSystem, app

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__description__",
    
    # Core
    "Settings",
    "get_settings",
    "setup_logging",
    "get_logger",
    "SecurityContext",
    "create_security_context",
    "comprehensive_security_analysis",
    "generate_secure_token",
    "hash_password",
    "verify_password",
    
    # Exceptions
    "AgentError",
    "AgentConfigurationError",
    "AgentExecutionError",
    "AgentStateError",
    "ToolError",
    "ToolConfigurationError", 
    "ToolExecutionError",
    "GuardrailViolationError",
    "GuardrailConfigurationError",
    "HookError",
    "HookExecutionError",
    "TracingError",
    "WorkflowError",
    "WorkflowExecutionError",
    "SecurityViolationError",
    
    # Agents
    "BaseAgent",
    "SimpleAgent", 
    "ToolAgent",
    "AgentCapabilities",
    "AgentState",
    
    # Models/Schemas
    "AgentStatus",
    "MessageRole",
    "SecurityClearance",
    "TaskPriority",
    "TaskStatus",
    "Message",
    "Task",
    "ConversationContext",
    "AgentConfiguration",
    "SystemHealth",
    "APIResponse",
    
    # Tools
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "WebSearchTool",
    "CalculatorTool",
    "EmailTool",
    "FileOperationTool",
    "default_tool_registry",
    
    # Guardrails
    "BaseGuardrail",
    "GuardrailAction",
    "GuardrailSeverity", 
    "GuardrailResult",
    "ContentLengthGuardrail",
    "ProfanityGuardrail",
    "PIIGuardrail",
    "RateLimitGuardrail",
    "TopicGuardrail",
    "GuardrailEngine",
    "create_default_guardrail_engine",
    
    # Hooks
    "BaseHook",
    "HookType",
    "HookPriority",
    "HookResult",
    "LoggingHook",
    "MetricsHook",
    "SecurityHook",
    "HookManager",
    "create_default_hook_manager",
    
    # Tracing
    "TraceLevel",
    "SpanType",
    "SpanStatus",
    "TraceEvent",
    "TraceSpan",
    "Trace",
    "TracingProcessor",
    "trace_operation",
    "global_tracing_processor",
    "get_tracing_processor",
    "create_tracing_integration",
    
    # Workflows
    "WorkflowStepType",
    "WorkflowExecutionStatus",
    "WorkflowExecution",
    "StepExecutionResult",
    "WorkflowEngine",
    "create_customer_support_workflow",
    
    # Main system
    "AgentSystem",
    "app"
]


def get_system_info() -> dict:
    """Get system information and version details."""
    return {
        "name": "Advanced Agent System",
        "version": __version__,
        "author": __author__, 
        "description": __description__,
        "features": [
            "Enterprise-grade agent management",
            "OpenAI SDK integration",
            "Comprehensive tool system",
            "Advanced guardrails",
            "Hook-based extensibility",
            "Distributed tracing",
            "Workflow orchestration",
            "Security and permissions",
            "Production-ready architecture"
        ],
        "components": {
            "agents": "Multi-type agent system with capabilities",
            "tools": "Extensible tool registry with security",
            "guardrails": "Content safety and compliance",
            "hooks": "Event-driven extensibility",
            "tracing": "Comprehensive observability",
            "workflows": "Agent orchestration",
            "security": "Role-based access control",
            "api": "RESTful API with FastAPI"
        }
    }


def print_banner():
    """Print system banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                    Advanced Agent System                     ║
║                         v{__version__}                            ║
║                                                              ║
║  Enterprise-grade AI agent system with OpenAI SDK          ║
║  Full-featured with guardrails, hooks, tracing & workflows  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main() -> None:
    """Main entry point for the Advanced Agent System."""
    print_banner()
    print("Advanced Agent System initialized!")
    print("Use 'advance-agent --help' for CLI commands or import components for programmatic use.")
