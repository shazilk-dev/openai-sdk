"""Main entry point and orchestration system for the Advanced Agent System."""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core.config import get_settings
from .core.exceptions import (
    AgentError, 
    AgentConfigurationError, 
    AgentExecutionError,
    GuardrailViolationError,
    ToolError,
    SecurityViolationError
)
from .core.logging import get_logger, setup_logging
from .core.security import SecurityContext, create_security_context, verify_access_token

from .agents.base import BaseAgent, SimpleAgent, ToolAgent, AgentCapabilities
from .tools.base import default_tool_registry, ToolRegistry, BaseTool
from .guardrails.base import create_default_guardrail_engine, GuardrailEngine
from .hooks.base import create_default_hook_manager, HookManager, HookType
from .tracing.base import (
    global_tracing_processor, 
    TracingProcessor, 
    SpanType, 
    trace_operation,
    create_tracing_integration
)

from .models.schemas import (
    APIResponse,
    AgentConfiguration,
    Task,
    ConversationContext,
    MessageRole,
    SecurityClearance,
    SystemHealth
)


# Request/Response Models
class CreateAgentRequest(BaseModel):
    config: AgentConfiguration
    security_context: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    agent_id: str
    message: str
    context: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    response: str
    agent_id: str
    tokens_used: int
    response_time: float
    conversation_id: str


class SystemStatusResponse(BaseModel):
    status: str
    timestamp: str
    agents: Dict[str, Any]
    metrics: Dict[str, Any]
    health: SystemHealth


# Global system components
agents_registry: Dict[str, BaseAgent] = {}
tool_registry: ToolRegistry = default_tool_registry
guardrail_engine: GuardrailEngine = create_default_guardrail_engine()
hook_manager: HookManager = create_default_hook_manager()
tracing_processor: TracingProcessor = global_tracing_processor
tracing_integration = create_tracing_integration()

# Authentication
security = HTTPBearer()
logger = get_logger("main")


class AgentSystem:
    """Main agent system orchestrator."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("agent_system")
        self.is_running = False
        self.startup_time = None
        
        # System metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "active_agents": 0,
            "total_agents_created": 0
        }
    
    async def startup(self):
        """Initialize the agent system."""
        self.startup_time = datetime.now(timezone.utc)
        self.is_running = True
        
        # Setup logging
        setup_logging()
        
        # Execute startup hooks
        await hook_manager.execute_hooks(
            HookType.BEFORE_AGENT_INIT,
            {"system": "starting"},
            {"startup_time": self.startup_time.isoformat()}
        )
        
        self.logger.info("Agent system started successfully")
    
    async def shutdown(self):
        """Shutdown the agent system gracefully."""
        self.logger.info("Shutting down agent system...")
        
        # Cleanup all agents
        for agent_id, agent in agents_registry.items():
            try:
                await agent.cleanup()
                self.logger.info(f"Agent {agent_id} cleaned up")
            except Exception as e:
                self.logger.error(f"Failed to cleanup agent {agent_id}: {str(e)}")
        
        # Finish any open traces
        for trace in tracing_processor.get_recent_traces():
            if trace.end_time is None:
                await tracing_processor.finish_trace(trace.trace_id)
        
        self.is_running = False
        self.logger.info("Agent system shutdown complete")
    
    async def create_agent(
        self,
        config: AgentConfiguration,
        security_context: Optional[SecurityContext] = None
    ) -> BaseAgent:
        """Create a new agent instance."""
        async with trace_operation(
            "create_agent",
            SpanType.AGENT_OPERATION,
            tracing_processor
        ) as span:
            span.set_attribute("agent.name", config.name)
            span.set_attribute("agent.model", config.model)
            
            try:
                # Create agent capabilities
                capabilities = AgentCapabilities(
                    can_use_tools=config.tools_enabled,
                    can_access_files=config.file_access_enabled,
                    can_generate_images=config.image_generation_enabled,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    allowed_models=[config.model],
                    security_clearance=config.security_clearance.value,
                    rate_limit_per_minute=config.rate_limit_per_minute
                )
                
                # Get tools for agent's security clearance
                available_tools = tool_registry.get_tools_for_clearance(
                    config.security_clearance.value
                )
                
                # Create appropriate agent type
                if config.tools_enabled and available_tools:
                    agent = ToolAgent(
                        name=config.name,
                        description=config.description,
                        capabilities=capabilities,
                        instructions=config.instructions,
                        tools=[tool.definition.to_openai_format() for tool in available_tools]
                    )
                    
                    # Register tools with agent
                    for tool in available_tools:
                        agent.register_tool(tool.definition.name, tool)
                else:
                    agent = SimpleAgent(
                        name=config.name,
                        description=config.description,
                        capabilities=capabilities,
                        instructions=config.instructions
                    )
                
                # Initialize agent
                await agent.initialize(security_context)
                
                # Register agent
                agents_registry[agent.id] = agent
                
                # Update metrics
                self.metrics["total_agents_created"] += 1
                self.metrics["active_agents"] = len(agents_registry)
                
                # Execute post-creation hooks
                await hook_manager.execute_hooks(
                    HookType.AFTER_AGENT_INIT,
                    {
                        "agent_id": agent.id,
                        "agent_name": config.name,
                        "agent_type": type(agent).__name__
                    },
                    security_context=security_context
                )
                
                span.set_attribute("agent.id", agent.id)
                span.set_attribute("agent.type", type(agent).__name__)
                
                self.logger.info(f"Created agent {agent.id} ({config.name})")
                
                return agent
            
            except Exception as e:
                span.set_attribute("error", str(e))
                self.logger.error(f"Failed to create agent: {str(e)}")
                raise
    
    async def process_message(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Process a message through an agent."""
        async with trace_operation(
            "process_message",
            SpanType.REQUEST,
            tracing_processor
        ) as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("message_length", len(message))
            
            try:
                self.metrics["total_requests"] += 1
                
                # Get agent
                agent = agents_registry.get(agent_id)
                if not agent:
                    raise AgentError(f"Agent {agent_id} not found")
                
                # Execute pre-processing hooks
                hook_data = {
                    "agent_id": agent_id,
                    "message": message,
                    "context": context
                }
                
                await hook_manager.execute_hooks(
                    HookType.BEFORE_MESSAGE_PROCESS,
                    hook_data,
                    context,
                    security_context
                )
                
                # Update message if modified by hooks
                message = hook_data.get("message", message)
                
                # Check guardrails
                async with trace_operation(
                    "guardrail_check",
                    SpanType.GUARDRAIL_CHECK,
                    tracing_processor
                ) as guardrail_span:
                    guardrail_result = await guardrail_engine.evaluate_all(
                        message, context, security_context
                    )
                    
                    guardrail_span.set_attribute("violations_count", len(guardrail_result.violations))
                    guardrail_span.set_attribute("is_allowed", guardrail_result.is_allowed)
                    
                    if not guardrail_result.is_allowed:
                        # Execute guardrail violation hooks
                        violation_data = {
                            "agent_id": agent_id,
                            "message": message,
                            "violations": [v.dict() for v in guardrail_result.violations],
                            "action": guardrail_result.action.value
                        }
                        
                        await hook_manager.execute_hooks(
                            HookType.ON_GUARDRAIL_VIOLATION,
                            violation_data,
                            context,
                            security_context
                        )
                        
                        raise GuardrailViolationError(
                            f"Guardrail violations detected: {[v.guardrail_name for v in guardrail_result.violations]}",
                            details={"violations": [v.dict() for v in guardrail_result.violations]}
                        )
                    
                    # Use modified content if available
                    if guardrail_result.modified_content:
                        message = guardrail_result.modified_content
                
                # Process message through agent
                async with trace_operation(
                    f"agent_process_{agent.name}",
                    SpanType.AGENT_OPERATION,
                    tracing_processor
                ) as agent_span:
                    result = await agent.process_message(message, context, security_context)
                    
                    agent_span.set_attribute("tokens_used", result.get("tokens_used", 0))
                    agent_span.set_attribute("response_time", result.get("response_time", 0))
                
                # Execute post-processing hooks
                post_hook_data = {
                    "agent_id": agent_id,
                    "original_message": message,
                    "response": result.get("response"),
                    "tokens_used": result.get("tokens_used", 0),
                    "response_time": result.get("response_time", 0),
                    "success": result.get("success", False)
                }
                
                await hook_manager.execute_hooks(
                    HookType.AFTER_MESSAGE_PROCESS,
                    post_hook_data,
                    context,
                    security_context
                )
                
                # Update metrics
                if result.get("success", False):
                    self.metrics["successful_requests"] += 1
                else:
                    self.metrics["failed_requests"] += 1
                
                span.set_attribute("success", result.get("success", False))
                span.set_attribute("tokens_used", result.get("tokens_used", 0))
                
                return result
            
            except Exception as e:
                self.metrics["failed_requests"] += 1
                
                # Execute error hooks
                error_data = {
                    "agent_id": agent_id,
                    "message": message,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                await hook_manager.execute_hooks(
                    HookType.ON_ERROR,
                    error_data,
                    context,
                    security_context
                )
                
                span.set_attribute("error", str(e))
                span.set_attribute("error_type", type(e).__name__)
                
                self.logger.error(f"Message processing failed: {str(e)}")
                raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        agent_statuses = {
            agent_id: agent.get_status()
            for agent_id, agent in agents_registry.items()
        }
        
        # Collect metrics from all components
        system_metrics = {
            "system": self.metrics,
            "agents": {
                agent_id: agent.metrics
                for agent_id, agent in agents_registry.items()
            },
            "tools": {
                tool_name: {
                    "call_count": tool.call_count,
                    "last_call": tool.last_call_time.isoformat() if tool.last_call_time else None
                }
                for tool_name, tool in tool_registry.get_all_tools().items()
            },
            "guardrails": guardrail_engine.get_metrics(),
            "hooks": hook_manager.get_all_metrics(),
            "tracing": tracing_processor.get_metrics()
        }
        
        # System health
        health = SystemHealth()
        health.add_service_status("agent_system", "healthy" if self.is_running else "stopped")
        health.add_service_status("agents", f"{len(agents_registry)} active")
        health.add_service_status("tools", f"{len(tool_registry.get_all_tools())} available")
        health.add_service_status("guardrails", f"{len(guardrail_engine.guardrails)} active")
        health.add_service_status("hooks", f"{sum(len(hooks) for hooks in hook_manager.hooks.values())} registered")
        
        health.add_metric("total_requests", self.metrics["total_requests"])
        health.add_metric("success_rate", 
                         self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1))
        
        return {
            "status": "healthy" if self.is_running else "stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                if self.startup_time else 0
            ),
            "agents": agent_statuses,
            "metrics": system_metrics,
            "health": health
        }


# Global system instance
agent_system = AgentSystem()


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await agent_system.startup()
    yield
    # Shutdown
    await agent_system.shutdown()


app = FastAPI(
    title="Advanced Agent System",
    description="Enterprise-grade AI agent system with OpenAI SDK integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication dependency
async def get_security_context(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> SecurityContext:
    """Extract security context from request."""
    token = credentials.credentials
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return SecurityContext(
        user_id=payload.get("sub", "unknown"),
        clearance_level=SecurityClearance(payload.get("clearance", "public")),
        permissions=payload.get("permissions", [])
    )


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return APIResponse.success_response({"status": "healthy"})


@app.get("/status")
async def get_system_status() -> SystemStatusResponse:
    """Get comprehensive system status."""
    status = agent_system.get_system_status()
    return SystemStatusResponse(**status)


@app.post("/agents", response_model=APIResponse)
async def create_agent(
    request: CreateAgentRequest,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Create a new agent."""
    try:
        agent = await agent_system.create_agent(request.config, security_context)
        
        return APIResponse.success_response(
            data={
                "agent_id": agent.id,
                "name": agent.name,
                "status": agent.state.status
            },
            message="Agent created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        return APIResponse.error_response(str(e))


@app.get("/agents/{agent_id}")
async def get_agent_status(
    agent_id: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get agent status."""
    agent = agents_registry.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return APIResponse.success_response(agent.get_status())


@app.post("/agents/{agent_id}/message", response_model=APIResponse)
async def send_message(
    agent_id: str,
    request: MessageRequest,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Send a message to an agent."""
    try:
        result = await agent_system.process_message(
            agent_id=request.agent_id,
            message=request.message,
            context=request.context,
            security_context=security_context
        )
        
        response_data = MessageResponse(
            response=result.get("response", ""),
            agent_id=agent_id,
            tokens_used=result.get("tokens_used", 0),
            response_time=result.get("response_time", 0.0),
            conversation_id=result.get("thread_id", str(uuid4()))
        )
        
        return APIResponse.success_response(
            data=response_data.dict(),
            message="Message processed successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to process message: {str(e)}")
        return APIResponse.error_response(str(e))


@app.get("/agents")
async def list_agents(
    security_context: SecurityContext = Depends(get_security_context)
):
    """List all agents."""
    agents_list = [
        {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "status": agent.state.status,
            "last_activity": agent.state.last_activity.isoformat()
        }
        for agent in agents_registry.values()
    ]
    
    return APIResponse.success_response(agents_list)


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Delete an agent."""
    agent = agents_registry.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        await agent.cleanup()
        del agents_registry[agent_id]
        agent_system.metrics["active_agents"] = len(agents_registry)
        
        return APIResponse.success_response(
            message=f"Agent {agent_id} deleted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
        return APIResponse.error_response(str(e))


@app.get("/tools")
async def list_tools(
    security_context: SecurityContext = Depends(get_security_context)
):
    """List available tools."""
    available_tools = tool_registry.get_tools_for_clearance(
        security_context.clearance_level.value
    )
    
    tools_list = [
        {
            "name": tool.definition.name,
            "description": tool.definition.description,
            "security_clearance": tool.definition.security_clearance,
            "parameters": [param.dict() for param in tool.definition.parameters]
        }
        for tool in available_tools
    ]
    
    return APIResponse.success_response(tools_list)


@app.get("/metrics")
async def get_metrics(
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get system metrics."""
    if not security_context.has_permission("view_metrics"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    status = agent_system.get_system_status()
    return APIResponse.success_response(status["metrics"])


@app.get("/traces")
async def get_traces(
    limit: int = 50,
    user_id: Optional[str] = None,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get trace data."""
    if not security_context.has_permission("view_traces"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if user_id:
        traces = tracing_processor.get_traces_by_user(user_id, limit)
    else:
        traces = tracing_processor.get_recent_traces(limit)
    
    traces_data = [trace.to_dict() for trace in traces]
    
    return APIResponse.success_response(traces_data)


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(agent_system.shutdown())
    sys.exit(0)


if __name__ == "__main__":
    import uvicorn
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the application
    settings = get_settings()
    
    uvicorn.run(
        "advance_agent.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
