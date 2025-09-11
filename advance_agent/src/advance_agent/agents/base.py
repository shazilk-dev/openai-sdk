"""Base agent classes and interfaces for the Advanced Agent System."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run, Message
from openai.types.beta.threads.runs import RunStep

from ..core.config import get_settings
from ..core.exceptions import (
    AgentError, 
    AgentConfigurationError, 
    AgentExecutionError,
    AgentStateError
)
from ..core.logging import get_logger
from ..core.security import SecurityContext, comprehensive_security_analysis


T = TypeVar('T')


class AgentState:
    """Represents the current state of an agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = "idle"
        self.current_task = None
        self.last_activity = datetime.now(timezone.utc)
        self.metadata = {}
        self.conversation_history = []
        self.context_variables = {}
    
    def update_status(self, status: str, task: Optional[str] = None):
        """Update agent status and current task."""
        self.status = status
        self.current_task = task
        self.last_activity = datetime.now(timezone.utc)
    
    def add_to_history(self, entry: Dict[str, Any]):
        """Add entry to conversation history."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.conversation_history.append(entry)
    
    def set_context_variable(self, key: str, value: Any):
        """Set a context variable."""
        self.context_variables[key] = value
    
    def get_context_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context_variables.get(key, default)


class AgentCapabilities:
    """Defines agent capabilities and constraints."""
    
    def __init__(
        self,
        can_use_tools: bool = True,
        can_access_files: bool = False,
        can_generate_images: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        allowed_models: Optional[List[str]] = None,
        security_clearance: str = "public",
        rate_limit_per_minute: int = 60
    ):
        self.can_use_tools = can_use_tools
        self.can_access_files = can_access_files
        self.can_generate_images = can_generate_images
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.allowed_models = allowed_models or ["gpt-4"]
        self.security_clearance = security_clearance
        self.rate_limit_per_minute = rate_limit_per_minute


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: AgentCapabilities,
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.instructions = instructions or f"You are {name}. {description}"
        self.tools = tools or []
        
        self.state = AgentState(self.id)
        self.logger = get_logger(f"agent.{self.name}")
        self.client = None
        self.assistant = None
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0
        }
    
    async def initialize(self, security_context: Optional[SecurityContext] = None):
        """Initialize the agent with OpenAI client and assistant."""
        settings = get_settings()
        
        try:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            # Create assistant
            self.assistant = await self.client.beta.assistants.create(
                name=self.name,
                instructions=self.instructions,
                model=self.capabilities.allowed_models[0],
                tools=[tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in self.tools],
                temperature=self.capabilities.temperature
            )
            
            self.state.update_status("initialized")
            self.logger.info(f"Agent {self.name} initialized with assistant ID: {self.assistant.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            raise AgentConfigurationError(
                f"Failed to initialize agent {self.name}",
                details={"error": str(e), "agent_id": self.id}
            )
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.assistant and self.client:
            try:
                await self.client.beta.assistants.delete(self.assistant.id)
                self.logger.info(f"Agent {self.name} assistant deleted")
            except Exception as e:
                self.logger.warning(f"Failed to delete assistant: {str(e)}")
        
        self.state.update_status("terminated")
    
    @abstractmethod
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Process a message and return response."""
        pass
    
    def validate_security_access(
        self,
        security_context: Optional[SecurityContext],
        required_clearance: str = "public"
    ) -> bool:
        """Validate security access for operation."""
        if not security_context:
            return required_clearance == "public"
        
        return security_context.can_access_clearance(required_clearance)
    
    async def analyze_content_security(self, content: str) -> Dict[str, Any]:
        """Analyze content for security issues."""
        try:
            analysis = comprehensive_security_analysis(content)
            
            if analysis.blocking_issues:
                self.logger.warning(
                    f"Security blocking issues detected: {analysis.blocking_issues}"
                )
            
            return {
                "is_safe": len(analysis.blocking_issues) == 0,
                "risk_score": analysis.overall_risk_score,
                "issues": analysis.blocking_issues,
                "pii_detected": analysis.pii_result.has_pii,
                "safety_violations": analysis.safety_result.violations
            }
            
        except Exception as e:
            self.logger.error(f"Security analysis failed: {str(e)}")
            return {
                "is_safe": False,
                "risk_score": 1.0,
                "issues": ["Security analysis failed"],
                "error": str(e)
            }
    
    def update_metrics(self, success: bool, response_time: float, tokens_used: int):
        """Update agent performance metrics."""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        total_time = self.metrics["average_response_time"] * (self.metrics["total_requests"] - 1)
        self.metrics["average_response_time"] = (total_time + response_time) / self.metrics["total_requests"]
        
        self.metrics["total_tokens_used"] += tokens_used
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.state.status,
            "current_task": self.state.current_task,
            "last_activity": self.state.last_activity.isoformat(),
            "metrics": self.metrics,
            "capabilities": {
                "can_use_tools": self.capabilities.can_use_tools,
                "can_access_files": self.capabilities.can_access_files,
                "can_generate_images": self.capabilities.can_generate_images,
                "security_clearance": self.capabilities.security_clearance
            }
        }


class SimpleAgent(BaseAgent):
    """A simple agent implementation for basic conversational tasks."""
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Process a message using the OpenAI assistant."""
        start_time = datetime.now()
        
        try:
            # Validate security
            if not self.validate_security_access(security_context, self.capabilities.security_clearance):
                raise AgentExecutionError(
                    "Insufficient security clearance",
                    details={"required": self.capabilities.security_clearance}
                )
            
            # Analyze content security
            security_analysis = await self.analyze_content_security(message)
            if not security_analysis["is_safe"]:
                raise AgentExecutionError(
                    "Content security violation",
                    details=security_analysis
                )
            
            # Update state
            self.state.update_status("processing", "message_processing")
            self.state.add_to_history({
                "type": "user_message",
                "content": message,
                "security_context": security_context.user_id if security_context else None
            })
            
            # Create thread and run
            thread = await self.client.beta.threads.create()
            
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message
            )
            
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
                max_prompt_tokens=self.capabilities.max_tokens
            )
            
            # Wait for completion
            while run.status in ["queued", "in_progress"]:
                await asyncio.sleep(1)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == "completed":
                # Get response
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                response_content = messages.data[0].content[0].text.value
                
                # Update state and metrics
                response_time = (datetime.now() - start_time).total_seconds()
                tokens_used = run.usage.total_tokens if run.usage else 0
                
                self.update_metrics(True, response_time, tokens_used)
                self.state.update_status("idle")
                self.state.add_to_history({
                    "type": "assistant_response",
                    "content": response_content,
                    "tokens_used": tokens_used,
                    "response_time": response_time
                })
                
                return {
                    "success": True,
                    "response": response_content,
                    "tokens_used": tokens_used,
                    "response_time": response_time,
                    "run_id": run.id,
                    "thread_id": thread.id
                }
            
            else:
                error_msg = f"Run failed with status: {run.status}"
                if run.last_error:
                    error_msg += f" - {run.last_error.message}"
                
                raise AgentExecutionError(error_msg, details={"run_status": run.status})
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(False, response_time, 0)
            self.state.update_status("error", str(e))
            
            self.logger.error(f"Message processing failed: {str(e)}")
            
            if isinstance(e, AgentError):
                raise
            else:
                raise AgentExecutionError(
                    f"Unexpected error in message processing: {str(e)}",
                    details={"agent_id": self.id, "original_error": str(e)}
                )


class ToolAgent(SimpleAgent):
    """An agent with enhanced tool-calling capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_registry = {}
        self.tool_call_history = []
    
    def register_tool(self, tool_name: str, tool_function: Callable):
        """Register a tool function for this agent."""
        if not self.capabilities.can_use_tools:
            raise AgentConfigurationError("Agent does not have tool usage capability")
        
        self.tool_registry[tool_name] = tool_function
        self.logger.info(f"Registered tool: {tool_name}")
    
    async def handle_tool_calls(self, tool_calls: List[Any], run_id: str, thread_id: str):
        """Handle tool calls during execution."""
        tool_outputs = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = eval(tool_call.function.arguments)
                
                self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                if tool_name in self.tool_registry:
                    # Execute tool function
                    tool_function = self.tool_registry[tool_name]
                    
                    if inspect.iscoroutinefunction(tool_function):
                        result = await tool_function(**tool_args)
                    else:
                        result = tool_function(**tool_args)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": str(result)
                    })
                    
                    # Log tool call
                    self.tool_call_history.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id
                    })
                
                else:
                    self.logger.warning(f"Unknown tool requested: {tool_name}")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Error: Tool '{tool_name}' not found"
                    })
            
            except Exception as e:
                self.logger.error(f"Tool execution failed: {str(e)}")
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Error: Tool execution failed - {str(e)}"
                })
        
        # Submit tool outputs
        await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Enhanced message processing with tool support."""
        start_time = datetime.now()
        
        try:
            # Validate security
            if not self.validate_security_access(security_context, self.capabilities.security_clearance):
                raise AgentExecutionError(
                    "Insufficient security clearance",
                    details={"required": self.capabilities.security_clearance}
                )
            
            # Analyze content security
            security_analysis = await self.analyze_content_security(message)
            if not security_analysis["is_safe"]:
                raise AgentExecutionError(
                    "Content security violation",
                    details=security_analysis
                )
            
            # Update state
            self.state.update_status("processing", "message_with_tools")
            
            # Create thread and run
            thread = await self.client.beta.threads.create()
            
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message
            )
            
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
                max_prompt_tokens=self.capabilities.max_tokens
            )
            
            # Wait for completion with tool call handling
            while run.status in ["queued", "in_progress", "requires_action"]:
                if run.status == "requires_action":
                    # Handle tool calls
                    await self.handle_tool_calls(
                        run.required_action.submit_tool_outputs.tool_calls,
                        run.id,
                        thread.id
                    )
                
                await asyncio.sleep(1)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == "completed":
                # Get response
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                response_content = messages.data[0].content[0].text.value
                
                # Update metrics
                response_time = (datetime.now() - start_time).total_seconds()
                tokens_used = run.usage.total_tokens if run.usage else 0
                
                self.update_metrics(True, response_time, tokens_used)
                self.state.update_status("idle")
                
                return {
                    "success": True,
                    "response": response_content,
                    "tokens_used": tokens_used,
                    "response_time": response_time,
                    "run_id": run.id,
                    "thread_id": thread.id,
                    "tools_used": len([call for call in self.tool_call_history if call["run_id"] == run.id])
                }
            
            else:
                error_msg = f"Run failed with status: {run.status}"
                if run.last_error:
                    error_msg += f" - {run.last_error.message}"
                
                raise AgentExecutionError(error_msg, details={"run_status": run.status})
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(False, response_time, 0)
            self.state.update_status("error", str(e))
            
            self.logger.error(f"Enhanced message processing failed: {str(e)}")
            
            if isinstance(e, AgentError):
                raise
            else:
                raise AgentExecutionError(
                    f"Unexpected error in enhanced message processing: {str(e)}",
                    details={"agent_id": self.id, "original_error": str(e)}
                )
