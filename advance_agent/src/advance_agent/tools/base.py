"""Tool definitions and implementations for the Advanced Agent System."""

import asyncio
import json
import re
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ..core.exceptions import ToolError, ToolConfigurationError, ToolExecutionError
from ..core.logging import get_logger
from ..core.security import SecurityContext, sanitize_input


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by agents."""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    security_clearance: str = "public"
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_def = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                param_def["enum"] = param.enum
            if param.pattern:
                param_def["pattern"] = param.pattern
            if param.minimum is not None:
                param_def["minimum"] = param.minimum
            if param.maximum is not None:
                param_def["maximum"] = param.maximum
            
            properties[param.name] = param_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self.logger = get_logger(f"tool.{definition.name}")
        self.call_count = 0
        self.last_call_time = None
        self.rate_limit_tracker = {}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool parameters."""
        validated = {}
        
        for param in self.definition.parameters:
            value = parameters.get(param.name)
            
            # Check required parameters
            if param.required and value is None:
                if param.default is not None:
                    value = param.default
                else:
                    raise ToolError(
                        f"Required parameter '{param.name}' is missing",
                        details={"tool": self.definition.name, "parameter": param.name}
                    )
            
            # Type validation and conversion
            if value is not None:
                validated[param.name] = self._validate_parameter_type(param, value)
        
        return validated
    
    def _validate_parameter_type(self, param: ToolParameter, value: Any) -> Any:
        """Validate and convert parameter type."""
        if param.type == "string":
            str_value = str(value)
            # Sanitize string input
            str_value = sanitize_input(str_value)
            
            if param.pattern:
                if not re.match(param.pattern, str_value):
                    raise ToolError(
                        f"Parameter '{param.name}' does not match pattern",
                        details={"pattern": param.pattern, "value": str_value}
                    )
            
            if param.enum and str_value not in param.enum:
                raise ToolError(
                    f"Parameter '{param.name}' must be one of {param.enum}",
                    details={"value": str_value, "allowed": param.enum}
                )
            
            return str_value
        
        elif param.type == "number":
            try:
                num_value = float(value)
                
                if param.minimum is not None and num_value < param.minimum:
                    raise ToolError(
                        f"Parameter '{param.name}' must be >= {param.minimum}",
                        details={"value": num_value, "minimum": param.minimum}
                    )
                
                if param.maximum is not None and num_value > param.maximum:
                    raise ToolError(
                        f"Parameter '{param.name}' must be <= {param.maximum}",
                        details={"value": num_value, "maximum": param.maximum}
                    )
                
                return num_value
            except (ValueError, TypeError):
                raise ToolError(
                    f"Parameter '{param.name}' must be a number",
                    details={"value": value, "type": type(value).__name__}
                )
        
        elif param.type == "boolean":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            else:
                return bool(value)
        
        elif param.type == "array":
            if not isinstance(value, list):
                raise ToolError(
                    f"Parameter '{param.name}' must be an array",
                    details={"value": value, "type": type(value).__name__}
                )
            return value
        
        elif param.type == "object":
            if not isinstance(value, dict):
                raise ToolError(
                    f"Parameter '{param.name}' must be an object",
                    details={"value": value, "type": type(value).__name__}
                )
            return value
        
        return value
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if rate limit is exceeded for user."""
        now = datetime.now(timezone.utc)
        minute_key = now.strftime("%Y-%m-%d-%H-%M")
        
        if user_id not in self.rate_limit_tracker:
            self.rate_limit_tracker[user_id] = {}
        
        user_tracker = self.rate_limit_tracker[user_id]
        current_count = user_tracker.get(minute_key, 0)
        
        if current_count >= self.definition.rate_limit_per_minute:
            return False
        
        user_tracker[minute_key] = current_count + 1
        
        # Clean old entries
        for key in list(user_tracker.keys()):
            if key != minute_key:
                del user_tracker[key]
        
        return True
    
    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    async def __call__(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Any:
        """Call the tool with validation and security checks."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate security clearance
            if security_context:
                if not security_context.can_access_clearance(self.definition.security_clearance):
                    raise ToolError(
                        f"Insufficient security clearance for tool '{self.definition.name}'",
                        details={
                            "required": self.definition.security_clearance,
                            "user_clearance": security_context.clearance_level
                        }
                    )
                
                # Check rate limit
                if not self.check_rate_limit(security_context.user_id):
                    raise ToolError(
                        f"Rate limit exceeded for tool '{self.definition.name}'",
                        details={"limit": self.definition.rate_limit_per_minute}
                    )
            
            # Validate parameters
            validated_params = self.validate_parameters(parameters)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self.execute(validated_params, security_context),
                    timeout=self.definition.timeout_seconds
                )
                
                self.call_count += 1
                self.last_call_time = datetime.now(timezone.utc)
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                self.logger.info(
                    f"Tool '{self.definition.name}' executed successfully",
                    extra={
                        "tool_name": self.definition.name,
                        "execution_time": execution_time,
                        "user_id": security_context.user_id if security_context else None
                    }
                )
                
                return result
                
            except asyncio.TimeoutError:
                raise ToolExecutionError(
                    f"Tool '{self.definition.name}' timed out",
                    details={"timeout": self.definition.timeout_seconds}
                )
        
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self.logger.error(
                f"Tool '{self.definition.name}' execution failed: {str(e)}",
                extra={
                    "tool_name": self.definition.name,
                    "execution_time": execution_time,
                    "error": str(e),
                    "user_id": security_context.user_id if security_context else None
                }
            )
            
            if isinstance(e, ToolError):
                raise
            else:
                raise ToolExecutionError(
                    f"Unexpected error in tool '{self.definition.name}': {str(e)}",
                    details={"original_error": str(e)}
                )


class WebSearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def __init__(self, api_key: Optional[str] = None):
        definition = ToolDefinition(
            name="web_search",
            description="Search the web for information using a search engine",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to execute",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of search results to return",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=20
                )
            ],
            security_clearance="public",
            rate_limit_per_minute=30,
            timeout_seconds=15
        )
        super().__init__(definition)
        self.api_key = api_key
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Execute web search."""
        query = parameters["query"]
        num_results = int(parameters.get("num_results", 5))
        
        try:
            # Simulate web search (replace with actual search API)
            results = []
            
            # Mock search results for demonstration
            for i in range(num_results):
                results.append({
                    "title": f"Search Result {i+1} for '{query}'",
                    "url": f"https://example.com/result-{i+1}",
                    "snippet": f"This is a mock search result snippet for query '{query}'. "
                             f"It contains relevant information about the search topic.",
                    "rank": i + 1
                })
            
            return {
                "query": query,
                "num_results": len(results),
                "results": results,
                "search_time": 0.15  # Mock search time
            }
        
        except Exception as e:
            raise ToolExecutionError(
                f"Web search failed: {str(e)}",
                details={"query": query, "error": str(e)}
            )


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations and evaluations",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')",
                    required=True,
                    pattern=r'^[0-9+\-*/().\s]+$'
                )
            ],
            security_clearance="public",
            rate_limit_per_minute=100,
            timeout_seconds=5
        )
        super().__init__(definition)
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Execute mathematical calculation."""
        expression = parameters["expression"]
        
        try:
            # Safely evaluate mathematical expression
            # Only allow basic math operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                raise ToolError(
                    "Invalid characters in mathematical expression",
                    details={"expression": expression}
                )
            
            # Evaluate the expression
            result = eval(expression)
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
        
        except ZeroDivisionError:
            raise ToolExecutionError(
                "Division by zero in mathematical expression",
                details={"expression": expression}
            )
        except Exception as e:
            raise ToolExecutionError(
                f"Mathematical evaluation failed: {str(e)}",
                details={"expression": expression, "error": str(e)}
            )


class EmailTool(BaseTool):
    """Tool for sending emails."""
    
    def __init__(self, smtp_config: Optional[Dict[str, Any]] = None):
        definition = ToolDefinition(
            name="send_email",
            description="Send an email to specified recipients",
            parameters=[
                ToolParameter(
                    name="to",
                    type="string",
                    description="Email address of the recipient",
                    required=True,
                    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                ),
                ToolParameter(
                    name="subject",
                    type="string",
                    description="Email subject line",
                    required=True
                ),
                ToolParameter(
                    name="body",
                    type="string",
                    description="Email body content",
                    required=True
                ),
                ToolParameter(
                    name="priority",
                    type="string",
                    description="Email priority level",
                    required=False,
                    default="normal",
                    enum=["low", "normal", "high"]
                )
            ],
            security_clearance="internal",
            rate_limit_per_minute=10,
            timeout_seconds=30
        )
        super().__init__(definition)
        self.smtp_config = smtp_config or {}
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Execute email sending."""
        to = parameters["to"]
        subject = parameters["subject"]
        body = parameters["body"]
        priority = parameters.get("priority", "normal")
        
        try:
            # Mock email sending for demonstration
            # In a real implementation, use smtplib or email service API
            
            email_id = str(uuid4())
            
            self.logger.info(
                f"Sending email to {to} with subject '{subject}'",
                extra={
                    "email_id": email_id,
                    "recipient": to,
                    "subject": subject,
                    "priority": priority,
                    "sender": security_context.user_id if security_context else "system"
                }
            )
            
            # Simulate email delivery time
            await asyncio.sleep(0.5)
            
            return {
                "email_id": email_id,
                "to": to,
                "subject": subject,
                "priority": priority,
                "status": "sent",
                "sent_at": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            raise ToolExecutionError(
                f"Email sending failed: {str(e)}",
                details={"to": to, "subject": subject, "error": str(e)}
            )


class FileOperationTool(BaseTool):
    """Tool for file operations."""
    
    def __init__(self, base_directory: str = "/tmp"):
        definition = ToolDefinition(
            name="file_operation",
            description="Perform file operations like read, write, or list",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="File operation to perform",
                    required=True,
                    enum=["read", "write", "list", "delete", "exists"]
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="File or directory path",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write (for write operation)",
                    required=False
                )
            ],
            security_clearance="restricted",
            rate_limit_per_minute=20,
            timeout_seconds=10
        )
        super().__init__(definition)
        self.base_directory = base_directory
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Execute file operation."""
        operation = parameters["operation"]
        path = parameters["path"]
        content = parameters.get("content")
        
        try:
            # Security check: ensure path is within base directory
            import os
            full_path = os.path.join(self.base_directory, path)
            if not full_path.startswith(self.base_directory):
                raise ToolError(
                    "Path traversal attempt detected",
                    details={"path": path, "base_directory": self.base_directory}
                )
            
            if operation == "read":
                # Mock file read
                return {
                    "operation": "read",
                    "path": path,
                    "content": f"Mock content of file: {path}",
                    "size": 42
                }
            
            elif operation == "write":
                if not content:
                    raise ToolError("Content is required for write operation")
                
                # Mock file write
                return {
                    "operation": "write",
                    "path": path,
                    "bytes_written": len(content),
                    "success": True
                }
            
            elif operation == "list":
                # Mock directory listing
                return {
                    "operation": "list",
                    "path": path,
                    "files": [
                        {"name": "file1.txt", "size": 123, "type": "file"},
                        {"name": "file2.txt", "size": 456, "type": "file"},
                        {"name": "subdirectory", "type": "directory"}
                    ]
                }
            
            elif operation == "exists":
                # Mock file existence check
                return {
                    "operation": "exists",
                    "path": path,
                    "exists": True
                }
            
            elif operation == "delete":
                # Mock file deletion
                return {
                    "operation": "delete",
                    "path": path,
                    "success": True
                }
        
        except Exception as e:
            raise ToolExecutionError(
                f"File operation failed: {str(e)}",
                details={"operation": operation, "path": path, "error": str(e)}
            )


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.logger = get_logger("tool_registry")
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self.tools[tool.definition.name] = tool
        self.logger.info(f"Registered tool: {tool.definition.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self.tools.copy()
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-formatted tool definitions."""
        return [
            tool.definition.to_openai_format()
            for tool in self.tools.values()
        ]
    
    def get_tools_for_clearance(self, clearance_level: str) -> List[BaseTool]:
        """Get tools available for a specific security clearance."""
        available_tools = []
        
        clearance_levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2,
            "restricted": 3,
            "top_secret": 4
        }
        
        user_level = clearance_levels.get(clearance_level.lower(), 0)
        
        for tool in self.tools.values():
            tool_level = clearance_levels.get(tool.definition.security_clearance.lower(), 0)
            if user_level >= tool_level:
                available_tools.append(tool)
        
        return available_tools


# Create default tool registry
default_tool_registry = ToolRegistry()

# Register default tools
default_tool_registry.register_tool(WebSearchTool())
default_tool_registry.register_tool(CalculatorTool())
default_tool_registry.register_tool(EmailTool())
default_tool_registry.register_tool(FileOperationTool())
