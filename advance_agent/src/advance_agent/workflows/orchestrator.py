"""Workflow orchestration system for the Advanced Agent System."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel

from ..core.exceptions import WorkflowError, WorkflowExecutionError
from ..core.logging import get_logger
from ..core.security import SecurityContext
from ..models.schemas import (
    Workflow, 
    WorkflowStep, 
    Task, 
    TaskStatus, 
    TaskPriority,
    AgentConfiguration
)
from ..agents.base import BaseAgent
from ..hooks.base import HookManager, HookType
from ..tracing.base import TracingProcessor, SpanType, trace_operation


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    WAIT = "wait"
    HANDOFF = "handoff"
    HUMAN_REVIEW = "human_review"
    DATA_TRANSFORM = "data_transform"


class WorkflowExecutionStatus(str, Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowExecution(BaseModel):
    """Represents an execution instance of a workflow."""
    execution_id: str
    workflow_id: str
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.PENDING
    current_step_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    error: Optional[str] = None
    created_by: str
    
    class Config:
        use_enum_values = True


class StepExecutionResult(BaseModel):
    """Result of executing a workflow step."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    next_steps: List[str] = []
    context_updates: Dict[str, Any] = {}
    execution_time: float = 0.0


class WorkflowEngine:
    """Engine for executing workflows with agent coordination."""
    
    def __init__(
        self,
        agent_system: Any,  # Reference to main agent system
        hook_manager: HookManager,
        tracing_processor: TracingProcessor
    ):
        self.agent_system = agent_system
        self.hook_manager = hook_manager
        self.tracing_processor = tracing_processor
        self.logger = get_logger("workflow_engine")
        
        # Storage
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Step executors
        self.step_executors: Dict[WorkflowStepType, Callable] = {
            WorkflowStepType.AGENT_MESSAGE: self._execute_agent_message,
            WorkflowStepType.TOOL_CALL: self._execute_tool_call,
            WorkflowStepType.CONDITIONAL: self._execute_conditional,
            WorkflowStepType.PARALLEL: self._execute_parallel,
            WorkflowStepType.WAIT: self._execute_wait,
            WorkflowStepType.HANDOFF: self._execute_handoff,
            WorkflowStepType.HUMAN_REVIEW: self._execute_human_review,
            WorkflowStepType.DATA_TRANSFORM: self._execute_data_transform
        }
        
        # Metrics
        self.metrics = {
            "workflows_created": 0,
            "executions_started": 0,
            "executions_completed": 0,
            "executions_failed": 0,
            "total_steps_executed": 0
        }
    
    def register_workflow(self, workflow: Workflow):
        """Register a workflow definition."""
        self.workflows[workflow.id] = workflow
        self.metrics["workflows_created"] += 1
        self.logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    async def start_execution(
        self,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None,
        created_by: str = "system"
    ) -> WorkflowExecution:
        """Start a new workflow execution."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise WorkflowError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            context=initial_context or {},
            created_by=created_by
        )
        
        self.executions[execution_id] = execution
        self.metrics["executions_started"] += 1
        
        # Execute workflow start hooks
        await self.hook_manager.execute_hooks(
            HookType.ON_WORKFLOW_START,
            {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "workflow_name": workflow.name
            },
            {"workflow": workflow.dict()},
            security_context
        )
        
        # Start execution asynchronously
        asyncio.create_task(self._execute_workflow(execution, security_context))
        
        self.logger.info(f"Started workflow execution: {execution_id} for workflow: {workflow.name}")
        
        return execution
    
    async def _execute_workflow(
        self,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ):
        """Execute a workflow from start to finish."""
        workflow = self.get_workflow(execution.workflow_id)
        if not workflow:
            execution.status = WorkflowExecutionStatus.FAILED
            execution.error = f"Workflow {execution.workflow_id} not found"
            return
        
        async with trace_operation(
            f"workflow_execution_{workflow.name}",
            SpanType.WORKFLOW_STEP,
            self.tracing_processor
        ) as span:
            span.set_attribute("workflow_id", execution.workflow_id)
            span.set_attribute("execution_id", execution.execution_id)
            
            try:
                execution.status = WorkflowExecutionStatus.RUNNING
                execution.started_at = datetime.now(timezone.utc)
                
                # Find first step (usually has no dependencies)
                first_steps = [
                    step for step in workflow.steps
                    if not any(
                        step.id in other_step.next_steps
                        for other_step in workflow.steps
                    )
                ]
                
                if not first_steps:
                    raise WorkflowError("No starting step found in workflow")
                
                # Execute steps
                pending_steps = first_steps.copy()
                completed_steps = set()
                
                while pending_steps:
                    current_step = pending_steps.pop(0)
                    execution.current_step_id = current_step.id
                    
                    self.logger.debug(f"Executing step: {current_step.name}")
                    
                    # Execute step
                    result = await self._execute_step(
                        current_step,
                        execution,
                        security_context
                    )
                    
                    self.metrics["total_steps_executed"] += 1
                    
                    if result.success:
                        completed_steps.add(current_step.id)
                        
                        # Update execution context
                        execution.context.update(result.context_updates)
                        execution.results[current_step.id] = result.result
                        
                        # Add next steps to pending if their dependencies are met
                        for next_step_id in current_step.next_steps:
                            next_step = workflow.get_step(next_step_id)
                            if next_step and next_step_id not in completed_steps:
                                # Check if all dependencies are completed
                                dependencies_met = all(
                                    dep_id in completed_steps
                                    for dep_id in getattr(next_step, 'dependencies', [])
                                )
                                
                                if dependencies_met and next_step not in pending_steps:
                                    pending_steps.append(next_step)
                    else:
                        # Step failed
                        if current_step.retry_count < current_step.max_retries:
                            current_step.retry_count += 1
                            pending_steps.insert(0, current_step)  # Retry immediately
                            self.logger.warning(
                                f"Step {current_step.name} failed, retrying ({current_step.retry_count}/{current_step.max_retries})"
                            )
                        else:
                            # Max retries reached
                            execution.status = WorkflowExecutionStatus.FAILED
                            execution.error = f"Step {current_step.name} failed: {result.error}"
                            break
                
                # Check final status
                if execution.status == WorkflowExecutionStatus.RUNNING:
                    execution.status = WorkflowExecutionStatus.COMPLETED
                    execution.completed_at = datetime.now(timezone.utc)
                    self.metrics["executions_completed"] += 1
                else:
                    self.metrics["executions_failed"] += 1
                
                # Execute completion hooks
                await self.hook_manager.execute_hooks(
                    HookType.ON_WORKFLOW_COMPLETE,
                    {
                        "workflow_id": execution.workflow_id,
                        "execution_id": execution.execution_id,
                        "status": execution.status.value,
                        "results": execution.results
                    },
                    {"execution": execution.dict()},
                    security_context
                )
                
                span.set_attribute("status", execution.status.value)
                span.set_attribute("steps_completed", len(completed_steps))
                
                self.logger.info(
                    f"Workflow execution {execution.execution_id} completed with status: {execution.status.value}"
                )
            
            except Exception as e:
                execution.status = WorkflowExecutionStatus.FAILED
                execution.error = str(e)
                execution.completed_at = datetime.now(timezone.utc)
                self.metrics["executions_failed"] += 1
                
                span.set_attribute("error", str(e))
                
                self.logger.error(f"Workflow execution {execution.execution_id} failed: {str(e)}")
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a single workflow step."""
        async with trace_operation(
            f"step_{step.name}",
            SpanType.WORKFLOW_STEP,
            self.tracing_processor
        ) as span:
            span.set_attribute("step_id", step.id)
            span.set_attribute("step_name", step.name)
            span.set_attribute("step_action", step.action)
            
            start_time = datetime.now()
            
            try:
                # Check conditions
                if step.conditions:
                    if not self._evaluate_conditions(step.conditions, execution.context):
                        return StepExecutionResult(
                            success=True,
                            result="Skipped due to conditions",
                            next_steps=step.next_steps
                        )
                
                # Execute based on action type
                step_type = WorkflowStepType(step.action)
                executor = self.step_executors.get(step_type)
                
                if not executor:
                    raise WorkflowExecutionError(f"Unknown step action: {step.action}")
                
                result = await executor(step, execution, security_context)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time = execution_time
                
                span.set_attribute("success", result.success)
                span.set_attribute("execution_time", execution_time)
                
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                span.set_attribute("error", str(e))
                span.set_attribute("execution_time", execution_time)
                
                return StepExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
    
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate step conditions against execution context."""
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            if isinstance(expected_value, dict):
                # Handle operators like {"$gt": 5}, {"$eq": "value"}
                for operator, value in expected_value.items():
                    if operator == "$eq" and actual_value != value:
                        return False
                    elif operator == "$ne" and actual_value == value:
                        return False
                    elif operator == "$gt" and actual_value <= value:
                        return False
                    elif operator == "$gte" and actual_value < value:
                        return False
                    elif operator == "$lt" and actual_value >= value:
                        return False
                    elif operator == "$lte" and actual_value > value:
                        return False
                    elif operator == "$in" and actual_value not in value:
                        return False
                    elif operator == "$nin" and actual_value in value:
                        return False
            else:
                # Direct comparison
                if actual_value != expected_value:
                    return False
        
        return True
    
    # Step Executors
    async def _execute_agent_message(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute an agent message step."""
        agent_id = step.parameters.get("agent_id")
        message = step.parameters.get("message", "")
        
        # Template message with context variables
        for key, value in execution.context.items():
            message = message.replace(f"{{{key}}}", str(value))
        
        try:
            result = await self.agent_system.process_message(
                agent_id=agent_id,
                message=message,
                context=execution.context,
                security_context=security_context
            )
            
            return StepExecutionResult(
                success=result.get("success", False),
                result=result.get("response"),
                next_steps=step.next_steps,
                context_updates={
                    f"{step.id}_response": result.get("response"),
                    f"{step.id}_tokens": result.get("tokens_used", 0)
                }
            )
        
        except Exception as e:
            return StepExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def _execute_tool_call(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a tool call step."""
        tool_name = step.parameters.get("tool_name")
        tool_args = step.parameters.get("arguments", {})
        
        # Template arguments with context variables
        templated_args = {}
        for key, value in tool_args.items():
            if isinstance(value, str):
                for ctx_key, ctx_value in execution.context.items():
                    value = value.replace(f"{{{ctx_key}}}", str(ctx_value))
            templated_args[key] = value
        
        try:
            # Get tool from registry
            from ..tools.base import default_tool_registry
            tool = default_tool_registry.get_tool(tool_name)
            
            if not tool:
                raise WorkflowExecutionError(f"Tool {tool_name} not found")
            
            result = await tool(templated_args, security_context)
            
            return StepExecutionResult(
                success=True,
                result=result,
                next_steps=step.next_steps,
                context_updates={f"{step.id}_result": result}
            )
        
        except Exception as e:
            return StepExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def _execute_conditional(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a conditional step."""
        condition = step.parameters.get("condition", {})
        true_steps = step.parameters.get("true_steps", [])
        false_steps = step.parameters.get("false_steps", [])
        
        condition_result = self._evaluate_conditions(condition, execution.context)
        next_steps = true_steps if condition_result else false_steps
        
        return StepExecutionResult(
            success=True,
            result={"condition_result": condition_result},
            next_steps=next_steps,
            context_updates={f"{step.id}_condition": condition_result}
        )
    
    async def _execute_parallel(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute parallel steps."""
        parallel_steps = step.parameters.get("parallel_steps", [])
        
        # Create tasks for parallel execution
        tasks = []
        for step_id in parallel_steps:
            workflow = self.get_workflow(execution.workflow_id)
            parallel_step = workflow.get_step(step_id) if workflow else None
            
            if parallel_step:
                task = asyncio.create_task(
                    self._execute_step(parallel_step, execution, security_context)
                )
                tasks.append((step_id, task))
        
        # Wait for all tasks to complete
        results = {}
        success = True
        errors = []
        
        for step_id, task in tasks:
            try:
                result = await task
                results[step_id] = result.result
                if not result.success:
                    success = False
                    errors.append(f"{step_id}: {result.error}")
            except Exception as e:
                success = False
                errors.append(f"{step_id}: {str(e)}")
        
        return StepExecutionResult(
            success=success,
            result=results,
            next_steps=step.next_steps if success else [],
            context_updates={f"{step.id}_parallel_results": results},
            error="; ".join(errors) if errors else None
        )
    
    async def _execute_wait(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a wait step."""
        wait_seconds = step.parameters.get("seconds", 1)
        
        await asyncio.sleep(wait_seconds)
        
        return StepExecutionResult(
            success=True,
            result=f"Waited {wait_seconds} seconds",
            next_steps=step.next_steps
        )
    
    async def _execute_handoff(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute an agent handoff step."""
        from_agent_id = step.parameters.get("from_agent")
        to_agent_id = step.parameters.get("to_agent")
        handoff_message = step.parameters.get("message", "Handoff initiated")
        
        # Template handoff message
        for key, value in execution.context.items():
            handoff_message = handoff_message.replace(f"{{{key}}}", str(value))
        
        # Record handoff in context
        handoff_data = {
            "from_agent": from_agent_id,
            "to_agent": to_agent_id,
            "message": handoff_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return StepExecutionResult(
            success=True,
            result=handoff_data,
            next_steps=step.next_steps,
            context_updates={
                f"{step.id}_handoff": handoff_data,
                "current_agent": to_agent_id
            }
        )
    
    async def _execute_human_review(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a human review step (pauses workflow)."""
        review_data = {
            "step_id": step.id,
            "execution_id": execution.execution_id,
            "review_prompt": step.parameters.get("prompt", "Human review required"),
            "context": execution.context,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Pause execution
        execution.status = WorkflowExecutionStatus.PAUSED
        
        return StepExecutionResult(
            success=True,
            result=review_data,
            context_updates={f"{step.id}_review_pending": True}
        )
    
    async def _execute_data_transform(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        security_context: Optional[SecurityContext] = None
    ) -> StepExecutionResult:
        """Execute a data transformation step."""
        transform_type = step.parameters.get("type", "copy")
        source_key = step.parameters.get("source")
        target_key = step.parameters.get("target")
        
        if source_key not in execution.context:
            return StepExecutionResult(
                success=False,
                error=f"Source key '{source_key}' not found in context"
            )
        
        source_value = execution.context[source_key]
        
        # Apply transformation
        if transform_type == "copy":
            result = source_value
        elif transform_type == "upper":
            result = str(source_value).upper()
        elif transform_type == "lower":
            result = str(source_value).lower()
        elif transform_type == "length":
            result = len(source_value) if hasattr(source_value, "__len__") else 0
        elif transform_type == "json_parse":
            import json
            result = json.loads(str(source_value))
        elif transform_type == "json_stringify":
            import json
            result = json.dumps(source_value)
        else:
            return StepExecutionResult(
                success=False,
                error=f"Unknown transform type: {transform_type}"
            )
        
        return StepExecutionResult(
            success=True,
            result=result,
            next_steps=step.next_steps,
            context_updates={target_key: result}
        )
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get a workflow execution by ID."""
        return self.executions.get(execution_id)
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active workflow executions."""
        return [
            execution for execution in self.executions.values()
            if execution.status in [WorkflowExecutionStatus.RUNNING, WorkflowExecutionStatus.PAUSED]
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics."""
        active_executions = len(self.get_active_executions())
        
        return {
            **self.metrics,
            "active_executions": active_executions,
            "total_executions": len(self.executions),
            "workflows_registered": len(self.workflows)
        }


# Example workflow definitions
def create_customer_support_workflow() -> Workflow:
    """Create a customer support workflow."""
    steps = [
        WorkflowStep(
            id="greeting",
            name="Initial Greeting",
            description="Greet the customer and understand their issue",
            action=WorkflowStepType.AGENT_MESSAGE.value,
            parameters={
                "agent_id": "support_agent",
                "message": "Hello! I'm here to help you with your issue. Can you please describe what you're experiencing?"
            },
            next_steps=["categorize_issue"]
        ),
        WorkflowStep(
            id="categorize_issue",
            name="Categorize Issue",
            description="Categorize the customer issue",
            action=WorkflowStepType.TOOL_CALL.value,
            parameters={
                "tool_name": "categorize_text",
                "arguments": {
                    "text": "{greeting_response}",
                    "categories": ["technical", "billing", "general"]
                }
            },
            next_steps=["route_to_specialist"]
        ),
        WorkflowStep(
            id="route_to_specialist",
            name="Route to Specialist",
            description="Route to appropriate specialist",
            action=WorkflowStepType.CONDITIONAL.value,
            parameters={
                "condition": {"categorize_issue_result": {"$eq": "technical"}},
                "true_steps": ["technical_support"],
                "false_steps": ["general_support"]
            }
        ),
        WorkflowStep(
            id="technical_support",
            name="Technical Support",
            description="Handle technical issues",
            action=WorkflowStepType.AGENT_MESSAGE.value,
            parameters={
                "agent_id": "tech_support_agent",
                "message": "I understand you're having a technical issue. Let me help you troubleshoot this."
            },
            next_steps=["resolution"]
        ),
        WorkflowStep(
            id="general_support",
            name="General Support",
            description="Handle general inquiries",
            action=WorkflowStepType.AGENT_MESSAGE.value,
            parameters={
                "agent_id": "general_support_agent",
                "message": "I'll help you with your inquiry. Let me look into this for you."
            },
            next_steps=["resolution"]
        ),
        WorkflowStep(
            id="resolution",
            name="Issue Resolution",
            description="Resolve the customer issue",
            action=WorkflowStepType.AGENT_MESSAGE.value,
            parameters={
                "agent_id": "current_agent",
                "message": "Based on our conversation, here's the solution to your issue: {solution}"
            },
            next_steps=["satisfaction_survey"]
        ),
        WorkflowStep(
            id="satisfaction_survey",
            name="Customer Satisfaction Survey",
            description="Collect feedback from customer",
            action=WorkflowStepType.AGENT_MESSAGE.value,
            parameters={
                "agent_id": "survey_agent",
                "message": "Thank you for contacting support. On a scale of 1-10, how satisfied were you with the service?"
            },
            next_steps=[]
        )
    ]
    
    return Workflow(
        name="Customer Support",
        description="Automated customer support workflow with intelligent routing",
        steps=steps,
        created_by="system"
    )
