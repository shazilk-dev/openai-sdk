"""Comprehensive example showcasing all Advanced Agent System features."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

from .core.config import get_settings
from .core.security import create_security_context, SecurityContext
from .core.logging import get_logger

from .agents.base import AgentCapabilities, SimpleAgent, ToolAgent
from .tools.base import (
    default_tool_registry, 
    WebSearchTool, 
    CalculatorTool, 
    EmailTool,
    FileOperationTool
)
from .guardrails.base import (
    create_default_guardrail_engine,
    ContentLengthGuardrail,
    ProfanityGuardrail,
    PIIGuardrail,
    RateLimitGuardrail,
    TopicGuardrail,
    GuardrailAction
)
from .hooks.base import (
    create_default_hook_manager,
    LoggingHook,
    MetricsHook,
    SecurityHook,
    HookType,
    HookPriority
)
from .tracing.base import (
    global_tracing_processor,
    trace_operation,
    SpanType,
    create_tracing_integration
)
from .workflows.orchestrator import WorkflowEngine, create_customer_support_workflow
from .models.schemas import (
    AgentConfiguration,
    SecurityClearance,
    MessageRole,
    TaskPriority
)


logger = get_logger("examples")


async def comprehensive_example():
    """Comprehensive example demonstrating all system features."""
    
    logger.info("=== Starting Comprehensive Advanced Agent System Example ===")
    
    # 1. Initialize core components
    logger.info("1. Initializing core components...")
    
    # Security context for admin user
    admin_security_context = create_security_context(
        user_id="admin_user",
        clearance_level="top_secret",
        permissions=["view_metrics", "view_traces", "manage_agents", "use_all_tools"]
    )
    
    # Regular user security context
    user_security_context = create_security_context(
        user_id="regular_user",
        clearance_level="public",
        permissions=["basic_chat"]
    )
    
    # Tool registry (already initialized with default tools)
    logger.info(f"Available tools: {list(default_tool_registry.get_all_tools().keys())}")
    
    # Guardrail engine
    guardrail_engine = create_default_guardrail_engine()
    
    # Add custom guardrails
    guardrail_engine.add_guardrail(
        ContentLengthGuardrail(
            max_length=5000,
            action=GuardrailAction.MODIFY
        )
    )
    
    guardrail_engine.add_guardrail(
        TopicGuardrail(
            forbidden_topics=["violence", "illegal", "hacking"],
            action=GuardrailAction.BLOCK
        )
    )
    
    # Hook manager
    hook_manager = create_default_hook_manager()
    
    # Add custom hooks
    custom_hook = LoggingHook(
        hook_type=HookType.AFTER_MESSAGE_PROCESS,
        log_level="INFO",
        priority=HookPriority.HIGH
    )
    hook_manager.register_hook(custom_hook)
    
    # Tracing
    tracing_processor = global_tracing_processor
    tracing_integration = create_tracing_integration()
    
    logger.info("✓ Core components initialized")
    
    # 2. Create and configure agents
    logger.info("2. Creating and configuring agents...")
    
    # Simple conversational agent
    simple_agent_config = AgentConfiguration(
        name="Assistant",
        description="A helpful AI assistant for general conversations",
        instructions="You are a helpful, friendly AI assistant. Answer questions clearly and concisely.",
        model="gpt-4",
        temperature=0.7,
        max_tokens=2048,
        tools_enabled=False,
        security_clearance=SecurityClearance.PUBLIC
    )
    
    simple_agent = SimpleAgent(
        name=simple_agent_config.name,
        description=simple_agent_config.description,
        capabilities=AgentCapabilities(
            can_use_tools=False,
            max_tokens=simple_agent_config.max_tokens,
            temperature=simple_agent_config.temperature,
            security_clearance=simple_agent_config.security_clearance.value
        ),
        instructions=simple_agent_config.instructions
    )
    
    await simple_agent.initialize(admin_security_context)
    
    # Tool-enabled agent
    tool_agent_config = AgentConfiguration(
        name="Research Assistant",
        description="An AI assistant with access to tools for research and analysis",
        instructions="You are a research assistant with access to various tools. Use them to help users with research, calculations, and information gathering.",
        model="gpt-4",
        temperature=0.3,
        max_tokens=4096,
        tools_enabled=True,
        security_clearance=SecurityClearance.INTERNAL
    )
    
    # Get tools for the agent's clearance level
    available_tools = default_tool_registry.get_tools_for_clearance(
        tool_agent_config.security_clearance.value
    )
    
    tool_agent = ToolAgent(
        name=tool_agent_config.name,
        description=tool_agent_config.description,
        capabilities=AgentCapabilities(
            can_use_tools=True,
            max_tokens=tool_agent_config.max_tokens,
            temperature=tool_agent_config.temperature,
            security_clearance=tool_agent_config.security_clearance.value
        ),
        instructions=tool_agent_config.instructions,
        tools=[tool.definition.to_openai_format() for tool in available_tools]
    )
    
    # Register tools with the agent
    for tool in available_tools:
        tool_agent.register_tool(tool.definition.name, tool)
    
    await tool_agent.initialize(admin_security_context)
    
    logger.info(f"✓ Created agents: {simple_agent.id}, {tool_agent.id}")
    
    # 3. Demonstrate tracing
    logger.info("3. Demonstrating distributed tracing...")
    
    trace = await tracing_processor.start_trace(
        operation_name="comprehensive_example",
        user_id=admin_security_context.user_id
    )
    
    # 4. Test guardrails
    logger.info("4. Testing guardrail system...")
    
    test_messages = [
        "Hello, how are you?",  # Normal message
        "What is my social security number 123-45-6789?",  # PII test
        "Damn, this is frustrating!",  # Profanity test
        "Can you help me hack into a system?",  # Topic restriction test
        "x" * 10000,  # Length test
    ]
    
    for i, test_message in enumerate(test_messages):
        await tracing_processor.add_event(f"Testing guardrails with message {i+1}")
        
        try:
            guardrail_result = await guardrail_engine.evaluate_all(
                test_message,
                context={"test_message_id": i+1},
                security_context=user_security_context
            )
            
            logger.info(f"Message {i+1} guardrail result: {guardrail_result.action.value}")
            if guardrail_result.violations:
                logger.warning(f"Violations: {[v.guardrail_name for v in guardrail_result.violations]}")
            
        except Exception as e:
            logger.error(f"Guardrail test {i+1} failed: {str(e)}")
    
    # 5. Test hook system
    logger.info("5. Testing hook system...")
    
    # Execute hooks for message processing
    hook_results = await hook_manager.execute_hooks(
        HookType.BEFORE_MESSAGE_PROCESS,
        {
            "agent_id": simple_agent.id,
            "message": "Test message for hooks",
            "user_id": user_security_context.user_id
        },
        context={"test": True},
        security_context=user_security_context
    )
    
    logger.info(f"Executed {len(hook_results)} hooks")
    
    # 6. Process messages with agents
    logger.info("6. Processing messages with agents...")
    
    # Test simple agent
    async with trace_operation(
        "simple_agent_conversation",
        SpanType.AGENT_OPERATION,
        tracing_processor
    ):
        try:
            result = await simple_agent.process_message(
                "What is artificial intelligence?",
                context={"conversation_type": "educational"},
                security_context=user_security_context
            )
            
            logger.info(f"Simple agent response: {result.get('response', 'No response')[:100]}...")
            logger.info(f"Tokens used: {result.get('tokens_used', 0)}")
            
        except Exception as e:
            logger.error(f"Simple agent test failed: {str(e)}")
    
    # Test tool agent with calculations
    async with trace_operation(
        "tool_agent_calculation",
        SpanType.AGENT_OPERATION,
        tracing_processor
    ):
        try:
            result = await tool_agent.process_message(
                "Calculate the square root of 144 and then multiply it by 5",
                context={"task_type": "calculation"},
                security_context=admin_security_context  # Admin has tool access
            )
            
            logger.info(f"Tool agent response: {result.get('response', 'No response')[:100]}...")
            logger.info(f"Tools used: {result.get('tools_used', 0)}")
            
        except Exception as e:
            logger.error(f"Tool agent test failed: {str(e)}")
    
    # 7. Test workflow orchestration
    logger.info("7. Testing workflow orchestration...")
    
    # Create a mock agent system for workflows
    class MockAgentSystem:
        async def process_message(self, agent_id, message, context=None, security_context=None):
            return {
                "success": True,
                "response": f"Mock response from {agent_id}: {message}",
                "tokens_used": 50
            }
    
    mock_agent_system = MockAgentSystem()
    
    # Create workflow engine
    workflow_engine = WorkflowEngine(
        agent_system=mock_agent_system,
        hook_manager=hook_manager,
        tracing_processor=tracing_processor
    )
    
    # Register and execute customer support workflow
    support_workflow = create_customer_support_workflow()
    workflow_engine.register_workflow(support_workflow)
    
    try:
        execution = await workflow_engine.start_execution(
            workflow_id=support_workflow.id,
            initial_context={"customer_issue": "Login problems"},
            security_context=admin_security_context,
            created_by=admin_security_context.user_id
        )
        
        logger.info(f"Started workflow execution: {execution.execution_id}")
        
        # Wait a bit for workflow to process
        await asyncio.sleep(2)
        
        # Check execution status
        final_execution = workflow_engine.get_execution(execution.execution_id)
        if final_execution:
            logger.info(f"Workflow status: {final_execution.status.value}")
        
    except Exception as e:
        logger.error(f"Workflow test failed: {str(e)}")
    
    # 8. Test advanced tool usage
    logger.info("8. Testing advanced tool usage...")
    
    # Direct tool usage
    calculator = default_tool_registry.get_tool("calculator")
    if calculator:
        try:
            calc_result = await calculator(
                {"expression": "2 + 3 * 4"},
                admin_security_context
            )
            logger.info(f"Calculator result: {calc_result}")
        except Exception as e:
            logger.error(f"Calculator test failed: {str(e)}")
    
    # Web search tool
    web_search = default_tool_registry.get_tool("web_search")
    if web_search:
        try:
            search_result = await web_search(
                {"query": "artificial intelligence", "num_results": 3},
                admin_security_context
            )
            logger.info(f"Search returned {len(search_result.get('results', []))} results")
        except Exception as e:
            logger.error(f"Web search test failed: {str(e)}")
    
    # 9. Security and permissions testing
    logger.info("9. Testing security and permissions...")
    
    # Test with restricted user trying to access high-clearance tool
    file_tool = default_tool_registry.get_tool("file_operation")
    if file_tool:
        try:
            # This should fail due to insufficient clearance
            await file_tool(
                {"operation": "read", "path": "test.txt"},
                user_security_context  # Public clearance trying to use restricted tool
            )
            logger.warning("Security test failed - restricted access was allowed")
        except Exception as e:
            logger.info(f"Security test passed - access denied: {str(e)}")
    
    # 10. Metrics and monitoring
    logger.info("10. Collecting metrics and monitoring data...")
    
    # Agent metrics
    simple_agent_metrics = simple_agent.get_status()
    tool_agent_metrics = tool_agent.get_status()
    
    logger.info(f"Simple agent metrics: {simple_agent_metrics['metrics']}")
    logger.info(f"Tool agent metrics: {tool_agent_metrics['metrics']}")
    
    # Guardrail metrics
    guardrail_metrics = guardrail_engine.get_metrics()
    logger.info(f"Guardrail metrics: Total evaluations: {guardrail_metrics['total_evaluations']}")
    
    # Hook metrics
    hook_metrics = hook_manager.get_all_metrics()
    total_hook_executions = sum(
        sum(hook['execution_count'] for hook in hooks_list)
        for hooks_list in hook_metrics.values()
    )
    logger.info(f"Hook metrics: Total executions: {total_hook_executions}")
    
    # Tracing metrics
    tracing_metrics = tracing_processor.get_metrics()
    logger.info(f"Tracing metrics: {tracing_metrics}")
    
    # Workflow metrics
    workflow_metrics = workflow_engine.get_metrics()
    logger.info(f"Workflow metrics: {workflow_metrics}")
    
    # 11. Error handling and recovery
    logger.info("11. Testing error handling and recovery...")
    
    try:
        # Test with invalid agent
        await simple_agent.process_message(
            None,  # Invalid message
            security_context=user_security_context
        )
    except Exception as e:
        logger.info(f"Error handling test passed: {type(e).__name__}")
    
    # 12. Finish tracing
    await tracing_processor.finish_trace(trace.trace_id)
    
    # 13. Final cleanup
    logger.info("13. Performing cleanup...")
    
    await simple_agent.cleanup()
    await tool_agent.cleanup()
    
    # Final system status
    logger.info("=== Final System Status ===")
    logger.info(f"Traces created: {tracing_metrics['traces_created']}")
    logger.info(f"Spans created: {tracing_metrics['spans_created']}")
    logger.info(f"Events recorded: {tracing_metrics['events_recorded']}")
    logger.info(f"Guardrail evaluations: {guardrail_metrics['total_evaluations']}")
    logger.info(f"Hook executions: {total_hook_executions}")
    logger.info(f"Workflow executions: {workflow_metrics['executions_started']}")
    
    logger.info("=== Comprehensive Example Completed Successfully ===")


async def agent_handoff_example():
    """Example demonstrating agent handoffs and collaboration."""
    
    logger.info("=== Agent Handoff Example ===")
    
    # Create specialized agents
    security_context = create_security_context("user", "internal", ["agent_handoff"])
    
    # Customer service agent
    cs_agent = SimpleAgent(
        name="Customer Service",
        description="Initial customer contact agent",
        capabilities=AgentCapabilities(security_clearance="public"),
        instructions="You handle initial customer inquiries and route them to specialists when needed."
    )
    await cs_agent.initialize(security_context)
    
    # Technical support agent
    tech_agent = ToolAgent(
        name="Technical Support",
        description="Technical specialist with diagnostic tools",
        capabilities=AgentCapabilities(can_use_tools=True, security_clearance="internal"),
        instructions="You are a technical specialist. Use available tools to diagnose and solve technical issues."
    )
    
    # Register diagnostic tools
    tech_agent.register_tool("calculator", default_tool_registry.get_tool("calculator"))
    await tech_agent.initialize(security_context)
    
    # Billing agent
    billing_agent = SimpleAgent(
        name="Billing Support",
        description="Billing and payment specialist",
        capabilities=AgentCapabilities(security_clearance="confidential"),
        instructions="You handle billing inquiries, payment issues, and account management."
    )
    await billing_agent.initialize(security_context)
    
    # Simulate customer interaction with handoffs
    conversation_context = {
        "customer_id": "CUST_001",
        "issue_type": "unknown",
        "conversation_history": []
    }
    
    # 1. Initial contact with customer service
    logger.info("1. Customer contacts customer service...")
    
    cs_response = await cs_agent.process_message(
        "Hi, I'm having trouble with my account. I can't log in and I think there might be an issue with my recent payment.",
        context=conversation_context,
        security_context=security_context
    )
    
    conversation_context["conversation_history"].append({
        "agent": "customer_service",
        "message": cs_response.get("response", "")
    })
    
    logger.info(f"CS Agent: {cs_response.get('response', '')[:100]}...")
    
    # 2. Analyze issue and determine handoff
    if "login" in cs_response.get("response", "").lower():
        # Technical issue - handoff to tech support
        logger.info("2. Issue identified as technical - handing off to tech support...")
        
        handoff_message = f"Customer {conversation_context['customer_id']} is experiencing login issues. Previous conversation: {cs_response.get('response', '')}"
        
        tech_response = await tech_agent.process_message(
            handoff_message,
            context=conversation_context,
            security_context=security_context
        )
        
        conversation_context["conversation_history"].append({
            "agent": "technical_support",
            "message": tech_response.get("response", "")
        })
        
        logger.info(f"Tech Agent: {tech_response.get('response', '')[:100]}...")
        
        # Follow up with technical diagnosis
        tech_followup = await tech_agent.process_message(
            "Please run a diagnostic check and calculate the probability of this being a server issue vs user error.",
            context=conversation_context,
            security_context=security_context
        )
        
        logger.info(f"Tech Diagnosis: {tech_followup.get('response', '')[:100]}...")
    
    # 3. If payment issue mentioned, also involve billing
    if "payment" in cs_response.get("response", "").lower():
        logger.info("3. Payment issue detected - involving billing support...")
        
        billing_message = f"Customer {conversation_context['customer_id']} mentioned payment issues. Context: {conversation_context['conversation_history']}"
        
        billing_response = await billing_agent.process_message(
            billing_message,
            context=conversation_context,
            security_context=security_context
        )
        
        conversation_context["conversation_history"].append({
            "agent": "billing_support",
            "message": billing_response.get("response", "")
        })
        
        logger.info(f"Billing Agent: {billing_response.get('response', '')[:100]}...")
    
    # 4. Final resolution coordination
    logger.info("4. Coordinating final resolution...")
    
    resolution_summary = await cs_agent.process_message(
        f"Please provide a final resolution summary based on the conversation history: {conversation_context['conversation_history']}",
        context=conversation_context,
        security_context=security_context
    )
    
    logger.info(f"Final Resolution: {resolution_summary.get('response', '')}")
    
    # Cleanup
    await cs_agent.cleanup()
    await tech_agent.cleanup()
    await billing_agent.cleanup()
    
    logger.info("=== Agent Handoff Example Completed ===")


async def streaming_example():
    """Example demonstrating streaming responses (simulated)."""
    
    logger.info("=== Streaming Response Example ===")
    
    security_context = create_security_context("user", "public")
    
    # Create streaming agent
    streaming_agent = SimpleAgent(
        name="Streaming Assistant",
        description="Agent that provides streaming responses",
        capabilities=AgentCapabilities(temperature=0.8),
        instructions="Provide detailed, helpful responses that would benefit from streaming."
    )
    
    await streaming_agent.initialize(security_context)
    
    # Simulate streaming by processing message and showing incremental response
    logger.info("Processing message with simulated streaming...")
    
    message = "Can you explain the concept of machine learning in detail, including its applications and future prospects?"
    
    # Start trace for streaming operation
    async with trace_operation(
        "streaming_response",
        SpanType.AGENT_OPERATION,
        global_tracing_processor
    ):
        response = await streaming_agent.process_message(
            message,
            context={"response_mode": "streaming"},
            security_context=security_context
        )
        
        # Simulate streaming chunks
        full_response = response.get("response", "")
        chunk_size = 50
        
        logger.info("Streaming response:")
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i+chunk_size]
            logger.info(f"Chunk {i//chunk_size + 1}: {chunk}")
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    await streaming_agent.cleanup()
    
    logger.info("=== Streaming Example Completed ===")


@tracing_integration.trace_function(
    operation_name="dynamic_instruction_example",
    span_type=SpanType.AGENT_OPERATION
)
async def dynamic_instruction_example():
    """Example demonstrating dynamic instruction updates."""
    
    logger.info("=== Dynamic Instruction Example ===")
    
    security_context = create_security_context("user", "internal")
    
    # Create agent with initial instructions
    adaptive_agent = SimpleAgent(
        name="Adaptive Assistant",
        description="Agent with dynamically updating instructions",
        capabilities=AgentCapabilities(),
        instructions="You are a helpful assistant. Be polite and professional."
    )
    
    await adaptive_agent.initialize(security_context)
    
    # Test 1: Normal interaction
    logger.info("1. Normal interaction...")
    response1 = await adaptive_agent.process_message(
        "Hello, how are you?",
        security_context=security_context
    )
    logger.info(f"Response 1: {response1.get('response', '')[:100]}...")
    
    # Test 2: Update instructions dynamically
    logger.info("2. Updating instructions to be more casual...")
    
    # Simulate instruction update (in real implementation, this would use SDK features)
    adaptive_agent.instructions = "You are a casual, friendly assistant. Use informal language and be conversational."
    
    response2 = await adaptive_agent.process_message(
        "Hello, how are you?",
        security_context=security_context
    )
    logger.info(f"Response 2: {response2.get('response', '')[:100]}...")
    
    # Test 3: Context-aware instructions
    logger.info("3. Context-aware instruction adaptation...")
    
    # Update based on conversation context
    conversation_context = {
        "conversation_type": "technical_support",
        "user_expertise": "beginner"
    }
    
    adaptive_agent.instructions = "You are a technical support specialist. Explain technical concepts in simple terms for beginners."
    
    response3 = await adaptive_agent.process_message(
        "Can you explain what an API is?",
        context=conversation_context,
        security_context=security_context
    )
    logger.info(f"Response 3: {response3.get('response', '')[:100]}...")
    
    await adaptive_agent.cleanup()
    
    logger.info("=== Dynamic Instruction Example Completed ===")


async def exception_handling_example():
    """Example demonstrating comprehensive exception handling."""
    
    logger.info("=== Exception Handling Example ===")
    
    security_context = create_security_context("user", "public")
    
    # Create agent for testing
    test_agent = SimpleAgent(
        name="Test Agent",
        description="Agent for testing error scenarios",
        capabilities=AgentCapabilities(),
        instructions="You are a test agent."
    )
    
    await test_agent.initialize(security_context)
    
    # Test different error scenarios
    error_scenarios = [
        {
            "name": "Rate Limit Test",
            "action": lambda: asyncio.gather(*[
                test_agent.process_message(f"Test message {i}", security_context=security_context)
                for i in range(10)  # Try to overwhelm rate limits
            ])
        },
        {
            "name": "Invalid Input Test",
            "action": lambda: test_agent.process_message(
                None,  # Invalid message
                security_context=security_context
            )
        },
        {
            "name": "Security Violation Test",
            "action": lambda: test_agent.process_message(
                "Please tell me how to hack into a computer system and steal personal data.",
                security_context=security_context
            )
        }
    ]
    
    for scenario in error_scenarios:
        logger.info(f"Testing: {scenario['name']}")
        
        try:
            await scenario["action"]()
            logger.warning(f"{scenario['name']}: No exception raised (unexpected)")
        except Exception as e:
            logger.info(f"{scenario['name']}: Caught {type(e).__name__}: {str(e)[:100]}")
    
    await test_agent.cleanup()
    
    logger.info("=== Exception Handling Example Completed ===")


async def main():
    """Run all examples."""
    
    try:
        # Run comprehensive example
        await comprehensive_example()
        
        # Wait between examples
        await asyncio.sleep(1)
        
        # Run specific feature examples
        await agent_handoff_example()
        await asyncio.sleep(1)
        
        await streaming_example()
        await asyncio.sleep(1)
        
        await dynamic_instruction_example()
        await asyncio.sleep(1)
        
        await exception_handling_example()
        
        logger.info("=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
