"""
🏃‍♂️ OPENAI AGENTS SDK - COMPLETE RUNNER GUIDE & TEMPLATE

This template covers everything about the Runner class and agent execution:
- Runner.run() vs Runner.run_sync() vs Runner.run_streamed()
- RunConfig and execution parameters
- Context passing and management
- Session management
- Error handling and max_turns
- Streaming patterns
- Custom AgentRunner implementation

📚 Based on: https://openai.github.io/openai-agents-python/ref/run/
"""

import os
import asyncio
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

from agents import (
    Agent, 
    Runner, 
    function_tool,
    RunConfig,
    ModelSettings,
    SQLiteSession
)
from agents.memory import Session
from agents.models.multi_provider import MultiProvider
from agents.lifecycle import RunHooks
from agents.run_context import RunContextWrapper

# =============================================================================
# 📖 UNDERSTANDING THE RUNNER CLASS
# =============================================================================

"""
🎯 WHY RUNNER IS DESIGNED THIS WAY:

1. ✅ SEPARATION OF CONCERNS:
   - Agent defines WHAT (capabilities, instructions)
   - Runner defines HOW (execution, context, configuration)

2. ✅ STATELESS EXECUTION:
   - Runner doesn't hold state between runs
   - Each run is independent and isolated
   - Enables easy testing and debugging

3. ✅ FLEXIBLE EXECUTION MODES:
   - run(): Async execution with full control
   - run_sync(): Synchronous wrapper for simple use cases  
   - run_streamed(): Real-time streaming of results

4. ✅ CONFIGURATION SEPARATION:
   - User input passed to run() method
   - Agent configuration stays in Agent class
   - Runtime configuration in RunConfig

RUNNER CLASS METHODS:
```python
class Runner:
    @classmethod
    async def run(
        cls,
        starting_agent: Agent,
        input: str | list[ResponseInputItem],
        context: Any = None,
        max_turns: int = 10,
        hooks: RunHooks = None,
        run_config: RunConfig = None,
        session: Session = None
    ) -> RunResult
    
    @classmethod  
    def run_sync(...) -> RunResult
    
    @classmethod
    def run_streamed(...) -> RunResultStreaming
```
"""

# =============================================================================
# 🔧 1. BASIC RUNNER USAGE PATTERNS
# =============================================================================

def basic_runner_examples():
    """Fundamental Runner usage patterns"""
    
    # Create a simple agent for testing
    test_agent = Agent(
        name="TestBot",
        instructions="You are a helpful assistant. Keep responses concise and clear."
    )
    
    print("🔧 BASIC RUNNER PATTERNS")
    print("=" * 50)
    
    # 1.1 Simplest Usage - Synchronous
    print("\n1.1 ✅ SIMPLEST SYNCHRONOUS RUN")
    try:
        result = Runner.run_sync(test_agent, "Hello, how are you?")
        print(f"Input: Hello, how are you?")
        print(f"Output: {result.final_output}")
        print(f"Agent Used: {result.last_agent.name}")
        print(f"Total Turns: {len(result.run_steps)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 1.2 With Multiple Inputs
    print("\n1.2 ✅ MULTIPLE SEQUENTIAL RUNS")
    inputs = [
        "What's 2 + 2?",
        "Explain what you just calculated.",
        "Can you help with other math?"
    ]
    
    for i, user_input in enumerate(inputs, 1):
        try:
            result = Runner.run_sync(test_agent, user_input)
            print(f"{i}. Input: {user_input}")
            print(f"   Output: {result.final_output}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return test_agent

async def async_runner_examples():
    """Asynchronous Runner patterns"""
    
    test_agent = Agent(
        name="AsyncBot", 
        instructions="You are a helpful assistant for async operations."
    )
    
    print("\n🔄 ASYNC RUNNER PATTERNS")
    print("=" * 50)
    
    # 2.1 Basic Async Run
    print("\n2.1 ✅ BASIC ASYNC RUN")
    result = await Runner.run(test_agent, "Hello from async world!")
    print(f"Async Result: {result.final_output}")
    
    # 2.2 Concurrent Runs (Multiple agents at once)
    print("\n2.2 ✅ CONCURRENT AGENT RUNS")
    
    tasks = [
        Runner.run(test_agent, "What's the weather like?"),
        Runner.run(test_agent, "Tell me a fun fact."),
        Runner.run(test_agent, "What's 15 * 7?")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"Concurrent Task {i}: {result.final_output}")

# =============================================================================
# ⚙️ 2. RUNCONFIG - GLOBAL EXECUTION SETTINGS
# =============================================================================

def run_config_examples():
    """RunConfig usage for customizing execution behavior"""
    
    test_agent = Agent(
        name="ConfigBot",
        instructions="You are a test agent for RunConfig examples."
    )
    
    print("\n⚙️ RUNCONFIG EXAMPLES")
    print("=" * 50)
    
    # 2.1 Basic RunConfig with Model Override
    print("\n2.1 ✅ MODEL OVERRIDE via RunConfig")
    
    config = RunConfig(
        model="gpt-4o-mini",  # Override agent's default model
        model_settings=ModelSettings(
            temperature=0.7,
            max_tokens=150
        )
    )
    
    try:
        result = Runner.run_sync(
            test_agent, 
            "Tell me about AI in 2 sentences.",
            run_config=config
        )
        print(f"With Model Override: {result.final_output}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2.2 Multiple Model Providers
    print("\n2.2 ✅ MULTI-PROVIDER CONFIGURATION")
    
    multi_config = RunConfig(
        model_provider=MultiProvider(),  # Supports multiple LLM providers
        model_settings=ModelSettings(
            temperature=0.3,  # Low temperature for consistent responses
            max_tokens=100
        )
    )
    
    try:
        result = Runner.run_sync(
            test_agent,
            "What is 1 + 1? Be precise.",
            run_config=multi_config
        )
        print(f"Multi-provider Result: {result.final_output}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2.3 Advanced RunConfig with All Options
    print("\n2.3 ✅ COMPREHENSIVE RUNCONFIG")
    
    advanced_config = RunConfig(
        model="gpt-4o",
        model_settings=ModelSettings(
            temperature=0.5,
            max_tokens=200,
            top_p=0.9
        ),
        # Add more configuration as needed
    )
    
    try:
        result = Runner.run_sync(
            test_agent,
            "Explain machine learning in simple terms.",
            run_config=advanced_config,
            max_turns=3  # Limit conversation turns
        )
        print(f"Advanced Config Result: {result.final_output}")
        print(f"Turns Used: {len(result.run_steps)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return [config, multi_config, advanced_config]

# =============================================================================
# 🎯 3. CONTEXT PASSING & MANAGEMENT
# =============================================================================

def context_examples():
    """Context passing and management patterns"""
    
    # Agent with dynamic instructions that use context
    def context_aware_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        user_context = ctx.context or {}
        user_name = user_context.get('name', 'User')
        user_role = user_context.get('role', 'general')
        user_preferences = user_context.get('preferences', {})
        
        base = f"Hello {user_name}! "
        
        if user_role == 'developer':
            persona = "You are a senior software engineer. Focus on technical accuracy and best practices."
        elif user_role == 'student':
            persona = "You are a patient tutor. Explain concepts step by step with examples."
        elif user_role == 'manager':
            persona = "You are a business consultant. Focus on practical implications and ROI."
        else:
            persona = "You are a helpful general assistant."
        
        # Add preference handling
        communication_style = user_preferences.get('style', 'balanced')
        if communication_style == 'concise':
            style_note = " Keep responses brief and to the point."
        elif communication_style == 'detailed':
            style_note = " Provide comprehensive explanations with examples."
        else:
            style_note = " Balance detail with clarity."
        
        return base + persona + style_note
    
    context_agent = Agent(
        name="ContextBot",
        instructions=context_aware_instructions
    )
    
    print("\n🎯 CONTEXT MANAGEMENT EXAMPLES")
    print("=" * 50)
    
    # 3.1 Basic Context Passing
    print("\n3.1 ✅ BASIC CONTEXT PASSING")
    
    basic_context = {
        'name': 'Alice',
        'role': 'developer',
        'preferences': {'style': 'concise'}
    }
    
    try:
        result = Runner.run_sync(
            context_agent,
            "How do I optimize database queries?",
            context=basic_context
        )
        print(f"Context: {basic_context}")
        print(f"Response: {result.final_output}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3.2 Different Contexts, Same Question
    print("\n3.2 ✅ SAME QUESTION, DIFFERENT CONTEXTS")
    
    contexts = [
        {'name': 'Bob', 'role': 'student', 'preferences': {'style': 'detailed'}},
        {'name': 'Carol', 'role': 'manager', 'preferences': {'style': 'concise'}},
        {'name': 'Dave', 'role': 'general', 'preferences': {'style': 'balanced'}}
    ]
    
    question = "What is machine learning?"
    
    for i, context in enumerate(contexts, 1):
        try:
            result = Runner.run_sync(context_agent, question, context=context)
            print(f"\n{i}. Context: {context}")
            print(f"   Response: {result.final_output[:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 3.3 Complex Context with Business Logic
    print("\n3.3 ✅ COMPLEX BUSINESS CONTEXT")
    
    business_context = {
        'user_id': 'user_12345',
        'company': 'TechCorp',
        'subscription_tier': 'premium',
        'usage_quota': {'remaining': 50, 'limit': 100},
        'permissions': ['read_data', 'analyze', 'export'],
        'session_data': {
            'login_time': '2025-01-01T10:00:00Z',
            'previous_queries': 5
        }
    }
    
    def business_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        context = ctx.context or {}
        
        user_id = context.get('user_id', 'unknown')
        company = context.get('company', 'Unknown Company')
        tier = context.get('subscription_tier', 'basic')
        
        base = f"Welcome {company} user ({user_id})! "
        
        if tier == 'premium':
            capabilities = "You have access to all premium features including advanced analytics and unlimited queries."
        elif tier == 'professional':
            capabilities = "You have professional-level access with standard analytics."
        else:
            capabilities = "You have basic access. Upgrade for more features!"
        
        return base + capabilities + " How can I assist you today?"
    
    business_agent = Agent(
        name="BusinessBot",
        instructions=business_instructions
    )
    
    try:
        result = Runner.run_sync(
            business_agent,
            "What can you help me with?",
            context=business_context
        )
        print(f"Business Context Result: {result.final_output}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return context_agent

# =============================================================================
# 💾 4. SESSION MANAGEMENT
# =============================================================================

def session_management_examples():
    """Session management for conversation history"""
    
    session_agent = Agent(
        name="SessionBot",
        instructions="You are a helpful assistant. Remember our conversation history and reference previous topics when relevant."
    )
    
    print("\n💾 SESSION MANAGEMENT EXAMPLES")
    print("=" * 50)
    
    # 4.1 SQLite Session (Persistent Memory)
    print("\n4.1 ✅ SQLITE SESSION - PERSISTENT MEMORY")
    
    # Create a session with unique ID
    session = SQLiteSession("conversation_123", "chat_history.db")
    
    conversation_turns = [
        "Hi, my name is Alice and I'm a Python developer.",
        "What's my name?",
        "What programming language do I work with?",
        "Can you help me with a Python question?",
        "Actually, let's talk about something else. What's the weather like?"
    ]
    
    for i, turn in enumerate(conversation_turns, 1):
        try:
            result = Runner.run_sync(
                session_agent,
                turn,
                session=session
            )
            print(f"\nTurn {i}:")
            print(f"  User: {turn}")
            print(f"  Bot: {result.final_output}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 4.2 Multiple Sessions (Different Conversations)
    print("\n4.2 ✅ MULTIPLE SESSIONS - SEPARATE CONVERSATIONS")
    
    session_alice = SQLiteSession("alice_chat", "chat_history.db")
    session_bob = SQLiteSession("bob_chat", "chat_history.db")
    
    # Alice's conversation
    try:
        result_alice = Runner.run_sync(
            session_agent,
            "Hi, I'm Alice. I love machine learning.",
            session=session_alice
        )
        print(f"Alice: {result_alice.final_output}")
    except Exception as e:
        print(f"❌ Alice Error: {e}")
    
    # Bob's conversation (separate session)
    try:
        result_bob = Runner.run_sync(
            session_agent,
            "Hello, I'm Bob. I'm into web development.",
            session=session_bob
        )
        print(f"Bob: {result_bob.final_output}")
    except Exception as e:
        print(f"❌ Bob Error: {e}")
    
    # Continue Alice's conversation
    try:
        result_alice2 = Runner.run_sync(
            session_agent,
            "What did I say I love?",
            session=session_alice
        )
        print(f"Alice (continued): {result_alice2.final_output}")
    except Exception as e:
        print(f"❌ Alice Error: {e}")
    
    return session

# =============================================================================
# 🚨 5. ERROR HANDLING & MAX_TURNS
# =============================================================================

def error_handling_examples():
    """Error handling and max_turns management"""
    
    print("\n🚨 ERROR HANDLING & MAX_TURNS")
    print("=" * 50)
    
    # 5.1 Max Turns Limiting
    print("\n5.1 ✅ MAX_TURNS CONTROL")
    
    # Agent that might make multiple tool calls
    @function_tool
    def slow_operation(task: str) -> str:
        """A tool that simulates work and might trigger multiple calls."""
        return f"Processing {task}... This might need follow-up."
    
    multi_turn_agent = Agent(
        name="MultiTurnBot",
        instructions="""
        You help with complex tasks that might require multiple steps.
        Break down complex requests and use tools as needed.
        """,
        tools=[slow_operation]
    )
    
    try:
        # Limit to 3 turns to prevent infinite loops
        result = Runner.run_sync(
            multi_turn_agent,
            "Help me with a very complex multi-step project analysis.",
            max_turns=3
        )
        print(f"Limited Result: {result.final_output}")
        print(f"Turns Used: {len(result.run_steps)}")
    except Exception as e:
        print(f"❌ Max Turns Error: {e}")
    
    # 5.2 Error Recovery Patterns
    print("\n5.2 ✅ ERROR RECOVERY PATTERNS")
    
    def safe_agent_run(agent: Agent, user_input: str, context: Dict = None) -> str:
        """Safe wrapper for agent runs with error recovery"""
        
        max_retries = 3
        base_max_turns = 5
        
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}")
                
                result = Runner.run_sync(
                    agent,
                    user_input,
                    context=context,
                    max_turns=base_max_turns + attempt  # Increase turns on retry
                )
                
                return result.final_output
                
            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    return f"Sorry, I encountered persistent errors: {e}"
                
                # Wait before retry
                import time
                time.sleep(1)
        
        return "Maximum retries exceeded."
    
    # Test error recovery
    test_agent = Agent(
        name="TestBot",
        instructions="You are a helpful assistant."
    )
    
    safe_result = safe_agent_run(
        test_agent,
        "Tell me about quantum computing.",
        context={'user_id': 'test_user'}
    )
    
    print(f"Safe Result: {safe_result}")
    
    # 5.3 Graceful Degradation
    print("\n5.3 ✅ GRACEFUL DEGRADATION")
    
    def run_with_fallback(primary_agent: Agent, fallback_agent: Agent, user_input: str) -> str:
        """Try primary agent, fallback to simpler agent on error"""
        
        try:
            print("  Trying primary agent...")
            result = Runner.run_sync(primary_agent, user_input, max_turns=2)
            return f"Primary: {result.final_output}"
            
        except Exception as e:
            print(f"  Primary failed: {e}")
            print("  Falling back to simple agent...")
            
            try:
                result = Runner.run_sync(fallback_agent, user_input, max_turns=1)
                return f"Fallback: {result.final_output}"
                
            except Exception as e2:
                return f"Both agents failed. Primary: {e}, Fallback: {e2}"
    
    # Complex agent (might fail)
    complex_agent = Agent(
        name="ComplexBot",
        instructions="You are a very sophisticated AI with complex reasoning capabilities.",
        # Might have issues with complex tools, models, etc.
    )
    
    # Simple fallback agent
    simple_agent = Agent(
        name="SimpleBot", 
        instructions="You are a basic helpful assistant. Give simple, direct answers."
    )
    
    fallback_result = run_with_fallback(
        complex_agent,
        simple_agent,
        "What's 2 + 2?"
    )
    
    print(f"Fallback Result: {fallback_result}")

# =============================================================================
# 🌊 6. STREAMING PATTERNS
# =============================================================================

async def streaming_examples():
    """Streaming execution patterns"""
    
    streaming_agent = Agent(
        name="StreamBot",
        instructions="You are a helpful assistant. Provide detailed responses that can be streamed."
    )
    
    print("\n🌊 STREAMING PATTERNS")
    print("=" * 50)
    
    # 6.1 Basic Streaming
    print("\n6.1 ✅ BASIC STREAMING")
    
    try:
        stream_result = Runner.run_streamed(
            streaming_agent,
            "Tell me a detailed story about space exploration."
        )
        
        print("Streaming response:")
        async for event in stream_result.stream_events():
            # Handle different event types
            event_type = type(event).__name__
            print(f"  Event: {event_type}")
            
            # You can process specific event types
            # and stream content to users in real-time
            
        # Get final result
        final_result = await stream_result
        print(f"Final: {final_result.final_output}")
        
    except Exception as e:
        print(f"❌ Streaming Error: {e}")
    
    # 6.2 Streaming with Progress Tracking
    print("\n6.2 ✅ STREAMING WITH PROGRESS")
    
    @function_tool
    def long_analysis(data: str) -> str:
        """Simulate a long-running analysis task."""
        return f"Analysis complete for: {data[:50]}..."
    
    analysis_agent = Agent(
        name="AnalysisBot",
        instructions="Perform detailed analysis and report progress.",
        tools=[long_analysis]
    )
    
    try:
        stream_result = Runner.run_streamed(
            analysis_agent,
            "Analyze the impact of AI on modern business operations."
        )
        
        print("Analysis streaming:")
        event_count = 0
        
        async for event in stream_result.stream_events():
            event_count += 1
            print(f"  Progress: Event {event_count} - {type(event).__name__}")
            
            # In a real app, you'd send these events to a frontend
            # for real-time progress updates
            
        final_result = await stream_result
        print(f"Analysis Complete: {final_result.final_output}")
        
    except Exception as e:
        print(f"❌ Analysis Streaming Error: {e}")

# =============================================================================
# 🏗️ 7. CUSTOM AGENT RUNNER
# =============================================================================

class CustomAgentRunner:
    """
    Custom AgentRunner implementation for advanced control
    WARNING: This is experimental API and may change
    """
    
    async def run(
        self,
        starting_agent: Agent,
        input: str,
        context: Any = None,
        max_turns: int = 10,
        **kwargs
    ):
        """Custom run implementation with additional logic"""
        
        print(f"🏗️ CustomAgentRunner: Starting run with {starting_agent.name}")
        print(f"   Input: {input[:50]}...")
        print(f"   Max turns: {max_turns}")
        
        # Add custom pre-processing
        if context:
            print(f"   Context keys: {list(context.keys()) if isinstance(context, dict) else 'Non-dict context'}")
        
        # You could add:
        # - Custom authentication
        # - Rate limiting
        # - Logging/monitoring
        # - Input validation
        # - Response filtering
        
        # For now, delegate to default runner
        # In practice, you'd implement your custom logic here
        try:
            from agents.run import DEFAULT_AGENT_RUNNER
            result = await DEFAULT_AGENT_RUNNER.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                **kwargs
            )
            
            print(f"🏗️ CustomAgentRunner: Run completed successfully")
            return result
            
        except Exception as e:
            print(f"🏗️ CustomAgentRunner: Run failed: {e}")
            raise
    
    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper"""
        return asyncio.run(self.run(*args, **kwargs))

def custom_runner_examples():
    """Custom AgentRunner examples"""
    
    print("\n🏗️ CUSTOM AGENT RUNNER")
    print("=" * 50)
    
    # Note: Custom runners are experimental
    # This is how you might use them in the future
    
    custom_runner = CustomAgentRunner()
    
    test_agent = Agent(
        name="CustomRunnerTest",
        instructions="You are a test agent for custom runner demonstration."
    )
    
    try:
        result = custom_runner.run_sync(
            test_agent,
            "Hello from custom runner!",
            context={'custom_runner': True}
        )
        
        print(f"Custom Runner Result: {result.final_output}")
        
    except Exception as e:
        print(f"❌ Custom Runner Error: {e}")

# =============================================================================
# 📊 8. RUNNER PERFORMANCE & MONITORING
# =============================================================================

def performance_monitoring_examples():
    """Performance monitoring and optimization patterns"""
    
    print("\n📊 PERFORMANCE MONITORING")
    print("=" * 50)
    
    # 8.1 Performance Measurement
    print("\n8.1 ✅ PERFORMANCE MEASUREMENT")
    
    import time
    from typing import List, Dict, Any
    
    def measure_agent_performance(
        agent: Agent, 
        test_inputs: List[str],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Measure agent performance across multiple runs"""
        
        results = []
        total_start = time.time()
        
        for i, input_text in enumerate(test_inputs):
            run_start = time.time()
            
            try:
                result = Runner.run_sync(agent, input_text, context=context)
                run_time = time.time() - run_start
                
                results.append({
                    'input': input_text,
                    'output': result.final_output,
                    'success': True,
                    'run_time': run_time,
                    'turns': len(result.run_steps),
                    'tokens_used': getattr(result, 'usage', {}).get('total_tokens', 0) if hasattr(result, 'usage') else 0
                })
                
            except Exception as e:
                run_time = time.time() - run_start
                results.append({
                    'input': input_text,
                    'output': None,
                    'success': False,
                    'error': str(e),
                    'run_time': run_time,
                    'turns': 0,
                    'tokens_used': 0
                })
        
        total_time = time.time() - total_start
        
        # Calculate metrics
        successful_runs = [r for r in results if r['success']]
        success_rate = len(successful_runs) / len(results) if results else 0
        avg_run_time = sum(r['run_time'] for r in successful_runs) / len(successful_runs) if successful_runs else 0
        avg_turns = sum(r['turns'] for r in successful_runs) / len(successful_runs) if successful_runs else 0
        
        return {
            'results': results,
            'metrics': {
                'total_runs': len(results),
                'successful_runs': len(successful_runs),
                'success_rate': success_rate,
                'total_time': total_time,
                'average_run_time': avg_run_time,
                'average_turns': avg_turns
            }
        }
    
    # Test performance
    perf_agent = Agent(
        name="PerfTestBot",
        instructions="You are a helpful assistant for performance testing."
    )
    
    test_inputs = [
        "Hello!",
        "What's 2 + 2?",
        "Tell me about Python.",
        "Help me solve a problem.",
        "What can you do?"
    ]
    
    performance_results = measure_agent_performance(perf_agent, test_inputs)
    
    print("\nPerformance Results:")
    print(f"  Total Runs: {performance_results['metrics']['total_runs']}")
    print(f"  Success Rate: {performance_results['metrics']['success_rate']:.2%}")
    print(f"  Average Run Time: {performance_results['metrics']['average_run_time']:.2f}s")
    print(f"  Average Turns: {performance_results['metrics']['average_turns']:.1f}")
    
    return performance_results

# =============================================================================
# 🎯 9. RUNNER BEST PRACTICES & PATTERNS
# =============================================================================

"""
🎯 RUNNER BEST PRACTICES

1. EXECUTION MODE SELECTION:
   ✅ Use run_sync() for: Simple scripts, testing, synchronous workflows
   ✅ Use run() for: Web applications, concurrent processing, production systems
   ✅ Use run_streamed() for: Real-time UIs, long-running tasks, progress updates

2. CONTEXT MANAGEMENT:
   ✅ Keep context lightweight and serializable
   ✅ Use context for user state, not temporary data
   ✅ Validate context data before passing to agents
   ❌ Don't put large objects or functions in context

3. SESSION USAGE:
   ✅ Use sessions for conversational agents
   ✅ Choose appropriate session backends (SQLite for dev, managed DBs for prod)
   ✅ Clean up old sessions periodically
   ❌ Don't use sessions for stateless operations

4. ERROR HANDLING:
   ✅ Always wrap runner calls in try-catch blocks
   ✅ Implement retry logic with exponential backoff
   ✅ Set appropriate max_turns based on agent complexity
   ❌ Don't ignore errors or let them crash your application

5. PERFORMANCE OPTIMIZATION:
   ✅ Use RunConfig to optimize model settings
   ✅ Monitor token usage and run times
   ✅ Implement caching for repeated queries
   ✅ Use streaming for better user experience

6. PRODUCTION PATTERNS:
   ✅ Implement proper logging and monitoring
   ✅ Use custom runners for advanced control
   ✅ Set up health checks and graceful degradation
   ✅ Configure proper timeouts and resource limits

📊 COMMON ANTI-PATTERNS TO AVOID:

❌ DON'T: Use run_sync() in async contexts (will block event loop)
❌ DON'T: Set max_turns too high (can cause infinite loops)
❌ DON'T: Ignore context validation (can cause runtime errors)
❌ DON'T: Use heavy objects in context (affects performance)
❌ DON'T: Run multiple agents without proper error handling
❌ DON'T: Use sessions for every interaction (unnecessary overhead)

🚀 PRODUCTION CHECKLIST:

☐ Error handling with retries
☐ Proper context validation
☐ Session management strategy
☐ Performance monitoring
☐ Resource limits (max_turns, timeouts)
☐ Logging and observability
☐ Health checks and graceful degradation
☐ Rate limiting and abuse prevention
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & EXAMPLES
# =============================================================================

async def run_all_examples():
    """Run all Runner examples comprehensively"""
    
    print("🏃‍♂️ OPENAI AGENTS SDK - COMPLETE RUNNER DEMONSTRATION")
    print("=" * 70)
    
    # 1. Basic Runner Examples
    basic_runner_examples()
    
    # 2. RunConfig Examples  
    run_config_examples()
    
    # 3. Context Examples
    context_examples()
    
    # 4. Session Management
    session_management_examples()
    
    # 5. Error Handling
    error_handling_examples()
    
    # 6. Async Examples
    await async_runner_examples()
    
    # 7. Streaming Examples
    await streaming_examples()
    
    # 8. Custom Runner
    custom_runner_examples()
    
    # 9. Performance Monitoring
    performance_monitoring_examples()
    
    print("\n✅ All Runner examples completed!")

def run_sync_examples():
    """Run synchronous examples only"""
    
    print("🏃‍♂️ OPENAI AGENTS SDK - RUNNER SYNC EXAMPLES")
    print("=" * 60)
    
    # Run only synchronous examples for immediate testing
    basic_runner_examples()
    run_config_examples()
    context_examples()
    session_management_examples()
    error_handling_examples()
    custom_runner_examples()
    performance_monitoring_examples()
    
    print("\n✅ Sync Runner examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🏃‍♂️ OpenAI Agents SDK - Complete Runner Template")
    print("This template demonstrates all Runner class features and patterns.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_examples())
    
    print("\n✅ Runner template demonstration complete!")
    print("💡 Use this template as reference for all your Runner execution needs.")
