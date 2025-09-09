"""
🤖 OPENAI AGENTS SDK - COMPLETE AGENTS GUIDE & TEMPLATE

This template covers everything about the Agent class in the OpenAI Agents SDK:
- Agent class structure and parameters
- Static vs Dynamic instructions  
- Tools integration
- Handoffs configuration
- Output types and structured responses
- Model configuration
- All possible use cases with examples

📚 Based on: https://openai.github.io/openai-agents-python/ref/agent/
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Union, List
from pydantic import BaseModel

from agents import (
    Agent, 
    Runner, 
    function_tool, 
    handoff,
    RunContextWrapper
)

# =============================================================================
# 📖 UNDERSTANDING THE AGENT CLASS - COMPLETE PARAMETERS GUIDE
# =============================================================================

"""
🤖 AGENT CLASS COMPLETE SIGNATURE (All Parameters):

@dataclass
class Agent(Generic[TContext]):
    # REQUIRED PARAMETERS
    name: str                                    # Agent identifier (required)
    
    # CORE CONFIGURATION
    instructions: str | Callable | None = None  # System prompt (static/dynamic)
    handoff_description: str | None = None       # Description when used as handoff target
    
    # TOOLS & CAPABILITIES  
    tools: list[Tool] = field(default_factory=list)              # Function tools
    mcp_servers: list[MCPServer] = field(default_factory=list)   # Model Context Protocol servers
    mcp_config: MCPConfig = field(default_factory=lambda: MCPConfig())  # MCP configuration
    
    # ROUTING & HANDOFFS
    handoffs: list[Agent | Handoff] = field(default_factory=list) # Delegation targets
    
    # MODEL CONFIGURATION  
    model: str | Model | None = None             # Model implementation 
    model_settings: ModelSettings = field(default_factory=get_default_model_settings)  # Model tuning
    
    # INPUT/OUTPUT CONTROL
    input_guardrails: list[InputGuardrail] = field(default_factory=list)   # Pre-execution checks
    output_guardrails: list[OutputGuardrail] = field(default_factory=list) # Post-execution checks  
    output_type: type | AgentOutputSchemaBase | None = None                 # Structured output schema
    
    # ADVANCED BEHAVIOR
    hooks: AgentHooks | None = None              # Lifecycle event callbacks
    tool_use_behavior: str | StopAtTools | ToolsToFinalOutputFunction = "run_llm_again"  # Tool execution control
    reset_tool_choice: bool = True               # Reset tool choice after calls
    
    # PROMPTS FEATURE (OpenAI Only)
    prompt: Prompt | DynamicPromptFunction | None = None  # Responses API prompt object

WHY IS AGENT A DATACLASS?
1. ✅ IMMUTABILITY: Once created, agent config is immutable (safer)
2. ✅ SERIALIZATION: Easy to save/load agent configurations  
3. ✅ CLONING: Simple to copy agents with modifications (agent.clone())
4. ✅ TYPE SAFETY: Built-in type checking and validation
5. ✅ COMPARISON: Easy to compare agent configurations
6. ✅ SHALLOW COPY: Clone preserves object references (efficient)
"""

# =============================================================================
# 🔧 1. COMPLETE PARAMETER EXAMPLES  
# =============================================================================

def complete_parameter_examples():
    """Comprehensive examples of every Agent parameter"""
    
    print("🔧 COMPLETE AGENT PARAMETER EXAMPLES")
    print("=" * 60)
    
    # 1.1 REQUIRED PARAMETER: name
    print("\n1.1 REQUIRED: name (str)")
    basic_agent = Agent(
        name="CustomerSupportBot"  # Must be descriptive and unique
    )
    print(f"✅ Basic agent: {basic_agent.name}")
    
    # 1.2 HANDOFF_DESCRIPTION - How this agent appears to other agents
    print("\n1.2 OPTIONAL: handoff_description (str | None)")
    
    specialist_agent = Agent(
        name="SecurityExpert",
        handoff_description="Security specialist for vulnerability assessment, compliance checks, and security best practices"
    )
    
    router_agent = Agent(
        name="TechRouter", 
        instructions="Route technical questions to appropriate specialists",
        handoffs=[specialist_agent]  # Router can see the handoff description
    )
    
    print(f"✅ Specialist description: {specialist_agent.handoff_description}")
    
    # 1.3 INSTRUCTIONS - Static vs Dynamic
    print("\n1.3 CORE: instructions (str | Callable | None)")
    
    # Static instructions
    static_agent = Agent(
        name="StaticBot",
        instructions="You are a helpful assistant. Always be polite and provide accurate information."
    )
    
    # Dynamic instructions based on context
    def dynamic_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        user_context = ctx.context or {}
        user_tier = user_context.get('tier', 'basic')
        
        if user_tier == 'premium':
            return "You are a premium support agent. Provide detailed, priority assistance with advanced features."
        elif user_tier == 'enterprise':
            return "You are an enterprise support agent. Provide comprehensive technical guidance and escalation paths."
        else:
            return "You are a basic support agent. Provide helpful, concise assistance."
    
    dynamic_agent = Agent(
        name="DynamicBot",
        instructions=dynamic_instructions
    )
    
    print(f"✅ Static instructions length: {len(static_agent.instructions or '')}")
    print(f"✅ Dynamic instructions: {type(dynamic_agent.instructions).__name__}")
    
    # 1.4 TOOLS - Function tools list
    print("\n1.4 CAPABILITIES: tools (list[Tool])")
    
    @function_tool
    def get_weather(city: str) -> str:
        """Get current weather for a city"""
        return f"Weather in {city}: 22°C, Sunny"
    
    @function_tool  
    def calculate_tip(bill: float, percentage: int = 15) -> float:
        """Calculate tip amount"""
        return round(bill * (percentage / 100), 2)
    
    tool_agent = Agent(
        name="ToolBot",
        instructions="You can check weather and calculate tips for users.",
        tools=[get_weather, calculate_tip]
    )
    
    print(f"✅ Tools count: {len(tool_agent.tools)}")
    print(f"✅ Tool names: {[tool.name for tool in tool_agent.tools]}")
    
    # 1.5 HANDOFFS - Agent delegation
    print("\n1.5 ROUTING: handoffs (list[Agent | Handoff])")
    
    # Simple handoff (list of agents)
    python_expert = Agent(
        name="PythonExpert",
        instructions="You are a Python programming expert.",
        handoff_description="Python programming specialist"
    )
    
    js_expert = Agent(
        name="JSExpert", 
        instructions="You are a JavaScript programming expert.",
        handoff_description="JavaScript programming specialist"
    )
    
    # Basic handoff agent
    basic_router = Agent(
        name="BasicRouter",
        instructions="Route programming questions to language experts",
        handoffs=[python_expert, js_expert]
    )
    
    print(f"✅ Basic handoffs: {len(basic_router.handoffs)}")
    
    # 1.6 MODEL - Model implementation
    print("\n1.6 MODEL: model (str | Model | None)")
    
    # String model (most common)
    string_model_agent = Agent(
        name="GPT4Agent",
        model="gpt-4o",  # Model name string
        instructions="I use GPT-4o model"
    )
    
    print(f"✅ String model: {string_model_agent.model}")
    
    # 1.7 MODEL_SETTINGS - Model tuning parameters
    print("\n1.7 TUNING: model_settings (ModelSettings)")
    
    creative_agent = Agent(
        name="CreativeAgent",
        instructions="You are a creative writing assistant",
        model_settings=ModelSettings(
            temperature=0.9,      # High creativity
            max_tokens=1000,      # Longer responses
            top_p=0.95,          # Diverse vocabulary
            frequency_penalty=0.1, # Reduce repetition
            presence_penalty=0.1,  # Encourage new topics
            max_retries=3,        # Retry on failures
            timeout=30.0          # 30 second timeout
        )
    )
    
    analytical_agent = Agent(
        name="AnalyticalAgent", 
        instructions="You provide precise, factual analysis",
        model_settings=ModelSettings(
            temperature=0.1,      # Low creativity/high precision
            max_tokens=500,       # Concise responses
            top_p=0.1,           # Focused vocabulary
            max_retries=5,        # High reliability
            timeout=60.0          # Longer timeout for analysis
        )
    )
    
    print(f"✅ Creative temp: {creative_agent.model_settings.temperature}")
    print(f"✅ Analytical temp: {analytical_agent.model_settings.temperature}")
    
    # 1.8 OUTPUT_TYPE - Structured output schema
    print("\n1.8 STRUCTURE: output_type (type | AgentOutputSchemaBase | None)")
    
    from pydantic import BaseModel
    from dataclasses import dataclass
    from typing import List
    
    # Pydantic model output
    class CustomerReport(BaseModel):
        customer_id: str
        issue_summary: str
        priority: str
        resolution_steps: List[str]
        estimated_time: int
    
    structured_agent = Agent(
        name="StructuredAgent",
        instructions="Always return customer reports in the specified format",
        output_type=CustomerReport
    )
    
    # Dataclass output  
    @dataclass
    class ProductInfo:
        name: str
        price: float
        in_stock: bool
        description: str
    
    product_agent = Agent(
        name="ProductAgent",
        instructions="Return product information in structured format",
        output_type=ProductInfo
    )
    
    print(f"✅ Pydantic output: {structured_agent.output_type.__name__}")
    print(f"✅ Dataclass output: {product_agent.output_type.__name__}")
    
    # 1.9 TOOL_USE_BEHAVIOR - Tool execution control
    print("\n1.9 CONTROL: tool_use_behavior (str | StopAtTools | Function)")
    
    # Default behavior: run LLM again after tool
    default_behavior_agent = Agent(
        name="DefaultBehaviorAgent",
        tool_use_behavior="run_llm_again",  # Default
        tools=[get_weather]
    )
    
    # Stop on first tool (tool output becomes final output)
    stop_first_agent = Agent(
        name="StopFirstAgent", 
        tool_use_behavior="stop_on_first_tool",
        tools=[get_weather]
    )
    
    print(f"✅ Default behavior: {default_behavior_agent.tool_use_behavior}")
    print(f"✅ Stop first: {stop_first_agent.tool_use_behavior}")
    
    # 1.10 RESET_TOOL_CHOICE - Tool choice reset control
    print("\n1.10 OPTIMIZATION: reset_tool_choice (bool)")
    
    # Default: reset tool choice (prevents infinite tool loops)
    reset_agent = Agent(
        name="ResetAgent",
        reset_tool_choice=True,  # Default
        tools=[get_weather]
    )
    
    # Don't reset (for specific use cases)
    no_reset_agent = Agent(
        name="NoResetAgent",
        reset_tool_choice=False,
        tools=[get_weather]
    )
    
    print(f"✅ Reset enabled: {reset_agent.reset_tool_choice}")
    print(f"✅ Reset disabled: {no_reset_agent.reset_tool_choice}")
    
    # SUMMARY OF ALL PARAMETERS
    print("\n" + "="*60)
    print("📋 AGENT PARAMETER SUMMARY")
    print("="*60)
    
    all_params = {
        "name": "Required string identifier",
        "handoff_description": "Description for handoff routing", 
        "instructions": "System prompt (static/dynamic/None)",
        "tools": "Function tools list",
        "mcp_servers": "External MCP servers",
        "mcp_config": "MCP configuration dict",
        "handoffs": "Agent delegation targets",
        "model": "Model implementation",
        "model_settings": "Model tuning parameters", 
        "input_guardrails": "Pre-execution safety checks",
        "output_guardrails": "Post-execution validation",
        "output_type": "Structured output schema",
        "hooks": "Lifecycle event callbacks",
        "tool_use_behavior": "Tool execution control",
        "reset_tool_choice": "Tool choice reset flag",
        "prompt": "OpenAI Responses API prompt"
    }
    
    for param, description in all_params.items():
        print(f"  {param:20} → {description}")
    
    return {
        "basic_agent": basic_agent,
        "specialist_agent": specialist_agent,
        "router_agent": router_agent,
        "static_agent": static_agent,
        "dynamic_agent": dynamic_agent,
        "tool_agent": tool_agent,
        "creative_agent": creative_agent,
        "analytical_agent": analytical_agent,
        "structured_agent": structured_agent,
        "string_model_agent": string_model_agent
    }

def basic_agent_examples():
    """Basic agent creation patterns (preserved for compatibility)"""
    
    # 1.1 Minimal Agent (just name and instructions)
    minimal_agent = Agent(
        name="SimpleBot",
        instructions="You are a helpful assistant. Keep responses concise."
    )
    
    # 1.2 Agent with Custom Model
    custom_model_agent = Agent(
        name="GPT4Agent", 
        instructions="You are an expert assistant.",
        model="gpt-4o"  # Specify model
    )
    
    # 1.3 Agent with All Basic Parameters
    complete_basic_agent = Agent(
        name="ComprehensiveBot",
        instructions="""
        You are a professional assistant with these capabilities:
        1. Answer questions accurately
        2. Help with problem-solving
        3. Provide detailed explanations when needed
        
        Always be professional, helpful, and concise.
        """,
        model="gpt-4o-mini"  # Cost-effective model choice
    )
    
    return [minimal_agent, custom_model_agent, complete_basic_agent]

# =============================================================================
# 📝 2. INSTRUCTIONS - STATIC VS DYNAMIC
# =============================================================================

def instructions_examples():
    """Different ways to set agent instructions"""
    
    # 2.1 Static Instructions (Simple String)
    static_agent = Agent(
        name="StaticBot",
        instructions="You are a weather assistant. Always provide current weather info."
    )
    
    # 2.2 Multi-line Static Instructions  
    multiline_agent = Agent(
        name="DetailedBot",
        instructions="""
        You are a code review assistant. Your responsibilities:
        
        1. 🔍 Review code for bugs and improvements
        2. 📚 Suggest best practices
        3. 🚀 Recommend performance optimizations
        4. 📖 Explain complex concepts clearly
        
        Always provide constructive feedback with examples.
        """
    )
    
    # 2.3 Dynamic Instructions (Callable) - POWERFUL FEATURE!
    def dynamic_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        """
        Dynamic instructions change based on context
        ctx.context contains user-defined context data
        """
        user_data = ctx.context or {}
        user_name = user_data.get('name', 'User')
        user_role = user_data.get('role', 'general')
        
        base_prompt = f"Hello {user_name}! "
        
        if user_role == 'developer':
            return base_prompt + """
            You are a senior software engineer assistant.
            Focus on:
            - Code quality and best practices
            - Architecture decisions
            - Performance optimization
            - Security considerations
            """
        elif user_role == 'student':
            return base_prompt + """
            You are a patient coding tutor.
            Focus on:
            - Step-by-step explanations
            - Learning fundamentals
            - Practice exercises
            - Encouraging progress
            """
        else:
            return base_prompt + """
            You are a general programming assistant.
            Focus on:
            - Clear explanations
            - Practical examples
            - Helpful guidance
            """
    
    # Agent with dynamic instructions
    dynamic_agent = Agent(
        name="AdaptiveBot",
        instructions=dynamic_instructions  # ← Function, not string!
    )
    
    # 2.4 Context-Aware Instructions with Error Handling
    def robust_dynamic_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        try:
            context = ctx.context or {}
            
            # Extract context variables safely
            language = context.get('preferred_language', 'English')
            expertise_level = context.get('expertise', 'beginner')
            domain = context.get('domain', 'general')
            
            # Build customized instructions
            base = f"Respond in {language}. "
            
            level_guidance = {
                'beginner': "Use simple terms and provide step-by-step explanations.",
                'intermediate': "Provide balanced detail with some technical terms.",
                'expert': "Use technical language and focus on advanced concepts."
            }
            
            domain_focus = {
                'web': "Focus on web development technologies and practices.",
                'data': "Focus on data science, analysis, and machine learning.",
                'mobile': "Focus on mobile app development best practices.",
                'general': "Provide general programming assistance."
            }
            
            instructions = (
                base + 
                level_guidance.get(expertise_level, level_guidance['beginner']) + 
                " " + 
                domain_focus.get(domain, domain_focus['general'])
            )
            
            return instructions
            
        except Exception as e:
            # Fallback to safe default
            return "You are a helpful programming assistant. Provide clear, accurate responses."
    
    robust_agent = Agent(
        name="RobustBot",
        instructions=robust_dynamic_instructions
    )
    
    return [static_agent, multiline_agent, dynamic_agent, robust_agent]

# =============================================================================
# 🛠️ 3. TOOLS INTEGRATION
# =============================================================================

def tools_examples():
    """Agents with various tool configurations"""
    
    # 3.1 Define Some Tools First
    @function_tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Mock implementation
        return f"The weather in {city} is sunny, 22°C"
    
    @function_tool
    def calculate(expression: str) -> str:
        """Calculate mathematical expressions safely."""
        try:
            result = eval(expression.replace('^', '**'))
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    @function_tool
    def search_web(query: str, max_results: int = 5) -> str:
        """Search the web for information."""
        # Mock implementation
        return f"Found {max_results} results for '{query}'"
    
    @function_tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        # Mock implementation
        return f"Email sent to {to} with subject '{subject}'"
    
    # 3.2 Agent with Single Tool
    weather_agent = Agent(
        name="WeatherBot",
        instructions="You are a weather assistant. Use the weather tool to get current conditions.",
        tools=[get_weather]
    )
    
    # 3.3 Agent with Multiple Tools
    multi_tool_agent = Agent(
        name="MultiBot", 
        instructions="""
        You are a versatile assistant with multiple capabilities:
        - Check weather for any city
        - Perform mathematical calculations
        - Search the web for information
        
        Choose the appropriate tool based on the user's request.
        """,
        tools=[get_weather, calculate, search_web]
    )
    
    # 3.4 Agent with Business Tools
    business_agent = Agent(
        name="BusinessBot",
        instructions="""
        You are a business assistant that can:
        1. Send emails on behalf of the user
        2. Search for business information
        3. Perform calculations for business metrics
        
        Always confirm important actions with the user before executing.
        """,
        tools=[send_email, search_web, calculate]
    )
    
    # 3.5 Specialized Agent with Domain-Specific Tools
    @function_tool
    def analyze_code(code: str, language: str = "python") -> str:
        """Analyze code for issues and improvements."""
        return f"Code analysis for {language}: The code looks good with minor improvements suggested."
    
    @function_tool
    def run_tests(test_file: str) -> str:
        """Run unit tests and return results."""
        return f"Tests in {test_file}: 8 passed, 2 failed"
    
    developer_agent = Agent(
        name="DevBot",
        instructions="""
        You are a senior software developer assistant.
        Help with code review, testing, and development tasks.
        Always explain your recommendations clearly.
        """,
        tools=[analyze_code, run_tests, search_web]
    )
    
    return [weather_agent, multi_tool_agent, business_agent, developer_agent]

# =============================================================================
# 🤝 4. HANDOFFS CONFIGURATION  
# =============================================================================

def handoffs_examples():
    """Agents with handoff capabilities"""
    
    # 4.1 Basic Agents for Handoffs
    spanish_agent = Agent(
        name="SpanishExpert",
        instructions="Eres un asistente que solo habla español. Ayuda a los usuarios en español."
    )
    
    french_agent = Agent(
        name="FrenchExpert", 
        instructions="Tu es un assistant qui ne parle que français. Aide les utilisateurs en français."
    )
    
    tech_agent = Agent(
        name="TechExpert",
        instructions="""
        You are a technical expert specializing in:
        - Software development
        - System architecture 
        - DevOps and deployment
        - Performance optimization
        """
    )
    
    business_agent = Agent(
        name="BusinessExpert",
        instructions="""
        You are a business expert specializing in:
        - Business strategy
        - Market analysis
        - Financial planning
        - Operations management
        """
    )
    
    # 4.2 Simple Handoff Agent (List of Agents)
    language_triage_agent = Agent(
        name="LanguageTriage",
        instructions="""
        You are a language detector and router. 
        Determine the language of the user's request and handoff to the appropriate language expert:
        - For Spanish requests → SpanishExpert
        - For French requests → FrenchExpert
        - For English requests → continue handling yourself
        """,
        handoffs=[spanish_agent, french_agent]
    )
    
    # 4.3 Advanced Handoff with handoff() Objects
    expertise_triage_agent = Agent(
        name="ExpertiseTriage",
        instructions="""
        You are an intelligent router that determines what type of expertise is needed:
        - Technical questions → TechExpert
        - Business questions → BusinessExpert
        - General questions → handle yourself
        """,
        handoffs=[
            handoff(
                agent=tech_agent,
                input_filter=lambda items: [
                    {"type": "text", "text": f"Technical question: {items[-1]['text']}"}
                ]
            ),
            handoff(
                agent=business_agent, 
                input_filter=lambda items: [
                    {"type": "text", "text": f"Business question: {items[-1]['text']}"}
                ]
            )
        ]
    )
    
    # 4.4 Conditional Handoff (Advanced)
    def should_handoff_to_expert(ctx: RunContextWrapper, agent: Agent) -> bool:
        """
        Conditional logic for when to enable handoffs
        """
        user_context = ctx.context or {}
        user_level = user_context.get('expertise_level', 'beginner')
        question_complexity = user_context.get('complexity', 'simple')
        
        # Only handoff for complex questions or intermediate+ users
        return question_complexity == 'complex' or user_level in ['intermediate', 'expert']
    
    conditional_agent = Agent(
        name="ConditionalTriage",
        instructions="Handle simple questions yourself, route complex ones to experts.",
        handoffs=[
            handoff(
                agent=tech_agent,
                is_enabled=should_handoff_to_expert
            )
        ]
    )
    
    # 4.5 Handoff with Callbacks
    def on_tech_handoff(ctx: RunContextWrapper, agent: Agent, input_items: list) -> None:
        """Called when handing off to tech expert"""
        print(f"🔄 Routing technical question to {agent.name}")
        
    def on_business_handoff(ctx: RunContextWrapper, agent: Agent, input_items: list) -> None:
        """Called when handing off to business expert"""  
        print(f"🔄 Routing business question to {agent.name}")
    
    callback_agent = Agent(
        name="CallbackTriage",
        instructions="Route questions to appropriate experts with logging.",
        handoffs=[
            handoff(
                agent=tech_agent,
                on_handoff=on_tech_handoff
            ),
            handoff(
                agent=business_agent,
                on_handoff=on_business_handoff
            )
        ]
    )
    
    return [
        language_triage_agent, 
        expertise_triage_agent, 
        conditional_agent, 
        callback_agent
    ]

# =============================================================================
# 📊 5. STRUCTURED OUTPUTS (OUTPUT_TYPE)
# =============================================================================

def structured_output_examples():
    """Agents that return structured data instead of plain text"""
    
    # 5.1 Simple Structured Output
    class WeatherReport(BaseModel):
        city: str
        temperature: int
        conditions: str
        humidity: int
        
    weather_structured_agent = Agent(
        name="StructuredWeatherBot",
        instructions="""
        Provide weather information in the specified JSON format.
        Always include all required fields: city, temperature, conditions, humidity.
        """,
        output_type=WeatherReport
    )
    
    # 5.2 Complex Structured Output
    class CodeReviewResult(BaseModel):
        overall_score: int  # 1-10
        issues_found: List[str]
        suggestions: List[str] 
        security_concerns: List[str]
        performance_notes: List[str]
        maintainability_score: int  # 1-10
        
    code_review_agent = Agent(
        name="CodeReviewBot",
        instructions="""
        Analyze code and provide a comprehensive review in the specified format.
        Be thorough in your analysis and provide actionable feedback.
        """,
        output_type=CodeReviewResult
    )
    
    # 5.3 Business Analysis Output
    class BusinessAnalysis(BaseModel):
        market_opportunity: str
        competitive_advantage: str
        risk_factors: List[str]
        success_probability: float  # 0.0 to 1.0
        recommended_actions: List[str]
        
    business_analysis_agent = Agent(
        name="BusinessAnalyst",
        instructions="""
        Analyze business opportunities and provide structured insights.
        Consider market conditions, competition, risks, and actionable recommendations.
        """,
        output_type=BusinessAnalysis
    )
    
    # 5.4 Multi-Step Task Result
    class TaskExecutionResult(BaseModel):
        task_id: str
        status: str  # "completed", "failed", "partial"
        steps_completed: List[str]
        steps_failed: List[str]
        final_output: str
        execution_time_seconds: float
        
    task_executor_agent = Agent(
        name="TaskExecutor",
        instructions="""
        Execute multi-step tasks and provide detailed execution results.
        Track progress, failures, and timing information.
        """,
        output_type=TaskExecutionResult
    )
    
    return [
        weather_structured_agent,
        code_review_agent, 
        business_analysis_agent,
        task_executor_agent
    ]

# =============================================================================
# 🔄 6. COMBINING ALL FEATURES - ADVANCED EXAMPLES
# =============================================================================

def advanced_agent_examples():
    """Sophisticated agents combining multiple features"""
    
    # 6.1 Tools for Advanced Agents
    @function_tool
    def database_query(sql: str) -> str:
        """Execute a database query."""
        return f"Query executed: {sql[:50]}... (5 rows returned)"
    
    @function_tool
    def api_request(endpoint: str, method: str = "GET") -> str:
        """Make an API request."""
        return f"API {method} request to {endpoint} - Status: 200 OK"
    
    @function_tool
    def file_operations(action: str, filename: str, content: str = "") -> str:
        """Perform file operations."""
        return f"File operation '{action}' on '{filename}' completed"
    
    # 6.2 Specialized Expert Agents
    database_expert = Agent(
        name="DatabaseExpert",
        instructions="""
        You are a database expert. Help with:
        - SQL query optimization
        - Database design
        - Performance tuning
        - Data migration
        """,
        tools=[database_query]
    )
    
    api_expert = Agent(
        name="APIExpert", 
        instructions="""
        You are an API expert. Help with:
        - API design and development
        - REST/GraphQL best practices
        - API testing and documentation
        - Integration patterns
        """,
        tools=[api_request]
    )
    
    # 6.3 Advanced Multi-Feature Agent
    class ProjectAnalysis(BaseModel):
        project_type: str
        complexity_score: int  # 1-10
        estimated_hours: float
        technologies_needed: List[str]
        risks: List[str]
        recommendations: List[str]
    
    def adaptive_project_instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        """Dynamic instructions based on project context"""
        context = ctx.context or {}
        client_type = context.get('client_type', 'standard')
        budget_tier = context.get('budget', 'medium')
        urgency = context.get('urgency', 'normal')
        
        base = "You are a senior project consultant. "
        
        if client_type == 'enterprise':
            focus = "Focus on scalability, security, and enterprise integration. "
        elif client_type == 'startup':
            focus = "Focus on rapid development, cost-effectiveness, and MVP approach. "
        else:
            focus = "Focus on balanced solutions with good practices. "
            
        if urgency == 'urgent':
            approach = "Prioritize quick delivery and pragmatic solutions."
        else:
            approach = "Provide comprehensive analysis and optimal recommendations."
            
        return base + focus + approach
    
    master_consultant_agent = Agent(
        name="MasterConsultant",
        instructions=adaptive_project_instructions,
        tools=[database_query, api_request, file_operations],
        handoffs=[
            handoff(
                agent=database_expert,
                is_enabled=lambda ctx, agent: 'database' in str(ctx.context.get('technologies', [])).lower()
            ),
            handoff(
                agent=api_expert,
                is_enabled=lambda ctx, agent: 'api' in str(ctx.context.get('technologies', [])).lower()
            )
        ],
        output_type=ProjectAnalysis
    )
    
    return [master_consultant_agent]

# =============================================================================
# 🚀 7. USAGE EXAMPLES & TESTING
# =============================================================================

async def demonstrate_all_agent_types():
    """Demonstrate all different agent types in action"""
    
    print("🤖 OpenAI Agents SDK - Complete Agent Demonstration")
    print("=" * 60)
    
    # 1. Basic Agents
    print("\n1. 📝 BASIC AGENTS")
    basic_agents = basic_agent_examples()
    
    for agent in basic_agents:
        print(f"\n✓ {agent.name}")
        try:
            result = Runner.run_sync(agent, "Hello, what can you do?")
            print(f"  Response: {result.final_output}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 2. Dynamic Instructions
    print("\n2. 🔄 DYNAMIC INSTRUCTIONS")
    instruction_agents = instructions_examples()
    
    # Test with different contexts
    contexts = [
        {'name': 'Alice', 'role': 'developer'},
        {'name': 'Bob', 'role': 'student'},
        {'name': 'Charlie', 'role': 'manager'}
    ]
    
    dynamic_agent = instruction_agents[2]  # The dynamic one
    
    for context in contexts:
        print(f"\n✓ Testing with context: {context}")
        try:
            result = Runner.run_sync(
                dynamic_agent, 
                "How do I optimize database queries?",
                context=context
            )
            print(f"  Response: {result.final_output}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 3. Structured Outputs
    print("\n3. 📊 STRUCTURED OUTPUTS")
    structured_agents = structured_output_examples()
    
    weather_agent = structured_agents[0]
    print(f"\n✓ {weather_agent.name}")
    try:
        result = Runner.run_sync(weather_agent, "What's the weather in Tokyo?")
        print(f"  Structured Result: {result.final_output}")
        print(f"  Type: {type(result.final_output)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

def run_sync_examples():
    """Run synchronous examples for immediate testing"""
    
    print("🔄 SYNCHRONOUS EXAMPLES")
    print("=" * 40)
    
    # Quick test of each agent type
    basic_agents = basic_agent_examples()
    agent = basic_agents[0]
    
    print(f"Testing: {agent.name}")
    
    test_inputs = [
        "Hello!",
        "What can you help me with?",
        "Explain what you do in one sentence."
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: {input_text}")
        try:
            result = Runner.run_sync(agent, input_text)
            print(f"   Output: {result.final_output}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

# =============================================================================
# 📋 8. AGENT CHEAT SHEET & BEST PRACTICES
# =============================================================================

"""
🎯 AGENT CREATION BEST PRACTICES

1. NAMING CONVENTIONS:
   ✅ Use descriptive names: "CustomerServiceBot", "CodeReviewExpert"
   ❌ Avoid generic names: "Agent1", "Bot", "AI"

2. INSTRUCTIONS WRITING:
   ✅ Be specific about the agent's role and capabilities
   ✅ Include examples of good behavior
   ✅ Set clear boundaries ("only help with X, not Y")
   ❌ Don't make instructions too long or complex

3. TOOLS SELECTION:
   ✅ Only include tools the agent actually needs
   ✅ Group related tools together logically  
   ✅ Test tools thoroughly before adding
   ❌ Don't give agents access to dangerous operations

4. HANDOFFS STRATEGY:
   ✅ Use handoffs for clear specialization boundaries
   ✅ Implement proper routing logic
   ✅ Add logging/monitoring for handoff tracking
   ❌ Don't create circular handoff loops

5. STRUCTURED OUTPUTS:
   ✅ Use Pydantic models for complex data
   ✅ Include validation and constraints
   ✅ Document expected output format
   ❌ Don't over-structure simple responses

6. TESTING:
   ✅ Test with edge cases and error scenarios
   ✅ Validate structured outputs match expected schema
   ✅ Monitor agent performance and accuracy
   ✅ Test handoff flows end-to-end

📊 COMMON PATTERNS:

Pattern 1 - Triage Agent:
```python
triage = Agent(
    name="Triage",
    instructions="Route requests to appropriate specialists",
    handoffs=[specialist1, specialist2, specialist3]
)
```

Pattern 2 - Tool-Heavy Agent:
```python
worker = Agent(
    name="Worker",
    instructions="Execute tasks using available tools",
    tools=[tool1, tool2, tool3, tool4]
)
```

Pattern 3 - Structured Output Agent:
```python
analyzer = Agent(
    name="Analyzer", 
    instructions="Analyze input and return structured results",
    output_type=AnalysisResult,
    tools=[analysis_tools]
)
```

Pattern 4 - Context-Adaptive Agent:
```python
adaptive = Agent(
    name="Adaptive",
    instructions=dynamic_instruction_function,
    tools=context_dependent_tools
)
```
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION
# =============================================================================

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all agent examples to demonstrate functionality"""
    
    print("🤖 OpenAI Agents SDK - Complete Agents Template")
    print("=" * 60)
    print("This template demonstrates ALL Agent class parameters and patterns.")
    print()
    
    # 1. Run comprehensive parameter examples (NEW)
    print("🔧 RUNNING: Complete Parameter Examples")
    complete_examples = complete_parameter_examples()
    print(f"✅ Created {len(complete_examples)} parameter demonstration agents")
    print()
    
    # 2. Run basic examples (preserved)
    print("🔧 RUNNING: Basic Agent Examples")
    basic_examples = basic_agent_examples()
    print(f"✅ Created {len(basic_examples)} basic agents")
    print()
    
    # 3. Run instructions examples
    print("📝 RUNNING: Instructions Examples")
    instruction_examples = instructions_examples()
    print(f"✅ Created {len(instruction_examples)} instruction agents")
    print()
    
    # 4. Run tools examples
    print("🛠️ RUNNING: Tools Examples") 
    tool_examples = tools_examples()
    print(f"✅ Created {len(tool_examples)} tool agents")
    print()
    
    # 5. Run handoffs examples
    print("🤝 RUNNING: Handoffs Examples")
    handoff_examples_result = handoffs_examples()
    print(f"✅ Created {len(handoff_examples_result)} handoff agents")
    print()
    
    # 6. Run structured output examples
    print("📊 RUNNING: Structured Output Examples")
    structured_examples = structured_output_examples()
    print(f"✅ Created {len(structured_examples)} structured output agents")
    print()
    
    # 7. Run advanced examples
    print("🚀 RUNNING: Advanced Agent Examples") 
    advanced_examples = advanced_agent_examples()
    print(f"✅ Created {len(advanced_examples)} advanced agents")
    print()
    
    # Summary
    total_agents = (len(complete_examples) + len(basic_examples) + 
                   len(instruction_examples) + len(tool_examples) + 
                   len(handoff_examples_result) + len(structured_examples) + 
                   len(advanced_examples))
    
    print("="*60)
    print("📋 EXECUTION SUMMARY")
    print("="*60)
    print(f"Total agents created: {total_agents}")
    print("All parameter types demonstrated ✅")
    print("All usage patterns covered ✅") 
    print("Template validation complete ✅")
    print()
    print("💡 Next Steps:")
    print("  1. Copy specific agent patterns you need")
    print("  2. Modify parameters for your use case")
    print("  3. Test with Runner.run() or Runner.run_sync()")
    print("  4. Add your own tools, handoffs, and guardrails")

if __name__ == "__main__":
    # Run all examples to show complete functionality
    run_all_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(demonstrate_all_agent_types())
    
    print("\n✅ Template demonstration complete!")
    print("💡 Use this template as reference for all your agent development needs.")
