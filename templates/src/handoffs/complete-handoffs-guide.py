"""
🤝 OPENAI AGENTS SDK - COMPLETE HANDOFFS GUIDE & TEMPLATE

This template covers everything about Handoffs in the OpenAI Agents SDK:
- Basic handoffs between agents
- Advanced handoff() objects with customization
- Conditional handoffs with is_enabled
- Input filtering and transformation
- Handoff callbacks and lifecycle hooks
- Multi-level handoff chains
- Best practices and patterns
- When to use handoffs vs agents-as-tools

📚 Based on: https://openai.github.io/openai-agents-python/ref/handoffs/
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
from datetime import datetime

from agents import (
    Agent, 
    Runner, 
    function_tool, 
    handoff,
    RunContextWrapper
)

# =============================================================================
# 📖 UNDERSTANDING HANDOFFS IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT ARE HANDOFFS?

Handoffs are a specialized tool that transfers control from one agent to another.
They enable:
1. ✅ SPECIALIZATION: Route requests to domain experts
2. ✅ WORKFLOW ORCHESTRATION: Create multi-stage processes
3. ✅ LOAD DISTRIBUTION: Balance work across agents
4. ✅ CONTEXT PRESERVATION: Maintain conversation flow
5. ✅ INTELLIGENT ROUTING: Dynamic agent selection

KEY CONCEPTS:

1. HANDOFF vs AGENT-AS-TOOL:
   - Handoff: Transfers control completely, new agent takes over
   - Agent-as-tool: Main agent calls another agent and gets result back

2. HANDOFF TYPES:
   - Simple: agent.handoffs = [other_agent]
   - Advanced: handoff(agent, input_filter=..., is_enabled=..., on_handoff=...)

3. HANDOFF FLOW:
   Agent A → Decides to handoff → Agent B takes control → Agent B responds

4. THE HANDOFF TOOL:
   - Agents see handoffs as tools named "transfer_to_<agent_name>"
   - LLM decides when to use the handoff tool
   - Control transfers to the target agent
"""

# =============================================================================
# 🔧 1. BASIC HANDOFFS PATTERNS
# =============================================================================

def basic_handoffs_examples():
    """Basic handoff patterns and usage"""
    
    print("🔧 BASIC HANDOFFS PATTERNS")
    print("=" * 50)
    
    # 1.1 Create Specialized Agents
    python_expert = Agent(
        name="PythonExpert",
        instructions="""
        You are a Python programming expert. You specialize in:
        - Python syntax and best practices
        - Library recommendations
        - Code optimization
        - Debugging assistance
        
        Provide detailed, technical responses for Python-related questions.
        """,
        tools=[]
    )
    
    javascript_expert = Agent(
        name="JavaScriptExpert", 
        instructions="""
        You are a JavaScript programming expert. You specialize in:
        - Modern JavaScript (ES6+)
        - Frontend and backend JavaScript
        - Framework recommendations (React, Vue, Node.js)
        - Performance optimization
        
        Provide detailed, technical responses for JavaScript-related questions.
        """,
        tools=[]
    )
    
    database_expert = Agent(
        name="DatabaseExpert",
        instructions="""
        You are a database expert specializing in:
        - SQL query optimization
        - Database design and normalization
        - Performance tuning
        - Different database systems (MySQL, PostgreSQL, MongoDB)
        
        Provide detailed technical guidance for database-related questions.
        """,
        tools=[]
    )
    
    # 1.2 Simple Triage Agent with List of Handoffs
    simple_triage = Agent(
        name="SimpleTriage",
        instructions="""
        You are a technical support triage agent. Your job is to:
        1. Understand the user's question
        2. Determine which expert can best help them
        3. Route them to the appropriate specialist:
           - Python questions → PythonExpert
           - JavaScript questions → JavaScriptExpert  
           - Database questions → DatabaseExpert
        
        For general questions, provide a brief answer yourself.
        """,
        handoffs=[python_expert, javascript_expert, database_expert]
    )
    
    # 1.3 Language-Based Routing
    spanish_agent = Agent(
        name="SpanishAssistant",
        instructions="""
        Eres un asistente que solo habla español. 
        Ayudas a los usuarios con sus preguntas en español.
        Proporciona respuestas detalladas y útiles.
        """
    )
    
    french_agent = Agent(
        name="FrenchAssistant",
        instructions="""
        Tu es un assistant qui ne parle que français.
        Tu aides les utilisateurs avec leurs questions en français.
        Fournis des réponses détaillées et utiles.
        """
    )
    
    language_router = Agent(
        name="LanguageRouter",
        instructions="""
        You detect the language of user input and route to appropriate language specialists:
        - Spanish text → SpanishAssistant
        - French text → FrenchAssistant  
        - English text → handle yourself
        
        If you can't determine the language, ask the user to clarify.
        """,
        handoffs=[spanish_agent, french_agent]
    )
    
    # 1.4 Business Process Handoffs
    sales_agent = Agent(
        name="SalesAgent",
        instructions="""
        You are a sales representative. You help with:
        - Product information and pricing
        - Sales inquiries and quotes
        - Lead qualification
        - Initial customer onboarding
        
        Be friendly, helpful, and focus on understanding customer needs.
        """
    )
    
    support_agent = Agent(
        name="SupportAgent", 
        instructions="""
        You are a technical support specialist. You handle:
        - Bug reports and troubleshooting
        - Account issues and access problems
        - Feature questions and how-to guidance
        - Escalation to development team when needed
        
        Be patient, thorough, and solution-focused.
        """
    )
    
    billing_agent = Agent(
        name="BillingAgent",
        instructions="""
        You are a billing specialist. You assist with:
        - Invoice questions and payment issues
        - Subscription changes and upgrades
        - Refund requests and billing disputes
        - Account billing history
        
        Be accurate, empathetic, and follow company policies.
        """
    )
    
    customer_service_router = Agent(
        name="CustomerServiceRouter",
        instructions="""
        You are the first point of contact for customer inquiries. Route customers to:
        - Sales questions → SalesAgent
        - Technical problems → SupportAgent
        - Billing/payment issues → BillingAgent
        
        For simple questions, provide quick answers yourself before routing.
        """,
        handoffs=[sales_agent, support_agent, billing_agent]
    )
    
    # Test basic handoffs
    test_scenarios = [
        # Technical routing
        (simple_triage, "I'm having trouble with a Python list comprehension. Can you help?"),
        (simple_triage, "How do I make an API call in JavaScript?"),
        (simple_triage, "My SQL query is running slowly. How can I optimize it?"),
        
        # Language routing  
        (language_router, "Hola, ¿cómo puedo mejorar mi código Python?"),
        (language_router, "Bonjour, j'ai une question sur JavaScript"),
        
        # Business routing
        (customer_service_router, "I want to upgrade my subscription plan"),
        (customer_service_router, "I'm getting an error when trying to log in"),
        (customer_service_router, "My last invoice seems incorrect")
    ]
    
    print("\n1. Testing Basic Handoffs:")
    for i, (agent, query) in enumerate(test_scenarios, 1):
        try:
            result = Runner.run_sync(agent, query)
            print(f"{i}. Agent: {agent.name}")
            print(f"   Query: {query}")
            print(f"   Final Agent: {result.last_agent.name}")
            print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return {
        "simple_triage": simple_triage,
        "language_router": language_router,
        "customer_service_router": customer_service_router
    }

# =============================================================================
# ⚙️ 2. ADVANCED HANDOFF OBJECTS  
# =============================================================================

def advanced_handoff_examples():
    """Advanced handoff() objects with customization"""
    
    print("\n⚙️ ADVANCED HANDOFF OBJECTS")
    print("=" * 50)
    
    # 2.1 Create Expert Agents
    security_expert = Agent(
        name="SecurityExpert", 
        instructions="""
        You are a cybersecurity expert specializing in:
        - Security vulnerabilities and fixes
        - Authentication and authorization
        - Encryption and data protection
        - Security best practices and compliance
        
        Always prioritize security in your recommendations.
        """
    )
    
    performance_expert = Agent(
        name="PerformanceExpert",
        instructions="""
        You are a performance optimization expert specializing in:
        - Code performance analysis
        - Database query optimization
        - System architecture improvements
        - Scalability solutions
        
        Focus on measurable performance improvements.
        """
    )
    
    architecture_expert = Agent(
        name="ArchitectureExpert",
        instructions="""
        You are a software architecture expert specializing in:
        - System design patterns
        - Microservices architecture
        - Cloud architecture decisions
        - Technology stack recommendations
        
        Provide strategic, long-term architectural guidance.
        """
    )
    
    # 2.2 Handoffs with Custom Tool Names and Descriptions
    expert_router = Agent(
        name="ExpertRouter",
        instructions="""
        You route complex technical questions to appropriate experts.
        Analyze the question to determine if it needs:
        - Security expertise
        - Performance optimization
        - Architecture guidance
        
        For simple questions, answer them yourself.
        """,
        handoffs=[
            handoff(
                agent=security_expert,
                tool_name="consult_security_expert",
                description="Route security-related questions to the security expert"
            ),
            handoff(
                agent=performance_expert,
                tool_name="consult_performance_expert", 
                description="Route performance optimization questions to the performance expert"
            ),
            handoff(
                agent=architecture_expert,
                tool_name="consult_architecture_expert",
                description="Route system architecture questions to the architecture expert"
            )
        ]
    )
    
    # 2.3 Handoffs with Input Filters
    def add_context_filter(items):
        """Add context to the handoff input"""
        original_message = items[-1]['text'] if items else ""
        
        enhanced_message = f"""
        CONTEXT: This question was routed from the main technical advisor.
        
        ORIGINAL QUESTION: {original_message}
        
        PRIORITY: High - Expert consultation requested
        
        Please provide a comprehensive technical response.
        """
        
        return [{"type": "text", "text": enhanced_message}]
    
    def security_context_filter(items):
        """Add security-specific context"""
        original_message = items[-1]['text'] if items else ""
        
        enhanced_message = f"""
        🔒 SECURITY CONSULTATION REQUEST
        
        Original Question: {original_message}
        
        Please analyze this from a security perspective and provide:
        1. Security risks and vulnerabilities
        2. Recommended security measures
        3. Best practices and compliance considerations
        
        Priority: Security-critical
        """
        
        return [{"type": "text", "text": enhanced_message}]
    
    enhanced_router = Agent(
        name="EnhancedRouter",
        instructions="""
        You are an advanced technical router that enhances questions with context
        before routing to experts. This helps experts provide more targeted responses.
        """,
        handoffs=[
            handoff(
                agent=security_expert,
                input_filter=security_context_filter,
                description="Route to security expert with enhanced security context"
            ),
            handoff(
                agent=performance_expert,
                input_filter=add_context_filter,
                description="Route to performance expert with enhanced context"
            )
        ]
    )
    
    # 2.4 Handoffs with Callbacks
    def on_security_handoff(ctx: RunContextWrapper, agent: Agent, input_items: List[Dict]) -> None:
        """Called when routing to security expert"""
        print(f"🔒 HANDOFF LOG: Routing security question to {agent.name}")
        print(f"    Context: {ctx.context}")
        print(f"    Timestamp: {datetime.now()}")
    
    def on_performance_handoff(ctx: RunContextWrapper, agent: Agent, input_items: List[Dict]) -> None:
        """Called when routing to performance expert"""
        print(f"⚡ HANDOFF LOG: Routing performance question to {agent.name}")
        print(f"    Input length: {len(input_items)}")
        print(f"    Timestamp: {datetime.now()}")
    
    callback_router = Agent(
        name="CallbackRouter",
        instructions="""
        You route technical questions to experts with logging and tracking.
        This helps monitor which types of questions get routed where.
        """,
        handoffs=[
            handoff(
                agent=security_expert,
                on_handoff=on_security_handoff,
                description="Route to security expert with logging"
            ),
            handoff(
                agent=performance_expert,
                on_handoff=on_performance_handoff,
                description="Route to performance expert with logging"
            )
        ]
    )
    
    # Test advanced handoffs
    advanced_test_scenarios = [
        (expert_router, "How should I implement authentication in my web application?"),
        (enhanced_router, "My database queries are slow. What should I do?"),
        (callback_router, "Is it safe to store user passwords in plain text?")
    ]
    
    print("\n2. Testing Advanced Handoff Objects:")
    for i, (agent, query) in enumerate(advanced_test_scenarios, 1):
        try:
            result = Runner.run_sync(agent, query)
            print(f"{i}. Agent: {agent.name}")
            print(f"   Query: {query}")
            print(f"   Final Agent: {result.last_agent.name}")
            print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return {
        "expert_router": expert_router,
        "enhanced_router": enhanced_router,
        "callback_router": callback_router
    }

# =============================================================================
# 🎯 3. CONDITIONAL HANDOFFS
# =============================================================================

def conditional_handoffs_examples():
    """Conditional handoffs based on context and logic"""
    
    print("\n🎯 CONDITIONAL HANDOFFS")
    print("=" * 50)
    
    # 3.1 Create Tier-Based Agents
    basic_support = Agent(
        name="BasicSupport",
        instructions="""
        You provide basic technical support for common issues:
        - Password resets
        - Account setup
        - Basic troubleshooting
        - FAQ responses
        
        Keep responses simple and easy to follow.
        """
    )
    
    advanced_support = Agent(
        name="AdvancedSupport",
        instructions="""
        You provide advanced technical support for complex issues:
        - System integration problems
        - Advanced configuration
        - Performance troubleshooting
        - Custom solution development
        
        Provide detailed technical solutions.
        """
    )
    
    enterprise_support = Agent(
        name="EnterpriseSupport",
        instructions="""
        You provide enterprise-level support:
        - Critical system issues
        - Custom enterprise solutions
        - SLA-bound support
        - Executive-level communication
        
        Prioritize business impact and provide white-glove service.
        """
    )
    
    # 3.2 Conditional Logic Functions
    def user_tier_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Check if user qualifies for advanced support"""
        user_context = ctx.context or {}
        user_tier = user_context.get('tier', 'basic')
        return user_tier in ['advanced', 'enterprise']
    
    def enterprise_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Check if user qualifies for enterprise support"""
        user_context = ctx.context or {}
        user_tier = user_context.get('tier', 'basic')
        priority = user_context.get('priority', 'normal')
        return user_tier == 'enterprise' or priority == 'critical'
    
    def complexity_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Check if question requires advanced support based on keywords"""
        # In real implementation, you might analyze the question content
        user_context = ctx.context or {}
        complexity = user_context.get('complexity', 'simple')
        return complexity in ['complex', 'advanced']
    
    # 3.3 Conditional Support Router
    conditional_support = Agent(
        name="ConditionalSupport",
        instructions="""
        You are a smart support router that escalates based on:
        - User tier and permissions
        - Question complexity
        - Business priority
        
        For basic questions from any user, handle them yourself.
        Escalate appropriately based on context.
        """,
        handoffs=[
            handoff(
                agent=advanced_support,
                is_enabled=user_tier_check,
                description="Escalate to advanced support for premium users"
            ),
            handoff(
                agent=enterprise_support,
                is_enabled=enterprise_check,
                description="Escalate to enterprise support for enterprise customers"
            )
        ]
    )
    
    # 3.4 Time-Based Conditional Handoffs
    def business_hours_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Check if it's business hours for live agent handoff"""
        import datetime
        now = datetime.datetime.now()
        # Business hours: 9 AM to 5 PM, Monday to Friday
        is_weekday = now.weekday() < 5
        is_business_hours = 9 <= now.hour < 17
        
        return is_weekday and is_business_hours
    
    def urgent_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Check if issue is urgent regardless of time"""
        user_context = ctx.context or {}
        priority = user_context.get('priority', 'normal')
        return priority in ['urgent', 'critical']
    
    live_agent = Agent(
        name="LiveAgent",
        instructions="""
        You are a live human agent providing real-time support.
        Handle urgent issues with immediate attention and personalized service.
        """
    )
    
    after_hours_agent = Agent(
        name="AfterHoursAgent",
        instructions="""
        You provide after-hours automated support:
        - Acknowledge urgent issues
        - Provide immediate workarounds
        - Schedule follow-up for business hours
        - Escalate true emergencies
        """
    )
    
    time_based_router = Agent(
        name="TimeBased Router",
        instructions="""
        You route support requests based on time and urgency:
        - Business hours: Route to live agent when available
        - After hours: Provide automated support
        - Urgent issues: Always escalate appropriately
        """,
        handoffs=[
            handoff(
                agent=live_agent,
                is_enabled=business_hours_check,
                description="Route to live agent during business hours"
            ),
            handoff(
                agent=after_hours_agent,
                is_enabled=urgent_check,
                description="Route urgent issues to after-hours support"
            )
        ]
    )
    
    # 3.5 Multi-Condition Logic
    def advanced_routing_logic(ctx: RunContextWrapper, agent: Agent) -> bool:
        """Complex routing logic with multiple conditions"""
        user_context = ctx.context or {}
        
        # Get context variables
        user_tier = user_context.get('tier', 'basic')
        priority = user_context.get('priority', 'normal')
        complexity = user_context.get('complexity', 'simple')
        previous_escalations = user_context.get('escalations', 0)
        
        # Complex logic for routing
        conditions = [
            user_tier in ['enterprise', 'premium'],
            priority in ['high', 'critical'],
            complexity in ['complex', 'advanced'],
            previous_escalations >= 2  # User has been escalated before
        ]
        
        # Route if any 2 conditions are met
        return sum(conditions) >= 2
    
    smart_router = Agent(
        name="SmartRouter",
        instructions="""
        You use sophisticated logic to route support requests.
        Consider multiple factors: user tier, priority, complexity, history.
        """,
        handoffs=[
            handoff(
                agent=enterprise_support,
                is_enabled=advanced_routing_logic,
                description="Route to enterprise support based on multiple factors"
            )
        ]
    )
    
    # Test conditional handoffs with different contexts
    conditional_test_scenarios = [
        # Basic user
        (conditional_support, "How do I reset my password?", {'tier': 'basic'}),
        # Advanced user
        (conditional_support, "I need help with API integration", {'tier': 'advanced', 'complexity': 'complex'}),
        # Enterprise user
        (conditional_support, "Critical system outage!", {'tier': 'enterprise', 'priority': 'critical'}),
        # Time-based routing (mock business hours)
        (time_based_router, "Urgent: Payment processing down", {'priority': 'urgent'}),
        # Smart routing
        (smart_router, "Complex technical issue", {'tier': 'premium', 'priority': 'high', 'escalations': 1})
    ]
    
    print("\n3. Testing Conditional Handoffs:")
    for i, (agent, query, context) in enumerate(conditional_test_scenarios, 1):
        try:
            result = Runner.run_sync(agent, query, context=context)
            print(f"{i}. Agent: {agent.name}")
            print(f"   Query: {query}")
            print(f"   Context: {context}")
            print(f"   Final Agent: {result.last_agent.name}")
            print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return {
        "conditional_support": conditional_support,
        "time_based_router": time_based_router,
        "smart_router": smart_router
    }

# =============================================================================
# 🔗 4. MULTI-LEVEL HANDOFF CHAINS  
# =============================================================================

def multi_level_handoff_examples():
    """Multi-level handoff chains and complex workflows"""
    
    print("\n🔗 MULTI-LEVEL HANDOFF CHAINS")
    print("=" * 50)
    
    # 4.1 Create a Multi-Level Support Hierarchy
    
    # Level 1: Frontline Support
    l1_support = Agent(
        name="L1Support",
        instructions="""
        You are Level 1 (frontline) support. Handle:
        - Password resets
        - Basic account questions
        - Simple troubleshooting
        - FAQ responses
        
        If you can't resolve the issue, escalate to L2 Support.
        """
    )
    
    # Level 2: Technical Support  
    l2_support = Agent(
        name="L2Support",
        instructions="""
        You are Level 2 (technical) support. Handle:
        - Complex technical issues
        - API problems
        - Integration questions
        - Advanced troubleshooting
        
        If the issue requires development expertise, escalate to L3 Support.
        You can also route back to L1 for simple issues.
        """,
        handoffs=[]  # Will add handoffs after L3 is defined
    )
    
    # Level 3: Expert/Engineering Support
    l3_support = Agent(
        name="L3Support", 
        instructions="""
        You are Level 3 (expert/engineering) support. Handle:
        - Critical system issues
        - Bug investigations
        - Custom development needs  
        - Architectural questions
        
        You are the highest level of technical support.
        Provide definitive solutions or escalate to product team.
        """,
        handoffs=[]  # This is the top level
    )
    
    # Set up the handoff chain
    l1_support.handoffs = [l2_support]
    l2_support.handoffs = [l3_support, l1_support]  # Can escalate up or route back down
    
    # 4.2 Create a Sales Process Chain
    
    # Lead Qualification
    lead_qualifier = Agent(
        name="LeadQualifier",
        instructions="""
        You qualify sales leads by asking about:
        - Company size and needs
        - Budget and timeline
        - Decision-making process
        - Current solutions
        
        Route qualified leads to Sales Consultant.
        Route unqualified leads to Marketing for nurturing.
        """
    )
    
    # Sales Consultant
    sales_consultant = Agent(
        name="SalesConsultant",
        instructions="""
        You are a sales consultant who:
        - Conducts product demos
        - Develops custom proposals
        - Negotiates pricing
        - Manages sales pipeline
        
        Route technical questions to Sales Engineer.
        Route contract questions to Sales Manager.
        """
    )
    
    # Sales Engineer
    sales_engineer = Agent(
        name="SalesEngineer",
        instructions="""
        You provide technical sales support:
        - Technical product demos
        - Integration feasibility
        - Architecture discussions
        - Technical proposal development
        
        Work closely with Sales Consultant on technical deals.
        """
    )
    
    # Sales Manager
    sales_manager = Agent(
        name="SalesManager",
        instructions="""
        You handle high-level sales activities:
        - Contract negotiations
        - Pricing approvals
        - Enterprise deals
        - Executive relationships
        
        You can close deals and make final decisions.
        """
    )
    
    # Marketing (for nurturing)
    marketing = Agent(
        name="Marketing",
        instructions="""
        You handle marketing and lead nurturing:
        - Educational content
        - Webinar invitations
        - Newsletter signups
        - Long-term relationship building
        
        Keep unqualified leads engaged for future opportunities.
        """
    )
    
    # Set up sales handoff chain
    lead_qualifier.handoffs = [sales_consultant, marketing]
    sales_consultant.handoffs = [sales_engineer, sales_manager]
    
    # 4.3 Create a Customer Journey Chain
    
    onboarding_specialist = Agent(
        name="OnboardingSpecialist",
        instructions="""
        You help new customers get started:
        - Account setup and configuration
        - Initial training and tutorials
        - Basic integration support
        - Success milestone tracking
        
        Route complex technical issues to Technical Consultant.
        Route completed onboarding to Account Manager.
        """
    )
    
    technical_consultant = Agent(
        name="TechnicalConsultant",
        instructions="""
        You provide ongoing technical guidance:
        - Advanced implementation support
        - Custom integration development
        - Performance optimization
        - Best practices consulting
        
        Work with customers throughout their technical journey.
        """
    )
    
    account_manager = Agent(
        name="AccountManager",
        instructions="""
        You manage ongoing customer relationships:
        - Renewal discussions
        - Expansion opportunities
        - Executive check-ins
        - Success measurement
        
        You own the long-term customer relationship.
        """
    )
    
    # Set up customer journey chain
    onboarding_specialist.handoffs = [technical_consultant, account_manager]
    
    # 4.4 Create Initial Routing Agent
    master_router = Agent(
        name="MasterRouter",
        instructions="""
        You are the main entry point that routes requests to appropriate workflows:
        
        SUPPORT ISSUES:
        - Technical problems → L1Support (will escalate as needed)
        
        SALES INQUIRIES:
        - New sales questions → LeadQualifier
        
        CUSTOMER SUCCESS:
        - New customer questions → OnboardingSpecialist
        - Existing customer questions → AccountManager
        
        Determine the intent and route appropriately.
        """,
        handoffs=[l1_support, lead_qualifier, onboarding_specialist, account_manager]
    )
    
    # Test multi-level handoffs
    multi_level_scenarios = [
        # Support escalation chain
        (master_router, "I'm having a critical system outage that L1 support couldn't fix"),
        
        # Sales process chain  
        (master_router, "I'm interested in your enterprise solution for my 500-person company"),
        
        # Customer success chain
        (master_router, "I'm a new customer and need help setting up my account"),
        
        # Direct L1 support (to test internal escalation)
        (l1_support, "My API integration is returning 500 errors consistently")
    ]
    
    print("\n4. Testing Multi-Level Handoff Chains:")
    for i, (agent, query) in enumerate(multi_level_scenarios, 1):
        try:
            result = Runner.run_sync(agent, query, max_turns=5)  # Allow more turns for chains
            print(f"{i}. Starting Agent: {agent.name}")
            print(f"   Query: {query}")
            print(f"   Final Agent: {result.last_agent.name}")
            print(f"   Total Turns: {len(result.run_steps)}")
            print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return {
        "master_router": master_router,
        "l1_support": l1_support,
        "sales_process": lead_qualifier,
        "customer_success": onboarding_specialist
    }

# =============================================================================
# 🎭 5. HANDOFF PATTERNS & USE CASES
# =============================================================================

def handoff_patterns_examples():
    """Common handoff patterns and when to use them"""
    
    print("\n🎭 HANDOFF PATTERNS & USE CASES")
    print("=" * 50)
    
    # 5.1 Pattern: Language Detection and Routing
    print("Pattern 1: Language Detection Router")
    
    multilingual_agents = {}
    languages = ['spanish', 'french', 'german', 'italian', 'portuguese']
    
    for lang in languages:
        multilingual_agents[lang] = Agent(
            name=f"{lang.title()}Agent",
            instructions=f"""
            You are a native {lang} speaker assistant.
            Respond only in {lang} and help users with their questions.
            Provide culturally appropriate and linguistically correct responses.
            """
        )
    
    language_detector = Agent(
        name="LanguageDetector",
        instructions="""
        You detect the language of user input and route to native speakers:
        - Spanish → SpanishAgent
        - French → FrenchAgent
        - German → GermanAgent
        - Italian → ItalianAgent
        - Portuguese → PortugueseAgent
        - English → handle yourself
        
        If language is unclear, ask user to specify.
        """,
        handoffs=list(multilingual_agents.values())
    )
    
    # 5.2 Pattern: Expertise-Based Routing
    print("Pattern 2: Domain Expertise Router")
    
    domain_experts = {
        'medical': Agent(
            name="MedicalExpert",
            instructions="""
            You are a medical information expert. Provide:
            - General health information
            - Medical terminology explanations
            - Guidance on when to see doctors
            
            IMPORTANT: Always recommend consulting healthcare providers for medical decisions.
            """
        ),
        'legal': Agent(
            name="LegalExpert", 
            instructions="""
            You are a legal information expert. Provide:
            - General legal concepts
            - Legal procedure explanations
            - Document templates and guidance
            
            IMPORTANT: Always recommend consulting lawyers for legal advice.
            """
        ),
        'financial': Agent(
            name="FinancialExpert",
            instructions="""
            You are a financial information expert. Provide:
            - Investment concepts
            - Financial planning guidance
            - Tax information (general)
            
            IMPORTANT: Always recommend consulting financial advisors for personal advice.
            """
        )
    }
    
    domain_router = Agent(
        name="DomainRouter",
        instructions="""
        You route questions to appropriate domain experts:
        - Health/medical questions → MedicalExpert
        - Legal questions → LegalExpert  
        - Financial questions → FinancialExpert
        
        For general questions, provide helpful answers yourself.
        """,
        handoffs=list(domain_experts.values())
    )
    
    # 5.3 Pattern: Workflow Stage Routing
    print("Pattern 3: Project Workflow Router")
    
    workflow_agents = {
        'planning': Agent(
            name="ProjectPlanner",
            instructions="""
            You help with project planning:
            - Requirements gathering
            - Timeline estimation
            - Resource planning
            - Risk assessment
            
            Focus on upfront planning and preparation.
            """
        ),
        'execution': Agent(
            name="ProjectExecutor",
            instructions="""
            You help with project execution:
            - Task management
            - Progress tracking
            - Issue resolution
            - Team coordination
            
            Focus on getting work done effectively.
            """
        ),
        'review': Agent(
            name="ProjectReviewer",
            instructions="""
            You help with project review and optimization:
            - Quality assessment
            - Performance analysis
            - Lessons learned
            - Process improvement
            
            Focus on learning and optimization.
            """
        )
    }
    
    workflow_router = Agent(
        name="WorkflowRouter",
        instructions="""
        You route project questions based on workflow stage:
        - Planning/design questions → ProjectPlanner
        - Implementation/execution questions → ProjectExecutor  
        - Review/optimization questions → ProjectReviewer
        
        Help users identify which stage they're in.
        """,
        handoffs=list(workflow_agents.values())
    )
    
    # 5.4 Pattern: Complexity-Based Routing
    print("Pattern 4: Complexity-Based Router")
    
    def simple_question_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        # In practice, you might use ML or keyword analysis
        user_context = ctx.context or {}
        return user_context.get('complexity', 'unknown') == 'simple'
    
    def complex_question_check(ctx: RunContextWrapper, agent: Agent) -> bool:
        user_context = ctx.context or {}
        return user_context.get('complexity', 'unknown') == 'complex'
    
    simple_agent = Agent(
        name="SimpleAgent",
        instructions="""
        You handle simple, straightforward questions:
        - Quick answers
        - Basic explanations
        - FAQ-style responses
        
        Keep responses concise and clear.
        """
    )
    
    complex_agent = Agent(
        name="ComplexAgent",
        instructions="""
        You handle complex, nuanced questions:
        - Detailed analysis
        - Multi-faceted explanations
        - Advanced problem-solving
        
        Provide comprehensive, thoughtful responses.
        """
    )
    
    complexity_router = Agent(
        name="ComplexityRouter",
        instructions="""
        You assess question complexity and route accordingly:
        - Simple questions → SimpleAgent
        - Complex questions → ComplexAgent
        
        For medium complexity, handle yourself.
        """,
        handoffs=[
            handoff(simple_agent, is_enabled=simple_question_check),
            handoff(complex_agent, is_enabled=complex_question_check)
        ]
    )
    
    # Test patterns
    pattern_scenarios = [
        # Language routing
        (language_detector, "Bonjour, comment allez-vous?"),
        
        # Domain expertise
        (domain_router, "What should I know about starting an LLC?"),
        
        # Workflow stage
        (workflow_router, "How do I plan a software development project?"),
        
        # Complexity-based (with context)
        (complexity_router, "What is 2+2?", {'complexity': 'simple'}),
        (complexity_router, "Explain quantum computing", {'complexity': 'complex'})
    ]
    
    print("\n5. Testing Handoff Patterns:")
    for i, scenario in enumerate(pattern_scenarios, 1):
        agent = scenario[0]
        query = scenario[1]
        context = scenario[2] if len(scenario) > 2 else None
        
        try:
            result = Runner.run_sync(agent, query, context=context)
            print(f"{i}. Pattern: {agent.name}")
            print(f"   Query: {query}")
            if context:
                print(f"   Context: {context}")
            print(f"   Final Agent: {result.last_agent.name}")
            print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")

# =============================================================================
# 💡 6. HANDOFFS BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 HANDOFFS BEST PRACTICES

1. 🎯 WHEN TO USE HANDOFFS:
   ✅ Clear specialization boundaries (Python expert, Sales agent)
   ✅ Sequential workflows (L1 → L2 → L3 support)
   ✅ Language or region routing
   ✅ User permission/tier-based routing
   ❌ Simple task delegation (use agent-as-tool instead)

2. 🤝 HANDOFF vs AGENT-AS-TOOL:
   HANDOFF:
   - Transfers complete control
   - New agent owns the conversation
   - Good for: specialization, workflows, routing
   
   AGENT-AS-TOOL:
   - Main agent stays in control
   - Sub-agent provides specific capability
   - Good for: quick tasks, data processing, calculations

3. 🏗️ HANDOFF DESIGN PATTERNS:

   Pattern A - Simple Routing:
   ```python
   router = Agent(
       name="Router",
       instructions="Route to appropriate specialist",
       handoffs=[specialist1, specialist2, specialist3]
   )
   ```

   Pattern B - Conditional Routing:
   ```python
   router = Agent(
       handoffs=[
           handoff(expert, is_enabled=condition_check),
           handoff(basic, is_enabled=lambda ctx, agent: not condition_check(ctx, agent))
       ]
   )
   ```

   Pattern C - Enhanced Context:
   ```python
   def add_context(items):
       enhanced = f"EXPERT CONSULTATION: {items[-1]['text']}"
       return [{"type": "text", "text": enhanced}]
   
   agent.handoffs = [handoff(expert, input_filter=add_context)]
   ```

   Pattern D - Workflow Chain:
   ```python
   step1.handoffs = [step2]
   step2.handoffs = [step3, step1]  # Can go forward or back
   step3.handoffs = []  # End of chain
   ```

4. ⚙️ CONDITIONAL LOGIC BEST PRACTICES:
   ✅ Use context for user-specific routing (tier, permissions)
   ✅ Use keywords/ML for content-based routing
   ✅ Use time/business rules for availability routing
   ✅ Combine multiple conditions for smart routing
   ❌ Don't make conditions too complex or hard to debug

5. 🔄 INPUT FILTERS BEST PRACTICES:
   ✅ Add helpful context for the target agent
   ✅ Reformat input for better processing
   ✅ Include metadata (priority, source, etc.)
   ❌ Don't completely change the user's intent

6. 📊 MONITORING & DEBUGGING:
   ✅ Use on_handoff callbacks for logging
   ✅ Track handoff patterns and success rates
   ✅ Monitor agent performance after handoffs
   ✅ Test handoff chains thoroughly

🚨 COMMON HANDOFF PITFALLS:

❌ DON'T create circular handoff loops
❌ DON'T use handoffs for simple task delegation
❌ DON'T make handoff conditions too complex
❌ DON'T forget to handle cases where no handoff applies
❌ DON'T lose important context during handoffs
❌ DON'T create too many handoff levels (keep chains manageable)

✅ HANDOFF SUCCESS CHECKLIST:

☐ Clear handoff purpose and criteria
☐ Proper agent specialization
☐ Tested handoff conditions
☐ Appropriate input filtering
☐ Logging and monitoring
☐ Error handling for failed handoffs
☐ Performance testing under load
☐ User experience validation

📈 SCALING HANDOFFS:

For high-volume systems:
- Use async handoffs where possible
- Monitor handoff latency and success rates
- Implement fallback agents for failed handoffs
- Cache conditional logic results
- Load balance across agent instances
- Implement circuit breakers for failing handoffs
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_handoff_examples():
    """Run all handoff examples comprehensively"""
    
    print("🤝 OPENAI AGENTS SDK - COMPLETE HANDOFFS DEMONSTRATION")
    print("=" * 70)
    
    # 1. Basic Handoffs
    basic_agents = basic_handoffs_examples()
    
    # 2. Advanced Handoff Objects
    advanced_agents = advanced_handoff_examples()
    
    # 3. Conditional Handoffs
    conditional_agents = conditional_handoffs_examples()
    
    # 4. Multi-Level Chains
    chain_agents = multi_level_handoff_examples()
    
    # 5. Common Patterns
    handoff_patterns_examples()
    
    print("\n✅ All handoff examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Handoffs transfer complete control between agents")
    print("- Use handoffs for specialization, not simple task delegation")
    print("- Conditional handoffs enable smart, context-aware routing")
    print("- Input filters help target agents provide better responses")
    print("- Multi-level chains enable complex workflows")
    print("- Always monitor and log handoff performance")

def run_sync_handoff_examples():
    """Run synchronous handoff examples for immediate testing"""
    
    print("🤝 OPENAI AGENTS SDK - HANDOFFS SYNC EXAMPLES")
    print("=" * 60)
    
    # Run only synchronous examples
    basic_handoffs_examples()
    advanced_handoff_examples()
    conditional_handoffs_examples() 
    multi_level_handoff_examples()
    handoff_patterns_examples()
    
    print("\n✅ Sync handoff examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🤝 OpenAI Agents SDK - Complete Handoffs Template")
    print("This template demonstrates all Handoff patterns and best practices.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_handoff_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_handoff_examples())
    
    print("\n✅ Handoffs template demonstration complete!")
    print("💡 Use this template as reference for all your handoff development needs.")
