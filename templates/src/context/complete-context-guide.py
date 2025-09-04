"""
🧠 OPENAI AGENTS SDK - COMPLETE CONTEXT GUIDE & TEMPLATE

This template covers everything about Context in the OpenAI Agents SDK:
- Context passing and management patterns
- Dynamic instructions and context injection
- Context validation and sanitization
- Business logic integration with context
- Memory and state management in context
- Context inheritance and composition
- Performance optimization for context handling
- Advanced context manipulation techniques

📚 Based on: https://openai.github.io/openai-agents-python/ref/context/
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from agents import Agent, Runner, function_tool

# =============================================================================
# 📖 UNDERSTANDING CONTEXT IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT IS CONTEXT IN AGENTS?

Context provides dynamic information and configuration to agents:
1. ✅ DYNAMIC INSTRUCTIONS: Context-aware agent behavior
2. ✅ BUSINESS LOGIC: Integrate real-time business rules
3. ✅ STATE MANAGEMENT: Maintain conversation and session state
4. ✅ PERSONALIZATION: User-specific context and preferences
5. ✅ SECURITY: Role-based access and permissions

KEY CONCEPTS:

1. CONTEXT TYPES:
   - Static Context: Fixed configuration and instructions
   - Dynamic Context: Runtime-generated context from functions
   - User Context: User-specific information and preferences
   - Session Context: Conversation state and history
   - Business Context: Current business rules and data

2. CONTEXT SOURCES:
   - Database queries for current state
   - API calls to external services
   - User profiles and preferences
   - Session and conversation history
   - Real-time business metrics

3. CONTEXT LIFECYCLE:
   - Initialization: Set up base context
   - Injection: Add dynamic context per request
   - Validation: Ensure context integrity
   - Transformation: Process context for agent
   - Cleanup: Remove sensitive or stale context

4. CONTEXT PATTERNS:
   - Context Functions: Generate context on demand
   - Context Middleware: Process context pipeline
   - Context Inheritance: Share context across agents
   - Context Composition: Combine multiple context sources
"""

# =============================================================================
# 🏗️ 1. BASIC CONTEXT PATTERNS
# =============================================================================

class ContextType(Enum):
    """Types of context"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    USER = "user"
    SESSION = "session"
    BUSINESS = "business"
    SECURITY = "security"

@dataclass
class ContextData:
    """Base context data structure"""
    type: ContextType
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if context data has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }

class ContextManager:
    """Manage context data for agents"""
    
    def __init__(self):
        self.contexts: Dict[str, ContextData] = {}
        self.context_functions: Dict[str, Callable] = {}
    
    def add_context(self, context: ContextData):
        """Add context data"""
        self.contexts[context.key] = context
    
    def get_context(self, key: str) -> Optional[ContextData]:
        """Get context data by key"""
        context = self.contexts.get(key)
        if context and context.is_expired():
            del self.contexts[key]
            return None
        return context
    
    def remove_context(self, key: str):
        """Remove context data"""
        self.contexts.pop(key, None)
    
    def register_context_function(self, key: str, func: Callable):
        """Register a context generation function"""
        self.context_functions[key] = func
    
    def generate_dynamic_context(self, key: str, **kwargs) -> Optional[ContextData]:
        """Generate dynamic context using registered function"""
        if key in self.context_functions:
            try:
                value = self.context_functions[key](**kwargs)
                return ContextData(
                    type=ContextType.DYNAMIC,
                    key=key,
                    value=value,
                    metadata={'generated_at': datetime.now().isoformat()}
                )
            except Exception as e:
                print(f"Error generating context {key}: {e}")
                return None
        return None
    
    def get_all_contexts(self, context_types: List[ContextType] = None) -> Dict[str, Any]:
        """Get all contexts as dictionary"""
        contexts = {}
        
        for key, context in self.contexts.items():
            if context.is_expired():
                continue
            
            if context_types is None or context.type in context_types:
                contexts[key] = context.value
        
        return contexts
    
    def cleanup_expired_contexts(self):
        """Remove expired contexts"""
        expired_keys = [
            key for key, context in self.contexts.items()
            if context.is_expired()
        ]
        
        for key in expired_keys:
            del self.contexts[key]

def basic_context_examples():
    """Basic context management examples"""
    
    print("🏗️ BASIC CONTEXT PATTERNS")
    print("=" * 50)
    
    # Create context manager
    context_manager = ContextManager()
    
    print("\n1. Static Context:")
    
    # Add static context
    static_contexts = [
        ContextData(
            type=ContextType.STATIC,
            key="company_name",
            value="Acme Corporation",
            metadata={"description": "Company name for branding"}
        ),
        ContextData(
            type=ContextType.STATIC,
            key="business_hours",
            value="9:00 AM - 6:00 PM EST",
            metadata={"timezone": "America/New_York"}
        ),
        ContextData(
            type=ContextType.STATIC,
            key="support_email",
            value="support@acme.com",
            metadata={"department": "customer_service"}
        )
    ]
    
    for context in static_contexts:
        context_manager.add_context(context)
        print(f"Added static context: {context.key} = {context.value}")
    
    print("\n2. Dynamic Context Functions:")
    
    # Register context functions
    def get_current_time():
        """Get current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_weather_info(location: str = "New York"):
        """Get weather information (simulated)"""
        import random
        conditions = ["Sunny", "Cloudy", "Rainy", "Snowy"]
        return {
            "location": location,
            "condition": random.choice(conditions),
            "temperature": random.randint(32, 85)
        }
    
    def get_user_preferences(user_id: str = "default"):
        """Get user preferences (simulated)"""
        preferences = {
            "default": {"language": "en", "timezone": "UTC", "theme": "light"},
            "user123": {"language": "es", "timezone": "America/Los_Angeles", "theme": "dark"},
            "user456": {"language": "fr", "timezone": "Europe/Paris", "theme": "auto"}
        }
        return preferences.get(user_id, preferences["default"])
    
    context_manager.register_context_function("current_time", get_current_time)
    context_manager.register_context_function("weather", get_weather_info)
    context_manager.register_context_function("user_prefs", get_user_preferences)
    
    print("Registered context functions:")
    for func_name in context_manager.context_functions.keys():
        print(f"  - {func_name}")
    
    print("\n3. Generating Dynamic Context:")
    
    # Generate dynamic contexts
    time_context = context_manager.generate_dynamic_context("current_time")
    weather_context = context_manager.generate_dynamic_context("weather", location="San Francisco")
    prefs_context = context_manager.generate_dynamic_context("user_prefs", user_id="user123")
    
    if time_context:
        context_manager.add_context(time_context)
        print(f"Time context: {time_context.value}")
    
    if weather_context:
        context_manager.add_context(weather_context)
        print(f"Weather context: {weather_context.value}")
    
    if prefs_context:
        context_manager.add_context(prefs_context)
        print(f"User preferences: {prefs_context.value}")
    
    print("\n4. Context Retrieval:")
    
    # Get all contexts
    all_contexts = context_manager.get_all_contexts()
    print(f"All contexts ({len(all_contexts)} items):")
    for key, value in all_contexts.items():
        print(f"  {key}: {value}")
    
    # Get specific context types
    static_only = context_manager.get_all_contexts([ContextType.STATIC])
    dynamic_only = context_manager.get_all_contexts([ContextType.DYNAMIC])
    
    print(f"\nStatic contexts: {len(static_only)} items")
    print(f"Dynamic contexts: {len(dynamic_only)} items")
    
    return {
        "context_manager": context_manager,
        "static_contexts": static_contexts
    }

# =============================================================================
# 🤖 2. AGENT CONTEXT INTEGRATION
# =============================================================================

class ContextualAgent:
    """Agent wrapper with context integration"""
    
    def __init__(self, agent: Agent, context_manager: ContextManager = None):
        self.agent = agent
        self.context_manager = context_manager or ContextManager()
        self.context_template = None
    
    def set_context_template(self, template: str):
        """Set context template for injection"""
        self.context_template = template
    
    def build_contextual_instructions(self, base_instructions: str = None) -> str:
        """Build instructions with injected context"""
        instructions = base_instructions or self.agent.instructions or ""
        
        # Get all current contexts
        contexts = self.context_manager.get_all_contexts()
        
        # Build context section
        context_section = "\n\n## CURRENT CONTEXT:\n"
        
        for key, value in contexts.items():
            if isinstance(value, dict):
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)
            
            context_section += f"- {key.replace('_', ' ').title()}: {formatted_value}\n"
        
        # Use template if provided
        if self.context_template:
            try:
                formatted_context = self.context_template.format(**contexts)
                return f"{instructions}\n\n{formatted_context}"
            except KeyError as e:
                print(f"Context template key error: {e}")
                return f"{instructions}{context_section}"
        
        return f"{instructions}{context_section}"
    
    def run_with_context(self, message: str, **run_kwargs):
        """Run agent with context injection"""
        
        # Clean up expired contexts
        self.context_manager.cleanup_expired_contexts()
        
        # Build contextual instructions
        contextual_instructions = self.build_contextual_instructions()
        
        # Create temporary agent with contextual instructions
        temp_agent = Agent(
            name=self.agent.name,
            instructions=contextual_instructions,
            tools=self.agent.tools,
            model=self.agent.model
        )
        
        # Run with context
        return Runner.run_sync(temp_agent, message, **run_kwargs)
    
    def add_business_context(self, business_data: Dict[str, Any]):
        """Add business context data"""
        for key, value in business_data.items():
            context = ContextData(
                type=ContextType.BUSINESS,
                key=key,
                value=value,
                expires_at=datetime.now() + timedelta(hours=1)  # Expire in 1 hour
            )
            self.context_manager.add_context(context)
    
    def add_user_context(self, user_data: Dict[str, Any]):
        """Add user-specific context"""
        for key, value in user_data.items():
            context = ContextData(
                type=ContextType.USER,
                key=f"user_{key}",
                value=value,
                expires_at=datetime.now() + timedelta(days=1)  # Expire in 1 day
            )
            self.context_manager.add_context(context)

def agent_context_examples():
    """Examples of agent context integration"""
    
    print("\n🤖 AGENT CONTEXT INTEGRATION")
    print("=" * 50)
    
    # Create tools for context demonstration
    @function_tool
    def get_account_balance(account_id: str) -> str:
        """Get account balance for a customer"""
        # Simulate database lookup
        balances = {
            "ACC001": "$5,240.50",
            "ACC002": "$12,875.25", 
            "ACC003": "$950.00"
        }
        return balances.get(account_id, "$0.00")
    
    @function_tool
    def get_order_status(order_id: str) -> str:
        """Get order status"""
        statuses = {
            "ORD001": "Shipped - Tracking: ABC123456",
            "ORD002": "Processing - Expected ship date: Tomorrow",
            "ORD003": "Delivered - Signed by: John Doe"
        }
        return statuses.get(order_id, "Order not found")
    
    @function_tool
    def schedule_appointment(date: str, time: str, service: str) -> str:
        """Schedule an appointment"""
        return f"Appointment scheduled for {service} on {date} at {time}"
    
    # Create base agent
    customer_service_agent = Agent(
        name="CustomerServiceAgent",
        instructions="""
        You are a helpful customer service representative.
        Provide friendly, professional assistance to customers.
        Use the available tools to look up account information and help with requests.
        """,
        tools=[get_account_balance, get_order_status, schedule_appointment]
    )
    
    # Create contextual agent wrapper
    contextual_agent = ContextualAgent(customer_service_agent)
    
    print("\n1. Setting Up Context Templates:")
    
    # Set context template
    context_template = """
## CURRENT CONTEXT:
Customer: {user_name} (ID: {user_id})
Account Type: {user_account_type}
Location: {user_location}
Current Time: {current_time}
Business Hours: {business_hours}
Today's Promotions: {todays_promotions}

## BUSINESS RULES:
- Always greet customers by name when available
- Mention current promotions when relevant
- Check if customer is within business hours
- Use formal language for Premium customers, casual for Standard customers
"""
    
    contextual_agent.set_context_template(context_template)
    print("✅ Context template set")
    
    print("\n2. Adding Context Data:")
    
    # Add user context
    user_context = {
        "name": "Sarah Johnson",
        "id": "CUST12345",
        "account_type": "Premium",
        "location": "New York, NY",
        "preferred_language": "English",
        "last_contact": "2024-03-10"
    }
    contextual_agent.add_user_context(user_context)
    
    # Add business context
    business_context = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "business_hours": "9:00 AM - 6:00 PM EST",
        "todays_promotions": "20% off premium services, Free shipping on orders over $100",
        "system_status": "All systems operational",
        "support_queue_length": "2 customers ahead"
    }
    contextual_agent.add_business_context(business_context)
    
    print(f"Added user context for: {user_context['name']}")
    print(f"Added business context: {len(business_context)} items")
    
    print("\n3. Testing Contextual Responses:")
    
    # Test queries with context
    test_queries = [
        "Hi, I'd like to check my account balance for account ACC001",
        "Can you help me track my order ORD002?",
        "I'd like to schedule an appointment for next Tuesday at 2 PM for account review"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Customer: {query}")
        
        try:
            # Run with context
            result = contextual_agent.run_with_context(query)
            print(f"Agent: {result.final_output}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n4. Context Template Demonstration:")
    
    # Show how instructions are built
    contextual_instructions = contextual_agent.build_contextual_instructions()
    print("Generated contextual instructions:")
    print("-" * 40)
    print(contextual_instructions[:500] + "..." if len(contextual_instructions) > 500 else contextual_instructions)
    print("-" * 40)
    
    return {
        "contextual_agent": contextual_agent,
        "customer_service_agent": customer_service_agent
    }

# =============================================================================
# 📊 3. BUSINESS CONTEXT INTEGRATION
# =============================================================================

class BusinessContextProvider:
    """Provide business-specific context"""
    
    def __init__(self):
        self.business_rules = {}
        self.data_sources = {}
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def register_data_source(self, key: str, source_func: Callable):
        """Register a business data source"""
        self.data_sources[key] = source_func
    
    def add_business_rule(self, rule_name: str, rule_func: Callable):
        """Add business rule"""
        self.business_rules[rule_name] = rule_func
    
    def get_business_data(self, key: str, force_refresh: bool = False, **kwargs) -> Any:
        """Get business data with caching"""
        cache_key = f"{key}_{hash(str(sorted(kwargs.items())))}"
        
        # Check cache
        if not force_refresh and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        # Fetch from source
        if key in self.data_sources:
            try:
                data = self.data_sources[key](**kwargs)
                self.cache[cache_key] = (data, time.time())
                return data
            except Exception as e:
                print(f"Error fetching business data {key}: {e}")
                return None
        
        return None
    
    def evaluate_business_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all business rules against context"""
        rule_results = {}
        
        for rule_name, rule_func in self.business_rules.items():
            try:
                result = rule_func(context)
                rule_results[rule_name] = result
            except Exception as e:
                print(f"Error evaluating rule {rule_name}: {e}")
                rule_results[rule_name] = {"error": str(e)}
        
        return rule_results
    
    def generate_business_context(self, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive business context"""
        context = {}
        
        # Get current business data
        business_data_keys = [
            "current_inventory",
            "pricing_tiers", 
            "active_promotions",
            "system_status",
            "queue_metrics"
        ]
        
        for key in business_data_keys:
            data = self.get_business_data(key, user_context=user_context)
            if data is not None:
                context[key] = data
        
        # Evaluate business rules
        if user_context:
            combined_context = {**context, **user_context}
            rule_results = self.evaluate_business_rules(combined_context)
            context["business_rules"] = rule_results
        
        return context

def business_context_examples():
    """Examples of business context integration"""
    
    print("\n📊 BUSINESS CONTEXT INTEGRATION")
    print("=" * 50)
    
    # Create business context provider
    business_provider = BusinessContextProvider()
    
    print("\n1. Setting Up Business Data Sources:")
    
    # Register business data sources
    def get_current_inventory(user_context: Dict = None):
        """Get current inventory levels"""
        return {
            "laptops": {"available": 45, "reserved": 5, "low_stock_threshold": 10},
            "phones": {"available": 123, "reserved": 15, "low_stock_threshold": 25},
            "tablets": {"available": 67, "reserved": 8, "low_stock_threshold": 15}
        }
    
    def get_pricing_tiers(user_context: Dict = None):
        """Get customer pricing tiers"""
        user_type = user_context.get("user_account_type", "Standard") if user_context else "Standard"
        
        pricing = {
            "Standard": {"discount": 0.0, "shipping": "Standard", "support": "Basic"},
            "Premium": {"discount": 0.10, "shipping": "Express", "support": "Priority"},
            "Enterprise": {"discount": 0.20, "shipping": "White Glove", "support": "Dedicated"}
        }
        
        return pricing.get(user_type, pricing["Standard"])
    
    def get_active_promotions(user_context: Dict = None):
        """Get current active promotions"""
        current_hour = datetime.now().hour
        
        promotions = []
        
        if current_hour < 12:
            promotions.append("Morning Special: 15% off electronics until noon")
        
        if datetime.now().weekday() == 4:  # Friday
            promotions.append("Friday Flash Sale: Free shipping on all orders")
        
        if user_context and user_context.get("user_account_type") == "Premium":
            promotions.append("Premium Member: Exclusive 25% off premium accessories")
        
        return promotions
    
    def get_system_status(user_context: Dict = None):
        """Get system status information"""
        return {
            "payment_processing": "operational",
            "inventory_system": "operational", 
            "shipping_api": "operational",
            "estimated_response_time": "< 2 minutes",
            "last_update": datetime.now().isoformat()
        }
    
    def get_queue_metrics(user_context: Dict = None):
        """Get customer service queue metrics"""
        import random
        return {
            "queue_length": random.randint(0, 8),
            "average_wait_time": f"{random.randint(2, 15)} minutes",
            "agents_available": random.randint(3, 8),
            "peak_hours": "10 AM - 2 PM, 6 PM - 8 PM"
        }
    
    # Register data sources
    data_sources = {
        "current_inventory": get_current_inventory,
        "pricing_tiers": get_pricing_tiers,
        "active_promotions": get_active_promotions,
        "system_status": get_system_status,
        "queue_metrics": get_queue_metrics
    }
    
    for key, func in data_sources.items():
        business_provider.register_data_source(key, func)
        print(f"Registered data source: {key}")
    
    print("\n2. Setting Up Business Rules:")
    
    # Add business rules
    def check_inventory_levels(context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if any products are low in stock"""
        inventory = context.get("current_inventory", {})
        alerts = []
        
        for product, data in inventory.items():
            available = data.get("available", 0)
            threshold = data.get("low_stock_threshold", 0)
            
            if available <= threshold:
                alerts.append(f"{product.title()} is low in stock ({available} available)")
        
        return {
            "has_low_stock": len(alerts) > 0,
            "low_stock_alerts": alerts,
            "total_products_checked": len(inventory)
        }
    
    def determine_shipping_options(context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine available shipping options"""
        pricing_tier = context.get("pricing_tiers", {})
        user_location = context.get("user_location", "")
        
        options = ["Standard (5-7 business days)"]
        
        if pricing_tier.get("shipping") in ["Express", "White Glove"]:
            options.append("Express (2-3 business days)")
        
        if pricing_tier.get("shipping") == "White Glove":
            options.append("White Glove (Next business day + setup)")
        
        if "NY" in user_location or "New York" in user_location:
            options.append("Same Day (NYC area only)")
        
        return {
            "available_options": options,
            "recommended": options[-1],  # Best available option
            "user_tier": pricing_tier.get("shipping", "Standard")
        }
    
    def calculate_final_pricing(context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final pricing with discounts"""
        pricing_tier = context.get("pricing_tiers", {})
        promotions = context.get("active_promotions", [])
        
        base_discount = pricing_tier.get("discount", 0.0)
        additional_discount = 0.0
        
        # Check for promotional discounts
        for promo in promotions:
            if "15%" in promo:
                additional_discount = max(additional_discount, 0.15)
            elif "25%" in promo:
                additional_discount = max(additional_discount, 0.25)
        
        total_discount = base_discount + additional_discount
        total_discount = min(total_discount, 0.5)  # Max 50% discount
        
        return {
            "base_discount": base_discount,
            "promotional_discount": additional_discount,
            "total_discount": total_discount,
            "discount_percentage": f"{total_discount * 100:.1f}%",
            "applicable_promotions": len(promotions)
        }
    
    # Register business rules
    business_rules = {
        "inventory_check": check_inventory_levels,
        "shipping_options": determine_shipping_options,
        "pricing_calculation": calculate_final_pricing
    }
    
    for rule_name, rule_func in business_rules.items():
        business_provider.add_business_rule(rule_name, rule_func)
        print(f"Added business rule: {rule_name}")
    
    print("\n3. Testing Business Context Generation:")
    
    # Test with different user contexts
    test_users = [
        {
            "user_name": "John Doe",
            "user_account_type": "Standard",
            "user_location": "Chicago, IL"
        },
        {
            "user_name": "Jane Smith",
            "user_account_type": "Premium", 
            "user_location": "New York, NY"
        },
        {
            "user_name": "Bob Johnson",
            "user_account_type": "Enterprise",
            "user_location": "Los Angeles, CA"
        }
    ]
    
    for i, user_context in enumerate(test_users, 1):
        print(f"\n--- User {i}: {user_context['user_name']} ({user_context['user_account_type']}) ---")
        
        # Generate business context
        business_context = business_provider.generate_business_context(user_context)
        
        print("Generated Business Context:")
        
        # Display key business data
        if "pricing_tiers" in business_context:
            pricing = business_context["pricing_tiers"]
            print(f"  Pricing Tier: {pricing['discount']*100:.0f}% discount, {pricing['shipping']} shipping")
        
        if "active_promotions" in business_context:
            promotions = business_context["active_promotions"]
            print(f"  Active Promotions: {len(promotions)} available")
            for promo in promotions:
                print(f"    - {promo}")
        
        # Display business rule results
        if "business_rules" in business_context:
            rules = business_context["business_rules"]
            
            if "inventory_check" in rules:
                inventory_result = rules["inventory_check"]
                if inventory_result["has_low_stock"]:
                    print(f"  ⚠️ Low Stock Alerts: {len(inventory_result['low_stock_alerts'])}")
                else:
                    print("  ✅ All products in stock")
            
            if "shipping_options" in rules:
                shipping_result = rules["shipping_options"]
                print(f"  🚚 Shipping Options: {len(shipping_result['available_options'])}")
                print(f"    Recommended: {shipping_result['recommended']}")
            
            if "pricing_calculation" in rules:
                pricing_result = rules["pricing_calculation"]
                print(f"  💰 Total Discount: {pricing_result['discount_percentage']}")
    
    return {
        "business_provider": business_provider,
        "data_sources": data_sources,
        "business_rules": business_rules
    }

# =============================================================================
# 🔄 4. CONTEXT INHERITANCE & COMPOSITION
# =============================================================================

class ContextLayer:
    """A layer of context that can be composed with others"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # Higher priority overrides lower
        self.contexts: Dict[str, ContextData] = {}
        self.validators: List[Callable] = []
        self.transformers: List[Callable] = []
    
    def add_context(self, context: ContextData):
        """Add context to this layer"""
        self.contexts[context.key] = context
    
    def add_validator(self, validator: Callable[[Dict[str, Any]], bool]):
        """Add context validator"""
        self.validators.append(validator)
    
    def add_transformer(self, transformer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add context transformer"""
        self.transformers.append(transformer)
    
    def validate(self, contexts: Dict[str, Any]) -> bool:
        """Validate contexts against all validators"""
        for validator in self.validators:
            try:
                if not validator(contexts):
                    return False
            except Exception as e:
                print(f"Validation error in layer {self.name}: {e}")
                return False
        return True
    
    def transform(self, contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transformers to contexts"""
        result = contexts.copy()
        
        for transformer in self.transformers:
            try:
                result = transformer(result)
            except Exception as e:
                print(f"Transform error in layer {self.name}: {e}")
        
        return result
    
    def get_contexts(self) -> Dict[str, Any]:
        """Get all contexts from this layer"""
        valid_contexts = {}
        
        for key, context in self.contexts.items():
            if not context.is_expired():
                valid_contexts[key] = context.value
        
        return valid_contexts

class CompositeContextManager:
    """Manage multiple context layers with inheritance and composition"""
    
    def __init__(self):
        self.layers: List[ContextLayer] = []
    
    def add_layer(self, layer: ContextLayer):
        """Add context layer"""
        self.layers.append(layer)
        # Sort by priority (higher priority first)
        self.layers.sort(key=lambda x: x.priority, reverse=True)
    
    def remove_layer(self, layer_name: str):
        """Remove context layer by name"""
        self.layers = [layer for layer in self.layers if layer.name != layer_name]
    
    def get_layer(self, layer_name: str) -> Optional[ContextLayer]:
        """Get layer by name"""
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        return None
    
    def compose_contexts(self) -> Dict[str, Any]:
        """Compose contexts from all layers"""
        composed = {}
        
        # Process layers in reverse priority order (lowest to highest)
        for layer in reversed(self.layers):
            layer_contexts = layer.get_contexts()
            
            # Apply layer transformations
            layer_contexts = layer.transform(layer_contexts)
            
            # Merge with composed contexts (higher priority overwrites)
            composed.update(layer_contexts)
        
        # Final validation across all layers
        for layer in self.layers:
            if not layer.validate(composed):
                print(f"Context validation failed for layer: {layer.name}")
        
        return composed
    
    def get_context_lineage(self, key: str) -> List[Dict[str, Any]]:
        """Get the lineage of a context key across layers"""
        lineage = []
        
        for layer in self.layers:
            if key in layer.contexts:
                context = layer.contexts[key]
                lineage.append({
                    "layer": layer.name,
                    "priority": layer.priority,
                    "value": context.value,
                    "timestamp": context.timestamp,
                    "metadata": context.metadata
                })
        
        return lineage

def context_composition_examples():
    """Examples of context inheritance and composition"""
    
    print("\n🔄 CONTEXT INHERITANCE & COMPOSITION")
    print("=" * 50)
    
    # Create composite context manager
    composite_manager = CompositeContextManager()
    
    print("\n1. Creating Context Layers:")
    
    # 1. Base/System Layer (Lowest Priority)
    base_layer = ContextLayer("base_system", priority=0)
    
    base_contexts = [
        ContextData(ContextType.STATIC, "app_name", "Customer Service Portal"),
        ContextData(ContextType.STATIC, "app_version", "2.1.0"),
        ContextData(ContextType.STATIC, "max_response_length", 500),
        ContextData(ContextType.STATIC, "default_language", "en"),
        ContextData(ContextType.STATIC, "support_level", "basic")
    ]
    
    for context in base_contexts:
        base_layer.add_context(context)
    
    print(f"Created base layer with {len(base_contexts)} contexts")
    
    # 2. Organization Layer (Medium Priority)
    org_layer = ContextLayer("organization", priority=50)
    
    org_contexts = [
        ContextData(ContextType.BUSINESS, "company_name", "TechCorp Solutions"),
        ContextData(ContextType.BUSINESS, "support_level", "professional"),  # Override base
        ContextData(ContextType.BUSINESS, "business_hours", "8:00 AM - 10:00 PM EST"),
        ContextData(ContextType.BUSINESS, "escalation_threshold", 3),
        ContextData(ContextType.BUSINESS, "customer_satisfaction_target", 0.95)
    ]
    
    for context in org_contexts:
        org_layer.add_context(context)
    
    print(f"Created organization layer with {len(org_contexts)} contexts")
    
    # 3. User Session Layer (High Priority)
    session_layer = ContextLayer("user_session", priority=100)
    
    session_contexts = [
        ContextData(ContextType.USER, "user_id", "CUST789"),
        ContextData(ContextType.USER, "user_name", "Alice Cooper"),
        ContextData(ContextType.USER, "support_level", "premium"),  # Override org
        ContextData(ContextType.SESSION, "session_id", "SESS_ABC123"),
        ContextData(ContextType.SESSION, "conversation_count", 3),
        ContextData(
            ContextType.SESSION, 
            "conversation_history", 
            ["Greeted customer", "Resolved billing question", "Current: Technical support"],
            expires_at=datetime.now() + timedelta(hours=2)
        )
    ]
    
    for context in session_contexts:
        session_layer.add_context(context)
    
    print(f"Created user session layer with {len(session_contexts)} contexts")
    
    print("\n2. Adding Context Validators and Transformers:")
    
    # Add validators
    def validate_user_session(contexts: Dict[str, Any]) -> bool:
        """Validate user session contexts"""
        required_fields = ["user_id", "user_name", "session_id"]
        for field in required_fields:
            if field not in contexts:
                print(f"Missing required field: {field}")
                return False
        return True
    
    def validate_support_level(contexts: Dict[str, Any]) -> bool:
        """Validate support level is valid"""
        valid_levels = ["basic", "professional", "premium", "enterprise"]
        support_level = contexts.get("support_level", "").lower()
        if support_level not in valid_levels:
            print(f"Invalid support level: {support_level}")
            return False
        return True
    
    session_layer.add_validator(validate_user_session)
    org_layer.add_validator(validate_support_level)
    
    # Add transformers
    def transform_user_greeting(contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Transform user name for greeting"""
        result = contexts.copy()
        
        user_name = contexts.get("user_name", "")
        support_level = contexts.get("support_level", "basic")
        
        if user_name:
            if support_level in ["premium", "enterprise"]:
                result["greeting"] = f"Good day, {user_name}. Thank you for being a valued {support_level} customer."
            else:
                result["greeting"] = f"Hello, {user_name}! How can we help you today?"
        
        return result
    
    def transform_response_settings(contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response settings based on support level"""
        result = contexts.copy()
        
        support_level = contexts.get("support_level", "basic")
        base_max_length = contexts.get("max_response_length", 500)
        
        # Adjust response length based on support level
        if support_level == "enterprise":
            result["max_response_length"] = base_max_length * 2
        elif support_level == "premium":
            result["max_response_length"] = int(base_max_length * 1.5)
        
        # Add response tone guidance
        tone_mapping = {
            "basic": "friendly and helpful",
            "professional": "professional and detailed",
            "premium": "personalized and comprehensive",
            "enterprise": "formal and thorough"
        }
        
        result["response_tone"] = tone_mapping.get(support_level, "friendly and helpful")
        
        return result
    
    session_layer.add_transformer(transform_user_greeting)
    org_layer.add_transformer(transform_response_settings)
    
    print("Added validators and transformers to layers")
    
    print("\n3. Composing Context Layers:")
    
    # Add layers to composite manager
    composite_manager.add_layer(base_layer)
    composite_manager.add_layer(org_layer)
    composite_manager.add_layer(session_layer)
    
    # Compose final context
    final_contexts = composite_manager.compose_contexts()
    
    print("Final Composed Context:")
    for key, value in sorted(final_contexts.items()):
        print(f"  {key}: {value}")
    
    print("\n4. Context Lineage Tracking:")
    
    # Show lineage for overridden values
    lineage_keys = ["support_level", "max_response_length", "greeting"]
    
    for key in lineage_keys:
        lineage = composite_manager.get_context_lineage(key)
        if lineage:
            print(f"\nLineage for '{key}':")
            for entry in lineage:
                print(f"  Layer: {entry['layer']} (priority {entry['priority']}) = {entry['value']}")
    
    print("\n5. Dynamic Layer Management:")
    
    # Add temporary override layer
    override_layer = ContextLayer("temporary_override", priority=200)
    override_context = ContextData(
        ContextType.DYNAMIC, 
        "special_message", 
        "🎉 Welcome! We're celebrating our 10th anniversary with special offers!",
        expires_at=datetime.now() + timedelta(minutes=30)
    )
    override_layer.add_context(override_context)
    
    composite_manager.add_layer(override_layer)
    print("Added temporary override layer")
    
    # Recompose with override
    updated_contexts = composite_manager.compose_contexts()
    if "special_message" in updated_contexts:
        print(f"Special message active: {updated_contexts['special_message']}")
    
    # Remove override layer
    composite_manager.remove_layer("temporary_override")
    print("Removed temporary override layer")
    
    return {
        "composite_manager": composite_manager,
        "base_layer": base_layer,
        "org_layer": org_layer,
        "session_layer": session_layer
    }

# =============================================================================
# 🎯 5. ADVANCED CONTEXT PATTERNS
# =============================================================================

class ContextMiddleware(ABC):
    """Abstract base for context middleware"""
    
    @abstractmethod
    def process(self, contexts: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process contexts with middleware logic"""
        pass

class SecurityContextMiddleware(ContextMiddleware):
    """Security-focused context middleware"""
    
    def __init__(self):
        self.sensitive_keys = ["password", "token", "secret", "key", "credit_card"]
        self.permission_rules = {}
    
    def add_permission_rule(self, role: str, allowed_contexts: List[str]):
        """Add role-based context access rule"""
        self.permission_rules[role] = allowed_contexts
    
    def process(self, contexts: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security filtering to contexts"""
        user_role = request_data.get("user_role", "guest")
        
        # Apply role-based filtering
        if user_role in self.permission_rules:
            allowed = set(self.permission_rules[user_role])
            contexts = {k: v for k, v in contexts.items() if k in allowed or k.startswith("public_")}
        
        # Remove sensitive data
        filtered_contexts = {}
        for key, value in contexts.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_keys):
                filtered_contexts[key] = "[REDACTED]"
            else:
                filtered_contexts[key] = value
        
        # Add security metadata
        filtered_contexts["security_applied"] = True
        filtered_contexts["user_role"] = user_role
        filtered_contexts["filtered_keys"] = len(contexts) - len(filtered_contexts) + 2  # +2 for added keys
        
        return filtered_contexts

class PerformanceContextMiddleware(ContextMiddleware):
    """Performance optimization middleware"""
    
    def __init__(self):
        self.size_limits = {
            "max_context_size": 10000,  # Max characters in context values
            "max_context_count": 50,    # Max number of context items
            "max_string_length": 1000   # Max length for string values
        }
    
    def process(self, contexts: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply performance optimizations"""
        
        # Limit number of contexts
        if len(contexts) > self.size_limits["max_context_count"]:
            # Keep most recent/important contexts
            sorted_contexts = sorted(contexts.items(), key=lambda x: x[0])
            contexts = dict(sorted_contexts[:self.size_limits["max_context_count"]])
        
        # Truncate large values
        optimized_contexts = {}
        total_size = 0
        
        for key, value in contexts.items():
            if isinstance(value, str) and len(value) > self.size_limits["max_string_length"]:
                truncated_value = value[:self.size_limits["max_string_length"]] + "..."
                optimized_contexts[key] = truncated_value
            else:
                optimized_contexts[key] = value
            
            # Track total size
            value_size = len(str(value))
            total_size += value_size
            
            # Stop if we exceed size limit
            if total_size > self.size_limits["max_context_size"]:
                break
        
        # Add performance metadata
        optimized_contexts["performance_optimized"] = True
        optimized_contexts["total_context_size"] = total_size
        optimized_contexts["context_count"] = len(optimized_contexts)
        
        return optimized_contexts

class ContextPipeline:
    """Pipeline for processing contexts through middleware"""
    
    def __init__(self):
        self.middleware: List[ContextMiddleware] = []
        self.metrics = {
            "total_requests": 0,
            "total_processing_time": 0,
            "middleware_execution_times": {}
        }
    
    def add_middleware(self, middleware: ContextMiddleware):
        """Add middleware to pipeline"""
        self.middleware.append(middleware)
        middleware_name = middleware.__class__.__name__
        self.metrics["middleware_execution_times"][middleware_name] = []
    
    def process(self, contexts: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process contexts through middleware pipeline"""
        start_time = time.time()
        
        result = contexts.copy()
        
        # Process through each middleware
        for middleware in self.middleware:
            middleware_start = time.time()
            
            try:
                result = middleware.process(result, request_data)
            except Exception as e:
                print(f"Middleware error ({middleware.__class__.__name__}): {e}")
            
            # Track middleware execution time
            middleware_time = time.time() - middleware_start
            middleware_name = middleware.__class__.__name__
            self.metrics["middleware_execution_times"][middleware_name].append(middleware_time)
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += total_time
        
        # Add pipeline metadata
        result["pipeline_processed"] = True
        result["processing_time_ms"] = round(total_time * 1000, 2)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        metrics = self.metrics.copy()
        
        if metrics["total_requests"] > 0:
            metrics["average_processing_time"] = metrics["total_processing_time"] / metrics["total_requests"]
        
        # Calculate middleware averages
        for middleware_name, times in metrics["middleware_execution_times"].items():
            if times:
                metrics[f"{middleware_name}_avg_time"] = sum(times) / len(times)
                metrics[f"{middleware_name}_executions"] = len(times)
        
        return metrics

def advanced_context_examples():
    """Examples of advanced context patterns"""
    
    print("\n🎯 ADVANCED CONTEXT PATTERNS")
    print("=" * 50)
    
    print("\n1. Setting Up Context Pipeline:")
    
    # Create context pipeline
    pipeline = ContextPipeline()
    
    # Create middleware
    security_middleware = SecurityContextMiddleware()
    performance_middleware = PerformanceContextMiddleware()
    
    # Configure security rules
    security_middleware.add_permission_rule("guest", ["public_info", "general_help"])
    security_middleware.add_permission_rule("user", ["public_info", "general_help", "user_profile", "order_history"])
    security_middleware.add_permission_rule("admin", ["public_info", "general_help", "user_profile", "order_history", "system_status", "admin_tools"])
    
    # Add middleware to pipeline
    pipeline.add_middleware(security_middleware)
    pipeline.add_middleware(performance_middleware)
    
    print("Created context pipeline with security and performance middleware")
    
    print("\n2. Testing Context Processing:")
    
    # Create test contexts
    test_contexts = {
        "public_info": "Company hours: 9 AM - 6 PM",
        "general_help": "For support, contact us at support@company.com",
        "user_profile": "User: Alice, Account: Premium, Location: NYC",
        "order_history": "Last 5 orders: #1001, #1002, #1003, #1004, #1005",
        "system_status": "All systems operational, 99.9% uptime",
        "admin_tools": "Admin panel access enabled",
        "sensitive_password": "super_secret_password_123",
        "api_secret": "sk-abcd1234567890",
        "large_data": "X" * 2000,  # Large string to test truncation
    }
    
    # Test with different user roles
    test_requests = [
        {"user_role": "guest", "user_id": "guest_001"},
        {"user_role": "user", "user_id": "user_123"},
        {"user_role": "admin", "user_id": "admin_456"}
    ]
    
    for request_data in test_requests:
        user_role = request_data["user_role"]
        print(f"\n--- Processing for role: {user_role} ---")
        
        # Process through pipeline
        processed_contexts = pipeline.process(test_contexts, request_data)
        
        print(f"Original contexts: {len(test_contexts)} items")
        print(f"Processed contexts: {len(processed_contexts)} items")
        
        # Show key results
        accessible_contexts = [k for k in processed_contexts.keys() 
                             if not k.startswith(('security_', 'performance_', 'pipeline_', 'processing_'))]
        print(f"Accessible contexts: {accessible_contexts}")
        
        # Show security and performance info
        if "security_applied" in processed_contexts:
            print(f"Security applied: filtered {processed_contexts.get('filtered_keys', 0)} keys")
        
        if "performance_optimized" in processed_contexts:
            print(f"Performance optimized: {processed_contexts.get('total_context_size', 0)} total characters")
        
        if "processing_time_ms" in processed_contexts:
            print(f"Processing time: {processed_contexts['processing_time_ms']} ms")
    
    print("\n3. Pipeline Performance Metrics:")
    
    # Get pipeline metrics
    metrics = pipeline.get_metrics()
    
    print(f"Total requests processed: {metrics['total_requests']}")
    print(f"Average processing time: {metrics.get('average_processing_time', 0):.4f}s")
    
    for key, value in metrics.items():
        if key.endswith('_avg_time'):
            middleware_name = key.replace('_avg_time', '')
            executions = metrics.get(f"{middleware_name}_executions", 0)
            print(f"{middleware_name}: {value:.4f}s avg ({executions} executions)")
    
    print("\n4. Advanced Context Features:")
    
    # Context caching example
    class CachedContextProvider:
        def __init__(self, ttl_seconds: int = 300):
            self.cache = {}
            self.ttl = ttl_seconds
        
        def get_cached_context(self, key: str, generator: Callable) -> Any:
            """Get context with caching"""
            now = time.time()
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                if now - timestamp < self.ttl:
                    return value
            
            # Generate new value
            value = generator()
            self.cache[key] = (value, now)
            return value
        
        def clear_cache(self):
            """Clear all cached contexts"""
            self.cache.clear()
    
    # Test cached context
    cached_provider = CachedContextProvider(ttl_seconds=5)
    
    def expensive_context_generator():
        """Simulate expensive context generation"""
        time.sleep(0.1)  # Simulate delay
        return f"Generated at {datetime.now().strftime('%H:%M:%S')}"
    
    print("Testing cached context:")
    
    # First call - should generate
    start_time = time.time()
    context1 = cached_provider.get_cached_context("expensive_data", expensive_context_generator)
    time1 = time.time() - start_time
    print(f"First call (generated): {context1} - took {time1:.3f}s")
    
    # Second call - should use cache
    start_time = time.time()
    context2 = cached_provider.get_cached_context("expensive_data", expensive_context_generator)
    time2 = time.time() - start_time
    print(f"Second call (cached): {context2} - took {time2:.3f}s")
    
    return {
        "pipeline": pipeline,
        "security_middleware": security_middleware,
        "performance_middleware": performance_middleware,
        "cached_provider": cached_provider
    }

# =============================================================================
# 💡 6. CONTEXT BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 CONTEXT BEST PRACTICES

1. 🏗️ CONTEXT ARCHITECTURE:

   Pattern A - Layered Context:
   ```python
   # Base layer (system defaults)
   # Organization layer (business rules)
   # User layer (personalization)
   # Session layer (temporary state)
   ```
   
   Pattern B - Contextual Functions:
   ```python
   @function_tool
   def get_user_context(user_id: str) -> str:
       return f"Current context for {user_id}"
   ```
   
   Pattern C - Dynamic Injection:
   ```python
   instructions = base_instructions + context_template.format(**contexts)
   ```

2. 🔄 CONTEXT LIFECYCLE:

   ✅ Initialize contexts early in request cycle
   ✅ Validate context data integrity
   ✅ Transform contexts for specific needs
   ✅ Cache expensive context generation
   ✅ Clean up expired or stale contexts

3. 🚦 PERFORMANCE OPTIMIZATION:

   ✅ Use caching for expensive context generation
   ✅ Limit context size and count
   ✅ Implement lazy loading for context data
   ✅ Use async operations for external data
   ✅ Monitor context processing performance

4. 🔒 SECURITY CONSIDERATIONS:

   ✅ Validate all context data sources
   ✅ Implement role-based context filtering
   ✅ Sanitize sensitive information
   ✅ Use secure transport for context APIs
   ✅ Audit context access patterns

5. 📊 CONTEXT COMPOSITION:

   ✅ Use priority-based layer composition
   ✅ Implement validation across layers
   ✅ Support context inheritance
   ✅ Enable dynamic layer management
   ✅ Track context lineage for debugging

6. 🎯 BUSINESS INTEGRATION:

   ✅ Integrate real-time business data
   ✅ Apply business rules to contexts
   ✅ Support multi-tenant context isolation
   ✅ Enable A/B testing through context
   ✅ Monitor business context effectiveness

🚨 COMMON CONTEXT PITFALLS:

❌ DON'T include sensitive data in context without filtering
❌ DON'T ignore context size and performance impact
❌ DON'T hardcode context values - use dynamic sources
❌ DON'T forget to validate external context data
❌ DON'T mix business logic directly in context generation
❌ DON'T assume context data is always available

✅ CONTEXT SUCCESS CHECKLIST:

☐ Clear context layer architecture
☐ Dynamic context generation functions
☐ Proper validation and sanitization
☐ Performance monitoring and optimization
☐ Security filtering and access control
☐ Business rule integration
☐ Error handling and fallbacks
☐ Context caching and expiration
☐ Multi-tenant support if needed
☐ Comprehensive testing coverage

📈 CONTEXT OPTIMIZATION STRATEGIES:

For high-performance context handling:
- Use connection pooling for external data sources
- Implement efficient caching with TTL
- Batch context generation operations
- Use async/await for non-blocking context fetching
- Monitor context processing latency
- Implement circuit breakers for external services
- Use content delivery networks for static context
- Optimize database queries for context data
- Implement proper error handling and fallbacks
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

def run_all_context_examples():
    """Run all context examples comprehensively"""
    
    print("🧠 OPENAI AGENTS SDK - COMPLETE CONTEXT DEMONSTRATION")
    print("=" * 70)
    
    # 1. Basic Context Patterns
    basic_results = basic_context_examples()
    
    # 2. Agent Context Integration
    agent_results = agent_context_examples()
    
    # 3. Business Context Integration
    business_results = business_context_examples()
    
    # 4. Context Inheritance & Composition
    composition_results = context_composition_examples()
    
    # 5. Advanced Context Patterns
    advanced_results = advanced_context_examples()
    
    print("\n✅ All context examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Context provides dynamic, personalized agent behavior")
    print("- Use layered architecture for context composition")
    print("- Implement proper security and performance optimizations") 
    print("- Integrate real-time business data through context functions")
    print("- Monitor context processing performance and effectiveness")
    print("- Always validate and sanitize context data sources")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🧠 OpenAI Agents SDK - Complete Context Template")
    print("This template demonstrates all Context management patterns and features.")
    
    # Run all context examples
    run_all_context_examples()
    
    print("\n✅ Context template demonstration complete!")
    print("💡 Use this template as reference for all your context management needs.")
