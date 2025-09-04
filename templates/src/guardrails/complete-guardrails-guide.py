"""
🛡️ OPENAI AGENTS SDK - COMPLETE GUARDRAILS GUIDE & TEMPLATE

This template covers everything about Guardrails in the OpenAI Agents SDK:
- Input validation and sanitization
- Output filtering and safety
- Content moderation and compliance
- Rate limiting and resource protection
- Custom guardrail implementations
- Security patterns and best practices
- Business rule enforcement
- Error handling and recovery

📚 Based on: https://openai.github.io/openai-agents-python/
"""

import os
import re
import json
import time
import hashlib
import asyncio
from typing import List, Dict, Optional, Callable, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from agents import Agent, Runner, function_tool

# =============================================================================
# 📖 UNDERSTANDING GUARDRAILS IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT ARE GUARDRAILS?

Guardrails are protective mechanisms that ensure:
1. ✅ SAFETY: Prevent harmful or inappropriate content
2. ✅ COMPLIANCE: Enforce business rules and regulations  
3. ✅ SECURITY: Protect against malicious inputs
4. ✅ QUALITY: Maintain response standards
5. ✅ RELIABILITY: Prevent system abuse and overload

KEY CONCEPTS:

1. INPUT GUARDRAILS:
   - Validate user input before processing
   - Sanitize potentially dangerous content
   - Check permissions and authorization
   - Rate limiting and abuse prevention

2. OUTPUT GUARDRAILS:
   - Filter AI responses for appropriateness
   - Ensure compliance with policies
   - Prevent information disclosure
   - Maintain brand voice and standards

3. PROCESS GUARDRAILS:
   - Monitor agent behavior during execution
   - Prevent infinite loops and resource exhaustion
   - Enforce business logic constraints
   - Log and audit agent activities

4. TYPES OF GUARDRAILS:
   - Content-based: Check for inappropriate content
   - Permission-based: Verify user authorization
   - Rate-based: Limit usage frequency
   - Business-rule-based: Enforce domain constraints
   - Security-based: Prevent attacks and exploitation
"""

# =============================================================================
# ⚠️ 1. INPUT VALIDATION GUARDRAILS
# =============================================================================

class InputValidationError(Exception):
    """Exception raised when input validation fails"""
    pass

class ContentFilter:
    """Content filtering and validation"""
    
    # Prohibited content patterns
    PROHIBITED_PATTERNS = [
        r'\b(?:password|passwd|pwd)\s*[:=]\s*\S+',  # Password patterns
        r'\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*\S+',  # API key patterns
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card patterns
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
    ]
    
    # Spam/abuse indicators
    SPAM_PATTERNS = [
        r'(.)\1{10,}',  # Repeated characters
        r'\b(?:buy|sale|offer|discount|free|win|winner)\b.*\$',  # Sales spam
        r'(?:http://|https://|www\.)\S+(?:\s+(?:http://|https://|www\.)\S+){3,}',  # Link spam
    ]
    
    # Inappropriate content (basic examples)
    INAPPROPRIATE_PATTERNS = [
        r'\b(?:hate|violent|illegal)\b.*\b(?:content|material)\b',
        r'\bhow\s+to\s+(?:hack|crack|bypass|exploit)\b',
    ]
    
    @classmethod
    def validate_input(cls, text: str) -> Dict[str, Any]:
        """Comprehensive input validation"""
        issues = []
        
        # Check for prohibited content
        for pattern in cls.PROHIBITED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append({
                    'type': 'prohibited_content',
                    'severity': 'high',
                    'message': 'Input contains potentially sensitive information'
                })
                break
        
        # Check for spam patterns
        for pattern in cls.SPAM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append({
                    'type': 'spam_content',
                    'severity': 'medium',
                    'message': 'Input appears to be spam or promotional content'
                })
                break
        
        # Check for inappropriate requests
        for pattern in cls.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append({
                    'type': 'inappropriate_content',
                    'severity': 'high',
                    'message': 'Input contains inappropriate or harmful requests'
                })
                break
        
        # Check input length
        if len(text) > 10000:  # 10K character limit
            issues.append({
                'type': 'length_exceeded',
                'severity': 'medium',
                'message': 'Input exceeds maximum length limit'
            })
        
        # Check for suspicious patterns
        if text.count('<script>') > 0 or text.count('javascript:') > 0:
            issues.append({
                'type': 'injection_attempt',
                'severity': 'high',
                'message': 'Input contains potential injection attempts'
            })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'processed_text': cls._sanitize_input(text) if len(issues) == 0 else text
        }
    
    @classmethod
    def _sanitize_input(cls, text: str) -> str:
        """Sanitize input text"""
        # Remove potentially dangerous HTML/JS
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

def input_validation_guardrail_examples():
    """Examples of input validation guardrails"""
    
    print("⚠️ INPUT VALIDATION GUARDRAILS")
    print("=" * 50)
    
    def validate_input_tool(input_text: str) -> str:
        """Validate user input before processing"""
        validation_result = ContentFilter.validate_input(input_text)
        
        if not validation_result['valid']:
            high_severity_issues = [
                issue for issue in validation_result['issues'] 
                if issue['severity'] == 'high'
            ]
            
            if high_severity_issues:
                raise InputValidationError(
                    f"Input validation failed: {high_severity_issues[0]['message']}"
                )
        
        return validation_result['processed_text']
    
    @function_tool
    def process_user_request(request: str) -> str:
        """Process user request with input validation"""
        try:
            # Validate input first
            clean_request = validate_input_tool(request)
            
            # Process the clean request
            return f"Processing your request: {clean_request[:100]}..."
            
        except InputValidationError as e:
            return f"❌ Request blocked: {str(e)}"
    
    # Create agent with input validation
    validated_agent = Agent(
        name="ValidatedAgent",
        instructions="""
        You process user requests safely by validating all inputs.
        You refuse to process requests that fail validation checks.
        Always prioritize user safety and data protection.
        """,
        tools=[process_user_request]
    )
    
    # Test input validation
    test_inputs = [
        "What's the weather like today?",  # Safe input
        "My password is: secret123",  # Prohibited content
        "BUY NOW!!! GET FREE MONEY!!! $$$$$$$",  # Spam content
        "How to hack into systems",  # Inappropriate content
        "A" * 15000,  # Too long input
        "<script>alert('xss')</script>What's 2+2?",  # Injection attempt
    ]
    
    print("\n1. Testing Input Validation:")
    for i, test_input in enumerate(test_inputs, 1):
        try:
            result = Runner.run_sync(validated_agent, test_input)
            print(f"{i}. Input: {test_input[:50]}{'...' if len(test_input) > 50 else ''}")
            print(f"   Result: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return validated_agent

# =============================================================================
# 🔒 2. PERMISSION & AUTHORIZATION GUARDRAILS
# =============================================================================

class UserPermissions:
    """User permission management"""
    
    def __init__(self):
        self.user_roles = {
            'admin': {'read', 'write', 'delete', 'admin', 'sensitive'},
            'manager': {'read', 'write', 'sensitive'},
            'user': {'read', 'write'},
            'guest': {'read'},
            'anonymous': set()
        }
        
        self.user_assignments = {
            'alice@company.com': 'admin',
            'bob@company.com': 'manager', 
            'charlie@company.com': 'user',
            'guest_user': 'guest'
        }
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get permissions for a user"""
        role = self.user_assignments.get(user_id, 'anonymous')
        return self.user_roles.get(role, set())
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions
    
    def require_permission(self, user_id: str, permission: str) -> bool:
        """Require permission or raise exception"""
        if not self.has_permission(user_id, permission):
            raise PermissionError(
                f"User {user_id} does not have required permission: {permission}"
            )
        return True

class PermissionError(Exception):
    """Exception raised when user lacks required permissions"""
    pass

def authorization_guardrail_examples():
    """Examples of authorization and permission guardrails"""
    
    print("\n🔒 AUTHORIZATION & PERMISSION GUARDRAILS")
    print("=" * 50)
    
    permissions = UserPermissions()
    
    @function_tool
    def read_document(document_id: str, user_id: str = "guest_user") -> str:
        """Read a document with permission checking"""
        try:
            permissions.require_permission(user_id, 'read')
            return f"📄 Document {document_id}: This is the document content."
        except PermissionError as e:
            return f"❌ Access denied: {str(e)}"
    
    @function_tool  
    def create_document(title: str, content: str, user_id: str = "guest_user") -> str:
        """Create a document with write permission checking"""
        try:
            permissions.require_permission(user_id, 'write')
            return f"✅ Created document '{title}' successfully."
        except PermissionError as e:
            return f"❌ Access denied: {str(e)}"
    
    @function_tool
    def delete_document(document_id: str, user_id: str = "guest_user") -> str:
        """Delete a document with delete permission checking"""
        try:
            permissions.require_permission(user_id, 'delete')
            return f"🗑️ Document {document_id} deleted successfully."
        except PermissionError as e:
            return f"❌ Access denied: {str(e)}"
    
    @function_tool
    def access_sensitive_data(query: str, user_id: str = "guest_user") -> str:
        """Access sensitive data with sensitive permission checking"""
        try:
            permissions.require_permission(user_id, 'sensitive')
            return f"🔐 Sensitive data for query '{query}': [CLASSIFIED INFORMATION]"
        except PermissionError as e:
            return f"❌ Access denied: {str(e)}"
    
    # Create agent with permission-based tools
    authorized_agent = Agent(
        name="AuthorizedAgent",
        instructions="""
        You are a document management agent that enforces strict permissions.
        Always check user permissions before performing any actions.
        Clearly explain permission requirements when access is denied.
        """,
        tools=[read_document, create_document, delete_document, access_sensitive_data]
    )
    
    # Test authorization with different users
    test_scenarios = [
        # Admin user (all permissions)
        ("alice@company.com", "Please delete document DOC123"),
        
        # Manager user (no delete permission)
        ("bob@company.com", "Please delete document DOC123"),
        
        # Regular user (read/write only)
        ("charlie@company.com", "Create a new document called 'Meeting Notes'"),
        
        # Guest user (read only)
        ("guest_user", "Show me document DOC456"),
        
        # Guest trying sensitive access
        ("guest_user", "Show me sensitive financial data"),
    ]
    
    print("\n2. Testing Authorization Guardrails:")
    for i, (user_id, request) in enumerate(test_scenarios, 1):
        try:
            # Add user context to the request
            context_request = f"[User: {user_id}] {request}"
            result = Runner.run_sync(authorized_agent, context_request, context={'user_id': user_id})
            print(f"{i}. User: {user_id}")
            print(f"   Request: {request}")
            print(f"   Result: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return authorized_agent

# =============================================================================
# 🚦 3. RATE LIMITING GUARDRAILS
# =============================================================================

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self):
        self.request_counts = defaultdict(deque)
        self.limits = {
            'per_minute': 60,
            'per_hour': 1000,
            'per_day': 10000
        }
    
    def is_rate_limited(self, user_id: str, window: str = 'per_minute') -> bool:
        """Check if user is rate limited"""
        now = datetime.now()
        user_requests = self.request_counts[f"{user_id}_{window}"]
        
        # Define time windows
        windows = {
            'per_minute': timedelta(minutes=1),
            'per_hour': timedelta(hours=1),
            'per_day': timedelta(days=1)
        }
        
        if window not in windows:
            return False
        
        cutoff_time = now - windows[window]
        
        # Remove old requests
        while user_requests and user_requests[0] < cutoff_time:
            user_requests.popleft()
        
        # Check if limit exceeded
        return len(user_requests) >= self.limits.get(window, float('inf'))
    
    def record_request(self, user_id: str) -> Dict[str, bool]:
        """Record a request and return rate limit status"""
        now = datetime.now()
        status = {}
        
        # Check all windows
        for window in ['per_minute', 'per_hour', 'per_day']:
            key = f"{user_id}_{window}"
            
            # Check current status
            is_limited = self.is_rate_limited(user_id, window)
            status[window] = is_limited
            
            # Record request if not limited
            if not is_limited:
                self.request_counts[key].append(now)
        
        return status
    
    def get_reset_times(self, user_id: str) -> Dict[str, datetime]:
        """Get reset times for rate limits"""
        now = datetime.now()
        reset_times = {}
        
        windows = {
            'per_minute': timedelta(minutes=1),
            'per_hour': timedelta(hours=1),
            'per_day': timedelta(days=1)
        }
        
        for window, duration in windows.items():
            user_requests = self.request_counts[f"{user_id}_{window}"]
            if user_requests:
                # Reset time is when the oldest request expires
                oldest_request = user_requests[0]
                reset_times[window] = oldest_request + duration
            else:
                reset_times[window] = now
        
        return reset_times

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass

def rate_limiting_guardrail_examples():
    """Examples of rate limiting guardrails"""
    
    print("\n🚦 RATE LIMITING GUARDRAILS")
    print("=" * 50)
    
    rate_limiter = RateLimiter()
    
    # Override limits for testing
    rate_limiter.limits = {
        'per_minute': 3,  # Only 3 requests per minute for testing
        'per_hour': 50,
        'per_day': 500
    }
    
    def check_rate_limit(user_id: str) -> bool:
        """Check and record rate limit"""
        status = rate_limiter.record_request(user_id)
        
        # Check if any window is exceeded
        for window, limited in status.items():
            if limited:
                reset_times = rate_limiter.get_reset_times(user_id)
                reset_time = reset_times[window]
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {window}. "
                    f"Resets at {reset_time.strftime('%H:%M:%S')}"
                )
        
        return True
    
    @function_tool
    def api_call_simulation(query: str, user_id: str = "default_user") -> str:
        """Simulate an API call with rate limiting"""
        try:
            check_rate_limit(user_id)
            # Simulate processing
            return f"✅ API Response: Processed query '{query}' successfully"
        except RateLimitExceeded as e:
            return f"🚦 Rate limit exceeded: {str(e)}"
    
    @function_tool
    def expensive_operation(task: str, user_id: str = "default_user") -> str:
        """Simulate an expensive operation with stricter rate limiting"""
        try:
            # Check rate limit
            check_rate_limit(user_id)
            
            # Additional check for expensive operations
            minute_requests = rate_limiter.request_counts[f"{user_id}_per_minute"]
            if len(minute_requests) >= 2:  # Only 2 expensive ops per minute
                raise RateLimitExceeded(
                    "Expensive operation limit exceeded (max 2 per minute)"
                )
            
            return f"⚡ Expensive operation '{task}' completed successfully"
        except RateLimitExceeded as e:
            return f"🚦 Operation blocked: {str(e)}"
    
    # Create agent with rate limiting
    rate_limited_agent = Agent(
        name="RateLimitedAgent",
        instructions="""
        You are an API agent that enforces rate limits to prevent abuse.
        You inform users when they hit rate limits and suggest when to retry.
        Always be polite when declining requests due to rate limits.
        """,
        tools=[api_call_simulation, expensive_operation]
    )
    
    # Test rate limiting
    print("\n3. Testing Rate Limiting Guardrails:")
    
    # Test with same user (should hit rate limit)
    test_user = "test_user_123"
    
    for i in range(6):  # Try 6 requests (limit is 3)
        try:
            request = f"API request #{i+1}"
            result = Runner.run_sync(
                rate_limited_agent, 
                request,
                context={'user_id': test_user}
            )
            print(f"{i+1}. Request: {request}")
            print(f"   Result: {result.final_output}")
            print()
            
            # Small delay to simulate real usage
            time.sleep(0.1)
            
        except Exception as e:
            print(f"{i+1}. Error: {e}")
    
    return rate_limited_agent

# =============================================================================
# 📊 4. CONTENT MODERATION GUARDRAILS
# =============================================================================

class ContentModerator:
    """Content moderation and safety checking"""
    
    # Content categories
    HARMFUL_CONTENT = {
        'violence': [
            r'\b(?:kill|murder|assault|attack|harm|hurt|violence|violent)\b',
            r'\b(?:weapon|gun|knife|bomb|explosive)\b'
        ],
        'hate_speech': [
            r'\b(?:hate|racist|discrimination|bigot|prejudice)\b',
            r'\b(?:supremacist|nazi|terrorist)\b'
        ],
        'illegal': [
            r'\b(?:illegal|crime|criminal|steal|theft|fraud|scam)\b',
            r'\b(?:drug|narcotics|cocaine|heroin|methamphetamine)\b'
        ],
        'adult_content': [
            r'\b(?:adult|porn|sex|sexual|explicit|nude|naked)\b'
        ]
    }
    
    BUSINESS_VIOLATIONS = {
        'financial_advice': [
            r'\b(?:invest|investment|stock|trading|financial|advice)\b.*\b(?:guaranteed|sure|certain)\b',
            r'\b(?:bitcoin|crypto|cryptocurrency)\b.*\b(?:buy|sell|invest)\b'
        ],
        'medical_advice': [
            r'\b(?:diagnosis|diagnose|treatment|cure|medicine|medication)\b',
            r'\b(?:doctor|physician|medical)\b.*\b(?:recommend|advice|suggest)\b'
        ],
        'legal_advice': [
            r'\b(?:legal|law|court|lawsuit|attorney|lawyer)\b.*\b(?:advice|recommend)\b'
        ]
    }
    
    @classmethod
    def moderate_content(cls, text: str, check_business_rules: bool = True) -> Dict[str, Any]:
        """Moderate content for harmful or inappropriate material"""
        violations = []
        
        # Check harmful content
        for category, patterns in cls.HARMFUL_CONTENT.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({
                        'category': 'harmful',
                        'subcategory': category,
                        'severity': 'high',
                        'message': f'Content contains {category.replace("_", " ")}'
                    })
                    break
        
        # Check business rule violations
        if check_business_rules:
            for category, patterns in cls.BUSINESS_VIOLATIONS.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        violations.append({
                            'category': 'business_violation',
                            'subcategory': category,
                            'severity': 'medium',
                            'message': f'Content may violate {category.replace("_", " ")} policies'
                        })
                        break
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': cls._calculate_risk_score(violations)
        }
    
    @classmethod
    def _calculate_risk_score(cls, violations: List[Dict]) -> float:
        """Calculate overall risk score (0-1)"""
        if not violations:
            return 0.0
        
        severity_weights = {'high': 0.8, 'medium': 0.5, 'low': 0.2}
        total_score = sum(severity_weights.get(v['severity'], 0.3) for v in violations)
        return min(total_score / len(violations), 1.0)

class ContentViolation(Exception):
    """Exception raised when content violates moderation policies"""
    pass

def content_moderation_guardrail_examples():
    """Examples of content moderation guardrails"""
    
    print("\n📊 CONTENT MODERATION GUARDRAILS")
    print("=" * 50)
    
    def moderate_input_and_output(input_text: str, output_text: str = None) -> Dict[str, Any]:
        """Moderate both input and output"""
        results = {
            'input_approved': True,
            'output_approved': True,
            'input_violations': [],
            'output_violations': []
        }
        
        # Moderate input
        input_moderation = ContentModerator.moderate_content(input_text)
        results['input_approved'] = input_moderation['approved']
        results['input_violations'] = input_moderation['violations']
        
        # Moderate output if provided
        if output_text:
            output_moderation = ContentModerator.moderate_content(output_text)
            results['output_approved'] = output_moderation['approved']
            results['output_violations'] = output_moderation['violations']
        
        return results
    
    @function_tool
    def content_processor(user_request: str) -> str:
        """Process content with moderation"""
        # Moderate input first
        moderation = moderate_input_and_output(user_request)
        
        if not moderation['input_approved']:
            violations = moderation['input_violations']
            high_severity = [v for v in violations if v['severity'] == 'high']
            
            if high_severity:
                return f"❌ Content blocked: {high_severity[0]['message']}"
            else:
                return f"⚠️ Content warning: {violations[0]['message']}. Please rephrase your request."
        
        # Process the request (simplified)
        response = f"I understand you're asking about: {user_request[:100]}..."
        
        # Moderate output
        output_moderation = ContentModerator.moderate_content(response)
        if not output_moderation['approved']:
            return "❌ Response blocked by content moderation."
        
        return response
    
    @function_tool
    def safe_information_provider(topic: str) -> str:
        """Provide safe information on various topics"""
        # Check if topic is appropriate
        moderation = ContentModerator.moderate_content(topic)
        
        if not moderation['approved']:
            violations = moderation['violations']
            return f"❌ Cannot provide information on this topic: {violations[0]['message']}"
        
        # Provide safe, general information
        safe_responses = {
            'health': "For health information, please consult qualified healthcare professionals.",
            'finance': "For financial decisions, please consult certified financial advisors.",
            'legal': "For legal matters, please consult qualified legal professionals.",
            'general': f"I can provide general, educational information about {topic}."
        }
        
        # Determine response type
        for violation in moderation['violations']:
            if violation['subcategory'] in safe_responses:
                return safe_responses[violation['subcategory']]
        
        return safe_responses['general']
    
    # Create moderated agent
    moderated_agent = Agent(
        name="ModeratedAgent",
        instructions="""
        You are a safety-first agent that moderates all content.
        You refuse to process harmful, inappropriate, or policy-violating content.
        You provide safe, helpful responses within appropriate boundaries.
        Always prioritize user safety and platform policies.
        """,
        tools=[content_processor, safe_information_provider]
    )
    
    # Test content moderation
    test_content = [
        "What's the weather like today?",  # Safe content
        "How to invest in cryptocurrency guaranteed returns",  # Business violation
        "Tell me about violence in movies",  # Potential harmful content
        "I need medical advice for my symptoms",  # Medical advice violation
        "How to hack into computer systems",  # Illegal content
        "What's the capital of France?",  # Safe educational content
    ]
    
    print("\n4. Testing Content Moderation Guardrails:")
    for i, content in enumerate(test_content, 1):
        try:
            result = Runner.run_sync(moderated_agent, content)
            print(f"{i}. Input: {content}")
            print(f"   Result: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return moderated_agent

# =============================================================================
# 🛠️ 5. CUSTOM BUSINESS RULE GUARDRAILS
# =============================================================================

class BusinessRuleEngine:
    """Custom business rule enforcement"""
    
    def __init__(self):
        self.rules = {}
    
    def add_rule(self, rule_name: str, condition: Callable, action: Callable, 
                 description: str = ""):
        """Add a business rule"""
        self.rules[rule_name] = {
            'condition': condition,
            'action': action,
            'description': description,
            'triggered_count': 0
        }
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all rules against context"""
        triggered_rules = []
        
        for rule_name, rule in self.rules.items():
            try:
                if rule['condition'](context):
                    result = rule['action'](context)
                    rule['triggered_count'] += 1
                    
                    triggered_rules.append({
                        'rule_name': rule_name,
                        'description': rule['description'],
                        'action_result': result
                    })
            except Exception as e:
                print(f"Error evaluating rule {rule_name}: {e}")
        
        return triggered_rules

def business_rule_guardrail_examples():
    """Examples of custom business rule guardrails"""
    
    print("\n🛠️ CUSTOM BUSINESS RULE GUARDRAILS")
    print("=" * 50)
    
    # Initialize business rule engine
    rule_engine = BusinessRuleEngine()
    
    # Define business rules
    
    # Rule 1: Working hours restriction
    def working_hours_condition(context: Dict) -> bool:
        import datetime
        now = datetime.datetime.now()
        return not (9 <= now.hour < 17 and now.weekday() < 5)
    
    def working_hours_action(context: Dict) -> str:
        return "Non-business hours: Limited support available"
    
    rule_engine.add_rule(
        "working_hours",
        working_hours_condition,
        working_hours_action,
        "Restrict certain actions outside business hours"
    )
    
    # Rule 2: High-value transaction approval
    def high_value_condition(context: Dict) -> bool:
        amount = context.get('transaction_amount', 0)
        return amount > 10000
    
    def high_value_action(context: Dict) -> str:
        return "High-value transaction requires manager approval"
    
    rule_engine.add_rule(
        "high_value_transaction",
        high_value_condition,
        high_value_action,
        "High-value transactions need approval"
    )
    
    # Rule 3: Sensitive data access logging
    def sensitive_data_condition(context: Dict) -> bool:
        request = context.get('request', '').lower()
        sensitive_keywords = ['confidential', 'secret', 'private', 'sensitive']
        return any(keyword in request for keyword in sensitive_keywords)
    
    def sensitive_data_action(context: Dict) -> str:
        user = context.get('user_id', 'unknown')
        print(f"🔍 AUDIT LOG: Sensitive data access by {user}")
        return "Sensitive data access logged"
    
    rule_engine.add_rule(
        "sensitive_data_access",
        sensitive_data_condition,
        sensitive_data_action,
        "Log all sensitive data access attempts"
    )
    
    # Rule 4: Geographic restrictions
    def geographic_restriction_condition(context: Dict) -> bool:
        country = context.get('user_country', 'unknown')
        restricted_countries = ['restricted_country_1', 'restricted_country_2']
        return country in restricted_countries
    
    def geographic_restriction_action(context: Dict) -> str:
        return "Access restricted in your geographic location"
    
    rule_engine.add_rule(
        "geographic_restriction",
        geographic_restriction_condition,
        geographic_restriction_action,
        "Restrict access based on geographic location"
    )
    
    @function_tool
    def business_transaction(transaction_type: str, amount: float = 0, 
                           user_id: str = "default") -> str:
        """Process business transaction with rule checking"""
        context = {
            'transaction_type': transaction_type,
            'transaction_amount': amount,
            'user_id': user_id,
            'request': f"{transaction_type} for ${amount}",
            'user_country': 'US'  # Default for testing
        }
        
        # Evaluate business rules
        triggered_rules = rule_engine.evaluate_rules(context)
        
        if triggered_rules:
            # Handle rule violations
            blocking_rules = ['geographic_restriction', 'high_value_transaction']
            
            for rule in triggered_rules:
                if rule['rule_name'] in blocking_rules:
                    return f"❌ Transaction blocked: {rule['action_result']}"
            
            # Non-blocking rules (warnings/logging)
            warnings = [rule['action_result'] for rule in triggered_rules]
            return f"⚠️ Transaction processed with warnings: {'; '.join(warnings)}"
        
        return f"✅ {transaction_type} transaction for ${amount} processed successfully"
    
    @function_tool
    def access_company_data(data_type: str, user_id: str = "employee") -> str:
        """Access company data with business rule checking"""
        context = {
            'request': f"access {data_type} data",
            'user_id': user_id,
            'data_type': data_type
        }
        
        # Evaluate rules
        triggered_rules = rule_engine.evaluate_rules(context)
        
        if triggered_rules:
            messages = [rule['action_result'] for rule in triggered_rules]
            return f"Data access: {'; '.join(messages)}"
        
        return f"✅ Access granted to {data_type} data for {user_id}"
    
    # Create agent with business rules
    business_agent = Agent(
        name="BusinessAgent",
        instructions="""
        You are a business operations agent that enforces company policies.
        You check all business rules before processing requests.
        You clearly communicate policy restrictions and requirements.
        """,
        tools=[business_transaction, access_company_data]
    )
    
    # Test business rules
    business_scenarios = [
        # Regular transaction
        (business_agent, "Process a purchase transaction for $500"),
        
        # High-value transaction
        (business_agent, "Process a purchase transaction for $15000"),
        
        # Sensitive data access
        (business_agent, "I need access to confidential employee records"),
        
        # Regular data access
        (business_agent, "Show me the public company directory"),
    ]
    
    print("\n5. Testing Business Rule Guardrails:")
    for i, (agent, request) in enumerate(business_scenarios, 1):
        try:
            result = Runner.run_sync(agent, request)
            print(f"{i}. Request: {request}")
            print(f"   Result: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return business_agent

# =============================================================================
# 🔄 6. COMPREHENSIVE GUARDRAIL SYSTEM
# =============================================================================

class GuardrailSystem:
    """Comprehensive guardrail system combining all protection mechanisms"""
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.permissions = UserPermissions()
        self.rate_limiter = RateLimiter()
        self.content_moderator = ContentModerator()
        self.business_rules = BusinessRuleEngine()
        
        # Configure rate limits
        self.rate_limiter.limits = {
            'per_minute': 10,
            'per_hour': 100,
            'per_day': 1000
        }
        
        # Add default business rules
        self._setup_default_business_rules()
    
    def _setup_default_business_rules(self):
        """Setup default business rules"""
        
        # Rule: Block after-hours sensitive operations
        def after_hours_sensitive_condition(context: Dict) -> bool:
            import datetime
            now = datetime.datetime.now()
            is_after_hours = not (9 <= now.hour < 17 and now.weekday() < 5)
            is_sensitive = context.get('requires_sensitive_permission', False)
            return is_after_hours and is_sensitive
        
        def after_hours_sensitive_action(context: Dict) -> str:
            return "Sensitive operations blocked outside business hours"
        
        self.business_rules.add_rule(
            "after_hours_sensitive",
            after_hours_sensitive_condition,
            after_hours_sensitive_action,
            "Block sensitive operations after hours"
        )
    
    def process_request(self, request: str, user_id: str, context: Dict = None) -> Dict[str, Any]:
        """Process request through all guardrails"""
        context = context or {}
        context['user_id'] = user_id
        context['request'] = request
        
        results = {
            'approved': True,
            'response': None,
            'violations': [],
            'warnings': []
        }
        
        try:
            # 1. Input Validation
            validation = self.content_filter.validate_input(request)
            if not validation['valid']:
                results['approved'] = False
                results['violations'].extend(validation['issues'])
                return results
            
            # 2. Rate Limiting
            rate_status = self.rate_limiter.record_request(user_id)
            for window, limited in rate_status.items():
                if limited:
                    results['approved'] = False
                    results['violations'].append({
                        'type': 'rate_limit',
                        'message': f'Rate limit exceeded for {window}'
                    })
                    return results
            
            # 3. Content Moderation
            moderation = self.content_moderator.moderate_content(request)
            if not moderation['approved']:
                high_risk = moderation['risk_score'] > 0.7
                if high_risk:
                    results['approved'] = False
                    results['violations'].extend(moderation['violations'])
                    return results
                else:
                    results['warnings'].extend(moderation['violations'])
            
            # 4. Permission Checking (if required)
            required_permission = context.get('required_permission')
            if required_permission:
                if not self.permissions.has_permission(user_id, required_permission):
                    results['approved'] = False
                    results['violations'].append({
                        'type': 'permission_denied',
                        'message': f'Missing required permission: {required_permission}'
                    })
                    return results
            
            # 5. Business Rules
            triggered_rules = self.business_rules.evaluate_rules(context)
            blocking_rules = ['after_hours_sensitive', 'geographic_restriction']
            
            for rule in triggered_rules:
                if rule['rule_name'] in blocking_rules:
                    results['approved'] = False
                    results['violations'].append({
                        'type': 'business_rule',
                        'message': rule['action_result']
                    })
                    return results
                else:
                    results['warnings'].append({
                        'type': 'business_rule',
                        'message': rule['action_result']
                    })
            
            # If we get here, request is approved
            results['response'] = "✅ Request approved and processed"
            
        except Exception as e:
            results['approved'] = False
            results['violations'].append({
                'type': 'system_error',
                'message': f'Guardrail system error: {str(e)}'
            })
        
        return results

def comprehensive_guardrail_examples():
    """Examples of comprehensive guardrail system"""
    
    print("\n🔄 COMPREHENSIVE GUARDRAIL SYSTEM")
    print("=" * 50)
    
    # Initialize comprehensive guardrail system
    guardrail_system = GuardrailSystem()
    
    @function_tool
    def secure_operation(operation: str, user_id: str = "guest") -> str:
        """Perform secure operation with comprehensive guardrails"""
        
        # Determine if operation requires special permissions
        context = {'required_permission': None}
        
        sensitive_ops = ['delete', 'admin', 'sensitive', 'confidential']
        if any(op in operation.lower() for op in sensitive_ops):
            context['required_permission'] = 'sensitive'
            context['requires_sensitive_permission'] = True
        
        write_ops = ['create', 'update', 'modify', 'change']
        if any(op in operation.lower() for op in write_ops):
            context['required_permission'] = 'write'
        
        # Process through guardrails
        result = guardrail_system.process_request(operation, user_id, context)
        
        if not result['approved']:
            violations = [v['message'] for v in result['violations']]
            return f"❌ Operation blocked: {'; '.join(violations)}"
        
        warnings = [w['message'] for w in result['warnings']]
        warning_text = f" (Warnings: {'; '.join(warnings)})" if warnings else ""
        
        return f"✅ Operation '{operation}' completed successfully{warning_text}"
    
    # Create comprehensive secure agent
    secure_agent = Agent(
        name="SecureAgent",
        instructions="""
        You are a highly secure agent with comprehensive guardrails.
        You enforce all security policies, permissions, and business rules.
        You clearly communicate security restrictions and provide guidance
        on how to meet requirements when possible.
        """,
        tools=[secure_operation]
    )
    
    # Test comprehensive guardrails
    comprehensive_scenarios = [
        # Safe operation
        ("alice@company.com", "Create a new document"),
        
        # Permission required operation 
        ("guest", "Delete sensitive data"),
        
        # Admin operation by authorized user
        ("alice@company.com", "Admin system configuration"),
        
        # Potentially harmful content
        ("bob@company.com", "How to hack the system"),
        
        # Regular user legitimate operation
        ("charlie@company.com", "Update my profile information"),
        
        # Guest trying to write
        ("guest", "Create a new file"),
    ]
    
    print("\n6. Testing Comprehensive Guardrail System:")
    for i, (user_id, operation) in enumerate(comprehensive_scenarios, 1):
        try:
            result = Runner.run_sync(
                secure_agent, 
                operation,
                context={'user_id': user_id}
            )
            print(f"{i}. User: {user_id}")
            print(f"   Operation: {operation}")
            print(f"   Result: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return secure_agent

# =============================================================================
# 💡 7. GUARDRAILS BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 GUARDRAILS BEST PRACTICES

1. 🛡️ DEFENSE IN DEPTH:
   ✅ Layer multiple guardrails (input → permission → rate limit → content)
   ✅ Each layer provides specific protection
   ✅ Redundancy prevents single points of failure
   ✅ Fail-safe defaults (deny when unsure)

2. 🚦 GUARDRAIL HIERARCHY:
   1. Input Validation (first line of defense)
   2. Authentication & Authorization
   3. Rate Limiting & Abuse Prevention  
   4. Content Moderation & Safety
   5. Business Rule Enforcement
   6. Output Filtering & Compliance

3. 🎯 WHEN TO USE EACH GUARDRAIL:

   INPUT VALIDATION:
   ✅ Always validate user input
   ✅ Sanitize potentially dangerous content
   ✅ Check input format and length limits

   PERMISSIONS:
   ✅ For resource access control
   ✅ Feature-based authorization
   ✅ Role-based access control

   RATE LIMITING:
   ✅ Prevent abuse and DoS attacks
   ✅ Fair usage enforcement
   ✅ Resource protection

   CONTENT MODERATION:
   ✅ Safety and policy compliance
   ✅ Brand protection
   ✅ Legal compliance

   BUSINESS RULES:
   ✅ Domain-specific constraints
   ✅ Workflow enforcement
   ✅ Compliance requirements

4. ⚙️ IMPLEMENTATION PATTERNS:

   Pattern A - Decorator Pattern:
   ```python
   @require_permission('read')
   @rate_limit(requests_per_minute=60)
   @validate_input()
   def secure_function(data):
       return process(data)
   ```

   Pattern B - Chain of Responsibility:
   ```python
   guardrails = [
       InputValidator(),
       PermissionChecker(),
       RateLimiter(),
       ContentModerator()
   ]
   
   for guardrail in guardrails:
       if not guardrail.check(request):
           return guardrail.get_error()
   ```

   Pattern C - Context-Based Enforcement:
   ```python
   def enforce_guardrails(context):
       if context.user_tier == 'premium':
           return PremiumGuardrails()
       return StandardGuardrails()
   ```

5. 🔍 MONITORING & ALERTING:
   ✅ Log all guardrail violations
   ✅ Monitor violation patterns and trends
   ✅ Alert on suspicious activity
   ✅ Track guardrail performance impact

6. 🚨 ERROR HANDLING:
   ✅ Graceful degradation when guardrails fail
   ✅ Clear error messages for users
   ✅ Logging for security team
   ✅ Fallback mechanisms

🚨 COMMON GUARDRAIL PITFALLS:

❌ DON'T rely on client-side validation only
❌ DON'T hardcode security rules in business logic
❌ DON'T ignore performance impact of guardrails
❌ DON'T forget to handle edge cases
❌ DON'T expose sensitive information in error messages
❌ DON'T create guardrails that block legitimate use

✅ GUARDRAIL SUCCESS CHECKLIST:

☐ Multiple layers of protection
☐ Clear security policies and rules
☐ Comprehensive testing coverage
☐ Performance impact assessment
☐ User experience consideration
☐ Monitoring and alerting setup
☐ Regular security reviews
☐ Documentation and training

📈 SCALING GUARDRAILS:

For high-volume systems:
- Use async guardrail processing
- Implement caching for expensive checks
- Distribute rate limiting across instances
- Use ML for advanced content moderation
- Implement circuit breakers for external checks
- Monitor and optimize guardrail performance
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_guardrail_examples():
    """Run all guardrail examples comprehensively"""
    
    print("🛡️ OPENAI AGENTS SDK - COMPLETE GUARDRAILS DEMONSTRATION")
    print("=" * 70)
    
    # 1. Input Validation
    validated_agent = input_validation_guardrail_examples()
    
    # 2. Authorization
    authorized_agent = authorization_guardrail_examples()
    
    # 3. Rate Limiting
    rate_limited_agent = rate_limiting_guardrail_examples()
    
    # 4. Content Moderation  
    moderated_agent = content_moderation_guardrail_examples()
    
    # 5. Business Rules
    business_agent = business_rule_guardrail_examples()
    
    # 6. Comprehensive System
    secure_agent = comprehensive_guardrail_examples()
    
    print("\n✅ All guardrail examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Layer multiple guardrails for defense in depth")
    print("- Always validate input and moderate content")
    print("- Enforce permissions and rate limits appropriately")
    print("- Implement business rules for domain constraints")
    print("- Monitor and log all security events")
    print("- Balance security with user experience")

def run_sync_guardrail_examples():
    """Run synchronous guardrail examples for immediate testing"""
    
    print("🛡️ OPENAI AGENTS SDK - GUARDRAILS SYNC EXAMPLES")
    print("=" * 60)
    
    # Run only synchronous examples
    input_validation_guardrail_examples()
    authorization_guardrail_examples()
    rate_limiting_guardrail_examples()
    content_moderation_guardrail_examples()
    business_rule_guardrail_examples()
    comprehensive_guardrail_examples()
    
    print("\n✅ Sync guardrail examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🛡️ OpenAI Agents SDK - Complete Guardrails Template")
    print("This template demonstrates all Guardrail patterns and security practices.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_guardrail_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_guardrail_examples())
    
    print("\n✅ Guardrails template demonstration complete!")
    print("💡 Use this template as reference for all your security and guardrail needs.")
