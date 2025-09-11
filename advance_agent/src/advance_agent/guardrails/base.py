"""Guardrails system for the Advanced Agent System."""

import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from pydantic import BaseModel

from ..core.exceptions import GuardrailViolationError, GuardrailConfigurationError
from ..core.logging import get_logger
from ..core.security import SecurityContext, comprehensive_security_analysis
from ..models.schemas import MessageRole


class GuardrailAction(str, Enum):
    """Actions that can be taken when a guardrail is triggered."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


class GuardrailSeverity(str, Enum):
    """Severity levels for guardrail violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailViolation(BaseModel):
    """Represents a guardrail violation."""
    guardrail_name: str
    severity: GuardrailSeverity
    action: GuardrailAction
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    content_sample: Optional[str] = None


class GuardrailResult(BaseModel):
    """Result of guardrail evaluation."""
    is_allowed: bool
    action: GuardrailAction
    violations: List[GuardrailViolation]
    modified_content: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = 1.0


class BaseGuardrail(ABC):
    """Abstract base class for all guardrails."""
    
    def __init__(
        self,
        name: str,
        description: str,
        severity: GuardrailSeverity = GuardrailSeverity.MEDIUM,
        action: GuardrailAction = GuardrailAction.WARN,
        enabled: bool = True
    ):
        self.name = name
        self.description = description
        self.severity = severity
        self.action = action
        self.enabled = enabled
        self.logger = get_logger(f"guardrail.{name}")
        
        # Metrics
        self.violation_count = 0
        self.evaluation_count = 0
        self.last_violation = None
    
    @abstractmethod
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content against this guardrail."""
        pass
    
    def create_violation(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None,
        content_sample: Optional[str] = None
    ) -> GuardrailViolation:
        """Create a guardrail violation."""
        return GuardrailViolation(
            guardrail_name=self.name,
            severity=self.severity,
            action=self.action,
            message=message,
            details=details or {},
            timestamp=datetime.now(timezone.utc),
            user_id=security_context.user_id if security_context else None,
            content_sample=content_sample
        )
    
    def update_metrics(self, has_violation: bool):
        """Update guardrail metrics."""
        self.evaluation_count += 1
        if has_violation:
            self.violation_count += 1
            self.last_violation = datetime.now(timezone.utc)


class ContentLengthGuardrail(BaseGuardrail):
    """Guardrail to check content length limits."""
    
    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 1,
        action: GuardrailAction = GuardrailAction.BLOCK,
        **kwargs
    ):
        super().__init__(
            name="content_length",
            description=f"Enforces content length between {min_length} and {max_length} characters",
            action=action,
            **kwargs
        )
        self.max_length = max_length
        self.min_length = min_length
    
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content length."""
        content_length = len(content)
        violations = []
        
        if content_length > self.max_length:
            violation = self.create_violation(
                f"Content too long: {content_length} characters (max: {self.max_length})",
                details={
                    "length": content_length,
                    "max_length": self.max_length,
                    "excess": content_length - self.max_length
                },
                security_context=security_context,
                content_sample=content[:100] + "..." if len(content) > 100 else content
            )
            violations.append(violation)
        
        elif content_length < self.min_length:
            violation = self.create_violation(
                f"Content too short: {content_length} characters (min: {self.min_length})",
                details={
                    "length": content_length,
                    "min_length": self.min_length,
                    "shortage": self.min_length - content_length
                },
                security_context=security_context,
                content_sample=content
            )
            violations.append(violation)
        
        has_violation = len(violations) > 0
        self.update_metrics(has_violation)
        
        # Handle modification for length issues
        modified_content = None
        if has_violation and self.action == GuardrailAction.MODIFY:
            if content_length > self.max_length:
                modified_content = content[:self.max_length] + "..."
            elif content_length < self.min_length:
                modified_content = content + " " * (self.min_length - content_length)
        
        return GuardrailResult(
            is_allowed=not has_violation or self.action != GuardrailAction.BLOCK,
            action=self.action if has_violation else GuardrailAction.ALLOW,
            violations=violations,
            modified_content=modified_content,
            explanation=f"Content length: {content_length} characters"
        )


class ProfanityGuardrail(BaseGuardrail):
    """Guardrail to detect and handle profanity."""
    
    def __init__(
        self,
        profanity_words: Optional[List[str]] = None,
        action: GuardrailAction = GuardrailAction.MODIFY,
        replacement: str = "***",
        **kwargs
    ):
        super().__init__(
            name="profanity_filter",
            description="Detects and filters profanity in content",
            action=action,
            **kwargs
        )
        
        # Default profanity list (simplified for demonstration)
        self.profanity_words = profanity_words or [
            "damn", "hell", "crap", "shit", "fuck", "bitch", "ass"
        ]
        self.replacement = replacement
        
        # Compile regex patterns for efficient matching
        self.profanity_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in self.profanity_words
        ]
    
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content for profanity."""
        violations = []
        detected_words = []
        modified_content = content
        
        # Check for profanity
        for pattern, word in zip(self.profanity_patterns, self.profanity_words):
            matches = pattern.findall(content)
            if matches:
                detected_words.extend(matches)
                if self.action == GuardrailAction.MODIFY:
                    modified_content = pattern.sub(self.replacement, modified_content)
        
        if detected_words:
            violation = self.create_violation(
                f"Profanity detected: {detected_words}",
                details={
                    "detected_words": detected_words,
                    "count": len(detected_words),
                    "replacement": self.replacement
                },
                security_context=security_context,
                content_sample=content[:200] + "..." if len(content) > 200 else content
            )
            violations.append(violation)
        
        has_violation = len(violations) > 0
        self.update_metrics(has_violation)
        
        return GuardrailResult(
            is_allowed=not has_violation or self.action != GuardrailAction.BLOCK,
            action=self.action if has_violation else GuardrailAction.ALLOW,
            violations=violations,
            modified_content=modified_content if has_violation and self.action == GuardrailAction.MODIFY else None,
            explanation=f"Profanity check: {len(detected_words)} violations found",
            confidence=0.9 if detected_words else 1.0
        )


class PIIGuardrail(BaseGuardrail):
    """Guardrail to detect personally identifiable information."""
    
    def __init__(
        self,
        action: GuardrailAction = GuardrailAction.BLOCK,
        mask_pii: bool = True,
        **kwargs
    ):
        super().__init__(
            name="pii_detection",
            description="Detects and protects personally identifiable information",
            severity=GuardrailSeverity.HIGH,
            action=action,
            **kwargs
        )
        self.mask_pii = mask_pii
    
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content for PII."""
        # Use the comprehensive security analysis
        analysis = comprehensive_security_analysis(content)
        pii_result = analysis.pii_result
        
        violations = []
        modified_content = None
        
        if pii_result.has_pii:
            violation = self.create_violation(
                f"PII detected: {', '.join(pii_result.detected_types)}",
                details={
                    "detected_types": pii_result.detected_types,
                    "confidence": pii_result.confidence_score,
                    "details": pii_result.details
                },
                security_context=security_context,
                content_sample=content[:100] + "..." if len(content) > 100 else content
            )
            violations.append(violation)
            
            if self.mask_pii and self.action in [GuardrailAction.MODIFY, GuardrailAction.WARN]:
                modified_content = pii_result.masked_content
        
        has_violation = len(violations) > 0
        self.update_metrics(has_violation)
        
        return GuardrailResult(
            is_allowed=not has_violation or self.action != GuardrailAction.BLOCK,
            action=self.action if has_violation else GuardrailAction.ALLOW,
            violations=violations,
            modified_content=modified_content,
            explanation=f"PII detection: {len(pii_result.detected_types)} types found",
            confidence=pii_result.confidence_score
        )


class RateLimitGuardrail(BaseGuardrail):
    """Guardrail to enforce rate limiting."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        action: GuardrailAction = GuardrailAction.BLOCK,
        **kwargs
    ):
        super().__init__(
            name="rate_limit",
            description=f"Enforces rate limits: {requests_per_minute}/min, {requests_per_hour}/hr",
            action=action,
            **kwargs
        )
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests by user
        self.user_requests = {}
    
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate rate limits."""
        if not security_context:
            # No user context, allow
            return GuardrailResult(
                is_allowed=True,
                action=GuardrailAction.ALLOW,
                violations=[],
                explanation="No user context for rate limiting"
            )
        
        user_id = security_context.user_id
        now = datetime.now(timezone.utc)
        
        # Initialize user tracking
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        user_requests = self.user_requests[user_id]
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Remove requests older than an hour
        self.user_requests[user_id] = [
            req_time for req_time in user_requests
            if req_time > hour_ago
        ]
        user_requests = self.user_requests[user_id]
        
        # Count recent requests
        requests_last_minute = len([
            req_time for req_time in user_requests
            if req_time > minute_ago
        ])
        
        requests_last_hour = len(user_requests)
        
        violations = []
        
        # Check minute limit
        if requests_last_minute >= self.requests_per_minute:
            violation = self.create_violation(
                f"Rate limit exceeded: {requests_last_minute} requests in last minute (limit: {self.requests_per_minute})",
                details={
                    "requests_last_minute": requests_last_minute,
                    "limit_per_minute": self.requests_per_minute,
                    "user_id": user_id
                },
                security_context=security_context
            )
            violations.append(violation)
        
        # Check hour limit
        elif requests_last_hour >= self.requests_per_hour:
            violation = self.create_violation(
                f"Rate limit exceeded: {requests_last_hour} requests in last hour (limit: {self.requests_per_hour})",
                details={
                    "requests_last_hour": requests_last_hour,
                    "limit_per_hour": self.requests_per_hour,
                    "user_id": user_id
                },
                security_context=security_context
            )
            violations.append(violation)
        
        # If no violations, record this request
        if not violations:
            self.user_requests[user_id].append(now)
        
        has_violation = len(violations) > 0
        self.update_metrics(has_violation)
        
        return GuardrailResult(
            is_allowed=not has_violation,
            action=self.action if has_violation else GuardrailAction.ALLOW,
            violations=violations,
            explanation=f"Rate check: {requests_last_minute}/min, {requests_last_hour}/hr"
        )


class TopicGuardrail(BaseGuardrail):
    """Guardrail to restrict conversation topics."""
    
    def __init__(
        self,
        forbidden_topics: Optional[List[str]] = None,
        allowed_topics: Optional[List[str]] = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
        **kwargs
    ):
        super().__init__(
            name="topic_restriction",
            description="Restricts conversation to allowed topics and blocks forbidden ones",
            action=action,
            **kwargs
        )
        
        # Default forbidden topics
        self.forbidden_topics = forbidden_topics or [
            "violence", "illegal activities", "hate speech", "harassment",
            "self-harm", "dangerous instructions", "malware", "hacking"
        ]
        
        self.allowed_topics = allowed_topics
        
        # Create keyword patterns
        self.forbidden_patterns = [
            re.compile(r'\b' + re.escape(topic.lower()) + r'\b', re.IGNORECASE)
            for topic in self.forbidden_topics
        ]
    
    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content for topic restrictions."""
        violations = []
        detected_forbidden = []
        
        content_lower = content.lower()
        
        # Check for forbidden topics
        for pattern, topic in zip(self.forbidden_patterns, self.forbidden_topics):
            if pattern.search(content):
                detected_forbidden.append(topic)
        
        if detected_forbidden:
            violation = self.create_violation(
                f"Forbidden topics detected: {detected_forbidden}",
                details={
                    "forbidden_topics": detected_forbidden,
                    "all_forbidden": self.forbidden_topics
                },
                security_context=security_context,
                content_sample=content[:150] + "..." if len(content) > 150 else content
            )
            violations.append(violation)
        
        has_violation = len(violations) > 0
        self.update_metrics(has_violation)
        
        return GuardrailResult(
            is_allowed=not has_violation,
            action=self.action if has_violation else GuardrailAction.ALLOW,
            violations=violations,
            explanation=f"Topic check: {len(detected_forbidden)} forbidden topics found"
        )


class GuardrailEngine:
    """Engine that manages and executes multiple guardrails."""
    
    def __init__(self):
        self.guardrails: List[BaseGuardrail] = []
        self.logger = get_logger("guardrail_engine")
        
        # Global metrics
        self.total_evaluations = 0
        self.total_violations = 0
        self.blocked_requests = 0
        self.modified_requests = 0
    
    def add_guardrail(self, guardrail: BaseGuardrail):
        """Add a guardrail to the engine."""
        self.guardrails.append(guardrail)
        self.logger.info(f"Added guardrail: {guardrail.name}")
    
    def remove_guardrail(self, name: str) -> bool:
        """Remove a guardrail by name."""
        for i, guardrail in enumerate(self.guardrails):
            if guardrail.name == name:
                del self.guardrails[i]
                self.logger.info(f"Removed guardrail: {name}")
                return True
        return False
    
    def get_guardrail(self, name: str) -> Optional[BaseGuardrail]:
        """Get a guardrail by name."""
        for guardrail in self.guardrails:
            if guardrail.name == name:
                return guardrail
        return None
    
    async def evaluate_all(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> GuardrailResult:
        """Evaluate content against all guardrails."""
        self.total_evaluations += 1
        
        all_violations = []
        final_content = content
        is_allowed = True
        final_action = GuardrailAction.ALLOW
        explanations = []
        
        # Evaluate each guardrail
        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue
            
            try:
                result = await guardrail.evaluate(final_content, context, security_context)
                
                if result.violations:
                    all_violations.extend(result.violations)
                    self.total_violations += len(result.violations)
                
                explanations.append(result.explanation)
                
                # Update content if modified
                if result.modified_content:
                    final_content = result.modified_content
                    self.modified_requests += 1
                
                # Check if request should be blocked
                if not result.is_allowed:
                    is_allowed = False
                    if result.action == GuardrailAction.BLOCK:
                        final_action = GuardrailAction.BLOCK
                        self.blocked_requests += 1
                        break
                    elif result.action == GuardrailAction.ESCALATE:
                        final_action = GuardrailAction.ESCALATE
                    elif final_action == GuardrailAction.ALLOW:
                        final_action = result.action
            
            except Exception as e:
                self.logger.error(f"Guardrail {guardrail.name} evaluation failed: {str(e)}")
                # Continue with other guardrails
        
        # Log summary
        if all_violations:
            self.logger.warning(
                f"Guardrail violations detected: {len(all_violations)} violations",
                extra={
                    "violations": [v.guardrail_name for v in all_violations],
                    "action": final_action.value,
                    "user_id": security_context.user_id if security_context else None
                }
            )
        
        return GuardrailResult(
            is_allowed=is_allowed,
            action=final_action,
            violations=all_violations,
            modified_content=final_content if final_content != content else None,
            explanation="; ".join(filter(None, explanations)),
            confidence=min(1.0, 1.0 - (len(all_violations) * 0.1))
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get guardrail engine metrics."""
        guardrail_metrics = {}
        for guardrail in self.guardrails:
            guardrail_metrics[guardrail.name] = {
                "enabled": guardrail.enabled,
                "evaluations": guardrail.evaluation_count,
                "violations": guardrail.violation_count,
                "violation_rate": (
                    guardrail.violation_count / guardrail.evaluation_count
                    if guardrail.evaluation_count > 0 else 0.0
                ),
                "last_violation": (
                    guardrail.last_violation.isoformat()
                    if guardrail.last_violation else None
                )
            }
        
        return {
            "total_evaluations": self.total_evaluations,
            "total_violations": self.total_violations,
            "blocked_requests": self.blocked_requests,
            "modified_requests": self.modified_requests,
            "violation_rate": (
                self.total_violations / self.total_evaluations
                if self.total_evaluations > 0 else 0.0
            ),
            "guardrails": guardrail_metrics
        }


# Create default guardrail engine with common guardrails
def create_default_guardrail_engine() -> GuardrailEngine:
    """Create a guardrail engine with default guardrails."""
    engine = GuardrailEngine()
    
    # Add default guardrails
    engine.add_guardrail(ContentLengthGuardrail(max_length=50000))
    engine.add_guardrail(ProfanityGuardrail())
    engine.add_guardrail(PIIGuardrail())
    engine.add_guardrail(RateLimitGuardrail())
    engine.add_guardrail(TopicGuardrail())
    
    return engine
