"""Security utilities and helpers for the Advanced Agent System."""

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Pattern, Tuple
from uuid import uuid4

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

from .config import get_settings
from .exceptions import SecurityViolationError


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityConfig:
    """Security configuration and patterns."""
    
    # PII Detection Patterns
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
    
    # Content Safety Patterns
    PROFANITY_PATTERNS = [
        re.compile(r'\b(fuck|shit|damn|hell|ass|bitch)\b', re.IGNORECASE),
        # Add more patterns as needed
    ]
    
    SPAM_INDICATORS = [
        re.compile(r'(click here|buy now|limited time|act now)', re.IGNORECASE),
        re.compile(r'(free money|earn \$|make money fast)', re.IGNORECASE),
    ]
    
    # Security threat patterns
    INJECTION_PATTERNS = [
        re.compile(r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b)', re.IGNORECASE),
        re.compile(r'(<script|javascript:|on\w+\s*=)', re.IGNORECASE),
    ]


class PIIDetectionResult(BaseModel):
    """Result of PII detection scan."""
    has_pii: bool
    detected_types: List[str]
    confidence_score: float
    masked_content: str
    details: Dict[str, List[str]]


class ContentSafetyResult(BaseModel):
    """Result of content safety analysis."""
    is_safe: bool
    safety_score: float
    violations: List[str]
    severity: str
    recommendations: List[str]


class SecurityAnalysisResult(BaseModel):
    """Comprehensive security analysis result."""
    pii_result: PIIDetectionResult
    safety_result: ContentSafetyResult
    overall_risk_score: float
    requires_review: bool
    blocking_issues: List[str]


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT access token."""
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None


def sanitize_input(text: str, strict: bool = False) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not text:
        return ""
    
    # Remove or escape potentially dangerous characters
    sanitized = text.strip()
    
    if strict:
        # Strict mode: only allow alphanumeric, spaces, and basic punctuation
        sanitized = re.sub(r'[^a-zA-Z0-9\s\.,!?;:\-_()]', '', sanitized)
    else:
        # Standard mode: escape HTML and SQL injection patterns
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
    
    return sanitized


def detect_pii(content: str) -> PIIDetectionResult:
    """Detect personally identifiable information in content."""
    detected_types = []
    details = {}
    masked_content = content
    
    # SSN detection
    ssn_matches = SecurityConfig.SSN_PATTERN.findall(content)
    if ssn_matches:
        detected_types.append("ssn")
        details["ssn"] = ssn_matches
        masked_content = SecurityConfig.SSN_PATTERN.sub("XXX-XX-XXXX", masked_content)
    
    # Credit card detection
    cc_matches = SecurityConfig.CREDIT_CARD_PATTERN.findall(content)
    if cc_matches:
        detected_types.append("credit_card")
        details["credit_card"] = cc_matches
        masked_content = SecurityConfig.CREDIT_CARD_PATTERN.sub("XXXX-XXXX-XXXX-XXXX", masked_content)
    
    # Email detection
    email_matches = SecurityConfig.EMAIL_PATTERN.findall(content)
    if email_matches:
        detected_types.append("email")
        details["email"] = email_matches
        masked_content = SecurityConfig.EMAIL_PATTERN.sub("email@masked.com", masked_content)
    
    # Phone number detection
    phone_matches = SecurityConfig.PHONE_PATTERN.findall(content)
    if phone_matches:
        detected_types.append("phone")
        details["phone"] = phone_matches
        masked_content = SecurityConfig.PHONE_PATTERN.sub("XXX-XXX-XXXX", masked_content)
    
    # Calculate confidence score
    total_patterns = len([
        SecurityConfig.SSN_PATTERN, SecurityConfig.CREDIT_CARD_PATTERN,
        SecurityConfig.EMAIL_PATTERN, SecurityConfig.PHONE_PATTERN
    ])
    confidence_score = len(detected_types) / total_patterns if detected_types else 0.0
    
    return PIIDetectionResult(
        has_pii=len(detected_types) > 0,
        detected_types=detected_types,
        confidence_score=confidence_score,
        masked_content=masked_content,
        details=details
    )


def analyze_content_safety(content: str) -> ContentSafetyResult:
    """Analyze content for safety violations."""
    violations = []
    severity = "low"
    
    # Check for profanity
    profanity_count = sum(
        len(pattern.findall(content)) 
        for pattern in SecurityConfig.PROFANITY_PATTERNS
    )
    if profanity_count > 0:
        violations.append("profanity")
        severity = "medium" if profanity_count <= 2 else "high"
    
    # Check for spam indicators
    spam_count = sum(
        len(pattern.findall(content)) 
        for pattern in SecurityConfig.SPAM_INDICATORS
    )
    if spam_count > 0:
        violations.append("spam")
    
    # Check for injection attempts
    injection_count = sum(
        len(pattern.findall(content)) 
        for pattern in SecurityConfig.INJECTION_PATTERNS
    )
    if injection_count > 0:
        violations.append("injection_attempt")
        severity = "critical"
    
    # Calculate safety score (inverse of violations)
    max_possible_violations = len([
        "profanity", "spam", "injection_attempt"
    ])
    safety_score = 1.0 - (len(violations) / max_possible_violations)
    
    # Generate recommendations
    recommendations = []
    if "profanity" in violations:
        recommendations.append("Remove or replace inappropriate language")
    if "spam" in violations:
        recommendations.append("Remove promotional language and excessive capitalization")
    if "injection_attempt" in violations:
        recommendations.append("Remove potentially malicious code patterns")
    
    return ContentSafetyResult(
        is_safe=len(violations) == 0,
        safety_score=safety_score,
        violations=violations,
        severity=severity,
        recommendations=recommendations
    )


def comprehensive_security_analysis(content: str) -> SecurityAnalysisResult:
    """Perform comprehensive security analysis on content."""
    pii_result = detect_pii(content)
    safety_result = analyze_content_safety(content)
    
    # Calculate overall risk score
    pii_risk = 0.7 if pii_result.has_pii else 0.0
    safety_risk = 1.0 - safety_result.safety_score
    overall_risk_score = max(pii_risk, safety_risk)
    
    # Determine if review is required
    requires_review = (
        pii_result.has_pii or 
        not safety_result.is_safe or 
        overall_risk_score > 0.5
    )
    
    # Identify blocking issues
    blocking_issues = []
    if "injection_attempt" in safety_result.violations:
        blocking_issues.append("Potential security injection detected")
    if pii_result.has_pii and "ssn" in pii_result.detected_types:
        blocking_issues.append("Social Security Number detected")
    if pii_result.has_pii and "credit_card" in pii_result.detected_types:
        blocking_issues.append("Credit card information detected")
    
    return SecurityAnalysisResult(
        pii_result=pii_result,
        safety_result=safety_result,
        overall_risk_score=overall_risk_score,
        requires_review=requires_review,
        blocking_issues=blocking_issues
    )


def validate_security_clearance(
    user_clearance: str, 
    required_clearance: str
) -> bool:
    """Validate if user has sufficient security clearance."""
    clearance_levels = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
        "top_secret": 4
    }
    
    user_level = clearance_levels.get(user_clearance.lower(), 0)
    required_level = clearance_levels.get(required_clearance.lower(), 0)
    
    return user_level >= required_level


def generate_audit_hash(data: Dict[str, Any]) -> str:
    """Generate a hash for audit trail integrity."""
    settings = get_settings()
    
    # Create deterministic string from data
    data_string = str(sorted(data.items()))
    
    # Create HMAC hash
    return hmac.new(
        settings.secret_key.encode(),
        data_string.encode(),
        hashlib.sha256
    ).hexdigest()


def verify_audit_integrity(
    data: Dict[str, Any], 
    provided_hash: str
) -> bool:
    """Verify audit trail integrity."""
    calculated_hash = generate_audit_hash(data)
    return hmac.compare_digest(calculated_hash, provided_hash)


class SecurityContext:
    """Security context for operations."""
    
    def __init__(
        self,
        user_id: str,
        clearance_level: str,
        permissions: List[str],
        session_id: Optional[str] = None
    ):
        self.user_id = user_id
        self.clearance_level = clearance_level
        self.permissions = permissions
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.utcnow()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def can_access_clearance(self, required_clearance: str) -> bool:
        """Check if user can access content with required clearance."""
        return validate_security_clearance(self.clearance_level, required_clearance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security context to dictionary."""
        return {
            "user_id": self.user_id,
            "clearance_level": self.clearance_level,
            "permissions": self.permissions,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat()
        }


def create_security_context(
    user_id: str,
    clearance_level: str = "public",
    permissions: Optional[List[str]] = None
) -> SecurityContext:
    """Create a security context for operations."""
    return SecurityContext(
        user_id=user_id,
        clearance_level=clearance_level,
        permissions=permissions or []
    )
