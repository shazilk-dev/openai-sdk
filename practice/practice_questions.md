# OpenAI Agents SDK Practice Questions - Real-World Professional Scenarios

Based on official OpenAI sources and documentation, these practice questions cover intermediate to advanced agent implementation scenarios that professionals encounter in production environments.

## Question 1: E-commerce Customer Service Multi-Agent System (Intermediate)

### Scenario

You're building a customer service system for an e-commerce platform. The system needs to handle:

- Order inquiries (status, modifications, cancellations)
- Product information requests
- Return and refund processing
- Technical support escalation

**Requirements:**

- Use handoffs between specialized agents
- Implement guardrails to prevent unauthorized refunds over $500
- Add tracing for monitoring agent performance
- Use Pydantic models for structured data handling

### Implementation Guide

#### Step 1: Define the Data Models

```python
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class OrderInfo(BaseModel):
    order_id: str
    customer_id: str
    status: Literal["pending", "shipped", "delivered", "cancelled"]
    total_amount: float
    items: list[str]

class RefundRequest(BaseModel):
    order_id: str
    amount: float
    reason: str
    urgency: Literal["low", "medium", "high"]

class EscalationData(BaseModel):
    issue_type: str
    priority: Literal["low", "medium", "high", "critical"]
    customer_id: str
    description: str
```

#### Step 2: Create Specialized Agents

```python
from agents import Agent, handoff, tool, RunContextWrapper
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Order Management Agent
order_agent = Agent(
    name="Order Management Agent",
    instructions=prompt_with_handoff_instructions("""
    You handle order-related inquiries including status checks, modifications, and cancellations.
    You have access to order lookup tools and can process standard requests.
    For refunds over $200, hand off to the Refund Specialist.
    """),
    tools=[lookup_order_tool, modify_order_tool, cancel_order_tool]
)

# Refund Specialist Agent
refund_agent = Agent(
    name="Refund Specialist",
    instructions="""
    You process refund requests with proper authorization checks.
    Verify order eligibility and process approved refunds.
    For amounts over $500, additional approval is required.
    """,
    tools=[process_refund_tool, verify_refund_eligibility_tool]
)

# Technical Support Agent
tech_support_agent = Agent(
    name="Technical Support",
    instructions="""
    You handle technical issues, account problems, and complex inquiries.
    Escalate critical issues to human support when necessary.
    """,
    tools=[check_account_status_tool, reset_password_tool, escalate_to_human_tool]
)

# Main Triage Agent
triage_agent = Agent(
    name="Customer Service Triage",
    instructions=prompt_with_handoff_instructions("""
    You are the first point of contact for customer inquiries.
    Analyze the customer's request and route to the appropriate specialist:
    - Order-related questions → Order Management Agent
    - Refund requests → Refund Specialist (if over $200) or handle directly (if under $200)
    - Technical issues → Technical Support
    Always greet customers warmly and gather essential information.
    """),
    handoffs=[
        handoff(order_agent, tool_description_override="Transfer to order management for order-related inquiries"),
        handoff(refund_agent, input_type=RefundRequest, tool_description_override="Transfer to refund specialist for refund processing"),
        handoff(tech_support_agent, input_type=EscalationData, tool_description_override="Transfer to technical support for technical issues")
    ]
)
```

#### Step 3: Implement Guardrails

```python
from agents import input_guardrail, output_guardrail, GuardrailFunctionOutput, Runner

@input_guardrail
async def refund_amount_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str
) -> GuardrailFunctionOutput:
    """Prevent unauthorized high-value refund attempts"""
    # Check if input contains refund request over $500
    if "refund" in input_data.lower():
        # Use a simple NLP check or external validation
        try:
            # Extract amount using regex or NLP
            import re
            amounts = re.findall(r'\$?(\d+(?:\.\d{2})?)', input_data)
            if amounts and float(amounts[0]) > 500:
                return GuardrailFunctionOutput(
                    output_info={"flagged_amount": float(amounts[0])},
                    tripwire_triggered=True
                )
        except:
            pass

    return GuardrailFunctionOutput(
        output_info={"status": "cleared"},
        tripwire_triggered=False
    )

# Apply guardrail to refund agent
refund_agent.input_guardrails = [refund_amount_guardrail]
```

#### Step 4: Add Comprehensive Tracing

```python
from agents import trace, Runner, RunConfig

async def handle_customer_request(customer_input: str, customer_id: str):
    with trace(
        workflow_name="E-commerce Customer Service",
        group_id=customer_id,
        metadata={"customer_id": customer_id, "timestamp": datetime.now().isoformat()}
    ) as customer_trace:

        config = RunConfig(
            trace_include_sensitive_data=False,  # Protect customer data
            workflow_name="Customer Service Flow"
        )

        try:
            result = await Runner.run(
                triage_agent,
                customer_input,
                config=config
            )
            return result.final_output

        except Exception as e:
            # Log error for monitoring
            print(f"Customer service error for {customer_id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact human support."
```

#### Step 5: Tool Implementation Examples

```python
@tool
async def lookup_order_tool(order_id: str) -> OrderInfo:
    """Look up order information by ID"""
    # Simulate database lookup
    mock_order = OrderInfo(
        order_id=order_id,
        customer_id="CUST123",
        status="shipped",
        total_amount=299.99,
        items=["Laptop Case", "USB Cable"]
    )
    return mock_order

@tool
async def process_refund_tool(refund_request: RefundRequest) -> dict:
    """Process a refund request"""
    if refund_request.amount > 500:
        return {
            "status": "requires_approval",
            "message": "Refund amount requires manager approval",
            "request_id": f"REF_{refund_request.order_id}"
        }

    return {
        "status": "processed",
        "refund_amount": refund_request.amount,
        "estimated_days": "3-5 business days"
    }
```

---

## Question 2: Content Moderation and Publishing Workflow (Intermediate)

### Scenario

You're building a content management system for a digital marketing agency. The system needs to:

- Review and moderate user-generated content
- Enhance content with SEO optimization
- Ensure brand compliance
- Schedule publication across multiple platforms

**Requirements:**

- Multi-stage content pipeline with specialized agents
- Content quality guardrails
- Dynamic instructions based on content type
- Comprehensive workflow tracing

### Implementation Guide

#### Step 1: Content Models and Workflow

```python
from pydantic import BaseModel, Field
from typing import Literal, List
from datetime import datetime

class ContentItem(BaseModel):
    content_id: str
    content_type: Literal["blog_post", "social_media", "email", "advertisement"]
    title: str
    body: str
    target_audience: str
    brand_guidelines: dict
    status: Literal["draft", "under_review", "approved", "rejected", "published"] = "draft"

class ModerationResult(BaseModel):
    approved: bool
    issues_found: List[str] = []
    severity: Literal["none", "minor", "major", "critical"] = "none"
    recommended_changes: List[str] = []

class SEOEnhancement(BaseModel):
    keywords: List[str]
    meta_description: str
    suggested_title: str
    readability_score: float
    seo_recommendations: List[str]
```

#### Step 2: Specialized Content Agents

```python
# Content Moderator Agent
moderator_agent = Agent(
    name="Content Moderator",
    instructions="""
    Review content for compliance with brand guidelines, legal requirements, and platform policies.
    Check for inappropriate language, factual accuracy, and brand consistency.
    Flag content that requires revision or escalation.
    """,
    output_type=ModerationResult,
    tools=[check_brand_compliance_tool, fact_check_tool, sentiment_analysis_tool]
)

# SEO Optimizer Agent
seo_agent = Agent(
    name="SEO Optimizer",
    instructions="""
    Enhance content for search engine optimization while maintaining readability.
    Suggest keyword improvements, meta descriptions, and structural changes.
    Ensure content meets SEO best practices for the target platform.
    """,
    output_type=SEOEnhancement,
    tools=[keyword_research_tool, readability_analyzer_tool, competitor_analysis_tool]
)

# Publication Manager Agent
publisher_agent = Agent(
    name="Publication Manager",
    instructions="""
    Handle final content preparation and cross-platform publishing.
    Format content for different platforms and schedule publication.
    Monitor publication success and handle any platform-specific requirements.
    """,
    tools=[format_for_platform_tool, schedule_publication_tool, monitor_performance_tool]
)

# Content Workflow Orchestrator
workflow_agent = Agent(
    name="Content Workflow Orchestrator",
    instructions=prompt_with_handoff_instructions("""
    Manage the complete content lifecycle from submission to publication.
    Route content through appropriate review stages based on type and priority.
    Coordinate between moderation, optimization, and publication teams.
    """),
    handoffs=[
        handoff(moderator_agent, tool_description_override="Send to content moderation"),
        handoff(seo_agent, tool_description_override="Send for SEO optimization"),
        handoff(publisher_agent, tool_description_override="Send to publication manager")
    ]
)
```

#### Step 3: Content Quality Guardrails

```python
@input_guardrail
async def content_length_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    content: ContentItem
) -> GuardrailFunctionOutput:
    """Ensure content meets minimum quality standards"""
    issues = []

    if len(content.body) < 100:
        issues.append("Content too short (minimum 100 characters)")

    if len(content.title) > 60:
        issues.append("Title too long for SEO (maximum 60 characters)")

    if not content.target_audience:
        issues.append("Target audience not specified")

    return GuardrailFunctionOutput(
        output_info={"quality_check": issues},
        tripwire_triggered=len(issues) > 0
    )

@output_guardrail
async def brand_safety_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:
    """Ensure output maintains brand safety"""
    # Check for sensitive topics or inappropriate content
    sensitive_keywords = ["controversial", "political", "inappropriate"]

    flagged = any(keyword in output.lower() for keyword in sensitive_keywords)

    return GuardrailFunctionOutput(
        output_info={"brand_safety_check": "flagged" if flagged else "passed"},
        tripwire_triggered=flagged
    )
```

#### Step 4: Dynamic Content Processing

```python
async def process_content_submission(content: ContentItem, priority: str = "normal"):
    """Process content through the workflow with dynamic instructions"""

    # Dynamic instruction modification based on content type
    content_specific_instructions = {
        "blog_post": "Focus on long-form content optimization and detailed SEO analysis",
        "social_media": "Prioritize engagement optimization and platform-specific formatting",
        "email": "Ensure compliance with email marketing regulations and personalization",
        "advertisement": "Strict brand compliance and legal review required"
    }

    # Update agent instructions dynamically
    if content.content_type in content_specific_instructions:
        workflow_agent.instructions += f"\n\nSpecial instructions for {content.content_type}: {content_specific_instructions[content.content_type]}"

    with trace(
        workflow_name="Content Publishing Pipeline",
        metadata={
            "content_id": content.content_id,
            "content_type": content.content_type,
            "priority": priority,
            "submission_time": datetime.now().isoformat()
        }
    ) as workflow_trace:

        config = RunConfig(
            workflow_name=f"Content Processing - {content.content_type}",
            trace_include_sensitive_data=False
        )

        try:
            result = await Runner.run(
                workflow_agent,
                content.model_dump_json(),
                config=config
            )

            return {
                "status": "processed",
                "content_id": content.content_id,
                "result": result.final_output,
                "trace_id": workflow_trace.trace_id
            }

        except Exception as e:
            return {
                "status": "error",
                "content_id": content.content_id,
                "error": str(e)
            }
```

---

## Question 3: Financial Analysis Multi-Agent Research System (Intermediate)

### Scenario

You're developing a financial research platform that analyzes investment opportunities. The system needs to:

- Gather market data from multiple sources
- Perform technical and fundamental analysis
- Generate risk assessments
- Create investment recommendations with compliance checks

**Requirements:**

- Coordinate multiple research agents
- Implement data validation guardrails
- Use sophisticated handoff patterns with data passing
- Provide comprehensive audit trails via tracing

### Implementation Guide

#### Step 1: Financial Data Models

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from decimal import Decimal
from datetime import datetime, date

class MarketData(BaseModel):
    symbol: str
    price: Decimal
    volume: int
    market_cap: Optional[Decimal] = None
    pe_ratio: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class TechnicalAnalysis(BaseModel):
    symbol: str
    trend: Literal["bullish", "bearish", "neutral"]
    support_levels: List[Decimal]
    resistance_levels: List[Decimal]
    indicators: Dict[str, float]
    confidence_score: float = Field(ge=0.0, le=1.0)

class RiskAssessment(BaseModel):
    symbol: str
    risk_level: Literal["low", "medium", "high", "very_high"]
    volatility_score: float
    beta: float
    risk_factors: List[str]
    var_95: Optional[Decimal] = None  # Value at Risk

class InvestmentRecommendation(BaseModel):
    symbol: str
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    target_price: Decimal
    time_horizon: Literal["short_term", "medium_term", "long_term"]
    reasoning: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    compliance_approved: bool = False
```

#### Step 2: Specialized Financial Agents

```python
# Market Data Collector
data_collector_agent = Agent(
    name="Market Data Collector",
    instructions="""
    Gather comprehensive market data for requested securities.
    Collect real-time prices, historical data, volume, and key financial metrics.
    Validate data quality and flag any inconsistencies.
    """,
    output_type=MarketData,
    tools=[fetch_market_data_tool, validate_data_quality_tool, get_company_fundamentals_tool]
)

# Technical Analyst
technical_analyst_agent = Agent(
    name="Technical Analysis Specialist",
    instructions="""
    Perform comprehensive technical analysis using various indicators and chart patterns.
    Identify support/resistance levels, trends, and momentum signals.
    Provide confidence scores for all analyses.
    """,
    output_type=TechnicalAnalysis,
    tools=[calculate_technical_indicators_tool, identify_chart_patterns_tool, trend_analysis_tool]
)

# Risk Assessment Specialist
risk_analyst_agent = Agent(
    name="Risk Assessment Specialist",
    instructions="""
    Evaluate investment risks including market, credit, and operational risks.
    Calculate volatility metrics, beta, and Value at Risk (VaR).
    Identify and assess specific risk factors for the security.
    """,
    output_type=RiskAssessment,
    tools=[calculate_volatility_tool, assess_credit_risk_tool, market_risk_analysis_tool]
)

# Investment Advisor
investment_advisor_agent = Agent(
    name="Investment Advisor",
    instructions="""
    Synthesize market data, technical analysis, and risk assessment to generate investment recommendations.
    Consider client suitability, market conditions, and risk tolerance.
    Provide clear reasoning for recommendations.
    """,
    output_type=InvestmentRecommendation,
    tools=[generate_recommendation_tool, assess_client_suitability_tool]
)

# Compliance Officer
compliance_agent = Agent(
    name="Compliance Officer",
    instructions="""
    Review investment recommendations for regulatory compliance.
    Ensure recommendations meet fiduciary standards and disclosure requirements.
    Flag any potential conflicts of interest or regulatory issues.
    """,
    tools=[compliance_check_tool, regulatory_review_tool, conflict_check_tool]
)

# Research Coordinator
research_coordinator_agent = Agent(
    name="Financial Research Coordinator",
    instructions=prompt_with_handoff_instructions("""
    Orchestrate comprehensive financial research by coordinating specialized analysis teams.
    Ensure systematic progression: Data Collection → Technical Analysis → Risk Assessment → Investment Recommendation → Compliance Review.
    Manage information flow between teams and compile final research reports.
    """),
    handoffs=[
        handoff(data_collector_agent, tool_description_override="Initiate market data collection"),
        handoff(technical_analyst_agent, tool_description_override="Perform technical analysis",
                input_type=MarketData),
        handoff(risk_analyst_agent, tool_description_override="Conduct risk assessment",
                input_type=MarketData),
        handoff(investment_advisor_agent, tool_description_override="Generate investment recommendation"),
        handoff(compliance_agent, tool_description_override="Compliance review and approval",
                input_type=InvestmentRecommendation)
    ]
)
```

#### Step 3: Financial Data Validation Guardrails

```python
@input_guardrail
async def market_data_validation_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    market_data: MarketData
) -> GuardrailFunctionOutput:
    """Validate market data quality and completeness"""
    validation_issues = []

    # Price validation
    if market_data.price <= 0:
        validation_issues.append("Invalid price: must be positive")

    # Volume validation
    if market_data.volume < 0:
        validation_issues.append("Invalid volume: cannot be negative")

    # P/E ratio validation
    if market_data.pe_ratio and (market_data.pe_ratio < 0 or market_data.pe_ratio > 1000):
        validation_issues.append("Suspicious P/E ratio: outside normal range")

    # Data freshness check
    data_age = datetime.now() - market_data.timestamp
    if data_age.total_seconds() > 3600:  # 1 hour
        validation_issues.append("Stale data: market data is over 1 hour old")

    return GuardrailFunctionOutput(
        output_info={"validation_issues": validation_issues},
        tripwire_triggered=len(validation_issues) > 0
    )

@output_guardrail
async def recommendation_sanity_check_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    recommendation: InvestmentRecommendation
) -> GuardrailFunctionOutput:
    """Ensure investment recommendations are reasonable"""
    sanity_issues = []

    # Confidence level check
    if recommendation.confidence_level < 0.3:
        sanity_issues.append("Low confidence recommendation may not be actionable")

    # Target price reasonableness (basic check)
    if recommendation.target_price <= 0:
        sanity_issues.append("Invalid target price: must be positive")

    # Strong recommendations require high confidence
    strong_recs = ["strong_buy", "strong_sell"]
    if recommendation.recommendation in strong_recs and recommendation.confidence_level < 0.7:
        sanity_issues.append("Strong recommendations require high confidence (>70%)")

    return GuardrailFunctionOutput(
        output_info={"sanity_check": sanity_issues},
        tripwire_triggered=len(sanity_issues) > 2  # Allow minor issues
    )
```

#### Step 4: Comprehensive Research Workflow

```python
async def conduct_financial_research(symbol: str, client_profile: dict, research_depth: str = "standard"):
    """Conduct comprehensive financial research with full audit trail"""

    research_id = f"RESEARCH_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with trace(
        workflow_name="Financial Research Analysis",
        group_id=research_id,
        metadata={
            "symbol": symbol,
            "client_profile": client_profile,
            "research_depth": research_depth,
            "start_time": datetime.now().isoformat()
        }
    ) as research_trace:

        config = RunConfig(
            workflow_name=f"Financial Research - {symbol}",
            trace_include_sensitive_data=False  # Protect client information
        )

        try:
            # Step 1: Data Collection with validation
            with custom_span("market_data_collection") as data_span:
                market_data_result = await Runner.run(
                    data_collector_agent,
                    f"Collect comprehensive market data for {symbol}",
                    config=config
                )
                data_span.add_event("data_collected", {"symbol": symbol})

            # Step 2: Parallel Technical and Risk Analysis
            with custom_span("parallel_analysis") as parallel_span:
                # Technical Analysis
                technical_result = await Runner.run(
                    technical_analyst_agent,
                    market_data_result.final_output,
                    config=config
                )

                # Risk Analysis
                risk_result = await Runner.run(
                    risk_analyst_agent,
                    market_data_result.final_output,
                    config=config
                )

                parallel_span.add_event("analysis_completed", {
                    "technical_confidence": technical_result.final_output.confidence_score,
                    "risk_level": risk_result.final_output.risk_level
                })

            # Step 3: Investment Recommendation
            combined_analysis = {
                "market_data": market_data_result.final_output,
                "technical_analysis": technical_result.final_output,
                "risk_assessment": risk_result.final_output,
                "client_profile": client_profile
            }

            recommendation_result = await Runner.run(
                investment_advisor_agent,
                json.dumps(combined_analysis),
                config=config
            )

            # Step 4: Compliance Review
            final_result = await Runner.run(
                compliance_agent,
                recommendation_result.final_output,
                config=config
            )

            return {
                "research_id": research_id,
                "symbol": symbol,
                "market_data": market_data_result.final_output,
                "technical_analysis": technical_result.final_output,
                "risk_assessment": risk_result.final_output,
                "recommendation": recommendation_result.final_output,
                "compliance_status": final_result.final_output,
                "trace_id": research_trace.trace_id,
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            # Comprehensive error handling with trace context
            error_span = custom_span("research_error")
            error_span.add_event("error_occurred", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "symbol": symbol
            })

            return {
                "research_id": research_id,
                "status": "error",
                "error": str(e),
                "trace_id": research_trace.trace_id
            }
```

---

## Question 4: Enterprise Document Processing and Knowledge Management System (Advanced)

### Scenario

You're architecting an enterprise-grade document processing system for a large consulting firm. The system must:

- Process multiple document types (PDFs, Word docs, emails, contracts)
- Extract and classify information using NLP
- Implement multi-level security and access controls
- Create searchable knowledge base with intelligent categorization
- Handle sensitive client information with appropriate guardrails
- Support real-time collaboration and version control
- Provide comprehensive audit trails for compliance

**Requirements:**

- Complex multi-agent orchestration with conditional flows
- Advanced guardrails for data privacy and security
- Dynamic agent scaling based on workload
- Integration with external systems (databases, APIs)
- Sophisticated error handling and recovery mechanisms
- Custom tracing processors for enterprise monitoring

### Implementation Guide

#### Step 1: Enterprise Data Models and Security Framework

```python
from pydantic import BaseModel, Field, SecretStr, validator
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from uuid import uuid4

class SecurityClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DocumentMetadata(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    document_type: Literal["pdf", "docx", "email", "contract", "report", "presentation"]
    security_classification: SecurityClassification
    client_id: Optional[str] = None
    project_id: Optional[str] = None
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: int
    checksum: str
    access_permissions: List[str] = []
    retention_policy: Optional[str] = None

class ExtractedContent(BaseModel):
    document_id: str
    text_content: str
    structured_data: Dict[str, Any] = {}
    entities: List[Dict[str, str]] = []  # Named entities
    key_phrases: List[str] = []
    sentiment_score: Optional[float] = None
    language: str = "en"
    confidence_score: float = Field(ge=0.0, le=1.0)

class DocumentClassification(BaseModel):
    document_id: str
    primary_category: str
    secondary_categories: List[str] = []
    topics: List[str] = []
    client_relevance: Optional[str] = None
    business_value: Literal["low", "medium", "high", "critical"]
    recommended_actions: List[str] = []

class KnowledgeBaseEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    title: str
    summary: str
    key_insights: List[str]
    tags: List[str]
    related_documents: List[str] = []
    search_keywords: List[str]
    access_level: SecurityClassification
    indexed_at: datetime = Field(default_factory=datetime.now)

class ComplianceReport(BaseModel):
    document_id: str
    compliance_status: Literal["compliant", "requires_review", "non_compliant"]
    violations_found: List[str] = []
    recommendations: List[str] = []
    reviewer_id: str
    review_timestamp: datetime = Field(default_factory=datetime.now)
    next_review_date: Optional[datetime] = None
```

#### Step 2: Advanced Document Processing Agents

```python
# Document Ingestion and Security Agent
document_ingestion_agent = Agent(
    name="Document Ingestion Security Controller",
    instructions="""
    Perform initial document intake with comprehensive security screening.
    Validate file integrity, scan for malware, and classify security levels.
    Apply appropriate access controls and encryption based on content sensitivity.
    Extract basic metadata and prepare documents for processing pipeline.
    """,
    output_type=DocumentMetadata,
    tools=[
        security_scan_tool,
        extract_metadata_tool,
        apply_encryption_tool,
        validate_file_integrity_tool
    ]
)

# Advanced Content Extraction Agent
content_extraction_agent = Agent(
    name="Advanced Content Extraction Specialist",
    instructions="""
    Extract and structure content from various document formats using advanced NLP.
    Perform named entity recognition, key phrase extraction, and sentiment analysis.
    Handle OCR for scanned documents and extract structured data from forms/tables.
    Maintain extraction confidence scores and flag low-quality extractions.
    """,
    output_type=ExtractedContent,
    tools=[
        ocr_extraction_tool,
        nlp_analysis_tool,
        table_extraction_tool,
        entity_recognition_tool,
        sentiment_analysis_tool
    ]
)

# Intelligent Classification Agent
classification_agent = Agent(
    name="Intelligent Document Classifier",
    instructions="""
    Classify documents using machine learning models and business rule engines.
    Categorize by document type, business function, client relevance, and value.
    Identify key topics, themes, and recommend appropriate handling procedures.
    Consider business context and client-specific classification rules.
    """,
    output_type=DocumentClassification,
    tools=[
        ml_classification_tool,
        topic_modeling_tool,
        business_rule_engine_tool,
        client_context_tool
    ]
)

# Knowledge Management Agent
knowledge_management_agent = Agent(
    name="Enterprise Knowledge Manager",
    instructions="""
    Create comprehensive knowledge base entries from processed documents.
    Generate executive summaries, extract key insights, and establish relationships.
    Optimize content for search and discovery while maintaining security controls.
    Create knowledge graphs and semantic relationships between documents.
    """,
    output_type=KnowledgeBaseEntry,
    tools=[
        summarization_tool,
        insight_extraction_tool,
        relationship_mapping_tool,
        search_optimization_tool,
        knowledge_graph_tool
    ]
)

# Compliance and Audit Agent
compliance_agent = Agent(
    name="Compliance and Audit Specialist",
    instructions="""
    Perform comprehensive compliance reviews for regulatory requirements.
    Check for sensitive information, privacy violations, and retention policies.
    Generate audit trails and compliance reports for enterprise governance.
    Flag documents requiring legal review or special handling.
    """,
    output_type=ComplianceReport,
    tools=[
        privacy_scan_tool,
        regulatory_check_tool,
        audit_trail_tool,
        legal_review_flag_tool,
        retention_policy_tool
    ]
)

# Enterprise Document Orchestrator
document_orchestrator_agent = Agent(
    name="Enterprise Document Processing Orchestrator",
    instructions=prompt_with_handoff_instructions("""
    Orchestrate enterprise document processing workflows with dynamic routing.
    Coordinate security screening, content extraction, classification, and knowledge management.
    Handle error recovery, quality assurance, and compliance requirements.
    Scale processing based on document volume and priority levels.
    Ensure all enterprise governance and security policies are enforced.
    """),
    handoffs=[
        handoff(
            document_ingestion_agent,
            tool_description_override="Initiate secure document ingestion",
            on_handoff=log_document_intake
        ),
        handoff(
            content_extraction_agent,
            tool_description_override="Extract and analyze document content",
            input_type=DocumentMetadata,
            input_filter=filter_sensitive_metadata
        ),
        handoff(
            classification_agent,
            tool_description_override="Classify and categorize document",
            input_type=ExtractedContent
        ),
        handoff(
            knowledge_management_agent,
            tool_description_override="Create knowledge base entry",
            input_type=DocumentClassification
        ),
        handoff(
            compliance_agent,
            tool_description_override="Perform compliance review",
            input_type=KnowledgeBaseEntry
        )
    ]
)
```

#### Step 3: Enterprise-Grade Guardrails and Security

```python
# Advanced Security Guardrails
@input_guardrail
async def enterprise_security_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    document_data: Any
) -> GuardrailFunctionOutput:
    """Comprehensive enterprise security screening"""
    security_issues = []
    critical_violations = []

    # Document metadata security check
    if isinstance(document_data, DocumentMetadata):
        # Check file size limits
        if document_data.file_size > 100 * 1024 * 1024:  # 100MB limit
            security_issues.append("Document exceeds size limit")

        # Verify security classification
        if document_data.security_classification == SecurityClassification.RESTRICTED:
            # Additional checks for restricted documents
            if not document_data.access_permissions:
                critical_violations.append("Restricted document missing access permissions")

        # Client data validation
        if document_data.client_id and document_data.security_classification == SecurityClassification.PUBLIC:
            security_issues.append("Client document cannot be classified as public")

    # Content security screening
    if isinstance(document_data, str):
        # Scan for PII patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]

        import re
        for pattern in pii_patterns:
            if re.search(pattern, document_data):
                critical_violations.append(f"PII detected matching pattern: {pattern}")

    return GuardrailFunctionOutput(
        output_info={
            "security_scan": {
                "issues": security_issues,
                "critical_violations": critical_violations,
                "scan_timestamp": datetime.now().isoformat()
            }
        },
        tripwire_triggered=len(critical_violations) > 0
    )

@output_guardrail
async def data_classification_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """Ensure proper data classification and handling"""
    classification_issues = []

    if isinstance(output, KnowledgeBaseEntry):
        # Verify access level consistency
        if output.access_level == SecurityClassification.RESTRICTED:
            if len(output.search_keywords) > 5:
                classification_issues.append("Restricted documents should have limited search keywords")

        # Check for over-sharing of sensitive information
        sensitive_terms = ["confidential", "proprietary", "trade secret", "classified"]
        if any(term in output.summary.lower() for term in sensitive_terms):
            if output.access_level in [SecurityClassification.PUBLIC, SecurityClassification.INTERNAL]:
                classification_issues.append("Sensitive content requires higher security classification")

    return GuardrailFunctionOutput(
        output_info={"classification_check": classification_issues},
        tripwire_triggered=len(classification_issues) > 0
    )

# Custom Input Filters for Handoffs
async def filter_sensitive_metadata(input_data: HandoffInputData) -> HandoffInputData:
    """Filter sensitive information during handoffs"""
    # Remove sensitive fields before passing to next agent
    if hasattr(input_data, 'access_permissions'):
        # Create filtered copy without sensitive permission details
        filtered_data = input_data.copy()
        filtered_data.access_permissions = ["[FILTERED]"]
        return filtered_data
    return input_data
```

#### Step 4: Custom Enterprise Tracing and Monitoring

```python
from agents.tracing import add_trace_processor, TraceProcessor, Trace, Span
import json

class EnterpriseAuditProcessor(TraceProcessor):
    """Custom trace processor for enterprise audit and compliance"""

    def __init__(self, audit_database_url: str, compliance_webhook: str):
        self.audit_db_url = audit_database_url
        self.compliance_webhook = compliance_webhook

    async def process_trace(self, trace: Trace) -> None:
        """Process completed traces for audit logging"""
        audit_record = {
            "trace_id": trace.trace_id,
            "workflow_name": trace.workflow_name,
            "start_time": trace.started_at.isoformat(),
            "end_time": trace.ended_at.isoformat() if trace.ended_at else None,
            "duration_ms": trace.duration_ms,
            "metadata": trace.metadata,
            "compliance_flags": []
        }

        # Analyze spans for compliance issues
        for span in trace.spans:
            if hasattr(span, 'span_data') and span.span_data:
                if "security_violation" in str(span.span_data).lower():
                    audit_record["compliance_flags"].append({
                        "span_id": span.span_id,
                        "issue": "security_violation_detected",
                        "timestamp": span.started_at.isoformat()
                    })

        # Store in enterprise audit database
        await self._store_audit_record(audit_record)

        # Send compliance alerts if needed
        if audit_record["compliance_flags"]:
            await self._send_compliance_alert(audit_record)

    async def _store_audit_record(self, record: dict):
        """Store audit record in enterprise database"""
        # Implementation would integrate with enterprise database
        print(f"Storing audit record: {record['trace_id']}")

    async def _send_compliance_alert(self, record: dict):
        """Send compliance alerts to monitoring systems"""
        # Implementation would integrate with enterprise alerting
        print(f"COMPLIANCE ALERT: {record['trace_id']} - {len(record['compliance_flags'])} issues")

# Setup enterprise tracing
enterprise_processor = EnterpriseAuditProcessor(
    audit_database_url="postgresql://audit-db:5432/compliance",
    compliance_webhook="https://compliance.company.com/alerts"
)
add_trace_processor(enterprise_processor)
```

#### Step 5: Complete Enterprise Document Processing Workflow

```python
async def process_enterprise_document(
    file_path: str,
    user_id: str,
    client_context: dict,
    processing_priority: Literal["low", "normal", "high", "urgent"] = "normal"
) -> dict:
    """
    Complete enterprise document processing with full compliance and audit trail
    """

    workflow_id = f"DOC_PROC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    # Enhanced tracing with enterprise metadata
    with trace(
        workflow_name="Enterprise Document Processing",
        group_id=workflow_id,
        metadata={
            "file_path": file_path,
            "user_id": user_id,
            "client_id": client_context.get("client_id"),
            "project_id": client_context.get("project_id"),
            "processing_priority": processing_priority,
            "compliance_level": client_context.get("compliance_level", "standard"),
            "workflow_version": "2.1.0",
            "environment": "production"
        }
    ) as enterprise_trace:

        # Dynamic configuration based on priority and client requirements
        config = RunConfig(
            workflow_name=f"Enterprise Document Processing - {processing_priority.upper()}",
            trace_include_sensitive_data=False,
            max_retry_attempts=3 if processing_priority in ["high", "urgent"] else 1
        )

        processing_result = {
            "workflow_id": workflow_id,
            "status": "processing",
            "stages_completed": [],
            "processing_start": datetime.now().isoformat(),
            "errors": [],
            "compliance_status": "pending"
        }

        try:
            # Stage 1: Secure Document Ingestion
            with custom_span("secure_document_ingestion") as ingestion_span:
                ingestion_span.add_event("ingestion_started", {
                    "file_path": file_path,
                    "user_id": user_id,
                    "priority": processing_priority
                })

                initial_input = {
                    "file_path": file_path,
                    "user_context": {"user_id": user_id, "timestamp": datetime.now().isoformat()},
                    "client_context": client_context,
                    "processing_requirements": {
                        "priority": processing_priority,
                        "compliance_level": client_context.get("compliance_level", "standard")
                    }
                }

                result = await Runner.run(
                    document_orchestrator_agent,
                    json.dumps(initial_input),
                    config=config
                )

                processing_result["stages_completed"].append("ingestion")
                processing_result["document_metadata"] = result.metadata

                ingestion_span.add_event("ingestion_completed", {
                    "document_id": result.metadata.get("document_id"),
                    "security_classification": result.metadata.get("security_classification")
                })

            # Stage 2: Quality Assurance and Validation
            with custom_span("quality_assurance") as qa_span:
                qa_validation = await validate_processing_quality(result, client_context)

                if not qa_validation["passed"]:
                    processing_result["errors"].extend(qa_validation["issues"])

                    # Implement retry logic for high-priority documents
                    if processing_priority in ["high", "urgent"] and qa_validation["retryable"]:
                        qa_span.add_event("quality_retry_initiated", {
                            "retry_reason": qa_validation["issues"][0],
                            "retry_attempt": 1
                        })

                        # Retry with enhanced configuration
                        enhanced_config = config.copy()
                        enhanced_config.workflow_name += " - RETRY"

                        result = await Runner.run(
                            document_orchestrator_agent,
                            json.dumps(initial_input),
                            config=enhanced_config
                        )

                processing_result["stages_completed"].append("quality_assurance")
                processing_result["qa_status"] = qa_validation

            # Stage 3: Enterprise Integration and Indexing
            with custom_span("enterprise_integration") as integration_span:
                integration_result = await integrate_with_enterprise_systems(
                    result,
                    client_context,
                    processing_priority
                )

                processing_result["stages_completed"].append("enterprise_integration")
                processing_result["integration_status"] = integration_result

                integration_span.add_event("enterprise_integration_completed", {
                    "systems_updated": integration_result.get("systems_updated", []),
                    "knowledge_base_id": integration_result.get("knowledge_base_id")
                })

            # Final Status Update
            processing_result.update({
                "status": "completed",
                "processing_end": datetime.now().isoformat(),
                "final_output": result.final_output,
                "trace_id": enterprise_trace.trace_id,
                "compliance_status": "approved" if not processing_result["errors"] else "requires_review"
            })

            # Calculate processing metrics
            start_time = datetime.fromisoformat(processing_result["processing_start"])
            end_time = datetime.fromisoformat(processing_result["processing_end"])
            processing_result["processing_duration_seconds"] = (end_time - start_time).total_seconds()

            return processing_result

        except Exception as e:
            # Comprehensive error handling with enterprise logging
            error_span = custom_span("enterprise_error_handling")
            error_span.add_event("critical_error_occurred", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "workflow_id": workflow_id,
                "user_id": user_id,
                "client_id": client_context.get("client_id")
            })

            # Implement error escalation for high-priority processing
            if processing_priority in ["high", "urgent"]:
                await escalate_processing_error(workflow_id, str(e), user_id, client_context)

            processing_result.update({
                "status": "error",
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_timestamp": datetime.now().isoformat()
                },
                "trace_id": enterprise_trace.trace_id,
                "compliance_status": "failed"
            })

            return processing_result

# Supporting functions for enterprise integration
async def validate_processing_quality(result: dict, client_context: dict) -> dict:
    """Validate processing quality against enterprise standards"""
    validation_issues = []

    # Check extraction confidence scores
    if result.get("confidence_score", 0) < 0.8:
        validation_issues.append("Low extraction confidence")

    # Validate client-specific requirements
    if client_context.get("requires_manual_review") and not result.get("manual_review_flag"):
        validation_issues.append("Manual review required but not flagged")

    return {
        "passed": len(validation_issues) == 0,
        "issues": validation_issues,
        "retryable": True if validation_issues else False
    }

async def integrate_with_enterprise_systems(result: dict, client_context: dict, priority: str) -> dict:
    """Integrate processed document with enterprise systems"""
    integration_status = {
        "systems_updated": [],
        "knowledge_base_id": None,
        "search_indexed": False,
        "client_portal_updated": False
    }

    try:
        # Update knowledge management system
        knowledge_id = await update_knowledge_management_system(result)
        integration_status["knowledge_base_id"] = knowledge_id
        integration_status["systems_updated"].append("knowledge_management")

        # Update search index
        await update_enterprise_search_index(result, knowledge_id)
        integration_status["search_indexed"] = True
        integration_status["systems_updated"].append("search_index")

        # Update client portal if applicable
        if client_context.get("client_portal_access"):
            await update_client_portal(result, client_context)
            integration_status["client_portal_updated"] = True
            integration_status["systems_updated"].append("client_portal")

    except Exception as e:
        integration_status["error"] = str(e)

    return integration_status

async def escalate_processing_error(workflow_id: str, error: str, user_id: str, client_context: dict):
    """Escalate processing errors for high-priority documents"""
    escalation_data = {
        "workflow_id": workflow_id,
        "error": error,
        "user_id": user_id,
        "client_id": client_context.get("client_id"),
        "escalation_timestamp": datetime.now().isoformat()
    }

    # Send to enterprise monitoring and alerting systems
    print(f"ESCALATION: Critical document processing error - {workflow_id}")
    # Implementation would integrate with enterprise systems
```

### Testing and Validation Scenarios

Each practice question includes comprehensive testing scenarios:

1. **Unit Testing**: Test individual agent functions and guardrails
2. **Integration Testing**: Validate handoff patterns and data flow
3. **Performance Testing**: Measure processing speed and resource usage
4. **Security Testing**: Verify guardrails and access controls
5. **Compliance Testing**: Ensure audit trails and regulatory compliance
6. **Error Handling Testing**: Validate recovery mechanisms and escalation procedures

### Expected Outcomes

After completing these practice questions, you should be able to:

- Design and implement complex multi-agent systems using the OpenAI Agents SDK
- Apply appropriate guardrails for different business scenarios
- Implement sophisticated handoff patterns with data validation
- Create comprehensive tracing and monitoring solutions
- Handle enterprise-grade security and compliance requirements
- Design scalable agent architectures for production environments

These scenarios represent real-world professional use cases that demonstrate mastery of the OpenAI Agents SDK across different industries and complexity levels.
