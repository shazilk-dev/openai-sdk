# OpenAI Agents SDK Practice Questions Guide - Implementation Requirements

This guide provides the requirements, specifications, and architectural guidance for implementing real-world professional agent scenarios using the OpenAI Agents SDK. Each question includes detailed requirements but **NO IMPLEMENTATION CODE** - you must implement the solutions yourself.

## Question 1: E-commerce Customer Service Multi-Agent System (Intermediate)

### Business Scenario

Build a comprehensive customer service system for an e-commerce platform that handles various customer inquiries efficiently through specialized agent routing.

### Core Requirements

#### System Capabilities

- **Order Management**: Handle order status checks, modifications, and cancellations
- **Product Support**: Provide product information and recommendations
- **Refund Processing**: Process return and refund requests with proper authorization
- **Technical Escalation**: Route complex technical issues to appropriate support levels

#### Agent Architecture Requirements

- **Triage Agent**: Main entry point that analyzes customer requests and routes appropriately
- **Order Management Agent**: Specializes in order-related operations
- **Refund Specialist Agent**: Handles refund processing with authorization controls
- **Technical Support Agent**: Manages technical issues and escalations

#### Data Models Needed

1. **OrderInfo Model**

   - Fields: order_id, customer_id, status, total_amount, items
   - Status types: pending, shipped, delivered, cancelled

2. **RefundRequest Model**

   - Fields: order_id, amount, reason, urgency
   - Urgency levels: low, medium, high

3. **EscalationData Model**
   - Fields: issue_type, priority, customer_id, description
   - Priority levels: low, medium, high, critical

#### Handoff Strategy

- **Basic Handoffs**: Simple agent-to-agent transfers
- **Typed Handoffs**: Use input_type for structured data passing
- **Conditional Routing**: Route based on request amount, complexity, or urgency
- **Handoff Descriptions**: Custom tool descriptions for clarity

#### Guardrail Requirements

1. **Input Guardrail for Refund Agent**

   - Purpose: Prevent unauthorized high-value refund attempts
   - Trigger: Refund requests over $500
   - Action: Block processing and require additional authorization
   - Implementation: Parse input for monetary amounts using regex

2. **General Security Guardrail**
   - Validate customer data integrity
   - Check for suspicious patterns
   - Ensure proper authentication context

#### Tracing Specifications

- **Workflow Name**: "E-commerce Customer Service"
- **Group ID**: Use customer_id for session tracking
- **Metadata Requirements**:
  - customer_id
  - timestamp
  - request_type
  - priority_level
- **Sensitive Data**: Disable sensitive data tracing for customer privacy
- **Error Handling**: Comprehensive error logging with customer-friendly responses

#### Tool Requirements

1. **Order Management Tools**

   - `lookup_order_tool`: Retrieve order information
   - `modify_order_tool`: Update order details
   - `cancel_order_tool`: Cancel orders with proper checks

2. **Refund Processing Tools**

   - `process_refund_tool`: Handle refund transactions
   - `verify_refund_eligibility_tool`: Check refund eligibility

3. **Technical Support Tools**
   - `check_account_status_tool`: Verify account status
   - `reset_password_tool`: Handle password resets
   - `escalate_to_human_tool`: Route to human agents

#### Success Criteria

- Proper routing of 95%+ customer requests
- Guardrails prevent unauthorized high-value transactions
- Complete audit trail for all customer interactions
- Response time under 30 seconds for standard requests

---

## Question 2: Content Moderation and Publishing Workflow (Intermediate)

### Business Scenario

Develop a content management system for a digital marketing agency that processes, moderates, optimizes, and publishes content across multiple platforms.

### Core Requirements

#### System Capabilities

- **Content Moderation**: Review content for compliance and quality
- **SEO Optimization**: Enhance content for search engine performance
- **Brand Compliance**: Ensure adherence to brand guidelines
- **Multi-Platform Publishing**: Format and schedule content across platforms

#### Agent Architecture Requirements

- **Content Workflow Orchestrator**: Manages the complete content lifecycle
- **Content Moderator Agent**: Reviews content for compliance and quality
- **SEO Optimizer Agent**: Enhances content for search performance
- **Publication Manager Agent**: Handles final formatting and publishing

#### Data Models Needed

1. **ContentItem Model**

   - Fields: content_id, content_type, title, body, target_audience, brand_guidelines, status
   - Content types: blog_post, social_media, email, advertisement
   - Status types: draft, under_review, approved, rejected, published

2. **ModerationResult Model**

   - Fields: approved, issues_found, severity, recommended_changes
   - Severity levels: none, minor, major, critical

3. **SEOEnhancement Model**
   - Fields: keywords, meta_description, suggested_title, readability_score, seo_recommendations

#### Dynamic Instruction Strategy

- **Content Type Adaptation**: Modify agent instructions based on content type
- **Platform Optimization**: Adjust processing based on target platform
- **Brand Guidelines**: Apply client-specific brand rules dynamically
- **Priority Handling**: Fast-track urgent content through the pipeline

#### Guardrail Requirements

1. **Content Quality Guardrail**

   - Minimum content length validation (100+ characters)
   - Title length optimization (under 60 characters for SEO)
   - Target audience specification requirement
   - Trigger on quality threshold violations

2. **Brand Safety Guardrail**
   - Scan for sensitive or inappropriate topics
   - Check against brand-specific prohibited terms
   - Validate compliance with industry regulations
   - Flag content requiring legal review

#### Workflow Orchestration

- **Sequential Processing**: Moderation → SEO → Publishing
- **Quality Gates**: Each stage must pass before proceeding
- **Error Recovery**: Retry mechanisms for failed processing
- **Parallel Processing**: Handle multiple content pieces simultaneously

#### Tracing Specifications

- **Workflow Name**: "Content Publishing Pipeline"
- **Metadata Requirements**:
  - content_id
  - content_type
  - priority_level
  - submission_time
  - processing_stages
- **Stage Tracking**: Monitor each processing stage completion
- **Performance Metrics**: Track processing time and success rates

#### Tool Requirements

1. **Moderation Tools**

   - `check_brand_compliance_tool`: Validate brand guidelines
   - `fact_check_tool`: Verify content accuracy
   - `sentiment_analysis_tool`: Analyze content tone

2. **SEO Tools**

   - `keyword_research_tool`: Identify optimal keywords
   - `readability_analyzer_tool`: Assess content readability
   - `competitor_analysis_tool`: Compare against competitor content

3. **Publishing Tools**
   - `format_for_platform_tool`: Adapt content for specific platforms
   - `schedule_publication_tool`: Schedule content release
   - `monitor_performance_tool`: Track publication success

#### Success Criteria

- Content approval rate above 90%
- SEO score improvement of 20%+ post-optimization
- Zero brand compliance violations in published content
- Platform-specific formatting accuracy of 98%+

---

## Question 3: Financial Analysis Multi-Agent Research System (Intermediate)

### Business Scenario

Create a comprehensive financial research platform that analyzes investment opportunities through coordinated specialist agents providing market data, technical analysis, risk assessment, and compliance-approved recommendations.

### Core Requirements

#### System Capabilities

- **Market Data Collection**: Gather real-time and historical financial data
- **Technical Analysis**: Perform chart pattern and indicator analysis
- **Risk Assessment**: Evaluate investment risks and volatility
- **Investment Recommendations**: Generate actionable investment advice
- **Compliance Review**: Ensure regulatory compliance for recommendations

#### Agent Architecture Requirements

- **Financial Research Coordinator**: Orchestrates the complete research workflow
- **Market Data Collector**: Gathers comprehensive market information
- **Technical Analysis Specialist**: Performs chart and indicator analysis
- **Risk Assessment Specialist**: Evaluates investment risks
- **Investment Advisor**: Generates investment recommendations
- **Compliance Officer**: Reviews recommendations for regulatory compliance

#### Data Models Needed

1. **MarketData Model**

   - Fields: symbol, price, volume, market_cap, pe_ratio, timestamp
   - Use Decimal for financial precision
   - Include data freshness validation

2. **TechnicalAnalysis Model**

   - Fields: symbol, trend, support_levels, resistance_levels, indicators, confidence_score
   - Trend types: bullish, bearish, neutral
   - Confidence score: 0.0 to 1.0 range

3. **RiskAssessment Model**

   - Fields: symbol, risk_level, volatility_score, beta, risk_factors, var_95
   - Risk levels: low, medium, high, very_high
   - Include Value at Risk (VaR) calculations

4. **InvestmentRecommendation Model**
   - Fields: symbol, recommendation, target_price, time_horizon, reasoning, confidence_level, compliance_approved
   - Recommendations: strong_buy, buy, hold, sell, strong_sell
   - Time horizons: short_term, medium_term, long_term

#### Sequential Handoff Strategy

- **Data Collection Phase**: Market data gathering and validation
- **Parallel Analysis Phase**: Technical and risk analysis simultaneously
- **Synthesis Phase**: Combine analyses for investment recommendation
- **Compliance Phase**: Final regulatory review and approval
- **Typed Handoffs**: Use structured data models between agents

#### Guardrail Requirements

1. **Market Data Validation Guardrail**

   - Price validation (must be positive)
   - Volume validation (cannot be negative)
   - P/E ratio sanity checks (0-1000 range)
   - Data freshness verification (under 1 hour old)
   - Trigger on any validation failure

2. **Recommendation Sanity Check Guardrail**
   - Confidence level minimums (30%+ for any recommendation)
   - Strong recommendation confidence requirements (70%+ confidence)
   - Target price validation (must be positive)
   - Allow minor issues but trigger on multiple violations

#### Advanced Tracing Strategy

- **Research Session Tracking**: Group related analyses under research_id
- **Parallel Analysis Monitoring**: Track simultaneous technical and risk analysis
- **Custom Spans**: Create spans for data collection, analysis phases
- **Performance Metrics**: Measure analysis confidence and completion times
- **Client Privacy**: Protect sensitive client information in traces

#### Tool Requirements

1. **Data Collection Tools**

   - `fetch_market_data_tool`: Retrieve current market data
   - `validate_data_quality_tool`: Verify data integrity
   - `get_company_fundamentals_tool`: Collect fundamental analysis data

2. **Analysis Tools**

   - `calculate_technical_indicators_tool`: Compute technical indicators
   - `identify_chart_patterns_tool`: Recognize chart patterns
   - `trend_analysis_tool`: Determine market trends
   - `calculate_volatility_tool`: Measure price volatility
   - `assess_credit_risk_tool`: Evaluate credit risks

3. **Recommendation Tools**
   - `generate_recommendation_tool`: Create investment recommendations
   - `assess_client_suitability_tool`: Match recommendations to client profiles
   - `compliance_check_tool`: Verify regulatory compliance

#### Success Criteria

- Data accuracy rate above 99.5%
- Analysis completion time under 5 minutes
- Recommendation confidence scores above 70% for actionable advice
- 100% compliance approval for published recommendations

---

## Question 4: Enterprise Document Processing and Knowledge Management System (Advanced)

### Business Scenario

Architect an enterprise-grade document processing system for a large consulting firm that handles sensitive client documents with comprehensive security, compliance, and knowledge management capabilities.

### Core Requirements

#### System Capabilities

- **Multi-Format Processing**: Handle PDFs, Word docs, emails, contracts, presentations
- **Advanced Content Extraction**: NLP-powered text and data extraction
- **Security Classification**: Multi-level security and access controls
- **Knowledge Management**: Intelligent categorization and searchable knowledge base
- **Compliance Monitoring**: Comprehensive audit trails and regulatory compliance
- **Real-Time Collaboration**: Version control and collaborative features

#### Agent Architecture Requirements

- **Enterprise Document Processing Orchestrator**: Manages complete document lifecycle
- **Document Ingestion Security Controller**: Handles secure document intake
- **Advanced Content Extraction Specialist**: Performs NLP and OCR processing
- **Intelligent Document Classifier**: Categorizes and tags documents
- **Enterprise Knowledge Manager**: Creates knowledge base entries
- **Compliance and Audit Specialist**: Ensures regulatory compliance

#### Complex Data Models Needed

1. **DocumentMetadata Model**

   - Security classification enum: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
   - Fields: document_id, title, document_type, security_classification, client_id, project_id, created_by, created_at, file_size, checksum, access_permissions, retention_policy
   - Include UUID generation and datetime handling

2. **ExtractedContent Model**

   - Fields: document_id, text_content, structured_data, entities, key_phrases, sentiment_score, language, confidence_score
   - Support named entity recognition results
   - Include confidence scoring for quality assessment

3. **DocumentClassification Model**

   - Fields: document_id, primary_category, secondary_categories, topics, client_relevance, business_value, recommended_actions
   - Business value levels: low, medium, high, critical

4. **KnowledgeBaseEntry Model**

   - Fields: entry_id, document_id, title, summary, key_insights, tags, related_documents, search_keywords, access_level, indexed_at
   - Support relationship mapping between documents

5. **ComplianceReport Model**
   - Fields: document_id, compliance_status, violations_found, recommendations, reviewer_id, review_timestamp, next_review_date
   - Compliance statuses: compliant, requires_review, non_compliant

#### Advanced Handoff Patterns

- **Conditional Routing**: Route based on document type, security level, and priority
- **Input Filters**: Strip sensitive information during handoffs
- **Callback Functions**: Execute logging and notification functions on handoff
- **Error Recovery**: Implement retry mechanisms with enhanced configurations
- **Dynamic Scaling**: Adjust processing based on document volume and priority

#### Enterprise Security Guardrails

1. **Enterprise Security Guardrail**

   - File size limits (100MB maximum)
   - Security classification validation
   - Access permission verification for restricted documents
   - Client data classification rules
   - PII detection using regex patterns (SSN, credit cards, emails)
   - Trigger on critical security violations

2. **Data Classification Guardrail**
   - Access level consistency validation
   - Search keyword limitations for restricted content
   - Sensitive content classification requirements
   - Over-sharing prevention mechanisms

#### Custom Enterprise Tracing

- **Enterprise Audit Processor**: Custom trace processor for audit logging
- **Compliance Integration**: Integration with enterprise monitoring systems
- **Audit Database Storage**: Store comprehensive audit records
- **Alert System**: Automated compliance violation alerts
- **Metadata Enrichment**: Enhanced trace metadata for enterprise requirements

#### Advanced Workflow Management

- **Priority-Based Processing**: Handle urgent, high, normal, and low priority documents
- **Quality Assurance Gates**: Multi-stage validation and retry mechanisms
- **Enterprise System Integration**: Update knowledge management, search indexes, client portals
- **Error Escalation**: Automatic escalation for high-priority document failures
- **Performance Monitoring**: Track processing duration and success rates

#### Comprehensive Tool Requirements

1. **Security Tools**

   - `security_scan_tool`: Malware and threat detection
   - `extract_metadata_tool`: Document metadata extraction
   - `apply_encryption_tool`: Content encryption for sensitive documents
   - `validate_file_integrity_tool`: File integrity verification

2. **Content Processing Tools**

   - `ocr_extraction_tool`: Optical character recognition for scanned documents
   - `nlp_analysis_tool`: Natural language processing for text analysis
   - `table_extraction_tool`: Structured data extraction from tables
   - `entity_recognition_tool`: Named entity recognition
   - `sentiment_analysis_tool`: Content sentiment analysis

3. **Classification Tools**

   - `ml_classification_tool`: Machine learning-based document classification
   - `topic_modeling_tool`: Topic identification and modeling
   - `business_rule_engine_tool`: Business rule application
   - `client_context_tool`: Client-specific classification rules

4. **Knowledge Management Tools**

   - `summarization_tool`: Document summarization
   - `insight_extraction_tool`: Key insight identification
   - `relationship_mapping_tool`: Document relationship mapping
   - `search_optimization_tool`: Search index optimization
   - `knowledge_graph_tool`: Knowledge graph creation

5. **Compliance Tools**
   - `privacy_scan_tool`: Privacy violation detection
   - `regulatory_check_tool`: Regulatory compliance verification
   - `audit_trail_tool`: Audit trail generation
   - `legal_review_flag_tool`: Legal review requirement flagging
   - `retention_policy_tool`: Data retention policy application

#### Enterprise Integration Requirements

- **Quality Validation**: Multi-criteria quality assessment with retry logic
- **Enterprise System Updates**: Knowledge management, search indexing, client portal updates
- **Error Escalation**: Automated escalation for critical processing failures
- **Performance Metrics**: Comprehensive processing duration and success tracking
- **Compliance Reporting**: Detailed compliance status and violation reporting

#### Success Criteria

- Document processing accuracy above 99%
- Security classification accuracy of 100%
- Average processing time under 10 minutes for standard documents
- Zero compliance violations in production
- Complete audit trail for all processed documents
- Enterprise system integration success rate above 98%

---

## General Implementation Guidelines

### Development Approach

1. **Start Simple**: Begin with basic agent setup and gradually add complexity
2. **Test Incrementally**: Test each component before integration
3. **Security First**: Implement security measures from the beginning
4. **Monitor Everything**: Add comprehensive logging and monitoring

### Code Organization

- **Separate Concerns**: Keep models, agents, tools, and workflows in separate modules
- **Configuration Management**: Use environment variables for sensitive data
- **Error Handling**: Implement comprehensive error handling and recovery
- **Documentation**: Document all components and integration points

### Testing Strategy

1. **Unit Testing**: Test individual agents, tools, and guardrails
2. **Integration Testing**: Test handoff patterns and data flow
3. **Performance Testing**: Measure response times and resource usage
4. **Security Testing**: Verify guardrail effectiveness and access controls
5. **End-to-End Testing**: Test complete workflows with realistic data

### Deployment Considerations

- **Environment Configuration**: Separate development, staging, and production configs
- **Scaling Strategy**: Design for horizontal scaling of agent processing
- **Monitoring Integration**: Integrate with enterprise monitoring systems
- **Backup and Recovery**: Implement data backup and disaster recovery procedures

### Best Practices

- **Prompt Engineering**: Craft clear, specific instructions for each agent
- **Data Validation**: Validate all inputs and outputs using Pydantic models
- **Tracing Strategy**: Use meaningful workflow names and comprehensive metadata
- **Security Compliance**: Follow enterprise security policies and regulations
- **Performance Optimization**: Monitor and optimize agent response times

Remember: These are comprehensive professional scenarios that require significant implementation effort. Focus on understanding the requirements thoroughly before beginning implementation, and build incrementally to ensure each component works correctly before adding complexity.
