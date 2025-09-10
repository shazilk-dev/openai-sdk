# OpenAI Agents SDK - Beginner to Intermediate Practice Guide

This comprehensive practice guide covers essential OpenAI Agents SDK concepts through progressive hands-on exercises. Each question builds upon previous concepts while introducing new SDK features following industry best practices.

## 🎯 Learning Objectives

By completing these exercises, you will master:

- General SDK concepts and defaults
- Handoffs (concept, usage, parameters, callbacks)
- Tool calls & error handling during execution
- Dynamic instructions & context objects
- Guardrails (purpose, timing, tripwires)
- Tracing (traces vs spans, multi-run traces)
- Hooks (RunHooks, AgentHooks)
- Exception handling (MaxTurnsExceeded, ModelBehaviorError, etc.)
- Runner methods (run, run_sync, run_streamed) and use cases
- ModelSettings and resolve() method
- output_type behavior and schema strictness

---

## Question 1: Personal Task Assistant Agent (Beginner)

### 🎯 Learning Focus

- Basic agent creation and configuration
- Simple tool implementation
- Understanding Runner methods
- Basic exception handling

### Business Scenario

Create a personal task assistant that helps users manage their daily tasks, set reminders, and get weather information.

### Core Requirements

#### Agent Basics

- **Agent Name**: "Personal Task Assistant"
- **Instructions**: Friendly assistant that helps with task management and daily information
- **Default Behavior**: Always greet users and offer help options

#### Tool Requirements

1. **Task Management Tools**
   - `add_task_tool`: Add new tasks to a list
   - `list_tasks_tool`: Display current tasks
   - `complete_task_tool`: Mark tasks as completed
   - `get_weather_tool`: Get current weather information

#### Runner Method Practice

- **Synchronous Execution**: Use `run_sync()` for simple interactions
- **Basic Error Handling**: Handle tool execution failures gracefully
- **Response Processing**: Extract and display agent responses properly

#### Exception Handling Requirements

- **Tool Errors**: Handle cases when tools fail (e.g., weather API unavailable)
- **Invalid Input**: Manage user inputs that don't match expected formats
- **Basic Logging**: Log errors for debugging purposes

#### Industry Best Practices to Implement

1. **Proper Tool Signatures**: Use type hints and clear parameter names
2. **Error Messages**: Provide user-friendly error messages
3. **Input Validation**: Validate user inputs before processing
4. **Resource Management**: Properly handle external API calls

#### Implementation Challenges

1. Create a simple task storage system (in-memory list)
2. Implement weather API integration with error handling
3. Handle malformed user requests gracefully
4. Create clear, conversational responses

#### Success Criteria

- Agent responds to basic task management requests
- Tools execute successfully with proper error handling
- User receives helpful feedback for both success and error cases
- Code follows Python best practices with proper type hints

---

## Question 2: Dynamic Learning Tutor Agent (Beginner-Intermediate)

### 🎯 Learning Focus

- Dynamic instructions modification
- Context objects and state management
- Basic tracing implementation
- ModelSettings configuration

### Business Scenario

Build an adaptive tutoring agent that adjusts its teaching style and difficulty based on student performance and subject matter.

### Core Requirements

#### Dynamic Instructions Strategy

- **Base Instructions**: Standard tutoring approach
- **Adaptive Behavior**: Modify instructions based on:
  - Student's current subject (math, science, history, etc.)
  - Performance level (beginner, intermediate, advanced)
  - Learning preferences (visual, auditory, kinesthetic)
  - Recent quiz scores

#### Context Management

- **Student Profile Context**: Store and update student information
- **Session Context**: Track current lesson progress
- **Performance Context**: Maintain running score of student performance
- **Subject Context**: Current subject and topic being studied

#### ModelSettings Requirements

- **Temperature Configuration**: Lower temperature (0.3) for factual subjects, higher (0.7) for creative subjects
- **Max Tokens**: Adjust based on complexity of explanation needed
- **Model Selection**: Use different models based on task complexity
- **Response Format**: Structured responses for different question types

#### Tool Requirements

1. **Learning Tools**

   - `ask_question_tool`: Generate questions based on difficulty level
   - `check_answer_tool`: Evaluate student responses
   - `provide_hint_tool`: Give contextual hints
   - `update_progress_tool`: Update student progress tracking

2. **Adaptation Tools**
   - `adjust_difficulty_tool`: Change question difficulty based on performance
   - `change_subject_tool`: Switch between subjects
   - `set_learning_style_tool`: Adapt to different learning preferences

#### Tracing Implementation

- **Workflow Name**: "Adaptive Tutoring Session"
- **Trace Metadata**: Student ID, subject, session duration, performance metrics
- **Span Tracking**: Track individual question-answer cycles
- **Performance Monitoring**: Monitor response times and accuracy

#### Industry Best Practices

1. **State Management**: Proper handling of student state across sessions
2. **Configuration Management**: Use environment variables for API keys and settings
3. **Data Privacy**: Ensure student data is handled securely
4. **Performance Optimization**: Efficient context updates and model calls

#### Implementation Challenges

1. Design a system to modify instructions dynamically based on context
2. Implement proper state management for student profiles
3. Create adaptive difficulty algorithms
4. Set up comprehensive tracing for learning analytics

#### Success Criteria

- Agent adapts teaching style based on student performance
- Dynamic instructions change appropriately for different subjects
- Proper tracing captures learning session analytics
- ModelSettings adjust automatically based on context

---

## Question 3: Customer Support Handoff System (Intermediate)

### 🎯 Learning Focus

- Handoff concepts and implementation
- Handoff parameters and callbacks
- Advanced tool error handling
- Agent specialization patterns

### Business Scenario

Create a customer support system where a triage agent routes customers to specialized support agents based on their needs.

### Core Requirements

#### Agent Architecture

1. **Triage Agent**: Initial customer contact and routing
2. **Technical Support Agent**: Handles technical issues
3. **Billing Support Agent**: Manages billing and payment issues
4. **Escalation Agent**: Handles complex cases requiring manager attention

#### Handoff Strategy

- **Basic Handoffs**: Simple routing between agents
- **Parametric Handoffs**: Pass structured data between agents
- **Conditional Handoffs**: Route based on issue complexity or customer tier
- **Callback Handoffs**: Execute functions when handoffs occur

#### Handoff Parameters

1. **Customer Information Handoff**
   - Customer ID, tier level, issue history
   - Structured data passing using Pydantic models
2. **Issue Context Handoff**
   - Issue type, priority level, urgency
   - Previous interaction history
3. **Escalation Data Handoff**
   - Escalation reason, attempted solutions
   - Customer satisfaction score

#### Callback Implementation

- **on_handoff Callbacks**: Log handoff events, notify supervisors
- **Pre-handoff Validation**: Verify customer information before transfer
- **Post-handoff Actions**: Update customer records, send notifications

#### Advanced Tool Error Handling

1. **Retry Mechanisms**: Implement exponential backoff for failed API calls
2. **Fallback Strategies**: Alternative tools when primary tools fail
3. **Error Context**: Provide detailed error information to agents
4. **Recovery Procedures**: Automatic recovery from common failure scenarios

#### Tool Requirements

1. **Customer Management Tools**

   - `lookup_customer_tool`: Retrieve customer information
   - `update_customer_record_tool`: Update customer data
   - `check_service_status_tool`: Verify service availability

2. **Issue Tracking Tools**

   - `create_ticket_tool`: Create support tickets
   - `update_ticket_tool`: Update ticket status
   - `escalate_ticket_tool`: Escalate to higher support tier

3. **Communication Tools**
   - `send_email_tool`: Send follow-up emails
   - `schedule_callback_tool`: Schedule customer callbacks
   - `notify_supervisor_tool`: Alert supervisors of escalations

#### Industry Best Practices

1. **Error Handling Hierarchy**: Graceful degradation from specific to general error handling
2. **Handoff Documentation**: Clear documentation of handoff purposes and data flow
3. **Monitoring Integration**: Track handoff success rates and customer satisfaction
4. **Security Considerations**: Secure data transmission between agents

#### Implementation Challenges

1. Design effective handoff routing logic
2. Implement robust error handling with multiple fallback strategies
3. Create callback functions that maintain system state consistency
4. Build comprehensive logging for handoff tracking

#### Success Criteria

- Smooth handoffs between specialized agents
- Proper error handling maintains customer experience
- Callback functions execute reliably
- System maintains data consistency across handoffs

---

## Question 4: Content Moderation Pipeline with Guardrails (Intermediate)

### 🎯 Learning Focus

- Guardrails purpose and implementation
- Tripwire mechanisms
- Input/output guardrail timing
- Pipeline safety controls

### Business Scenario

Build a content moderation system for a social media platform that ensures content meets community guidelines before publication.

### Core Requirements

#### Guardrail Strategy

1. **Input Guardrails**: Check content before processing
2. **Output Guardrails**: Validate agent responses before publication
3. **Tripwire Implementation**: Immediate stops for policy violations
4. **Multi-layer Validation**: Progressive filtering with different strictness levels

#### Content Safety Requirements

- **Language Detection**: Identify inappropriate language and hate speech
- **Image Analysis**: Scan for inappropriate visual content
- **Spam Detection**: Identify promotional or repetitive content
- **Privacy Protection**: Detect and flag personal information

#### Guardrail Implementation

1. **Content Length Guardrail**

   - Purpose: Ensure content meets platform length requirements
   - Timing: Input validation before processing
   - Tripwire: Trigger on content too short/long

2. **Toxicity Detection Guardrail**

   - Purpose: Prevent harmful content from reaching users
   - Timing: Both input and output validation
   - Tripwire: Immediate rejection of toxic content

3. **PII Protection Guardrail**

   - Purpose: Protect user privacy by detecting personal information
   - Timing: Input validation and output scrubbing
   - Tripwire: Block content containing personal data

4. **Brand Safety Guardrail**
   - Purpose: Ensure content aligns with platform values
   - Timing: Output validation before publication
   - Tripwire: Flag content for manual review

#### Tool Requirements

1. **Content Analysis Tools**

   - `analyze_text_toxicity_tool`: Detect harmful language
   - `detect_pii_tool`: Identify personal information
   - `check_spam_score_tool`: Calculate spam probability
   - `analyze_image_content_tool`: Visual content analysis

2. **Moderation Tools**
   - `approve_content_tool`: Approve content for publication
   - `reject_content_tool`: Reject inappropriate content
   - `flag_for_review_tool`: Queue content for human review
   - `apply_content_warnings_tool`: Add warning labels

#### Exception Handling

- **GuardrailTripwireTriggered**: Handle guardrail violations gracefully
- **ModerationServiceError**: Manage external service failures
- **ContentProcessingError**: Handle content analysis failures

#### Industry Best Practices

1. **Progressive Filtering**: Multiple validation layers with increasing strictness
2. **Audit Trail**: Complete logging of moderation decisions
3. **Appeal Process**: Mechanism for users to contest moderation decisions
4. **Performance Monitoring**: Track false positive/negative rates

#### Implementation Challenges

1. Design effective guardrail trigger thresholds
2. Implement proper error handling for moderation service failures
3. Create audit trails for compliance and appeals
4. Balance security with user experience

#### Success Criteria

- Guardrails effectively prevent policy violations
- Tripwires trigger appropriately without false positives
- System maintains high throughput while ensuring safety
- Complete audit trail for all moderation decisions

---

## Question 5: Research Assistant with Advanced Tracing (Intermediate)

### 🎯 Learning Focus

- Advanced tracing concepts
- Traces vs spans understanding
- Multi-run trace coordination
- Custom span creation and management

### Business Scenario

Create a research assistant that conducts comprehensive research across multiple sources and provides detailed analysis with complete audit trails.

### Core Requirements

#### Tracing Architecture

1. **Multi-Run Traces**: Coordinate multiple research queries under single investigation
2. **Custom Spans**: Create detailed spans for different research phases
3. **Nested Span Hierarchy**: Organize spans to reflect research workflow
4. **Trace Metadata**: Rich metadata for research session analysis

#### Research Workflow

1. **Query Planning Phase**: Analyze research request and plan approach
2. **Source Collection Phase**: Gather information from multiple sources
3. **Analysis Phase**: Process and synthesize collected information
4. **Synthesis Phase**: Create comprehensive research report

#### Span Strategy

- **Investigation Span**: Top-level span for entire research session
- **Query Spans**: Individual search operations
- **Analysis Spans**: Data processing and evaluation steps
- **Synthesis Spans**: Report generation and formatting

#### Custom Tracing Implementation

1. **Trace Groups**: Link related research activities under investigation ID
2. **Span Events**: Track significant milestones in research process
3. **Custom Metadata**: Source credibility scores, processing times, confidence levels
4. **Error Tracking**: Detailed error information within trace context

#### Tool Requirements

1. **Research Tools**

   - `web_search_tool`: Search across multiple web sources
   - `academic_search_tool`: Search academic databases
   - `fact_check_tool`: Verify information accuracy
   - `source_credibility_tool`: Evaluate source reliability

2. **Analysis Tools**
   - `summarize_content_tool`: Create content summaries
   - `extract_key_points_tool`: Identify important information
   - `compare_sources_tool`: Compare information across sources
   - `generate_report_tool`: Create final research report

#### Trace Metadata Requirements

- **Research Session**: session_id, start_time, research_topic, complexity_level
- **Source Information**: source_count, source_types, credibility_scores
- **Performance Metrics**: total_duration, query_count, success_rate
- **Quality Metrics**: confidence_score, source_diversity, fact_check_results

#### Industry Best Practices

1. **Trace Organization**: Logical hierarchy reflecting research methodology
2. **Performance Monitoring**: Track research efficiency and accuracy
3. **Quality Assurance**: Trace-based validation of research quality
4. **Reproducibility**: Sufficient trace data to reproduce research results

#### Implementation Challenges

1. Design effective span hierarchy for complex research workflows
2. Implement custom span creation with meaningful metadata
3. Coordinate multi-run traces for comprehensive investigations
4. Create trace-based quality metrics and reporting

#### Success Criteria

- Clear trace hierarchy reflects research methodology
- Custom spans provide detailed insight into research process
- Multi-run traces successfully coordinate related activities
- Trace metadata enables comprehensive research analytics

---

## Question 6: Streaming Chat Agent with Hooks (Intermediate)

### 🎯 Learning Focus

- Runner methods comparison (run, run_sync, run_streamed)
- RunHooks and AgentHooks implementation
- Real-time response handling
- Hook-based monitoring and controls

### Business Scenario

Build a real-time customer service chat agent that provides streaming responses with comprehensive monitoring and control mechanisms.

### Core Requirements

#### Runner Method Strategy

1. **run_streamed()**: Primary method for real-time chat responses
2. **run_sync()**: Fallback for non-streaming operations
3. **run()**: Async operations for background processing

#### Hook Implementation

1. **RunHooks**: Monitor entire conversation flows
2. **AgentHooks**: Track individual agent interactions
3. **Performance Hooks**: Monitor response times and resource usage
4. **Quality Hooks**: Assess response quality in real-time

#### Streaming Requirements

- **Real-time Response**: Stream responses as they're generated
- **Partial Response Handling**: Process and display partial responses
- **Stream Interruption**: Handle user interruptions gracefully
- **Progressive Enhancement**: Improve responses as more context becomes available

#### Hook Strategy

1. **Pre-execution Hooks**

   - Validate user input and context
   - Log conversation starts
   - Initialize performance monitoring

2. **During-execution Hooks**

   - Monitor streaming progress
   - Track token usage and costs
   - Detect response quality issues

3. **Post-execution Hooks**
   - Log conversation completion
   - Calculate performance metrics
   - Update user satisfaction scores

#### Tool Requirements

1. **Chat Management Tools**

   - `process_user_message_tool`: Handle incoming messages
   - `format_response_tool`: Format streaming responses
   - `manage_context_tool`: Update conversation context
   - `escalate_to_human_tool`: Transfer to human agent

2. **Monitoring Tools**
   - `track_performance_tool`: Monitor response metrics
   - `assess_quality_tool`: Evaluate response quality
   - `log_interaction_tool`: Record conversation events
   - `update_metrics_tool`: Update performance dashboards

#### Exception Handling

- **StreamingError**: Handle streaming connection issues
- **TokenLimitExceeded**: Manage token limit constraints
- **QualityThresholdError**: Handle low-quality response detection
- **UserDisconnectionError**: Manage unexpected disconnections

#### Industry Best Practices

1. **Graceful Degradation**: Fallback mechanisms when streaming fails
2. **Performance Optimization**: Efficient streaming and hook execution
3. **User Experience**: Smooth streaming with appropriate visual feedback
4. **Monitoring Integration**: Comprehensive performance and quality tracking

#### Implementation Challenges

1. Design effective hook points for comprehensive monitoring
2. Implement smooth streaming with proper error handling
3. Create performance metrics that guide system improvements
4. Balance real-time response with quality assurance

#### Success Criteria

- Smooth streaming responses enhance user experience
- Hooks provide comprehensive monitoring without performance impact
- Proper fallback mechanisms ensure service reliability
- Performance metrics guide system optimization

---

## Question 7: Schema-Strict Output Agent (Intermediate)

### 🎯 Learning Focus

- output_type behavior and implementation
- Schema strictness and validation
- Structured output patterns
- Type safety and error handling

### Business Scenario

Create a financial report generator that produces strictly structured financial analysis reports with guaranteed schema compliance.

### Core Requirements

#### Output Type Strategy

1. **Strict Schema Enforcement**: Use Pydantic models for guaranteed structure
2. **Validation Hierarchies**: Nested models for complex data structures
3. **Type Safety**: Comprehensive type hints and validation
4. **Error Recovery**: Handle schema validation failures gracefully

#### Schema Design Requirements

1. **Financial Report Schema**

   - Required fields with strict typing
   - Optional fields with defaults
   - Nested objects for complex data
   - Validation rules for financial data

2. **Analysis Schema**
   - Numerical constraints and ranges
   - Enumerated values for categories
   - Date/time validation
   - Cross-field validation rules

#### Pydantic Model Structure

1. **Base Models**: Common fields and validation patterns
2. **Composite Models**: Complex nested structures
3. **Union Types**: Flexible yet type-safe alternatives
4. **Custom Validators**: Business logic validation

#### Output Type Implementation

- **Model Definition**: Comprehensive Pydantic models with validation
- **Agent Configuration**: Proper output_type parameter usage
- **Response Processing**: Handle validated structured responses
- **Error Handling**: Manage schema validation failures

#### Tool Requirements

1. **Data Collection Tools**

   - `fetch_financial_data_tool`: Retrieve financial metrics
   - `calculate_ratios_tool`: Compute financial ratios
   - `get_market_data_tool`: Collect market information
   - `validate_data_tool`: Verify data accuracy

2. **Report Generation Tools**
   - `create_summary_tool`: Generate executive summaries
   - `format_tables_tool`: Create structured data tables
   - `generate_charts_tool`: Create visual representations
   - `compile_report_tool`: Assemble final report structure

#### Validation Requirements

1. **Data Validation**: Ensure financial data meets business rules
2. **Format Validation**: Verify proper formatting and structure
3. **Completeness Validation**: Check for required information
4. **Consistency Validation**: Ensure data consistency across sections

#### Industry Best Practices

1. **Schema Versioning**: Manage schema evolution over time
2. **Validation Documentation**: Clear error messages and validation rules
3. **Type Safety**: Comprehensive type checking and validation
4. **Error Recovery**: Graceful handling of validation failures

#### Implementation Challenges

1. Design comprehensive Pydantic models with proper validation
2. Implement robust error handling for schema violations
3. Create clear validation error messages for debugging
4. Balance schema strictness with usability

#### Success Criteria

- All outputs conform to defined schemas without exceptions
- Schema validation provides clear, actionable error messages
- Type safety prevents runtime errors in data processing
- Structured outputs integrate seamlessly with downstream systems

---

## Question 8: Multi-Agent Exception Handling System (Advanced Intermediate)

### 🎯 Learning Focus

- Advanced exception handling patterns
- MaxTurnsExceeded and ModelBehaviorError handling
- Custom exception types and recovery
- Resilient multi-agent coordination

### Business Scenario

Build a complex document processing system that coordinates multiple agents with comprehensive error handling and recovery mechanisms.

### Core Requirements

#### Exception Handling Strategy

1. **Built-in Exception Management**: Handle SDK-specific exceptions
2. **Custom Exception Types**: Create domain-specific error handling
3. **Recovery Mechanisms**: Automatic recovery from common failures
4. **Escalation Procedures**: Progressive error handling approaches

#### Exception Types to Handle

1. **MaxTurnsExceeded**: Agent conversation limits
2. **ModelBehaviorError**: Unexpected model responses
3. **ToolExecutionError**: Tool failure handling
4. **HandoffError**: Inter-agent communication failures
5. **GuardrailTripwireTriggered**: Safety mechanism violations

#### Agent Coordination Requirements

1. **Document Ingestion Agent**: Handle file processing errors
2. **Content Analysis Agent**: Manage analysis failures
3. **Quality Control Agent**: Handle validation errors
4. **Output Generation Agent**: Manage formatting failures

#### Recovery Patterns

1. **Retry with Backoff**: Exponential backoff for transient failures
2. **Graceful Degradation**: Reduced functionality when components fail
3. **Circuit Breaker**: Prevent cascade failures across agents
4. **Fallback Strategies**: Alternative approaches when primary methods fail

#### Tool Requirements

1. **Processing Tools**

   - `extract_text_tool`: Text extraction with error handling
   - `analyze_content_tool`: Content analysis with fallbacks
   - `validate_quality_tool`: Quality validation with recovery
   - `generate_output_tool`: Output generation with alternatives

2. **Recovery Tools**
   - `retry_operation_tool`: Implement retry logic
   - `log_error_tool`: Comprehensive error logging
   - `escalate_error_tool`: Error escalation procedures
   - `recover_state_tool`: State recovery mechanisms

#### Error Context Management

- **Error Tracking**: Maintain detailed error context across agents
- **State Recovery**: Restore system state after failures
- **Audit Trail**: Complete error and recovery logging
- **Performance Impact**: Monitor error handling overhead

#### Industry Best Practices

1. **Error Classification**: Categorize errors for appropriate handling
2. **Recovery Testing**: Comprehensive failure scenario testing
3. **Monitoring Integration**: Error metrics and alerting
4. **Documentation**: Clear error handling documentation

#### Implementation Challenges

1. Design comprehensive exception hierarchy and handling
2. Implement effective recovery mechanisms for different failure types
3. Create error context that survives agent handoffs
4. Balance error handling complexity with system performance

#### Success Criteria

- System handles all exception types gracefully without data loss
- Recovery mechanisms restore normal operation automatically
- Error context provides sufficient debugging information
- Exception handling doesn't significantly impact system performance

---

## 📚 Implementation Guidelines

### Getting Started

1. **Environment Setup**: Configure development environment with proper dependencies
2. **Basic Structure**: Create modular code structure for agents, tools, and utilities
3. **Testing Framework**: Set up comprehensive testing for all components
4. **Documentation**: Document all components and their interactions

### Development Best Practices

1. **Code Organization**: Separate concerns into distinct modules
2. **Configuration Management**: Use environment variables and configuration files
3. **Error Handling**: Implement comprehensive error handling from the start
4. **Performance Monitoring**: Include performance tracking in all implementations

### Testing Strategy

1. **Unit Tests**: Test individual components thoroughly
2. **Integration Tests**: Test agent interactions and handoffs
3. **Error Scenario Tests**: Test all error handling paths
4. **Performance Tests**: Validate system performance under load

### Production Considerations

1. **Scalability**: Design for horizontal scaling from the beginning
2. **Monitoring**: Implement comprehensive logging and monitoring
3. **Security**: Follow security best practices for all components
4. **Maintenance**: Design for easy updates and maintenance

### Learning Path Recommendations

1. **Start Simple**: Begin with basic agents and gradually add complexity
2. **Practice Concepts**: Focus on understanding each concept thoroughly
3. **Build Incrementally**: Add features one at a time with proper testing
4. **Industry Standards**: Follow established patterns and best practices

Remember: These exercises are designed to build your expertise progressively. Take time to understand each concept thoroughly before moving to the next level of complexity.
