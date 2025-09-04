"""
🏗️ OPENAI AGENTS SDK - COMPLETE STRUCTURED OUTPUTS GUIDE & TEMPLATE

This template covers everything about Structured Outputs in the OpenAI Agents SDK:
- Pydantic model definitions and validation
- output_type parameter usage patterns
- Complex data structure modeling
- Validation and error handling
- Nested models and relationships
- Custom validators and serialization
- Performance optimization for structured outputs
- Real-world use case implementations

📚 Based on: https://openai.github.io/openai-agents-python/ref/structured-outputs/
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any, Union, Literal
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator
from agents import Agent, Runner, function_tool

# =============================================================================
# 📖 UNDERSTANDING STRUCTURED OUTPUTS IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT ARE STRUCTURED OUTPUTS?

Structured Outputs ensure AI responses follow specific data formats:
1. ✅ TYPE SAFETY: Guarantee response format and types
2. ✅ VALIDATION: Automatic input/output validation
3. ✅ CONSISTENCY: Reliable data structures across calls
4. ✅ INTEGRATION: Easy API and database integration
5. ✅ ERROR HANDLING: Clear validation error messages

KEY CONCEPTS:

1. PYDANTIC MODELS:
   - Define data structure with types
   - Built-in validation and serialization
   - Custom validators for business logic
   - Nested models for complex data

2. OUTPUT_TYPE PARAMETER:
   - Specify expected response format
   - Works with Agent instructions and tools
   - Enforces structure at runtime
   - Provides type hints for development

3. VALIDATION PATTERNS:
   - Field validation (type, range, format)
   - Model validation (cross-field logic)
   - Custom validators for domain rules
   - Error handling and user feedback

4. USE CASES:
   - API responses and data exchange
   - Database record creation/updates
   - Report generation with consistent format
   - Form processing and data entry
   - Configuration and settings management
"""

# =============================================================================
# 📊 1. BASIC STRUCTURED OUTPUT MODELS
# =============================================================================

# Basic data models
class PersonalInfo(BaseModel):
    """Basic personal information model"""
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    phone: Optional[str] = Field(None, regex=r'^\+?1?-?\d{3}-?\d{3}-?\d{4}$')
    
    @validator('email')
    def validate_email_domain(cls, v):
        """Custom email domain validation"""
        if '@' in v:
            domain = v.split('@')[1]
            blocked_domains = ['spam.com', 'fake.org']
            if domain in blocked_domains:
                raise ValueError(f'Email domain {domain} is not allowed')
        return v

class Address(BaseModel):
    """Address information model"""
    street: str = Field(..., min_length=1, max_length=100)
    city: str = Field(..., min_length=1, max_length=50)
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str = Field(..., regex=r'^\d{5}(-\d{4})?$')
    country: str = Field(default="US", max_length=2)

class ContactInfo(BaseModel):
    """Combined contact information"""
    personal: PersonalInfo
    address: Address
    emergency_contact: Optional[str] = Field(None, max_length=100)
    
    @root_validator
    def validate_emergency_contact(cls, values):
        """Ensure emergency contact is provided for minors"""
        personal = values.get('personal')
        emergency_contact = values.get('emergency_contact')
        
        if personal and personal.age < 18 and not emergency_contact:
            raise ValueError('Emergency contact required for minors')
        
        return values

def basic_structured_outputs_examples():
    """Basic structured output examples"""
    
    print("📊 BASIC STRUCTURED OUTPUT MODELS")
    print("=" * 50)
    
    # Agent that collects contact information
    contact_collector = Agent(
        name="ContactCollector",
        instructions="""
        You collect and organize contact information from users.
        Extract all relevant details and structure them properly.
        Ask for missing required information.
        """,
        output_type=ContactInfo
    )
    
    print("\n1. Testing Basic Structured Outputs:")
    
    # Test data collection scenarios
    contact_scenarios = [
        "My name is Alice Johnson, I'm 28 years old. My email is alice@example.com and I live at 123 Main St, Springfield, IL 62701",
        "I'm Bob Smith, 17 years old, email bob@test.com, living at 456 Oak Ave, Chicago, IL 60601. My emergency contact is Sarah Smith.",
        "Jane Doe, age 35, jane.doe@company.com, 789 Pine St, Boston, MA 02101"
    ]
    
    for i, scenario in enumerate(contact_scenarios, 1):
        try:
            result = Runner.run_sync(contact_collector, scenario)
            print(f"{i}. Input: {scenario}")
            
            if hasattr(result, 'structured_output'):
                output = result.structured_output
                print(f"   Structured Output: {output}")
                print(f"   Name: {output.personal.first_name} {output.personal.last_name}")
                print(f"   Age: {output.personal.age}")
                print(f"   Address: {output.address.street}, {output.address.city}, {output.address.state}")
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return contact_collector

# =============================================================================
# 💼 2. BUSINESS DATA MODELS
# =============================================================================

class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(str, Enum):
    """Task status options"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    CANCELLED = "cancelled"

class Task(BaseModel):
    """Task management model"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Priority = Field(default=Priority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.TODO)
    assigned_to: Optional[str] = Field(None, max_length=100)
    due_date: Optional[date] = None
    estimated_hours: Optional[float] = Field(None, ge=0, le=1000)
    tags: List[str] = Field(default_factory=list)
    
    @validator('due_date')
    def validate_due_date(cls, v):
        """Ensure due date is not in the past"""
        if v and v < date.today():
            raise ValueError('Due date cannot be in the past')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and clean tags"""
        cleaned_tags = []
        for tag in v:
            cleaned_tag = tag.strip().lower()
            if cleaned_tag and len(cleaned_tag) <= 50:
                cleaned_tags.append(cleaned_tag)
        return cleaned_tags

class Project(BaseModel):
    """Project management model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    tasks: List[Task] = Field(default_factory=list)
    start_date: date
    end_date: Optional[date] = None
    budget: Optional[Decimal] = Field(None, ge=0)
    team_members: List[str] = Field(default_factory=list)
    
    @root_validator
    def validate_dates(cls, values):
        """Validate project date logic"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date and end_date < start_date:
            raise ValueError('End date must be after start date')
        
        return values
    
    def get_progress(self) -> Dict[str, Any]:
        """Calculate project progress"""
        if not self.tasks:
            return {'completion_rate': 0.0, 'task_counts': {}}
        
        task_counts = {}
        for status in TaskStatus:
            task_counts[status.value] = sum(1 for task in self.tasks if task.status == status)
        
        completed = task_counts.get('done', 0)
        completion_rate = completed / len(self.tasks) if self.tasks else 0.0
        
        return {
            'completion_rate': completion_rate,
            'task_counts': task_counts,
            'total_tasks': len(self.tasks)
        }

class ProjectReport(BaseModel):
    """Project status report"""
    project: Project
    report_date: datetime = Field(default_factory=datetime.now)
    summary: str = Field(..., min_length=1, max_length=1000)
    progress: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    next_milestones: List[str] = Field(default_factory=list)
    
    @root_validator
    def calculate_progress(cls, values):
        """Auto-calculate progress from project data"""
        project = values.get('project')
        if project:
            values['progress'] = project.get_progress()
        return values

def business_data_models_examples():
    """Examples of business data model usage"""
    
    print("\n💼 BUSINESS DATA MODELS")
    print("=" * 50)
    
    # Project management agent
    project_manager = Agent(
        name="ProjectManager",
        instructions="""
        You are a project management assistant that helps organize tasks and projects.
        Extract project information, create tasks, and generate structured reports.
        Always ensure data is properly validated and organized.
        """,
        output_type=Project
    )
    
    # Report generation agent
    report_generator = Agent(
        name="ReportGenerator",
        instructions="""
        You generate detailed project reports with analysis and insights.
        Include progress metrics, issue identification, and future milestones.
        Be comprehensive but concise in your analysis.
        """,
        output_type=ProjectReport
    )
    
    print("\n1. Testing Project Management Models:")
    
    # Test project creation
    project_scenarios = [
        """Create a project called "Website Redesign" that starts today and includes these tasks:
        - Design mockups (high priority, assigned to Sarah, 20 hours)
        - Develop frontend (medium priority, assigned to Mike, 40 hours)  
        - Backend integration (high priority, assigned to Alex, 30 hours)
        - Testing and QA (medium priority, assigned to Lisa, 15 hours)
        The project has a budget of $50000 and should be completed in 3 months.""",
        
        """Set up a project "Mobile App Development" starting next week with these tasks:
        - User research (low priority, 10 hours)
        - UI/UX design (high priority, 25 hours)
        - iOS development (critical priority, 60 hours)
        - Android development (critical priority, 60 hours)
        - App store submission (medium priority, 5 hours)
        Budget is $75000, team includes John, Emma, David, and Maria."""
    ]
    
    created_projects = []
    
    for i, scenario in enumerate(project_scenarios, 1):
        try:
            result = Runner.run_sync(project_manager, scenario)
            print(f"{i}. Project Creation Scenario:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                project = result.structured_output
                print(f"   Project: {project.name}")
                print(f"   Tasks: {len(project.tasks)}")
                print(f"   Team: {project.team_members}")
                print(f"   Budget: ${project.budget}")
                
                # Calculate and show progress
                progress = project.get_progress()
                print(f"   Progress: {progress['completion_rate']:.1%}")
                print(f"   Task Status: {progress['task_counts']}")
                
                created_projects.append(project)
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    print("\n2. Testing Report Generation:")
    
    # Generate reports for created projects
    for i, project in enumerate(created_projects, 1):
        try:
            # Create a context for report generation
            report_request = f"""Generate a status report for the {project.name} project. 
            Analyze the current progress, identify any potential issues, and suggest next milestones."""
            
            # Use project as context
            result = Runner.run_sync(
                report_generator, 
                report_request,
                context={'project_data': project.dict()}
            )
            
            print(f"{i}. Report for {project.name}:")
            if hasattr(result, 'structured_output'):
                report = result.structured_output
                print(f"   Summary: {report.summary}")
                print(f"   Progress: {report.progress}")
                print(f"   Issues: {report.issues}")
                print(f"   Next Milestones: {report.next_milestones}")
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Report Error: {e}")
    
    return {
        "project_manager": project_manager,
        "report_generator": report_generator,
        "projects": created_projects
    }

# =============================================================================
# 🛒 3. E-COMMERCE DATA MODELS
# =============================================================================

class ProductCategory(str, Enum):
    """Product categories"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME_GARDEN = "home_garden"
    SPORTS = "sports"
    HEALTH_BEAUTY = "health_beauty"

class Product(BaseModel):
    """E-commerce product model"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    category: ProductCategory
    price: Decimal = Field(..., ge=0, max_digits=10, decimal_places=2)
    sku: str = Field(..., min_length=1, max_length=50)
    in_stock: bool = Field(default=True)
    stock_quantity: int = Field(default=0, ge=0)
    images: List[str] = Field(default_factory=list)
    specifications: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @validator('sku')
    def validate_sku(cls, v):
        """Validate SKU format"""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('SKU must contain only alphanumeric characters, hyphens, and underscores')
        return v.upper()
    
    @validator('images')
    def validate_images(cls, v):
        """Validate image URLs"""
        valid_images = []
        for url in v:
            if url.startswith(('http://', 'https://')) or url.startswith('/'):
                valid_images.append(url)
        return valid_images

class OrderItem(BaseModel):
    """Order item model"""
    product_sku: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=1)
    unit_price: Decimal = Field(..., ge=0, max_digits=10, decimal_places=2)
    total_price: Optional[Decimal] = None
    
    @root_validator
    def calculate_total(cls, values):
        """Auto-calculate total price"""
        quantity = values.get('quantity', 0)
        unit_price = values.get('unit_price', 0)
        values['total_price'] = Decimal(quantity) * unit_price
        return values

class ShippingAddress(BaseModel):
    """Shipping address model"""
    recipient_name: str = Field(..., min_length=1, max_length=100)
    street_address: str = Field(..., min_length=1, max_length=200)
    city: str = Field(..., min_length=1, max_length=50)
    state_province: str = Field(..., min_length=1, max_length=50)
    postal_code: str = Field(..., min_length=1, max_length=20)
    country: str = Field(default="US", max_length=2)
    phone: Optional[str] = Field(None, regex=r'^\+?[\d\s\-\(\)]+$')

class Order(BaseModel):
    """E-commerce order model"""
    order_id: str = Field(..., min_length=1)
    customer_email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    items: List[OrderItem] = Field(..., min_items=1)
    shipping_address: ShippingAddress
    subtotal: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    shipping_cost: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None
    order_date: datetime = Field(default_factory=datetime.now)
    status: Literal["pending", "confirmed", "shipped", "delivered", "cancelled"] = "pending"
    notes: Optional[str] = Field(None, max_length=500)
    
    @root_validator
    def calculate_totals(cls, values):
        """Calculate order totals"""
        items = values.get('items', [])
        
        # Calculate subtotal
        subtotal = sum(item.total_price for item in items if item.total_price)
        values['subtotal'] = subtotal
        
        # Calculate tax (example: 8.5%)
        tax_rate = Decimal('0.085')
        tax_amount = subtotal * tax_rate
        values['tax_amount'] = tax_amount.quantize(Decimal('0.01'))
        
        # Shipping cost (example logic)
        shipping_cost = Decimal('9.99') if subtotal < Decimal('50.00') else Decimal('0.00')
        values['shipping_cost'] = shipping_cost
        
        # Total amount
        total = subtotal + tax_amount + shipping_cost
        values['total_amount'] = total.quantize(Decimal('0.01'))
        
        return values

class ProductCatalog(BaseModel):
    """Product catalog response model"""
    products: List[Product]
    total_count: int
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    
    @root_validator
    def validate_pagination(cls, values):
        """Validate pagination data"""
        products = values.get('products', [])
        page = values.get('page', 1)
        page_size = values.get('page_size', 20)
        
        # Update total count if not provided
        if 'total_count' not in values:
            values['total_count'] = len(products)
        
        return values

def ecommerce_data_models_examples():
    """Examples of e-commerce data model usage"""
    
    print("\n🛒 E-COMMERCE DATA MODELS")
    print("=" * 50)
    
    # Product catalog agent
    catalog_agent = Agent(
        name="CatalogAgent",
        instructions="""
        You manage product catalogs and help customers find products.
        Extract product information and organize it into proper data structures.
        Ensure all product details are complete and valid.
        """,
        output_type=ProductCatalog
    )
    
    # Order processing agent
    order_agent = Agent(
        name="OrderAgent", 
        instructions="""
        You process customer orders and handle order management.
        Extract order details, validate information, and calculate totals.
        Ensure shipping addresses are complete and properly formatted.
        """,
        output_type=Order
    )
    
    print("\n1. Testing Product Catalog Management:")
    
    catalog_scenarios = [
        """Create a catalog with these electronics products:
        1. iPhone 15 Pro - $999, SKU: IPH15PRO, in stock (50 units), 128GB storage, Space Black
        2. MacBook Air M2 - $1199, SKU: MBA-M2, in stock (25 units), 13-inch, 8GB RAM, 256GB SSD
        3. AirPods Pro - $249, SKU: APP-GEN2, in stock (100 units), Active Noise Cancellation
        All are electronics category.""",
        
        """Add clothing products to catalog:
        1. Men's Cotton T-Shirt - $19.99, SKU: MCT-001, various sizes, Navy Blue
        2. Women's Jeans - $79.99, SKU: WJ-SLIM, size 28-34, Dark Wash
        3. Winter Jacket - $149.99, SKU: WJ-WINTER, sizes S-XL, Water resistant
        All in clothing category."""
    ]
    
    created_catalogs = []
    
    for i, scenario in enumerate(catalog_scenarios, 1):
        try:
            result = Runner.run_sync(catalog_agent, scenario)
            print(f"{i}. Catalog Creation:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                catalog = result.structured_output
                print(f"   Products Count: {len(catalog.products)}")
                print(f"   Total in Catalog: {catalog.total_count}")
                
                for product in catalog.products[:2]:  # Show first 2 products
                    print(f"   - {product.name}: ${product.price} (SKU: {product.sku})")
                
                created_catalogs.append(catalog)
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    print("\n2. Testing Order Processing:")
    
    order_scenarios = [
        """Process this order:
        Customer: john.doe@email.com
        Items: 2x iPhone 15 Pro at $999 each, 1x AirPods Pro at $249
        Ship to: John Doe, 123 Main St, San Francisco, CA 94105, USA, phone: 555-123-4567
        Order ID: ORD-2024-001""",
        
        """Create order:
        Customer: jane.smith@example.com  
        Items: 1x MacBook Air M2 at $1199, 1x Men's Cotton T-Shirt at $19.99
        Ship to: Jane Smith, 456 Oak Avenue, Austin, TX 78701, USA, phone: 555-987-6543
        Order ID: ORD-2024-002
        Special notes: Leave at front door if not home"""
    ]
    
    for i, scenario in enumerate(order_scenarios, 1):
        try:
            result = Runner.run_sync(order_agent, scenario)
            print(f"{i}. Order Processing:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                order = result.structured_output
                print(f"   Order ID: {order.order_id}")
                print(f"   Customer: {order.customer_email}")
                print(f"   Items: {len(order.items)}")
                print(f"   Subtotal: ${order.subtotal}")
                print(f"   Tax: ${order.tax_amount}")
                print(f"   Shipping: ${order.shipping_cost}")
                print(f"   Total: ${order.total_amount}")
                print(f"   Ship to: {order.shipping_address.recipient_name}")
                
                if order.notes:
                    print(f"   Notes: {order.notes}")
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return {
        "catalog_agent": catalog_agent,
        "order_agent": order_agent,
        "catalogs": created_catalogs
    }

# =============================================================================
# 📋 4. SURVEY & FORM DATA MODELS
# =============================================================================

class QuestionType(str, Enum):
    """Survey question types"""
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT = "text"
    NUMBER = "number"
    RATING = "rating"
    YES_NO = "yes_no"
    DATE = "date"

class SurveyQuestion(BaseModel):
    """Survey question model"""
    id: str = Field(..., min_length=1)
    question_text: str = Field(..., min_length=1, max_length=500)
    question_type: QuestionType
    required: bool = Field(default=True)
    options: Optional[List[str]] = None  # For multiple choice
    min_value: Optional[int] = None  # For number/rating
    max_value: Optional[int] = None  # For number/rating
    help_text: Optional[str] = Field(None, max_length=200)
    
    @root_validator
    def validate_question_options(cls, values):
        """Validate question type specific options"""
        question_type = values.get('question_type')
        options = values.get('options')
        
        if question_type == QuestionType.MULTIPLE_CHOICE and not options:
            raise ValueError('Multiple choice questions must have options')
        
        if question_type == QuestionType.RATING:
            min_val = values.get('min_value', 1)
            max_val = values.get('max_value', 5)
            if min_val >= max_val:
                raise ValueError('min_value must be less than max_value for rating questions')
        
        return values

class SurveyResponse(BaseModel):
    """Survey response model"""
    question_id: str = Field(..., min_length=1)
    response_value: Union[str, int, bool, float, date] = Field(...)
    response_text: Optional[str] = None  # For additional comments
    
    @validator('response_value')
    def validate_response_format(cls, v):
        """Basic response format validation"""
        if isinstance(v, str) and len(v) > 1000:
            raise ValueError('Text responses cannot exceed 1000 characters')
        return v

class Survey(BaseModel):
    """Complete survey model"""
    survey_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    questions: List[SurveyQuestion] = Field(..., min_items=1)
    created_date: datetime = Field(default_factory=datetime.now)
    expires_date: Optional[datetime] = None
    is_active: bool = Field(default=True)
    
    @validator('expires_date')
    def validate_expiry(cls, v, values):
        """Validate expiry date"""
        if v and 'created_date' in values and v <= values['created_date']:
            raise ValueError('Expiry date must be after creation date')
        return v

class SurveySubmission(BaseModel):
    """Survey submission model"""
    submission_id: str = Field(..., min_length=1)
    survey_id: str = Field(..., min_length=1)
    respondent_email: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    responses: List[SurveyResponse] = Field(..., min_items=1)
    submission_date: datetime = Field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    completion_time_seconds: Optional[int] = Field(None, ge=0)
    
    def get_response_by_question(self, question_id: str) -> Optional[SurveyResponse]:
        """Get response for specific question"""
        for response in self.responses:
            if response.question_id == question_id:
                return response
        return None
    
    def get_completion_rate(self, total_questions: int) -> float:
        """Calculate completion rate"""
        return len(self.responses) / total_questions if total_questions > 0 else 0.0

class SurveyAnalysis(BaseModel):
    """Survey analysis results"""
    survey_id: str
    analysis_date: datetime = Field(default_factory=datetime.now)
    total_submissions: int = Field(ge=0)
    average_completion_rate: float = Field(ge=0, le=1)
    question_analytics: Dict[str, Any] = Field(default_factory=dict)
    demographic_breakdown: Dict[str, Any] = Field(default_factory=dict)
    summary_insights: List[str] = Field(default_factory=list)

def survey_form_data_models_examples():
    """Examples of survey and form data model usage"""
    
    print("\n📋 SURVEY & FORM DATA MODELS")
    print("=" * 50)
    
    # Survey creation agent
    survey_creator = Agent(
        name="SurveyCreator",
        instructions="""
        You create surveys and forms with proper question structures.
        Extract survey requirements and create well-structured questionnaires.
        Ensure questions are clear, properly typed, and logically organized.
        """,
        output_type=Survey
    )
    
    # Survey response processor
    response_processor = Agent(
        name="ResponseProcessor",
        instructions="""
        You process survey responses and validate submissions.
        Extract response data and structure it properly for analysis.
        Ensure all required questions are answered and data is valid.
        """,
        output_type=SurveySubmission
    )
    
    # Survey analyzer
    survey_analyzer = Agent(
        name="SurveyAnalyzer",
        instructions="""
        You analyze survey data and provide insights.
        Calculate metrics, identify trends, and generate actionable insights.
        Provide clear summaries and recommendations based on the data.
        """,
        output_type=SurveyAnalysis
    )
    
    print("\n1. Testing Survey Creation:")
    
    survey_scenarios = [
        """Create a customer satisfaction survey with these questions:
        1. How satisfied are you with our product? (Rating 1-5)
        2. What is your age group? (Multiple choice: 18-25, 26-35, 36-45, 46-55, 55+)
        3. How did you hear about us? (Multiple choice: Social media, Friend referral, Google search, Advertisement)
        4. Any additional feedback? (Text, optional)
        5. Would you recommend us to others? (Yes/No)
        Title: Customer Satisfaction Survey 2024""",
        
        """Build an employee feedback survey:
        1. Rate your job satisfaction (1-10 scale)
        2. What department do you work in? (Text)
        3. How long have you been with the company? (Multiple choice: <1 year, 1-3 years, 3-5 years, 5+ years)
        4. What could we improve? (Text, optional)
        5. Do you feel valued at work? (Yes/No)
        6. When did you start working here? (Date)
        Title: Annual Employee Feedback Survey"""
    ]
    
    created_surveys = []
    
    for i, scenario in enumerate(survey_scenarios, 1):
        try:
            result = Runner.run_sync(survey_creator, scenario)
            print(f"{i}. Survey Creation:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                survey = result.structured_output
                print(f"   Survey: {survey.title}")
                print(f"   Questions: {len(survey.questions)}")
                print(f"   Survey ID: {survey.survey_id}")
                
                # Show first few questions
                for q in survey.questions[:3]:
                    print(f"   - {q.question_text} ({q.question_type.value})")
                
                created_surveys.append(survey)
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    print("\n2. Testing Survey Response Processing:")
    
    # Create mock responses for the first survey
    if created_surveys:
        survey = created_surveys[0]
        response_scenarios = [
            f"""Process survey response for {survey.survey_id}:
            Respondent: user@example.com
            Responses:
            - Satisfaction rating: 4
            - Age group: 26-35
            - How heard about us: Social media
            - Additional feedback: Great product, keep it up!
            - Would recommend: Yes
            Submission ID: RESP-001, completed in 120 seconds""",
            
            f"""Survey response for {survey.survey_id}:
            Respondent: customer2@test.com
            Responses:
            - Satisfaction: 5
            - Age: 36-45  
            - Source: Friend referral
            - Feedback: Excellent service and support
            - Recommend: Yes
            Submission ID: RESP-002, completed in 95 seconds"""
        ]
        
        processed_submissions = []
        
        for i, scenario in enumerate(response_scenarios, 1):
            try:
                result = Runner.run_sync(response_processor, scenario)
                print(f"{i}. Response Processing:")
                print(f"   Input: {scenario[:100]}...")
                
                if hasattr(result, 'structured_output'):
                    submission = result.structured_output
                    print(f"   Submission ID: {submission.submission_id}")
                    print(f"   Survey ID: {submission.survey_id}")
                    print(f"   Respondent: {submission.respondent_email}")
                    print(f"   Responses: {len(submission.responses)}")
                    print(f"   Completion Time: {submission.completion_time_seconds}s")
                    
                    # Show completion rate
                    completion_rate = submission.get_completion_rate(len(survey.questions))
                    print(f"   Completion Rate: {completion_rate:.1%}")
                    
                    processed_submissions.append(submission)
                else:
                    print(f"   Response: {result.final_output}")
                print()
            except Exception as e:
                print(f"{i}. Error: {e}")
        
        print("\n3. Testing Survey Analysis:")
        
        if processed_submissions:
            analysis_request = f"""Analyze survey data for {survey.survey_id}:
            - Total submissions: {len(processed_submissions)}
            - Survey has {len(survey.questions)} questions
            - Analyze satisfaction ratings, demographics, and feedback themes
            - Provide insights and recommendations"""
            
            try:
                result = Runner.run_sync(survey_analyzer, analysis_request)
                print("Survey Analysis:")
                print(f"   Input: {analysis_request[:100]}...")
                
                if hasattr(result, 'structured_output'):
                    analysis = result.structured_output
                    print(f"   Survey ID: {analysis.survey_id}")
                    print(f"   Total Submissions: {analysis.total_submissions}")
                    print(f"   Avg Completion Rate: {analysis.average_completion_rate:.1%}")
                    print(f"   Insights: {len(analysis.summary_insights)}")
                    
                    for insight in analysis.summary_insights[:3]:
                        print(f"   - {insight}")
                else:
                    print(f"   Response: {result.final_output}")
                print()
            except Exception as e:
                print(f"Analysis Error: {e}")
    
    return {
        "survey_creator": survey_creator,
        "response_processor": response_processor,
        "survey_analyzer": survey_analyzer,
        "surveys": created_surveys
    }

# =============================================================================
# 🔧 5. CUSTOM VALIDATORS & ADVANCED PATTERNS
# =============================================================================

from pydantic import validator, root_validator, Field
from typing import Any, Dict, List

class CustomValidationError(ValueError):
    """Custom validation error for business logic"""
    pass

class BankAccount(BaseModel):
    """Bank account with custom validation"""
    account_number: str = Field(..., min_length=8, max_length=12)
    routing_number: str = Field(..., min_length=9, max_length=9)
    account_type: Literal["checking", "savings"] = "checking"
    balance: Decimal = Field(..., ge=0)
    account_holder: str = Field(..., min_length=1)
    
    @validator('account_number')
    def validate_account_number(cls, v):
        """Custom account number validation"""
        if not v.isdigit():
            raise CustomValidationError('Account number must contain only digits')
        
        # Simple checksum validation (Luhn algorithm simulation)
        checksum = sum(int(digit) * (2 if i % 2 == 0 else 1) for i, digit in enumerate(v[::-1]))
        if checksum % 10 != 0:
            raise CustomValidationError('Invalid account number checksum')
        
        return v
    
    @validator('routing_number')
    def validate_routing_number(cls, v):
        """Validate routing number format"""
        if not v.isdigit() or len(v) != 9:
            raise CustomValidationError('Routing number must be exactly 9 digits')
        return v
    
    @root_validator
    def validate_business_rules(cls, values):
        """Apply business validation rules"""
        account_type = values.get('account_type')
        balance = values.get('balance', 0)
        
        # Minimum balance requirements
        min_balances = {
            'checking': Decimal('25.00'),
            'savings': Decimal('100.00')
        }
        
        min_required = min_balances.get(account_type, Decimal('0'))
        if balance < min_required:
            raise CustomValidationError(
                f'{account_type.title()} account requires minimum balance of ${min_required}'
            )
        
        return values

class Transaction(BaseModel):
    """Financial transaction with validation"""
    transaction_id: str = Field(..., min_length=1)
    from_account: str = Field(..., min_length=8)
    to_account: str = Field(..., min_length=8)
    amount: Decimal = Field(..., gt=0, max_digits=15, decimal_places=2)
    transaction_type: Literal["transfer", "deposit", "withdrawal", "payment"] = "transfer"
    description: Optional[str] = Field(None, max_length=200)
    transaction_date: datetime = Field(default_factory=datetime.now)
    
    @validator('amount')
    def validate_transaction_limits(cls, v, values):
        """Apply transaction amount limits"""
        transaction_type = values.get('transaction_type', 'transfer')
        
        limits = {
            'transfer': Decimal('10000.00'),
            'withdrawal': Decimal('5000.00'),
            'payment': Decimal('25000.00'),
            'deposit': Decimal('50000.00')
        }
        
        max_limit = limits.get(transaction_type, Decimal('1000.00'))
        if v > max_limit:
            raise CustomValidationError(
                f'{transaction_type.title()} amount cannot exceed ${max_limit}'
            )
        
        return v
    
    @root_validator
    def validate_accounts(cls, values):
        """Validate account relationships"""
        from_account = values.get('from_account')
        to_account = values.get('to_account')
        transaction_type = values.get('transaction_type')
        
        # Same account validation
        if from_account == to_account and transaction_type == 'transfer':
            raise CustomValidationError('Cannot transfer to the same account')
        
        # Account format validation (should be digits)
        for account_field, account_num in [('from_account', from_account), ('to_account', to_account)]:
            if account_num and not account_num.isdigit():
                raise CustomValidationError(f'{account_field} must contain only digits')
        
        return values

class FinancialReport(BaseModel):
    """Financial report with complex validation"""
    report_id: str = Field(..., min_length=1)
    account: BankAccount
    transactions: List[Transaction] = Field(default_factory=list)
    report_period_start: date
    report_period_end: date
    opening_balance: Decimal = Field(..., ge=0)
    closing_balance: Decimal = Field(..., ge=0)
    total_deposits: Optional[Decimal] = None
    total_withdrawals: Optional[Decimal] = None
    transaction_count: Optional[int] = None
    
    @root_validator
    def validate_and_calculate(cls, values):
        """Validate report and calculate totals"""
        start_date = values.get('report_period_start')
        end_date = values.get('report_period_end')
        transactions = values.get('transactions', [])
        opening_balance = values.get('opening_balance', Decimal('0'))
        
        # Date validation
        if start_date and end_date and end_date < start_date:
            raise CustomValidationError('End date must be after start date')
        
        # Calculate transaction totals
        total_deposits = Decimal('0')
        total_withdrawals = Decimal('0')
        
        for transaction in transactions:
            if transaction.transaction_type in ['deposit']:
                total_deposits += transaction.amount
            elif transaction.transaction_type in ['withdrawal', 'payment']:
                total_withdrawals += transaction.amount
        
        values['total_deposits'] = total_deposits
        values['total_withdrawals'] = total_withdrawals
        values['transaction_count'] = len(transactions)
        
        # Validate closing balance calculation
        expected_closing = opening_balance + total_deposits - total_withdrawals
        actual_closing = values.get('closing_balance', Decimal('0'))
        
        if abs(expected_closing - actual_closing) > Decimal('0.01'):
            raise CustomValidationError(
                f'Closing balance mismatch: expected {expected_closing}, got {actual_closing}'
            )
        
        return values

def custom_validators_examples():
    """Examples of custom validators and advanced patterns"""
    
    print("\n🔧 CUSTOM VALIDATORS & ADVANCED PATTERNS")
    print("=" * 50)
    
    # Financial services agent
    financial_agent = Agent(
        name="FinancialAgent",
        instructions="""
        You process financial data with strict validation and business rules.
        Extract account information, validate transactions, and generate reports.
        Always ensure data integrity and compliance with financial regulations.
        """,
        output_type=FinancialReport
    )
    
    # Transaction processor
    transaction_processor = Agent(
        name="TransactionProcessor",
        instructions="""
        You process financial transactions with comprehensive validation.
        Validate account numbers, apply business rules, and ensure compliance.
        Check transaction limits and business logic constraints.
        """,
        output_type=Transaction
    )
    
    print("\n1. Testing Custom Validation Rules:")
    
    # Test financial data processing
    financial_scenarios = [
        """Create a financial report for account:
        Account Number: 1234567890 (checking)
        Routing Number: 123456789
        Account Holder: John Smith
        Balance: $2500.00
        Report period: January 1, 2024 to January 31, 2024
        Opening balance: $2000.00
        Transactions:
        - Deposit $750 on Jan 5
        - Withdrawal $250 on Jan 15
        Closing balance: $2500.00""",
        
        """Process account information:
        Account: 9876543210 (savings)
        Routing: 987654321
        Holder: Jane Doe
        Current Balance: $5000
        Period: Feb 1-28, 2024
        Opening: $4500
        Transactions:
        - Deposit $800 on Feb 10
        - Payment $300 on Feb 20
        Closing: $5000"""
    ]
    
    for i, scenario in enumerate(financial_scenarios, 1):
        try:
            result = Runner.run_sync(financial_agent, scenario)
            print(f"{i}. Financial Report Processing:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                report = result.structured_output
                print(f"   Report ID: {report.report_id}")
                print(f"   Account: {report.account.account_number} ({report.account.account_type})")
                print(f"   Account Holder: {report.account.account_holder}")
                print(f"   Opening Balance: ${report.opening_balance}")
                print(f"   Closing Balance: ${report.closing_balance}")
                print(f"   Total Deposits: ${report.total_deposits}")
                print(f"   Total Withdrawals: ${report.total_withdrawals}")
                print(f"   Transaction Count: {report.transaction_count}")
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Validation Error: {e}")
    
    print("\n2. Testing Transaction Validation:")
    
    transaction_scenarios = [
        """Process transaction:
        ID: TXN001
        From account: 1234567890
        To account: 9876543210
        Amount: $500.00
        Type: transfer
        Description: Monthly payment""",
        
        """Large withdrawal:
        ID: TXN002
        From account: 9876543210
        To account: 0000000000
        Amount: $6000.00
        Type: withdrawal
        Description: Cash withdrawal""",  # This should fail validation
        
        """Valid deposit:
        ID: TXN003
        From account: 0000000000
        To account: 1234567890
        Amount: $1000.00
        Type: deposit
        Description: Salary deposit"""
    ]
    
    for i, scenario in enumerate(transaction_scenarios, 1):
        try:
            result = Runner.run_sync(transaction_processor, scenario)
            print(f"{i}. Transaction Processing:")
            print(f"   Input: {scenario[:100]}...")
            
            if hasattr(result, 'structured_output'):
                transaction = result.structured_output
                print(f"   Transaction ID: {transaction.transaction_id}")
                print(f"   From: {transaction.from_account}")
                print(f"   To: {transaction.to_account}")
                print(f"   Amount: ${transaction.amount}")
                print(f"   Type: {transaction.transaction_type}")
                print("   ✅ Transaction validated successfully")
            else:
                print(f"   Response: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. ❌ Transaction failed validation: {e}")
    
    print("\n3. Custom Validation Benefits:")
    print("💡 Custom Validation Advantages:")
    print("  - Business rule enforcement at data level")
    print("  - Consistent validation across application")
    print("  - Clear error messages for invalid data")
    print("  - Type safety with runtime checks")
    print("  - Automatic data transformation and cleanup")
    print("  - Integration with existing validation frameworks")
    
    return {
        "financial_agent": financial_agent,
        "transaction_processor": transaction_processor
    }

# =============================================================================
# 💡 6. STRUCTURED OUTPUTS BEST PRACTICES
# =============================================================================

"""
💡 STRUCTURED OUTPUTS BEST PRACTICES

1. 🏗️ MODEL DESIGN PRINCIPLES:

   ✅ Single Responsibility: Each model has a clear, focused purpose
   ✅ Composition over Inheritance: Use nested models instead of complex inheritance
   ✅ Explicit over Implicit: Clear field names and types
   ✅ Fail Fast: Validate early and provide clear error messages
   ✅ Immutability: Use frozen models where data shouldn't change

2. 📊 FIELD VALIDATION STRATEGIES:

   Basic Validation:
   ```python
   name: str = Field(..., min_length=1, max_length=100)
   age: int = Field(..., ge=0, le=150)
   email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
   ```
   
   Custom Validation:
   ```python
   @validator('field_name')
   def validate_field(cls, v):
       if not meets_criteria(v):
           raise ValueError('Validation message')
       return v
   ```
   
   Cross-Field Validation:
   ```python
   @root_validator
   def validate_model(cls, values):
       # Validate relationships between fields
       return values
   ```

3. 🔄 OUTPUT_TYPE USAGE PATTERNS:

   Agent with Structured Output:
   ```python
   agent = Agent(
       name="DataAgent",
       instructions="Extract and structure data...",
       output_type=MyModel
   )
   ```
   
   Tool with Structured Output:
   ```python
   @function_tool
   def structured_tool(input: str) -> MyModel:
       # Tool returns structured data
       return MyModel(...)
   ```

4. 🚨 ERROR HANDLING:

   ✅ Provide meaningful validation error messages
   ✅ Handle partial data gracefully
   ✅ Use optional fields for non-critical data
   ✅ Implement fallback values where appropriate
   ✅ Log validation failures for debugging

5. ⚡ PERFORMANCE OPTIMIZATION:

   ✅ Use simple types where possible (str, int, bool)
   ✅ Limit nested model depth
   ✅ Consider field ordering for validation efficiency
   ✅ Use lazy validation for expensive checks
   ✅ Cache validation results when appropriate

6. 🔧 MAINTENANCE & EVOLUTION:

   ✅ Version your models for backward compatibility
   ✅ Use aliases for field name changes
   ✅ Document model changes and migration paths
   ✅ Test model validation thoroughly
   ✅ Monitor validation failure rates in production

🚨 COMMON STRUCTURED OUTPUT PITFALLS:

❌ DON'T create overly complex nested models
❌ DON'T ignore validation errors
❌ DON'T use structured outputs for simple string responses
❌ DON'T forget to handle optional fields properly
❌ DON'T make all fields required without consideration
❌ DON'T ignore performance impact of complex validation

✅ STRUCTURED OUTPUT SUCCESS CHECKLIST:

☐ Clear model purpose and responsibility
☐ Appropriate field types and constraints
☐ Comprehensive validation rules
☐ Error handling and user feedback
☐ Performance testing with large datasets
☐ Documentation and examples
☐ Integration testing with agents
☐ Monitoring and alerting for validation failures

📈 SCALING STRUCTURED OUTPUTS:

For high-volume applications:
- Use efficient serialization formats
- Implement validation caching
- Consider async validation for expensive rules
- Monitor validation performance
- Use schema evolution strategies
- Implement validation result caching
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_structured_outputs_examples():
    """Run all structured outputs examples comprehensively"""
    
    print("🏗️ OPENAI AGENTS SDK - COMPLETE STRUCTURED OUTPUTS DEMONSTRATION")
    print("=" * 80)
    
    # 1. Basic Structured Output Models
    basic_agent = basic_structured_outputs_examples()
    
    # 2. Business Data Models
    business_results = business_data_models_examples()
    
    # 3. E-commerce Data Models
    ecommerce_results = ecommerce_data_models_examples()
    
    # 4. Survey & Form Data Models
    survey_results = survey_form_data_models_examples()
    
    # 5. Custom Validators & Advanced Patterns
    validator_results = custom_validators_examples()
    
    print("\n✅ All structured outputs examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Structured outputs ensure consistent, validated data formats")
    print("- Pydantic models provide powerful validation and serialization")
    print("- Custom validators enforce business rules at the data level")
    print("- Nested models handle complex data relationships")
    print("- Proper error handling guides users to provide valid input")
    print("- Performance optimization becomes important with complex validation")

def run_sync_structured_outputs_examples():
    """Run synchronous structured outputs examples for immediate testing"""
    
    print("🏗️ OPENAI AGENTS SDK - STRUCTURED OUTPUTS SYNC EXAMPLES")
    print("=" * 70)
    
    # Run all structured outputs examples synchronously
    basic_structured_outputs_examples()
    business_data_models_examples()
    ecommerce_data_models_examples()
    survey_form_data_models_examples()
    custom_validators_examples()
    
    print("\n✅ Sync structured outputs examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🏗️ OpenAI Agents SDK - Complete Structured Outputs Template")
    print("This template demonstrates all Structured Output patterns and validation strategies.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_structured_outputs_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_structured_outputs_examples())
    
    print("\n✅ Structured outputs template demonstration complete!")
    print("💡 Use this template as reference for all your structured data needs.")
