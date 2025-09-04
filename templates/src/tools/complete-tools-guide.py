"""
🛠️ OPENAI AGENTS SDK - COMPLETE TOOLS GUIDE & TEMPLATE

This template covers everything about Tools in the OpenAI Agents SDK:
- @function_tool decorator usage and patterns
- Tool parameter types and validation
- Return value handling and formatting
- Tool documentation and descriptions
- Error handling in tools
- Advanced tool patterns
- Tool composition and organization
- Best practices and common pitfalls

📚 Based on: https://openai.github.io/openai-agents-python/ref/tools/
"""

import os
import json
import asyncio
import requests
from typing import List, Dict, Optional, Union, Any, Literal
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool
from agents.tool import Tool, FunctionTool

# =============================================================================
# 📖 UNDERSTANDING TOOLS IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHY TOOLS ARE ESSENTIAL:

Tools extend agent capabilities beyond text generation:
1. ✅ ACCESS REAL DATA: APIs, databases, files
2. ✅ PERFORM ACTIONS: Send emails, create files, make calculations
3. ✅ INTEGRATE SYSTEMS: Connect to external services
4. ✅ VALIDATE INPUTS: Type checking and data validation
5. ✅ STRUCTURED OPERATIONS: Convert natural language to function calls

TOOL CREATION PATTERNS:

1. @function_tool Decorator:
   - Automatically converts Python functions to agent tools
   - Uses function signature for parameter validation
   - Uses docstring for tool description
   - Supports type hints for proper validation

2. Function Signature → Tool Schema:
   ```python
   @function_tool
   def my_tool(param1: str, param2: int = 5) -> str:
       \"\"\"Tool description for the agent.\"\"\"
       return f"Processed {param1} with {param2}"
   ```

3. Agent Integration:
   ```python
   agent = Agent(
       name="ToolAgent",
       instructions="Use tools to help users",
       tools=[my_tool, other_tool]
   )
   ```
"""

# =============================================================================
# 🔧 1. BASIC TOOL CREATION PATTERNS
# =============================================================================

def basic_tool_examples():
    """Basic tool creation patterns and usage"""
    
    print("🔧 BASIC TOOL CREATION PATTERNS")
    print("=" * 50)
    
    # 1.1 Simple Tool - No Parameters
    @function_tool
    def get_current_time() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1.2 Tool with Required Parameters
    @function_tool
    def calculate_tip(bill_amount: float, tip_percentage: float) -> str:
        """Calculate tip amount and total bill.
        
        Args:
            bill_amount: The original bill amount in dollars
            tip_percentage: The tip percentage (e.g., 15.0 for 15%)
        """
        tip_amount = bill_amount * (tip_percentage / 100)
        total_amount = bill_amount + tip_amount
        
        return f"Bill: ${bill_amount:.2f}, Tip: ${tip_amount:.2f}, Total: ${total_amount:.2f}"
    
    # 1.3 Tool with Optional Parameters
    @function_tool
    def greet_user(name: str, greeting: str = "Hello", include_time: bool = False) -> str:
        """Greet a user with customizable greeting.
        
        Args:
            name: The user's name
            greeting: The greeting to use (default: "Hello")
            include_time: Whether to include current time (default: False)
        """
        message = f"{greeting}, {name}!"
        
        if include_time:
            current_time = datetime.now().strftime("%H:%M")
            message += f" It's currently {current_time}."
        
        return message
    
    # 1.4 Tool with Complex Return Value
    @function_tool
    def analyze_text(text: str) -> str:
        """Analyze text and return statistics.
        
        Args:
            text: The text to analyze
        """
        words = text.split()
        sentences = text.split('.')
        characters = len(text)
        
        analysis = {
            "character_count": characters,
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
        
        return json.dumps(analysis, indent=2)
    
    # Create agent with basic tools
    basic_agent = Agent(
        name="BasicToolBot",
        instructions="""
        You have access to several useful tools:
        - Get current time
        - Calculate tips
        - Greet users
        - Analyze text
        
        Use these tools to help users with their requests.
        """,
        tools=[get_current_time, calculate_tip, greet_user, analyze_text]
    )
    
    # Test basic tools
    test_cases = [
        "What time is it?",
        "Calculate a 20% tip on a $50 bill",
        "Greet me as Alice with a friendly hello",
        "Analyze this text: 'The quick brown fox jumps over the lazy dog.'"
    ]
    
    print("\n1. Testing Basic Tools:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = Runner.run_sync(basic_agent, test_case)
            print(f"{i}. Input: {test_case}")
            print(f"   Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return basic_agent, [get_current_time, calculate_tip, greet_user, analyze_text]

# =============================================================================
# 📊 2. ADVANCED PARAMETER TYPES & VALIDATION
# =============================================================================

def advanced_parameter_examples():
    """Advanced parameter types and validation patterns"""
    
    print("\n📊 ADVANCED PARAMETER TYPES")
    print("=" * 50)
    
    # 2.1 Enum Parameters (Restricted Choices)
    class Priority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        URGENT = "urgent"
    
    @function_tool
    def create_task(
        title: str, 
        priority: Priority, 
        due_date: Optional[str] = None,
        assignee: Optional[str] = None
    ) -> str:
        """Create a new task with specified parameters.
        
        Args:
            title: Task title/description
            priority: Task priority level
            due_date: Due date in YYYY-MM-DD format (optional)
            assignee: Person assigned to the task (optional)
        """
        task = {
            "title": title,
            "priority": priority.value,
            "due_date": due_date,
            "assignee": assignee,
            "created_at": datetime.now().isoformat()
        }
        
        return f"Task created: {json.dumps(task, indent=2)}"
    
    # 2.2 Literal Types (Specific String Values)
    @function_tool
    def convert_temperature(
        value: float, 
        from_unit: Literal["celsius", "fahrenheit", "kelvin"],
        to_unit: Literal["celsius", "fahrenheit", "kelvin"]
    ) -> str:
        """Convert temperature between different units.
        
        Args:
            value: Temperature value to convert
            from_unit: Source temperature unit
            to_unit: Target temperature unit
        """
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (value - 32) * 5/9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        else:
            celsius = value
        
        # Convert from Celsius to target unit
        if to_unit == "fahrenheit":
            result = celsius * 9/5 + 32
        elif to_unit == "kelvin":
            result = celsius + 273.15
        else:
            result = celsius
        
        return f"{value}° {from_unit.title()} = {result:.2f}° {to_unit.title()}"
    
    # 2.3 List Parameters
    @function_tool
    def calculate_statistics(numbers: List[float]) -> str:
        """Calculate basic statistics for a list of numbers.
        
        Args:
            numbers: List of numbers to analyze
        """
        if not numbers:
            return "Error: Empty list provided"
        
        stats = {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers)
        }
        
        return json.dumps(stats, indent=2)
    
    # 2.4 Dictionary Parameters
    @function_tool
    def process_user_data(user_info: Dict[str, Any]) -> str:
        """Process user information and generate a summary.
        
        Args:
            user_info: Dictionary containing user information
        """
        required_fields = ["name", "email"]
        missing_fields = [field for field in required_fields if field not in user_info]
        
        if missing_fields:
            return f"Error: Missing required fields: {', '.join(missing_fields)}"
        
        summary = f"User Profile:\n"
        summary += f"Name: {user_info.get('name', 'N/A')}\n"
        summary += f"Email: {user_info.get('email', 'N/A')}\n"
        summary += f"Age: {user_info.get('age', 'Not specified')}\n"
        summary += f"Location: {user_info.get('location', 'Not specified')}\n"
        
        return summary
    
    # 2.5 Complex Nested Types
    @function_tool
    def analyze_sales_data(
        sales_records: List[Dict[str, Union[str, float, int]]]
    ) -> str:
        """Analyze sales data and generate insights.
        
        Args:
            sales_records: List of sales records, each containing product, amount, date
        """
        if not sales_records:
            return "No sales data provided"
        
        total_sales = sum(float(record.get('amount', 0)) for record in sales_records)
        product_sales = {}
        
        for record in sales_records:
            product = record.get('product', 'Unknown')
            amount = float(record.get('amount', 0))
            product_sales[product] = product_sales.get(product, 0) + amount
        
        # Find top product
        top_product = max(product_sales.items(), key=lambda x: x[1]) if product_sales else ("None", 0)
        
        analysis = {
            "total_sales": total_sales,
            "total_records": len(sales_records),
            "average_sale": total_sales / len(sales_records),
            "top_product": f"{top_product[0]} (${top_product[1]:.2f})",
            "unique_products": len(product_sales)
        }
        
        return json.dumps(analysis, indent=2)
    
    # Create agent with advanced parameter tools
    advanced_agent = Agent(
        name="AdvancedToolBot",
        instructions="""
        You have access to advanced tools with complex parameter types:
        - Task creation with priority levels
        - Temperature conversion between units
        - Statistical analysis of number lists
        - User data processing
        - Sales data analysis
        
        Always validate inputs and provide clear feedback.
        """,
        tools=[create_task, convert_temperature, calculate_statistics, process_user_data, analyze_sales_data]
    )
    
    # Test advanced tools
    advanced_test_cases = [
        "Create a high priority task called 'Fix bug in login system' assigned to John",
        "Convert 100 degrees fahrenheit to celsius",
        "Calculate statistics for these numbers: 10, 20, 30, 40, 50",
        "Process user data: name=Alice, email=alice@example.com, age=30"
    ]
    
    print("\n2. Testing Advanced Parameter Tools:")
    for i, test_case in enumerate(advanced_test_cases, 1):
        try:
            result = Runner.run_sync(advanced_agent, test_case)
            print(f"{i}. Input: {test_case}")
            print(f"   Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return advanced_agent

# =============================================================================
# 🌐 3. EXTERNAL API INTEGRATION TOOLS
# =============================================================================

def external_api_examples():
    """Tools that integrate with external APIs and services"""
    
    print("\n🌐 EXTERNAL API INTEGRATION TOOLS")
    print("=" * 50)
    
    # 3.1 Weather API Tool (Mock Implementation)
    @function_tool
    def get_weather_info(city: str, country_code: Optional[str] = None) -> str:
        """Get current weather information for a city.
        
        Args:
            city: Name of the city
            country_code: Optional 2-letter country code (e.g., 'US', 'UK')
        """
        # Mock implementation - in real use, you'd call a weather API
        location = f"{city}, {country_code}" if country_code else city
        
        # Simulate API response
        mock_weather = {
            "location": location,
            "temperature": "22°C",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "wind": "10 km/h",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(mock_weather, indent=2)
    
    # 3.2 News API Tool (Mock Implementation)
    @function_tool
    def get_news_headlines(category: str = "general", country: str = "us") -> str:
        """Get latest news headlines by category.
        
        Args:
            category: News category (business, entertainment, general, health, science, sports, technology)
            country: Country code for news (us, uk, ca, etc.)
        """
        # Mock implementation
        mock_headlines = [
            f"Breaking: Major development in {category} sector",
            f"Scientists discover new breakthrough in {category}",
            f"Market analysis shows trends in {category} industry",
            f"Government announces new policies affecting {category}",
            f"International cooperation grows in {category} field"
        ]
        
        result = {
            "category": category,
            "country": country,
            "headlines": mock_headlines[:3],  # Return top 3
            "total_articles": len(mock_headlines),
            "last_updated": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2)
    
    # 3.3 Search Tool (Mock Implementation)
    @function_tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (1-10)
        """
        # Mock implementation
        mock_results = []
        
        for i in range(min(max_results, 5)):
            mock_results.append({
                "title": f"Result {i+1}: {query} - Comprehensive Guide",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This comprehensive guide covers everything about {query}. Learn more about the latest developments and best practices."
            })
        
        result = {
            "query": query,
            "results_count": len(mock_results),
            "results": mock_results
        }
        
        return json.dumps(result, indent=2)
    
    # 3.4 Email Tool (Mock Implementation)
    @function_tool
    def send_email(to: str, subject: str, body: str, cc: Optional[str] = None) -> str:
        """Send an email message.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            cc: Optional CC recipients (comma-separated)
        """
        # Mock implementation - validate email format
        if "@" not in to:
            return "Error: Invalid email address format"
        
        email_data = {
            "to": to,
            "cc": cc,
            "subject": subject,
            "body": body[:100] + "..." if len(body) > 100 else body,
            "status": "sent",
            "timestamp": datetime.now().isoformat()
        }
        
        return f"Email sent successfully: {json.dumps(email_data, indent=2)}"
    
    # 3.5 Database Query Tool (Mock Implementation)
    @function_tool
    def database_query(table: str, operation: Literal["select", "count", "insert"], filters: Optional[Dict[str, Any]] = None) -> str:
        """Execute a database operation.
        
        Args:
            table: Table name to query
            operation: Database operation to perform
            filters: Optional filters for the query
        """
        # Mock implementation
        if operation == "select":
            mock_data = [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ]
            
            # Apply mock filtering
            if filters:
                mock_data = [row for row in mock_data if any(str(row.get(k)) == str(v) for k, v in filters.items())]
            
            return json.dumps({"table": table, "operation": operation, "data": mock_data}, indent=2)
        
        elif operation == "count":
            count = 2 if not filters else 1  # Mock count
            return json.dumps({"table": table, "operation": operation, "count": count}, indent=2)
        
        else:
            return json.dumps({"table": table, "operation": operation, "status": "success"}, indent=2)
    
    # Create agent with API integration tools
    api_agent = Agent(
        name="APIIntegrationBot",
        instructions="""
        You can access external services and data through various tools:
        - Weather information for any city
        - Latest news headlines by category
        - Web search capabilities
        - Email sending functionality
        - Database operations
        
        Always provide helpful information and handle errors gracefully.
        When using these tools, explain what you're doing and interpret the results for the user.
        """,
        tools=[get_weather_info, get_news_headlines, web_search, send_email, database_query]
    )
    
    # Test API integration tools
    api_test_cases = [
        "What's the weather like in London?",
        "Get me the latest technology news",
        "Search for information about Python programming",
        "Send an email to alice@example.com about the meeting tomorrow",
        "Query the users table to count all records"
    ]
    
    print("\n3. Testing API Integration Tools:")
    for i, test_case in enumerate(api_test_cases, 1):
        try:
            result = Runner.run_sync(api_agent, test_case)
            print(f"{i}. Input: {test_case}")
            print(f"   Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return api_agent

# =============================================================================
# 🛡️ 4. ERROR HANDLING & VALIDATION TOOLS
# =============================================================================

def error_handling_examples():
    """Tools with robust error handling and validation"""
    
    print("\n🛡️ ERROR HANDLING & VALIDATION TOOLS")
    print("=" * 50)
    
    # 4.1 Tool with Input Validation
    @function_tool
    def validate_and_format_phone(phone_number: str, country_code: str = "US") -> str:
        """Validate and format a phone number.
        
        Args:
            phone_number: Phone number to validate
            country_code: Country code for formatting (default: US)
        """
        try:
            # Remove all non-digit characters
            digits_only = ''.join(filter(str.isdigit, phone_number))
            
            # Validation based on country
            if country_code.upper() == "US":
                if len(digits_only) == 10:
                    formatted = f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
                    return f"Valid US phone number: {formatted}"
                elif len(digits_only) == 11 and digits_only[0] == '1':
                    formatted = f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
                    return f"Valid US phone number: {formatted}"
                else:
                    return f"Error: Invalid US phone number. Expected 10 or 11 digits, got {len(digits_only)}"
            else:
                # Basic international validation
                if len(digits_only) >= 7 and len(digits_only) <= 15:
                    return f"Phone number for {country_code}: +{digits_only}"
                else:
                    return f"Error: Invalid phone number length for {country_code}"
        
        except Exception as e:
            return f"Error processing phone number: {str(e)}"
    
    # 4.2 Tool with Safe File Operations
    @function_tool
    def safe_file_operation(operation: Literal["read", "write", "list"], filename: str, content: Optional[str] = None) -> str:
        """Perform safe file operations with error handling.
        
        Args:
            operation: File operation to perform
            filename: Name of the file
            content: Content to write (required for write operation)
        """
        try:
            # Security: Basic path validation
            if ".." in filename or filename.startswith("/") or "\\" in filename:
                return "Error: Invalid filename - security violation"
            
            # Simulate file operations (mock implementation)
            if operation == "read":
                # Mock reading file
                mock_content = f"Mock content from {filename}\nLine 2 of the file\nEnd of file"
                return f"File '{filename}' contents:\n{mock_content}"
            
            elif operation == "write":
                if content is None:
                    return "Error: Content is required for write operation"
                
                # Mock writing file
                return f"Successfully wrote {len(content)} characters to '{filename}'"
            
            elif operation == "list":
                # Mock directory listing
                mock_files = ["file1.txt", "file2.txt", "data.json", filename]
                return f"Files in directory: {', '.join(mock_files)}"
            
            else:
                return f"Error: Unknown operation '{operation}'"
        
        except Exception as e:
            return f"File operation error: {str(e)}"
    
    # 4.3 Tool with Data Validation and Cleanup
    @function_tool
    def process_csv_data(csv_content: str, expected_columns: List[str]) -> str:
        """Process CSV data with validation and cleanup.
        
        Args:
            csv_content: CSV data as string
            expected_columns: List of expected column names
        """
        try:
            lines = csv_content.strip().split('\n')
            if not lines:
                return "Error: Empty CSV data"
            
            # Parse header
            header = [col.strip() for col in lines[0].split(',')]
            
            # Validate expected columns
            missing_columns = [col for col in expected_columns if col not in header]
            if missing_columns:
                return f"Error: Missing columns: {', '.join(missing_columns)}"
            
            # Process data rows
            processed_rows = []
            errors = []
            
            for i, line in enumerate(lines[1:], start=2):
                try:
                    values = [val.strip() for val in line.split(',')]
                    if len(values) != len(header):
                        errors.append(f"Row {i}: Column count mismatch")
                        continue
                    
                    row_dict = dict(zip(header, values))
                    processed_rows.append(row_dict)
                
                except Exception as e:
                    errors.append(f"Row {i}: {str(e)}")
            
            result = {
                "total_rows": len(lines) - 1,
                "processed_rows": len(processed_rows),
                "errors": errors,
                "sample_data": processed_rows[:3] if processed_rows else [],
                "columns": header
            }
            
            return json.dumps(result, indent=2)
        
        except Exception as e:
            return f"CSV processing error: {str(e)}"
    
    # 4.4 Tool with Timeout and Rate Limiting (Mock)
    @function_tool
    def api_with_limits(endpoint: str, method: str = "GET", timeout_seconds: int = 30) -> str:
        """Make API call with timeout and rate limiting.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method
            timeout_seconds: Request timeout in seconds
        """
        try:
            # Mock rate limiting check
            import time
            from random import random
            
            if random() < 0.1:  # 10% chance of rate limit
                return "Error: Rate limit exceeded. Please wait before making another request."
            
            # Mock timeout simulation
            if timeout_seconds < 1:
                return "Error: Timeout too short"
            
            # Mock API response
            response_data = {
                "endpoint": endpoint,
                "method": method,
                "status": "success",
                "data": f"Mock response from {endpoint}",
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": int(random() * 1000)
            }
            
            return json.dumps(response_data, indent=2)
        
        except Exception as e:
            return f"API call error: {str(e)}"
    
    # 4.5 Tool with Comprehensive Error Reporting
    @function_tool
    def comprehensive_validator(data: Dict[str, Any], schema: Dict[str, str]) -> str:
        """Validate data against a schema with detailed error reporting.
        
        Args:
            data: Data to validate
            schema: Schema definition (field_name: type_name)
        """
        try:
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "processed_fields": []
            }
            
            # Check required fields
            for field_name, field_type in schema.items():
                if field_name not in data:
                    validation_results["errors"].append(f"Missing required field: {field_name}")
                    validation_results["valid"] = False
                    continue
                
                value = data[field_name]
                
                # Type validation
                if field_type == "string" and not isinstance(value, str):
                    validation_results["errors"].append(f"Field '{field_name}' must be a string, got {type(value).__name__}")
                    validation_results["valid"] = False
                elif field_type == "number" and not isinstance(value, (int, float)):
                    validation_results["errors"].append(f"Field '{field_name}' must be a number, got {type(value).__name__}")
                    validation_results["valid"] = False
                elif field_type == "email" and not isinstance(value, str):
                    validation_results["errors"].append(f"Field '{field_name}' must be an email string")
                    validation_results["valid"] = False
                elif field_type == "email" and "@" not in str(value):
                    validation_results["errors"].append(f"Field '{field_name}' is not a valid email format")
                    validation_results["valid"] = False
                else:
                    validation_results["processed_fields"].append({
                        "field": field_name,
                        "type": field_type,
                        "value": str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    })
            
            # Check for extra fields
            extra_fields = [field for field in data.keys() if field not in schema.keys()]
            if extra_fields:
                validation_results["warnings"].append(f"Extra fields found: {', '.join(extra_fields)}")
            
            return json.dumps(validation_results, indent=2)
        
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    # Create agent with error handling tools
    error_handling_agent = Agent(
        name="ErrorHandlingBot",
        instructions="""
        You have access to robust tools with comprehensive error handling:
        - Phone number validation and formatting
        - Safe file operations
        - CSV data processing with validation
        - API calls with timeout and rate limiting
        - Data validation against schemas
        
        Always handle errors gracefully and provide clear feedback about what went wrong and how to fix it.
        """,
        tools=[validate_and_format_phone, safe_file_operation, process_csv_data, api_with_limits, comprehensive_validator]
    )
    
    # Test error handling tools
    error_test_cases = [
        "Validate this phone number: 555-123-4567",
        "Read the contents of file test.txt",
        "Process this CSV: 'name,email\\nJohn,john@example.com\\nJane,jane@example.com' with expected columns: name, email",
        "Make an API call to /users endpoint",
        "Validate this data: {'name': 'Alice', 'age': 30} against schema: {'name': 'string', 'email': 'email'}"
    ]
    
    print("\n4. Testing Error Handling Tools:")
    for i, test_case in enumerate(error_test_cases, 1):
        try:
            result = Runner.run_sync(error_handling_agent, test_case)
            print(f"{i}. Input: {test_case}")
            print(f"   Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return error_handling_agent

# =============================================================================
# 🔄 5. ASYNC & CONCURRENT TOOLS
# =============================================================================

async def async_tools_examples():
    """Tools that handle async operations and concurrency"""
    
    print("\n🔄 ASYNC & CONCURRENT TOOLS")
    print("=" * 50)
    
    # 5.1 Async Tool with Delays
    @function_tool
    async def fetch_data_async(source: str, delay_seconds: float = 1.0) -> str:
        """Fetch data asynchronously with configurable delay.
        
        Args:
            source: Data source identifier
            delay_seconds: Simulated network delay
        """
        await asyncio.sleep(delay_seconds)
        
        mock_data = {
            "source": source,
            "data": f"Fetched data from {source}",
            "timestamp": datetime.now().isoformat(),
            "delay_used": delay_seconds
        }
        
        return json.dumps(mock_data, indent=2)
    
    # 5.2 Batch Processing Tool
    @function_tool
    async def batch_process_items(items: List[str], batch_size: int = 3) -> str:
        """Process items in batches asynchronously.
        
        Args:
            items: List of items to process
            batch_size: Number of items to process in each batch
        """
        if not items:
            return "No items to process"
        
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Simulate async processing of batch
            await asyncio.sleep(0.5)  # Simulate work
            
            batch_result = {
                "batch_number": (i // batch_size) + 1,
                "items_processed": batch,
                "processing_time": 0.5,
                "status": "completed"
            }
            
            results.append(batch_result)
        
        summary = {
            "total_items": len(items),
            "total_batches": len(results),
            "batch_size": batch_size,
            "batches": results
        }
        
        return json.dumps(summary, indent=2)
    
    # 5.3 Concurrent Operations Tool
    @function_tool
    async def concurrent_operations(operations: List[Dict[str, Any]]) -> str:
        """Execute multiple operations concurrently.
        
        Args:
            operations: List of operations to execute, each with 'name' and 'duration'
        """
        if not operations:
            return "No operations specified"
        
        async def execute_operation(op):
            name = op.get('name', 'unknown')
            duration = float(op.get('duration', 1.0))
            
            await asyncio.sleep(duration)
            
            return {
                "name": name,
                "duration": duration,
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            }
        
        # Execute all operations concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[execute_operation(op) for op in operations])
        end_time = asyncio.get_event_loop().time()
        
        summary = {
            "total_operations": len(operations),
            "concurrent_execution_time": round(end_time - start_time, 2),
            "sequential_time_would_be": sum(float(op.get('duration', 1.0)) for op in operations),
            "results": results
        }
        
        return json.dumps(summary, indent=2)
    
    # Create agent with async tools
    async_agent = Agent(
        name="AsyncToolBot",
        instructions="""
        You have access to asynchronous tools that can handle concurrent operations:
        - Fetch data with configurable delays
        - Process items in batches
        - Execute multiple operations concurrently
        
        These tools are optimized for performance and can handle multiple requests efficiently.
        """,
        tools=[fetch_data_async, batch_process_items, concurrent_operations]
    )
    
    # Test async tools
    async_test_cases = [
        "Fetch data from 'api.example.com' with a 2 second delay",
        "Process these items in batches: ['item1', 'item2', 'item3', 'item4', 'item5'] with batch size 2",
        "Execute these operations concurrently: [{'name': 'task1', 'duration': 1}, {'name': 'task2', 'duration': 2}]"
    ]
    
    print("\n5. Testing Async Tools:")
    for i, test_case in enumerate(async_test_cases, 1):
        try:
            result = await Runner.run(async_agent, test_case)
            print(f"{i}. Input: {test_case}")
            print(f"   Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    return async_agent

# =============================================================================
# 🏗️ 6. TOOL COMPOSITION & ORGANIZATION
# =============================================================================

def tool_organization_examples():
    """Examples of organizing and composing tools effectively"""
    
    print("\n🏗️ TOOL COMPOSITION & ORGANIZATION")
    print("=" * 50)
    
    # 6.1 Tool Categories - Math Tools
    class MathTools:
        """Collection of mathematical tools"""
        
        @staticmethod
        @function_tool
        def basic_calculator(expression: str) -> str:
            """Calculate basic mathematical expressions safely.
            
            Args:
                expression: Mathematical expression (e.g., '2 + 3 * 4')
            """
            try:
                # Safe evaluation - only allow basic math
                allowed_chars = set('0123456789+-*/(). ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Expression contains invalid characters"
                
                result = eval(expression)
                return f"{expression} = {result}"
            except Exception as e:
                return f"Error calculating '{expression}': {str(e)}"
        
        @staticmethod
        @function_tool
        def percentage_calculator(value: float, percentage: float) -> str:
            """Calculate percentage of a value.
            
            Args:
                value: Base value
                percentage: Percentage to calculate
            """
            result = value * (percentage / 100)
            return f"{percentage}% of {value} = {result}"
        
        @staticmethod
        @function_tool
        def compound_interest(principal: float, rate: float, time: float, compound_frequency: int = 12) -> str:
            """Calculate compound interest.
            
            Args:
                principal: Initial amount
                rate: Annual interest rate (as percentage)
                time: Time in years
                compound_frequency: Compounding frequency per year (default: 12 for monthly)
            """
            amount = principal * (1 + (rate / 100) / compound_frequency) ** (compound_frequency * time)
            interest = amount - principal
            
            result = {
                "principal": principal,
                "rate": f"{rate}%",
                "time": f"{time} years",
                "compound_frequency": f"{compound_frequency} times/year",
                "final_amount": round(amount, 2),
                "interest_earned": round(interest, 2)
            }
            
            return json.dumps(result, indent=2)
    
    # 6.2 Tool Categories - Text Tools
    class TextTools:
        """Collection of text processing tools"""
        
        @staticmethod
        @function_tool
        def text_statistics(text: str) -> str:
            """Get comprehensive text statistics.
            
            Args:
                text: Text to analyze
            """
            words = text.split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            stats = {
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(' ', '')),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "average_words_per_sentence": round(len(words) / len(sentences), 2) if sentences else 0,
                "reading_time_minutes": round(len(words) / 200, 2)  # Assuming 200 WPM
            }
            
            return json.dumps(stats, indent=2)
        
        @staticmethod
        @function_tool
        def text_formatter(text: str, format_type: Literal["uppercase", "lowercase", "title", "sentence"]) -> str:
            """Format text in different ways.
            
            Args:
                text: Text to format
                format_type: Type of formatting to apply
            """
            if format_type == "uppercase":
                return text.upper()
            elif format_type == "lowercase":
                return text.lower()
            elif format_type == "title":
                return text.title()
            elif format_type == "sentence":
                return text.capitalize()
            else:
                return f"Unknown format type: {format_type}"
        
        @staticmethod
        @function_tool
        def find_and_replace(text: str, find: str, replace: str, case_sensitive: bool = True) -> str:
            """Find and replace text.
            
            Args:
                text: Source text
                find: Text to find
                replace: Replacement text
                case_sensitive: Whether search is case sensitive
            """
            if not case_sensitive:
                # Case insensitive replacement
                import re
                pattern = re.compile(re.escape(find), re.IGNORECASE)
                result_text = pattern.sub(replace, text)
            else:
                result_text = text.replace(find, replace)
            
            count = text.count(find) if case_sensitive else text.lower().count(find.lower())
            
            result = {
                "original_text": text,
                "find": find,
                "replace": replace,
                "replacements_made": count,
                "result_text": result_text
            }
            
            return json.dumps(result, indent=2)
    
    # 6.3 Tool Categories - Data Tools  
    class DataTools:
        """Collection of data processing tools"""
        
        @staticmethod
        @function_tool
        def json_validator(json_string: str) -> str:
            """Validate and format JSON data.
            
            Args:
                json_string: JSON string to validate
            """
            try:
                parsed = json.loads(json_string)
                formatted = json.dumps(parsed, indent=2)
                
                result = {
                    "valid": True,
                    "formatted_json": formatted,
                    "type": type(parsed).__name__,
                    "size": len(json_string)
                }
                
                return json.dumps(result, indent=2)
                
            except json.JSONDecodeError as e:
                return json.dumps({
                    "valid": False,
                    "error": str(e),
                    "error_line": getattr(e, 'lineno', None),
                    "error_column": getattr(e, 'colno', None)
                }, indent=2)
        
        @staticmethod
        @function_tool
        def data_converter(data: str, from_format: Literal["csv", "json"], to_format: Literal["csv", "json"]) -> str:
            """Convert data between different formats.
            
            Args:
                data: Data to convert
                from_format: Source format
                to_format: Target format
            """
            try:
                if from_format == to_format:
                    return f"Data is already in {to_format} format"
                
                if from_format == "csv" and to_format == "json":
                    lines = data.strip().split('\n')
                    if not lines:
                        return "Error: Empty CSV data"
                    
                    headers = [h.strip() for h in lines[0].split(',')]
                    rows = []
                    
                    for line in lines[1:]:
                        values = [v.strip() for v in line.split(',')]
                        if len(values) == len(headers):
                            rows.append(dict(zip(headers, values)))
                    
                    return json.dumps(rows, indent=2)
                
                elif from_format == "json" and to_format == "csv":
                    parsed = json.loads(data)
                    if not isinstance(parsed, list) or not parsed:
                        return "Error: JSON must be a non-empty array of objects"
                    
                    headers = list(parsed[0].keys())
                    csv_lines = [','.join(headers)]
                    
                    for row in parsed:
                        values = [str(row.get(h, '')) for h in headers]
                        csv_lines.append(','.join(values))
                    
                    return '\n'.join(csv_lines)
                
                else:
                    return f"Conversion from {from_format} to {to_format} not supported"
                    
            except Exception as e:
                return f"Conversion error: {str(e)}"
    
    # 6.4 Create Specialized Agents with Tool Categories
    
    # Math-focused agent
    math_agent = Agent(
        name="MathExpert",
        instructions="""
        You are a mathematics expert with access to calculation tools.
        Help users with:
        - Basic calculations and expressions
        - Percentage calculations
        - Compound interest and financial math
        
        Always show your work and explain the results.
        """,
        tools=[MathTools.basic_calculator, MathTools.percentage_calculator, MathTools.compound_interest]
    )
    
    # Text processing agent
    text_agent = Agent(
        name="TextProcessor",
        instructions="""
        You are a text processing expert with tools for:
        - Text analysis and statistics
        - Text formatting and transformation
        - Find and replace operations
        
        Help users analyze and manipulate text effectively.
        """,
        tools=[TextTools.text_statistics, TextTools.text_formatter, TextTools.find_and_replace]
    )
    
    # Data processing agent
    data_agent = Agent(
        name="DataProcessor",
        instructions="""
        You are a data processing expert with tools for:
        - JSON validation and formatting
        - Data format conversion (CSV ↔ JSON)
        
        Help users work with structured data efficiently.
        """,
        tools=[DataTools.json_validator, DataTools.data_converter]
    )
    
    # Multi-purpose agent with all tools
    swiss_army_agent = Agent(
        name="SwissArmyBot",
        instructions="""
        You are a versatile assistant with comprehensive tool access:
        
        📊 Math Tools:
        - Basic calculations
        - Percentage calculations  
        - Compound interest calculations
        
        📝 Text Tools:
        - Text statistics and analysis
        - Text formatting
        - Find and replace
        
        💾 Data Tools:
        - JSON validation
        - Data format conversion
        
        Choose the appropriate tools based on the user's needs and explain your process.
        """,
        tools=[
            # Math tools
            MathTools.basic_calculator,
            MathTools.percentage_calculator,
            MathTools.compound_interest,
            # Text tools
            TextTools.text_statistics,
            TextTools.text_formatter,
            TextTools.find_and_replace,
            # Data tools
            DataTools.json_validator,
            DataTools.data_converter
        ]
    )
    
    # Test tool organization
    test_cases = [
        ("Math", math_agent, "Calculate 15% of 250 and also compute compound interest on $1000 at 5% for 2 years"),
        ("Text", text_agent, "Analyze this text and convert it to title case: 'hello world this is a test'"),
        ("Data", data_agent, "Validate this JSON: '{\"name\": \"Alice\", \"age\": 30}'"),
        ("Swiss Army", swiss_army_agent, "I need help with math: what's 25 * 4, text: make 'HELLO' lowercase, and data: convert CSV 'name,age\\nAlice,30' to JSON")
    ]
    
    print("\n6. Testing Tool Organization:")
    for category, agent, test_case in test_cases:
        try:
            result = Runner.run_sync(agent, test_case)
            print(f"{category} Agent:")
            print(f"  Input: {test_case}")
            print(f"  Output: {result.final_output}")
            print()
        except Exception as e:
            print(f"{category} Agent Error: {e}")
    
    return {
        "math_agent": math_agent,
        "text_agent": text_agent, 
        "data_agent": data_agent,
        "swiss_army_agent": swiss_army_agent
    }

# =============================================================================
# 💡 7. TOOL BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 TOOL DEVELOPMENT BEST PRACTICES

1. 📝 DOCUMENTATION:
   ✅ Write clear, detailed docstrings
   ✅ Explain all parameters and return values
   ✅ Include usage examples in docstrings
   ❌ Don't use vague or missing descriptions

2. 🔍 PARAMETER VALIDATION:
   ✅ Use type hints for all parameters
   ✅ Provide sensible defaults
   ✅ Validate inputs and provide clear error messages
   ❌ Don't assume inputs are always valid

3. 🛡️ ERROR HANDLING:
   ✅ Wrap operations in try-catch blocks
   ✅ Return meaningful error messages
   ✅ Handle edge cases gracefully
   ❌ Don't let exceptions crash the agent

4. 🔄 RETURN VALUES:
   ✅ Return consistent, structured data
   ✅ Use JSON for complex data
   ✅ Include status/success indicators
   ❌ Don't return raw exceptions or unclear data

5. ⚡ PERFORMANCE:
   ✅ Use async for I/O operations
   ✅ Implement timeouts for external calls
   ✅ Cache results when appropriate
   ❌ Don't block the agent with long operations

6. 🔐 SECURITY:
   ✅ Validate and sanitize all inputs
   ✅ Use safe evaluation methods
   ✅ Implement access controls
   ❌ Don't execute arbitrary code or access sensitive data

📊 COMMON TOOL PATTERNS:

Pattern 1 - Data Processor:
```python
@function_tool
def process_data(data: Dict[str, Any], operation: Literal["validate", "transform"]) -> str:
    try:
        # Validate inputs
        if not data:
            return "Error: No data provided"
        
        # Process based on operation
        result = perform_operation(data, operation)
        
        # Return structured response
        return json.dumps({
            "status": "success",
            "operation": operation,
            "result": result
        }, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"
```

Pattern 2 - External Service:
```python
@function_tool
def call_service(endpoint: str, params: Optional[Dict] = None) -> str:
    try:
        # Input validation
        if not endpoint:
            return "Error: Endpoint required"
        
        # Make service call with timeout
        response = make_request(endpoint, params, timeout=30)
        
        # Return formatted response
        return json.dumps(response, indent=2)
        
    except TimeoutError:
        return "Error: Service request timed out"
    except Exception as e:
        return f"Error: {str(e)}"
```

Pattern 3 - Batch Processor:
```python
@function_tool
def batch_process(items: List[str], batch_size: int = 10) -> str:
    try:
        results = []
        errors = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                batch_result = process_batch(batch)
                results.extend(batch_result)
            except Exception as e:
                errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
        
        return json.dumps({
            "processed": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors
        }, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"
```

🚨 COMMON PITFALLS TO AVOID:

❌ DON'T: Return raw Python objects (use JSON strings)
❌ DON'T: Use print() for output (return strings instead)
❌ DON'T: Ignore type hints (they enable validation)
❌ DON'T: Make tools overly complex (split into smaller tools)
❌ DON'T: Forget error handling (always wrap in try-catch)
❌ DON'T: Use blocking operations without timeouts
❌ DON'T: Hard-code values (use parameters instead)

✅ TESTING YOUR TOOLS:

1. Test with valid inputs
2. Test with invalid inputs
3. Test edge cases (empty, null, extreme values)
4. Test error conditions
5. Test with real agents
6. Test performance with large inputs

🔧 TOOL DEBUGGING CHECKLIST:

☐ Tool description is clear and helpful
☐ All parameters have type hints
☐ Default values are sensible
☐ Error handling covers all cases
☐ Return values are consistent
☐ Function works independently
☐ Agent can understand and use the tool
☐ Performance is acceptable
☐ Security considerations addressed
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_tool_examples():
    """Run all tool examples comprehensively"""
    
    print("🛠️ OPENAI AGENTS SDK - COMPLETE TOOLS DEMONSTRATION")
    print("=" * 70)
    
    # 1. Basic Tools
    basic_tools = basic_tool_examples()
    
    # 2. Advanced Parameters
    advanced_agent = advanced_parameter_examples()
    
    # 3. API Integration
    api_agent = external_api_examples()
    
    # 4. Error Handling
    error_agent = error_handling_examples()
    
    # 5. Async Tools
    await async_tools_examples()
    
    # 6. Tool Organization
    organized_tools = tool_organization_examples()
    
    print("\n✅ All tool examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Tools extend agent capabilities beyond text generation")
    print("- Use type hints and validation for robust tools")
    print("- Organize tools by category for better maintainability")
    print("- Always handle errors gracefully")
    print("- Use async tools for I/O operations")
    print("- Document tools clearly for both agents and developers")

def run_sync_tool_examples():
    """Run synchronous tool examples for immediate testing"""
    
    print("🛠️ OPENAI AGENTS SDK - TOOLS SYNC EXAMPLES")
    print("=" * 60)
    
    # Run only synchronous examples
    basic_tool_examples()
    advanced_parameter_examples()
    external_api_examples()
    error_handling_examples()
    tool_organization_examples()
    
    print("\n✅ Sync tool examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🛠️ OpenAI Agents SDK - Complete Tools Template")
    print("This template demonstrates all Tool creation patterns and best practices.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_tool_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_tool_examples())
    
    print("\n✅ Tools template demonstration complete!")
    print("💡 Use this template as reference for all your tool development needs.")
