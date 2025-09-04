"""
📡 OPENAI AGENTS SDK - COMPLETE STREAMING GUIDE & TEMPLATE

This template covers everything about Streaming in the OpenAI Agents SDK:
- Real-time response streaming patterns
- Progress tracking and status updates
- Event handling and callbacks
- Async streaming implementations
- Performance optimization for streaming
- Error handling in streaming contexts
- UI integration patterns for streaming
- Advanced streaming use cases

📚 Based on: https://openai.github.io/openai-agents-python/ref/streaming/
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator, Generator, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue
import threading

from agents import Agent, Runner, function_tool

# =============================================================================
# 📖 UNDERSTANDING STREAMING IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT IS STREAMING?

Streaming provides real-time response delivery:
1. ✅ REAL-TIME FEEDBACK: Users see responses as they're generated
2. ✅ PROGRESS TRACKING: Monitor long-running operations
3. ✅ BETTER UX: Reduced perceived latency
4. ✅ INCREMENTAL PROCESSING: Handle large responses efficiently
5. ✅ INTERRUPTIBLE: Cancel operations mid-stream

KEY CONCEPTS:

1. STREAMING TYPES:
   - Text Streaming: Character-by-character or token-by-token
   - Event Streaming: Structured events with metadata
   - Progress Streaming: Status updates for long operations
   - Data Streaming: Large datasets in chunks

2. STREAMING PATTERNS:
   - Generator Functions: yield results incrementally
   - Async Generators: async yield for non-blocking streaming
   - Event Callbacks: Function calls on stream events
   - Observer Pattern: Subscribe to stream updates

3. STREAMING LIFECYCLE:
   - Initialize: Set up stream and handlers
   - Stream: Emit data/events incrementally
   - Complete: Signal end of stream
   - Error: Handle and communicate failures

4. USE CASES:
   - Chat interfaces with typing indicators
   - Long document processing with progress
   - Real-time data analysis and reporting
   - File uploads/downloads with progress bars
   - Live code generation and execution
"""

# =============================================================================
# 📺 1. BASIC STREAMING PATTERNS
# =============================================================================

class StreamEvent:
    """Base class for streaming events"""
    
    def __init__(self, event_type: str, data: Any = None, timestamp: datetime = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

class TextStreamEvent(StreamEvent):
    """Text streaming event"""
    
    def __init__(self, text_chunk: str, is_complete: bool = False):
        super().__init__('text_chunk', {
            'text': text_chunk,
            'is_complete': is_complete
        })

class ProgressStreamEvent(StreamEvent):
    """Progress streaming event"""
    
    def __init__(self, progress: float, message: str = "", step: str = ""):
        super().__init__('progress', {
            'progress': progress,
            'message': message,
            'step': step
        })

class ErrorStreamEvent(StreamEvent):
    """Error streaming event"""
    
    def __init__(self, error_message: str, error_code: str = None):
        super().__init__('error', {
            'message': error_message,
            'code': error_code
        })

def basic_streaming_examples():
    """Basic streaming pattern examples"""
    
    print("📺 BASIC STREAMING PATTERNS")
    print("=" * 50)
    
    # 1.1 Simple Text Streaming Generator
    def stream_text_simple(text: str, chunk_size: int = 5) -> Generator[str, None, None]:
        """Stream text in chunks"""
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield chunk
            time.sleep(0.1)  # Simulate processing delay
    
    # 1.2 Event-Based Streaming
    def stream_with_events(text: str, chunk_size: int = 10) -> Generator[StreamEvent, None, None]:
        """Stream text with structured events"""
        total_chars = len(text)
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            
            # Emit progress event
            progress = (i + len(chunk)) / total_chars
            yield ProgressStreamEvent(progress, f"Processing... {progress:.1%}")
            
            # Emit text event
            is_complete = (i + len(chunk)) >= total_chars
            yield TextStreamEvent(chunk, is_complete)
            
            time.sleep(0.05)
    
    # 1.3 Async Streaming Generator
    async def stream_async_text(text: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
        """Async text streaming"""
        words = text.split()
        
        for word in words:
            yield word + " "
            await asyncio.sleep(delay)
    
    print("\n1. Testing Basic Streaming Patterns:")
    
    # Test simple text streaming
    sample_text = "This is a sample text that will be streamed in chunks to demonstrate real-time processing."
    
    print("Simple Text Streaming:")
    streamed_text = ""
    for chunk in stream_text_simple(sample_text, chunk_size=8):
        streamed_text += chunk
        print(f"Chunk: '{chunk}' | Total so far: '{streamed_text}'")
    
    print("\nEvent-Based Streaming:")
    full_text = ""
    for event in stream_with_events(sample_text, chunk_size=12):
        if event.event_type == 'progress':
            print(f"Progress: {event.data['progress']:.1%} - {event.data['message']}")
        elif event.event_type == 'text_chunk':
            full_text += event.data['text']
            print(f"Text chunk: '{event.data['text']}' | Complete: {event.data['is_complete']}")
    
    print(f"\nFinal text: {full_text}")
    
    # Test async streaming
    async def test_async_streaming():
        print("\nAsync Streaming:")
        async_text = ""
        async for word in stream_async_text("Async streaming provides non-blocking real-time updates"):
            async_text += word
            print(f"Word: '{word.strip()}' | Total: '{async_text.strip()}'")
    
    # Run async test
    try:
        asyncio.run(test_async_streaming())
    except Exception as e:
        print(f"Async streaming error: {e}")
    
    return {
        "text_streamer": stream_text_simple,
        "event_streamer": stream_with_events,
        "async_streamer": stream_async_text
    }

# =============================================================================
# 🔄 2. AGENT STREAMING INTEGRATION
# =============================================================================

class StreamingAgent:
    """Agent wrapper with streaming capabilities"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.stream_handlers = []
    
    def add_stream_handler(self, handler: Callable[[StreamEvent], None]):
        """Add a stream event handler"""
        self.stream_handlers.append(handler)
    
    def emit_event(self, event: StreamEvent):
        """Emit event to all handlers"""
        for handler in self.stream_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Stream handler error: {e}")
    
    async def stream_run(self, message: str, **kwargs) -> AsyncGenerator[StreamEvent, None]:
        """Run agent with streaming output"""
        try:
            # Emit start event
            yield StreamEvent('stream_start', {'message': message})
            
            # Simulate streaming processing (in real implementation, 
            # this would integrate with the actual Agent streaming)
            steps = [
                "Analyzing input...",
                "Processing request...",
                "Generating response...",
                "Finalizing output..."
            ]
            
            for i, step in enumerate(steps):
                progress = (i + 1) / len(steps)
                yield ProgressStreamEvent(progress, step, step)
                await asyncio.sleep(0.5)  # Simulate processing
            
            # Run the actual agent (non-streaming for now)
            result = Runner.run_sync(self.agent, message, **kwargs)
            
            # Stream the response text
            response_text = result.final_output
            chunk_size = 20
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                is_complete = (i + len(chunk)) >= len(response_text)
                yield TextStreamEvent(chunk, is_complete)
                await asyncio.sleep(0.1)
            
            # Emit completion event
            yield StreamEvent('stream_complete', {
                'total_length': len(response_text),
                'final_result': result.final_output
            })
            
        except Exception as e:
            yield ErrorStreamEvent(str(e), 'processing_error')

def agent_streaming_examples():
    """Examples of agent streaming integration"""
    
    print("\n🔄 AGENT STREAMING INTEGRATION")
    print("=" * 50)
    
    # Create agents for streaming
    @function_tool
    def analyze_data(data: str) -> str:
        """Analyze provided data"""
        time.sleep(1)  # Simulate processing time
        return f"Analysis complete: {data[:50]}... contains {len(data)} characters"
    
    @function_tool
    def generate_report(topic: str) -> str:
        """Generate a detailed report"""
        time.sleep(1.5)  # Simulate longer processing
        return f"Comprehensive report on {topic}: Executive summary shows positive trends with key recommendations for improvement."
    
    streaming_test_agent = Agent(
        name="StreamingTestAgent",
        instructions="""
        You are a data analysis agent that provides detailed responses.
        Take time to process requests thoroughly and provide comprehensive answers.
        """,
        tools=[analyze_data, generate_report]
    )
    
    # Create streaming wrapper
    streaming_agent = StreamingAgent(streaming_test_agent)
    
    # Stream event handlers
    def progress_handler(event: StreamEvent):
        """Handle progress events"""
        if event.event_type == 'progress':
            progress_data = event.data
            print(f"📊 Progress: {progress_data['progress']:.1%} - {progress_data['message']}")
    
    def text_handler(event: StreamEvent):
        """Handle text chunk events"""
        if event.event_type == 'text_chunk':
            text_data = event.data
            print(f"📝 Text: '{text_data['text']}' (Complete: {text_data['is_complete']})")
    
    def completion_handler(event: StreamEvent):
        """Handle completion events"""
        if event.event_type == 'stream_complete':
            print(f"✅ Stream Complete: {event.data['total_length']} characters")
        elif event.event_type == 'stream_start':
            print(f"🚀 Stream Started: {event.data['message']}")
    
    def error_handler(event: StreamEvent):
        """Handle error events"""
        if event.event_type == 'error':
            print(f"❌ Error: {event.data['message']}")
    
    # Add handlers
    streaming_agent.add_stream_handler(progress_handler)
    streaming_agent.add_stream_handler(text_handler)
    streaming_agent.add_stream_handler(completion_handler)
    streaming_agent.add_stream_handler(error_handler)
    
    print("\n1. Testing Agent Streaming:")
    
    # Test streaming queries
    streaming_queries = [
        "Analyze this data: Customer satisfaction scores have improved by 15% over the last quarter",
        "Generate a report on quarterly sales performance trends"
    ]
    
    async def test_agent_streaming():
        for i, query in enumerate(streaming_queries, 1):
            print(f"\n--- Streaming Query {i} ---")
            print(f"Query: {query}")
            print("Stream Events:")
            
            full_response = ""
            async for event in streaming_agent.stream_run(query):
                # Events are handled by registered handlers
                if event.event_type == 'text_chunk':
                    full_response += event.data['text']
            
            print(f"Final Response: {full_response}")
    
    # Run async streaming test
    try:
        asyncio.run(test_agent_streaming())
    except Exception as e:
        print(f"Agent streaming test error: {e}")
    
    return {
        "streaming_agent": streaming_agent,
        "test_agent": streaming_test_agent
    }

# =============================================================================
# 📈 3. PROGRESS TRACKING & MONITORING
# =============================================================================

class ProgressTracker:
    """Advanced progress tracking for streaming operations"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_history = []
        self.is_complete = False
    
    def update_progress(self, step: int = None, message: str = "") -> ProgressStreamEvent:
        """Update progress and return event"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = min(self.current_step / self.total_steps, 1.0)
        
        # Calculate ETA
        elapsed = datetime.now() - self.start_time
        if progress > 0:
            estimated_total = elapsed / progress
            eta = estimated_total - elapsed
        else:
            eta = None
        
        step_data = {
            'step': self.current_step,
            'total': self.total_steps,
            'progress': progress,
            'message': message,
            'elapsed_seconds': elapsed.total_seconds(),
            'eta_seconds': eta.total_seconds() if eta else None
        }
        
        self.step_history.append(step_data)
        
        if progress >= 1.0:
            self.is_complete = True
        
        return ProgressStreamEvent(
            progress=progress,
            message=f"{message} ({progress:.1%})",
            step=f"Step {self.current_step}/{self.total_steps}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        elapsed = datetime.now() - self.start_time
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'progress': self.current_step / self.total_steps,
            'is_complete': self.is_complete,
            'elapsed_seconds': elapsed.total_seconds(),
            'steps_per_second': self.current_step / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        }

class LongRunningTask:
    """Simulate long-running task with progress tracking"""
    
    def __init__(self, name: str, total_items: int = 50):
        self.name = name
        self.total_items = total_items
        self.tracker = ProgressTracker(total_items)
    
    async def execute_with_streaming(self) -> AsyncGenerator[StreamEvent, None]:
        """Execute task with progress streaming"""
        try:
            yield StreamEvent('task_start', {
                'task_name': self.name,
                'total_items': self.total_items
            })
            
            # Simulate processing items
            for i in range(self.total_items):
                # Simulate work
                await asyncio.sleep(0.05)  # 50ms per item
                
                # Update progress
                message = f"Processing item {i + 1}: {self.name}_item_{i + 1}"
                progress_event = self.tracker.update_progress(message=message)
                yield progress_event
                
                # Emit item completion
                yield StreamEvent('item_complete', {
                    'item_id': f"{self.name}_item_{i + 1}",
                    'item_number': i + 1
                })
                
                # Simulate occasional detailed updates
                if (i + 1) % 10 == 0:
                    summary = self.tracker.get_summary()
                    yield StreamEvent('progress_summary', summary)
            
            # Task completion
            final_summary = self.tracker.get_summary()
            yield StreamEvent('task_complete', {
                'task_name': self.name,
                'summary': final_summary
            })
            
        except Exception as e:
            yield ErrorStreamEvent(f"Task {self.name} failed: {str(e)}", 'task_error')

def progress_tracking_examples():
    """Examples of progress tracking and monitoring"""
    
    print("\n📈 PROGRESS TRACKING & MONITORING")
    print("=" * 50)
    
    # Progress display handler
    class ProgressDisplay:
        def __init__(self):
            self.current_progress = 0
            self.current_message = ""
        
        def display_progress_bar(self, progress: float, message: str = "", width: int = 30):
            """Display a text progress bar"""
            filled = int(width * progress)
            bar = "█" * filled + "░" * (width - filled)
            percentage = progress * 100
            
            # Update display
            print(f"\r{bar} {percentage:5.1f}% | {message}", end="", flush=True)
            
            if progress >= 1.0:
                print()  # New line when complete
        
        def handle_event(self, event: StreamEvent):
            """Handle streaming events for display"""
            if event.event_type == 'progress':
                progress = event.data['progress']
                message = event.data.get('message', '')
                self.display_progress_bar(progress, message)
            elif event.event_type == 'task_start':
                print(f"\n🚀 Starting task: {event.data['task_name']}")
                print(f"Total items: {event.data['total_items']}")
            elif event.event_type == 'task_complete':
                summary = event.data['summary']
                print(f"\n✅ Task complete!")
                print(f"Processed {summary['completed_steps']} items in {summary['elapsed_seconds']:.1f} seconds")
                print(f"Rate: {summary['steps_per_second']:.1f} items/second")
            elif event.event_type == 'error':
                print(f"\n❌ Error: {event.data['message']}")
    
    print("\n1. Testing Progress Tracking:")
    
    # Create tasks
    tasks = [
        LongRunningTask("DataProcessing", 25),
        LongRunningTask("ImageResize", 30),
        LongRunningTask("DocumentAnalysis", 20)
    ]
    
    async def run_tasks_with_progress():
        """Run tasks with progress display"""
        for task in tasks:
            print(f"\n--- Running Task: {task.name} ---")
            display = ProgressDisplay()
            
            async for event in task.execute_with_streaming():
                display.handle_event(event)
                
                # Additional event logging
                if event.event_type == 'progress_summary':
                    summary = event.data
                    print(f"\n📊 Progress Summary: {summary['completed_steps']}/{summary['total_steps']} items "
                          f"({summary['steps_per_second']:.1f} items/sec)")
    
    # Run progress tracking test
    try:
        asyncio.run(run_tasks_with_progress())
    except Exception as e:
        print(f"Progress tracking test error: {e}")
    
    print("\n2. Advanced Progress Features:")
    
    # Demonstrate advanced progress tracking
    tracker = ProgressTracker(100)
    print("Advanced Progress Tracker:")
    
    for i in range(0, 101, 20):
        event = tracker.update_progress(i, f"Processing phase {i//20 + 1}")
        summary = tracker.get_summary()
        
        print(f"Step {i}/100: {event.data['progress']:.1%} complete")
        print(f"  Rate: {summary['steps_per_second']:.2f} steps/second")
        if summary['elapsed_seconds'] > 0:
            print(f"  Elapsed: {summary['elapsed_seconds']:.1f}s")
        
        time.sleep(0.1)  # Simulate work
    
    return {
        "progress_tracker": ProgressTracker,
        "long_running_task": LongRunningTask,
        "progress_display": ProgressDisplay
    }

# =============================================================================
# 🌊 4. REAL-TIME DATA STREAMING
# =============================================================================

class DataStreamManager:
    """Manage real-time data streams"""
    
    def __init__(self):
        self.active_streams = {}
        self.stream_subscribers = {}
    
    def create_stream(self, stream_id: str, data_source: Callable = None):
        """Create a new data stream"""
        self.active_streams[stream_id] = {
            'id': stream_id,
            'created_at': datetime.now(),
            'data_source': data_source,
            'is_active': True,
            'message_count': 0
        }
        self.stream_subscribers[stream_id] = []
    
    def subscribe(self, stream_id: str, handler: Callable[[Any], None]):
        """Subscribe to a data stream"""
        if stream_id not in self.stream_subscribers:
            self.stream_subscribers[stream_id] = []
        
        self.stream_subscribers[stream_id].append(handler)
    
    def emit_to_stream(self, stream_id: str, data: Any):
        """Emit data to stream subscribers"""
        if stream_id in self.stream_subscribers:
            self.active_streams[stream_id]['message_count'] += 1
            
            for handler in self.stream_subscribers[stream_id]:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Stream handler error for {stream_id}: {e}")
    
    async def start_data_stream(self, stream_id: str, 
                               data_generator: AsyncGenerator[Any, None]):
        """Start streaming data from generator"""
        try:
            async for data in data_generator:
                if stream_id in self.active_streams and self.active_streams[stream_id]['is_active']:
                    self.emit_to_stream(stream_id, data)
                else:
                    break  # Stream was stopped
        except Exception as e:
            error_event = ErrorStreamEvent(f"Stream {stream_id} error: {str(e)}")
            self.emit_to_stream(stream_id, error_event)
    
    def stop_stream(self, stream_id: str):
        """Stop a data stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['is_active'] = False
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics for all streams"""
        stats = {
            'total_streams': len(self.active_streams),
            'active_streams': sum(1 for s in self.active_streams.values() if s['is_active']),
            'total_subscribers': sum(len(subs) for subs in self.stream_subscribers.values()),
            'streams': []
        }
        
        for stream_id, stream_info in self.active_streams.items():
            subscriber_count = len(self.stream_subscribers.get(stream_id, []))
            stats['streams'].append({
                'id': stream_id,
                'is_active': stream_info['is_active'],
                'message_count': stream_info['message_count'],
                'subscriber_count': subscriber_count,
                'uptime_seconds': (datetime.now() - stream_info['created_at']).total_seconds()
            })
        
        return stats

# Simulated data generators
async def stock_price_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Simulate stock price updates"""
    import random
    
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    prices = {stock: 100 + random.uniform(-50, 200) for stock in stocks}
    
    while True:
        stock = random.choice(stocks)
        change = random.uniform(-5, 5)
        prices[stock] = max(0.01, prices[stock] + change)
        
        yield {
            'timestamp': datetime.now().isoformat(),
            'symbol': stock,
            'price': round(prices[stock], 2),
            'change': round(change, 2),
            'change_percent': round((change / (prices[stock] - change)) * 100, 2)
        }
        
        await asyncio.sleep(0.5)  # Update every 500ms

async def system_metrics_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Simulate system metrics updates"""
    import random
    
    while True:
        yield {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': round(random.uniform(10, 90), 1),
            'memory_usage': round(random.uniform(30, 85), 1),
            'disk_io': round(random.uniform(0, 100), 1),
            'network_io': round(random.uniform(0, 1000), 1)
        }
        
        await asyncio.sleep(1.0)  # Update every second

async def chat_message_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Simulate chat messages"""
    import random
    
    users = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    messages = [
        "Hello everyone!", "How's the project going?", "Great work team!",
        "Can someone review this?", "Meeting in 10 minutes",
        "Thanks for the update", "Looking good!", "Any questions?"
    ]
    
    while True:
        yield {
            'timestamp': datetime.now().isoformat(),
            'user': random.choice(users),
            'message': random.choice(messages),
            'channel': 'general'
        }
        
        await asyncio.sleep(random.uniform(2, 8))  # Variable timing

def real_time_streaming_examples():
    """Examples of real-time data streaming"""
    
    print("\n🌊 REAL-TIME DATA STREAMING")
    print("=" * 50)
    
    # Create stream manager
    stream_manager = DataStreamManager()
    
    # Data handlers
    def stock_handler(data):
        """Handle stock price updates"""
        if isinstance(data, dict) and 'symbol' in data:
            symbol = data['symbol']
            price = data['price']
            change = data['change']
            change_pct = data['change_percent']
            
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{direction} {symbol}: ${price:.2f} ({change:+.2f}, {change_pct:+.1f}%)")
        else:
            print(f"Stock data: {data}")
    
    def metrics_handler(data):
        """Handle system metrics"""
        if isinstance(data, dict) and 'cpu_usage' in data:
            cpu = data['cpu_usage']
            memory = data['memory_usage']
            print(f"💻 System: CPU {cpu}%, Memory {memory}%")
        else:
            print(f"Metrics: {data}")
    
    def chat_handler(data):
        """Handle chat messages"""
        if isinstance(data, dict) and 'user' in data:
            user = data['user']
            message = data['message']
            timestamp = data['timestamp'][:19]  # Remove microseconds
            print(f"💬 [{timestamp}] {user}: {message}")
        else:
            print(f"Chat: {data}")
    
    # Error handler
    def error_handler(data):
        """Handle stream errors"""
        if isinstance(data, ErrorStreamEvent):
            print(f"❌ Stream Error: {data.data['message']}")
    
    print("\n1. Setting up Real-Time Streams:")
    
    # Create streams
    stream_manager.create_stream('stocks', stock_price_stream)
    stream_manager.create_stream('metrics', system_metrics_stream)
    stream_manager.create_stream('chat', chat_message_stream)
    
    # Subscribe handlers
    stream_manager.subscribe('stocks', stock_handler)
    stream_manager.subscribe('stocks', error_handler)
    stream_manager.subscribe('metrics', metrics_handler)
    stream_manager.subscribe('metrics', error_handler)
    stream_manager.subscribe('chat', chat_handler)
    stream_manager.subscribe('chat', error_handler)
    
    print("✅ Streams created and handlers subscribed")
    
    async def run_streaming_demo():
        """Run streaming demonstration"""
        print("\n2. Starting Real-Time Data Streams:")
        print("(Streaming for 10 seconds...)\n")
        
        # Start streams
        stock_task = asyncio.create_task(
            stream_manager.start_data_stream('stocks', stock_price_stream())
        )
        metrics_task = asyncio.create_task(
            stream_manager.start_data_stream('metrics', system_metrics_stream())
        )
        chat_task = asyncio.create_task(
            stream_manager.start_data_stream('chat', chat_message_stream())
        )
        
        # Let streams run for 10 seconds
        await asyncio.sleep(10)
        
        # Stop streams
        stream_manager.stop_stream('stocks')
        stream_manager.stop_stream('metrics')
        stream_manager.stop_stream('chat')
        
        # Cancel tasks
        stock_task.cancel()
        metrics_task.cancel()
        chat_task.cancel()
        
        # Get final stats
        stats = stream_manager.get_stream_stats()
        print(f"\n📊 Final Stream Statistics:")
        print(f"Total Streams: {stats['total_streams']}")
        print(f"Active Streams: {stats['active_streams']}")
        print(f"Total Subscribers: {stats['total_subscribers']}")
        
        for stream_info in stats['streams']:
            print(f"  Stream {stream_info['id']}:")
            print(f"    Messages: {stream_info['message_count']}")
            print(f"    Subscribers: {stream_info['subscriber_count']}")
            print(f"    Uptime: {stream_info['uptime_seconds']:.1f}s")
    
    # Run streaming demo
    try:
        asyncio.run(run_streaming_demo())
    except Exception as e:
        print(f"Streaming demo error: {e}")
    
    return {
        "stream_manager": stream_manager,
        "stock_stream": stock_price_stream,
        "metrics_stream": system_metrics_stream,
        "chat_stream": chat_message_stream
    }

# =============================================================================
# 🎨 5. UI INTEGRATION PATTERNS
# =============================================================================

class StreamingUIAdapter:
    """Adapter for integrating streaming with UI frameworks"""
    
    def __init__(self):
        self.ui_callbacks = {}
        self.state = {}
    
    def register_ui_callback(self, component_id: str, callback: Callable):
        """Register UI update callback"""
        self.ui_callbacks[component_id] = callback
    
    def update_ui_component(self, component_id: str, data: Any):
        """Update specific UI component"""
        if component_id in self.ui_callbacks:
            try:
                self.ui_callbacks[component_id](data)
            except Exception as e:
                print(f"UI callback error for {component_id}: {e}")
    
    def handle_streaming_event(self, event: StreamEvent):
        """Handle streaming events for UI updates"""
        
        if event.event_type == 'text_chunk':
            # Update text display
            self.update_ui_component('text_display', {
                'action': 'append_text',
                'text': event.data['text'],
                'is_complete': event.data['is_complete']
            })
        
        elif event.event_type == 'progress':
            # Update progress bar
            self.update_ui_component('progress_bar', {
                'action': 'set_progress',
                'progress': event.data['progress'],
                'message': event.data['message']
            })
        
        elif event.event_type == 'stream_start':
            # Show loading state
            self.update_ui_component('loading_indicator', {
                'action': 'show',
                'message': 'Processing...'
            })
        
        elif event.event_type == 'stream_complete':
            # Hide loading state
            self.update_ui_component('loading_indicator', {
                'action': 'hide'
            })
            
            # Update completion status
            self.update_ui_component('status_display', {
                'action': 'set_status',
                'status': 'completed',
                'message': 'Processing completed successfully'
            })
        
        elif event.event_type == 'error':
            # Show error state
            self.update_ui_component('error_display', {
                'action': 'show_error',
                'message': event.data['message']
            })

# Mock UI components for demonstration
class MockUIComponent:
    """Mock UI component for demonstration"""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.state = {}
    
    def update(self, data: Dict[str, Any]):
        """Handle UI update"""
        action = data.get('action', 'update')
        
        if action == 'append_text':
            current_text = self.state.get('text', '')
            self.state['text'] = current_text + data['text']
            print(f"[{self.component_id}] Text: '{data['text']}' | Total: '{self.state['text']}'")
        
        elif action == 'set_progress':
            self.state['progress'] = data['progress']
            progress_bar = "█" * int(30 * data['progress']) + "░" * (30 - int(30 * data['progress']))
            print(f"[{self.component_id}] {progress_bar} {data['progress']:.1%} - {data['message']}")
        
        elif action == 'show':
            print(f"[{self.component_id}] 🔄 {data['message']}")
        
        elif action == 'hide':
            print(f"[{self.component_id}] ✅ Loading complete")
        
        elif action == 'set_status':
            print(f"[{self.component_id}] Status: {data['status']} - {data['message']}")
        
        elif action == 'show_error':
            print(f"[{self.component_id}] ❌ Error: {data['message']}")
        
        else:
            print(f"[{self.component_id}] Update: {data}")

def ui_integration_examples():
    """Examples of UI integration patterns"""
    
    print("\n🎨 UI INTEGRATION PATTERNS")
    print("=" * 50)
    
    # Create UI adapter
    ui_adapter = StreamingUIAdapter()
    
    # Create mock UI components
    text_display = MockUIComponent('text_display')
    progress_bar = MockUIComponent('progress_bar')
    loading_indicator = MockUIComponent('loading_indicator')
    status_display = MockUIComponent('status_display')
    error_display = MockUIComponent('error_display')
    
    # Register UI callbacks
    ui_adapter.register_ui_callback('text_display', text_display.update)
    ui_adapter.register_ui_callback('progress_bar', progress_bar.update)
    ui_adapter.register_ui_callback('loading_indicator', loading_indicator.update)
    ui_adapter.register_ui_callback('status_display', status_display.update)
    ui_adapter.register_ui_callback('error_display', error_display.update)
    
    print("\n1. Testing UI Integration:")
    
    # Simulate streaming events
    async def simulate_ui_streaming():
        """Simulate streaming with UI updates"""
        
        # Start event
        start_event = StreamEvent('stream_start', {'message': 'Processing user request'})
        ui_adapter.handle_streaming_event(start_event)
        await asyncio.sleep(0.5)
        
        # Progress events
        for i in range(1, 6):
            progress = i / 5
            progress_event = ProgressStreamEvent(
                progress=progress,
                message=f"Step {i}/5 complete",
                step=f"processing_step_{i}"
            )
            ui_adapter.handle_streaming_event(progress_event)
            await asyncio.sleep(0.5)
        
        # Text streaming events
        response_text = "This is a streaming response that will be displayed incrementally in the UI."
        chunk_size = 10
        
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            is_complete = (i + len(chunk)) >= len(response_text)
            
            text_event = TextStreamEvent(chunk, is_complete)
            ui_adapter.handle_streaming_event(text_event)
            await asyncio.sleep(0.3)
        
        # Completion event
        complete_event = StreamEvent('stream_complete', {
            'total_length': len(response_text),
            'processing_time': 4.0
        })
        ui_adapter.handle_streaming_event(complete_event)
    
    # Run UI simulation
    try:
        asyncio.run(simulate_ui_streaming())
    except Exception as e:
        print(f"UI integration test error: {e}")
    
    print("\n2. UI Integration Best Practices:")
    print("💡 UI Streaming Guidelines:")
    print("  - Buffer small chunks to reduce UI update frequency")
    print("  - Use debouncing for rapid updates")
    print("  - Provide visual feedback for all stream states")
    print("  - Handle errors gracefully with user-friendly messages")
    print("  - Allow users to cancel long-running streams")
    print("  - Implement proper cleanup on component unmount")
    
    return {
        "ui_adapter": ui_adapter,
        "components": {
            "text_display": text_display,
            "progress_bar": progress_bar,
            "loading_indicator": loading_indicator,
            "status_display": status_display,
            "error_display": error_display
        }
    }

# =============================================================================
# 💡 6. STREAMING BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 STREAMING BEST PRACTICES

1. 📡 STREAMING ARCHITECTURE:

   Pattern A - Generator-Based:
   ```python
   def stream_data():
       for item in data_source:
           yield process(item)
   ```
   
   Pattern B - Event-Driven:
   ```python
   class StreamProcessor:
       def on_data(self, data):
           self.emit_event(ProcessEvent(data))
   ```
   
   Pattern C - Async Streaming:
   ```python
   async def async_stream():
       async for item in async_data_source():
           yield await process_async(item)
   ```

2. 🔄 STREAM LIFECYCLE MANAGEMENT:

   ✅ Initialize streams with proper error handling
   ✅ Implement graceful shutdown mechanisms
   ✅ Handle backpressure and flow control
   ✅ Monitor stream health and performance
   ✅ Clean up resources on stream completion

3. 🚦 BACKPRESSURE HANDLING:

   ✅ Implement buffering for burst traffic
   ✅ Use flow control to manage stream rate
   ✅ Drop or sample data when overwhelmed
   ✅ Provide feedback to data producers
   ✅ Monitor queue sizes and latencies

4. 📊 PROGRESS & FEEDBACK:

   ✅ Provide real-time progress indicators
   ✅ Show estimated completion times
   ✅ Allow cancellation of long operations
   ✅ Display meaningful status messages
   ✅ Handle network interruptions gracefully

5. 🎨 UI INTEGRATION:

   ✅ Update UI incrementally, not on every event
   ✅ Use smooth animations for progress indicators
   ✅ Provide visual feedback for all states
   ✅ Handle offline/online state changes
   ✅ Optimize for mobile and desktop experiences

6. 🔒 SECURITY CONSIDERATIONS:

   ✅ Validate all streaming data
   ✅ Implement proper authentication for streams
   ✅ Use secure transport (WSS, HTTPS)
   ✅ Rate limit stream connections
   ✅ Monitor for abuse and anomalies

🚨 COMMON STREAMING PITFALLS:

❌ DON'T stream without proper error handling
❌ DON'T ignore backpressure and memory usage
❌ DON'T update UI too frequently (causes jank)
❌ DON'T forget to clean up stream resources
❌ DON'T stream sensitive data without encryption
❌ DON'T assume network connections are reliable

✅ STREAMING SUCCESS CHECKLIST:

☐ Proper stream lifecycle management
☐ Error handling and recovery mechanisms
☐ Backpressure and flow control
☐ Progress tracking and user feedback
☐ Resource cleanup and memory management
☐ Performance monitoring and optimization
☐ Security validation and authentication
☐ UI responsiveness and user experience
☐ Testing under various network conditions
☐ Documentation and usage examples

📈 PERFORMANCE OPTIMIZATION:

For high-throughput streaming:
- Use connection pooling and keep-alives
- Implement efficient serialization (protobuf, msgpack)
- Use compression for large payloads
- Batch small updates to reduce overhead
- Monitor memory usage and implement bounds
- Use async/await for non-blocking operations
- Consider WebSockets for bidirectional streams
- Implement client-side buffering and batching
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_streaming_examples():
    """Run all streaming examples comprehensively"""
    
    print("📡 OPENAI AGENTS SDK - COMPLETE STREAMING DEMONSTRATION")
    print("=" * 70)
    
    # 1. Basic Streaming Patterns
    basic_results = basic_streaming_examples()
    
    # 2. Agent Streaming Integration  
    agent_results = agent_streaming_examples()
    
    # 3. Progress Tracking & Monitoring
    progress_results = progress_tracking_examples()
    
    # 4. Real-Time Data Streaming
    realtime_results = real_time_streaming_examples()
    
    # 5. UI Integration Patterns
    ui_results = ui_integration_examples()
    
    print("\n✅ All streaming examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Streaming provides real-time feedback and better user experience")
    print("- Use generators and async patterns for efficient streaming")
    print("- Implement proper progress tracking and error handling")
    print("- Consider UI integration patterns for smooth user interfaces")
    print("- Monitor performance and handle backpressure appropriately")
    print("- Always clean up streaming resources properly")

def run_sync_streaming_examples():
    """Run synchronous streaming examples for immediate testing"""
    
    print("📡 OPENAI AGENTS SDK - STREAMING SYNC EXAMPLES")
    print("=" * 60)
    
    # Run synchronous parts of streaming examples
    basic_streaming_examples()
    # Note: Some streaming features require async execution
    print("\n⚠️ Some streaming features require async execution.")
    print("Run with asyncio.run(run_all_streaming_examples()) for full demonstration.")
    
    print("\n✅ Sync streaming examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("📡 OpenAI Agents SDK - Complete Streaming Template")
    print("This template demonstrates all Streaming patterns and real-time features.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_streaming_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_streaming_examples())
    
    print("\n✅ Streaming template demonstration complete!")
    print("💡 Use this template as reference for all your real-time streaming needs.")
