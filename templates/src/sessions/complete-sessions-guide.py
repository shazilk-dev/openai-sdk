"""
💾 OPENAI AGENTS SDK - COMPLETE SESSIONS GUIDE & TEMPLATE

This template covers everything about Sessions in the OpenAI Agents SDK:
- SQLiteSession for conversation memory
- Custom session implementations
- Session lifecycle management
- Multi-user session handling
- Session persistence strategies
- Memory optimization patterns
- Session security considerations
- Advanced session patterns

📚 Based on: https://openai.github.io/openai-agents-python/ref/sessions/
"""

import os
import json
import sqlite3
import asyncio
import hashlib
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod

from agents import Agent, Runner, function_tool, SQLiteSession
from agents.session import BaseSession

# =============================================================================
# 📖 UNDERSTANDING SESSIONS IN OPENAI AGENTS SDK
# =============================================================================

"""
🎯 WHAT ARE SESSIONS?

Sessions provide conversation memory and state management:
1. ✅ CONVERSATION MEMORY: Maintain chat history across interactions
2. ✅ STATE PERSISTENCE: Store agent state between runs
3. ✅ MULTI-USER SUPPORT: Separate conversations per user
4. ✅ CUSTOM STORAGE: Implement your own storage backends
5. ✅ LIFECYCLE MANAGEMENT: Control memory usage and cleanup

KEY CONCEPTS:

1. SESSION TYPES:
   - SQLiteSession: Built-in file-based storage
   - Custom Sessions: Implement BaseSession for other backends
   - In-Memory: Temporary sessions for testing

2. SESSION LIFECYCLE:
   - Creation: Initialize new session
   - Usage: Store and retrieve conversation history
   - Cleanup: Remove old or unused sessions
   - Persistence: Save state across application restarts

3. CONVERSATION HISTORY:
   - Messages: User inputs and agent responses
   - Metadata: Timestamps, user info, context
   - State: Agent internal state and variables

4. MULTI-USER PATTERNS:
   - Session per user: Separate memory for each user
   - Session per conversation: Multiple conversations per user
   - Global sessions: Shared memory across users
"""

# =============================================================================
# 💾 1. SQLITE SESSION FUNDAMENTALS
# =============================================================================

def sqlite_session_examples():
    """Basic SQLiteSession usage patterns"""
    
    print("💾 SQLITE SESSION FUNDAMENTALS")
    print("=" * 50)
    
    # 1.1 Basic SQLiteSession Setup
    
    # Create a simple agent with memory
    @function_tool
    def remember_fact(fact: str) -> str:
        """Remember a fact for later"""
        return f"📝 I'll remember: {fact}"
    
    @function_tool
    def recall_conversation() -> str:
        """Recall what we've discussed"""
        return "🧠 Let me review our conversation history..."
    
    memory_agent = Agent(
        name="MemoryAgent",
        instructions="""
        You are an agent with excellent memory. You remember everything
        we discuss and can recall it later. Use your tools to:
        - Remember important facts users tell you
        - Recall previous conversations when asked
        
        Maintain context across our conversations.
        """,
        tools=[remember_fact, recall_conversation]
    )
    
    # 1.2 Create SQLite session with custom database path
    session_dir = Path("./sessions")
    session_dir.mkdir(exist_ok=True)
    
    basic_session = SQLiteSession(
        db_path=str(session_dir / "basic_conversations.db")
    )
    
    print("\n1. Testing Basic SQLiteSession:")
    
    # Test conversation continuity
    conversation_tests = [
        "Hi, I'm Alice. My favorite color is blue.",
        "I work as a software engineer at TechCorp.",
        "What do you remember about me?",
        "What's my favorite color?",
        "Where do I work?"
    ]
    
    for i, message in enumerate(conversation_tests, 1):
        try:
            result = Runner.run_sync(
                memory_agent, 
                message,
                session=basic_session
            )
            print(f"{i}. User: {message}")
            print(f"   Agent: {result.final_output}")
            print()
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    # 1.3 Multiple Sessions for Different Users
    
    # Create separate sessions for different users
    alice_session = SQLiteSession(
        db_path=str(session_dir / "alice_conversations.db")
    )
    
    bob_session = SQLiteSession(
        db_path=str(session_dir / "bob_conversations.db")
    )
    
    print("2. Testing Multi-User Sessions:")
    
    # Alice's conversation
    alice_messages = [
        "Hi, I'm Alice. I love Python programming.",
        "I'm working on a machine learning project.",
    ]
    
    for i, message in enumerate(alice_messages, 1):
        result = Runner.run_sync(memory_agent, message, session=alice_session)
        print(f"Alice {i}: {message}")
        print(f"   Response: {result.final_output}")
    
    # Bob's conversation  
    bob_messages = [
        "Hello, I'm Bob. I'm a designer.",
        "I specialize in UI/UX design.",
    ]
    
    for i, message in enumerate(bob_messages, 1):
        result = Runner.run_sync(memory_agent, message, session=bob_session)
        print(f"Bob {i}: {message}")
        print(f"   Response: {result.final_output}")
    
    # Test memory separation
    print("\n3. Testing Memory Separation:")
    
    # Ask Alice's session about Bob
    result = Runner.run_sync(
        memory_agent, 
        "What do you know about Bob?",
        session=alice_session
    )
    print("Alice's session asked about Bob:")
    print(f"Response: {result.final_output}")
    
    # Ask Bob's session about Alice
    result = Runner.run_sync(
        memory_agent,
        "What do you know about Alice?", 
        session=bob_session
    )
    print("\nBob's session asked about Alice:")
    print(f"Response: {result.final_output}")
    
    return {
        "basic_session": basic_session,
        "alice_session": alice_session,
        "bob_session": bob_session,
        "memory_agent": memory_agent
    }

# =============================================================================
# 🔧 2. CUSTOM SESSION IMPLEMENTATIONS
# =============================================================================

class MemorySession(BaseSession):
    """In-memory session implementation for testing"""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages = []
        self.created_at = datetime.now()
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the session"""
        return self.messages.copy()
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the session"""
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        self.messages.append(message)
    
    def clear_messages(self) -> None:
        """Clear all messages from the session"""
        self.messages.clear()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session metadata"""
        return {
            'session_id': self.session_id,
            'message_count': len(self.messages),
            'created_at': self.created_at.isoformat(),
            'last_message_at': self.messages[-1]['timestamp'] if self.messages else None
        }

class FileSession(BaseSession):
    """File-based session implementation"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.messages = []
        self._load_messages()
    
    def _load_messages(self) -> None:
        """Load messages from file"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
            except Exception as e:
                print(f"Error loading session from {self.file_path}: {e}")
                self.messages = []
    
    def _save_messages(self) -> None:
        """Save messages to file"""
        try:
            data = {
                'messages': self.messages,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session to {self.file_path}: {e}")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the session"""
        return self.messages.copy()
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the session"""
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        self.messages.append(message)
        self._save_messages()
    
    def clear_messages(self) -> None:
        """Clear all messages from the session"""
        self.messages.clear()
        self._save_messages()

class RedisSession(BaseSession):
    """Redis-based session implementation (simulation)"""
    
    def __init__(self, session_id: str, redis_client=None):
        self.session_id = session_id
        self.redis_client = redis_client or {}  # Simulated Redis client
        self.key_prefix = f"session:{session_id}"
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get messages from Redis"""
        # Simulate Redis operation
        messages_key = f"{self.key_prefix}:messages"
        messages_data = self.redis_client.get(messages_key, '[]')
        
        if isinstance(messages_data, str):
            return json.loads(messages_data)
        return []
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to Redis"""
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        messages = self.get_messages()
        messages.append(message)
        
        # Simulate Redis operation
        messages_key = f"{self.key_prefix}:messages"
        self.redis_client[messages_key] = json.dumps(messages)
        
        # Set expiration (simulate Redis EXPIRE)
        expiry_key = f"{self.key_prefix}:expires"
        self.redis_client[expiry_key] = (
            datetime.now() + timedelta(days=30)
        ).isoformat()
    
    def clear_messages(self) -> None:
        """Clear messages from Redis"""
        messages_key = f"{self.key_prefix}:messages"
        if messages_key in self.redis_client:
            del self.redis_client[messages_key]

def custom_session_examples():
    """Examples of custom session implementations"""
    
    print("\n🔧 CUSTOM SESSION IMPLEMENTATIONS")
    print("=" * 50)
    
    # Create a conversational agent
    @function_tool
    def set_preference(key: str, value: str) -> str:
        """Set a user preference"""
        return f"✅ Set {key} = {value}"
    
    @function_tool
    def get_preferences() -> str:
        """Get all user preferences"""
        return "📋 Retrieving your saved preferences..."
    
    preference_agent = Agent(
        name="PreferenceAgent",
        instructions="""
        You help users manage their preferences and maintain conversation context.
        Remember what users tell you and refer back to previous conversations naturally.
        """,
        tools=[set_preference, get_preferences]
    )
    
    # Test different session implementations
    
    print("\n1. Testing MemorySession (In-Memory):")
    
    memory_session = MemorySession("user123")
    
    memory_tests = [
        "Hi, I'm testing the memory session.",
        "Please set my theme preference to dark mode.",
        "What preferences have I set?",
    ]
    
    for i, message in enumerate(memory_tests, 1):
        result = Runner.run_sync(preference_agent, message, session=memory_session)
        print(f"{i}. {message}")
        print(f"   Response: {result.final_output}")
    
    print(f"Session Info: {memory_session.get_session_info()}")
    
    print("\n2. Testing FileSession (JSON File):")
    
    file_session = FileSession("./sessions/file_session_example.json")
    
    file_tests = [
        "This is a file-based session test.",
        "Set my notification preference to email.",
        "Show me my current preferences.",
    ]
    
    for i, message in enumerate(file_tests, 1):
        result = Runner.run_sync(preference_agent, message, session=file_session)
        print(f"{i}. {message}")
        print(f"   Response: {result.final_output}")
    
    print("\n3. Testing RedisSession (Simulated):")
    
    # Simulated Redis client
    simulated_redis = {}
    redis_session = RedisSession("redis_user456", simulated_redis)
    
    redis_tests = [
        "Testing Redis session implementation.",
        "Set my language preference to Spanish.",
        "What's my current language setting?",
    ]
    
    for i, message in enumerate(redis_tests, 1):
        result = Runner.run_sync(preference_agent, message, session=redis_session)
        print(f"{i}. {message}")
        print(f"   Response: {result.final_output}")
    
    print(f"Simulated Redis Data: {simulated_redis}")
    
    return {
        "memory_session": memory_session,
        "file_session": file_session,
        "redis_session": redis_session,
        "preference_agent": preference_agent
    }

# =============================================================================
# 🏗️ 3. SESSION LIFECYCLE MANAGEMENT
# =============================================================================

class SessionManager:
    """Manages session lifecycle and cleanup"""
    
    def __init__(self, base_path: str = "./sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.active_sessions = {}
        self.session_metadata = {}
    
    def create_session(self, session_id: str, session_type: str = "sqlite") -> BaseSession:
        """Create a new session"""
        
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        if session_type == "sqlite":
            db_path = self.base_path / f"{session_id}.db"
            session = SQLiteSession(str(db_path))
        elif session_type == "file":
            file_path = self.base_path / f"{session_id}.json"
            session = FileSession(str(file_path))
        elif session_type == "memory":
            session = MemorySession(session_id)
        else:
            raise ValueError(f"Unknown session type: {session_type}")
        
        self.active_sessions[session_id] = session
        self.session_metadata[session_id] = {
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'type': session_type,
            'access_count': 0
        }
        
        return session
    
    def get_session(self, session_id: str) -> Optional[BaseSession]:
        """Get an existing session"""
        session = self.active_sessions.get(session_id)
        
        if session and session_id in self.session_metadata:
            self.session_metadata[session_id]['last_accessed'] = datetime.now()
            self.session_metadata[session_id]['access_count'] += 1
        
        return session
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions with metadata"""
        return self.session_metadata.copy()
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> List[str]:
        """Clean up sessions older than max_age_days"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_sessions = []
        
        for session_id, metadata in list(self.session_metadata.items()):
            if metadata['last_accessed'] < cutoff_date:
                self._remove_session(session_id)
                cleaned_sessions.append(session_id)
        
        return cleaned_sessions
    
    def cleanup_inactive_sessions(self, min_access_count: int = 1) -> List[str]:
        """Clean up sessions with low activity"""
        cleaned_sessions = []
        
        for session_id, metadata in list(self.session_metadata.items()):
            if metadata['access_count'] < min_access_count:
                self._remove_session(session_id)
                cleaned_sessions.append(session_id)
        
        return cleaned_sessions
    
    def _remove_session(self, session_id: str) -> None:
        """Remove a session and its files"""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove metadata
        if session_id in self.session_metadata:
            session_type = self.session_metadata[session_id]['type']
            del self.session_metadata[session_id]
            
            # Remove files
            if session_type == "sqlite":
                db_path = self.base_path / f"{session_id}.db"
                if db_path.exists():
                    db_path.unlink()
            elif session_type == "file":
                file_path = self.base_path / f"{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions"""
        if not self.session_metadata:
            return {
                'total_sessions': 0,
                'active_sessions': 0,
                'session_types': {},
                'avg_access_count': 0
            }
        
        total_sessions = len(self.session_metadata)
        active_sessions = len(self.active_sessions)
        
        session_types = {}
        total_access = 0
        
        for metadata in self.session_metadata.values():
            session_type = metadata['type']
            session_types[session_type] = session_types.get(session_type, 0) + 1
            total_access += metadata['access_count']
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'session_types': session_types,
            'avg_access_count': total_access / total_sessions if total_sessions > 0 else 0
        }

def session_lifecycle_examples():
    """Examples of session lifecycle management"""
    
    print("\n🏗️ SESSION LIFECYCLE MANAGEMENT")
    print("=" * 50)
    
    # Create session manager
    session_manager = SessionManager("./managed_sessions")
    
    # Create a simple agent for testing
    @function_tool
    def track_activity(activity: str) -> str:
        """Track user activity"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"📊 [{timestamp}] Tracked: {activity}"
    
    activity_agent = Agent(
        name="ActivityAgent",
        instructions="""
        You track user activities and maintain conversation history.
        Remember all activities users mention and provide summaries when asked.
        """,
        tools=[track_activity]
    )
    
    print("\n1. Creating and Managing Sessions:")
    
    # Create different types of sessions
    sessions_info = []
    
    session_types = [
        ("user_alice_work", "sqlite"),
        ("user_bob_personal", "file"), 
        ("temp_session_123", "memory"),
        ("user_charlie_mobile", "sqlite")
    ]
    
    for session_id, session_type in session_types:
        session = session_manager.create_session(session_id, session_type)
        sessions_info.append((session_id, session_type, session))
        print(f"✅ Created {session_type} session: {session_id}")
    
    print("\n2. Using Sessions:")
    
    # Use sessions with different activities
    test_activities = [
        ("user_alice_work", "Started working on project documentation"),
        ("user_bob_personal", "Planning weekend trip to mountains"),
        ("temp_session_123", "Testing temporary session functionality"),
        ("user_alice_work", "Completed first draft of requirements"),
        ("user_charlie_mobile", "Checking emails on mobile device"),
    ]
    
    for session_id, activity in test_activities:
        session = session_manager.get_session(session_id)
        if session:
            result = Runner.run_sync(
                activity_agent,
                f"Track this activity: {activity}",
                session=session
            )
            print(f"[{session_id}] {activity}")
            print(f"   Response: {result.final_output}")
    
    print("\n3. Session Statistics:")
    stats = session_manager.get_session_stats()
    print(f"Total Sessions: {stats['total_sessions']}")
    print(f"Active Sessions: {stats['active_sessions']}")
    print(f"Session Types: {stats['session_types']}")
    print(f"Average Access Count: {stats['avg_access_count']:.1f}")
    
    print("\n4. Session Metadata:")
    for session_id, metadata in session_manager.list_sessions().items():
        print(f"{session_id}:")
        print(f"  Created: {metadata['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Last Access: {metadata['last_accessed'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Access Count: {metadata['access_count']}")
        print(f"  Type: {metadata['type']}")
        print()
    
    print("5. Session Cleanup:")
    
    # Test cleanup of inactive sessions
    inactive = session_manager.cleanup_inactive_sessions(min_access_count=2)
    print(f"Cleaned up inactive sessions: {inactive}")
    
    # Test cleanup of old sessions (simulate old sessions)
    # In real usage, this would clean sessions older than specified days
    print("Session cleanup completed")
    
    return {
        "session_manager": session_manager,
        "activity_agent": activity_agent,
        "stats": stats
    }

# =============================================================================
# 👥 4. MULTI-USER SESSION PATTERNS
# =============================================================================

class MultiUserSessionManager:
    """Advanced session manager for multi-user applications"""
    
    def __init__(self, base_path: str = "./multi_user_sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.user_sessions = {}  # user_id -> {conversation_id: session}
        self.user_metadata = {}  # user_id -> metadata
    
    def get_user_session(self, user_id: str, conversation_id: str = "default") -> BaseSession:
        """Get or create a session for a specific user and conversation"""
        
        # Initialize user if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
            self.user_metadata[user_id] = {
                'created_at': datetime.now(),
                'conversation_count': 0,
                'last_active': datetime.now()
            }
        
        # Get or create conversation session
        if conversation_id not in self.user_sessions[user_id]:
            session_path = self.base_path / user_id / f"{conversation_id}.db"
            session_path.parent.mkdir(exist_ok=True)
            
            session = SQLiteSession(str(session_path))
            self.user_sessions[user_id][conversation_id] = session
            self.user_metadata[user_id]['conversation_count'] += 1
        
        # Update last active time
        self.user_metadata[user_id]['last_active'] = datetime.now()
        
        return self.user_sessions[user_id][conversation_id]
    
    def get_user_conversations(self, user_id: str) -> List[str]:
        """Get all conversation IDs for a user"""
        if user_id in self.user_sessions:
            return list(self.user_sessions[user_id].keys())
        return []
    
    def create_conversation(self, user_id: str, conversation_name: str = None) -> str:
        """Create a new conversation for a user"""
        if not conversation_name:
            conversation_name = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create the session
        session = self.get_user_session(user_id, conversation_name)
        return conversation_name
    
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary for a specific user"""
        if user_id not in self.user_metadata:
            return None
        
        metadata = self.user_metadata[user_id]
        conversations = self.get_user_conversations(user_id)
        
        return {
            'user_id': user_id,
            'created_at': metadata['created_at'].isoformat(),
            'last_active': metadata['last_active'].isoformat(),
            'conversation_count': len(conversations),
            'conversations': conversations
        }
    
    def get_all_users_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all users"""
        return {
            user_id: self.get_user_summary(user_id)
            for user_id in self.user_metadata.keys()
        }

def multi_user_session_examples():
    """Examples of multi-user session patterns"""
    
    print("\n👥 MULTI-USER SESSION PATTERNS") 
    print("=" * 50)
    
    # Create multi-user session manager
    multi_session_manager = MultiUserSessionManager()
    
    # Create agents for different contexts
    @function_tool
    def save_note(note: str, category: str = "general") -> str:
        """Save a personal note"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"📝 Saved note ({category}) at {timestamp}: {note}"
    
    @function_tool
    def list_my_notes(category: str = None) -> str:
        """List personal notes"""
        if category:
            return f"📋 Showing notes in category: {category}"
        return "📋 Showing all your notes..."
    
    personal_assistant = Agent(
        name="PersonalAssistant",
        instructions="""
        You are a personal assistant that helps users manage their notes,
        tasks, and information. You maintain separate conversations for 
        different topics and remember context within each conversation.
        """,
        tools=[save_note, list_my_notes]
    )
    
    print("\n1. Multiple Users with Separate Sessions:")
    
    # Simulate different users
    users = [
        {
            'id': 'alice@company.com',
            'conversations': [
                ('work_notes', "Save a note: Weekly team meeting scheduled for Friday"),
                ('personal', "Save a note about weekend plans: Visit art museum"),
                ('work_notes', "List my work notes please")
            ]
        },
        {
            'id': 'bob@company.com', 
            'conversations': [
                ('project_alpha', "Save a note: Code review completed for authentication module"),
                ('personal', "Save a note: Buy groceries on Thursday"),
                ('project_alpha', "Show me my project alpha notes")
            ]
        },
        {
            'id': 'charlie@company.com',
            'conversations': [
                ('default', "Save a note: Research new development frameworks"),
                ('default', "List all my notes")
            ]
        }
    ]
    
    for user in users:
        user_id = user['id']
        print(f"\n--- User: {user_id} ---")
        
        for conversation_id, message in user['conversations']:
            session = multi_session_manager.get_user_session(user_id, conversation_id)
            
            try:
                result = Runner.run_sync(personal_assistant, message, session=session)
                print(f"[{conversation_id}] {message}")
                print(f"   Response: {result.final_output}")
            except Exception as e:
                print(f"Error: {e}")
    
    print("\n2. User Session Summary:")
    
    all_users = multi_session_manager.get_all_users_summary()
    for user_id, summary in all_users.items():
        print(f"\nUser: {user_id}")
        print(f"  Created: {summary['created_at']}")
        print(f"  Last Active: {summary['last_active']}")
        print(f"  Conversations: {summary['conversations']}")
        print(f"  Total Conversations: {summary['conversation_count']}")
    
    print("\n3. Cross-Session Isolation Test:")
    
    # Test that users can't see each other's data
    alice_session = multi_session_manager.get_user_session('alice@company.com', 'work_notes')
    bob_session = multi_session_manager.get_user_session('bob@company.com', 'project_alpha')
    
    # Alice asks about Bob's notes (should not have access)
    result_alice = Runner.run_sync(
        personal_assistant,
        "What notes has Bob saved?",
        session=alice_session
    )
    print("Alice asking about Bob's notes:")
    print(f"Response: {result_alice.final_output}")
    
    # Bob asks about Alice's notes (should not have access)
    result_bob = Runner.run_sync(
        personal_assistant,
        "Show me Alice's work notes",
        session=bob_session
    )
    print("\nBob asking about Alice's notes:")
    print(f"Response: {result_bob.final_output}")
    
    return {
        "multi_session_manager": multi_session_manager,
        "personal_assistant": personal_assistant,
        "all_users_summary": all_users
    }

# =============================================================================
# 🔒 5. SESSION SECURITY & PRIVACY
# =============================================================================

class SecureSessionManager:
    """Session manager with security and privacy features"""
    
    def __init__(self, base_path: str = "./secure_sessions", encryption_key: str = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.encryption_key = encryption_key or self._generate_key()
        self.sessions = {}
        self.access_logs = []
    
    def _generate_key(self) -> str:
        """Generate a simple encryption key (in production, use proper key management)"""
        return hashlib.sha256(f"session_key_{datetime.now()}".encode()).hexdigest()[:32]
    
    def _hash_session_id(self, session_id: str) -> str:
        """Hash session ID for privacy"""
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]
    
    def _log_access(self, session_id: str, action: str, user_info: Dict = None):
        """Log session access for audit"""
        self.access_logs.append({
            'timestamp': datetime.now().isoformat(),
            'session_id_hash': self._hash_session_id(session_id),
            'action': action,
            'user_info': user_info or {}
        })
    
    def create_secure_session(self, session_id: str, user_info: Dict = None) -> BaseSession:
        """Create a secure session with logging"""
        
        # Hash the session ID for privacy
        hashed_id = self._hash_session_id(session_id)
        session_path = self.base_path / f"{hashed_id}.db"
        
        session = SQLiteSession(str(session_path))
        self.sessions[session_id] = session
        
        # Log session creation
        self._log_access(session_id, "create", user_info)
        
        return session
    
    def get_secure_session(self, session_id: str, user_info: Dict = None) -> Optional[BaseSession]:
        """Get secure session with access logging"""
        
        session = self.sessions.get(session_id)
        if session:
            self._log_access(session_id, "access", user_info)
        
        return session
    
    def expire_session(self, session_id: str, user_info: Dict = None) -> bool:
        """Expire and remove a session"""
        
        if session_id in self.sessions:
            # Clear session data
            session = self.sessions[session_id]
            if hasattr(session, 'clear_messages'):
                session.clear_messages()
            
            # Remove from active sessions
            del self.sessions[session_id]
            
            # Remove session file
            hashed_id = self._hash_session_id(session_id)
            session_path = self.base_path / f"{hashed_id}.db"
            if session_path.exists():
                session_path.unlink()
            
            # Log expiration
            self._log_access(session_id, "expire", user_info)
            
            return True
        
        return False
    
    def get_access_logs(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Get access logs for auditing"""
        
        if session_id:
            session_hash = self._hash_session_id(session_id)
            return [
                log for log in self.access_logs
                if log['session_id_hash'] == session_hash
            ]
        
        return self.access_logs.copy()
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> List[str]:
        """Clean up sessions older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id in list(self.sessions.keys()):
            # Check session age from logs
            session_logs = self.get_access_logs(session_id)
            if session_logs:
                last_access = datetime.fromisoformat(session_logs[-1]['timestamp'])
                if last_access < cutoff_time:
                    self.expire_session(session_id, {'reason': 'expired'})
                    expired_sessions.append(session_id)
        
        return expired_sessions

def session_security_examples():
    """Examples of session security and privacy features"""
    
    print("\n🔒 SESSION SECURITY & PRIVACY")
    print("=" * 50)
    
    # Create secure session manager
    secure_manager = SecureSessionManager()
    
    # Create a financial agent that needs security
    @function_tool
    def record_transaction(amount: float, description: str) -> str:
        """Record a financial transaction"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"💰 Transaction recorded: ${amount:.2f} - {description} at {timestamp}"
    
    @function_tool
    def get_account_balance() -> str:
        """Get account balance (sensitive operation)"""
        # In real app, this would retrieve actual balance
        return "🏦 Current account balance: $2,547.83"
    
    financial_agent = Agent(
        name="FinancialAgent",
        instructions="""
        You are a secure financial assistant that helps users manage their money.
        You maintain strict confidentiality and security for all financial data.
        Always be cautious with sensitive financial information.
        """,
        tools=[record_transaction, get_account_balance]
    )
    
    print("\n1. Creating Secure Sessions:")
    
    # Create secure sessions for different users
    user_sessions = {}
    
    users = [
        {'id': 'user_12345', 'info': {'name': 'Alice', 'role': 'customer', 'ip': '192.168.1.100'}},
        {'id': 'user_67890', 'info': {'name': 'Bob', 'role': 'premium', 'ip': '192.168.1.101'}},
    ]
    
    for user in users:
        session = secure_manager.create_secure_session(user['id'], user['info'])
        user_sessions[user['id']] = session
        print(f"✅ Created secure session for {user['info']['name']} ({user['id']})")
    
    print("\n2. Using Secure Sessions:")
    
    # Test secure financial operations
    financial_operations = [
        ('user_12345', "Record a transaction: $50 for groceries"),
        ('user_67890', "Record a transaction: $1200 for rent payment"),
        ('user_12345', "What's my current account balance?"),
        ('user_67890', "Show me my account balance"),
    ]
    
    for user_id, operation in financial_operations:
        session = secure_manager.get_secure_session(user_id, {'action': 'financial_query'})
        if session:
            result = Runner.run_sync(financial_agent, operation, session=session)
            print(f"[{user_id}] {operation}")
            print(f"   Response: {result.final_output}")
    
    print("\n3. Security Audit Logs:")
    
    all_logs = secure_manager.get_access_logs()
    print("All Access Logs:")
    for log in all_logs:
        print(f"  {log['timestamp']}: {log['action']} - Session: {log['session_id_hash']}")
        if log['user_info']:
            print(f"    User Info: {log['user_info']}")
    
    print("\n4. User-Specific Access Logs:")
    
    for user in users:
        user_logs = secure_manager.get_access_logs(user['id'])
        print(f"\nLogs for {user['info']['name']} ({user['id']}):")
        for log in user_logs:
            print(f"  {log['timestamp']}: {log['action']}")
    
    print("\n5. Session Security Features:")
    
    # Test session expiration
    print("Testing session expiration...")
    expired = secure_manager.cleanup_expired_sessions(max_age_hours=0.001)  # Very short for testing
    if expired:
        print(f"✅ Expired sessions: {expired}")
    else:
        print("No sessions expired (they're too new)")
    
    # Test session data isolation
    print("\nTesting session data isolation:")
    alice_session = secure_manager.get_secure_session('user_12345')
    if alice_session:
        result = Runner.run_sync(
            financial_agent,
            "What transactions has Bob made?",
            session=alice_session
        )
        print("Alice asking about Bob's transactions:")
        print(f"Response: {result.final_output}")
    
    return {
        "secure_manager": secure_manager,
        "financial_agent": financial_agent,
        "access_logs": all_logs
    }

# =============================================================================
# 📈 6. SESSION OPTIMIZATION & PERFORMANCE
# =============================================================================

class OptimizedSessionManager:
    """Optimized session manager for high-performance applications"""
    
    def __init__(self, base_path: str = "./optimized_sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.session_cache = {}  # In-memory cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.max_cache_size = 100
        self.message_limit = 1000  # Limit messages per session
    
    def _get_cache_key(self, session_id: str) -> str:
        """Generate cache key for session"""
        return f"session_{session_id}"
    
    def _evict_from_cache(self) -> None:
        """Evict least recently used session from cache"""
        if len(self.session_cache) >= self.max_cache_size:
            # Simple LRU: remove oldest entry (in practice, use proper LRU)
            oldest_key = next(iter(self.session_cache))
            del self.session_cache[oldest_key]
            self.cache_stats['evictions'] += 1
    
    def get_optimized_session(self, session_id: str) -> BaseSession:
        """Get session with caching and optimization"""
        cache_key = self._get_cache_key(session_id)
        
        # Check cache first
        if cache_key in self.session_cache:
            self.cache_stats['hits'] += 1
            return self.session_cache[cache_key]
        
        # Cache miss - load from disk
        self.cache_stats['misses'] += 1
        
        # Evict if necessary
        self._evict_from_cache()
        
        # Create/load session
        session_path = self.base_path / f"{session_id}.db"
        session = SQLiteSession(str(session_path))
        
        # Add to cache
        self.session_cache[cache_key] = session
        
        return session
    
    def optimize_session_messages(self, session: BaseSession) -> int:
        """Optimize session by removing old messages"""
        if hasattr(session, 'get_messages'):
            messages = session.get_messages()
            
            if len(messages) > self.message_limit:
                # Keep only recent messages
                messages_to_keep = messages[-self.message_limit:]
                
                # Clear and re-add messages
                session.clear_messages()
                for message in messages_to_keep:
                    session.add_message(message)
                
                removed_count = len(messages) - len(messages_to_keep)
                return removed_count
        
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'cache_evictions': self.cache_stats['evictions'],
            'hit_rate': hit_rate,
            'cache_size': len(self.session_cache),
            'max_cache_size': self.max_cache_size
        }
    
    def batch_optimize_sessions(self) -> Dict[str, int]:
        """Optimize all cached sessions"""
        optimization_results = {}
        
        for cache_key, session in self.session_cache.items():
            session_id = cache_key.replace('session_', '')
            removed_count = self.optimize_session_messages(session)
            optimization_results[session_id] = removed_count
        
        return optimization_results

def session_optimization_examples():
    """Examples of session optimization and performance"""
    
    print("\n📈 SESSION OPTIMIZATION & PERFORMANCE")
    print("=" * 50)
    
    # Create optimized session manager
    optimized_manager = OptimizedSessionManager()
    optimized_manager.max_cache_size = 5  # Small cache for testing
    optimized_manager.message_limit = 10  # Small limit for testing
    
    # Create a chatty agent that generates many messages
    @function_tool
    def generate_content(topic: str, length: str = "medium") -> str:
        """Generate content on a topic"""
        lengths = {
            "short": "Brief overview",
            "medium": "Detailed explanation with examples",
            "long": "Comprehensive analysis with multiple perspectives and detailed examples"
        }
        content_type = lengths.get(length, "medium")
        return f"📝 Generated {content_type} content about {topic}"
    
    content_agent = Agent(
        name="ContentAgent",
        instructions="""
        You generate content on various topics. You remember what content
        you've created for users and can build upon previous conversations.
        """,
        tools=[generate_content]
    )
    
    print("\n1. Testing Session Caching:")
    
    # Create multiple sessions and test caching
    session_ids = [f"session_{i}" for i in range(8)]  # More than cache size
    
    # First access (cache misses)
    print("First access to sessions (cache misses):")
    for session_id in session_ids:
        session = optimized_manager.get_optimized_session(session_id)
        Runner.run_sync(
            content_agent,
            f"Generate content about topic {session_id[-1]}",
            session=session
        )
    
    stats = optimized_manager.get_cache_stats()
    print(f"Cache Stats: Hits={stats['cache_hits']}, Misses={stats['cache_misses']}")
    print(f"Cache Size: {stats['cache_size']}/{stats['max_cache_size']}")
    print(f"Hit Rate: {stats['hit_rate']:.2%}")
    
    # Second access to some sessions (cache hits)
    print("\nSecond access to some sessions (cache hits expected):")
    for session_id in session_ids[-3:]:  # Access last 3 sessions
        session = optimized_manager.get_optimized_session(session_id)
        Runner.run_sync(
            content_agent,
            f"Generate more content about topic {session_id[-1]}",
            session=session
        )
    
    stats = optimized_manager.get_cache_stats()
    print(f"Updated Cache Stats: Hits={stats['cache_hits']}, Misses={stats['cache_misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.2%}")
    print(f"Evictions: {stats['cache_evictions']}")
    
    print("\n2. Testing Message Optimization:")
    
    # Create a session with many messages
    busy_session = optimized_manager.get_optimized_session("busy_session")
    
    # Add many messages
    print("Adding many messages to test optimization...")
    for i in range(15):  # More than message_limit
        Runner.run_sync(
            content_agent,
            f"Generate content batch {i+1}",
            session=busy_session
        )
    
    # Check message count before optimization
    if hasattr(busy_session, 'get_messages'):
        message_count_before = len(busy_session.get_messages())
        print(f"Messages before optimization: {message_count_before}")
    
    # Optimize session
    removed = optimized_manager.optimize_session_messages(busy_session)
    print(f"Messages removed during optimization: {removed}")
    
    # Check message count after optimization
    if hasattr(busy_session, 'get_messages'):
        message_count_after = len(busy_session.get_messages())
        print(f"Messages after optimization: {message_count_after}")
    
    print("\n3. Batch Optimization:")
    
    # Run batch optimization on all sessions
    optimization_results = optimized_manager.batch_optimize_sessions()
    print("Batch optimization results:")
    for session_id, removed_count in optimization_results.items():
        if removed_count > 0:
            print(f"  {session_id}: {removed_count} messages removed")
        else:
            print(f"  {session_id}: no optimization needed")
    
    print("\n4. Performance Recommendations:")
    print("💡 Session Performance Tips:")
    print("  - Use in-memory caching for frequently accessed sessions")
    print("  - Implement message limits to prevent unbounded growth")
    print("  - Regular cleanup of old/inactive sessions")
    print("  - Consider compression for long-term storage")
    print("  - Monitor cache hit rates and adjust cache size")
    print("  - Use async operations for I/O-heavy session operations")
    
    return {
        "optimized_manager": optimized_manager,
        "cache_stats": stats,
        "content_agent": content_agent
    }

# =============================================================================
# 💡 7. SESSION BEST PRACTICES & PATTERNS
# =============================================================================

"""
💡 SESSION BEST PRACTICES

1. 🏗️ SESSION DESIGN PATTERNS:

   Pattern A - Single Session Per User:
   ✅ Simple to implement
   ✅ Good for basic chat applications
   ❌ Can become cluttered with mixed topics
   
   Pattern B - Multiple Sessions Per User:
   ✅ Topic-based conversation separation
   ✅ Better organization and context
   ❌ More complex to manage
   
   Pattern C - Hierarchical Sessions:
   ✅ Organized by user → project → topic
   ✅ Great for enterprise applications
   ❌ Complex navigation and management

2. 💾 STORAGE STRATEGIES:

   SQLiteSession:
   ✅ File-based, good for development
   ✅ ACID compliance, reliable
   ❌ Not suitable for high concurrency
   
   Database Sessions:
   ✅ Scalable, concurrent access
   ✅ Backup and replication support
   ❌ More complex setup and maintenance
   
   In-Memory Sessions:
   ✅ Fastest performance
   ✅ Good for testing
   ❌ Data loss on restart

3. 🔄 SESSION LIFECYCLE:

   Creation:
   ✅ Create on first user interaction
   ✅ Initialize with user context
   ✅ Set appropriate permissions
   
   Management:
   ✅ Regular cleanup of old sessions
   ✅ Monitor session size and performance
   ✅ Implement session expiration policies
   
   Cleanup:
   ✅ Remove inactive sessions
   ✅ Archive important conversations
   ✅ Secure deletion of sensitive data

4. 🚀 PERFORMANCE OPTIMIZATION:

   ✅ Cache frequently accessed sessions
   ✅ Limit message history size
   ✅ Use compression for long-term storage
   ✅ Implement lazy loading for large sessions
   ✅ Batch operations where possible
   ✅ Monitor and optimize database queries

5. 🔒 SECURITY CONSIDERATIONS:

   ✅ Encrypt sensitive session data
   ✅ Implement proper access controls
   ✅ Log session access for auditing
   ✅ Use secure session identifiers
   ✅ Regular security reviews
   ✅ Secure session cleanup procedures

6. 👥 MULTI-USER PATTERNS:

   ✅ Isolate user data completely
   ✅ Implement proper user authentication
   ✅ Use hashed session identifiers
   ✅ Separate storage per user/tenant
   ✅ Monitor cross-user access attempts

🚨 COMMON SESSION PITFALLS:

❌ DON'T store sessions in global variables
❌ DON'T ignore session size limits
❌ DON'T expose session data across users
❌ DON'T forget to clean up old sessions
❌ DON'T store sensitive data unencrypted
❌ DON'T assume sessions persist forever

✅ SESSION SUCCESS CHECKLIST:

☐ Clear session management strategy
☐ Appropriate storage backend chosen
☐ Session lifecycle properly managed
☐ Security and privacy measures implemented
☐ Performance monitoring in place
☐ Regular cleanup procedures
☐ Multi-user isolation verified
☐ Error handling for session failures
☐ Documentation and operational procedures

📊 MONITORING & METRICS:

Key metrics to track:
- Session creation/deletion rates
- Average session size and message count
- Session access patterns and frequency
- Cache hit/miss rates (if using caching)
- Storage usage and growth trends
- Session-related error rates
- User engagement per session
- Session cleanup effectiveness

🛠️ TROUBLESHOOTING GUIDE:

Common issues and solutions:
1. Session not found → Check session creation and ID handling
2. Slow session access → Implement caching or optimize queries
3. Memory leaks → Ensure proper session cleanup
4. Cross-user data leaks → Verify session isolation
5. Session corruption → Implement validation and recovery
6. Storage exhaustion → Implement cleanup policies
"""

# =============================================================================
# 🏃‍♂️ MAIN EXECUTION & DEMONSTRATIONS
# =============================================================================

async def run_all_session_examples():
    """Run all session examples comprehensively"""
    
    print("💾 OPENAI AGENTS SDK - COMPLETE SESSIONS DEMONSTRATION")
    print("=" * 70)
    
    # 1. SQLite Session Fundamentals
    sqlite_results = sqlite_session_examples()
    
    # 2. Custom Session Implementations
    custom_results = custom_session_examples()
    
    # 3. Session Lifecycle Management
    lifecycle_results = session_lifecycle_examples()
    
    # 4. Multi-User Session Patterns
    multi_user_results = multi_user_session_examples()
    
    # 5. Session Security & Privacy
    security_results = session_security_examples()
    
    # 6. Session Optimization & Performance
    optimization_results = session_optimization_examples()
    
    print("\n✅ All session examples completed!")
    print("\n📚 Key Takeaways:")
    print("- Sessions provide conversation memory and state management")
    print("- SQLiteSession is great for development, custom sessions for production")
    print("- Always implement proper session lifecycle management")
    print("- Security and privacy are crucial for multi-user applications")
    print("- Performance optimization becomes important at scale")
    print("- Choose session patterns based on your application's needs")

def run_sync_session_examples():
    """Run synchronous session examples for immediate testing"""
    
    print("💾 OPENAI AGENTS SDK - SESSIONS SYNC EXAMPLES")
    print("=" * 60)
    
    # Run all session examples synchronously
    sqlite_session_examples()
    custom_session_examples()
    session_lifecycle_examples()
    multi_user_session_examples()
    session_security_examples()
    session_optimization_examples()
    
    print("\n✅ Sync session examples completed!")

# =============================================================================
# 🏁 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("💾 OpenAI Agents SDK - Complete Sessions Template")
    print("This template demonstrates all Session patterns and management strategies.")
    print("\nChoose execution mode:")
    print("1. Sync examples only (immediate)")
    print("2. Full async demonstration (comprehensive)")
    
    # For immediate testing, run sync examples
    run_sync_session_examples()
    
    # Uncomment to run full async demonstration:
    # asyncio.run(run_all_session_examples())
    
    print("\n✅ Sessions template demonstration complete!")
    print("💡 Use this template as reference for all your session management needs.")
