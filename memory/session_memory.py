# memory/session_memory.py
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
import logging
from typing import List, Dict, Any, Optional
import datetime
import json

logger = logging.getLogger('me_agent_orchestrator')

class SessionMemory:
    """Enhanced memory management for ME.ai sessions with persistence options"""
    
    def __init__(self, session_id, memory_type="buffer", window_size=10, persistence=None):
        """
        Initialize session memory
        
        Args:
            session_id: Unique identifier for the session
            memory_type: Type of memory ("buffer" for all history, "window" for limited)
            window_size: Number of turns to remember if using window memory
            persistence: Persistence method (None, "redis")
        """
        self.session_id = session_id
        self.memory_type = memory_type
        self.window_size = window_size
        self.persistence = persistence
        self.session_data = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "user_info": {},
            "device_info": [],
            "issue_data": {},
            "metadata": {}
        }
        
        # Initialize memory system based on type and persistence
        self.memory = self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize the appropriate memory system"""
        try:
            # Initialize chat history based on persistence method
            if self.persistence == "redis":
                try:
                    # This requires Redis to be configured and available
                    chat_history = RedisChatMessageHistory(
                        session_id=self.session_id,
                        url="redis://localhost:6379/0"  # Should be configurable
                    )
                    logger.info(f"Initialized Redis-backed chat history for session {self.session_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize Redis chat history: {str(e)}")
                    logger.info("Falling back to in-memory chat history")
                    chat_history = None
            else:
                chat_history = None
            
            # Create the appropriate memory type
            if self.memory_type == "window":
                if chat_history:
                    memory = ConversationBufferWindowMemory(
                        chat_memory=chat_history,
                        memory_key="chat_history",
                        return_messages=True,
                        k=self.window_size
                    )
                else:
                    memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        k=self.window_size
                    )
                logger.info(f"Initialized window memory with size {self.window_size}")
            else:
                if chat_history:
                    memory = ConversationBufferMemory(
                        chat_memory=chat_history,
                        memory_key="chat_history",
                        return_messages=True
                    )
                else:
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                logger.info("Initialized buffer memory with full history")
            
            return memory
        except Exception as e:
            logger.error(f"Error initializing memory: {str(e)}", exc_info=True)
            # Fallback to basic memory
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
    def save_context(self, inputs, outputs):
        """Save context to memory"""
        try:
            self.memory.save_context(inputs, outputs)
            self.session_data["last_updated"] = datetime.datetime.now().isoformat()
            logger.info(f"Saved context to memory for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving context to memory: {str(e)}")
    
    def load_memory_variables(self):
        """Load memory variables"""
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(f"Error loading memory variables: {str(e)}")
            return {"chat_history": []}
    
    def clear(self):
        """Clear memory"""
        try:
            self.memory.clear()
            logger.info(f"Cleared memory for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def add_user_message(self, message):
        """Add a user message to memory"""
        try:
            self.memory.chat_memory.add_user_message(message)
            self.session_data["last_updated"] = datetime.datetime.now().isoformat()
            logger.info(f"Added user message to memory")
        except Exception as e:
            logger.error(f"Error adding user message to memory: {str(e)}")
    
    def add_ai_message(self, message):
        """Add an AI message to memory"""
        try:
            self.memory.chat_memory.add_ai_message(message)
            self.session_data["last_updated"] = datetime.datetime.now().isoformat()
            logger.info(f"Added AI message to memory")
        except Exception as e:
            logger.error(f"Error adding AI message to memory: {str(e)}")
    
    def add_system_context(self, context_data):
        """Add system context to session data"""
        try:
            if "user_info" in context_data:
                self.session_data["user_info"].update(context_data["user_info"])
            
            if "device_info" in context_data:
                if isinstance(context_data["device_info"], list):
                    self.session_data["device_info"] = context_data["device_info"]
                else:
                    self.session_data["device_info"].append(context_data["device_info"])
            
            if "issue_data" in context_data:
                self.session_data["issue_data"].update(context_data["issue_data"])
            
            if "metadata" in context_data:
                self.session_data["metadata"].update(context_data["metadata"])
            
            self.session_data["last_updated"] = datetime.datetime.now().isoformat()
            logger.info(f"Added system context to session data")
        except Exception as e:
            logger.error(f"Error adding system context: {str(e)}")
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        try:
            chat_history = self.load_memory_variables().get("chat_history", [])
            
            # Format chat history as text
            summary = ""
            for message in chat_history:
                if hasattr(message, "type"):
                    message_type = message.type
                    content = message.content
                else:
                    # Fallback for different message format
                    message_type = getattr(message, "role", "unknown")
                    content = getattr(message, "content", "")
                
                if message_type == "human" or message_type == "user":
                    summary += f"User: {content}\n"
                elif message_type == "ai" or message_type == "assistant":
                    summary += f"Assistant: {content}\n"
                else:
                    summary += f"{message_type.capitalize()}: {content}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return "Error retrieving conversation history"
    
    def export_session_data(self):
        """Export the full session data including memory"""
        try:
            # Get the chat history
            chat_history = self.load_memory_variables().get("chat_history", [])
            
            # Format messages for export
            formatted_messages = []
            for message in chat_history:
                if hasattr(message, "type"):
                    message_type = message.type
                    content = message.content
                else:
                    # Fallback for different message format
                    message_type = getattr(message, "role", "unknown")
                    content = getattr(message, "content", "")
                
                formatted_messages.append({
                    "role": message_type,
                    "content": content,
                    "timestamp": datetime.datetime.now().isoformat()  # Ideally this would be the actual timestamp
                })
            
            # Combine session data with messages
            export_data = {
                "session_id": self.session_id,
                "created_at": self.session_data["created_at"],
                "last_updated": datetime.datetime.now().isoformat(),
                "user_info": self.session_data["user_info"],
                "device_info": self.session_data["device_info"],
                "issue_data": self.session_data["issue_data"],
                "metadata": self.session_data["metadata"],
                "messages": formatted_messages
            }
            
            return export_data
        except Exception as e:
            logger.error(f"Error exporting session data: {str(e)}")
            return {
                "session_id": self.session_id,
                "error": str(e)
            }