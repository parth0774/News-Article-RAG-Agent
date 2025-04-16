from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging
from datetime import datetime
import os
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ConversationManager')

class ConversationManager:
    def __init__(self):
        """Initialize the conversation manager."""
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.shared_context: Dict[str, Any] = {}
        self.agent_interactions: Dict[str, List[Dict[str, str]]] = {}  # Track agent interactions
        logger.info("Conversation manager initialized")
    
    def initialize(self):
        """Initialize the conversation manager and create necessary directories."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # Clear any existing conversations
            self.conversations.clear()
            self.shared_context.clear()
            self.agent_interactions.clear()
            
            logger.info("Conversation manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing conversation manager: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources and save any necessary data."""
        try:
            # Clear all conversations
            self.conversations.clear()
            self.shared_context.clear()
            
            logger.info("Conversation manager cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up conversation manager: {str(e)}")
            return False
    
    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        try:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
            logger.info(f"Created new conversation: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return "default"
    
    def add_agent_interaction(self, conversation_id: str, from_agent: str, to_agent: str, content: str) -> bool:
        """Add an interaction between agents to the conversation."""
        try:
            if conversation_id not in self.agent_interactions:
                self.agent_interactions[conversation_id] = []
            
            self.agent_interactions[conversation_id].append({
                "from": from_agent,
                "to": to_agent,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only the last 10 interactions
            if len(self.agent_interactions[conversation_id]) > 10:
                self.agent_interactions[conversation_id] = self.agent_interactions[conversation_id][-10:]
            
            logger.info(f"Added agent interaction from {from_agent} to {to_agent}")
            return True
        except Exception as e:
            logger.error(f"Error adding agent interaction: {str(e)}")
            return False
    
    def get_agent_interactions(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get all agent interactions for a conversation."""
        try:
            return self.agent_interactions.get(conversation_id, [])
        except Exception as e:
            logger.error(f"Error getting agent interactions: {str(e)}")
            return []
    
    def add_message(self, conversation_id: str, role: str, content: str, agent: Optional[str] = None) -> bool:
        """Add a message to a conversation with optional agent information."""
        try:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            if agent:
                message["agent"] = agent
            
            self.conversations[conversation_id].append(message)
            
            # Keep only the last 10 messages
            if len(self.conversations[conversation_id]) > 10:
                self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
            
            logger.info(f"Added message to conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def get_conversation(self, conversation_id: str, include_agent_interactions: bool = False) -> List[Dict[str, str]]:
        """Get the messages from a conversation, optionally including agent interactions."""
        try:
            messages = self.conversations.get(conversation_id, [])
            
            if include_agent_interactions:
                interactions = self.agent_interactions.get(conversation_id, [])
                # Combine and sort by timestamp
                all_messages = messages + [
                    {
                        "role": "system",
                        "content": f"Agent {i['from']} to {i['to']}: {i['content']}",
                        "timestamp": i["timestamp"]
                    }
                    for i in interactions
                ]
                return sorted(all_messages, key=lambda x: x["timestamp"])
            
            return messages
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            return []
    
    def update_shared_context(self, key: str, value: Any) -> bool:
        """Update the shared context between agents."""
        try:
            self.shared_context[key] = value
            logger.info(f"Updated shared context with key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating shared context: {str(e)}")
            return False
    
    def get_shared_context(self, key: str) -> Any:
        """Get a value from the shared context."""
        try:
            return self.shared_context.get(key)
            
        except Exception as e:
            logger.error(f"Error getting shared context: {str(e)}")
            return None
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation."""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                logger.info(f"Cleared conversation: {conversation_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            return False
    
    def log_agent_interaction(self, agent_name: str, action: str, details: str):
        """Log agent interactions."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] [{agent_name}] {action}: {details}")

# Create a singleton instance
conversation_manager = ConversationManager() 