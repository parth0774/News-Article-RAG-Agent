from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging
from datetime import datetime
import os
import uuid

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
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.shared_context: Dict[str, Any] = {}
        logger.info("Conversation manager initialized")
    
    def initialize(self):
        try:
            os.makedirs('logs', exist_ok=True)
            
            self.conversations.clear()
            self.shared_context.clear()
            
            logger.info("Conversation manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing conversation manager: {str(e)}")
            return False
    
    def cleanup(self):
        try:
            self.conversations.clear()
            self.shared_context.clear()
            
            logger.info("Conversation manager cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up conversation manager: {str(e)}")
            return False
    
    def create_conversation(self) -> str:
        try:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
            logger.info(f"Created new conversation: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return "default"
    
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        try:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            self.conversations[conversation_id].append({
                "role": role,
                "content": content
            })
            
            if len(self.conversations[conversation_id]) > 5:
                self.conversations[conversation_id] = self.conversations[conversation_id][-5:]
            
            logger.info(f"Added message to conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        try:
            return self.conversations.get(conversation_id, [])
            
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            return []
    
    def update_shared_context(self, key: str, value: Any) -> bool:
        try:
            self.shared_context[key] = value
            logger.info(f"Updated shared context with key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating shared context: {str(e)}")
            return False
    
    def get_shared_context(self, key: str) -> Any:
        try:
            return self.shared_context.get(key)
            
        except Exception as e:
            logger.error(f"Error getting shared context: {str(e)}")
            return None
    
    def clear_conversation(self, conversation_id: str) -> bool:
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] [{agent_name}] {action}: {details}")

conversation_manager = ConversationManager() 