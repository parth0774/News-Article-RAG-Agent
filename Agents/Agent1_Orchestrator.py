import os
import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .RAG_Agent2 import RAGSystem
from .LinkedIn_Agent3 import LinkedInAgent
from conversation_manager import conversation_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Agent1_Orchestrator')

class Orchestrator:
    def __init__(self):
        """Initialize the orchestrator with RAG and LinkedIn agents."""
        self.rag_agent = RAGSystem()
        self.linkedin_agent = LinkedInAgent()
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1"))
        )
        self.prompt = """Analyze the user's query and determine the appropriate action.
        Return a JSON response with the following structure:
        {
            "agent": "rag" or "linkedin",
            "query_type": "news_query" or "linkedin_post",
            "needs_news": true or false,
            "content": "original query or extracted content"
        }
        
        Rules:
        1. If the query is about news or current events -> route to RAG
        2. If the query is about creating a LinkedIn post with provided content -> route to LinkedIn
        3. If the query is about creating a LinkedIn post about a topic -> route to RAG first, then LinkedIn
        """
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route the query to the appropriate agent."""
        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query}
            ])
            
            routing_decision = json.loads(response.content)
            logger.info(f"Routing decision: {routing_decision}")
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error in routing: {str(e)}")
            return {
                "agent": "rag",
                "query_type": "news_query",
                "needs_news": True,
                "content": query
            }
    
    def process_query(self, query: str) -> Dict[str, str]:
        """Process a user query through the appropriate agent."""
        try:
            # Get existing conversation ID or create a new one
            conversation_id = conversation_manager.get_shared_context("current_conversation_id")
            if not conversation_id:
                conversation_id = conversation_manager.create_conversation()
                conversation_manager.update_shared_context("current_conversation_id", conversation_id)
            
            routing_decision = self.route_query(query)
            
            # Add user query to conversation
            conversation_manager.add_message(conversation_id, "user", query)
            
            if routing_decision["agent"] == "rag":
                # Add interaction from orchestrator to RAG agent
                conversation_manager.add_agent_interaction(
                    conversation_id,
                    "orchestrator",
                    "rag_agent",
                    f"Processing news query: {routing_decision['content']}"
                )
                
                result = self.rag_agent.query(routing_decision["content"])
                
                # Add RAG agent's response to conversation
                conversation_manager.add_message(
                    conversation_id,
                    "assistant",
                    f"Here's the news information:\n{result['answer']}\nSources: {[s['headline'] for s in result['sources']]}",
                    "rag_agent"
                )
                
                if routing_decision["query_type"] == "linkedin_post":
                    # Add interaction from RAG to LinkedIn agent
                    conversation_manager.add_agent_interaction(
                        conversation_id,
                        "rag_agent",
                        "linkedin_agent",
                        f"Generating LinkedIn post from news content: {result['answer'][:200]}..."
                    )
                    
                    post = self.linkedin_agent.generate_post(
                        result["answer"],
                        context={"conversation_id": conversation_id}
                    )
                    
                    # Add LinkedIn agent's response to conversation
                    conversation_manager.add_message(
                        conversation_id,
                        "assistant",
                        f"Here's your LinkedIn post:\n{post}",
                        "linkedin_agent"
                    )
                    
                    return {
                        "agent": "RAG + LinkedIn",
                        "response": post,
                        "details": "First retrieved news information, then generated LinkedIn post"
                    }
                
                return {
                    "agent": "RAG",
                    "response": result["answer"],
                    "details": "Retrieved and summarized news information"
                }
                
            elif routing_decision["agent"] == "linkedin":
                # Add interaction from orchestrator to LinkedIn agent
                conversation_manager.add_agent_interaction(
                    conversation_id,
                    "orchestrator",
                    "linkedin_agent",
                    f"Generating LinkedIn post from content: {routing_decision['content'][:200]}..."
                )
                
                post = self.linkedin_agent.generate_post(
                    routing_decision["content"],
                    context={"conversation_id": conversation_id}
                )
                
                # Add LinkedIn agent's response to conversation
                conversation_manager.add_message(
                    conversation_id,
                    "assistant",
                    f"Here's your LinkedIn post:\n{post}",
                    "linkedin_agent"
                )
                
                return {
                    "agent": "LinkedIn",
                    "response": post,
                    "details": "Generated LinkedIn post directly from provided content"
                }
            
            return {
                "agent": "Unknown",
                "response": "I couldn't determine how to process your query.",
                "details": "No suitable agent found for the query"
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "agent": "Error",
                "response": "I encountered an error while processing your request.",
                "details": str(e)
            }

# Create a singleton instance
orchestrator = Orchestrator()

def process_query(query: str) -> Dict[str, str]:
    """Process a query using the orchestrator."""
    return orchestrator.process_query(query)

def main(query: str):
        response = process_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 