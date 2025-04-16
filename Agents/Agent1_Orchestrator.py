import os
import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from Agents.RAG_Agent2 import RAGSystem
from Agents.LinkedIn_Agent3 import LinkedInAgent
from conversation_manager import conversation_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_orchestrator.log'),
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
            model="gpt-4-turbo-preview",
            temperature=0.1
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
            # Generate a unique conversation ID
            conversation_id = conversation_manager.create_conversation()
            
            # Route the query
            routing_decision = self.route_query(query)
            
            # Process based on routing decision
            if routing_decision["agent"] == "rag":
                # Get news information
                result = self.rag_agent.query(routing_decision["content"])
                
                # Update conversation history
                conversation_manager.add_message(
                    conversation_id,
                    "assistant",
                    f"Here's the news information:\n{result['answer']}\nSources: {[s['headline'] for s in result['sources']]}"
                )
                
                # If this was for a LinkedIn post, generate the post
                if routing_decision["query_type"] == "linkedin_post":
                    post = self.linkedin_agent.generate_post(
                        result["answer"],
                        context={"conversation_id": conversation_id}
                    )
                    conversation_manager.add_message(
                        conversation_id,
                        "assistant",
                        f"Here's your LinkedIn post:\n{post}"
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
                # Generate LinkedIn post directly
                post = self.linkedin_agent.generate_post(
                    routing_decision["content"],
                    context={"conversation_id": conversation_id}
                )
                
                # Update conversation history
                conversation_manager.add_message(
                    conversation_id,
                    "assistant",
                    f"Here's your LinkedIn post:\n{post}"
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

def main():
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = process_query(query)
        print(f"Response: {response}")
    else:
        print("Please provide a query as a command line argument.")
        print("Example: python Agent1_Orchestrator.py 'What's the latest news about AI?'")

if __name__ == "__main__":
    main() 