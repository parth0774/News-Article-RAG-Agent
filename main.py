import os
import logging
from dotenv import load_dotenv
from Agents.Agent1_Orchestrator import process_query, orchestrator
from conversation_manager import conversation_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MultiAgentSystem')

def initialize_system():
    """Initialize the multi-agent system."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Verify required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize conversation manager
        conversation_manager.initialize()
        
        logger.info("Multi-agent system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def main():
    """Main entry point for the multi-agent system."""
    print("\n" + "="*50)
    print("Welcome to the News and LinkedIn Post Generator!")
    print("="*50)
    print("\nThis system can help you with:")
    print("1. Finding and summarizing news articles")
    print("2. Generating professional LinkedIn posts")
    print("3. Creating LinkedIn posts based on news topics")
    print("\nType 'exit' to quit or 'help' for more information.")
    
    if not initialize_system():
        print("\nFailed to initialize the system. Please check the logs for details.")
        return
    
    try:
        while True:
            user_query = input("\nEnter your query (or 'exit' to quit): ")
            
            if user_query.lower() == 'exit':
                break
                
            response = process_query(user_query)
            
            # Display the response
            print("\n" + "="*50)
            print(f"Agent: {response['agent']}")
            print("="*50)
            print("\nResponse:")
            print(response['response'])
            print("\nDetails:", response['details'])
            print("="*50)
            
            # Get and display conversation history
            conversation_id = conversation_manager.get_shared_context("current_conversation_id")
            if conversation_id:
                print("\nConversation History:")
                messages = conversation_manager.get_conversation(conversation_id, include_agent_interactions=True)
                for msg in messages:
                    role = msg.get("role", "unknown")
                    agent = msg.get("agent", "")
                    content = msg.get("content", "")
                    print(f"\n{role.upper()}{f' ({agent})' if agent else ''}:")
                    print(content)
            
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")
    finally:
        conversation_manager.cleanup()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main() 