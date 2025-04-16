import os
import logging
from dotenv import load_dotenv
from Agents import process_query
from conversation_manager import conversation_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MultiAgentSystem')

def initialize_system():
    try:
        load_dotenv()
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        conversation_manager.initialize()
        
        logger.info("Multi-agent system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def main():
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
            
            print("\nAgent Used:", response.get('agent_used', 'Unknown'))
            print("Processing Details:", response.get('processing_details', 'No details available'))
            print("\nResponse Content:")
            print(response.get('content', 'No content available'))
            
            if 'sources' in response:
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source}")
                    
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