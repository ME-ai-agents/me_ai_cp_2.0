# initialize_meai_enhanced.py
import os
import logging
import argparse
import json
from dotenv import load_dotenv

# Import ME.ai enhanced integration
from me_ai_integration import MEAIEnhancedOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meai_initialization')

def load_config(config_path=None):
    """Load configuration from .env file or config file"""
    # Load .env file if exists
    load_dotenv()
    
    # Start with environment variables
    config = {
        "aws_region": os.environ.get('AWS_REGION', 'us-east-1'),
        "neo4j_uri": os.environ.get('NEO4J_URI', 'neo4j+s://a18e3b72.databases.neo4j.io'),
        "neo4j_username": os.environ.get('NEO4J_USERNAME', 'neo4j'),
        "neo4j_password": os.environ.get('NEO4J_PASSWORD', ''),
        "knowledge_base_id": os.environ.get('BEDROCK_KNOWLEDGE_BASE_ID', 'CIGAVU9WLM'),
        "db_service_url": os.environ.get('MEAI_DB_SERVICE', 'http://127.0.0.1:5000/api'),
        "db_username": os.environ.get('DB_USERNAME', 'testadmin'),
        "db_password": os.environ.get('DB_PASSWORD', 'testpass'),
        "bedrock_model_id": os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    }
    
    # If config file provided, override with its values
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    return config



def test_system_components(orchestrator):
    """Test the various components of the system"""
    success = True
    
    # Test Neo4j connection
    try:
        concepts = orchestrator.ontology_manager.query_incident_management_process()
        if concepts:
            logger.info(f"Neo4j ITSM Ontology connection successful - found {len(concepts)} concepts")
        else:
            logger.warning("Neo4j ITSM Ontology connection successful but no concepts found")
    except Exception as e:
        logger.error(f"Error connecting to Neo4j ITSM Ontology: {str(e)}")
        success = False
    
    # Test RAG Knowledge Base
    try:
        documents = orchestrator.knowledge_base.query("slow laptop performance")
        if documents:
            logger.info(f"RAG Knowledge Base query successful - found {len(documents)} documents")
        else:
            logger.warning("RAG Knowledge Base query successful but no documents found")
    except Exception as e:
        logger.error(f"Error querying RAG Knowledge Base: {str(e)}")
        success = False
    
    # Test DB Service for Semantic Profiles
    try:
        token = orchestrator.profile_manager._get_db_token()
        if token:
            logger.info("DB Service connection successful")
        else:
            logger.warning("DB Service connection failed - could not obtain token")
            success = False
    except Exception as e:
        logger.error(f"Error connecting to DB Service: {str(e)}")
        success = False
    
    return success

def run_interactive_session(orchestrator):
    """Run an interactive session for testing"""
    from existing.session_manager import Session
    
    print("\n=== ME.ai Enhanced Interactive Session ===")
    print("Type 'exit' to quit")
    
    # Create a test session
    session_id = f"test-session-{os.urandom(4).hex()}"
    session = Session(session_id)
    
    # Optionally set user info
    email = input("Enter test user email (optional): ")
    if email:
        session.customer_email = email
    
    phone = input("Enter test user phone (optional): ")
    if phone:
        session.customer_number = phone
    
    # Main interaction loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        # Process message
        response = orchestrator.process_message(user_input, session_id, email, phone)
        print(f"\nME.ai: {response}")

def main():
    """Main function to initialize and optionally test ME.ai enhanced system"""
    parser = argparse.ArgumentParser(description='Initialize ME.ai Enhanced System')
    parser.add_argument('--config', '-c', help='Path to configuration file (JSON)')
    parser.add_argument('--test', '-t', action='store_true', help='Test system components')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run interactive session')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize orchestrator
    logger.info("Initializing ME.ai Enhanced Orchestrator...")
    orchestrator = MEAIEnhancedOrchestrator(config)
    logger.info("ME.ai Enhanced Orchestrator initialized")
    
    # Test components if requested
    if args.test:
        logger.info("Testing system components...")
        success = test_system_components(orchestrator)
        if success:
            logger.info("All system components tested successfully")
        else:
            logger.warning("Some system components failed testing")
    
    # Run interactive session if requested
    if args.interactive:
        run_interactive_session(orchestrator)
    
    logger.info("ME.ai Enhanced System initialization complete")

if __name__ == "__main__":
    main()