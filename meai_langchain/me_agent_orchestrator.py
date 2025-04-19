# me_agent_orchestrator.py
import os
import logging
import uuid
from typing import Dict, Any, List, Optional

import datetime  # Add this import to fix the error


# LangChain imports
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory

# Internal imports
from agent.orchestrator import AgentOrchestrator
from agent.hardware_agent import HardwareAgent
from agent.software_agent import SoftwareAgent
from agent.password_agent import PasswordAgent
from existing.session_manager import SessionManager, Session
from existing.db_service import (
    find_employee_by_contact,
    get_employee_devices,
    find_agent_by_specialization,
    log_conversation_to_db
)
from existing.response_generator import generate_initial_greeting
from chains.conversation import MEConversationChain
from chains.workflow import WorkflowChain
from memory.session_memory import SessionMemory
from tools.db_tools import create_db_tools
from tools.device_tools import create_device_tools
from prompts.templates import PromptTemplateManager

# Load configuration
from config import (
    SECRET_KEY, 
    DEBUG, 
    PORT, 
    DEEPSEEK_API_KEY, 
    DEEPSEEK_API_URL,
    AWS_REGION, 
    BEDROCK_MODEL_ID, 
    MEAI_DB_SERVICE,
    DB_USERNAME, 
    DB_PASSWORD,
    LOG_LEVEL,
    LOG_FORMAT
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger('me_agent_orchestrator')

class MEAgentOrchestrator:
    """Main orchestrator for ME.ai agents using LangChain"""
    
    def __init__(self, config=None):
        # Load configuration
        self.config = config or {
            "aws_region": AWS_REGION,
            "model_id": BEDROCK_MODEL_ID,
            "db_service_url": MEAI_DB_SERVICE,
            "db_username": DB_USERNAME,
            "db_password": DB_PASSWORD
        }
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize session manager
        self.session_manager = SessionManager()
        
        # Initialize prompt template manager
        self.prompt_manager = PromptTemplateManager()
        
        # Initialize specialized agents
        self.agents = {
            "Hardware": HardwareAgent(self.config["aws_region"], self.config["model_id"]),
            "Software": SoftwareAgent(self.config["aws_region"], self.config["model_id"]),
            "Password": PasswordAgent(self.config["aws_region"], self.config["model_id"]),
        }
        
        # Initialize workflow chain
        self.workflow_chain = WorkflowChain(self.llm)
        
        # Default to hardware agent for unclassified issues
        self.default_agent = "Hardware"
        
        # Session ID to memory mapping
        self.session_memories = {}
        
        # Tools
        self.db_tools = create_db_tools()
        self.device_tools = create_device_tools()
        
        logger.info("ME.ai Agent Orchestrator initialized")
    
    def _initialize_llm(self):
        """Initialize the LLM (using AWS Bedrock)"""
        try:
            logger.info(f"Initializing LLM with model {self.config['model_id']}")
            return ChatBedrock(
                model_id=self.config["model_id"],
                region_name=self.config["aws_region"],
                model_kwargs={"temperature": 0.7, "max_tokens": 1000}
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # This would ideally be a fallback to another provider
            raise e
    
    def _get_or_create_memory(self, session_id):
        """Get or create memory for a session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = SessionMemory(
                session_id=session_id,
                memory_type="buffer",
                window_size=10,
                persistence=None  # Could be 'redis' if configured
            )
        return self.session_memories[session_id]
    
    def classify_issue_type(self, query):
        """Classify the issue type using the workflow chain"""
        try:
            # Format the conversation string - simple version for now
            conversation = f"User: {query}"
            
            # Classify using the workflow chain
            classification = self.workflow_chain.classify_issue_detailed(conversation)
            category = classification["category"]
            
            # Map category to agent type
            if "Hardware" in category or "Device" in category:
                return "Hardware"
            elif "Software" in category or "Application" in category:
                return "Software"
            elif "Password" in category or "Access" in category or "Account" in category:
                return "Password"
            else:
                return "General"
        except Exception as e:
            logger.error(f"Error classifying issue: {str(e)}")
            return "General"  # Default to General if classification fails
    
    def process_query(self, query, session):
        """Process a user query, selecting appropriate agent and returning response"""
        try:
            # Get session memory
            memory = self._get_or_create_memory(session.session_id)
            
            # Add user message to memory
            memory.add_user_message(query)
            
            # If no issue type in session, classify it
            if not session.issue_type or session.issue_type == "General":
                issue_type = self.classify_issue_type(query)
                session.issue_type = issue_type
                logger.info(f"Classified issue as: {issue_type}")
                
                # Add issue classification to session memory metadata
                memory.add_system_context({
                    "issue_data": {
                        "type": issue_type,
                        "classified_at": str(datetime.datetime.now())
                    }
                })
            else:
                issue_type = session.issue_type
            
            # Select agent based on issue type
            if issue_type in self.agents:
                agent = self.agents[issue_type]
                logger.info(f"Using {issue_type} agent for processing")
            else:
                # Fall back to default agent
                agent = self.agents[self.default_agent]
                logger.info(f"Using default agent ({self.default_agent}) for processing")
            
            # If no agent_id set for session yet, try to find one
            if not session.agent_id:
                agent_info = find_agent_by_specialization(issue_type)
                if agent_info:
                    session.agent_id = agent_info.get('agent_id')
                    logger.info(f"Assigned agent ID: {session.agent_id}")
            
            # Process query with selected agent
            response = agent.process(query, session)
            
            # Add AI response to memory
            memory.add_ai_message(response)
            
            # Log to database if we have user and agent IDs
            if session.employee_id and session.agent_id:
                try:
                    conversation_id = getattr(session, 'conversation_id', str(uuid.uuid4()))
                    log_conversation_to_db(
                        conversation_id,
                        session.employee_id,
                        session.agent_id,
                        response,
                        "AI response",
                        "In Progress"
                    )
                    logger.info(f"Logged conversation to database: {conversation_id}")
                except Exception as e:
                    logger.error(f"Error logging to database: {str(e)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in orchestrator processing: {str(e)}")
            # Fallback response
            from existing.response_generator import generate_fallback_response
            return generate_fallback_response(query, session)
    
    def get_initial_greeting(self, session):
        """Generate an initial greeting for the user"""
        try:
            # Update session with employee info if not already present
            if not session.employee_id and (session.customer_email or session.customer_number):
                # Try email first
                employee = None
                if session.customer_email:
                    employee = find_employee_by_contact('email', session.customer_email)
                
                # Try phone if email didn't work
                if not employee and session.customer_number:
                    employee = find_employee_by_contact('phone', session.customer_number)
                
                # Update session if employee found
                if employee:
                    session.employee_id = employee.get('employee_id')
                    session.employee_info = employee
                    logger.info(f"Identified employee for greeting: {employee.get('name')}")
                    
                    # Get employee devices
                    devices = get_employee_devices(employee)
                    session.devices = devices
                    logger.info(f"Found {len(devices)} devices for employee")
                    
                    # Add employee info to session memory
                    memory = self._get_or_create_memory(session.session_id)
                    memory.add_system_context({
                        "user_info": employee,
                        "device_info": devices
                    })
            
            # Use greeting prompt based on employee info
            if hasattr(session, 'employee_info') and session.employee_info:
                employee_name = session.employee_info.get('name', '').split()[0]  # First name
                department = session.employee_info.get('department', 'Unknown Department')
                
                # Get appropriate greeting prompt
                greeting_prompt = self.prompt_manager.get_prompt("greeting", "english")
                
                # Create a basic conversation chain for the greeting
                greeting_chain = MEConversationChain(self.llm)
                
                # Generate personalized greeting
                greeting = greeting_chain.process(
                    "Hello",
                    {"name": employee_name, "department": department}
                )
                
                logger.info(f"Generated personalized greeting for {employee_name}")
                return greeting
            else:
                # Use existing generator for unknown users
                return generate_initial_greeting(session)
                
        except Exception as e:
            logger.error(f"Error generating greeting: {str(e)}")
            # Simple fallback greeting
            return "Hello! I'm ME.ai Assistant, your IT support specialist. How can I help you today?"
    
    def process_message(self, message, session_id, user_email=None, user_phone=None, language=None):
        """
        Process a message from any channel
        
        Args:
            message: The user's message text
            session_id: Unique session identifier
            user_email: Optional user email
            user_phone: Optional user phone number
            language: Optional language code
            
        Returns:
            Response text from the appropriate agent
        """
        try:
            # Get or create session
            session = self.session_manager.get_session(session_id)
            
            # Update session contact info if provided
            if user_email:
                session.customer_email = user_email
            if user_phone:
                session.customer_number = user_phone
            
            # Add language to session if provided
            if language:
                if not hasattr(session, 'language'):
                    setattr(session, 'language', language)
                    logger.info(f"Set session language to: {language}")
            
            # If message is a greeting and we haven't greeted yet, send initial greeting
            if (not hasattr(session, 'greeted') or not session.greeted) and \
               any(greeting in message.lower() for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                greeting = self.get_initial_greeting(session)
                session.greeted = True
                
                # Add greeting to session messages
                if not hasattr(session, 'messages'):
                    session.messages = []
                
                session.messages.append({
                    "role": "user",
                    "content": message
                })
                
                session.messages.append({
                    "role": "assistant",
                    "content": greeting
                })
                
                # Save session
                self.session_manager.save_session(session)
                
                return greeting
            
            # Add user message to session
            if not hasattr(session, 'messages'):
                session.messages = []
            
            session.messages.append({
                "role": "user",
                "content": message
            })
            
            # Process with appropriate agent
            response = self.process_query(message, session)
            
            # Add response to session
            session.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Save session
            self.session_manager.save_session(session)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our IT support team directly if your issue is urgent."
    
    def export_conversation(self, session_id):
        """Export the full conversation history with metadata"""
        try:
            session = self.session_manager.get_session(session_id)
            memory = self._get_or_create_memory(session_id)
            
            # Get all session data
            export_data = memory.export_session_data()
            
            # Add any session attributes not in memory
            if hasattr(session, 'issue_type'):
                if 'issue_data' not in export_data:
                    export_data['issue_data'] = {}
                export_data['issue_data']['type'] = session.issue_type
            
            return export_data
        except Exception as e:
            logger.error(f"Error exporting conversation: {str(e)}")
            return {"error": str(e)}


# Direct execution for testing
if __name__ == "__main__":
    # Simple test routine
    orchestrator = MEAgentOrchestrator()
    
    # Create test session
    session_id = str(uuid.uuid4())
    session = Session(session_id)
    session.customer_email = "test.user@example.com"
    
    # Process a few test messages
    test_messages = [
        "Hello",
        "My laptop is running really slow",
        "It's a Dell Latitude, about 2 years old",
        "Yes, the fan seems to be running loudly too"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = orchestrator.process_message(message, session_id, user_email="test.user@example.com")
        print(f"ME.ai: {response}")