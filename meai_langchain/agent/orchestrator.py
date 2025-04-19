# agent/orchestrator.py
from langchain.agents import AgentExecutor
from langchain.agents import ConversationalAgent
from langchain_aws import ChatBedrock
#from langchain.agents.conversational_agent.base import ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain_aws import ChatBedrock

#from langchain_aws import BedrockLLM
from langchain.chains import LLMChain
from langchain.callbacks import StdOutCallbackHandler

from existing.response_generator import classify_issue
from existing.session_manager import Session
from .hardware_agent import HardwareAgent
from .software_agent import SoftwareAgent
from .password_agent import PasswordAgent

import logging

logger = logging.getLogger('me_agent_orchestrator')

class AgentOrchestrator:
    """Orchestrator that manages different specialized agents"""
    
    def __init__(self, aws_region="us-east-1", model_id="anthropic.claude-3-sonnet-20240229"):
        self.aws_region = aws_region
        self.model_id = model_id
        self.llm = self._initialize_llm()
        
        # Initialize specialized agents
        self.agents = {
            "Hardware": HardwareAgent(aws_region, model_id),
            "Software": SoftwareAgent(aws_region, model_id),
            "Password": PasswordAgent(aws_region, model_id),
        }
        
        # Default to hardware agent for now
        self.default_agent = "Hardware"
        
        # Create classifier chain
        self.classifier_chain = self._create_classifier_chain()
    
    def _initialize_llm(self):
        """Initialize the LLM (using AWS Bedrock)"""
        try:
            # You can use other LLMs as well
            return ChatBedrock(
                model_id=self.model_id,
                region_name=self.aws_region,
                model_kwargs={"temperature": 0.7, "max_tokens": 1000}
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # Fallback to DeepSeek if configured
            # This would need to be implemented
            raise e
    
    def _create_classifier_chain(self):
        """Create a chain for classifying the issue type"""
        classifier_template = """
You are a helpful assistant that categorizes IT support issues into one of these categories:
1. Hardware - Issues with physical devices, computers, printers, etc.
2. Software - Issues with applications, operating systems, etc.
3. Password - Issues with account access, login problems, etc.
4. General - Other IT issues that don't fit the above categories

USER QUERY: {query}

CATEGORY:"""
        
        classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template=classifier_template
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=classifier_prompt,
            verbose=True
        )
    
    def classify_issue_type(self, query):
        """Classify the issue type using LangChain"""
        try:
            # First try the existing classifier function
            issue_type = classify_issue(query)
            
            # If it returns General, try the LLM-based classifier
            if issue_type == "General":
                response = self.classifier_chain.run(query=query)
                # Extract the category from response
                response = response.strip()
                
                # Map to valid categories
                if "Hardware" in response:
                    return "Hardware"
                elif "Software" in response:
                    return "Software"  
                elif "Password" in response:
                    return "Password"
                else:
                    return "General"
            
            return issue_type
        except Exception as e:
            logger.error(f"Error classifying issue: {str(e)}")
            return "General"  # Default to General if classification fails
    
    def process_query(self, query, session=None):
        """Process a user query, selecting appropriate agent and returning response"""
        try:
            # If no issue type in session, classify it
            if session and not session.issue_type:
                issue_type = self.classify_issue_type(query)
                session.issue_type = issue_type
                logger.info(f"Classified issue as: {issue_type}")
            elif session and session.issue_type:
                issue_type = session.issue_type
            else:
                issue_type = self.classify_issue_type(query)
                logger.info(f"Classified issue as: {issue_type} (no session)")
            
            # Select agent based on issue type
            if issue_type in self.agents:
                agent = self.agents[issue_type]
                logger.info(f"Using {issue_type} agent for processing")
            else:
                # Fall back to default agent
                agent = self.agents[self.default_agent]
                logger.info(f"Using default agent ({self.default_agent}) for processing")
            
            # Process query with selected agent
            response = agent.process(query, session)
            return response
            
        except Exception as e:
            logger.error(f"Error in orchestrator processing: {str(e)}")
            # Fallback to your existing response generator
            from existing.response_generator import generate_fallback_response
            return generate_fallback_response(query, session)
    
    def get_initial_greeting(self, session):
        """Generate an initial greeting for the user"""
        from existing.response_generator import generate_initial_greeting
        return generate_initial_greeting(session)
