# me_ai_integration.py
import os
import logging
import json
from typing import Dict, Any, List, Optional, Tuple

# Import custom components
from enhanced.semantic_profile_manager import SemanticProfileManager
from enhanced.neo4j_itsm_manager import ITSMOntologyManager
from enhanced.rag_knowledge_base import RAGKnowledgeBase
from enhanced.enhanced_prompt_templates import SemanticProfilePromptManager

# Import existing ME.ai components
from me_agent_orchestrator import MEAgentOrchestrator
from existing.session_manager import Session
from existing.db_service import find_employee_by_contact

logger = logging.getLogger('me_ai_integration')

class MEAIEnhancedOrchestrator:
    """
    Enhanced ME.ai orchestrator that integrates:
    - RAG: Retrieval-Augmented Generation with AWS Bedrock and OpenSearch
    - KG: Knowledge Graph with Neo4j ITSM Ontology
    - Semantic Profiles: User profiles for personalized interactions
    """
    
    def __init__(self, config=None):
        # Load configuration
        self.config = config or {
            "aws_region": os.environ.get('AWS_REGION', 'us-east-1'),
            "neo4j_uri": os.environ.get('NEO4J_URI', 'neo4j+s://a18e3b72.databases.neo4j.io'),
            "neo4j_username": os.environ.get('NEO4J_USERNAME', 'neo4j'),
            "neo4j_password": os.environ.get('NEO4J_PASSWORD', 'S2LZZNCJoRtAbwB3VE-e-uKBjD9QKGyEibgI7ygad9M'),
            "knowledge_base_id": os.environ.get('BEDROCK_KNOWLEDGE_BASE_ID', 'CIGAVU9WLM'),
            "db_service_url": os.environ.get('MEAI_DB_SERVICE', 'http://127.0.0.1:5000/api'),
            "db_username": os.environ.get('DB_USERNAME', 'testadmin'),
            "db_password": os.environ.get('DB_PASSWORD', 'testpass'),
            "bedrock_model_id": os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        }
        
        # Initialize base orchestrator
        self.base_orchestrator = MEAgentOrchestrator()
        
        # Initialize enhanced components
        self.profile_manager = SemanticProfileManager(
            self.config["db_service_url"],
            self.config["db_username"],
            self.config["db_password"]
        )
        
        self.ontology_manager = ITSMOntologyManager(
            self.config["neo4j_uri"],
            self.config["neo4j_username"],
            self.config["neo4j_password"]
        )
        
        # Initialize enhanced components
        self.knowledge_base = RAGKnowledgeBase(
            self.config["aws_region"],
            self.config["knowledge_base_id"],
            model_id=self.config.get("bedrock_model_id")
        )
        
        # Initialize semantic-aware prompt manager
        self.prompt_manager = SemanticProfilePromptManager(
            base_prompt_manager=self.base_orchestrator.prompt_manager
        )
        
        logger.info("ME.ai Enhanced Orchestrator initialized")
    
    def _extract_keywords(self, text):
        """Extract key terms from the query for searching"""
        # Simple keyword extraction
        # In production, you might want to use NLP techniques or AWS Comprehend
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like", "through", "over", "before", "between", "after", "since", "without", "under", "within", "along", "following", "across", "behind", "beyond", "plus", "except", "but", "up", "out", "around", "down", "off", "above", "near"}
        
        # Split text into words, lowercase, and remove common words
        words = [word.lower() for word in text.split() if word.lower() not in common_words and len(word) > 2]
        
        # Remove punctuation from words
        import string
        translator = str.maketrans('', '', string.punctuation)
        words = [word.translate(translator) for word in words]
        
        # Remove duplicates while preserving order
        seen = set()
        keywords = [word for word in words if not (word in seen or seen.add(word))]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _get_user_profile(self, session):
        """Get user semantic profile from session information"""
        user_profile = None
        
        # Try to get profile from email
        if hasattr(session, 'customer_email') and session.customer_email:
            user_profile = self.profile_manager.get_profile_by_email(session.customer_email)
            if user_profile:
                logger.info(f"Found semantic profile by email: {session.customer_email}")
        
        # If no profile from email, try phone number
        if not user_profile and hasattr(session, 'customer_number') and session.customer_number:
            user_profile = self.profile_manager.get_profile_by_phone(session.customer_number)
            if user_profile:
                logger.info(f"Found semantic profile by phone: {session.customer_number}")
        
        return user_profile
    
    def _enhance_prompt_with_context(self, base_prompt, issue_type, keywords, semantic_profile, query):
        """Enhance the prompt with RAG, KG, and semantic profile context"""
        enhanced_prompt = base_prompt
        
        # Add semantic profile information if available
        if semantic_profile:
            profile_section = self.profile_manager.create_profile_prompt_section(semantic_profile)
            profile_instructions = self.profile_manager.get_tailored_instructions(semantic_profile)
            
            # Add profile information after system instructions but before user query
            if "User:" in enhanced_prompt:
                parts = enhanced_prompt.split("User:")
                enhanced_prompt = parts[0] + "\n" + profile_section + "\n" + profile_instructions + "\n\nUser:" + parts[1]
            else:
                enhanced_prompt += "\n\n" + profile_section + "\n" + profile_instructions
        
        # Add ontology information
        ontology_concepts = self.ontology_manager.query_concepts_by_issue(issue_type, keywords)
        if ontology_concepts:
            ontology_section = self.ontology_manager.format_ontology_for_prompt(ontology_concepts)
            
            # Add troubleshooting steps
            troubleshooting_steps = self.ontology_manager.get_standardized_troubleshooting_steps(issue_type)
            
            # Add ontology context after system instructions but before user query
            if "User:" in enhanced_prompt:
                parts = enhanced_prompt.split("User:")
                enhanced_prompt = parts[0] + "\n\nITSM ONTOLOGY CONTEXT:\n" + ontology_section + "\n" + troubleshooting_steps + "\n\nUser:" + parts[1]
            else:
                enhanced_prompt += "\n\nITSM ONTOLOGY CONTEXT:\n" + ontology_section + "\n" + troubleshooting_steps
        
        # Add RAG knowledge base information
        documents = self.knowledge_base.get_relevant_documents(query, issue_type)
        if documents:
            kb_section = self.knowledge_base.format_documents_for_prompt(documents)
            
            # Add knowledge base context after system instructions but before user query
            if "User:" in enhanced_prompt:
                parts = enhanced_prompt.split("User:")
                enhanced_prompt = parts[0] + "\n\nKNOWLEDGE BASE CONTEXT:\n" + kb_section + "\n\nUser:" + parts[1]
            else:
                enhanced_prompt += "\n\nKNOWLEDGE BASE CONTEXT:\n" + kb_section
        
        return enhanced_prompt
    
    def get_initial_greeting(self, session):
        """Generate an enhanced personalized greeting"""
        try:
            # Get semantic profile if available
            user_profile = self._get_user_profile(session)
            
            # If we have a user profile, use it to customize the greeting
            if user_profile:
                # Get profile-specific information
                tech_level = self.profile_manager.get_tech_proficiency_level(user_profile)
                comm_prefs = self.profile_manager.get_communication_preferences(user_profile)
                
                # Get language preference if available
                language = comm_prefs.get('language', 'english').lower()
                
                # Update session with language preference
                if not hasattr(session, 'language'):
                    setattr(session, 'language', language)
                
                # Get demographic information
                demographics = {}
                for field in ['demog_age', 'demog_gender', 'demog_occupation', 'demog_education', 'demog_location']:
                    if field in user_profile and user_profile[field]:
                        key = field.replace('demog_', '')
                        demographics[key] = user_profile[field]
                
                # Get a profile-aware greeting template
                greeting_prompt = self.prompt_manager.create_profile_aware_prompt(
                    "greeting", 
                    language,
                    user_profile,
                    session.employee_info if hasattr(session, 'employee_info') else None
                )
                
                # For MVP, fall back to base orchestrator with profile info
                return self.base_orchestrator.get_initial_greeting(session)
            else:
                # No profile, use base orchestrator
                return self.base_orchestrator.get_initial_greeting(session)
        except Exception as e:
            logger.error(f"Error generating enhanced greeting: {str(e)}")
            return self.base_orchestrator.get_initial_greeting(session)
    
    def process_query(self, query, session):
        """Process a user query with enhanced context from RAG, KG, and semantic profile"""
        try:
            # Get user profile
            user_profile = self._get_user_profile(session)
            
            # Extract keywords for ontology and knowledge base search
            keywords = self._extract_keywords(query)
            
            # Classify issue type
            issue_classification = self.ontology_manager.get_issue_classification(query)
            issue_type = issue_classification["category"]
            agent_type = issue_classification["agent_category"]
            
            # Store issue type in session
            session.issue_type = agent_type
            
            # Select agent based on issue type
            if agent_type in self.base_orchestrator.agents:
                agent = self.base_orchestrator.agents[agent_type]
                logger.info(f"Using {agent_type} agent for processing")
            else:
                # Fall back to default agent
                agent = self.base_orchestrator.agents[self.base_orchestrator.default_agent]
                logger.info(f"Using default agent ({self.base_orchestrator.default_agent}) for processing")
            
            # Get base prompt from agent
            if hasattr(agent, '_create_base_prompt'):
                base_prompt = agent._create_base_prompt(session.employee_info)
                base_prompt_text = base_prompt.template
            else:
                # Fallback to a generic prompt
                base_prompt_text = """
You are ME.ai Assistant, a helpful and empathetic IT support specialist.

USER INFORMATION:
Name: {employee_name}
Department: {department}
Role: {role}
Devices: {devices}

BEHAVIOR GUIDELINES:
- Be friendly, professional, and empathetic in your responses
- If you need more information to troubleshoot, ask specific questions
- Focus on solving their problem efficiently
- Ask only one question at a time to avoid overwhelming the user
- Keep responses concise and easy to understand (around 2-3 paragraphs maximum)
- Avoid technical jargon unless the user appears technically proficient
- Address the user by their first name if available

User: {input}
ME.ai Assistant:
"""
            
            # Enhance prompt with contextual information
            enhanced_prompt = self._enhance_prompt_with_context(
                base_prompt_text,
                issue_type,
                keywords,
                user_profile,
                query
            )
            
            # For debugging
            logger.info(f"Enhanced prompt with RAG+KG+Semantic Profile context, length: {len(enhanced_prompt)}")
            
            # For the MVP, we'll use the base orchestrator's process function
            # but in a production implementation, you'd modify the agent to use the enhanced prompt
            response = agent.process(query, session)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced processing: {str(e)}")
            # Fall back to base orchestrator
            return self.base_orchestrator.process_query(query, session)
    
    def process_message(self, message, session_id, user_email=None, user_phone=None):
        """Process a message from any channel with enhanced capabilities"""
        try:
            # Get or create session
            from existing.session_manager import SessionManager
            session_manager = SessionManager()
            session = session_manager.get_session(session_id)
            
            # Update session contact info if provided
            if user_email:
                session.customer_email = user_email
            if user_phone:
                session.customer_number = user_phone
            
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
                session_manager.save_session(session)
                
                return greeting
            
            # Add user message to session
            if not hasattr(session, 'messages'):
                session.messages = []
            
            session.messages.append({
                "role": "user",
                "content": message
            })
            
            # Process with enhanced capabilities
            response = self.process_query(message, session)
            
            # Add response to session
            session.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Save session
            session_manager.save_session(session)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our IT support team directly if your issue is urgent."

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "aws_region": "us-east-1",
        "neo4j_uri": "neo4j+s://a18e3b72.databases.neo4j.io", 
        "neo4j_username": "neo4j",
        "neo4j_password": "S2LZZNCJoRtAbwB3VE-e-uKBjD9QKGyEibgI7ygad9M",
        "knowledge_base_id": 'CIGAVU9WLM',
        "db_service_url": "http://localhost:5000/api",
        "db_username": "testadmin",
        "db_password": "testpass",
        "bedrock_model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    }
    
    # Initialize the enhanced orchestrator
    orchestrator = MEAIEnhancedOrchestrator(config)
    
    # Create a test session
    from existing.session_manager import Session
    session_id = "test-session-123"
    session = Session(session_id)
    session.customer_email = "test.user@example.com"
    session.customer_number = "+1234567890"
    
    # Process a test message
    response = orchestrator.process_message(
        "My laptop is running very slow and the battery drains quickly",
        session_id,
        user_email="test.user@example.com"
    )
    
    print("Response:", response)