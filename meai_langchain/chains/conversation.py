# chains/conversation.py
from typing import List, Dict, Any, Optional
import logging
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM

logger = logging.getLogger('me_agent_orchestrator')

class MEConversationChain:
    """Enhanced conversation chain for ME.ai with empathetic responses and multilingual support"""
    
    def __init__(self, llm, memory=None, language="english"):
        self.llm = llm
        self.language = language.lower()
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create the conversation chain with appropriate prompt template"""
        # Base template in English
        base_template = """
You are ME.ai Assistant, a helpful and empathetic IT support specialist.

Current conversation:
{chat_history}

User: {input}
ME.ai Assistant:"""

        # Add language-specific elements if not English
        if self.language != "english":
            empathy_keywords = self._get_language_specific_elements()
            prompt_template = f"""
You are ME.ai Assistant, a helpful and empathetic IT support specialist.
Respond in {self.language}.

Current conversation:
{{chat_history}}

User: {{input}}
ME.ai Assistant:"""
        else:
            prompt_template = base_template
            
        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=prompt_template
        )
        
        return ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=True
        )
    
    def _get_language_specific_elements(self):
        """Get language-specific elements for empathy"""
        # This could be expanded for more languages
        language_elements = {
            "spanish": {
                "greetings": ["Hola", "Buenos días", "Buenas tardes", "Buenas noches"],
                "empathy": ["Entiendo", "Comprendo", "Lamento escuchar eso", "Puedo ayudarte con eso"]
            },
            "french": {
                "greetings": ["Bonjour", "Salut", "Bonsoir"],
                "empathy": ["Je comprends", "Je suis désolé d'entendre ça", "Je peux vous aider"]
            },
            "german": {
                "greetings": ["Hallo", "Guten Tag", "Guten Morgen", "Guten Abend"],
                "empathy": ["Ich verstehe", "Das tut mir leid", "Ich kann Ihnen helfen"]
            }
        }
        
        return language_elements.get(self.language, {})
    
    def set_language(self, language):
        """Update the conversation language"""
        self.language = language.lower()
        # Recreate the chain with the new language
        self.chain = self._create_chain()
        logger.info(f"Conversation language updated to: {language}")
    
    def process(self, user_input, employee_info=None):
        """Process user input and return a response"""
        try:
            # Add any employee context to the memory if provided
            if employee_info and not self.memory.chat_memory.messages:
                context = f"[System: User is {employee_info.get('name', 'an employee')}, department: {employee_info.get('department', 'unknown')}, role: {employee_info.get('role', 'unknown')}]"
                # Add this context to memory as a system message
                self.memory.chat_memory.add_message("system", context)
            
            # Run the conversation chain
            response = self.chain.run(input=user_input)
            logger.info(f"Generated conversation response of length {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error in conversation chain: {str(e)}", exc_info=True)
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our IT team directly if your issue is urgent."