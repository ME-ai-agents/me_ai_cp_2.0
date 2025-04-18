# agent/base_agent.py
from langchain.agents import AgentExecutor
from langchain.agents import ConversationalAgent

# agent/base_agent.py
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain_aws import ChatBedrock

from existing.response_generator import get_agent_prompt
from existing.db_service import find_employee_by_contact, get_employee_devices
import logging

logger = logging.getLogger('me_agent_orchestrator')

class MeAIBaseAgent:
    """Base agent class for ME.ai agents using LangChain"""
    
    def __init__(self, agent_type, aws_region="us-east-1", model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        self.agent_type = agent_type
        self.model_id = model_id
        self.aws_region = aws_region
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = self._initialize_llm()
        self.tools = self._get_tools()
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()
    
    def _initialize_llm(self):
        """Initialize the LLM (using AWS Bedrock)"""
        try:
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
    
    def _create_base_prompt(self, employee_info=None):
        """Create a base prompt for the agent based on agent type"""
        # Reuse your existing prompts
        base_prompt = get_agent_prompt(self.agent_type, employee_info, "")
        
        # Add LangChain agent components
        agent_prompt = """
{base_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
{agent_scratchpad}
"""
        
        # Replace base_prompt placeholder with actual content
        agent_prompt = agent_prompt.replace("{base_prompt}", base_prompt)
        
        template = PromptTemplate(
            input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
            template=agent_prompt
        )
        
        return template
    
    def _get_tools(self):
        """Get the tools available to this agent"""
        # Default empty list of tools, to be overridden by subclasses
        return []
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = self._create_base_prompt()
        
        return ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def _create_agent_executor(self):
        """Create the agent executor"""
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process(self, user_input, session=None):
        """Process user input and return agent response"""
        try:
            # Update employee info in the prompt if available
            if session and hasattr(session, 'employee_info') and session.employee_info:
                # Create a new agent with updated prompt rather than trying to modify the existing one
                prompt = self._create_base_prompt(session.employee_info)
                agent = ConversationalAgent.from_llm_and_tools(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=prompt
                )
                agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=self.tools,
                    memory=self.memory,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5
                )
                response = agent_executor.run(input=user_input)
            else:
                # Use the existing agent executor
                response = self.agent_executor.run(input=user_input)
            
            return response
        except Exception as e:
            logger.error(f"Error processing input with agent: {str(e)}")
            # Fallback to your existing response generator
            from existing.response_generator import generate_fallback_response
            return generate_fallback_response(user_input, session, self.agent_type)