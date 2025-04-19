# chains/workflow.py
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger('me_agent_orchestrator')

class WorkflowChain:
    """Chain that manages the workflow of the conversation and determines next actions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.planning_chain = self._create_planning_chain()
        self.issue_chain = self._create_issue_classification_chain()
        self.action_chain = self._create_action_recommendation_chain()
        
    def _create_planning_chain(self):
        """Create a chain for planning the conversation flow"""
        planning_template = """
You are an IT support workflow planner. Given the conversation history and context, determine the best next steps.

CONVERSATION HISTORY:
{conversation_history}

USER INFORMATION:
Employee Name: {employee_name}
Department: {department}
Role: {role}

CURRENT ISSUE:
Issue Type: {issue_type}
Progress: {progress_stage}

Your task is to determine the optimal next step in the conversation to efficiently resolve the user's issue.
Think step by step about what information has been gathered, what is still needed, and what actions should be taken.

Output your analysis as:
ANALYSIS: [Your detailed analysis of the current state]
NEXT STEP: [What the agent should focus on next: "gather_info", "troubleshoot", "verify_solution", "escalate", or "close"]
REASONING: [Why this is the appropriate next step]
"""
        
        planning_prompt = PromptTemplate(
            input_variables=["conversation_history", "employee_name", "department", "role", "issue_type", "progress_stage"],
            template=planning_template
        )
        
        return LLMChain(llm=self.llm, prompt=planning_prompt, verbose=True)
    
    def _create_issue_classification_chain(self):
        """Create a chain for classifying issues more precisely"""
        issue_template = """
You are an IT issue classifier. Based on the conversation and user description, classify the specific issue.

CONVERSATION:
{conversation}

Top-level issue categories:
1. Hardware
2. Software
3. Password/Access
4. Network
5. Other

Analyze the issue and provide:
CATEGORY: [The most appropriate category from above]
SUBCATEGORY: [More specific subcategory]
PRIORITY: [High/Medium/Low]
REASONING: [Why you classified it this way]
"""
        
        issue_prompt = PromptTemplate(
            input_variables=["conversation"],
            template=issue_template
        )
        
        return LLMChain(llm=self.llm, prompt=issue_prompt, verbose=True)
    
    def _create_action_recommendation_chain(self):
        """Create a chain for recommending specific actions"""
        action_template = """
You are an IT support action recommender. Based on the issue and conversation, recommend specific actions.

ISSUE TYPE: {issue_type}
SUBCATEGORY: {subcategory}
CONVERSATION: {conversation}
CURRENT STAGE: {stage}

Based on this information, recommend the next best action(s) for resolving this issue.
Provide step-by-step guidance that would be appropriate for the user's technical level.

RECOMMENDED ACTIONS:
"""
        
        action_prompt = PromptTemplate(
            input_variables=["issue_type", "subcategory", "conversation", "stage"],
            template=action_template
        )
        
        return LLMChain(llm=self.llm, prompt=action_prompt, verbose=True)
    
    def plan_next_step(self, session):
        """Plan the next step in the conversation workflow"""
        try:
            # Extract information from session
            employee_name = "Unknown"
            department = "Unknown"
            role = "Unknown"
            
            if hasattr(session, 'employee_info') and session.employee_info:
                employee_name = session.employee_info.get('name', 'Unknown')
                department = session.employee_info.get('department', 'Unknown')
                role = session.employee_info.get('role', 'Unknown')
            
            issue_type = getattr(session, 'issue_type', 'Unknown')
            
            # Determine progress stage based on conversation length and content
            messages = getattr(session, 'messages', [])
            if len(messages) <= 2:
                progress_stage = "initial"
            elif len(messages) <= 6:
                progress_stage = "information_gathering"
            elif len(messages) <= 10:
                progress_stage = "troubleshooting"
            else:
                progress_stage = "resolution"
            
            # Format conversation history
            conversation_history = ""
            for msg in messages[-6:]:  # Only use the last 6 messages to keep context manageable
                role = msg.get('role', '').capitalize()
                content = msg.get('content', '')
                conversation_history += f"{role}: {content}\n\n"
            
            # Run the planning chain
            result = self.planning_chain.run(
                conversation_history=conversation_history,
                employee_name=employee_name,
                department=department,
                role=role,
                issue_type=issue_type,
                progress_stage=progress_stage
            )
            
            logger.info(f"Workflow planning result: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error in workflow planning: {str(e)}", exc_info=True)
            return "NEXT STEP: gather_info"  # Default to information gathering
    
    def classify_issue_detailed(self, conversation):
        """Classify the issue in more detail"""
        try:
            result = self.issue_chain.run(conversation=conversation)
            logger.info(f"Issue classification result: {result}")
            
            # Parse the output to extract structured information
            category = "Unknown"
            subcategory = "Unknown"
            priority = "Medium"
            
            if "CATEGORY:" in result:
                category_line = result.split("CATEGORY:")[1].split("\n")[0].strip()
                category = category_line
            
            if "SUBCATEGORY:" in result:
                subcategory_line = result.split("SUBCATEGORY:")[1].split("\n")[0].strip()
                subcategory = subcategory_line
                
            if "PRIORITY:" in result:
                priority_line = result.split("PRIORITY:")[1].split("\n")[0].strip()
                priority = priority_line
            
            return {
                "category": category,
                "subcategory": subcategory,
                "priority": priority,
                "full_response": result
            }
        except Exception as e:
            logger.error(f"Error in issue classification: {str(e)}", exc_info=True)
            return {
                "category": "Unknown",
                "subcategory": "Unknown",
                "priority": "Medium",
                "full_response": "Error in classification"
            }
    
    def recommend_actions(self, issue_type, subcategory, conversation, stage):
        """Recommend specific actions for the issue"""
        try:
            result = self.action_chain.run(
                issue_type=issue_type,
                subcategory=subcategory,
                conversation=conversation,
                stage=stage
            )
            
            logger.info(f"Action recommendation result: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error in action recommendation: {str(e)}", exc_info=True)
            return "Please gather more information about the specific issue the user is experiencing."