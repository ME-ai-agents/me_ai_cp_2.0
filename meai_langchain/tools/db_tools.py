# tools/db_tools.py
from langchain.tools import Tool
from existing.db_service import (
    find_employee_by_contact,
    get_employee_devices,
    find_agent_by_specialization,
    log_conversation_to_db
)
import logging

logger = logging.getLogger('me_agent_orchestrator')

def create_db_tools():
    """Create tools that interact with the database service"""
    
    tools = [
        Tool(
            name="find_employee_by_email",
            func=lambda email: find_employee_by_contact('email', email),
            description="Find an employee by their email address. Input should be the email address."
        ),
        
        Tool(
            name="find_employee_by_phone",
            func=lambda phone: find_employee_by_contact('phone', phone),
            description="Find an employee by their phone number. Input should be the phone number."
        ),
        
        Tool(
            name="get_employee_devices",
            func=lambda employee_id: get_employee_devices({"employee_id": employee_id}),
            description="Get devices assigned to an employee. Input should be the employee ID."
        ),
        
        Tool(
            name="find_agent_by_specialization",
            func=find_agent_by_specialization,
            description="Find an agent with a specific specialization. Input should be the specialization (e.g., 'Hardware', 'Software', 'Password')."
        ),
        
        Tool(
            name="log_conversation",
            func=lambda params: _log_conversation_helper(params),
            description="Log a conversation message to the database. Input should be a JSON string with conversation_id, user_id, agent_id, message_text, message_type, and issue_status."
        )
    ]
    
    return tools

def _log_conversation_helper(params_str):
    """Helper function to parse parameters for conversation logging"""
    try:
        import json
        params = json.loads(params_str)
        
        # Extract required parameters
        conversation_id = params.get('conversation_id')
        user_id = params.get('user_id')
        agent_id = params.get('agent_id')
        message_text = params.get('message_text')
        message_type = params.get('message_type', 'AI response')
        issue_status = params.get('issue_status', 'In Progress')
        
        # Validate required parameters
        if not all([conversation_id, user_id, agent_id, message_text]):
            return "Error: Missing required parameters. Need conversation_id, user_id, agent_id, and message_text."
        
        # Log conversation to DB
        result = log_conversation_to_db(
            conversation_id,
            user_id,
            agent_id,
            message_text,
            message_type,
            issue_status
        )
        
        if result:
            return "Successfully logged conversation message."
        else:
            return "Failed to log conversation message."
    except Exception as e:
        logger.error(f"Error logging conversation: {str(e)}")
        return f"Error logging conversation: {str(e)}"
