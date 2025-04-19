# db_service.py
import logging
import requests
import os

logger = logging.getLogger('me_agent_orchestrator')

# Global token cache
db_service_token = None

def get_db_service_token():
    """Get a token for communicating with the ME.ai DB Service"""
    global db_service_token
    
    # If we already have a token, return it
    if db_service_token:
        return db_service_token
    
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        # Login to DB service
        response = requests.post(
            f"{db_service_url}/login",
            json={
                "username": "testadmin", 
                "password": "testpass"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            db_service_token = data.get('token')
            logger.info(f"Successfully obtained DB service token")
            return db_service_token
        else:
            logger.error(f"Failed to get DB service token: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting DB service token: {str(e)}")
        return None

def find_employee_by_contact(contact_type, contact_value):
    """Find an employee by contact info with enhanced phone and email matching"""
    token = get_db_service_token()
    if not token:
        logger.error("Failed to get token for DB service")
        return None
    
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        if contact_type == 'phone':
            # Log the phone number search attempt
            logger.info(f"Searching for employee with phone: {contact_value}")
            
            # Better normalize phone for comparison (handle international formats)
            normalized_search = ''.join(c for c in contact_value if c.isdigit() or c == '+')
            
            # For international numbers that start with +, also try without the + as some systems strip it
            alternative_search = normalized_search.replace('+', '') if normalized_search.startswith('+') else None
            
            # Get all employees
            response = requests.get(
                f"{db_service_url}/employees",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                employees = response.json()
                logger.info(f"Found {len(employees)} employees in database")
                
                # First try exact match with both forms
                for employee in employees:
                    if employee.get('phone'):
                        normalized_employee_phone = ''.join(c for c in employee.get('phone') if c.isdigit() or c == '+')
                        if (normalized_employee_phone == normalized_search or 
                            (alternative_search and normalized_employee_phone.replace('+', '') == alternative_search)):
                            logger.info(f"Exact phone match for employee: {employee.get('name')}")
                            return employee
                
                # Then try partial match (number is contained or last digits match)
                for employee in employees:
                    if employee.get('phone'):
                        normalized_employee_phone = ''.join(c for c in employee.get('phone') if c.isdigit() or c == '+')
                        # Compare last 8 digits if both numbers are long enough
                        emp_last_digits = normalized_employee_phone[-8:] if len(normalized_employee_phone) >= 8 else normalized_employee_phone
                        search_last_digits = normalized_search[-8:] if len(normalized_search) >= 8 else normalized_search
                        
                        if (normalized_search in normalized_employee_phone or 
                            normalized_employee_phone in normalized_search or
                            emp_last_digits == search_last_digits):
                            logger.info(f"Partial phone match for employee: {employee.get('name')}")
                            return employee
                            
                logger.warning(f"No employee found with phone: {contact_value}")
                return None
            else:
                logger.error(f"Error retrieving employees: {response.text}")
                return None
        
        elif contact_type == 'email':
            # Improved email matching - case insensitive and partial match
            logger.info(f"Searching for employee with email: {contact_value}")
            
            # Get all employees and filter by email
            response = requests.get(
                f"{db_service_url}/employees",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                employees = response.json()
                
                # First try exact match (case insensitive)
                for employee in employees:
                    if employee.get('email') and contact_value.lower() == employee.get('email').lower():
                        logger.info(f"Exact email match for employee: {employee.get('name')}")
                        return employee
                
                # Then try partial match
                for employee in employees:
                    if employee.get('email') and contact_value.lower() in employee.get('email').lower():
                        logger.info(f"Partial email match for employee: {employee.get('name')}")
                        return employee
                
                logger.warning(f"No employee found with email: {contact_value}")
            else:
                logger.error(f"Error retrieving employees: {response.text}")
            
            return None
            
        elif contact_type == 'id':
            # Get specific employee by ID
            response = requests.get(
                f"{db_service_url}/employees/{contact_value}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                employee = response.json()
                logger.info(f"Found employee by ID: {employee.get('name')}")
                return employee
            else:
                logger.error(f"Error retrieving employee by ID: {response.text}")
                return None
        else:
            logger.error(f"Unsupported contact type: {contact_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error finding employee: {str(e)}")
        return None

def get_employee_devices(employee_info):
    """Get devices for an employee"""
    token = get_db_service_token()
    if not token:
        logger.error("Failed to get token for DB service")
        return []
    
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        # Get devices by employee ID
        if employee_info and employee_info.get('employee_id'):
            response = requests.get(
                f"{db_service_url}/devices",
                headers={"Authorization": f"Bearer {token}"},
                params={"employee_id": employee_info.get('employee_id')}
            )
            
            if response.status_code == 200:
                devices = response.json()
                logger.info(f"Found {len(devices)} devices for employee {employee_info.get('name')}")
                return devices
            else:
                logger.error(f"Error retrieving devices: {response.text}")
                return []
        else:
            logger.warning("No employee ID provided to fetch devices")
            return []
            
    except Exception as e:
        logger.error(f"Error getting employee devices: {str(e)}")
        return []

def find_agent_by_specialization(specialization):
    """Find an agent with the given specialization"""
    token = get_db_service_token()
    if not token:
        logger.error("Failed to get token for DB service")
        return None
        
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        # Get all agents
        response = requests.get(
            f"{db_service_url}/agents",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            logger.error(f"Error retrieving agents: {response.text}")
            return None
            
        agents = response.json()
        
        # Find first active agent with matching specialization
        for agent in agents:
            if agent.get('status') == 'Active' and agent.get('specialization') == specialization:
                logger.info(f"Found agent for specialization {specialization}: {agent.get('agent_name')}")
                return agent
                
        # If no exact match, return first active agent
        for agent in agents:
            if agent.get('status') == 'Active':
                logger.info(f"No specialized agent found, using: {agent.get('agent_name')}")
                return agent
                
        logger.warning(f"No active agent found")
        return None
    except Exception as e:
        logger.error(f"Error finding agent: {str(e)}")
        return None

def log_conversation_to_db(conversation_id, user_id, agent_id, message_text, message_type, issue_status):
    """Log a conversation message to the database"""
    token = get_db_service_token()
    if not token:
        logger.error("Failed to get token for conversation logging")
        return None
        
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        conversation_data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "message_text": message_text,
            "message_type": message_type,
            "issue_status": issue_status
        }
        
        response = requests.post(
            f"{db_service_url}/conversations",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=conversation_data
        )
        
        if response.status_code == 201:
            logger.info(f"Successfully logged conversation message")
            return response.json()
        else:
            logger.error(f"Failed to log conversation: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error logging conversation: {str(e)}")
        return None
        
# Add this function to your db_service.py file

def get_employee_devices(employee_info):
    """Get devices for an employee"""
    token = get_db_service_token()
    if not token:
        logger.error("Failed to get token for DB service")
        return []
    
    try:
        # Use local development server
        db_service_url = "http://127.0.0.1:5000/api"
        
        # Get devices by employee ID
        if employee_info and employee_info.get('employee_id'):
            response = requests.get(
                f"{db_service_url}/devices",
                headers={"Authorization": f"Bearer {token}"},
                params={"employee_id": employee_info.get('employee_id')}
            )
            
            if response.status_code == 200:
                devices = response.json()
                logger.info(f"Found {len(devices)} devices for employee {employee_info.get('name')}")
                return devices
            else:
                logger.error(f"Error retrieving devices: {response.text}")
                return []
        else:
            logger.warning("No employee ID provided to fetch devices")
            return []
            
    except Exception as e:
        logger.error(f"Error getting employee devices: {str(e)}")
        return []