# session_manager.py
import logging
import datetime
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .db_service import log_conversation_to_db 

logger = logging.getLogger('me_agent_orchestrator')


        
        
class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.customer_number = None
        self.customer_email = None
        self.employee_id = None
        self.agent_id = None
        self.employee_info = None  # Store full employee info
        self.devices = []  # Store employee devices
        self.conversation_id = str(uuid.uuid4())
        self.messages = []
        self.channel_status = {'telephony': False, 'chat': False}
        self.issue_type = None
        self.created_at = datetime.datetime.now()
        self.last_updated = datetime.datetime.now()
        self.call_data = {}
        self.asked_about_devices = False  # Track if we've asked about devices
        self.selected_device = None  # Store which device the user selected
        self.initial_greeting = None  # Store the initial greeting
    
    def add_message(self, message, channel_type='chat'):
        self.messages.append(message)
        self.last_updated = datetime.datetime.now()
        
        # Log message to database if we have employee and agent IDs
        if self.employee_id and self.agent_id:
            log_conversation_to_db(
                self.conversation_id,
                self.employee_id,
                self.agent_id,
                message['content'],
                "User input" if message['role'] == 'user' else "AI response",
                "In Progress"
            )
    
    def update_channel_status(self, channel, status):
        self.channel_status[channel] = status
        self.last_updated = datetime.datetime.now()
    
    def update_call_data(self, call_data):
        self.call_data.update(call_data)
        self.last_updated = datetime.datetime.now()

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]
    
    def save_session(self, session):
        self.sessions[session.session_id] = session
        logger.info(f"Session {session.session_id} saved/updated")
    
    def end_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} ended")