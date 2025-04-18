# me_agent_orchestrator_fastapi.py
import os
import uuid
import datetime
import json
import logging
import time
import asyncio
from typing import Optional, Dict, Any, List
import requests

import uvicorn
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('me_agent_orchestrator')

# Configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-643f3d962fef49949b7a719e2dc38e83')
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')
MEAI_DB_SERVICE = os.environ.get('MEAI_DB_SERVICE', 'http://127.0.0.1:5000/api')
DB_USERNAME = os.environ.get('DB_USERNAME', 'testadmin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'testpass')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

# Import existing modules
from existing.session_manager import SessionManager, Session
from existing.db_service import (
    get_db_service_token, 
    find_employee_by_contact, 
    get_employee_devices,
    log_conversation_to_db
)
from existing.response_generator import (
    classify_issue, 
    generate_initial_greeting,
    generate_fallback_response
)

# Import LangChain components
from agent.orchestrator import AgentOrchestrator

# Initialize FastAPI app
app = FastAPI(
    title="ME.ai Agent Orchestrator",
    description="An AI agent orchestrator for IT support using LangChain",
    version="1.0.0"
)

# Initialize session manager
session_manager = SessionManager()

# Initialize agent orchestrator
agent_orchestrator = AgentOrchestrator(
    aws_region=AWS_REGION,
    model_id=BEDROCK_MODEL_ID
)

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class TelephonyRequest(BaseModel):
    call: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    phone: Optional[str] = None
    message: Optional[str] = None
    messages: Optional[List[Message]] = None
    model: Optional[str] = None

class TeamsRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    framework: str
    version: str

def generate_response(message, session, channel_type):
    """Generate a response for the user using LangChain agents"""
    try:
        # If no employee ID yet, try to identify the user
        if not session.employee_id:
            if channel_type == 'telephony' and session.customer_number:
                employee = find_employee_by_contact('phone', session.customer_number)
            elif channel_type == 'chat' and session.customer_email:
                employee = find_employee_by_contact('email', session.customer_email)
            # Fallback to phone for chat if email didn't work
            elif channel_type == 'chat' and session.customer_number:
                employee = find_employee_by_contact('phone', session.customer_number)
            else:
                employee = None
            
            if employee:
                session.employee_id = employee.get('employee_id')
                logger.info(f"Identified employee: {employee.get('name')} ({session.employee_id})")
                
                # Get employee devices
                devices = get_employee_devices(employee)
                session.devices = devices
                logger.info(f"Found {len(devices)} devices for employee")
                
                # Store employee info in session
                session.employee_info = employee
        
        # Handle initial greeting/welcome message
        if len(session.messages) <= 1 and any(greeting in message.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
            greeting = agent_orchestrator.get_initial_greeting(session)
            logger.info(f"Generated initial greeting: {greeting[:50]}...")
            return greeting
        
        # Process message with the agent orchestrator
        ai_response = agent_orchestrator.process_query(message, session)
        
        # Log AI response to DB
        if session.employee_id and session.agent_id:
            try:
                log_conversation_to_db(
                    getattr(session, 'conversation_id', str(uuid.uuid4())),
                    session.employee_id,
                    session.agent_id,
                    ai_response,
                    "AI response",
                    "In Progress"
                )
                logger.info("Successfully logged AI response to database")
            except Exception as e:
                logger.error(f"Error logging AI response to database: {str(e)}")
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our IT support team directly if your issue is urgent."
        
        
@app.post("/telephony/chat/completions")
async def handle_telephony_chat(request: TelephonyRequest):
    """Handle telephony chat completions with streaming support"""
    try:
        # Extract key information
        call_data = request.call or {}
        session_id = call_data.get('id') or request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
        
        customer_data = call_data.get('customer', {})
        customer_number = customer_data.get('number') or request.phone
        user_message = request.message or ""
        
        # For messages sent in array format
        if not user_message and request.messages:
            for msg in request.messages:
                if msg.role == 'user' and msg.content:
                    user_message = msg.content
                    logger.info(f"Found user message: '{user_message}'")
                    break
        
        # Get or create session
        session = session_manager.get_session(session_id)
        
        # Update session info
        if customer_number:
            session.customer_number = customer_number
            session.update_channel_status('telephony', True)
        
        # Define a streaming response generator in the format your telephony system expects
        async def stream_response():
            try:
                # Determine the response content
                if not user_message:
                    # Generate a personalized greeting
                    response = agent_orchestrator.get_initial_greeting(session)
                    logger.info(f"Generated greeting: '{response[:30]}...'")
                    
                    # Add to conversation history
                    session.add_message({
                        "role": "assistant",
                        "content": response
                    }, 'telephony')
                    logger.info("Added greeting to conversation history")
                else:
                    # Add user message to session
                    session.add_message({
                        "role": "user",
                        "content": user_message
                    }, 'telephony')
                    
                    # Generate response using LangChain agent orchestrator
                    response = generate_response(user_message, session, 'telephony')
                    
                    # Add bot message to session
                    session.add_message({
                        "role": "assistant",
                        "content": response
                    }, 'telephony')
                
                # Save session
                session_manager.save_session(session)
                
                # Check for end call trigger
                if user_message:
                    should_end_call = any(word in user_message.lower() for word in ['end', 'bye', 'goodbye', 'quit'])
                    if should_end_call:
                        session.update_channel_status('telephony', False)
                        session_manager.save_session(session)
                
                # Format the response in the exact format expected by the telephony system
                yield f"data: {json.dumps({
                    'id': f'chatcmpl-{int(time.time())}',
                    'choices': [{'delta': {'content': response}}]
                })}\n\n"
                
                # End the stream
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in telephony response stream: {str(e)}")
                yield f"data: {json.dumps({
                    'id': session_id,
                    'choices': [{'delta': {'content': 'I apologize, but there was an error processing your request.'}}]
                })}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Error handling telephony chat: {str(e)}", exc_info=True)
        
        async def error_stream():
            yield f"data: {json.dumps({
                'id': f'chatcmpl-{int(time.time())}',
                'choices': [{'delta': {'content': 'I apologize, but I\'m having trouble processing your request. Could you please try again?'}}]
            })}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

@app.post("/teams/chat/completions")
async def handle_teams_chat(request: Request):
    """Handle Teams chat messages with enhanced debugging"""
    try:
        # Get raw request data
        body = await request.body()
        body_str = body.decode()
        logger.info(f"Teams request body: {body_str[:100]}...")
        
        # Parse request data
        data = await request.json()
        logger.info(f"Teams request data keys: {list(data.keys())}")
        
        # Extract key information with fallbacks
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Try multiple possible locations for the message
        message = data.get('message', '')
        if not message and 'text' in data:
            message = data.get('text', '')
        if not message and 'content' in data:
            message = data.get('content', '')
        if not message and 'messages' in data:
            messages = data.get('messages', [])
            if messages and isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        message = msg.get('content', '')
                        if message:
                            break
        
        # Extract user identifier with fallbacks
        user_email = data.get('email', '')
        if not user_email and 'from' in data:
            user_from = data.get('from', {})
            if isinstance(user_from, dict):
                user_email = user_from.get('email', '')
        
        user_phone = data.get('phone', '')
        
        logger.info(f"Teams processed: session_id={session_id}, message='{message}', email={user_email}")
        
        # Get or create session
        session = session_manager.get_session(session_id)
        
        # Update session info
        if user_email:
            session.customer_email = user_email
        if user_phone:
            session.customer_number = user_phone
            
        # Try to identify the user if not yet identified
        if not session.employee_id:
            employee = None
            # Try email first for Teams
            if user_email:
                employee = find_employee_by_contact('email', user_email)
                if employee:
                    logger.info(f"Identified Teams user by email: {employee.get('name')}")
            
            # Try phone if email didn't work
            if not employee and user_phone:
                employee = find_employee_by_contact('phone', user_phone)
                if employee:
                    logger.info(f"Identified Teams user by phone: {employee.get('name')}")
            
            # Update session with employee info if found
            if employee:
                session.employee_id = employee.get('employee_id')
                session.employee_info = employee
                logger.info(f"Identified employee: {employee.get('name')} ({session.employee_id})")
                
                # Get employee devices
                devices = get_employee_devices(employee)
                session.devices = devices
                logger.info(f"Found {len(devices)} devices for employee")
        
        # Process message if provided
        if message:
            logger.info(f"Processing Teams message: '{message}'")
            
            # Add user message to session
            session.add_message({
                "role": "user",
                "content": message
            }, 'teams')
            
            # Generate response using agent orchestrator
            bot_response = generate_response(message, session, 'teams')
            logger.info(f"Generated Teams response: '{bot_response[:50]}...'")
            
            # Add bot response to session
            session.add_message({
                "role": "assistant",
                "content": bot_response
            }, 'teams')
            
            # Save session
            session_manager.save_session(session)
            
            # Try different response formats based on examining the request
            if 'channel' in data and data.get('channel') == 'teams':
                # Likely using the MS Teams connector format
                return {"text": bot_response, "type": "message"}
            else:
                # Use streaming response format
                async def stream_response():
                    yield f"data: {json.dumps({
                        'id': session_id,
                        'choices': [{'delta': {'content': bot_response}}]
                    })}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    }
                )
        else:
            # No message provided, return a greeting
            greeting = agent_orchestrator.get_initial_greeting(session)
            logger.info(f"No message, sending greeting: '{greeting[:50]}...'")
            
            # Add greeting to session
            session.add_message({
                "role": "assistant",
                "content": greeting
            }, 'teams')
            
            # Save session
            session_manager.save_session(session)
            
            # Try different response formats
            if 'channel' in data and data.get('channel') == 'teams':
                # Likely using the MS Teams connector format
                return {"text": greeting, "type": "message"}
            else:
                # Use streaming response format
                async def stream_response():
                    yield f"data: {json.dumps({
                        'id': session_id,
                        'choices': [{'delta': {'content': greeting}}]
                    })}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    }
                )
            
    except Exception as e:
        logger.error(f"Error handling Teams chat: {str(e)}", exc_info=True)
        
        # Return format that Teams is most likely to accept
        return {"text": "I apologize, but I'm having trouble processing your request. Please try again.", "type": "message"}


@app.post("/debug/teams")
async def debug_teams(request: Request):
    """Debug endpoint to log Teams request format"""
    data = await request.json()
    logger.info(f"DEBUG TEAMS REQUEST: {json.dumps(data)}")
    return {"status": "logged"}
    

@app.post("/debug/log_request")
async def debug_log_request(request: Request):
    """Debug endpoint to log the exact request structure"""
    try:
        # Get raw body for logging
        body = await request.body()
        body_str = body.decode()
        
        # Extract headers for additional context
        headers = dict(request.headers)
        
        # Log everything
        logger.info(f"DEBUG REQUEST PATH: {request.url.path}")
        logger.info(f"DEBUG REQUEST HEADERS: {json.dumps(headers)}")
        logger.info(f"DEBUG REQUEST BODY: {body_str}")
        
        # Try to parse as JSON if possible
        try:
            data = json.loads(body_str)
            logger.info(f"DEBUG REQUEST JSON: {json.dumps(data, indent=2)}")
        except:
            logger.info("DEBUG REQUEST BODY is not valid JSON")
        
        return {"status": "request logged"}
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}
@app.post("/webhook")
async def handle_webhook(request: Request):
    """Legacy webhook handler - redirects to telephony handler"""
    body = await request.json()
    return await handle_telephony_chat(TelephonyRequest(**body))

@app.post("/webhook/chat/completions")
async def handle_chat(request: Request):
    """Legacy chat completions handler - redirects to the appropriate channel"""
    body = await request.json()
    channel = body.get('channel', 'chat')
    
    if channel == 'teams':
        return await handle_teams_chat(TeamsRequest(**body))
    else:
        return await handle_telephony_chat(TelephonyRequest(**body))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ME.ai Agent Orchestrator",
        "framework": "FastAPI",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("me_agent_orchestrator_fastapi:app", host="127.0.0.1", port=8000, reload=True)