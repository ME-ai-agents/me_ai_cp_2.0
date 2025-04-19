# meai_enhanced_fastapi.py
import os
import logging
import json
import time
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import ME.ai enhanced orchestrator
from me_ai_integration import MEAIEnhancedOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meai_enhanced_api')

# Load configuration
load_dotenv()

# Initialize configuration
config = {
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

# Initialize FastAPI app
app = FastAPI(
    title="ME.ai Enhanced Agent Orchestrator",
    description="An enhanced AI agent orchestrator for IT support using RAG, KG, and Semantic Profiles",
    version="2.0.0"
)

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


# Initialize the enhanced orchestrator
orchestrator = MEAIEnhancedOrchestrator(config)

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    messages: Optional[List[Message]] = None
    channel: Optional[str] = "general"

class TelephonyRequest(BaseModel):
    call: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    phone: Optional[str] = None
    message: Optional[str] = None
    messages: Optional[List[Message]] = None

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

@app.post("/api/chat/completions")
async def handle_chat(request: ChatRequest):
    """Handle chat completions from any channel"""
    try:
        # Extract key information
        session_id = request.session_id or str(uuid.uuid4())
        user_message = request.message or ""
        user_email = request.email
        user_phone = request.phone
        channel = request.channel or "general"
        
        # For messages sent in array format
        if not user_message and request.messages:
            for msg in request.messages:
                if msg.role == 'user' and msg.content:
                    user_message = msg.content
                    logger.info(f"Found user message: '{user_message}'")
                    break
        
        logger.info(f"Processing {channel} message: '{user_message[:50]}...' (session: {session_id})")
        
        # Use the enhanced orchestrator to process the message
        response = orchestrator.process_message(
            user_message,
            session_id,
            user_email,
            user_phone
        )
        
        # Return a streaming response (compatible with OpenAI format)
        async def stream_response():
            yield f"data: {json.dumps({
                'id': f'chatcmpl-{int(time.time())}',
                'choices': [{'delta': {'content': response}}]
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
        logger.error(f"Error handling chat: {str(e)}", exc_info=True)
        
        async def error_stream():
            yield f"data: {json.dumps({
                'id': f'chatcmpl-{int(time.time())}',
                'choices': [{'delta': {'content': 'I apologize, but I\'m having trouble processing your request. Please try again.'}}]
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

# @app.post("/telephony/chat/completions")
# async def handle_telephony_chat(request: TelephonyRequest):
#     """Handle telephony chat completions"""
#     try:
#         # Extract key information
#         call_data = request.call or {}
#         session_id = call_data.get('id') or request.session_id
#         if not session_id:
#             session_id = str(uuid.uuid4())
        
#         customer_data = call_data.get('customer', {})
#         customer_number = customer_data.get('number') or request.phone
#         user_message = request.message or ""
        
#         # For messages sent in array format
#         if not user_message and request.messages:
#             for msg in request.messages:
#                 if msg.role == 'user' and msg.content:
#                     user_message = msg.content
#                     break
        
#         logger.info(f"Processing telephony message: '{user_message[:50]}...' (session: {session_id})")
        
#         # Use the enhanced orchestrator to process the message
#         response = orchestrator.process_message(
#             user_message,
#             session_id,
#             None,  # No email for telephony
#             customer_number
#         )
        
#         # Return a streaming response
#         async def stream_response():
#             yield f"data: {json.dumps({
#                 'id': f'chatcmpl-{int(time.time())}',
#                 'choices': [{'delta': {'content': response}}]
#             })}\n\n"
#             yield "data: [DONE]\n\n"
        
#         return StreamingResponse(
#             stream_response(),
#             media_type="text/event-stream",
#             headers={
#                 'Cache-Control': 'no-cache',
#                 'X-Accel-Buffering': 'no'
#             }
#         )
    
#     except Exception as e:
#         logger.error(f"Error handling telephony chat: {str(e)}", exc_info=True)
        
#         async def error_stream():
#             yield f"data: {json.dumps({
#                 'id': f'chatcmpl-{int(time.time())}',
#                 'choices': [{'delta': {'content': 'I apologize, but I\'m having trouble processing your request. Please try again.'}}]
#             })}\n\n"
#             yield "data: [DONE]\n\n"
        
#         return StreamingResponse(
#             error_stream(),
#             media_type="text/event-stream",
#             headers={
#                 'Cache-Control': 'no-cache',
#                 'X-Accel-Buffering': 'no'
#             }
#         )




# @app.post("/teams/chat/completions")
# async def handle_teams_chat(request: TeamsRequest):
#     """Handle Teams chat messages with support for both JSON and event stream formats"""
#     try:
#         # Extract key information
#         session_id = request.session_id or str(uuid.uuid4())
#         user_message = request.message or ""
#         user_email = request.email
#         user_phone = request.phone
        
#         logger.info(f"Processing Teams message: '{user_message[:50]}...' (session: {session_id})")
        
        
#         # Use the orchestrator to process the message
#         response = orchestrator.process_message(
#             user_message,
#             session_id,
#             user_email,
#             user_phone
#         )
        
        
#         # Return in Teams-compatible format - directly as JSON
#         # This format works with both the Teams client types you're using
#         return {
#             "text": response, 
#             "type": "message",
#             # Include the full response data in case the client needs it
#             "reply": response,
#             "id": f"chatcmpl-{int(time.time())}",
#             "choices": [{"delta": {"content": response}}]
#         }
    
#     except Exception as e:
#         logger.error(f"Error handling Teams chat: {str(e)}", exc_info=True)
#         return {
#             "text": "I apologize, but I'm having trouble processing your request. Please try again.", 
#             "type": "message"
#         }
    

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
                    #response = generate_response(user_message, session, 'telephony')

                    response = orchestrator.process_message(user_message,session_id,None,customer_number
                    )
                    
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













@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ME.ai Enhanced Agent Orchestrator",
        "framework": "FastAPI",
        "version": "2.0.0",
        "components": {
            "rag": orchestrator.knowledge_base is not None,
            "kg": orchestrator.ontology_manager is not None,
            "semantic_profiles": orchestrator.profile_manager is not None
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("meai_enhanced_fastapi:app", host="0.0.0.0", port=port, reload=False)