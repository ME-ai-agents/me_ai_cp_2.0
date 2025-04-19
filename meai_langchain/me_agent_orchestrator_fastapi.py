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
    """Generate a response for the user using direct issue handling"""
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
        
        # Direct response handling based on message content
        message_lower = message.lower()
        
        # USB issues
        if "usb" in message_lower:
            return """
I see you're having USB connection issues. Let's try these troubleshooting steps:

1. Restart your laptop - Sometimes a simple restart can resolve driver issues.

2. Try different USB ports - If one port isn't working, try another one.

3. Check Device Manager:
   - Right-click Start and select "Device Manager"
   - Look for any devices with yellow warning icons
   - If you see "Unknown USB Device" or similar, right-click and select "Update driver"

4. Reinstall USB drivers:
   - In Device Manager, expand "Universal Serial Bus controllers"
   - Right-click each USB controller and select "Uninstall device"
   - After uninstalling all, restart your laptop
   - Windows will automatically reinstall the drivers

5. Try the device on another computer if possible to determine if the issue is with the device or your laptop.

Let me know which of these steps you've tried or if any of them help resolve the issue.
"""
        
        # Speaker/audio issues
        elif any(word in message_lower for word in ["speaker", "audio", "sound"]):
            return """
I understand you're having laptop speaker issues. Let's quickly troubleshoot:

1. Check volume settings:
   - Make sure your volume isn't muted (check the speaker icon in taskbar)
   - Try pressing volume up keys on your keyboard

2. Test with headphones:
   - If you have headphones available, plug them in to see if sound works
   - This will help determine if it's a hardware or software issue

3. Check audio devices:
   - Right-click on the volume icon in taskbar
   - Select "Open Sound settings" 
   - Make sure the correct output device is selected

4. Quick restart of audio services:
   - Right-click Start menu > Run
   - Type "services.msc" and press Enter
   - Find "Windows Audio" service
   - Right-click it and select "Restart"

5. Run Windows troubleshooter:
   - Right-click volume icon in taskbar
   - Select "Troubleshoot sound problems"
   - Follow the on-screen instructions

Let me know which steps you've tried or if any of them helped.
"""
        
        # Slow laptop issues
        elif any(word in message_lower for word in ["slow", "speed", "performance", "fast"]):
            return """
Here are some steps to help speed up your slow laptop:

1. Close unnecessary programs:
   - Check your system tray and close programs you're not using
   - Use Task Manager (Ctrl+Shift+Esc) to identify resource-heavy applications

2. Clean up your disk:
   - Run Disk Cleanup: Search for "Disk Cleanup" in Start menu
   - Delete temporary files, downloaded program files, and empty recycle bin
   - Consider using CCleaner for a more thorough cleanup

3. Check startup programs:
   - Open Task Manager > Startup tab
   - Disable programs you don't need to start automatically

4. Update Windows and drivers:
   - Check for Windows updates
   - Update your graphics and other device drivers

5. Scan for malware:
   - Run a full system scan with Windows Defender or your antivirus

6. Optimize power settings:
   - Set power plan to "High performance" if you're plugged in

7. Consider hardware upgrades if possible:
   - Adding more RAM or switching to an SSD can dramatically improve performance

Let me know if you need more specific guidance on any of these steps.
"""
        
        # Printer issues
        elif "printer" in message_lower:
            return """
Here's how to troubleshoot your printer connection issues:

1. First, let's check the basics:
   - Make sure the printer is powered on and has paper
   - Check all cable connections (USB or network cables)
   - For wireless printers, verify it's connected to the same network as your computer

2. Restart both devices:
   - Turn off the printer, wait 30 seconds, then turn it back on
   - Restart your computer

3. Check printer status:
   - On Windows: Go to Settings > Devices > Printers & scanners
   - Make sure your printer is listed and not showing any errors
   - If it shows as offline, right-click and select "Use printer online"

4. Update or reinstall printer drivers:
   - Download the latest driver from the manufacturer's website
   - Uninstall the current printer software, then install the new drivers

5. Run the printer troubleshooter:
   - Go to Settings > Update & Security > Troubleshoot
   - Select "Printer" and follow the steps

6. Check for Windows updates:
   - Sometimes Windows updates can fix printer connection issues

Let me know which of these steps you've tried or if you need more specific guidance.
"""
        
        # Password issues
        elif any(word in message_lower for word in ["password", "login", "account", "forgot", "reset"]):
            return """
I understand you're having issues with your password or account access. Here's how I can help:

1. Which system are you trying to access? (Windows login, email, company applications, etc.)

2. For Windows password reset:
   - You'll need to contact IT support directly at extension 1234
   - They'll verify your identity and provide a temporary password
   - You'll need to change it at first login

3. For company application passwords:
   - Most company apps have a "Forgot Password" option on the login screen
   - This will send a reset link to your registered email
   - If that doesn't work, contact IT support

4. For email password issues:
   - Use the password reset option on the email login page
   - If you can't access your recovery email, contact IT support

5. Security reminder:
   - Never share your password with anyone, including IT staff
   - Create strong, unique passwords for different systems
   - Consider using a password manager

For urgent password resets, please contact IT Support at extension 1234 or support@company.com.
"""

        # Network issues
        elif any(word in message_lower for word in ["network", "wifi", "internet", "connection"]):
            return """
Here are troubleshooting steps for your network connection issues:

1. Basic connectivity checks:
   - Check if other devices can connect to the same network
   - Look at the WiFi icon in your taskbar - are you connected?

2. Restart networking equipment:
   - Turn off your laptop's WiFi (airplane mode on/off)
   - Restart your router/access point if possible

3. Run Windows network diagnostics:
   - Right-click on the network icon in the taskbar
   - Select "Troubleshoot problems"

4. Reset network settings:
   - Open Settings > Network & Internet > Status
   - Click "Network reset" at the bottom

5. Update network drivers:
   - Open Device Manager (right-click Start menu)
   - Expand "Network adapters"
   - Right-click your WiFi adapter and select "Update driver"

6. Check for interference:
   - Move closer to the WiFi router/access point
   - Try connecting in a different location

7. For wired connections:
   - Try a different Ethernet cable
   - Try a different port on the router

Let me know which of these steps you've tried and if they help resolve your issue.
"""
        
        # Email issues
        elif any(word in message_lower for word in ["email", "outlook", "gmail", "mail"]):
            return """
Let's troubleshoot your email issues:

1. Check your internet connection:
   - Make sure you're connected to the internet
   - Try loading a website to confirm connectivity

2. For Outlook issues:
   - Restart Outlook
   - Check if you're in Offline mode (Look for "Working Offline" in the status bar)
   - Run Outlook in Safe Mode (hold Ctrl while opening Outlook)
   - Repair Office: Control Panel > Programs > Uninstall > Office > Change > Repair

3. Check email server status:
   - Ask colleagues if they're experiencing similar issues
   - Check company IT status page if available

4. Verify account settings:
   - In Outlook: File > Account Settings
   - Confirm server settings are correct

5. Clear cache (Outlook):
   - Close Outlook
   - Search for %localappdata%\\Microsoft\\Outlook
   - Rename the .ost or .pst file (add .old to the end)
   - Restart Outlook to create a new file

6. If emails are sending but not receiving:
   - Check your junk/spam folder
   - Check mail rules that might be filtering messages

Let me know which email client you're using and what specific issues you're experiencing for more targeted help.
"""

        # Try using the agent orchestrator for other queries
        else:
            try:
                # Fall back to issue type classifier and direct response
                issue_type = classify_issue(message)
                
                if issue_type == "Hardware":
                    return """
I understand you're experiencing hardware issues. To help you better, I need some specific information:

1. What device are you having trouble with? (laptop, desktop, printer, etc.)
2. What are the specific symptoms you're experiencing?
3. When did this issue start?
4. Have you made any recent changes to your system?

Once you provide these details, I can give you targeted troubleshooting steps.
"""
                elif issue_type == "Software":
                    return """
I understand you're experiencing software issues. To help you better, I need some specific information:

1. Which application or program is causing problems?
2. What operating system are you using?
3. What exactly happens when you try to use the software?
4. Have you installed any updates recently?

Once you provide these details, I can give you targeted troubleshooting steps.
"""
                elif issue_type == "Password":
                    return """
I understand you're having password or account access issues. To help you better, could you tell me:

1. Which system or application are you trying to access?
2. What happens when you try to log in?
3. Have you tried resetting your password yet?

I can then provide specific guidance for your situation.
"""
                else:
                    # Process message with the agent orchestrator for general inquiries
                    # This will likely fail, but we'll handle that in the except block
                    ai_response = agent_orchestrator.process_query(message, session)
                    return ai_response
            except Exception as e:
                logger.error(f"Error processing with agent orchestrator: {str(e)}")
                # Return a generic but helpful response
                return f"""
I'd be happy to help you with your IT issue. To assist you better, could you provide more specific details about what you're experiencing? Some helpful information would be:

1. What specific device or software is involved?
2. What exactly happens when the issue occurs?
3. When did you first notice this problem?
4. Have you already tried any troubleshooting steps?

The more details you can provide, the better I can assist you in resolving your issue promptly.
"""
        
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
        return """
I apologize, but I'm experiencing technical difficulties. Please try one of these options:

1. Try describing your issue again with more details
2. For urgent IT support, please call the helpdesk at extension 1234
3. Email support@company.com with details of your issue
"""


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