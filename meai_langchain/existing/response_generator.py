# response_generator.py
import logging
import requests
import time
import json
import datetime

logger = logging.getLogger('me_agent_orchestrator')

def classify_issue(message):
    """Classify issue type based on message content with improved detection"""
    # Enhanced keyword lists for better classification
    hardware_keywords = [
        'device', 'computer', 'laptop', 'desktop', 'slow', 'broken', 
        'screen', 'keyboard', 'mouse', 'printer', 'hardware', 'wifi',
        'network', 'internet', 'connection', 'battery', 'power', 'crash',
        'frozen', 'blue screen', 'bsod', 'restart', 'boot', 'monitor',
        'display', 'black screen', 'webcam', 'camera', 'microphone', 'audio',
        'sound', 'speaker', 'usb', 'drive', 'disk', 'storage'
    ]
                     
    password_keywords = [
        'password', 'login', 'forgot', 'reset', 'locked', 'account',
        'access', 'credentials', "can't log in", 'authentication',
        'username', 'locked out', 'security', 'signin', 'sign in',
        'log in', 'cannot access', 'password expired', 'change password',
        'identity', 'verification', 'two-factor', '2fa', 'mfa'
    ]
    
    software_keywords = [
        'software', 'application', 'app', 'program', 'install',
        'update', 'upgrade', 'microsoft', 'office', 'excel', 'word',
        'outlook', 'email', 'browser', 'chrome', 'edge', 'firefox',
        'safari', 'teams', 'slack', 'zoom', 'license', 'activation',
        'windows', 'macos', 'os', 'operating system', 'error message'
    ]
    
    # If the message is too short or vague, default to a generic welcome without classification
    if len(message.strip()) < 10 or message.lower() in ['hi', 'hello', 'hey', 'hi there', 'hello there', 'greetings']:
        return "General"
    
    # Count keyword matches with word boundary checks for better accuracy
    hardware_count = 0
    password_count = 0
    software_count = 0
    
    message_lower = message.lower()
    
    for word in hardware_keywords:
        if word.lower() in message_lower:
            hardware_count += 1
    
    for word in password_keywords:
        if word.lower() in message_lower:
            password_count += 1
            
    for word in software_keywords:
        if word.lower() in message_lower:
            software_count += 1
    
    # Apply weights to categories
    password_count *= 1.2  # Give priority to password issues
    
    # Add logging for debugging
    logger.info(f"Issue classification scores - Hardware: {hardware_count}, Password: {password_count}, Software: {software_count}")
    
    # Return the category with more matches
    if password_count > hardware_count and password_count > software_count:
        return "Password"
    elif software_count > hardware_count and software_count > password_count:
        return "Software"
    elif hardware_count > 0:
        return "Hardware"
    else:
        # Default to General if no clear match
        return "General"

def get_agent_prompt(issue_type, employee_info, issue_description):
    """Generate a prompt for the AI model based on issue type with enhanced prompting"""
    # Ensure employee_info is a dictionary even if None was passed
    if employee_info is None:
        employee_info = {}
    
    # Get current time for time-appropriate greetings
    current_hour = datetime.datetime.now().hour
    time_greeting = "Good morning" if 5 <= current_hour < 12 else "Good afternoon" if 12 <= current_hour < 18 else "Good evening"
    
    # Common instructions for all agent types
    common_instructions = f"""
You are ME.ai Assistant, an AI helper for enterprise IT support. {time_greeting}!

USER INFORMATION:
Name: {employee_info.get('name', 'Anonymous User')}
Department: {employee_info.get('department', 'Unknown Department')}
Role: {employee_info.get('role', 'Employee')}

The user has contacted IT support regarding: "{issue_description}"

YOUR BEHAVIOR GUIDELINES:
- Be friendly, professional, and empathetic in your responses
- If you need more information to troubleshoot, ask specific questions
- Focus on solving their problem efficiently
- Ask only one question at a time to avoid overwhelming the user
- Keep responses concise and easy to understand (around 2-3 paragraphs maximum)
- Avoid technical jargon unless the user appears technically proficient
- Address the user by their first name if available
"""

    if issue_type == "Hardware":
        return f"""{common_instructions}

YOU ARE: ME.ai TechBot, specializing in hardware and technical support.

FOR HARDWARE ISSUES:
1. Determine which specific device they're having an issue with
2. Ask about the symptoms they're experiencing (error messages, behavior)
3. Find out when the problem started and any recent changes
4. Ask if they've tried any troubleshooting steps already

ADDITIONAL INSTRUCTIONS:
- If the issue seems to be affecting multiple devices, explore potential network or account-related causes
- For critical issues (device won't start, data loss risk), prioritize immediate solutions
- Offer step-by-step instructions with clear indicators of progress
"""

    elif issue_type == "Password":
        return f"""{common_instructions}

YOU ARE: ME.ai SecurityBot, specializing in password and account issues.

FOR PASSWORD/ACCOUNT ISSUES:
1. Determine which specific system or application they're trying to access
2. Find out what specific error message they're seeing
3. Ask when they last successfully logged in
4. DO NOT ask for their current password under any circumstances

ADDITIONAL INSTRUCTIONS:
- For security reasons, NEVER ask for current passwords
- If this is a password reset request, explain the secure reset process
- If the issue involves MFA/2FA, provide guidance on backup verification methods
- Be extra clear about security protocols and why they exist
"""

    elif issue_type == "Software":
        return f"""{common_instructions}

YOU ARE: ME.ai SoftwareBot, specializing in software and application issues.

FOR SOFTWARE ISSUES:
1. Determine which application or software they're having trouble with
2. Ask about specific error messages or unexpected behaviors
3. Find out what version of the software they're using
4. Ask if the issue occurred after an update, install, or system change

ADDITIONAL INSTRUCTIONS:
- For widely used applications (Office, Teams, etc.), check if the issue is affecting other users
- Suggest alternatives if a particular application is completely unavailable
- Explain any technical terms you need to use in simple language
- For licensing issues, be clear about company policies and procedures
"""

    else:  # General or default case
        return f"""{common_instructions}

YOU ARE: ME.ai Assistant, a general IT support assistant.

FOR GENERAL SUPPORT:
1. First determine the nature of their issue (hardware, software, account, etc.)
2. Ask about specific symptoms or error messages
3. Find out when the problem started occurring
4. Ask about any troubleshooting steps they've already tried

ADDITIONAL INSTRUCTIONS:
- Be adaptable as you learn more about their specific issue
- If it's a complex issue that might need escalation, let them know that option exists
- Provide general best practices for IT hygiene where appropriate
- Check if there are any urgent aspects to their request
"""

def generate_ai_response(prompt, user_message, session, api_key, api_url):
    """Generate a response using the DeepSeek API with improved error handling and fallbacks"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Build conversation history from session
        messages = [{"role": "system", "content": prompt}]
        
        # Add conversation history (last 6 messages)
        if hasattr(session, 'messages'):
            # Filter out system messages from the history
            user_assistant_messages = [msg for msg in session.messages[-10:] if msg.get('role') in ['user', 'assistant']]
            # Take the last 6 messages
            history_messages = user_assistant_messages[-6:]
            messages.extend(history_messages)
        
        # Add current user message if not already in session
        if not hasattr(session, 'messages') or not session.messages or session.messages[-1]['role'] != 'user' or session.messages[-1]['content'] != user_message:
            messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        logger.info(f"Sending request to AI API for issue: {user_message[:50]}...")
        logger.debug(f"Full API request payload: {json.dumps(payload, indent=2)}")
        
        # Set reasonable timeout
        timeout_seconds = 30
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            
            # Log the response status and first part of the content for debugging
            logger.debug(f"API response status: {response.status_code}")
            if response.text:
                logger.debug(f"API response preview: {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    ai_message = result['choices'][0]['message']['content']
                    logger.info(f"Successfully received AI response of length {len(ai_message)}")
                    return ai_message
                except (KeyError, IndexError) as e:
                    logger.error(f"Invalid response format from AI API: {str(e)}")
                    logger.debug(f"Response content: {response.text}")
                    return generate_fallback_response(user_message, session, issue_type=getattr(session, 'issue_type', None))
            else:
                logger.error(f"Error from AI API: {response.status_code} - {response.text}")
                # Fall back to fallback response
                return generate_fallback_response(user_message, session, issue_type=getattr(session, 'issue_type', None))
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to AI API after {timeout_seconds} seconds")
            return generate_fallback_response(user_message, session, issue_type=getattr(session, 'issue_type', None), 
                                             error_type="timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error connecting to AI API: {str(e)}")
            return generate_fallback_response(user_message, session, issue_type=getattr(session, 'issue_type', None))
            
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        return "I apologize, but I'm experiencing technical difficulties. Please try again later or contact our IT support team directly at support@meai.com if your issue is urgent."

def generate_fallback_response(user_message, session, issue_type=None, error_type=None):
    """Generate a fallback response when AI API is unavailable"""
    logger.info(f"Generating fallback response. Issue type: {issue_type}, Error type: {error_type}")
    
    # Get personalized greeting if we have user information
    greeting = ""
    if hasattr(session, 'employee_info') and session.employee_info and session.employee_info.get('name'):
        first_name = session.employee_info.get('name').split()[0]
        greeting = f"Hi {first_name}, "
    else:
        greeting = "Hello, "
    
    # If we're having connectivity issues
    if error_type == "timeout":
        return f"{greeting}I apologize for the delay. Our system is experiencing some momentary slowness. Could you please provide some additional details about your issue so I can assist you better once our systems are back to normal speed?"
    
    # Simple keyword-based response generator based on issue type and message content
    message_lower = user_message.lower()
    
    # If the message appears to be a simple greeting, respond accordingly
    if message_lower in ['hi', 'hello', 'hey', 'hi there', 'hello there', 'greetings']:
        return f"{greeting}I'm ME.ai Assistant, your IT support specialist. How can I help you today?"
    
    if issue_type == "Password":
        if "reset" in message_lower or "forgot" in message_lower:
            return f"{greeting}I understand you need to reset your password. I'd be happy to help with that. For security reasons, I'll need to verify your identity first. Could you please confirm your department and employee ID?"
        
        if "locked" in message_lower or "locked out" in message_lower:
            return f"{greeting}I see that your account is locked. This typically happens after multiple incorrect password attempts. Let me help you regain access. First, could you tell me which system or application you're trying to access?"
        
        return f"{greeting}I understand you're having an issue with authentication or accessing your account. To help you better, could you specify which system or application you're having trouble accessing?"
    
    elif issue_type == "Hardware":
        if any(word in message_lower for word in ['slow', 'performance', 'freezing', 'frozen']):
            return f"{greeting}I'm sorry to hear your device is running slowly. This could be due to several factors such as low disk space, too many applications running, or outdated software. Could you tell me which operating system you're using, and approximately when you started noticing the issue?"
        
        if any(word in message_lower for word in ['printer', 'print', 'scanning']):
            return f"{greeting}I understand you're having an issue with a printer. Let me help troubleshoot that. First, could you tell me the model of the printer, and whether it's connected via network or USB?"
        
        if any(word in message_lower for word in ['wifi', 'internet', 'connection', 'network']):
            return f"{greeting}I see you're experiencing network connectivity issues. Let's try to resolve this. Are you having trouble connecting to the WiFi, or is your device connected but you can't access specific websites or services?"
        
        return f"{greeting}Thank you for reaching out about your hardware issue. To help me troubleshoot effectively, could you tell me which specific device you're having problems with, and what symptoms you're experiencing?"
    
    elif issue_type == "Software":
        if any(word in message_lower for word in ['install', 'download', 'setup']):
            return f"{greeting}I understand you need help installing software. To assist you better, could you tell me which application you're trying to install, and what error or issue you're encountering during the installation process?"
        
        if any(word in message_lower for word in ['update', 'upgrade', 'patch']):
            return f"{greeting}I see you're having issues with a software update. These can sometimes be tricky. Could you let me know which program needs updating, and what happens when you try to update it?"
        
        if any(word in message_lower for word in ['office', 'excel', 'word', 'powerpoint', 'outlook']):
            return f"{greeting}I understand you're experiencing an issue with Microsoft Office. To help you more effectively, could you specify which Office application is giving you trouble, and describe what happens when the problem occurs?"
        
        return f"{greeting}I understand you're having a software issue. To help me troubleshoot effectively, could you tell me which specific application you're having problems with, and what error messages or unexpected behaviors you're seeing?"
    
    # Default response for any other issue
    return f"{greeting}Thank you for reaching out to IT support. I'd like to help with your issue, but I need a bit more information. Could you provide more details about what you're experiencing so I can better assist you?"

def generate_initial_greeting(session):
    """Generate a personalized initial greeting based on user info"""
    current_hour = datetime.datetime.now().hour
    time_greeting = "Good morning" if 5 <= current_hour < 12 else "Good afternoon" if 12 <= current_hour < 18 else "Good evening"
    
    # Check if we have employee info
    if hasattr(session, 'employee_info') and session.employee_info:
        # Use first name for more personal greeting
        employee_name = session.employee_info.get('name', '').split()[0]
        department = session.employee_info.get('department', '')
        
        greeting = f"{time_greeting}, {employee_name}! I'm ME.ai Assistant, your IT support specialist."
        
        # Add department-specific greeting if available
        if department:
            greeting += f" I see you're from the {department} department."
        
        # Add question about issue
        greeting += " How can I help you with your IT needs today?"
    else:
        # Generic greeting for unknown users
        greeting = f"{time_greeting}! I'm ME.ai Assistant, your IT support specialist. How can I help you today?"
        
    return greeting