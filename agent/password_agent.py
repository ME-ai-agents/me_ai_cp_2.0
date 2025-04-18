# agent/password_agent.py
from langchain.tools import Tool
from .base_agent import MeAIBaseAgent
import logging

logger = logging.getLogger('me_agent_orchestrator')

class PasswordAgent(MeAIBaseAgent):
    """Agent specializing in password and authentication issues"""
    
    def __init__(self, aws_region="us-east-1", model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__("Password", aws_region, model_id)
    
    def _get_tools(self):
        """Get password-specific tools"""
        tools = [
            Tool(
                name="get_reset_procedure",
                func=self._get_reset_procedure,
                description="Get instructions for resetting passwords for various systems. Input should be the system name."
            ),
            Tool(
                name="check_password_policy",
                func=self._check_password_policy,
                description="Get information about password policies for various systems. Input should be the system name."
            ),
            Tool(
                name="get_mfa_help",
                func=self._get_mfa_help,
                description="Get help with Multi-Factor Authentication. Input should be the system name and issue description separated by a semicolon."
            ),
            Tool(
                name="get_account_lockout_info",
                func=self._get_account_lockout_info,
                description="Get information about account lockout policies and resolution. Input should be the system name."
            )
        ]
        return tools
    
    def _get_reset_procedure(self, system_name):
        """Tool function to get password reset procedures"""
        try:
            system_name = system_name.lower()
            
            # Reset procedures database (mock)
            reset_procedures = {
                "windows": """
Password Reset Procedure for Windows Login:

1. For a standard Windows account:
   - Contact IT Helpdesk at support@meai.com
   - Provide your employee ID and complete identity verification
   - A temporary password will be provided
   - You will be prompted to change it at next login

2. Self-Service Option (if enabled):
   - On the login screen, click "I forgot my password"
   - Follow the prompts to verify your identity
   - Set a new password following our security guidelines

Note: All password resets require multi-factor authentication verification.
                """,
                
                "office 365": """
Password Reset Procedure for Office 365:

1. Self-Service Option:
   - Go to https://portal.office.com
   - Click "Can't access your account?"
   - Follow the prompts to verify your identity via phone or email
   - Create a new password following the security requirements

2. IT Support Option:
   - Contact IT Helpdesk at support@meai.com
   - Provide your employee ID and complete identity verification
   - A temporary password will be provided
   - You will be prompted to change it at next login

Note: Your Office 365 password is synchronized with your company email and Teams access.
                """,
                
                "email": """
Password Reset Procedure for Company Email:

1. Self-Service Option:
   - Go to https://mail.company.com
   - Click "Forgot password" link
   - Follow the prompts to verify your identity
   - Set a new password following our security guidelines

2. IT Support Option:
   - Contact IT Helpdesk at support@meai.com
   - Provide your employee ID and complete identity verification
   - A temporary password will be provided via SMS to your registered mobile number
   - You must change this password upon first login

Note: Your email password also affects access to other Microsoft services if you use Office 365.
                """,
                
                "vpn": """
Password Reset Procedure for VPN Access:

1. VPN passwords cannot be reset through self-service options due to security policy.

2. To reset your VPN password:
   - Contact IT Security team at security@meai.com
   - Provide your employee ID and complete enhanced identity verification
   - A temporary password will be provided via a secure channel
   - You must change this password upon first connection

Note: VPN access requires approval from your department manager for remote workers.
                """,
                
                "teams": """
Password Reset Procedure for Microsoft Teams:

1. Your Teams password is the same as your Office 365/Email password.
   - Follow the Office 365 reset procedure
   - Go to https://portal.office.com
   - Click "Can't access your account?"
   - Follow the prompts to verify your identity
   - Create a new password

2. If you continue to have issues:
   - Contact IT Helpdesk at support@meai.com
   - Specify that you're having Teams access issues
   - They will verify your account status

Note: Teams access is linked to your active directory account.
                """
            }
            
            # Generic procedure for systems not in our database
            generic_procedure = """
Generic Password Reset Procedure:

1. Self-Service Option:
   - Look for "Forgot Password" or "Reset Password" link on the login page
   - Follow the verification steps provided
   - Create a new password following the system requirements

2. IT Support Option:
   - Contact IT Helpdesk at support@meai.com
   - Provide your employee ID and the system you need access to
   - Complete identity verification
   - Follow the instructions provided by the support team

Note: Different systems have different security requirements. Always use strong, unique passwords.
            """
            
            # Look for matching system
            for system_key, procedure in reset_procedures.items():
                if system_key in system_name or system_name in system_key:
                    return procedure
            
            # If no specific match, return generic procedure
            return f"No specific reset procedure found for {system_name}. Here is our general password reset guidance:\n\n{generic_procedure}"
            
        except Exception as e:
            logger.error(f"Error getting reset procedure: {str(e)}")
            return f"Error retrieving password reset information: {str(e)}"
    
    def _check_password_policy(self, system_name):
        """Tool function to check password policies"""
        try:
            system_name = system_name.lower()
            
            # Password policies database (mock)
            password_policies = {
                "windows": """
Windows Password Policy:

- Minimum length: 12 characters
- Must include at least 3 of the following:
  * Uppercase letters (A-Z)
  * Lowercase letters (a-z)
  * Numbers (0-9)
  * Special characters (!@#$%^&*()_+)
- Cannot contain your username or parts of your full name
- Cannot reuse any of your last 5 passwords
- Expires every 90 days
- Lockout occurs after 5 failed attempts
                """,
                
                "office 365": """
Office 365 Password Policy:

- Minimum length: 12 characters
- Must include at least 3 of the following:
  * Uppercase letters (A-Z)
  * Lowercase letters (a-z)
  * Numbers (0-9)
  * Special characters (!@#$%^&*()_+)
- Cannot contain your username or email address
- Cannot reuse any of your last 5 passwords
- Expires every 90 days
- Lockout occurs after 10 failed attempts
                """,
                
                "vpn": """
VPN Password Policy:

- Minimum length: 16 characters
- Must include ALL of the following:
  * Uppercase letters (A-Z)
  * Lowercase letters (a-z)
  * Numbers (0-9)
  * Special characters (!@#$%^&*()_+)
- Cannot contain dictionary words
- Cannot contain your username or parts of your name
- Cannot reuse any of your last 10 passwords
- Expires every 60 days
- Lockout occurs after 3 failed attempts
- Requires MFA for all connections
                """,
                
                "database": """
Database Access Password Policy:

- Minimum length: 16 characters
- Must include ALL of the following:
  * Uppercase letters (A-Z)
  * Lowercase letters (a-z)
  * Numbers (0-9)
  * Special characters (!@#$%^&*()_+)
- Cannot contain dictionary words
- Cannot contain your username or parts of your name
- Cannot reuse any of your previous passwords
- Expires every 45 days
- Lockout occurs after 3 failed attempts
- Requires approval for each reset
                """
            }
            
            # Generic policy for systems not in our database
            generic_policy = """
Enterprise Standard Password Policy:

- Minimum length: 12 characters
- Must include at least 3 of the following:
  * Uppercase letters (A-Z)
  * Lowercase letters (a-z)
  * Numbers (0-9)
  * Special characters (!@#$%^&*()_+)
- Cannot contain easily guessable information (name, username, birth date)
- Cannot reuse recent passwords
- Regular password changes required (typically 90 days)
- Account lockout protection after multiple failed attempts

For specific system requirements, please contact IT Security.
            """
            
            # Look for matching system
            for system_key, policy in password_policies.items():
                if system_key in system_name or system_name in system_key:
                    return policy
            
            # If no specific match, return generic policy
            return f"No specific password policy found for {system_name}. Here is our general enterprise password policy:\n\n{generic_policy}"
            
        except Exception as e:
            logger.error(f"Error checking password policy: {str(e)}")
            return f"Error retrieving password policy information: {str(e)}"
    
    def _get_mfa_help(self, input_str):
        """Tool function to get MFA help"""
        try:
            # Parse input
            parts = input_str.split(';')
            if len(parts) != 2:
                return "Invalid input format. Please provide system name and issue description separated by a semicolon."
            
            system_name = parts[0].strip().lower()
            issue = parts[1].strip().lower()
            
            # MFA help database (mock)
            mfa_help = {
                "office 365": {
                    "setup": """
Office 365 MFA Setup:

1. Sign in to https://portal.office.com
2. Go to My Account > Security & privacy > Additional security verification
3. Choose your verification method:
   - Mobile app (recommended)
   - Text messages
   - Phone call
4. Follow the prompts to complete setup
5. Make sure to save your backup codes in a secure location
                    """,
                    "reset": """
Reset Office 365 MFA:

1. If you have access to your account but need to change MFA method:
   - Sign in to https://portal.office.com
   - Go to My Account > Security & privacy > Additional security verification
   - Update your verification methods

2. If you cannot access your account due to lost MFA device:
   - Contact IT Helpdesk at support@meai.com
   - Provide your employee ID and complete enhanced identity verification
   - IT Security will reset your MFA
   - You'll need to set up MFA again after sign-in
                    """,
                    "not working": """
Troubleshooting Office 365 MFA Issues:

1. Mobile App Not Working:
   - Check your phone's time is accurate (incorrect time can cause authentication failures)
   - Ensure you have internet connectivity
   - Try using the backup codes provided during setup
   - Reinstall the authentication app

2. Text Message/Call Not Received:
   - Verify your phone number is correct in your security settings
   - Ensure your phone has service
   - Try using backup verification method

3. Other Issues:
   - Use recovery codes if available
   - Contact IT Helpdesk at support@meai.com for assistance
                    """
                },
                "vpn": {
                    "setup": """
VPN MFA Setup:

1. Download the company-approved authenticator app:
   - Microsoft Authenticator (recommended)
   - Google Authenticator
   - Duo Mobile

2. Contact IT Security to initiate MFA setup for VPN
   - Email security@meai.com with your employee ID
   - You'll receive setup instructions via secure email
   - Follow the instructions to scan QR code and complete setup

3. Test your VPN connection with new MFA configured
                    """,
                    "reset": """
Reset VPN MFA:

VPN MFA resets require elevated security verification:

1. Submit reset request to security@meai.com
2. Include your employee ID and VPN username
3. IT Security will contact you to schedule a video verification call
4. After verification, your MFA will be reset
5. You'll need to set up MFA again following the standard procedure

Note: This process typically takes 1 business day to complete.
                    """,
                    "not working": """
Troubleshooting VPN MFA Issues:

1. Authentication App Issues:
   - Verify your phone's time and date are set to automatic
   - Check internet connectivity
   - Ensure you're using the correct account in your authenticator app

2. Connection Issues:
   - Some networks block VPN connections - try a different network
   - Ensure you're entering the correct code before it expires
   - Check VPN status at status.company.com

3. Other Issues:
   - Contact security@meai.com for assistance
   - Include screenshots of any error messages (do not include passwords)
                    """
                }
            }
            
            # Generic MFA help
            generic_mfa_help = """
General MFA Guidance:

1. Common MFA Methods:
   - Authenticator apps (Microsoft Authenticator, Google Authenticator)
   - SMS text codes
   - Email codes
   - Security keys (YubiKey, etc.)
   - Biometric verification

2. Best Practices:
   - Always set up backup verification methods
   - Save recovery codes in a secure location
   - Keep your authentication app and devices updated
   - Never share verification codes with anyone

3. Common Issues:
   - Time synchronization problems on your device
   - Network connectivity issues
   - Expired or incorrect codes
   - Device not recognized

For system-specific MFA help, contact IT Helpdesk at support@meai.com.
            """
            
            # First check for system and issue match
            if system_name in mfa_help:
                system_mfa_help = mfa_help[system_name]
                
                # Check for issue match
                for issue_key, help_text in system_mfa_help.items():
                    if issue_key in issue:
                        return help_text
                
                # If no issue match, but system match, return all help for that system
                combined_help = f"MFA Help for {system_name.title()}:\n\n"
                for issue_key, help_text in system_mfa_help.items():
                    combined_help += f"--- {issue_key.title()} ---\n{help_text}\n\n"
                return combined_help
            
            # If no match, return generic help
            return f"No specific MFA guidance found for {system_name}. Here is our general MFA guidance:\n\n{generic_mfa_help}"
            
        except Exception as e:
            logger.error(f"Error getting MFA help: {str(e)}")
            return f"Error retrieving MFA information: {str(e)}"
    
    def _get_account_lockout_info(self, system_name):
        """Tool function to get account lockout information"""
        try:
            system_name = system_name.lower()
            
            # Account lockout info database (mock)
            lockout_info = {
                "windows": """
Windows Account Lockout Information:

- Lockout Threshold: 5 failed login attempts
- Lockout Duration: 30 minutes auto-unlock
- Reset Counter: 15 minutes of inactivity

How to Resolve:
1. Wait 30 minutes for auto-unlock, or
2. Contact IT Helpdesk for immediate unlock:
   - Call: ext. 1234
   - Email: support@meai.com
   - Provide your employee ID and complete identity verification

Prevention:
- Use the "Forgot Password" option before multiple failed attempts
- Ensure Caps Lock is not enabled when typing passwords
- Don't use shared credentials
                """,
                
                "office 365": """
Office 365 Account Lockout Information:

- Lockout Threshold: 10 failed login attempts
- Lockout Duration: 24 hours auto-unlock
- Reset Counter: 1 hour of inactivity

How to Resolve:
1. Wait for auto-unlock (up to 24 hours), or
2. Contact IT Helpdesk for immediate unlock:
   - Call: ext. 1234
   - Email: support@meai.com
   - Provide your employee ID and verification
   - Specify it's an Office 365 lockout

Prevention:
- Use the password reset option after 1-2 failed attempts
- Verify you're on the legitimate Office 365 login page
- Check for browser password autofill issues
                """,
                
                "vpn": """
VPN Account Lockout Information:

- Lockout Threshold: 3 failed login attempts
- Lockout Duration: No auto-unlock (security policy)
- Reset Counter: Never (strict security)

How to Resolve:
1. Contact IT Security for unlock:
   - Email: security@meai.com
   - Call: ext. 5678 (business hours only)
   - Enhanced identity verification required
   - Manager approval may be needed

Prevention:
- Use password manager to avoid mistyping
- Ensure MFA device is accessible before attempting login
- Request password reset if unsure of credentials
                """,
                
                "salesforce": """
Salesforce Account Lockout Information:

- Lockout Threshold: 5 failed login attempts
- Lockout Duration: 15 minutes auto-unlock
- Reset Counter: 2 hours of inactivity

How to Resolve:
1. Wait 15 minutes for auto-unlock, or
2. Contact Salesforce Admin:
   - Email: salesforce.admin@meai.com
   - Provide your employee ID and username
   - Verification questions will be asked

Prevention:
- Use SSO when available rather than direct login
- Use the "Forgot Password" feature after 1-2 failed attempts
- Ensure you're logging in through the correct instance URL
                """
            }
            
            # Generic lockout info
            generic_lockout = """
General Account Lockout Information:

- Most systems lock after 3-10 failed login attempts
- Lockout duration varies by system importance:
  * Standard systems: 15-30 minutes
  * High-security systems: 24 hours or require manual unlock
- Frequent lockouts may trigger security alerts

How to Resolve Generic Lockouts:
1. Wait for the auto-unlock period to expire, or
2. Contact IT Helpdesk:
   - Call: ext. 1234
   - Email: support@meai.com
   - Provide your employee ID and system name
   - Complete identity verification

Prevention:
- Use password managers to avoid typing errors
- Click "Forgot Password" after 1-2 failed attempts
- Ensure you're using the correct username for each system
            """
            
            # Look for matching system
            for system_key, info in lockout_info.items():
                if system_key in system_name or system_name in system_key:
                    return info
            
            # If no specific match, return generic info
            return f"No specific account lockout information found for {system_name}. Here is our general account lockout guidance:\n\n{generic_lockout}"
            
        except Exception as e:
            logger.error(f"Error getting account lockout info: {str(e)}")
            return f"Error retrieving account lockout information: {str(e)}"
