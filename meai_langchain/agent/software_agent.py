# agent/software_agent.py
from langchain.tools import Tool
from .base_agent import MeAIBaseAgent
import logging

logger = logging.getLogger('me_agent_orchestrator')

class SoftwareAgent(MeAIBaseAgent):
    """Agent specializing in software issues"""
    
    def __init__(self, aws_region="us-east-1", model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__("Software", aws_region, model_id)
    
    def _get_tools(self):
        """Get software-specific tools"""
        tools = [
            Tool(
                name="get_software_info",
                func=self._get_software_info,
                description="Get information about standard enterprise software. Input should be the software name."
            ),
            Tool(
                name="troubleshoot_software",
                func=self._troubleshoot_software,
                description="Get troubleshooting steps for common software issues. Input should be software name and issue description separated by a semicolon."
            ),
            Tool(
                name="check_software_compatibility",
                func=self._check_software_compatibility,
                description="Check compatibility between software and operating systems. Input should be software name and OS separated by a semicolon."
            ),
            Tool(
                name="get_software_alternatives",
                func=self._get_software_alternatives,
                description="Get alternative software options. Input should be the software name."
            )
        ]
        return tools
    
    def _get_software_info(self, software_name):
        """Tool function to get software information"""
        try:
            # Lowercase for case-insensitive comparison
            software_name = software_name.lower()
            
            # Software database (mock - would be replaced with real DB query)
            software_db = {
                "microsoft office": {
                    "name": "Microsoft Office",
                    "description": "Productivity suite including Word, Excel, PowerPoint, and Outlook",
                    "current_version": "Microsoft 365 (formerly Office 365)",
                    "support_link": "https://support.microsoft.com/office",
                    "license_type": "Subscription"
                },
                "office": {
                    "name": "Microsoft Office",
                    "description": "Productivity suite including Word, Excel, PowerPoint, and Outlook",
                    "current_version": "Microsoft 365 (formerly Office 365)",
                    "support_link": "https://support.microsoft.com/office",
                    "license_type": "Subscription"
                },
                "word": {
                    "name": "Microsoft Word",
                    "description": "Word processing software",
                    "current_version": "Microsoft 365 Word",
                    "support_link": "https://support.microsoft.com/word",
                    "license_type": "Part of Microsoft 365 subscription"
                },
                "excel": {
                    "name": "Microsoft Excel",
                    "description": "Spreadsheet software",
                    "current_version": "Microsoft 365 Excel",
                    "support_link": "https://support.microsoft.com/excel",
                    "license_type": "Part of Microsoft 365 subscription"
                },
                "outlook": {
                    "name": "Microsoft Outlook",
                    "description": "Email and calendar application",
                    "current_version": "Microsoft 365 Outlook",
                    "support_link": "https://support.microsoft.com/outlook",
                    "license_type": "Part of Microsoft 365 subscription"
                },
                "teams": {
                    "name": "Microsoft Teams",
                    "description": "Collaboration platform with chat, meetings, and file sharing",
                    "current_version": "Microsoft Teams",
                    "support_link": "https://support.microsoft.com/teams",
                    "license_type": "Available with Microsoft 365 subscription"
                },
                "zoom": {
                    "name": "Zoom",
                    "description": "Video conferencing software",
                    "current_version": "Zoom Desktop Client",
                    "support_link": "https://support.zoom.us",
                    "license_type": "Freemium with paid tiers"
                },
                "chrome": {
                    "name": "Google Chrome",
                    "description": "Web browser from Google",
                    "current_version": "Automatically updated",
                    "support_link": "https://support.google.com/chrome",
                    "license_type": "Free"
                },
                "firefox": {
                    "name": "Mozilla Firefox",
                    "description": "Web browser from Mozilla",
                    "current_version": "Automatically updated",
                    "support_link": "https://support.mozilla.org",
                    "license_type": "Free"
                },
                "edge": {
                    "name": "Microsoft Edge",
                    "description": "Web browser from Microsoft",
                    "current_version": "Automatically updated",
                    "support_link": "https://support.microsoft.com/edge",
                    "license_type": "Free"
                },
                "adobe acrobat": {
                    "name": "Adobe Acrobat",
                    "description": "PDF reader and editor",
                    "current_version": "Acrobat DC",
                    "support_link": "https://helpx.adobe.com/acrobat.html",
                    "license_type": "Subscription (Adobe Creative Cloud)"
                },
                "acrobat": {
                    "name": "Adobe Acrobat",
                    "description": "PDF reader and editor",
                    "current_version": "Acrobat DC",
                    "support_link": "https://helpx.adobe.com/acrobat.html",
                    "license_type": "Subscription (Adobe Creative Cloud)"
                },
                "windows": {
                    "name": "Microsoft Windows",
                    "description": "Operating system",
                    "current_version": "Windows 11",
                    "support_link": "https://support.microsoft.com/windows",
                    "license_type": "OEM or Retail license"
                },
                "macos": {
                    "name": "Apple macOS",
                    "description": "Operating system for Apple computers",
                    "current_version": "macOS Ventura",
                    "support_link": "https://support.apple.com/macos",
                    "license_type": "Included with Apple hardware"
                }
            }
            
            # Check if software exists in our database
            for key, info in software_db.items():
                if software_name in key or key in software_name:
                    # Format the output
                    return f"""
Software Information:
- Name: {info['name']}
- Description: {info['description']}
- Current Version: {info['current_version']}
- Support Link: {info['support_link']}
- License Type: {info['license_type']}
"""
            
            # If not found
            return f"Software '{software_name}' not found in our database. Please check spelling or provide more details."
            
        except Exception as e:
            logger.error(f"Error getting software info: {str(e)}")
            return f"Error retrieving software information: {str(e)}"
    
    def _troubleshoot_software(self, input_str):
        """Tool function to get software troubleshooting steps"""
        try:
            # Parse input
            parts = input_str.split(';')
            if len(parts) != 2:
                return "Invalid input format. Please provide software name and issue description separated by a semicolon."
            
            software_name = parts[0].strip().lower()
            issue = parts[1].strip().lower()
            
            # Common troubleshooting steps based on software and issue
            troubleshooting_steps = {
                "microsoft office": {
                    "activation": """
1. Check your Microsoft account is properly signed in
2. Go to File > Account to verify your subscription status
3. Run the Office repair tool: File > Account > Office Account > Repair
4. If still not working, try deactivating and reactivating: File > Account > Sign out, then sign back in
5. Contact IT for license verification if issues persist
                    """,
                    "crashes": """
1. Save your work and restart the application
2. Update Office to the latest version
3. Try running Office in Safe Mode (hold Ctrl while starting the application)
4. Repair Office installation: Control Panel > Programs > Uninstall a Program > Office > Change > Repair
5. Check for conflicting add-ins: File > Options > Add-ins > Manage: COM Add-ins > Go > Uncheck all and test
                    """,
                    "slow": """
1. Close other applications to free up memory
2. Check for large files or complex documents that might slow performance
3. Clear Office cache: %LOCALAPPDATA%\\Microsoft\\Office\\16.0\\OfficeFileCache
4. Check for Windows updates
5. Verify your computer meets minimum requirements for Office
                    """
                },
                "outlook": {
                    "not sending emails": """
1. Check your internet connection
2. Verify your email account settings: File > Account Settings
3. Try sending a test email to yourself
4. Check if you're working in Offline mode: Send/Receive tab > Work Offline (should be unchecked)
5. Create a new Outlook profile: Control Panel > Mail > Show Profiles > Add
                    """,
                    "search not working": """
1. Rebuild the Outlook search index: File > Options > Search > Indexing Options > Advanced > Rebuild
2. Verify the correct mailbox is being searched
3. Try restarting Outlook
4. Check Windows Search service is running (services.msc)
5. Repair Office installation if issues persist
                    """,
                    "calendar": """
1. Check calendar permissions if viewing shared calendars
2. Verify calendar sync settings
3. Try toggling calendar view options
4. Restart Outlook
5. Check for conflicts with other calendar applications
                    """
                },
                "teams": {
                    "audio issues": """
1. Check your speakers/headphones are properly connected and selected in Teams
2. Test your audio devices in Teams settings: Profile picture > Settings > Devices
3. Check Windows sound settings: Right-click speaker icon > Sound settings
4. Try a different headset/speaker
5. Restart Teams and your computer
                    """,
                    "video issues": """
1. Check your camera is properly connected and not in use by another application
2. Verify camera permissions in Windows settings
3. Test camera in Teams settings: Profile picture > Settings > Devices
4. Update camera drivers
5. Try joining without video, then enabling once connected
                    """,
                    "can't join meetings": """
1. Check your internet connection
2. Verify you're signed in with the correct account
3. Try joining via the web client (teams.microsoft.com)
4. Clear Teams cache: %appdata%\\Microsoft\\Teams
5. Reinstall Teams
                    """
                },
                "chrome": {
                    "crashes": """
1. Close and reopen Chrome
2. Update Chrome to the latest version
3. Clear browsing data: Three dots > Settings > Privacy and security > Clear browsing data
4. Disable extensions: Three dots > Extensions
5. Reset Chrome settings: Three dots > Settings > Advanced > Reset settings
                    """,
                    "slow": """
1. Close unnecessary tabs
2. Clear cache and cookies: Three dots > Settings > Privacy and security > Clear browsing data
3. Disable or remove unused extensions
4. Check for malware with Chrome's built-in scanner
5. Update Chrome to the latest version
                    """,
                    "won't load websites": """
1. Check your internet connection
2. Try opening the site in Incognito mode (Ctrl+Shift+N)
3. Clear DNS cache: Open Command Prompt and run 'ipconfig /flushdns'
4. Reset network settings in Windows
5. Check if the website is down for everyone or just you
                    """
                },
                "windows": {
                    "slow startup": """
1. Disable unnecessary startup programs: Task Manager > Startup tab
2. Check for malware with Windows Defender
3. Run Disk Cleanup: Search for 'Disk Cleanup' in Start
4. Defragment hard drive (for HDD, not SSD): Search for 'Defragment' in Start
5. Consider upgrading hardware if your computer is older
                    """,
                    "blue screen": """
1. Note the error code displayed on the blue screen
2. Check for Windows updates
3. Update device drivers, especially graphics and network drivers
4. Run System File Checker: Open Command Prompt as admin and type 'sfc /scannow'
5. Check for hardware issues, particularly RAM (run Memory Diagnostics)
                    """,
                    "updates failing": """
1. Run Windows Update Troubleshooter: Settings > Update & Security > Troubleshoot
2. Clear Windows Update cache: Stop Windows Update service, delete files in C:\\Windows\\SoftwareDistribution, restart service
3. Check for adequate disk space
4. Try updating in Safe Mode
5. Use the Windows Update Assistant from Microsoft's website
                    """
                }
            }
            
            # Look for matching software and issue
            for software_key, issues in troubleshooting_steps.items():
                if software_key in software_name or software_name in software_key:
                    # Try to find exact match first
                    if issue in issues:
                        return f"Troubleshooting steps for {software_key} - {issue}:\n{issues[issue]}"
                    
                    # Try partial match
                    for known_issue, steps in issues.items():
                        if known_issue in issue or issue in known_issue:
                            return f"Troubleshooting steps for {software_key} - {known_issue}:\n{steps}"
                    
                    # No specific match found, return general steps
                    general_steps = """
1. Restart the application
2. Check for software updates
3. Verify your internet connection if the application requires it
4. Restart your computer
5. Repair or reinstall the application
"""
                    return f"No specific troubleshooting steps found for '{issue}' with {software_key}. Here are general troubleshooting steps:\n{general_steps}"
            
            # No matching software found
            return f"No troubleshooting information available for software: {software_name}"
        except Exception as e:
            logger.error(f"Error providing troubleshooting steps: {str(e)}")
            return f"Error retrieving troubleshooting information: {str(e)}"
    
    def _check_software_compatibility(self, input_str):
        """Tool function to check software compatibility with OS"""
        try:
            # Parse input
            parts = input_str.split(';')
            if len(parts) != 2:
                return "Invalid input format. Please provide software name and OS separated by a semicolon."
            
            software_name = parts[0].strip().lower()
            os_name = parts[1].strip().lower()
            
            # Compatibility database (mock)
            compatibility = {
                "microsoft office": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Compatible (macOS version available)",
                    "linux": "Not officially supported"
                },
                "adobe creative cloud": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Compatible",
                    "linux": "Not supported"
                },
                "autocad": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Compatible with latest version",
                    "macos": "Not supported natively (use virtualization)",
                    "linux": "Not supported"
                },
                "zoom": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Fully compatible",
                    "linux": "Compatible (limited features)"
                },
                "chrome": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Fully compatible",
                    "linux": "Fully compatible"
                },
                "firefox": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Fully compatible",
                    "linux": "Fully compatible"
                },
                "edge": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Compatible",
                    "linux": "Not officially supported"
                },
                "teams": {
                    "windows 10": "Fully compatible",
                    "windows 11": "Fully compatible",
                    "macos": "Compatible (some features limited)",
                    "linux": "Limited compatibility (web version recommended)"
                }
            }
            
            # Check compatibility
            for software_key, os_compatibility in compatibility.items():
                if software_key in software_name or software_name in software_key:
                    for os_key, status in os_compatibility.items():
                        if os_key in os_name or os_name in os_key:
                            return f"{software_key.title()} compatibility with {os_key.title()}: {status}"
                    
                    # OS not found in our database
                    return f"No compatibility information available for {software_key.title()} with {os_name}. Please contact IT support for more information."
            
            # Software not found
            return f"No compatibility information available for {software_name} with any operating system."
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {str(e)}")
            return f"Error retrieving compatibility information: {str(e)}"
    
    def _get_software_alternatives(self, software_name):
        """Tool function to suggest software alternatives"""
        try:
            software_name = software_name.lower()
            
            # Alternatives database (mock)
            alternatives = {
                "microsoft office": [
                    {"name": "Google Workspace", "description": "Cloud-based productivity suite including Docs, Sheets, Slides"},
                    {"name": "LibreOffice", "description": "Free and open-source office suite"},
                    {"name": "WPS Office", "description": "Office suite with good Microsoft Office compatibility"}
                ],
                "word": [
                    {"name": "Google Docs", "description": "Cloud-based word processor with real-time collaboration"},
                    {"name": "LibreOffice Writer", "description": "Free and open-source word processor"},
                    {"name": "WPS Writer", "description": "Word processor with MS Word compatibility"}
                ],
                "excel": [
                    {"name": "Google Sheets", "description": "Cloud-based spreadsheet application"},
                    {"name": "LibreOffice Calc", "description": "Free and open-source spreadsheet application"},
                    {"name": "WPS Spreadsheets", "description": "Spreadsheet application with Excel compatibility"}
                ],
                "powerpoint": [
                    {"name": "Google Slides", "description": "Cloud-based presentation software"},
                    {"name": "LibreOffice Impress", "description": "Free and open-source presentation software"},
                    {"name": "WPS Presentation", "description": "Presentation software with PowerPoint compatibility"}
                ],
                "outlook": [
                    {"name": "Gmail", "description": "Free email service from Google with calendar integration"},
                    {"name": "Thunderbird", "description": "Free and open-source email client"},
                    {"name": "eM Client", "description": "Email client with calendar and tasks"}
                ],
                "adobe acrobat": [
                    {"name": "Foxit PDF Reader", "description": "Free PDF reader with annotation capabilities"},
                    {"name": "PDF-XChange Editor", "description": "Feature-rich PDF editor"},
                    {"name": "Nitro PDF", "description": "PDF creation and editing software"}
                ],
                "photoshop": [
                    {"name": "GIMP", "description": "Free and open-source image editor"},
                    {"name": "Affinity Photo", "description": "Professional photo editing software (one-time purchase)"},
                    {"name": "Pixlr", "description": "Online photo editor with free and premium versions"}
                ],
                "chrome": [
                    {"name": "Firefox", "description": "Open-source web browser focusing on privacy"},
                    {"name": "Microsoft Edge", "description": "Chromium-based browser from Microsoft"},
                    {"name": "Safari", "description": "Default browser for Apple devices"}
                ],
                "zoom": [
                    {"name": "Microsoft Teams", "description": "Collaboration platform with video meetings"},
                    {"name": "Google Meet", "description": "Video conferencing platform from Google"},
                    {"name": "Webex", "description": "Enterprise video conferencing solution"}
                ],
                "teams": [
                    {"name": "Slack", "description": "Business communication platform"},
                    {"name": "Zoom", "description": "Video conferencing with chat capabilities"},
                    {"name": "Google Meet & Chat", "description": "Google's communication and meeting tools"}
                ]
            }
            
            # Look for matching software
            for software_key, alts in alternatives.items():
                if software_key in software_name or software_name in software_key:
                    # Format the output
                    result = f"Alternatives to {software_key.title()}:\n\n"
                    for alt in alts:
                        result += f"- {alt['name']}: {alt['description']}\n"
                    
                    return result
            
            # Software not found
            return f"No alternative suggestions available for {software_name}. Please contact IT support for recommendations."
            
        except Exception as e:
            logger.error(f"Error getting software alternatives: {str(e)}")
            return f"Error retrieving software alternatives: {str(e)}"