# agent/hardware_agent.py
from langchain.tools import Tool
from .base_agent import MeAIBaseAgent
from existing.db_service import get_employee_devices
import logging

logger = logging.getLogger('me_agent_orchestrator')

class HardwareAgent(MeAIBaseAgent):
    """Agent specializing in hardware issues"""
    
    def __init__(self, aws_region="us-east-1", model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__("Hardware", aws_region, model_id)
    
    def _get_tools(self):
        """Get hardware-specific tools"""
        tools = [
            Tool(
                name="get_employee_devices",
                func=self._get_employee_devices,
                description="Get all devices registered to an employee. Input should be the employee ID."
            ),
            Tool(
                name="check_device_status",
                func=self._check_device_status,
                description="Check the status of a specific device. Input should be a device ID."
            ),
            Tool(
                name="troubleshoot_hardware",
                func=self._troubleshoot_hardware,
                description="Get troubleshooting steps for common hardware issues. Input should be device type (laptop, desktop, printer) and issue description separated by a semicolon."
            )
        ]
        return tools
    
    def _get_employee_devices(self, employee_id):
        """Tool function to get employee devices"""
        try:
            # Create a mock employee_info with just the employee_id for the DB service
            employee_info = {"employee_id": employee_id}
            devices = get_employee_devices(employee_info)
            
            if not devices:
                return "No devices found for this employee."
            
            # Format device information
            device_info = "Employee devices:\n"
            for i, device in enumerate(devices):
                device_info += f"{i+1}. {device.get('device_name', 'Unknown Device')} - {device.get('os_type', 'Unknown OS')} {device.get('os_version', '')}\n"
            
            return device_info
        except Exception as e:
            logger.error(f"Error getting employee devices: {str(e)}")
            return f"Error retrieving device information: {str(e)}"
    
    def _check_device_status(self, device_id):
        """Tool function to check device status"""
        # This would connect to your device monitoring system
        # For MVP, we'll return mock data
        try:
            # Mock implementation - in a real system this would query device status
            statuses = {
                "DEV001": {"status": "Online", "last_check": "2025-04-17 08:30:00", "issues": "None"},
                "DEV002": {"status": "Offline", "last_check": "2025-04-16 17:45:00", "issues": "Connectivity issues"},
                "DEV003": {"status": "Warning", "last_check": "2025-04-17 09:15:00", "issues": "Low disk space"}
            }
            
            if device_id in statuses:
                status = statuses[device_id]
                return f"Device Status: {status['status']}\nLast Checked: {status['last_check']}\nIssues: {status['issues']}"
            else:
                return "Device not found in monitoring system."
        except Exception as e:
            logger.error(f"Error checking device status: {str(e)}")
            return f"Error checking device status: {str(e)}"
    
    def _troubleshoot_hardware(self, input_str):
        """Tool function to get hardware troubleshooting steps"""
        try:
            # Parse input
            parts = input_str.split(';')
            if len(parts) != 2:
                return "Invalid input format. Please provide device type and issue description separated by a semicolon."
            
            device_type = parts[0].strip().lower()
            issue = parts[1].strip().lower()
            
            # Common troubleshooting steps based on device type and issue
            troubleshooting_steps = {
                "laptop": {
                    "won't power on": """
1. Check power connection and cable
2. Remove battery, hold power button for 30 seconds, reinsert battery
3. Try a different power outlet
4. Check for any physical damage to the power port
5. If still not working, contact IT support for hardware assessment
                    """,
                    "slow performance": """
1. Restart the computer
2. Check for available disk space (need at least 10% free)
3. Close unnecessary applications running in the background
4. Check for malware or viruses
5. Verify your computer is not overheating
                    """,
                    "blue screen": """
1. Note any error codes displayed on the blue screen
2. Restart the computer
3. Check for recent software or driver updates
4. Boot in Safe Mode to isolate issues
5. Run hardware diagnostics from BIOS/UEFI
                    """
                },
                "desktop": {
                    "won't power on": """
1. Check power cable connections at both computer and wall outlet
2. Test with a different power cable if available
3. Check if the power supply switch is turned on
4. Listen for any beep codes during startup
5. Verify monitor is powered on and connected properly
                    """,
                    "slow performance": """
1. Restart the computer
2. Check for available disk space
3. Check CPU and memory usage in Task Manager
4. Close unnecessary applications and background processes
5. Scan for malware and viruses
                    """,
                    "strange noises": """
1. Identify source of noise (fan, hard drive, power supply)
2. Check for dust buildup and clean if necessary
3. Ensure all fans are functioning properly
4. Check for loose components or cables
5. For grinding noises from hard drives, backup data immediately
                    """
                },
                "printer": {
                    "not printing": """
1. Check physical connections (power, network/USB)
2. Verify printer is online and ready (no error lights)
3. Check for paper jams
4. Restart the printer
5. Clear print queue on computer
6. Check if correct printer is selected
                    """,
                    "poor print quality": """
1. Run printer cleaning cycle
2. Check toner/ink levels
3. Verify paper type settings match paper being used
4. Check for any obstructions in paper path
5. Update printer drivers
                    """,
                    "paper jam": """
1. Power off the printer
2. Open all access panels
3. Gently remove jammed paper (pull in direction of normal paper path)
4. Check for torn paper remaining inside
5. Close all panels and restart printer
                    """
                }
            }
            
            # Look for matching device and issue
            if device_type in troubleshooting_steps:
                device_issues = troubleshooting_steps[device_type]
                
                # Try to find exact match first
                if issue in device_issues:
                    return f"Troubleshooting steps for {device_type} - {issue}:\n{device_issues[issue]}"
                
                # Try partial match
                for known_issue, steps in device_issues.items():
                    if known_issue in issue or issue in known_issue:
                        return f"Troubleshooting steps for {device_type} - {known_issue}:\n{steps}"
                
                # No specific match found, return general steps
                return f"No specific troubleshooting steps found for '{issue}' with {device_type}. Here are general troubleshooting steps:\n1. Restart the device\n2. Check all physical connections\n3. Update drivers/firmware\n4. Run built-in diagnostics if available\n5. Document any error messages"
            else:
                return f"No troubleshooting information available for device type: {device_type}"
        except Exception as e:
            logger.error(f"Error providing troubleshooting steps: {str(e)}")
            return f"Error retrieving troubleshooting information: {str(e)}"
