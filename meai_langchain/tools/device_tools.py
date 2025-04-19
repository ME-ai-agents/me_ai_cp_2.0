# tools/device_tools.py
from langchain.tools import Tool
import logging
import json
import datetime

logger = logging.getLogger('me_agent_orchestrator')

def create_device_tools():
    """Create tools for interacting with devices"""
    
    tools = [
        Tool(
            name="get_device_status",
            func=get_device_status,
            description="Get the current status of a device. Input should be the device ID."
        ),
        Tool(
            name="get_device_specs",
            func=get_device_specs,
            description="Get detailed specifications for a device. Input should be the device ID."
        ),
        Tool(
            name="get_device_history",
            func=get_device_history,
            description="Get maintenance and issue history for a device. Input should be the device ID."
        ),
        Tool(
            name="check_software_compatibility",
            func=check_software_compatibility,
            description="Check if a software is compatible with a device. Input should be a JSON string with device_id and software_name."
        ),
        Tool(
            name="get_common_issue_solutions",
            func=get_common_issue_solutions,
            description="Get solutions for common issues with a device type. Input should be the device type (laptop, desktop, printer, etc.)."
        )
    ]
    
    return tools

# Tool implementation functions

def get_device_status(device_id):
    """Get the current status of a device"""
    try:
        # This would connect to your device monitoring system in production
        # For now, we'll return mock data
        
        # Mock device status database
        device_statuses = {
            "D001": {
                "status": "Online",
                "last_check": "2025-04-17 09:45:23",
                "uptime": "13 days, 7 hours",
                "cpu_usage": "23%",
                "memory_usage": "45%",
                "disk_usage": "67%",
                "issues_detected": []
            },
            "D002": {
                "status": "Warning",
                "last_check": "2025-04-17 08:30:12",
                "uptime": "45 days, 3 hours",
                "cpu_usage": "12%",
                "memory_usage": "34%",
                "disk_usage": "92%",  # High disk usage
                "issues_detected": ["Disk space low (92% used)"]
            },
            "D003": {
                "status": "Offline",
                "last_check": "2025-04-16 17:15:45",
                "uptime": "Unknown",
                "cpu_usage": "Unknown",
                "memory_usage": "Unknown",
                "disk_usage": "Unknown",
                "issues_detected": ["Device not responding"]
            },
            "D004": {
                "status": "Warning",
                "last_check": "2025-04-17 10:05:33",
                "uptime": "7 days, 14 hours",
                "cpu_usage": "87%",  # High CPU usage
                "memory_usage": "76%",
                "disk_usage": "54%",
                "issues_detected": ["High CPU usage (87%)"]
            },
            "D005": {
                "status": "Online",
                "last_check": "2025-04-17 10:12:09",
                "uptime": "2 days, 5 hours",
                "cpu_usage": "18%",
                "memory_usage": "42%",
                "disk_usage": "38%",
                "issues_detected": []
            }
        }
        
        # Check if the device ID exists in our database
        if device_id in device_statuses:
            status = device_statuses[device_id]
            
            # Format the response
            response = f"""
Device Status for {device_id}:
- Status: {status['status']}
- Last Check: {status['last_check']}
- Uptime: {status['uptime']}
- CPU Usage: {status['cpu_usage']}
- Memory Usage: {status['memory_usage']}
- Disk Usage: {status['disk_usage']}
"""
            
            # Add issues if any
            if status['issues_detected']:
                response += "- Issues Detected:\n"
                for issue in status['issues_detected']:
                    response += f"  * {issue}\n"
            else:
                response += "- No issues detected\n"
                
            return response
        else:
            return f"Device ID {device_id} not found in monitoring system."
        
    except Exception as e:
        logger.error(f"Error getting device status: {str(e)}")
        return f"Error retrieving device status: {str(e)}"

def get_device_specs(device_id):
    """Get detailed specifications for a device"""
    try:
        # Mock device specifications database
        device_specs = {
            "D001": {
                "type": "Laptop",
                "manufacturer": "Dell",
                "model": "Latitude 7420",
                "os": "Windows 11 Pro",
                "os_version": "22H2",
                "cpu": "Intel Core i7-1185G7 @ 3.0GHz",
                "ram": "16GB DDR4",
                "storage": "512GB NVMe SSD",
                "display": "14-inch FHD (1920 x 1080)",
                "graphics": "Intel Iris Xe Graphics",
                "network": "Intel Wi-Fi 6 AX201, Bluetooth 5.1",
                "last_updated": "2025-02-15"
            },
            "D002": {
                "type": "Desktop",
                "manufacturer": "HP",
                "model": "EliteDesk 800 G5",
                "os": "Windows 10 Enterprise",
                "os_version": "21H2",
                "cpu": "Intel Core i5-9500 @ 3.0GHz",
                "ram": "32GB DDR4",
                "storage": "1TB NVMe SSD",
                "display": "N/A (Connected to 24-inch monitor)",
                "graphics": "Intel UHD Graphics 630",
                "network": "Intel Ethernet Connection I219-LM, Wi-Fi 6",
                "last_updated": "2025-03-01"
            },
            "D003": {
                "type": "Laptop",
                "manufacturer": "Apple",
                "model": "MacBook Pro 14",
                "os": "macOS",
                "os_version": "Ventura 13.3",
                "cpu": "Apple M1 Pro",
                "ram": "16GB Unified Memory",
                "storage": "512GB SSD",
                "display": "14.2-inch Liquid Retina XDR (3024 x 1964)",
                "graphics": "14-core GPU (integrated)",
                "network": "Wi-Fi 6, Bluetooth 5.0",
                "last_updated": "2025-01-20"
            },
            "D004": {
                "type": "Desktop",
                "manufacturer": "Dell",
                "model": "OptiPlex 7090",
                "os": "Windows 11 Enterprise",
                "os_version": "22H2",
                "cpu": "Intel Core i7-10700 @ 2.9GHz",
                "ram": "32GB DDR4",
                "storage": "512GB NVMe SSD + 2TB HDD",
                "display": "N/A (Connected to dual 27-inch monitors)",
                "graphics": "NVIDIA GeForce GTX 1660 Super",
                "network": "Intel Ethernet I219-LM, Wi-Fi 6",
                "last_updated": "2025-03-15"
            },
            "D005": {
                "type": "Printer",
                "manufacturer": "HP",
                "model": "LaserJet Enterprise M607",
                "os": "HP Firmware",
                "os_version": "2.92.7",
                "cpu": "N/A",
                "ram": "512MB",
                "storage": "N/A",
                "display": "2.7-inch Color LCD Control Panel",
                "graphics": "N/A",
                "network": "Gigabit Ethernet, Wi-Fi Direct",
                "last_updated": "2025-02-10"
            }
        }
        
        # Check if the device ID exists in our database
        if device_id in device_specs:
            specs = device_specs[device_id]
            
            # Format the response
            response = f"""
Device Specifications for {device_id}:
- Type: {specs['type']}
- Manufacturer: {specs['manufacturer']}
- Model: {specs['model']}
- Operating System: {specs['os']} {specs['os_version']}
- CPU: {specs['cpu']}
- RAM: {specs['ram']}
- Storage: {specs['storage']}
- Display: {specs['display']}
- Graphics: {specs['graphics']}
- Network: {specs['network']}
- Last Updated: {specs['last_updated']}
"""
            return response
        else:
            return f"Device ID {device_id} not found in the asset database."
        
    except Exception as e:
        logger.error(f"Error getting device specifications: {str(e)}")
        return f"Error retrieving device specifications: {str(e)}"

def get_device_history(device_id):
    """Get maintenance and issue history for a device"""
    try:
        # Mock device history database
        device_history = {
            "D001": [
                {"date": "2025-03-25", "type": "Maintenance", "description": "Annual security audit and updates"},
                {"date": "2025-01-15", "type": "Issue", "description": "Blue screen error during startup - Resolved by updating graphics drivers"},
                {"date": "2024-11-10", "type": "Maintenance", "description": "OS upgrade to Windows 11"}
            ],
            "D002": [
                {"date": "2025-04-01", "type": "Issue", "description": "Low disk space warning - Resolved by cleanup"},
                {"date": "2025-02-15", "type": "Maintenance", "description": "RAM upgrade from 16GB to 32GB"},
                {"date": "2024-12-20", "type": "Issue", "description": "Network connectivity issues - Resolved by NIC driver update"}
            ],
            "D003": [
                {"date": "2025-03-10", "type": "Maintenance", "description": "macOS security updates"},
                {"date": "2025-01-05", "type": "Issue", "description": "Battery not charging properly - Replaced battery"},
                {"date": "2024-09-30", "type": "Maintenance", "description": "Initial setup and deployment"}
            ],
            "D004": [
                {"date": "2025-04-05", "type": "Issue", "description": "High CPU usage - Resolved by malware removal"},
                {"date": "2025-02-20", "type": "Maintenance", "description": "Added secondary 2TB HDD"},
                {"date": "2024-10-15", "type": "Maintenance", "description": "Graphics card upgrade"}
            ],
            "D005": [
                {"date": "2025-03-15", "type": "Maintenance", "description": "Firmware update"},
                {"date": "2025-02-01", "type": "Issue", "description": "Paper jam errors - Cleaned rollers and sensors"},
                {"date": "2024-12-10", "type": "Maintenance", "description": "Toner replacement"}
            ]
        }
        
        # Check if the device ID exists in our database
        if device_id in device_history:
            history = device_history[device_id]
            
            # Format the response
            response = f"Device History for {device_id}:\n\n"
            
            for entry in history:
                response += f"Date: {entry['date']}\n"
                response += f"Type: {entry['type']}\n"
                response += f"Description: {entry['description']}\n\n"
                
            return response
        else:
            return f"No history found for device ID {device_id}."
        
    except Exception as e:
        logger.error(f"Error getting device history: {str(e)}")
        return f"Error retrieving device history: {str(e)}"

def check_software_compatibility(input_str):
    """Check if a software is compatible with a device"""
    try:
        # Parse input
        try:
            input_data = json.loads(input_str)
            device_id = input_data.get("device_id")
            software_name = input_data.get("software_name")
        except json.JSONDecodeError:
            # Try parsing as device_id;software_name format
            parts = input_str.split(';')
            if len(parts) != 2:
                return "Invalid input format. Please provide input as JSON with device_id and software_name, or as 'device_id;software_name'."
            device_id = parts[0].strip()
            software_name = parts[1].strip()
        
        # Mock device database
        device_specs = {
            "D001": {
                "type": "Laptop",
                "os": "Windows 11 Pro",
                "os_version": "22H2",
                "cpu": "Intel Core i7-1185G7 @ 3.0GHz",
                "ram": "16GB"
            },
            "D002": {
                "type": "Desktop",
                "os": "Windows 10 Enterprise",
                "os_version": "21H2",
                "cpu": "Intel Core i5-9500 @ 3.0GHz",
                "ram": "32GB"
            },
            "D003": {
                "type": "Laptop",
                "os": "macOS",
                "os_version": "Ventura 13.3",
                "cpu": "Apple M1 Pro",
                "ram": "16GB"
            },
            "D004": {
                "type": "Desktop",
                "os": "Windows 11 Enterprise",
                "os_version": "22H2",
                "cpu": "Intel Core i7-10700 @ 2.9GHz",
                "ram": "32GB"
            },
            "D005": {
                "type": "Printer",
                "os": "HP Firmware",
                "os_version": "2.92.7",
                "cpu": "N/A",
                "ram": "512MB"
            }
        }
        
        # Mock software compatibility database
        software_requirements = {
            "microsoft office": {
                "windows": {
                    "min_os_version": "10.0",
                    "recommended_ram": "8GB",
                    "recommended_cpu": "1.6 GHz"
                },
                "macos": {
                    "min_os_version": "12.0",
                    "recommended_ram": "8GB",
                    "recommended_cpu": "Intel or Apple Silicon"
                }
            },
            "adobe creative cloud": {
                "windows": {
                    "min_os_version": "10.0",
                    "recommended_ram": "16GB",
                    "recommended_cpu": "2.0 GHz"
                },
                "macos": {
                    "min_os_version": "12.0",
                    "recommended_ram": "16GB",
                    "recommended_cpu": "Intel or Apple Silicon"
                }
            },
            "autocad": {
                "windows": {
                    "min_os_version": "10.0",
                    "recommended_ram": "16GB",
                    "recommended_cpu": "3.0 GHz"
                },
                "macos": {
                    "compatible": False,
                    "notes": "AutoCAD is not available for macOS. Consider using via virtualization."
                }
            }
        }
        
        # Check if the device exists
        if device_id not in device_specs:
            return f"Device ID {device_id} not found in the asset database."
        
        # Get device specs
        device = device_specs[device_id]
        
        # Normalize software name for lookup
        software_lower = software_name.lower()
        
        # Check if we have compatibility info for this software
        matching_software = None
        for sw_name in software_requirements:
            if sw_name in software_lower:
                matching_software = sw_name
                break
        
        if not matching_software:
            return f"No compatibility information found for {software_name}."
        
        # Get OS platform
        os_platform = "unknown"
        if "windows" in device["os"].lower():
            os_platform = "windows"
        elif "macos" in device["os"].lower():
            os_platform = "macos"
        
        # Check compatibility based on OS
        sw_info = software_requirements[matching_software]
        if os_platform not in sw_info:
            return f"{software_name} is not compatible with {device['os']}."
        
        platform_info = sw_info[os_platform]
        
        # Check if explicitly marked as incompatible
        if platform_info.get("compatible") is False:
            return f"{software_name} is not compatible with {device['os']}. {platform_info.get('notes', '')}"
        
        # For OS version comparison, we would need more sophisticated version parsing
        # This is a simplified check
        min_os_version = platform_info.get("min_os_version", "0.0")
        
        # Convert RAM to numeric value (GB)
        device_ram = device["ram"]
        ram_gb = 0
        if "GB" in device_ram:
            try:
                ram_gb = int(device_ram.split("GB")[0].strip())
            except ValueError:
                ram_gb = 0
        
        recommended_ram = platform_info.get("recommended_ram", "0GB")
        recommended_ram_gb = 0
        if "GB" in recommended_ram:
            try:
                recommended_ram_gb = int(recommended_ram.split("GB")[0].strip())
            except ValueError:
                recommended_ram_gb = 0
        
        # Determine compatibility
        is_compatible = True
        compatibility_notes = []
        
        # Add RAM check
        if ram_gb < recommended_ram_gb:
            is_compatible = False
            compatibility_notes.append(f"RAM: Device has {device_ram} but {software_name} recommends {recommended_ram}")
        
        # Format result
        if is_compatible:
            result = f"{software_name} is compatible with the device (ID: {device_id}, {device['type']}, {device['os']} {device['os_version']})."
        else:
            result = f"{software_name} may not work optimally with the device (ID: {device_id}, {device['type']}, {device['os']} {device['os_version']}).\n\nCompatibility issues:\n"
            for note in compatibility_notes:
                result += f"- {note}\n"
        
        return result
    
    except Exception as e:
        logger.error(f"Error checking software compatibility: {str(e)}")
        return f"Error checking software compatibility: {str(e)}"

def get_common_issue_solutions(device_type):
    """Get solutions for common issues with a device type"""
    try:
        device_type = device_type.lower()
        
        # Mock database of common issues and solutions
        common_issues = {
            "laptop": [
                {
                    "issue": "Battery drains quickly",
                    "solutions": [
                        "Reduce screen brightness",
                        "Close unused applications and browser tabs",
                        "Check for resource-intensive background processes",
                        "Disable unused wireless connections (Bluetooth, Wi-Fi)",
                        "Consider replacing the battery if it's more than 2 years old"
                    ]
                },
                {
                    "issue": "Overheating",
                    "solutions": [
                        "Ensure vents are not blocked",
                        "Use laptop on hard, flat surfaces (not on beds/sofas)",
                        "Clean dust from cooling vents with compressed air",
                        "Close unused applications to reduce CPU load",
                        "Consider using a cooling pad"
                    ]
                },
                {
                    "issue": "Blue screen errors",
                    "solutions": [
                        "Update all device drivers",
                        "Check for and install Windows updates",
                        "Run memory diagnostic test",
                        "Scan for malware",
                        "If persistent, check event logs for specific error codes"
                    ]
                },
                {
                    "issue": "Wi-Fi connectivity issues",
                    "solutions": [
                        "Toggle Wi-Fi off and on",
                        "Forget network and reconnect",
                        "Update wireless drivers",
                        "Reset network settings",
                        "Check if issue persists on other networks"
                    ]
                }
            ],
            "desktop": [
                {
                    "issue": "Slow startup/performance",
                    "solutions": [
                        "Disable unnecessary startup programs",
                        "Run disk cleanup and defragmentation",
                        "Check for and remove malware",
                        "Ensure adequate free disk space (at least 15%)",
                        "Consider hardware upgrades (SSD, more RAM)"
                    ]
                },
                {
                    "issue": "No display output",
                    "solutions": [
                        "Check monitor power and connections",
                        "Try a different display cable",
                        "Test with an alternative monitor if available",
                        "Verify graphics card is properly seated",
                        "Listen for beep codes during startup that may indicate hardware issues"
                    ]
                },
                {
                    "issue": "Random restarts",
                    "solutions": [
                        "Check for overheating issues",
                        "Test power supply",
                        "Run memory diagnostic tests",
                        "Scan for malware",
                        "Check event logs for hardware errors"
                    ]
                }
            ],
            "printer": [
                {
                    "issue": "Paper jams",
                    "solutions": [
                        "Remove jammed paper carefully following manufacturer guidelines",
                        "Use recommended paper type and weight",
                        "Don't overfill paper tray",
                        "Keep paper properly stored to avoid moisture",
                        "Clean paper feed rollers"
                    ]
                },
                {
                    "issue": "Poor print quality",
                    "solutions": [
                        "Run printer cleaning utility",
                        "Replace low ink/toner cartridges",
                        "Use high-quality paper",
                        "Update printer drivers",
                        "Check for and clear any paper dust"
                    ]
                },
                {
                    "issue": "Printer offline",
                    "solutions": [
                        "Check physical connections (power, network/USB)",
                        "Restart the printer",
                        "Clear print queue",
                        "Set printer as default",
                        "Reinstall printer drivers"
                    ]
                }
            ],
            "monitor": [
                {
                    "issue": "No signal",
                    "solutions": [
                        "Check cable connections",
                        "Try a different input port",
                        "Test with a different computer",
                        "Verify computer is not in sleep mode",
                        "Check monitor input source setting"
                    ]
                },
                {
                    "issue": "Flickering display",
                    "solutions": [
                        "Update graphics drivers",
                        "Try a different cable",
                        "Adjust refresh rate settings",
                        "Test with a different power outlet",
                        "Check for electromagnetic interference"
                    ]
                }
            ]
        }
        
        # Find matching device type
        matching_type = None
        for dt in common_issues:
            if dt in device_type or device_type in dt:
                matching_type = dt
                break
        
        if not matching_type:
            return f"No common issue solutions found for device type: {device_type}"
        
        # Format the response
        issues = common_issues[matching_type]
        
        response = f"Common Issues and Solutions for {matching_type.capitalize()}:\n\n"
        
        for issue_info in issues:
            response += f"Issue: {issue_info['issue']}\n"
            response += "Solutions:\n"
            
            for i, solution in enumerate(issue_info['solutions']):
                response += f"  {i+1}. {solution}\n"
            
            response += "\n"
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting common issue solutions: {str(e)}")
        return f"Error retrieving common issue solutions: {str(e)}"