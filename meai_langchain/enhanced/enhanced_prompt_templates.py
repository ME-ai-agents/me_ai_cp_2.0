# enhanced_prompt_templates.py
import logging
import datetime
from langchain.prompts import PromptTemplate

logger = logging.getLogger('semantic_prompt_manager')

class SemanticProfilePromptManager:
    """Enhanced prompt manager that incorporates semantic user profiles"""
    
    def __init__(self, base_prompt_manager=None):
        self.base_prompt_manager = base_prompt_manager
        
    def create_profile_aware_prompt(self, prompt_type, language, profile_data, user_info=None):
        """Create a prompt that incorporates semantic profile data"""
        # Get base prompt template either from provided manager or default templates
        if self.base_prompt_manager:
            base_template = self.base_prompt_manager.get_prompt(prompt_type, language)
        else:
            base_template = self._get_default_template(prompt_type, language)
        
        # Extract key profile attributes
        tech_level = self._get_tech_level(profile_data)
        communication_style = self._get_communication_style(profile_data)
        
        # Customize prompt based on profile
        customized_template = self._customize_template(
            base_template.template, 
            tech_level, 
            communication_style, 
            profile_data,
            user_info
        )
        
        # Create new prompt template
        return PromptTemplate(
            input_variables=base_template.input_variables,
            template=customized_template
        )
    
    def _get_tech_level(self, profile_data):
        """Extract technical proficiency level from profile data"""
        if not profile_data:
            return "intermediate"  # Default level
        
        # Check tech proficiency fields
        if profile_data.get('tech_advanced'):
            return "advanced"
        elif (profile_data.get('tech_specialized_a') or 
              profile_data.get('tech_specialized_b') or 
              profile_data.get('tech_specialized_c')):
            return "specialized"
        elif profile_data.get('tech_intermediate'):
            return "intermediate"
        elif profile_data.get('tech_basic'):
            return "basic"
        else:
            return "intermediate"  # Default if no specific level found
    
    def _get_communication_style(self, profile_data):
        """Extract communication style preferences from profile data"""
        if not profile_data or not profile_data.get('pref_communication_style'):
            return "neutral"  # Default style
        
        style = profile_data.get('pref_communication_style', '').lower()
        
        # Map communication style preferences
        if 'concise' in style or 'brief' in style:
            return "concise"
        elif 'detailed' in style or 'thorough' in style:
            return "detailed"
        elif 'formal' in style:
            return "formal"
        elif 'casual' in style or 'informal' in style:
            return "casual"
        elif 'simple' in style or 'plain' in style:
            return "simple"
        elif 'technical' in style:
            return "technical"
        else:
            return "neutral"
    
    def _customize_template(self, base_template, tech_level, communication_style, profile_data, user_info):
        """Customize the template based on profile attributes"""
        # Start with the base template
        template = base_template
        
        # Add profile-specific instructions
        profile_instructions = self._generate_profile_instructions(
            tech_level, 
            communication_style, 
            profile_data
        )
        
        # Insert profile instructions before the final prompt section
        if "User:" in template:
            parts = template.split("User:")
            template = parts[0] + profile_instructions + "\n\nUser:" + parts[1]
        else:
            # Fallback - append to end
            template += "\n\n" + profile_instructions
        
        return template
    
    def _generate_profile_instructions(self, tech_level, communication_style, profile_data):
        """Generate specific instructions based on profile attributes"""
        instructions = "\nUSER PROFILE CONSIDERATIONS:"
        
        # Technical level instructions
        instructions += "\n- Technical Proficiency: "
        if tech_level == "advanced":
            instructions += "User has advanced technical knowledge. You can use technical terminology and provide detailed technical explanations."
        elif tech_level == "specialized":
            instructions += "User has specialized technical knowledge in certain areas. Use appropriate technical terminology for their domain."
        elif tech_level == "intermediate":
            instructions += "User has intermediate technical knowledge. Balance technical details with clear explanations."
        elif tech_level == "basic":
            instructions += "User has basic technical knowledge. Avoid technical jargon and provide simple step-by-step instructions."
        
        # Communication style instructions
        instructions += "\n- Communication Style: "
        if communication_style == "concise":
            instructions += "Be brief and to the point. Minimize unnecessary details."
        elif communication_style == "detailed":
            instructions += "Provide thorough explanations with appropriate details."
        elif communication_style == "formal":
            instructions += "Use formal, professional language and structure."
        elif communication_style == "casual":
            instructions += "Use a more conversational, approachable tone."
        elif communication_style == "simple":
            instructions += "Use plain language and simple explanations."
        elif communication_style == "technical":
            instructions += "Include technical details that help solve the problem efficiently."
        else:
            instructions += "Use a balanced, neutral tone."
        
        # Add demographic-aware instructions if available
        if profile_data:
            # Age-based considerations
            if profile_data.get('demog_age'):
                age = profile_data.get('demog_age')
                if "senior" in age.lower() or "65+" in age:
                    instructions += "\n- Age Considerations: User may prefer clearer visual instructions and patience with technical concepts."
            
            # Language preferences
            if profile_data.get('demog_language'):
                language = profile_data.get('demog_language')
                if language.lower() != "english":
                    instructions += f"\n- Language Considerations: User's native language is {language}. Use straightforward language."
            
            # Occupation-based customization
            if profile_data.get('demog_occupation'):
                occupation = profile_data.get('demog_occupation')
                instructions += f"\n- Occupation Context: Consider user's role as {occupation} when providing examples or analogies."
            
            # Goal-based customization
            if profile_data.get('goal_need'):
                need = profile_data.get('goal_need')
                instructions += f"\n- User Needs: Address user's primary need: {need}"
        
        return instructions
    
    def _get_default_template(self, prompt_type, language="english"):
        """Get a default template if base_prompt_manager is not available"""
        # Basic template for hardware issues
        if prompt_type == "hardware_agent":
            template = """
You are ME.ai TechBot, specializing in hardware and technical support.

USER INFORMATION:
Name: {employee_name}
Department: {department}
Role: {role}
Devices: {devices}

FOR HARDWARE ISSUES:
1. Determine which specific device they're having an issue with
2. Ask about the symptoms they're experiencing (error messages, behavior)
3. Find out when the problem started and any recent changes
4. Ask if they've tried any troubleshooting steps already

ADDITIONAL INSTRUCTIONS:
- If the issue seems to be affecting multiple devices, explore potential network or account-related causes
- For critical issues (device won't start, data loss risk), prioritize immediate solutions
- Offer step-by-step instructions with clear indicators of progress

User: {input}
ME.ai TechBot:
"""
            return PromptTemplate(
                input_variables=["employee_name", "department", "role", "devices", "input"],
                template=template
            )
        
        # Basic template for software issues
        elif prompt_type == "software_agent":
            template = """
You are ME.ai SoftwareBot, specializing in software and application issues.

USER INFORMATION:
Name: {employee_name}
Department: {department}
Role: {role}
Devices: {devices}

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

User: {input}
ME.ai SoftwareBot:
"""
            return PromptTemplate(
                input_variables=["employee_name", "department", "role", "devices", "input"],
                template=template
            )
        
        # Basic template for password issues
        elif prompt_type == "password_agent":
            template = """
You are ME.ai SecurityBot, specializing in password and account issues.

USER INFORMATION:
Name: {employee_name}
Department: {department}
Role: {role}
Devices: {devices}

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

User: {input}
ME.ai SecurityBot:
"""
            return PromptTemplate(
                input_variables=["employee_name", "department", "role", "devices", "input"],
                template=template
            )
        
        # Default generic template
        else:
            template = """
You are ME.ai Assistant, a helpful and empathetic IT support specialist.

USER INFORMATION:
Name: {employee_name}
Department: {department}
Role: {role}
Devices: {devices}

BEHAVIOR GUIDELINES:
- Be friendly, professional, and empathetic in your responses
- If you need more information to troubleshoot, ask specific questions
- Focus on solving their problem efficiently
- Ask only one question at a time to avoid overwhelming the user
- Keep responses concise and easy to understand (around 2-3 paragraphs maximum)
- Avoid technical jargon unless the user appears technically proficient
- Address the user by their first name if available

User: {input}
ME.ai Assistant:
"""
            return PromptTemplate(
                input_variables=["employee_name", "department", "role", "devices", "input"],
                template=template
            )