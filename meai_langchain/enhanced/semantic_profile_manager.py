# semantic_profile_manager.py
import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger('semantic_profile_manager')

class SemanticProfileManager:
    """Manager for retrieving and processing semantic user profiles"""
    
    def __init__(self, db_service_url, db_username, db_password):
        self.db_service_url = db_service_url
        self.db_username = db_username
        self.db_password = db_password
        self.token = None
        self._get_db_token()
    
    def _get_db_token(self):
        """Get a token for the DB service"""
        try:
            response = requests.post(
                f"{self.db_service_url}/login",
                json={
                    "username": self.db_username,
                    "password": self.db_password
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get('token')
                logger.info(f"Successfully obtained DB service token")
                return self.token
            else:
                logger.error(f"Failed to get DB service token: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting DB service token: {str(e)}")
            return None
    
    def get_profile_by_email(self, email) -> Optional[Dict[str, Any]]:
        """Get semantic profile by email"""
        try:
            if not self.token:
                self._get_db_token()
                if not self.token:
                    return None
            
            # Query the profile endpoint
            response = requests.get(
                f"{self.db_service_url}/profiles/search",
                headers={"Authorization": f"Bearer {self.token}"},
                params={"email": email}
            )
            
            if response.status_code == 200:
                profiles = response.json()
                if profiles and len(profiles) > 0:
                    logger.info(f"Found semantic profile for email: {email}")
                    return profiles[0]
            
            logger.warning(f"No semantic profile found for email: {email}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving semantic profile by email: {str(e)}")
            return None
    
    def get_profile_by_phone(self, phone) -> Optional[Dict[str, Any]]:
        """Get semantic profile by phone number"""
        try:
            if not self.token:
                self._get_db_token()
                if not self.token:
                    return None
            
            # Query the profile endpoint
            response = requests.get(
                f"{self.db_service_url}/profiles/search",
                headers={"Authorization": f"Bearer {self.token}"},
                params={"phone": phone}
            )
            
            if response.status_code == 200:
                profiles = response.json()
                if profiles and len(profiles) > 0:
                    logger.info(f"Found semantic profile for phone: {phone}")
                    return profiles[0]
            
            logger.warning(f"No semantic profile found for phone: {phone}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving semantic profile by phone: {str(e)}")
            return None
    
    def process_profile_data(self, profile_data) -> Dict[str, Any]:
        """Process and normalize semantic profile data"""
        if not profile_data:
            return {
                "tech_level": "intermediate",
                "communication_style": "neutral",
                "demographics": {},
                "preferences": {},
                "goals": {}
            }
        
        processed_data = {
            "tech_level": self.get_tech_proficiency_level(profile_data),
            "communication_style": self.get_communication_style(profile_data),
            "demographics": self.extract_demographics(profile_data),
            "preferences": self.extract_preferences(profile_data),
            "goals": self.extract_goals(profile_data),
            "behavioral": self.extract_behavioral_patterns(profile_data)
        }
        
        return processed_data
    
    def get_tech_proficiency_level(self, profile) -> str:
        """Extract technical proficiency level from profile"""
        if not profile:
            return "intermediate"  # Default level
        
        # Check tech proficiency fields
        if profile.get('tech_advanced'):
            return "advanced"
        elif profile.get('tech_specialized_a') or profile.get('tech_specialized_b') or profile.get('tech_specialized_c'):
            return "specialized"
        elif profile.get('tech_intermediate'):
            return "intermediate"
        elif profile.get('tech_basic'):
            return "basic"
        else:
            return "intermediate"  # Default if no specific level found
    
    def get_communication_style(self, profile) -> str:
        """Extract communication style from profile"""
        if not profile or not profile.get('pref_communication_style'):
            return "neutral"  # Default style
        
        style = profile.get('pref_communication_style', '').lower()
        
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
    
    def extract_demographics(self, profile) -> Dict[str, str]:
        """Extract demographic information from profile"""
        demographics = {}
        
        demo_fields = ['demog_age', 'demog_gender', 'demog_occupation', 
                       'demog_education', 'demog_location', 'demog_language']
        
        for field in demo_fields:
            if field in profile and profile[field]:
                key = field.replace('demog_', '')
                demographics[key] = profile[field]
        
        return demographics
    
    def extract_preferences(self, profile) -> Dict[str, str]:
        """Extract user preferences from profile"""
        preferences = {}
        
        pref_fields = ['pref_terms', 'pref_products', 'pref_services', 
                        'pref_communication_style', 'pref_expectation']
        
        for field in pref_fields:
            if field in profile and profile[field]:
                key = field.replace('pref_', '')
                preferences[key] = profile[field]
        
        return preferences
    
    def extract_goals(self, profile) -> Dict[str, str]:
        """Extract user goals and motivations from profile"""
        goals = {}
        
        goal_fields = ['goal_pain', 'goal_need', 'goal_driver', 'goal_aim']
        
        for field in goal_fields:
            if field in profile and profile[field]:
                key = field.replace('goal_', '')
                goals[key] = profile[field]
        
        return goals
    
    def extract_behavioral_patterns(self, profile) -> Dict[str, str]:
        """Extract behavioral patterns from profile"""
        behavioral = {}
        
        behav_fields = ['behv_actions', 'behv_habits', 'behv_activity', 'behv_interactions']
        
        for field in behav_fields:
            if field in profile and profile[field]:
                key = field.replace('behv_', '')
                behavioral[key] = profile[field]
        
        return behavioral
    
    def create_profile_prompt_section(self, profile) -> str:
        """Create a section for the prompt that includes relevant profile information"""
        if not profile:
            return ""
        
        processed_data = self.process_profile_data(profile)
        
        prompt_section = "USER SEMANTIC PROFILE:\n"
        
        # Add tech level
        prompt_section += f"- Technical Proficiency: {processed_data['tech_level']}\n"
        
        # Add communication style
        prompt_section += f"- Communication Style: {processed_data['communication_style']}\n"
        
        # Add demographics if available
        demographics = processed_data['demographics']
        if demographics:
            prompt_section += "- Demographics:\n"
            for key, value in demographics.items():
                prompt_section += f"  * {key.capitalize()}: {value}\n"
        
        # Add preferences if available
        preferences = processed_data['preferences']
        if preferences:
            prompt_section += "- Preferences:\n"
            for key, value in preferences.items():
                if key != 'communication_style':  # Already included above
                    prompt_section += f"  * {key.capitalize()}: {value}\n"
        
        # Add goals if available
        goals = processed_data['goals']
        if goals:
            prompt_section += "- Goals and Motivations:\n"
            for key, value in goals.items():
                prompt_section += f"  * {key.capitalize()}: {value}\n"
        
        # Add behavioral patterns if relevant
        behavioral = processed_data['behavioral']
        if behavioral:
            prompt_section += "- Behavioral Patterns:\n"
            for key, value in behavioral.items():
                prompt_section += f"  * {key.capitalize()}: {value}\n"
        
        return prompt_section
    
    def get_tailored_instructions(self, profile) -> str:
        """Generate tailored instructions based on profile"""
        if not profile:
            return ""
        
        processed_data = self.process_profile_data(profile)
        tech_level = processed_data['tech_level']
        comm_style = processed_data['communication_style']
        
        instructions = "RESPONSE CUSTOMIZATION:\n"
        
        # Technical level instructions
        instructions += "- Based on technical proficiency:\n"
        if tech_level == "advanced":
            instructions += "  * You can use technical terminology and provide detailed technical explanations\n"
            instructions += "  * Focus on efficiency and advanced solutions\n"
            instructions += "  * Assume familiarity with IT concepts and tools\n"
        elif tech_level == "specialized":
            instructions += "  * Use appropriate technical terminology for their domain\n"
            instructions += "  * Provide specialized insights when relevant to their field\n"
            instructions += "  * Balance technical detail with clear explanations\n"
        elif tech_level == "intermediate":
            instructions += "  * Balance technical details with clear explanations\n"
            instructions += "  * Explain important concepts without oversimplifying\n"
            instructions += "  * Provide context for technical terms\n"
        elif tech_level == "basic":
            instructions += "  * Avoid technical jargon and provide simple step-by-step instructions\n"
            instructions += "  * Use analogies and examples to explain concepts\n"
            instructions += "  * Focus on visual guidance and clear indicators of progress\n"
        
        # Communication style instructions
        instructions += "- Based on communication preferences:\n"
        if comm_style == "concise":
            instructions += "  * Be brief and to the point\n"
            instructions += "  * Prioritize actionable information\n"
            instructions += "  * Use bullet points and short paragraphs\n"
        elif comm_style == "detailed":
            instructions += "  * Provide thorough explanations with appropriate details\n"
            instructions += "  * Include contextual information and background\n"
            instructions += "  * Explain the reasoning behind recommendations\n"
        elif comm_style == "formal":
            instructions += "  * Use formal, professional language and structure\n"
            instructions += "  * Maintain a respectful and business-like tone\n"
            instructions += "  * Avoid casual expressions and slang\n"
        elif comm_style == "casual":
            instructions += "  * Use a more conversational, approachable tone\n"
            instructions += "  * Include friendly rapport-building elements\n"
            instructions += "  * Balance warmth with professionalism\n"
        elif comm_style == "simple":
            instructions += "  * Use plain language and simple explanations\n"
            instructions += "  * Avoid complex sentence structures\n"
            instructions += "  * Focus on clarity above all\n"
        
        # Add demographic-specific instructions
        demographics = processed_data['demographics']
        if demographics:
            if 'age' in demographics:
                age = demographics['age'].lower()
                if "senior" in age or "65+" in age:
                    instructions += "- Age considerations:\n"
                    instructions += "  * Provide clearer visual instructions\n"
                    instructions += "  * Allow more time for technical concepts\n"
                    instructions += "  * Avoid assuming familiarity with newer technologies\n"
            
            if 'language' in demographics and demographics['language'].lower() != "english":
                instructions += "- Language considerations:\n"
                instructions += "  * Use straightforward language\n"
                instructions += "  * Avoid idioms and complex expressions\n"
                instructions += "  * Use simple sentence structures\n"
        
        return instructions