# neo4j_itsm_manager.py
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

logger = logging.getLogger('neo4j_itsm_manager')

class ITSMOntologyManager:
    """Manager for interacting with the ITSM ontology in Neo4j"""
    
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to the Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def query_ontology(self, query, params=None):
        """Run a Cypher query against the ontology"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}")
            return []
    
    def query_troubleshooting_steps(self, issue_type, device_type=None):
        """Query troubleshooting steps for a specific issue type and device"""
        # First try to get specific troubleshooting for the combination
        if device_type:
            query = """
            MATCH (c:Class)
            WHERE (c.name CONTAINS $issueType OR c.label CONTAINS $issueType)
            AND (c.name CONTAINS $deviceType OR c.label CONTAINS $deviceType)
            MATCH (c)-[:HAS]->(step:TroubleshootingStep)
            RETURN step.name as step_name, step.description as step_description, step.order as step_order
            ORDER BY step.order
            """
            
            result = self.query_ontology(query, {"issueType": issue_type, "deviceType": device_type})
            
            # If we found specific steps, return them
            if result:
                return result
        
        # Fall back to general troubleshooting for the issue type
        query = """
        MATCH (c:Class)
        WHERE (c.name CONTAINS $issueType OR c.label CONTAINS $issueType)
        MATCH (c)-[:HAS]->(step:TroubleshootingStep)
        RETURN step.name as step_name, step.description as step_description, step.order as step_order
        ORDER BY step.order
        """
        
        return self.query_ontology(query, {"issueType": issue_type})
    
    def query_potential_solutions(self, issue_type, keywords=None):
        """Query potential solutions for an issue type"""
        # Start with base query
        query = """
        MATCH (i:Class)-[:MAY_INDICATE]->(p:Problem)
        WHERE (i.name CONTAINS $issueType OR i.label CONTAINS $issueType)
        MATCH (p)-[:RESOLVED_BY]->(s:Solution)
        """
        
        params = {"issueType": issue_type}
        
        # Add keyword filters if provided
        if keywords:
            keyword_conditions = []
            for i, kw in enumerate(keywords):
                if kw:  # Skip empty keywords
                    param_name = f"kw{i}"
                    keyword_conditions.append(f"s.description CONTAINS ${param_name}")
                    params[param_name] = kw
            
            if keyword_conditions:
                query += "WHERE " + " OR ".join(keyword_conditions)
        
        # Complete the query
        query += """
        RETURN p.name as problem_name, p.description as problem_description,
               s.name as solution_name, s.description as solution_description,
               s.effectiveness as solution_effectiveness
        ORDER BY s.effectiveness DESC
        LIMIT 5
        """
        
        return self.query_ontology(query, params)
    
    def query_service_dependencies(self, service_name):
        """Query dependencies for a service"""
        query = """
        MATCH (s:Class {name: $serviceName})-[:DEPENDS_ON]->(dep:Class)
        RETURN dep.name as dependency_name, dep.label as dependency_label, 
               dep.description as dependency_description
        """
        
        return self.query_ontology(query, {"serviceName": service_name})
    
    def format_ontology_for_prompt(self, concepts):
        """Format ontology concepts for inclusion in prompt"""
        if not concepts:
            return ""
        
        ontology_prompt = "ITSM ONTOLOGY CONCEPTS:\n"
        
        for concept in concepts:
            if all(k in concept for k in ['source_name', 'related_name']):
                # Format graph relationship
                source_name = concept.get('source_name', '')
                source_label = concept.get('source_label', source_name)
                source_desc = concept.get('source_description', '')
                
                rel_type = concept.get('relationship_type', 'relates to')
                
                related_name = concept.get('related_name', '')
                related_label = concept.get('related_label', related_name)
                related_desc = concept.get('related_description', '')
                
                ontology_prompt += f"- {source_label}"
                if rel_type:
                    ontology_prompt += f" {rel_type} "
                else:
                    ontology_prompt += " relates to "
                ontology_prompt += f"{related_label}\n"
                
                # Add descriptions if available
                if source_desc:
                    ontology_prompt += f"  * {source_label}: {source_desc}\n"
                if related_desc:
                    ontology_prompt += f"  * {related_label}: {related_desc}\n"
            
            elif 'name' in concept or 'label' in concept:
                # Format single concept
                name = concept.get('name', '')
                label = concept.get('label', name)
                description = concept.get('description', '')
                
                ontology_prompt += f"- {label}"
                if description:
                    ontology_prompt += f": {description}"
                ontology_prompt += "\n"
            
            elif 'problem_name' in concept and 'solution_name' in concept:
                # Format problem/solution
                problem = concept.get('problem_name', '')
                problem_desc = concept.get('problem_description', '')
                solution = concept.get('solution_name', '')
                solution_desc = concept.get('solution_description', '')
                
                ontology_prompt += f"- Problem: {problem}\n"
                if problem_desc:
                    ontology_prompt += f"  * Description: {problem_desc}\n"
                
                ontology_prompt += f"  * Solution: {solution}\n"
                if solution_desc:
                    ontology_prompt += f"    - {solution_desc}\n"
            
            elif 'step_name' in concept:
                # Format troubleshooting step
                step_name = concept.get('step_name', '')
                step_desc = concept.get('step_description', '')
                step_order = concept.get('step_order', 0)
                
                ontology_prompt += f"- Step {step_order}: {step_name}\n"
                if step_desc:
                    ontology_prompt += f"  * {step_desc}\n"
        
        return ontology_prompt
    
    def get_standardized_troubleshooting_steps(self, issue_type, device_type=None):
        """Get standardized troubleshooting steps from the ontology"""
        # First try to get specific steps from the ontology
        ontology_steps = self.query_troubleshooting_steps(issue_type, device_type)
        
        if ontology_steps:
            return self.format_ontology_for_prompt(ontology_steps)
        
        # If no steps found in ontology, provide generic steps based on issue type
        steps = "STANDARDIZED TROUBLESHOOTING STEPS:\n"
        
        if issue_type == "Hardware":
            steps += """
1. Verify the device is powered on and properly connected
2. Check for any physical damage or loose connections
3. Restart the device
4. Check device drivers and firmware are up to date
5. Test device functionality in Safe Mode (if applicable)
6. Try the device on another system (if possible)
7. Check manufacturer's website for known issues
"""
        elif issue_type == "Software":
            steps += """
1. Close and reopen the application
2. Restart your computer
3. Verify software version is current 
4. Check for available updates
5. Verify sufficient disk space and memory
6. Clear application cache and temporary files
7. Repair or reinstall the application
"""
        elif issue_type == "Password":
            steps += """
1. Verify caps lock is not accidentally enabled
2. Try alternative authentication methods if available
3. Use "Forgot Password" functionality for self-service reset
4. Contact IT support for assisted password reset
5. Check if account is locked due to too many failed attempts
6. Verify you're using the correct username/account
"""
        elif issue_type == "Network":
            steps += """
1. Verify physical network connections
2. Restart networking devices (router, modem, etc.)
3. Check wireless signal strength
4. Run network troubleshooter
5. Verify network settings (IP, DNS, etc.)
6. Check if issue affects all devices or just one
7. Contact ISP if the issue persists across all devices
"""
        else:
            steps += """
1. Document the specific symptoms and error messages
2. Try restarting the affected systems
3. Check for recent changes or updates
4. Look for similar issues in knowledge base
5. Test in different environments if possible
6. Contact IT support with detailed information
"""
        
        return steps
    
    def get_issue_classification(self, issue_description):
        """Classify an issue based on ontology concepts"""
        # Map common issue keywords to ontology concepts
        hardware_keywords = ["laptop", "desktop", "printer", "device", "hardware", "keyboard", 
                            "mouse", "monitor", "screen", "battery", "power", "usb", "disk"]
        
        software_keywords = ["software", "application", "program", "app", "windows", "office", 
                            "excel", "word", "outlook", "browser", "update", "install",
                            "license", "version", "freeze", "crash"]
        
        network_keywords = ["network", "wifi", "internet", "connection", "lan", "vpn", 
                            "ethernet", "dns", "ip", "wireless", "connect", "access point"]
        
        security_keywords = ["password", "login", "security", "authentication", "access", 
                            "account", "credentials", "reset", "locked", "mfa", "permission"]
        
        # Simple text matching for classification
        issue_lower = issue_description.lower()
        
        # Count keyword matches
        hw_count = sum(1 for kw in hardware_keywords if kw in issue_lower)
        sw_count = sum(1 for kw in software_keywords if kw in issue_lower)
        net_count = sum(1 for kw in network_keywords if kw in issue_lower)
        sec_count = sum(1 for kw in security_keywords if kw in issue_lower)
        
        # Determine main category and query the ontology
        category = "General"
        primary_keywords = []
        
        if sec_count > max(hw_count, sw_count, net_count):
            category = "Password"
            primary_keywords = [kw for kw in security_keywords if kw in issue_lower]
        elif hw_count > max(sw_count, net_count, sec_count):
            category = "Hardware"
            primary_keywords = [kw for kw in hardware_keywords if kw in issue_lower]
        elif sw_count > max(hw_count, net_count, sec_count):
            category = "Software" 
            primary_keywords = [kw for kw in software_keywords if kw in issue_lower]
        elif net_count > max(hw_count, sw_count, sec_count):
            category = "Network"
            primary_keywords = [kw for kw in network_keywords if kw in issue_lower]
        
        # If issue is network-related, treat as a subtype of hardware for agent selection
        agent_category = "Hardware" if category == "Network" else category
        
        return {
            "category": category,
            "agent_category": agent_category,
            "primary_keywords": primary_keywords,
            "confidence": max(hw_count, sw_count, net_count, sec_count) / 5  # Simple confidence score
        }
    
    def query_incident_management_process(self):
        """Query incident management process from ontology"""
        query = """
        MATCH path = (im:Class {uri: "http://ontology.it/itsmo/v1#IncidentManagement"})-[*1..2]-(related:Class)
        RETURN related.name as name, related.label as label, related.description as description
        """
        
        return self.query_ontology(query)
    
    def query_concepts_by_issue(self, issue_type, keywords=None):
        """Query concepts related to a specific issue type"""
        # Map issue types to ontology classes
        type_mapping = {
            "Hardware": ["Hardware", "Device", "ConfigurationItem", "Asset"],
            "Software": ["Software", "Application", "Program", "ConfigurationItem"],
            "Password": ["Authentication", "Access", "Security", "Account"],
            "Network": ["Network", "Connectivity", "Communication"]
        }
        
        class_types = type_mapping.get(issue_type, [issue_type])
        
        # Convert keywords to a list if it's a string
        if isinstance(keywords, str):
            keywords = keywords.split()
        
        # Build the query
        params = {}
        
        # Base query to search for matching classes
        query = """
        MATCH (c:Class)
        WHERE 
        """
        
        # Add class type conditions
        class_conditions = []
        for i, ct in enumerate(class_types):
            param_name = f"class{i}"
            class_conditions.append(f"c.name CONTAINS ${param_name} OR c.label CONTAINS ${param_name}")
            params[param_name] = ct
        
        query += "(" + " OR ".join(class_conditions) + ")"
        
        # Add keyword conditions if provided
        if keywords:
            query += " AND ("
            keyword_conditions = []
            for i, kw in enumerate(keywords):
                if kw:  # Skip empty keywords
                    param_name = f"kw{i}"
                    keyword_conditions.append(f"c.name CONTAINS ${param_name} OR c.label CONTAINS ${param_name} OR c.description CONTAINS ${param_name}")
                    params[param_name] = kw
            
            if keyword_conditions:
                query += " OR ".join(keyword_conditions)
            else:
                query += "1=1"  # Always true if no valid keywords
            
            query += ")"
        
        # Complete query to get related concepts
        query += """
        MATCH path = (c)-[r*0..2]-(related:Class)
        RETURN c.name as source_name, c.label as source_label, c.description as source_description,
               type(r[0]) as relationship_type,
               related.name as related_name, related.label as related_label, related.description as related_description
        LIMIT 20
        """
        
        try:
            return self.query_ontology(query, params)
        except Exception as e:
            logger.error(f"Error querying concepts by issue: {str(e)}")
            return []