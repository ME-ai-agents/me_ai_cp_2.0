# rag_knowledge_base_bedrock.py
import logging
import boto3
import json
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger('rag_knowledge_base')

class RAGKnowledgeBase:
    """Retrieval-Augmented Generation Knowledge Base using AWS Bedrock Knowledge Base"""
    
    def __init__(self, aws_region, knowledge_base_id, 
                 aws_access_key=None, aws_secret_key=None,
                 model_id="amazon.titan-embed-text-v1"):
        self.aws_region = aws_region
        self.knowledge_base_id = knowledge_base_id
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.model_id = model_id
        
        self.bedrock_client = None
        self.bedrock_kb_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS Bedrock clients"""
        try:
            # Create AWS session
            session = boto3.Session(
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            
            # Initialize Bedrock Runtime client (for embeddings if needed)
            self.bedrock_client = session.client('bedrock-runtime')
            
            # Initialize Bedrock Agent Runtime client (for knowledge base queries)
            self.bedrock_kb_client = session.client('bedrock-agent-runtime')
            
            logger.info(f"Successfully initialized AWS Bedrock clients in region {self.aws_region}")
        except Exception as e:
            logger.error(f"Error initializing AWS Bedrock clients: {str(e)}")
    
    def query(self, question, k=3):
        """Query the knowledge base for relevant documents"""
        if not self.bedrock_kb_client:
            logger.error("No Bedrock Knowledge Base client available")
            return []
        
        try:
            # Call Bedrock Knowledge Base API
            response = self.bedrock_kb_client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={
                    'text': question
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': k
                    }
                }
            )
            
            # Process results
            retrieval_results = response.get('retrievalResults', [])
            
            documents = []
            for result in retrieval_results:
                # Extract content and metadata
                content = result.get('content', {}).get('text', '')
                
                # Get metadata (location, title, etc.)
                metadata = {}
                location = result.get('location', {})
                if location:
                    metadata['source'] = location.get('s3Location', {}).get('uri', '')
                
                # Get score (confidence)
                score = result.get('score', 0)
                
                documents.append({
                    "content": content,
                    "metadata": metadata,
                    "score": score
                })
            
            logger.info(f"Retrieved {len(documents)} documents from Bedrock Knowledge Base")
            return documents
        except Exception as e:
            logger.error(f"Error querying Bedrock Knowledge Base: {str(e)}")
            return []
    
    def format_documents_for_prompt(self, documents):
        """Format retrieved documents for inclusion in prompt"""
        if not documents:
            return ""
        
        docs_prompt = "KNOWLEDGE BASE INFORMATION:\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            
            docs_prompt += f"Document {i}:\n"
            
            # Add metadata if available
            if metadata:
                source = metadata.get("source", "Unknown source")
                docs_prompt += f"Source: {source}\n"
            
            # Add content with formatting
            docs_prompt += f"Content:\n{content}\n\n"
        
        return docs_prompt
    
    def get_relevant_documents(self, query, issue_type=None, k=3):
        """Get relevant documents with optional issue type filtering"""
        # Get documents based on the query
        documents = self.query(query, k=k)
        
        # If issue type is provided, we could add it to the query
        # For Bedrock Knowledge Base, we might need to do filtering client-side
        # For now, we'll just return the documents as is
        return documents