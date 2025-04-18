# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_orchestrator_key')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
PORT = int(os.environ.get('PORT', 8000))

# AI Models configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-643f3d962fef49949b7a719e2dc38e83')
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')


# Qwen configuration (if used)
QWEN_API_KEY = os.environ.get('QWEN_API_KEY', 'sk-ce99dbfb59df4f8c94d6f78aba7d6221')
QWEN_API_URL = os.environ.get('QWEN_API_URL', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1')

# Database service configuration
MEAI_DB_SERVICE = os.environ.get('MEAI_DB_SERVICE', 'http://127.0.0.1:5000/api')
DB_USERNAME = os.environ.get('DB_USERNAME', 'testadmin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'testpass')

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Feature flags
ENABLE_MFA = os.environ.get('ENABLE_MFA', 'False').lower() == 'true'
USE_VECTOR_DB = os.environ.get('USE_VECTOR_DB', 'False').lower() == 'true'
