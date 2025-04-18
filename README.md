# ME.ai - Multilingual Empathetic AI for IT Support

An AI-powered IT support system that uses specialized agents to handle different types of IT issues.

## Architecture Overview

This system uses LangChain as the framework for orchestrating specialized AI agents:

- **Hardware Agent**: Handles hardware-related issues
- **Software Agent**: Handles software-related issues
- **Password Agent**: Handles authentication and access issues

## Setup Instructions

### Prerequisites

- Python 3.9+
- AWS account with Bedrock access
- Database service running (uses existing ME.ai database)

### Installation

1. Clone the repository:


ME.ai LangChain-Based Architecture
1. Core Components to Add

LangChain Integration Layer

Chain orchestrator for agent workflows
Memory system for conversation context
Prompt templates manager
Tool integration system


Agent Framework

Primary orchestrator agent
Specialized agents (hardware, software, password)
Planning agent for workflow decisions


Knowledge Infrastructure

Vector database for IT knowledge (optional at MVP stage)
Enhanced ontology management



Implementation Plan
Let me outline a step-by-step implementation that builds on your existing code:
project/
├── agent/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── hardware_agent.py
│   ├── software_agent.py
│   ├── password_agent.py
│   └── orchestrator.py
├── chains/
│   ├── __init__.py
│   ├── conversation.py
│   └── workflow.py
├── memory/
│   ├── __init__.py
│   └── session_memory.py
├── tools/
│   ├── __init__.py
│   ├── db_tools.py
│   └── device_tools.py
├── prompts/
│   ├── __init__.py
│   └── templates.py
├── existing/
│   ├── db_service.py
│   ├── session_manager.py
│   └── response_generator.py
├── me_agent_orchestrator.py
└── requirements.txt
