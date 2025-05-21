# Agentic Framework

A flexible and extensible framework for creating LLM-powered agents with different models and parsers.

## Features

- Support for multiple LLM providers (OpenAI, Groq, GitHub Copilot, DeepInfra)
- Flexible parsing options (JSON, YAML, Pydantic models)
- Easy agent creation with customizable prompts
- Detailed logging and metadata tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-framework.git
cd agentic-framework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from agentic_framework import Agent

# Create a simple agent
agent = Agent(
    llm_type="openai",
    model="gpt-4",
    prompt="Answer the following question: {query}"
)

# Invoke the agent
response = agent.invoke("What is the capital of France?")
print(response)
```

## Creating Agents

The framework makes it easy to create agents with different LLM providers and parsers:

### Basic Agent

```python
agent = Agent(
    llm_type="openai",  # Service provider (openai, groq, github)
    model="gpt-4",      # Model name
    prompt="Answer the following question: {query}"
)
```

### Agent with JSON Parser

```python
agent = Agent(
    llm_type="openai",
    model="gpt-4",
    model_args={"temperature": 0.2},
    parser_type="json",
    prompt="""
    Answer the following question and format your response as a JSON object:
    
    {query}
    
    Your response should be a valid JSON object.
    """
)
```

### Agent with Pydantic Parser

```python
from pydantic import BaseModel, Field
from typing import List
from agentic_framework import Agent, PydanticParser

class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    skills: List[str] = Field(description="List of person's skills")

parser = PydanticParser(model_class=PersonInfo)

agent = Agent(
    llm_type="openai",
    model="gpt-4",
    model_args={"temperature": 0.2},
    parser=parser,
    prompt="""
    Create a profile for a fictional person based on the following description:
    
    {query}
    
    Return your response as a JSON object with the following structure:
    {
        "name": "Full Name",
        "age": 30,
        "occupation": "Job Title",
        "skills": ["skill1", "skill2", "skill3"]
    }
    """
)
```

## Environment Variables

The framework uses environment variables for API keys:

```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
DEEPINFRA_API_KEY=your_deepinfra_api_key
BASE_URL=https://api.deepinfra.com/v1/openai  # For DeepInfra
```

You can also set these in a `.env` file in the project root.

## Using DeepInfra with Llama Models

The framework supports using DeepInfra's API to access Llama models:

```python
agent = Agent(
    llm_type="openai",  # Using OpenAI client with DeepInfra base URL
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    model_args={
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_key": os.getenv("DEEPINFRA_API_KEY"),
        "base_url": "https://api.deepinfra.com/v1/openai"
    },
    prompt="Answer the following question: {query}"
)
```

## Examples

See the `examples.py` file for more examples of how to use the framework.

## License

MIT