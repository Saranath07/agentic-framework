import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from agentic_framework import Agent, JsonParser, YamlParser, PydanticParser


def create_simple_agent():
    """
    Create a simple agent that uses OpenAI's GPT model with default settings.
    """
    agent = Agent(
        llm_type="openai",
        model="gpt-4",
        prompt="Answer the following question: {query}"
    )
    return agent


def create_json_agent():
    """
    Create an agent that returns structured JSON data.
    """
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
    return agent


def create_yaml_agent():
    """
    Create an agent that returns structured YAML data.
    """
    agent = Agent(
        llm_type="openai",
        model="gpt-4",
        model_args={"temperature": 0.2},
        parser_type="yaml",
        prompt="""
        Answer the following question and format your response as YAML:
        
        {query}
        
        Your response should be valid YAML.
        """
    )
    return agent


class PersonInfo(BaseModel):
    """Example Pydantic model for structured output."""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    skills: List[str] = Field(description="List of person's skills")


def create_pydantic_agent():
    """
    Create an agent that returns data structured according to a Pydantic model.
    """
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
    return agent


def create_deepinfra_agent():
    """
    Create an agent that uses DeepInfra's Llama-3.3-70B-Instruct-Turbo model.
    """
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
    return agent


def create_groq_agent():
    """
    Create an agent that uses Groq's LLM.
    """
    agent = Agent(
        llm_type="groq",
        model="llama-3.1-70b-versatile",
        model_args={"temperature": 0.7, "max_tokens": 1024},
        prompt="Answer the following question: {query}"
    )
    return agent


def create_github_agent():
    """
    Create an agent that uses GitHub's Copilot.
    """
    agent = Agent(
        llm_type="github",
        model="copilot-chat",
        prompt="Answer the following question: {query}"
    )
    return agent


# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create a DeepInfra Llama agent
    print("Creating DeepInfra Llama agent...")
    deepinfra_agent = create_deepinfra_agent()
    response = deepinfra_agent.invoke("What is the capital of France?")
    print(f"DeepInfra Llama Agent Response: {response}")
    print("\n" + "-" * 50 + "\n")
    
    # Create a simple agent
    simple_agent = create_simple_agent()
    response = simple_agent.invoke("What is the capital of France?")
    print(f"Simple Agent Response: {response}")
    
    # Create a JSON agent
    json_agent = create_json_agent()
    response = json_agent.invoke("List the top 3 programming languages in 2023")
    print(f"JSON Agent Response: {response}")
    
    # Create a YAML agent
    yaml_agent = create_yaml_agent()
    response = yaml_agent.invoke("Describe a recipe for chocolate chip cookies")
    print(f"YAML Agent Response: {response}")
    
    # Create a Pydantic agent
    pydantic_agent = create_pydantic_agent()
    response = pydantic_agent.invoke("A 35-year-old software engineer who loves hiking")
    print(f"Pydantic Agent Response: {response}")