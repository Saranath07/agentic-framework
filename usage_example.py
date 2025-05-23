import os
import re
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from agentic_framework import Agent, JsonParser, PydanticParser
from agentic_framework.parsers.base_parser import BaseParser


class BatchProcessingAgent:
    """
    Agent that processes a list of items by substituting each item into a prompt template.
    """
    
    def __init__(
        self,
        base_agent: Agent,
        placeholder: str = "{item}"
    ):
        """
        Initialize a batch processing agent.
        
        Args:
            base_agent: The base agent to use for processing each item
            placeholder: The placeholder string to replace in the prompt (default: "{item}")
        """
        self.base_agent = base_agent
        self.placeholder = placeholder
        self.results = {}
    
    def process_batch(self, items: List[str], prompt_template: str) -> Dict[str, Any]:
        """
        Process a batch of items using the base agent.
        
        Args:
            items: List of items to process
            prompt_template: Prompt template with a placeholder to be replaced with each item
            
        Returns:
            Dictionary mapping items to their results
        """
        results = {}
        
        for item in items:
            # Replace the placeholder with the current item
            prompt = prompt_template.replace(self.placeholder, item)
            
            # Process the prompt using the base agent
            result = self.base_agent.invoke(prompt)
            
            # Store the result
            results[item] = result
            
        self.results = results
        return results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the batch processing.
        
        Returns:
            Dictionary mapping items to their results
        """
        return self.results

class KeyValueParser(BaseParser):
    """Parser that converts text with ###key markers into a dictionary."""
    
    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse text with ###key format into a dictionary.
        
        Format:
        ###key1
        value1
        
        ###key2
        value2
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary with keys and values
        """
        # Split by ### markers, ignore the first empty part
        parts = re.split(r'###', text)
        result = {}
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # First line is the key, rest is the value
            lines = part.split('\n', 1)
            if len(lines) == 2:
                key = lines[0].strip()
                value = lines[1].strip()
                result[key] = value
        
        return result


class KeyValueData(BaseModel):
    """Pydantic model for key-value data."""
    key1: str = Field(description="Value for key1")
    key2: str = Field(description="Value for key2")


class SentimentAnalysis(BaseModel):
    """Pydantic model for sentiment analysis results."""
    sentiment: str = Field(description="Overall sentiment (positive, negative, neutral)")
    score: float = Field(description="Sentiment score from -1.0 (negative) to 1.0 (positive)")
    key_phrases: List[str] = Field(description="Key phrases that influenced the sentiment")


def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example text to analyze
    text = """
    The new smartphone has an excellent camera and impressive battery life.
    However, the user interface is somewhat confusing and the price is quite high.
    Overall, it's a good device for photography enthusiasts who don't mind the learning curve.
    """
    
    print("Original text:")
    print(text)
    print("\n" + "-" * 50 + "\n")
    
    # Create a summarization agent using DeepInfra's Llama model
    summarizer = Agent(
        llm_type="openai",  # Using OpenAI client with DeepInfra base URL
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 100,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="""
        Please summarize the following text in a concise way:
        
        {query}
        
        Provide a summary that captures the main points in 1-2 sentences.
        """
    )
    
    # Create a sentiment analysis agent with a Pydantic parser
    sentiment_parser = PydanticParser(model_class=SentimentAnalysis)
    sentiment_analyzer = Agent(
        llm_type="openai",  # Using OpenAI client with DeepInfra base URL
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.2,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser=sentiment_parser,
        system_prompt="""You are a sentiment analysis assistant that ONLY returns valid JSON.
        Your responses must ALWAYS be valid JSON objects without any additional text, explanations, or markdown formatting.
        Do not include ```json or ``` markers in your response.
        Do not explain your reasoning or provide any text outside of the JSON object.""",
        prompt="""
        Analyze the sentiment of the following text:
        
        {query}
        
        Return ONLY a JSON object with the following structure, without any explanation or markdown formatting:
        {{
            "sentiment": "positive/negative/neutral",
            "score": 0.0,  # A value between -1.0 (very negative) and 1.0 (very positive)
            "key_phrases": ["phrase1", "phrase2", "phrase3"]
        }}
        
        IMPORTANT: Return ONLY the JSON object, with no additional text, explanations, or markdown formatting.
        """
    )
    
    # Use the summarization agent
    summary = summarizer.invoke(text)
    print("Summary:")
    print(summary)
    print("\n" + "-" * 50 + "\n")
    
    # Use the sentiment analysis agent
    sentiment = sentiment_analyzer.invoke(text)
    print("Sentiment Analysis:")
    print(f"Sentiment: {sentiment.sentiment}")
    print(f"Score: {sentiment.score}")
    print("Key phrases:")
    for phrase in sentiment.key_phrases:
        print(f"- {phrase}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Example custom input with ###key format
    custom_input = """
    ###key1
    This is the value for key1
    
    ###key2
    This is the value for key2
    """
    
    # Create a key-value parser and agent
    kv_parser = KeyValueParser()
    kv_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.2,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser=kv_parser,
        system_prompt="""You are an assistant that processes structured data.
        Your task is to extract and return the exact content provided in the input without modification.""",
        prompt="""
        Process the following structured data:
        
        {query}
        
        Return the data in the same format without any additional text or explanations.
        """
    )
    
    # Use the key-value agent
    result = kv_agent.invoke(custom_input)
    print("Key-Value Parsing Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Example of batch processing with a list of people
    people = ["Virat kohli", "Naga Chaitanya", "Mukesh Ambani"]
    prompt_template = "What is the father's name of {person} and his {surname}?"

    my_dict = {
        "person" : ["Virat kohli", "Naga Chaitanya", "Mukesh Ambani"],
        "surname" : ["Kohli", "Akkineni", "Ambani"]
    }
    
    # There could be combinations
    # 1. All combinations
    # 2. One one mapping
    # 3. random sampling


    # Create a simple agent for answering questions
    qa_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 100,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="{query}"
    )

    # Either model or multiple model

    # Uniform 
    
    # Create a batch processing agent
    batch_agent = BatchProcessingAgent(
        base_agent=qa_agent,
        placeholder="{person}"
    )

    # batch mode
    
    # Ouptut of the first agent is a list and then use this list to get another list using another agent

    # Multiprocessing 


    
    # Process the batch of people
    print("Batch Processing Results:")
    results = batch_agent.process_batch(people, prompt_template)
    
    # Print the results
    for person, result in results.items():
        print(f"{person}'s father: {result}")


if __name__ == "__main__":
    main()