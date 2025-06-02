import os
import asyncio
from dotenv import load_dotenv

from agentic_framework import (
    Agent, 
    HierarchyProcessor, 
    HierarchyLevel, 
    JsonParser, 
    YamlParser, 
    ListParser, 
    PydanticParser
)
from pydantic import BaseModel
from typing import List, Dict, Any

# Pydantic models for structured parsing
class ProductInfo(BaseModel):
    name: str
    category: str
    price: float
    features: List[str]

class CompanyInfo(BaseModel):
    name: str
    industry: str
    founded: int
    employees: int

async def test_json_parser():
    """Test JSON parser with product information extraction."""
    print("\n" + "="*60)
    print("TESTING JSON PARSER")
    print("="*60)
    
    agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 300,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser_type="json",
        prompt="""
        Create a JSON object with information about {product}.
        Include name, category, price (as a number), and a list of 3 key features.
        
        Return ONLY valid JSON in this exact format:
        {{
            "name": "product name",
            "category": "product category", 
            "price": 999.99,
            "features": ["feature1", "feature2", "feature3"]
        }}
        """
    )
    
    products = ["iPhone 15", "Tesla Model 3", "MacBook Pro"]
    
    for product in products:
        print(f"\nProcessing: {product}")
        result = await agent.invoke("", product=product)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

async def test_yaml_parser():
    """Test YAML parser with configuration generation."""
    print("\n" + "="*60)
    print("TESTING YAML PARSER")
    print("="*60)
    
    agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 300,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser_type="yaml",
        prompt="""
        Create a YAML configuration for a {service_type} service.
        Include service name, port, environment variables, and dependencies.
        
        Return ONLY valid YAML in this format:
        ```yaml
        service:
          name: service-name
          port: 8080
          environment:
            - NODE_ENV=production
            - LOG_LEVEL=info
          dependencies:
            - database
            - redis
        ```
        """
    )
    
    services = ["web application", "API gateway", "microservice"]
    
    for service in services:
        print(f"\nProcessing: {service}")
        result = await agent.invoke("", service_type=service)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

async def test_list_parser():
    """Test List parser with recommendation generation."""
    print("\n" + "="*60)
    print("TESTING LIST PARSER")
    print("="*60)
    
    agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser_type="list",
        prompt="""
        Give me a list of 5 {item_type} recommendations.
        Return a valid JSON array of strings containing the recommendations and nothing else.
        Format your response exactly like this: ["item1", "item2", "item3", "item4", "item5"]
        Do not include any explanations, headers, or additional text.
        """
    )
    
    categories = ["programming books", "productivity tools", "healthy snacks"]
    
    for category in categories:
        print(f"\nProcessing: {category}")
        result = await agent.invoke("", item_type=category)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

async def test_pydantic_parser():
    """Test Pydantic parser with structured data extraction."""
    print("\n" + "="*60)
    print("TESTING PYDANTIC PARSER")
    print("="*60)
    
    # Create parser with Pydantic model
    pydantic_parser = PydanticParser(CompanyInfo)
    
    agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser=pydantic_parser,
        prompt="""
        Extract information about {company} and return it as JSON.
        Include: name, industry, founded year (as integer), and approximate number of employees (as integer).
        
        Return ONLY valid JSON in this exact format:
        {{
            "name": "Company Name",
            "industry": "Industry Type",
            "founded": 2000,
            "employees": 50000
        }}
        """
    )
    
    companies = ["Apple", "Netflix", "SpaceX"]
    
    for company in companies:
        print(f"\nProcessing: {company}")
        try:
            result = await agent.invoke("", company=company)
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            if hasattr(result, 'model_dump'):
                print(f"Structured data: {result.model_dump()}")
        except Exception as e:
            print(f"Error processing {company}: {e}")

async def test_hierarchy_with_different_parsers():
    """Test hierarchy processing with different parsers at each level."""
    print("\n" + "="*60)
    print("TESTING HIERARCHY WITH DIFFERENT PARSERS")
    print("="*60)
    
    # Create the hierarchy processor
    processor = HierarchyProcessor(
        output_dir="outputs/parser_showcase_results",
        parallel_processing=True,
        max_workers=2
    )
    
    # Level 1: List parser for generating topics
    topic_level = HierarchyLevel(
        name="topic",
        prompt="""
        Give me a list of 3 popular {domain} topics.
        Return a valid JSON array of strings containing the topics and nothing else.
        Format your response exactly like this: ["topic1", "topic2", "topic3"]
        Do not include any explanations, headers, or additional text.
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="domain",
        parser_type="list",
        output_format="list"
    )
    
    # Level 2: JSON parser for structured information
    info_level = HierarchyLevel(
        name="info",
        prompt="""
        Create a JSON object with detailed information about {topic}.
        Include: title, description, difficulty_level (beginner/intermediate/advanced), and estimated_time_hours (as number).
        
        Return ONLY valid JSON in this exact format:
        {{
            "title": "Topic Title",
            "description": "Brief description of the topic",
            "difficulty_level": "beginner",
            "estimated_time_hours": 10
        }}
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 300,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="topic",
        parser_type="json",
        output_format="text"
    )
    
    # Level 3: Plain text for resources
    resource_level = HierarchyLevel(
        name="resource",
        prompt="""
        Recommend one excellent learning resource for {info}.
        Provide just the resource name and a brief reason why it's good.
        Keep the response to 1-2 sentences maximum.
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.5,
            "max_tokens": 150,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="info",
        output_format="text"
    )
    
    # Add levels to the hierarchy
    processor.add_level(topic_level)
    processor.add_level(info_level, parent=topic_level)
    processor.add_level(resource_level, parent=info_level)
    
    # Process the hierarchy
    initial_data = {"domain": ["machine learning", "web development"]}
    results = await processor.process(initial_data)
    
    print(f"\nHierarchy processing completed!")
    print(f"Results keys: {list(results.keys())}")

async def main():
    """Main function to run all parser tests."""
    print("PARSER SHOWCASE - Testing Different Parser Types")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Test individual parsers
    await test_json_parser()
    await test_yaml_parser()
    await test_list_parser()
    await test_pydantic_parser()
    
    # Test hierarchy with different parsers
    await test_hierarchy_with_different_parsers()
    
    print("\n" + "="*80)
    print("ALL PARSER TESTS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())