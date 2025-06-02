import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from agentic_framework.agents.hierarchy_agent import HierarchyProcessor, HierarchyLevel
from agentic_framework.parsers.base_parser import BaseParser
from agentic_framework.parsers.json_parser import JsonParser
from agentic_framework.parsers.yaml_parser import YamlParser
from agentic_framework.parsers.list_parser import ListParser
from agentic_framework.parsers.base_parser import PydanticParser

# Custom Parser Example
class ProductSpecParser(BaseParser):
    """Custom parser for product specifications in a specific format."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse product specifications from text format:
        Name: Product Name
        Price: $999.99
        Rating: 4.5/5
        Features: feature1, feature2, feature3
        """
        try:
            lines = text.strip().split('\n')
            result = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'name':
                        result['name'] = value
                    elif key == 'price':
                        # Extract numeric value from price
                        price_str = value.replace('$', '').replace(',', '')
                        try:
                            result['price'] = float(price_str)
                        except ValueError:
                            result['price'] = value
                    elif key == 'rating':
                        # Extract rating number
                        if '/' in value:
                            rating_str = value.split('/')[0]
                            try:
                                result['rating'] = float(rating_str)
                            except ValueError:
                                result['rating'] = value
                        else:
                            result['rating'] = value
                    elif key == 'features':
                        # Split features by comma
                        result['features'] = [f.strip() for f in value.split(',')]
                    else:
                        result[key] = value
            
            return result if result else {"raw_text": text}
            
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}", "raw_text": text}

# Pydantic Models for structured parsing
class ProductInfo(BaseModel):
    name: str
    category: str
    price: float
    rating: Optional[float] = None
    features: List[str]

class TechSpecification(BaseModel):
    component: str
    specification: str
    importance_level: str  # high, medium, low
    technical_details: str

async def create_enhanced_hierarchy():
    """Create a hierarchy with different parsers at each level."""
    print("\n" + "="*80)
    print("ENHANCED HIERARCHY WITH MULTIPLE PARSER TYPES")
    print("="*80)
    
    # Create the hierarchy processor
    processor = HierarchyProcessor(
        output_dir="outputs/enhanced_hierarchy_results",
        parallel_processing=True,
        max_workers=3
    )
    
    # Level 1: List parser for product categories
    category_level = HierarchyLevel(
        name="category",
        prompt="""
        Give me a list of 3 popular {domain} product categories.
        Return a valid JSON array of strings containing the categories and nothing else.
        Format your response exactly like this: ["category1", "category2", "category3"]
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
    
    # Level 2: JSON parser for structured product information
    product_level = HierarchyLevel(
        name="product",
        prompt="""
        Create a JSON object with information about a popular {category} product.
        Include name, category, price (as a number), rating (as a number out of 5), and a list of 3 key features.
        
        Return ONLY valid JSON in this exact format:
        {{
            "name": "Product Name",
            "category": "{category}",
            "price": 999.99,
            "rating": 4.5,
            "features": ["feature1", "feature2", "feature3"]
        }}
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.5,
            "max_tokens": 300,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="category",
        parser_type="json",
        output_format="text"
    )
    
    # Level 3: Custom parser for product specifications
    spec_level = HierarchyLevel(
        name="specification",
        prompt="""
        For the product {product}, provide detailed specifications in this exact format:

        Name: [Product Name]
        Price: $[Price]
        Rating: [Rating]/5
        Features: [feature1, feature2, feature3]
        
        Make sure to follow this format exactly with each item on a new line.
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 250,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="product",
        parser=ProductSpecParser(),
        output_format="text"
    )
    
    # Level 4: YAML parser for configuration
    config_level = HierarchyLevel(
        name="config",
        prompt="""
        Create a YAML configuration for monitoring and managing {specification} in a system.
        Include monitoring settings, alerts, and maintenance schedules.
        
        Return ONLY valid YAML in this format:
        ```yaml
        product_monitoring:
          name: {specification}
          monitoring:
            enabled: true
            interval: 300
            metrics:
              - performance
              - availability
              - user_satisfaction
          alerts:
            email: admin@company.com
            threshold: 95
          maintenance:
            schedule: weekly
            backup: daily
        ```
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 400,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="specification",
        parser_type="yaml",
        output_format="text"
    )
    
    # Add levels to the hierarchy
    processor.add_level(category_level)
    processor.add_level(product_level, parent=category_level)
    processor.add_level(spec_level, parent=product_level)
    processor.add_level(config_level, parent=spec_level)
    
    # Process the hierarchy
    initial_data = {"domain": ["technology", "home appliances"]}
    results = await processor.process(initial_data)
    
    print(f"\nHierarchy processing completed!")
    print(f"Results structure: {type(results)}")
    
    # Print a sample of the results
    if isinstance(results, dict):
        print(f"\nTop-level keys: {list(results.keys())}")
        
        # Show a sample of the hierarchy
        for domain, domain_data in list(results.items())[:1]:  # Show first domain only
            print(f"\nSample hierarchy for '{domain}':")
            if isinstance(domain_data, dict):
                for category, category_data in list(domain_data.items())[:1]:  # Show first category
                    print(f"  Category: {category}")
                    if isinstance(category_data, dict):
                        for product, product_data in list(category_data.items())[:1]:  # Show first product
                            print(f"    Product: {product}")
                            if isinstance(product_data, dict):
                                print(f"    Product data keys: {list(product_data.keys())}")
    
    return results

async def test_pydantic_in_hierarchy():
    """Test Pydantic parser within a hierarchy."""
    print("\n" + "="*80)
    print("HIERARCHY WITH PYDANTIC PARSER")
    print("="*80)
    
    # Create the hierarchy processor
    processor = HierarchyProcessor(
        output_dir="outputs/pydantic_hierarchy_results",
        parallel_processing=True,
        max_workers=2
    )
    
    # Level 1: List parser for tech components
    component_level = HierarchyLevel(
        name="component",
        prompt="""
        Give me a list of 3 important {device_type} components.
        Return a valid JSON array of strings containing the components and nothing else.
        Format your response exactly like this: ["component1", "component2", "component3"]
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
        input_key="device_type",
        parser_type="list",
        output_format="list"
    )
    
    # Level 2: Pydantic parser for structured tech specifications
    spec_level = HierarchyLevel(
        name="tech_spec",
        prompt="""
        Extract technical specification information about {component} and return it as JSON.
        Include: component name, specification details, importance_level (high/medium/low), and technical_details.
        
        Return ONLY valid JSON in this exact format:
        {{
            "component": "{component}",
            "specification": "Brief specification description",
            "importance_level": "high",
            "technical_details": "Detailed technical information"
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
        input_key="component",
        parser=PydanticParser(TechSpecification),
        output_format="text"
    )
    
    # Add levels to the hierarchy
    processor.add_level(component_level)
    processor.add_level(spec_level, parent=component_level)
    
    # Process the hierarchy
    initial_data = {"device_type": ["smartphone", "laptop"]}
    results = await processor.process(initial_data)
    
    print(f"\nPydantic hierarchy processing completed!")
    print(f"Results type: {type(results)}")
    
    return results

async def demonstrate_custom_parser():
    """Demonstrate the custom ProductSpecParser."""
    print("\n" + "="*80)
    print("CUSTOM PARSER DEMONSTRATION")
    print("="*80)
    
    # Create an agent with the custom parser
    from agentic_framework.agents.base_agent import Agent
    
    agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.3,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser=ProductSpecParser(),
        prompt="""
        Provide product information for {product} in this exact format:

        Name: [Product Name]
        Price: $[Price]
        Rating: [Rating]/5
        Features: [feature1, feature2, feature3]
        
        Make sure to follow this format exactly with each item on a new line.
        """
    )
    
    products = ["iPhone 15 Pro", "MacBook Air M2"]
    
    for product in products:
        print(f"\nProcessing: {product}")
        result = await agent.invoke("", product=product)
        print(f"Result type: {type(result)}")
        print(f"Parsed result: {result}")

async def main():
    """Main function to run all enhanced hierarchy tests."""
    print("ENHANCED HIERARCHY WITH MULTIPLE PARSERS")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Demonstrate custom parser
    await demonstrate_custom_parser()
    
    # Test enhanced hierarchy with multiple parsers
    await create_enhanced_hierarchy()
    
    # Test Pydantic parser in hierarchy
    await test_pydantic_in_hierarchy()
    
    print("\n" + "="*80)
    print("ALL ENHANCED HIERARCHY TESTS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())