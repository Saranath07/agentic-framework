import os
from dotenv import load_dotenv

from agentic_framework import HierarchyProcessor, HierarchyLevel, create_domain_hierarchy

def main():
    # Load environment variables
    load_dotenv()
    
    print("Hierarchy Processor Example")
    print("=" * 50)
    
    # Example 1: Using the convenience function for domain-subdomain hierarchy
    print("\nExample 1: Domain-Subdomain Hierarchy using convenience function")
    print("-" * 50)
    
    domains = ["cricket", "football", "dance"]
    
    # Use the convenience function to create and process the hierarchy
    results = create_domain_hierarchy(
        domains=domains,
        output_dir="hierarchy_results",
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        }
    )
    
    # Example 2: Creating a custom 3-level hierarchy
    print("\nExample 2: Custom 3-level hierarchy")
    print("-" * 50)
    
    # Create the hierarchy processor
    processor = HierarchyProcessor(output_dir="hierarchy_results/custom")
    
    # Define the levels
    category_level = HierarchyLevel(
        name="category",
        prompt="""
        Give me a list of 3 popular {category} products.
        Return a valid JSON array of strings containing the products and nothing else.
        Format your response exactly like this: ["product1", "product2", "product3"]
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
        input_key="category",  # Explicitly set the input key
        parser_type="list",
        output_format="list"
    )
    
    product_level = HierarchyLevel(
        name="product",
        prompt="""
        List 3 key features of {product}.
        Return a valid JSON array of strings containing the features and nothing else.
        Format your response exactly like this: ["feature1", "feature2", "feature3"]
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
        input_key="product",  # Explicitly set the input key
        parser_type="list",
        output_format="list"
    )
    
    feature_level = HierarchyLevel(
        name="feature",
        prompt="""
        Explain the benefit of {feature} in a single sentence.
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        input_key="feature",  # Explicitly set the input key
        output_format="text"
    )
    
    # Add levels to the hierarchy
    processor.add_level(category_level)
    processor.add_level(product_level, parent=category_level)
    processor.add_level(feature_level, parent=product_level)
    
    # Process the hierarchy
    initial_data = {"category": ["smartphones", "laptops"]}
    custom_results = processor.process(initial_data)
    
    print("\nProcessing complete! Check the output directories for results.")

if __name__ == "__main__":
    main()