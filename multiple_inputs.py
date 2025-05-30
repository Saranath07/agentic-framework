import os
from dotenv import load_dotenv

from agentic_framework import HierarchyProcessor, HierarchyLevel, create_domain_hierarchy

def main():
    # Load environment variables
    load_dotenv()
    

    
    # Create the hierarchy processor with parallel processing for multiple placeholders
    multi_processor = HierarchyProcessor(
        output_dir="hierarchy_results/multi",
        parallel_processing=True, 
        max_workers=4  
    )
    
    # Define the levels with multiple placeholders
    product_review_level = HierarchyLevel(
        name="product",
        prompt="""
        Write a short review for {product} considering the {aspect} aspect.
        Keep it under 100 words.
        """,
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        # Define multiple input keys
        input_key=["product", "aspect"],
        output_format="text"
    )
    
 
    multi_processor.add_level(product_review_level)
    
    # Process the hierarchy with multiple input values
    initial_data = {
        "product": ["iPhone 15", "MacBook Pro"],
        "aspect": ["battery life", "performance", "design"]
    }

    results_2 = multi_processor.process(initial_data)

    print(results_2)
    
    
   
if __name__ == "__main__":
    main()