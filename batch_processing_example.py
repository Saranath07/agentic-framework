import os
from typing import Dict, List, Any
import json
from dotenv import load_dotenv

from agentic_framework import Agent, BatchProcessingAgent

def main():
    # Load environment variables
    load_dotenv()
    
    print("Batch Processing Example")
    print("=" * 50)
    
    # Example 1: Create an agent for getting subdomains
    subdomain_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="Give me a list of 5 subdomains of {domain}. Return only the list, one subdomain per line."
    )
    
    # Create a batch processing agent for subdomains
    batch_agent = BatchProcessingAgent(
        base_agent=subdomain_agent,
        output_dir="batch_results"
    )
    
    # Example 1: Process domains to get subdomains
    print("\nExample 1: Getting subdomains for different domains")
    print("-" * 50)
    
    domains = {
        "domain": ["cricket", "football", "dance"]
    }
    
    # Process the batch
    results = batch_agent.process_batch(domains)
    
    # Save results to a JSONL file
    output_file = batch_agent.save_results("domain_subdomains")
    
    print(f"Processed {len(results)} domains")
    print(f"Results saved to: {output_file}")
    
    # Print sample results
    print("\nSample results:")
    for key, result in list(results.items())[:1]:  # Show first result only
        print(f"\nDomain: {result['input']['domain']}")
        print("Subdomains:")
        print(result['output'])
    
    # Example 2: Process with multiple placeholders
    print("\nExample 2: Processing with multiple placeholders")
    print("-" * 50)
    
    # Create an agent for father information
    father_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="What is the father's name of {person} {surname}?"
    )
    
    # Create a batch processing agent for father information
    father_batch_agent = BatchProcessingAgent(
        base_agent=father_agent,
        output_dir="batch_results"
    )
    
    # Dictionary with multiple placeholders
    person_data = {
        "person": ["Virat", "Naga Chaitanya", "Mukesh"],
        "surname": ["Kohli", "Akkineni", "Ambani"]
    }
    
    # Process the batch with one-to-one mapping
    results = father_batch_agent.process_batch(
        person_data,
        combination_method="one_to_one"  # Match items at the same index
    )
    
    # Save results to a JSONL file
    output_file = father_batch_agent.save_results("person_father_info")
    
    print(f"Processed {len(results)} person queries")
    print(f"Results saved to: {output_file}")
    
    # Print sample results
    print("\nSample results:")
    for key, result in list(results.items())[:1]:  # Show first result only
        print(f"\nPerson: {result['input']['person']}")
        print(f"Surname: {result['input']['surname']}")
        print("Response:")
        print(result['output'])
    
    # Example 3: Process with all combinations
    print("\nExample 3: Processing with all combinations")
    print("-" * 50)
    
    # Create an agent for product features
    features_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="List 3 key features of a {brand} {product}. Return only the list, one feature per line."
    )
    
    # Create a batch processing agent for product features
    features_batch_agent = BatchProcessingAgent(
        base_agent=features_agent,
        output_dir="batch_results"
    )
    
    # Dictionary with multiple placeholders for all combinations
    combo_data = {
        "product": ["smartphone", "laptop"],
        "brand": ["Apple", "Samsung", "Google"]
    }
    
    # Process the batch with all combinations
    results = features_batch_agent.process_batch(
        combo_data,
        combination_method="all_combinations"  # Generate all possible combinations
    )
    
    # Save results to a JSONL file
    output_file = features_batch_agent.save_results("product_features")
    
    print(f"Processed {len(results)} product-brand combinations")
    print(f"Results saved to: {output_file}")
    
    # Print sample results
    print("\nSample results:")
    for key, result in list(results.items())[:1]:  # Show first result only
        print(f"\nProduct: {result['input']['product']}")
        print(f"Brand: {result['input']['brand']}")
        print("Features:")
        print(result['output'])
    
    # Example 4: Process and save in one operation
    print("\nExample 4: Process and save in one operation")
    print("-" * 50)
    
    # Dictionary with domains for subdomain processing
    more_domains = {
        "domain": ["basketball", "swimming", "chess"]
    }
    
    # Process the batch and save results in one operation
    output_file = batch_agent.process_and_save(
        more_domains,
        filename="more_domain_subdomains"
    )
    
    print(f"Processed and saved {len(batch_agent.get_results())} domains")
    print(f"Results saved to: {output_file}")
    
    # Example 5: Using the results for further processing
    print("\nExample 5: Using the results for further processing")
    print("-" * 50)
    
    # Function to extract subdomains from the output
    def extract_subdomains(output_text):
        return [line.strip() for line in output_text.strip().split('\n') if line.strip()]
    
    # Load the results from the first example
    subdomains_by_domain = {}
    with open("batch_results/domain_subdomains.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            domain = data["input"]["domain"]
            subdomains = extract_subdomains(data["output"])
            subdomains_by_domain[domain] = subdomains
    
    print("Extracted subdomains by domain:")
    for domain, subdomains in subdomains_by_domain.items():
        print(f"\n{domain.capitalize()}:")
        for i, subdomain in enumerate(subdomains, 1):
            print(f"  {i}. {subdomain}")
    
    # Now we can use these subdomains for further processing
    # For example, we can create a new batch processing job for each subdomain
    
    # Select one domain for demonstration
    selected_domain = "cricket"
    if selected_domain in subdomains_by_domain:
        print(f"\nFurther processing for {selected_domain} subdomains:")
        
        # Create a dictionary with the subdomains
        subdomain_data = {
            "subdomain": subdomains_by_domain[selected_domain]
        }
        
        # Create an agent for players in subdomains
        players_agent = Agent(
            llm_type="openai",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            model_args={
                "temperature": 0.7,
                "max_tokens": 200,
                "api_key": os.getenv("DEEPINFRA_API_KEY"),
                "base_url": "https://api.deepinfra.com/v1/openai"
            },
            prompt="Give me 3 famous players in {subdomain}. Return only the list, one player per line."
        )
        
        # Create a batch processing agent for players
        players_batch_agent = BatchProcessingAgent(
            base_agent=players_agent,
            output_dir="batch_results"
        )
        
        # Process the batch
        output_file = players_batch_agent.process_and_save(
            subdomain_data,
            filename=f"{selected_domain}_subdomain_players"
        )
        
        print(f"Processed {len(players_batch_agent.get_results())} subdomains of {selected_domain}")
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()