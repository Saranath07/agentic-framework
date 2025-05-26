import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

from agentic_framework import Agent, BatchProcessingAgent

def main():
    # Load environment variables
    load_dotenv()
    
    print("Domain-Subdomain Processor")
    print("=" * 50)
    
    # Create an agent for getting subdomains
    subdomain_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        parser_type="list",  # Use the new list parser
        prompt="""
        Give me a list of 5 subdomains of {domain}.
        Return a valid JSON array of strings containing the subdomains and nothing else.
        Format your response exactly like this: ["subdomain1", "subdomain2", "subdomain3", "subdomain4", "subdomain5"]
        Do not include any explanations, headers, or additional text.
        """
    )
    
    # Create a batch processing agent for subdomains
    subdomain_batch_agent = BatchProcessingAgent(
        base_agent=subdomain_agent,
        output_dir="domain_results"
    )
    
    # List of domains to process
    domains = {
        "domain": [
            "cricket", 
            "football", 
            "dance", 
            "music", 
            "technology"
        ]
    }
    
    # Step 1: Get subdomains for each domain
    print("\nStep 1: Getting subdomains for each domain")
    print("-" * 50)
    
    # Process the batch and save results
    subdomains_file = subdomain_batch_agent.process_and_save(
        domains,
        filename="domain_subdomains"
    )
    
    print(f"Processed {len(subdomain_batch_agent.get_results())} domains")
    print(f"Subdomains saved to: {subdomains_file}")
    
    # Step 2: Extract subdomains from the results
    print("\nStep 2: Extracting subdomains from results")
    print("-" * 50)
    
    # Load the results
    subdomains_by_domain = {}
    with open(subdomains_file, "r") as f:
        for line in f:
            data = json.loads(line)
            domain = data["input"]["domain"]
            # The output is already a list thanks to the list parser
            subdomains = data["output"] if isinstance(data["output"], list) else []
            subdomains_by_domain[domain] = subdomains
    
    # Print the extracted subdomains
    for domain, subdomains in subdomains_by_domain.items():
        print(f"\n{domain.capitalize()} subdomains:")
        for subdomain in subdomains:
            print(f"  - {subdomain}")
    
    # Step 3: Process each subdomain to get more information
    print("\nStep 3: Processing each subdomain for more information")
    print("-" * 50)
    
    # Create an agent for getting facts about subdomains
    facts_agent = Agent(
        llm_type="openai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args={
            "temperature": 0.7,
            "max_tokens": 200,
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        prompt="""
        Provide 3 key facts about {subdomain}.
        Return ONLY the facts, one per line, without numbering or any other text.
        """
    )
    
    # Create a batch processing agent for facts
    facts_batch_agent = BatchProcessingAgent(
        base_agent=facts_agent,
        output_dir="domain_results"
    )
    
    # Process each domain's subdomains separately
    for domain, subdomains in subdomains_by_domain.items():
        print(f"\nProcessing subdomains of {domain}...")
        
        # Create a dictionary with the subdomains
        subdomain_data = {
            "subdomain": subdomains
        }
        
        # Process the batch and save results
        output_file = facts_batch_agent.process_and_save(
            subdomain_data,
            filename=f"{domain}_subdomain_facts"
        )
        
        print(f"Processed {len(facts_batch_agent.get_results())} subdomains of {domain}")
        print(f"Results saved to: {output_file}")
    
    # Step 4: Combine all results into a single structured file
    print("\nStep 4: Combining all results into a structured file")
    print("-" * 50)
    
    # Create a hierarchical structure of domains, subdomains, and facts
    domain_hierarchy = {}
    
    # First, add all domains and their subdomains
    for domain, subdomains in subdomains_by_domain.items():
        domain_hierarchy[domain] = {subdomain: [] for subdomain in subdomains}
    
    # Then, add facts for each subdomain
    for domain in domain_hierarchy:
        facts_file = f"domain_results/{domain}_subdomain_facts.jsonl"
        try:
            with open(facts_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    subdomain = data["input"]["subdomain"]
                    # For facts, we need to split by lines instead of parsing as JSON
                    facts = data["output"].strip().split('\n') if data["output"].strip() else []
                    if subdomain in domain_hierarchy[domain]:
                        domain_hierarchy[domain][subdomain] = facts
        except FileNotFoundError:
            print(f"Warning: Facts file not found for {domain}")
    
    # Save the combined results
    combined_file = "domain_results/combined_domain_hierarchy.json"
    with open(combined_file, "w") as f:
        json.dump(domain_hierarchy, f, indent=2)
    
    print(f"Combined results saved to: {combined_file}")
    
    # Print a sample of the combined results
    print("\nSample of combined results:")
    sample_domain = list(domain_hierarchy.keys())[0]
    sample_subdomain = list(domain_hierarchy[sample_domain].keys())[0]
    print(f"\nDomain: {sample_domain}")
    print(f"  Subdomain: {sample_subdomain}")
    print("    Facts:")
    for fact in domain_hierarchy[sample_domain][sample_subdomain]:
        print(f"      - {fact}")

if __name__ == "__main__":
    main()