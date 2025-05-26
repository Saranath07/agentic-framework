# Batch Processing Agent

The BatchProcessingAgent is a powerful tool for processing multiple items using a base agent. It supports multiple placeholders in prompt templates and can save results to JSONL files for further processing.

## Features

- Process batches of items with multiple placeholders
- Support for different combination methods:
  - `one_to_one`: Match items at the same index (requires all lists to be the same length)
  - `all_combinations`: Generate all possible combinations of placeholder values
- Save results to JSONL files
- Process and save in one operation
- Hierarchical processing (process results of one batch in another batch)

## Installation

The BatchProcessingAgent is included in the agentic_framework package. No additional installation is required.

## Usage

### Basic Usage

```python
from agentic_framework import Agent, BatchProcessingAgent

# Create a base agent with a prompt template
base_agent = Agent(
    llm_type="openai",
    model="your-model",
    model_args={
        "temperature": 0.7,
        "api_key": "your-api-key"
    },
    prompt="Give me a list of 5 subdomains of {domain}."  # Prompt template with placeholders
)

# Create a batch processing agent
batch_agent = BatchProcessingAgent(
    base_agent=base_agent,
    output_dir="results"  # Directory to save results
)

# Define placeholder values
placeholder_dict = {
    "domain": ["cricket", "football", "dance"]
}

# Process the batch
results = batch_agent.process_batch(placeholder_dict)

# Save results to a JSONL file
output_file = batch_agent.save_results("domain_subdomains")
```

### Multiple Placeholders

```python
# Create an agent with a prompt template that has multiple placeholders
father_agent = Agent(
    llm_type="openai",
    model="your-model",
    model_args={
        "temperature": 0.7,
        "api_key": "your-api-key"
    },
    prompt="What is the father's name of {person} and his {surname}?"
)

# Create a batch processing agent
father_batch_agent = BatchProcessingAgent(
    base_agent=father_agent,
    output_dir="results"
)

# Dictionary with multiple placeholders
person_data = {
    "person": ["Virat Kohli", "Naga Chaitanya", "Mukesh Ambani"],
    "surname": ["Kohli", "Akkineni", "Ambani"]
}

# Process the batch with one-to-one mapping
results = father_batch_agent.process_batch(
    person_data,
    combination_method="one_to_one"  # Match items at the same index
)
```

### All Combinations

```python
# Create an agent with a prompt template
features_agent = Agent(
    llm_type="openai",
    model="your-model",
    model_args={
        "temperature": 0.7,
        "api_key": "your-api-key"
    },
    prompt="List 3 key features of a {brand} {product}."
)

# Create a batch processing agent
features_batch_agent = BatchProcessingAgent(
    base_agent=features_agent,
    output_dir="results"
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
```

### Process and Save in One Operation

```python
output_file = batch_agent.process_and_save(
    placeholder_dict,
    filename="results_filename"  # Optional, will use timestamp if not provided
)
```

## Example Scripts

### batch_processing_example.py

This script demonstrates various ways to use the BatchProcessingAgent:

1. Processing domains to get subdomains
2. Processing with multiple placeholders (one-to-one)
3. Processing with all combinations
4. Processing and saving in one operation
5. Using the results for further processing

Run the script:

```bash
python batch_processing_example.py
```

### domain_subdomain_processor.py

This script demonstrates a specific use case for domain-subdomain processing:

1. Get subdomains for each domain
2. Extract subdomains from the results
3. Process each subdomain to get more information
4. Combine all results into a single structured file

Run the script:

```bash
python domain_subdomain_processor.py
```

## JSONL File Format

The BatchProcessingAgent saves results in JSONL (JSON Lines) format, where each line is a valid JSON object. Each object contains:

- `key`: A unique identifier for the combination
- `input`: The input values used for the placeholders
- `output`: The output from the base agent

Example:

```json
{"key": "domain:cricket", "input": {"domain": "cricket"}, "output": "Test cricket\nT20 cricket\nODI cricket\nDomestic cricket\nWomen's cricket"}
{"key": "domain:football", "input": {"domain": "football"}, "output": "American football\nAssociation football (soccer)\nAustralian rules football\nRugby football\nCanadian football"}
```

## Further Processing

You can load the JSONL files for further processing:

```python
import json

# Load results from a JSONL file
results = []
with open("results/domain_subdomains.jsonl", "r") as f:
    for line in f:
        results.append(json.loads(line))

# Process the results
for item in results:
    domain = item["input"]["domain"]
    subdomains = item["output"].strip().split('\n')
    print(f"{domain}: {len(subdomains)} subdomains found")