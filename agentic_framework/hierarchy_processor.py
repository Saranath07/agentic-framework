import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .base_agent import Agent, BatchProcessingAgent


class HierarchyLevel:
    """
    Represents a level in a data hierarchy with its own prompt and processing logic.
    """
    
    def __init__(
        self,
        name: str,
        prompt: str,
        llm_type: str = "openai",
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_args: Optional[Dict[str, Any]] = None,
        parser_type: Optional[str] = None,
        input_key: Optional[Union[str, List[str]]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_format: str = "list",
    ):
        """
        Initialize a hierarchy level.
        
        Args:
            name: Name of this hierarchy level (used for file naming and hierarchy keys)
            prompt: Prompt template to use for this level
            llm_type: The type of LLM to use
            model: The specific model to use
            model_args: Additional arguments to pass to the LLM
            parser_type: The type of parser to use (e.g., "json", "list", "yaml")
            input_key: The key(s) to use for input values in the prompt template
                      If None, uses the name of the parent level
                      Can be a single string or a list of strings for multiple placeholders
            input_mapping: Optional mapping from placeholder names to input keys
                          For example: {"category": "product_category", "country": "market_region"}
            output_format: Format of the expected output ("list" or "text")
        """
        self.name = name
        self.prompt = prompt
        self.llm_type = llm_type
        self.model = model
        self.model_args = model_args or {}
        self.parser_type = parser_type
        
        # Handle input keys
        if isinstance(input_key, list):
            self.input_keys = input_key
        elif input_key is not None:
            self.input_keys = [input_key]
        else:
            self.input_keys = []
            
        self.input_mapping = input_mapping or {}
        self.output_format = output_format
        
        # Will be set when added to a hierarchy
        self.parent = None
        self.children = []
        
        # Extract placeholders from the prompt
        self._extract_placeholders()
        
    def create_agent(self) -> Agent:
        """
        Create an agent for this hierarchy level.
        
        Returns:
            An Agent configured for this hierarchy level
        """
        # We don't need to modify the prompt here since the Agent.invoke method
        # will handle the formatting with the provided kwargs
        return Agent(
            llm_type=self.llm_type,
            model=self.model,
            model_args=self.model_args,
            parser_type=self.parser_type if self.parser_type else
                       "list" if self.output_format == "list" else None,
            prompt=self.prompt
        )
    
    def _extract_placeholders(self) -> None:
        """
        Extract placeholders from the prompt template.
        """
        import re
        # Find all {placeholder} patterns in the prompt
        placeholders = re.findall(r'\{([^}]+)\}', self.prompt)
        
        # Add any placeholders not already in input_keys
        for placeholder in placeholders:
            # Skip if this placeholder is already mapped
            if placeholder in self.input_mapping:
                continue
                
            # If not already in input_keys, add it
            mapped_key = self.input_mapping.get(placeholder, placeholder)
            if mapped_key not in self.input_keys:
                self.input_keys.append(mapped_key)
    
    def get_input_keys(self) -> List[str]:
        """
        Get the input keys for this level.
        
        Returns:
            The list of input keys to use in the prompt template
        """
        if self.input_keys:
            return self.input_keys
        elif self.parent:
            return [self.parent.name]
        else:
            return [self.name]
    
    def get_placeholder_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from placeholders to input keys.
        
        Returns:
            Dictionary mapping placeholder names to input keys
        """
        # Start with the explicit mapping
        mapping = dict(self.input_mapping)
        
        # Add any placeholders that map to themselves
        import re
        placeholders = re.findall(r'\{([^}]+)\}', self.prompt)
        for placeholder in placeholders:
            if placeholder not in mapping:
                mapping[placeholder] = placeholder
                
        return mapping


class HierarchyProcessor:
    """
    Processes hierarchical data using a chain of agents.
    """
    
    def __init__(
        self,
        output_dir: str = "hierarchy_results",
        combine_results: bool = True,
        parallel_processing: bool = False,
        max_workers: int = None
    ):
        """
        Initialize a hierarchy processor.
        
        Args:
            output_dir: Directory to save results
            combine_results: Whether to combine results into a single hierarchical file
            parallel_processing: Whether to process batches in parallel
            max_workers: Maximum number of worker processes to use for parallel processing
        """
        self.output_dir = output_dir
        self.combine_results = combine_results
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        self.levels = []
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_level(self, level: HierarchyLevel, parent: Optional[HierarchyLevel] = None) -> HierarchyLevel:
        """
        Add a level to the hierarchy.
        
        Args:
            level: The hierarchy level to add
            parent: The parent level (if any)
            
        Returns:
            The added level for chaining
        """
        level.parent = parent
        if parent:
            parent.children.append(level)
        
        self.levels.append(level)
        return level
    
    def process(self, initial_data: Dict[str, List[Any]], combination_method="all_combinations") -> Dict[str, Any]:
        """
        Process the hierarchy using the provided initial data.
        
        Args:
            initial_data: Dictionary mapping input keys to lists of values
            
        Returns:
            A hierarchical dictionary of results
        """
        if not self.levels:
            raise ValueError("No hierarchy levels defined")
        
        # Process each level
        level_results = {}
        current_data = initial_data
        
        for i, level in enumerate(self.levels):
            print(f"\nProcessing level {i+1}: {level.name}")
            print("-" * 50)
            
            # Create agent for this level
            agent = level.create_agent()
            batch_agent = BatchProcessingAgent(
                base_agent=agent,
                output_dir=self.output_dir
            )
            
            # Process the batch
            # Check if all required input keys are present
            input_keys = level.get_input_keys()
            primary_key = level.name
            
            # Ensure at least the primary key is present
            if primary_key not in current_data:
                raise ValueError(f"Primary input key '{primary_key}' not found in data")
            
            # Check for any missing required keys
            missing_keys = [key for key in input_keys if key not in current_data]
            if missing_keys:
                print(f"Warning: The following input keys are missing: {missing_keys}")
            
            # Determine if we should use parallel processing
            # Only use parallel processing if there are multiple items to process
            use_parallel = self.parallel_processing and len(current_data.get(primary_key, [])) > 1
            if use_parallel:
                print(f"Processing {len(current_data.get(primary_key, []))} items in parallel")
            
            filename = f"{level.name}_results"
            output_file = batch_agent.process_and_save(
                current_data,
                combination_method=combination_method,
                filename=filename,
                parallel=use_parallel,
                max_workers=self.max_workers
            )
            
            print(f"Processed {len(batch_agent.get_results())} items")
            print(f"Results saved to: {output_file}")
            
            # Store results for this level
            level_results[level.name] = {
                "file": output_file,
                "results": batch_agent.get_results(),
                "level": level
            }
            
            # Prepare data for the next level if there is one
            if i < len(self.levels) - 1 and level.children:
                next_level = level.children[0]
                next_data = self._prepare_next_level_data(level, next_level, output_file)
                current_data = next_data
        
        # Combine results if requested
        if self.combine_results:
            combined_results = self._combine_results(level_results)
            self.results = combined_results
            return combined_results
        
        self.results = level_results
        return level_results
    
    def _prepare_next_level_data(
        self,
        current_level: HierarchyLevel,
        next_level: HierarchyLevel,
        output_file: str
    ) -> Dict[str, List[Any]]:
        """
        Prepare data for the next level based on the current level's output.
        
        Args:
            current_level: The current hierarchy level
            next_level: The next hierarchy level
            output_file: Path to the current level's output file
            
        Returns:
            Data dictionary for the next level
        """
        next_data = {}
        next_input_keys = next_level.get_input_keys()
        primary_key = next_level.name  # Use the level's name as the primary key
        all_values = []
        
        # Read the output file
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                
                # Extract the output based on the format
                if current_level.output_format == "list" and isinstance(data["output"], list):
                    values = data["output"]
                elif current_level.output_format == "text" and isinstance(data["output"], str):
                    # Split text by lines for text output
                    values = data["output"].strip().split('\n')
                else:
                    # Try to handle other formats
                    if isinstance(data["output"], list):
                        values = data["output"]
                    elif isinstance(data["output"], str):
                        try:
                            values = json.loads(data["output"])
                            if not isinstance(values, list):
                                values = [values]
                        except json.JSONDecodeError:
                            values = [data["output"]]
                    else:
                        values = [str(data["output"])]
                
                # Add the values to the list
                all_values.extend(values)
        
        # Set the next level's input data for the primary key
        next_data[primary_key] = all_values
        
        # For any additional input keys, check if they're in the parent's input
        # or if they should be passed through from the current level
        for key in next_input_keys:
            if key != primary_key and key not in next_data:
                # If this is a key from the parent level, pass it through
                if current_level.parent and key in current_level.parent.get_input_keys():
                    # Find the value in the parent's input
                    parent_file = os.path.join(self.output_dir, f"{current_level.parent.name}_results.jsonl")
                    if os.path.exists(parent_file):
                        with open(parent_file, "r") as f:
                            for line in f:
                                parent_data = json.loads(line)
                                if key in parent_data["input"]:
                                    # Use the first value we find
                                    next_data[key] = [parent_data["input"][key]]
                                    break
        
        return next_data
    
    def _combine_results(self, level_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from all levels into a hierarchical structure.
        
        Args:
            level_results: Dictionary of results from each level
            
        Returns:
            A hierarchical dictionary of results
        """
        print("\nCombining results into a hierarchical structure")
        print("-" * 50)
        
        # Start with the top level
        top_level = self.levels[0]
        hierarchy = {}
        
        # Build the hierarchy recursively
        self._build_hierarchy(hierarchy, top_level, level_results)
        
        # Save the combined results
        combined_file = os.path.join(self.output_dir, "combined_hierarchy.json")
        with open(combined_file, "w") as f:
            json.dump(hierarchy, f, indent=2)
        
        print(f"Combined results saved to: {combined_file}")
        
        # Print a sample of the combined results
        if hierarchy:
            print("\nSample of combined results:")
            self._print_sample(hierarchy)
        
        return hierarchy
    
    def _build_hierarchy(
        self, 
        current_dict: Dict[str, Any], 
        current_level: HierarchyLevel,
        level_results: Dict[str, Dict[str, Any]],
        parent_key: Optional[str] = None,
        parent_value: Optional[str] = None
    ) -> None:
        """
        Recursively build the hierarchy structure.
        
        Args:
            current_dict: The current dictionary being built
            current_level: The current hierarchy level
            level_results: Dictionary of results from each level
            parent_key: The key in the parent level (if any)
            parent_value: The value in the parent level (if any)
        """
        level_info = level_results.get(current_level.name)
        if not level_info:
            return
        
        # Get the results file for this level
        results_file = level_info["file"]
        
        # Read the results
        with open(results_file, "r") as f:
            for line in f:
                data = json.loads(line)
                
                # Get the input value for this item
                primary_key = current_level.name
                input_value = data["input"].get(primary_key)
                
                # Skip if this doesn't match the parent value (for child levels)
                if parent_key and parent_value and input_value != parent_value:
                    continue
                
                # Get the output value
                if current_level.output_format == "list" and isinstance(data["output"], list):
                    output_values = data["output"]
                elif current_level.output_format == "text" and isinstance(data["output"], str):
                    output_values = data["output"].strip().split('\n')
                else:
                    # Try to handle other formats
                    if isinstance(data["output"], list):
                        output_values = data["output"]
                    elif isinstance(data["output"], str):
                        try:
                            output_values = json.loads(data["output"])
                            if not isinstance(output_values, list):
                                output_values = [output_values]
                        except json.JSONDecodeError:
                            output_values = [data["output"]]
                    else:
                        output_values = [str(data["output"])]
                
                # Add to the hierarchy
                if parent_key is None:  # Top level
                    if input_value not in current_dict:
                        current_dict[input_value] = {}
                    
                    # If this level has children, prepare for them
                    if current_level.children:
                        for output_value in output_values:
                            current_dict[input_value][output_value] = {}
                            
                            # Process each child level
                            for child in current_level.children:
                                self._build_hierarchy(
                                    current_dict[input_value], 
                                    child, 
                                    level_results,
                                    current_level.name,
                                    output_value
                                )
                    else:
                        # Leaf level, just store the values
                        current_dict[input_value] = output_values
                else:  # Child level
                    # If this level has children, prepare for them
                    if current_level.children:
                        for output_value in output_values:
                            if output_value not in current_dict:
                                current_dict[output_value] = {}
                            
                            # Process each child level
                            for child in current_level.children:
                                self._build_hierarchy(
                                    current_dict[output_value], 
                                    child, 
                                    level_results,
                                    current_level.name,
                                    output_value
                                )
                    else:
                        # Leaf level, just store the values
                        current_dict[input_value] = output_values
    
    def _print_sample(self, hierarchy: Dict[str, Any], level: int = 0, max_items: int = 1) -> None:
        """
        Print a sample of the hierarchy.
        
        Args:
            hierarchy: The hierarchy to print
            level: The current indentation level
            max_items: Maximum number of items to print at each level
        """
        if not hierarchy:
            return
        
        indent = "  " * level
        count = 0
        
        for key, value in hierarchy.items():
            if count >= max_items:
                break
                
            print(f"{indent}{key}")
            
            if isinstance(value, dict):
                self._print_sample(value, level + 1, max_items)
            elif isinstance(value, list):
                for item in value[:max_items]:
                    print(f"{indent}  - {item}")
                if len(value) > max_items:
                    print(f"{indent}  ... ({len(value) - max_items} more)")
            else:
                print(f"{indent}  {value}")
                
            count += 1
        
        if len(hierarchy) > max_items:
            print(f"{indent}... ({len(hierarchy) - max_items} more)")


def create_domain_hierarchy(
    domains: List[str],
    output_dir: str = "domain_results",
    llm_type: str = "openai",
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    model_args: Optional[Dict[str, Any]] = None,
    parallel_processing: bool = False,
    max_workers: int = None
) -> Dict[str, Any]:
    """
    Create a domain-subdomain-facts hierarchy using the provided domains.
    
    Args:
        domains: List of domains to process
        output_dir: Directory to save results
        llm_type: The type of LLM to use
        model: The specific model to use
        model_args: Additional arguments to pass to the LLM
        
    Returns:
        A hierarchical dictionary of results
    """
    # Create the hierarchy processor
    processor = HierarchyProcessor(
        output_dir=output_dir,
        parallel_processing=parallel_processing,
        max_workers=max_workers
    )
    
    # Define the domain level
    domain_level = HierarchyLevel(
        name="domain",
        prompt="""
        Give me a list of 5 subdomains of {domain}.
        Return a valid JSON array of strings containing the subdomains and nothing else.
        Format your response exactly like this: ["subdomain1", "subdomain2", "subdomain3", "subdomain4", "subdomain5"]
        Do not include any explanations, headers, or additional text.
        """,
        llm_type=llm_type,
        model=model,
        model_args=model_args,
        parser_type="list",
        output_format="list"
    )
    
    # Define the subdomain level
    subdomain_level = HierarchyLevel(
        name="subdomain",
        prompt="""
        Provide 3 key facts about {subdomain}.
        Return ONLY the facts, one per line, without numbering or any other text.
        """,
        llm_type=llm_type,
        model=model,
        model_args=model_args,
        input_key="subdomain",  # Explicitly set the input key
        output_format="text"
    )
    
    # Add levels to the hierarchy
    processor.add_level(domain_level)
    processor.add_level(subdomain_level, parent=domain_level)
    
    # Process the hierarchy
    initial_data = {"domain": domains}
    results = processor.process(initial_data)
    
    return results