from typing import Any, Dict, List, Optional, Type, Union, Tuple, Iterator
import json
import os
import itertools
import concurrent.futures # asyncio along with concurrent features for parallel processing
from pathlib import Path

from pydantic import BaseModel, Field

from baseLLM import LLM, LLMResponse
from .parsers.base_parser import BaseParser


class Agent:
    """
    Base Agent class that uses an LLM to generate responses and a parser to structure the output.
    """
    
    def __init__(
        self,
        llm_type: str,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_type: Optional[str] = None,
        parser: Optional[Union[BaseParser, Type[BaseParser]]] = None,
        prompt: str = "",
        system_prompt: str = "",
    ):
        """
        Initialize an agent with an LLM and parser.
        
        Args:
            llm_type: The type of LLM to use (e.g., "openai", "groq", "github")
            model: The specific model to use (e.g., "gpt-4", "llama-3.1")
            model_args: Additional arguments to pass to the LLM (e.g., temperature, max_tokens)
            parser_type: The type of parser to use (e.g., "json", "yaml", "pydantic")
            parser: A parser instance or class to use for parsing LLM output
            prompt: Default prompt template to use for queries
            system_prompt: System prompt to use for the LLM
        """
        # Initialize the LLM
        model_args = model_args or {}
        self.llm = LLM(
            service_provider=llm_type,
            llm_model_name=model,
            **model_args
        )
        
        # Initialize the parser
        self.parser = None
        if parser is not None:
            if isinstance(parser, BaseParser):
                self.parser = parser
            elif issubclass(parser, BaseParser):
                self.parser = parser()
        elif parser_type is not None:
            # Import parsers dynamically based on parser_type
            if parser_type.lower() == "json":
                from .parsers.json_parser import JsonParser
                self.parser = JsonParser()
            elif parser_type.lower() == "yaml":
                from .parsers.yaml_parser import YamlParser
                self.parser = YamlParser()
            elif parser_type.lower() == "list":
                from .parsers.list_parser import ListParser
                self.parser = ListParser()
            elif parser_type.lower() == "pydantic":
                from .parsers.base_parser import PydanticParser
                # Note: PydanticParser requires a model class, which should be provided separately
                raise ValueError("PydanticParser requires a model class to be provided directly")
        
        # Store prompt templates
        self.prompt_template = prompt
        self.system_prompt = system_prompt
        
        # Initialize conversation history
        self.conversation_history = []
    
    def format_prompt(self, query: str, **kwargs) -> str:
        """
        Format the prompt template with the query and any additional arguments.
        
        Args:
            query: The query to include in the prompt
            **kwargs: Additional arguments to format the prompt template with
            
        Returns:
            The formatted prompt
        """
        if not self.prompt_template:
            return query
            
        # Format the prompt template with the query and any additional arguments
        prompt_vars = {"query": query, **kwargs}
        return self.prompt_template.format(**prompt_vars)
    
    def invoke(self, query: str, **kwargs) -> Any:
        """
        Invoke the agent with a query and return the parsed response.
        
        Args:
            query: The query to send to the LLM
            **kwargs: Additional arguments to format the prompt with
            
        Returns:
            The parsed response from the LLM
        """
        # Format the prompt
        formatted_prompt = self.format_prompt(query, **kwargs)
        
        # Add system prompt if provided
        if self.system_prompt:
            # This is a simplified approach; in a real implementation,
            # you would need to handle system prompts according to the LLM's API
            formatted_prompt = f"{self.system_prompt}\n\n{formatted_prompt}"
        
        # Invoke the LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Store the conversation
        self.conversation_history.append({
            "query": query,
            "formatted_prompt": formatted_prompt,
            "response": response.content,
            "metadata": response.metadata
        })
        
        # Parse the response if a parser is provided
        if self.parser is not None:
            try:
                return self.parser.parse(response.content)
            except Exception as e:
                # If parsing fails, return the raw response
                return response.content
        
        # Return the raw response if no parser is provided
        return response.content
    
    def get_raw_response(self, query: str, **kwargs) -> LLMResponse:
        """
        Get the raw LLM response without parsing.
        
        Args:
            query: The query to send to the LLM
            **kwargs: Additional arguments to format the prompt with
            
        Returns:
            The raw LLMResponse object
        """
        formatted_prompt = self.format_prompt(query, **kwargs)
        
        if self.system_prompt:
            formatted_prompt = f"{self.system_prompt}\n\n{formatted_prompt}"
        
        return self.llm.invoke(formatted_prompt)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            A list of conversation turns, each containing the query, formatted prompt,
            response, and metadata
        """
        return self.conversation_history


# Helper function for parallel processing - must be at module level for pickling
def _process_combo(agent, combo_tuple):
    """
    Process a single combination using the agent.
    
    Args:
        agent: The agent to use for processing
        combo_tuple: A tuple of (combo_values, combo_key)
        
    Returns:
        A tuple of (combo_key, result_data)
    """
    combo_values, combo_key = combo_tuple
    # Create a new agent instance with the same parameters
    # This is necessary because the original agent might not be picklable
    model_args = {
        "temperature": agent.llm.temperature,
        "max_tokens": agent.llm.max_tokens,
        "api_key": agent.llm.api_key,
        "base_url": agent.llm.base_url
    }
    
    new_agent = Agent(
        llm_type=agent.llm.service_provider,
        model=agent.llm.llm_model_name,
        model_args=model_args,
        prompt=agent.prompt_template,
        system_prompt=agent.system_prompt
    )
    
    # If there's a parser, set it up
    if hasattr(agent, 'parser') and agent.parser is not None:
        new_agent.parser = agent.parser
    
    # Process the prompt
    result = new_agent.invoke("", **combo_values)
    
    return combo_key, {
        "input": combo_values,
        "output": result
    }
    

class BatchProcessingAgent:
    """
    Agent that processes a batch of items by substituting multiple placeholders in a prompt template.
    Results can be saved to a jsonl file for further processing.
    """
    
    def __init__(
        self,
        base_agent: Agent,
        output_dir: str = "batch_results"
    ):
        """
        Initialize a batch processing agent.
        
        Args:
            base_agent: The base agent to use for processing each item
            output_dir: Directory to save results (default: "batch_results")
        """
        self.base_agent = base_agent
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_batch(
        self,
        placeholder_dict: Dict[str, List[str]],
        combination_method: str = "one_to_one",
        parallel: bool = False,
        max_workers: int = None
    ) -> Dict[str, Any]:
        """
        Process a batch of items using the base agent with multiple placeholders.
        
        Args:
            placeholder_dict: Dictionary mapping placeholder names to lists of values
            combination_method: Method to combine placeholders:
                - "one_to_one": Match items at same index (requires all lists to be same length)
                - "all_combinations": Generate all possible combinations of placeholder values
            parallel: Whether to process items in parallel using multiple processes
            max_workers: Maximum number of worker processes to use (None = auto-determine)
            
        Returns:
            Dictionary mapping combination keys to their results
        """
        results = {}
        combinations = list(self._generate_combinations(placeholder_dict, combination_method))
        
        if parallel and len(combinations) > 1:
            results = self._process_batch_parallel(combinations, max_workers)
        else:
            # Sequential processing
            for combo_values, combo_key in combinations:
                # Process the prompt using the base agent with the placeholder values as kwargs
                result = self.base_agent.invoke("", **combo_values)
                
                # Store the result
                results[combo_key] = {
                    "input": combo_values,
                    "output": result
                }
        
        self.results = results
        return results
        
    def _process_batch_parallel(
        self,
        combinations: List[Tuple[Dict[str, str], str]],
        max_workers: int = None
    ) -> Dict[str, Any]:
        """
        Process a batch of items in parallel using multiple processes.
        
        Args:
            combinations: List of (combination_dict, combination_key) tuples
            max_workers: Maximum number of worker processes to use
            
        Returns:
            Dictionary mapping combination keys to their results
        """
        results = {}
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        # This avoids pickling issues while still providing concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all combinations to the executor
            future_to_combo = {
                executor.submit(lambda c: _process_combo(self.base_agent, c), combo): combo
                for combo in combinations
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_combo):
                try:
                    combo_key, result_data = future.result()
                    results[combo_key] = result_data
                except Exception as exc:
                    combo = future_to_combo[future]
                    print(f"Combination {combo[1]} generated an exception: {exc}")
                    # Store the error in results
                    results[combo[1]] = {
                        "input": combo[0],
                        "output": f"ERROR: {str(exc)}",
                        "error": True
                    }
        
        return results
    
    def _generate_combinations(
        self,
        placeholder_dict: Dict[str, List[str]],
        method: str
    ) -> Iterator[Tuple[Dict[str, str], str]]:
        """
        Generate combinations of placeholder values based on the specified method.
        
        Args:
            placeholder_dict: Dictionary mapping placeholder names to lists of values
            method: Combination method ("one_to_one" or "all_combinations")
            
        Returns:
            Iterator of (combination_dict, combination_key) tuples
        """
        if method == "one_to_one":
            # Verify all lists have the same length
            list_lengths = [len(values) for values in placeholder_dict.values()]
            if len(set(list_lengths)) > 1:
                raise ValueError("For one_to_one method, all placeholder lists must have the same length")
            
            # Generate one-to-one combinations
            placeholders = list(placeholder_dict.keys())
            for i in range(list_lengths[0]):
                combo = {p: placeholder_dict[p][i] for p in placeholders}
                # Create a key for this combination
                combo_key = "_".join(f"{p}:{combo[p]}" for p in placeholders)
                yield combo, combo_key
                
        elif method == "all_combinations":
            # Generate all possible combinations
            placeholders = list(placeholder_dict.keys())
            placeholder_values = [placeholder_dict[p] for p in placeholders]
            
            for values in itertools.product(*placeholder_values):
                combo = {placeholders[i]: values[i] for i in range(len(placeholders))}
                # Create a key for this combination
                combo_key = "_".join(f"{p}:{combo[p]}" for p in placeholders)
                yield combo, combo_key
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def save_results(self, filename: str = None) -> str:
        """
        Save the results to a JSONL file.
        
        Args:
            filename: Name of the file to save results to (without extension)
                     If None, a timestamp-based filename will be used
        
        Returns:
            Path to the saved file
        """
        if not self.results:
            raise ValueError("No results to save. Run process_batch first.")
        
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{timestamp}"
        
        # Ensure the filename has .jsonl extension
        if not filename.endswith('.jsonl'):
            filename += '.jsonl'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            for combo_key, result_data in self.results.items():
                # Convert the result to a serializable format
                serializable_result = {
                    "key": combo_key,
                    "input": result_data["input"],
                    "output": result_data["output"] if isinstance(result_data["output"], (str, int, float, bool, list, dict))
                              else str(result_data["output"])
                }
                f.write(json.dumps(serializable_result) + '\n')
        
        return filepath
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the batch processing.
        
        Returns:
            Dictionary mapping combination keys to their results
        """
        return self.results
    
    def process_and_save(
        self,
        placeholder_dict: Dict[str, List[str]],
        combination_method: str = "one_to_one",
        filename: str = None,
        parallel: bool = False,
        max_workers: int = None
    ) -> str:
        """
        Process a batch and save the results in one operation.
        
        Args:
            placeholder_dict: Dictionary mapping placeholder names to lists of values
            combination_method: Method to combine placeholders
            filename: Name of the file to save results to (without extension)
            parallel: Whether to process items in parallel using multiple processes
            max_workers: Maximum number of worker processes to use
        
        Returns:
            Path to the saved file
        """
        self.process_batch(placeholder_dict, combination_method, parallel, max_workers)
        return self.save_results(filename)