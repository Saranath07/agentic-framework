from typing import Any, Dict, List, Optional, Type, Union

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