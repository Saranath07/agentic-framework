from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar('T')

class BaseParser(ABC):
    """
    Base class for all parsers in the agent framework.
    
    Parsers are responsible for converting raw LLM output strings into structured data.
    """
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parse the raw text output from an LLM into a structured format.
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            Parsed data in the appropriate format
        """
        pass
    
    @classmethod
    def get_parser_type(cls) -> str:
        """
        Get the type identifier for this parser.
        
        Returns:
            String identifier for the parser type
        """
        return cls.__name__.lower().replace('parser', '')


class PydanticParser(BaseParser):
    """
    Parser that converts LLM output to a Pydantic model instance.
    """
    
    def __init__(self, model_class: Type[BaseModel]):
        """
        Initialize the parser with a Pydantic model class.
        
        Args:
            model_class: The Pydantic model class to parse the output into
        """
        self.model_class = model_class
    
    def parse(self, text: str) -> BaseModel:
        """
        Parse the raw text output from an LLM into a Pydantic model instance.
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            An instance of the specified Pydantic model
        """
        # This is a simple implementation that assumes the text is valid JSON
        # In a real implementation, you might want to add more robust parsing logic
        import json
        try:
            data = json.loads(text)
            return self.model_class.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse text as JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse text into {self.model_class.__name__}: {e}")