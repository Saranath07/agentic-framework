import json
from typing import Any, List, Optional, Union

from .base_parser import BaseParser


class ListParser(BaseParser):
    """
    Parser that converts LLM output to a Python list.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize the List parser.
        
        Args:
            strict: If True, raises an error if parsing fails. If False, returns empty list on failure.
        """
        self.strict = strict
    
    def parse(self, text: str) -> List[Any]:
        """
        Parse the raw text output from an LLM into a Python list.
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            A list representing the parsed output, or empty list if parsing fails and strict=False
        """
        try:
            # Try to extract list from the text if it's embedded in other content
            list_start = text.find('[')
            
            if list_start == -1:
                if self.strict:
                    raise ValueError("No list found in the text")
                return []
            
            # Find the matching closing bracket
            open_char, close_char = '[', ']'
            count = 0
            for i in range(list_start, len(text)):
                if text[i] == open_char:
                    count += 1
                elif text[i] == close_char:
                    count -= 1
                    if count == 0:
                        end_idx = i + 1
                        break
            else:
                if self.strict:
                    raise ValueError(f"No matching closing {close_char} found")
                return []
            
            list_str = text[list_start:end_idx]
            
            # Parse the list using json.loads for safety
            return json.loads(list_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            if self.strict:
                raise ValueError(f"Failed to parse text as list: {e}")
            return []