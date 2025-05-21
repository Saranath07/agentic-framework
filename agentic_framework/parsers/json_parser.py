import json
from typing import Any, Dict, List, Optional, Union

from .base_parser import BaseParser


class JsonParser(BaseParser):
    """
    Parser that converts LLM output to a JSON object.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize the JSON parser.
        
        Args:
            strict: If True, raises an error if parsing fails. If False, returns None on failure.
        """
        self.strict = strict
    
    def parse(self, text: str) -> Union[Dict[str, Any], List[Any], None]:
        """
        Parse the raw text output from an LLM into a JSON object.
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            A dictionary or list representing the parsed JSON, or None if parsing fails and strict=False
        """
        try:
            # Try to extract JSON from the text if it's embedded in other content
            json_start = text.find('{')
            json_array_start = text.find('[')
            
            # Determine which comes first, a JSON object or array
            if json_start == -1 and json_array_start == -1:
                if self.strict:
                    raise ValueError("No JSON object or array found in the text")
                return None
                
            if json_start == -1:
                start_idx = json_array_start
            elif json_array_start == -1:
                start_idx = json_start
            else:
                start_idx = min(json_start, json_array_start)
            
            # Find the matching closing bracket/brace
            if text[start_idx] == '{':
                open_char, close_char = '{', '}'
            else:
                open_char, close_char = '[', ']'
                
            count = 0
            for i in range(start_idx, len(text)):
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
                return None
            
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            if self.strict:
                raise ValueError(f"Failed to parse text as JSON: {e}")
            return None