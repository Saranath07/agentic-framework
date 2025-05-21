from typing import Any, Dict, List, Optional, Union

from .base_parser import BaseParser


class YamlParser(BaseParser):
    """
    Parser that converts LLM output to a YAML object.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize the YAML parser.
        
        Args:
            strict: If True, raises an error if parsing fails. If False, returns None on failure.
        """
        self.strict = strict
        # Import yaml here to avoid making it a hard dependency
        try:
            import yaml
            self.yaml = yaml
        except ImportError:
            if strict:
                raise ImportError("PyYAML is required for YamlParser. Install it with 'pip install pyyaml'.")
            self.yaml = None
    
    def parse(self, text: str) -> Union[Dict[str, Any], List[Any], None]:
        """
        Parse the raw text output from an LLM into a YAML object.
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            A dictionary or list representing the parsed YAML, or None if parsing fails and strict=False
        """
        if self.yaml is None and not self.strict:
            return None
            
        try:
            # Try to extract YAML from the text if it's embedded in markdown code blocks
            yaml_block_start = text.find('```yaml')
            if yaml_block_start != -1:
                yaml_content_start = text.find('\n', yaml_block_start) + 1
                yaml_block_end = text.find('```', yaml_content_start)
                if yaml_block_end != -1:
                    text = text[yaml_content_start:yaml_block_end].strip()
            
            # Also check for yml code blocks
            if yaml_block_start == -1:
                yaml_block_start = text.find('```yml')
                if yaml_block_start != -1:
                    yaml_content_start = text.find('\n', yaml_block_start) + 1
                    yaml_block_end = text.find('```', yaml_content_start)
                    if yaml_block_end != -1:
                        text = text[yaml_content_start:yaml_block_end].strip()
            
            return self.yaml.safe_load(text)
            
        except Exception as e:
            if self.strict:
                raise ValueError(f"Failed to parse text as YAML: {e}")
            return None