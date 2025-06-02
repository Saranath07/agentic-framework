# Enhanced Hierarchy Parser Guide

This guide demonstrates how to use different parsers (including custom ones) within hierarchical data processing using the agentic framework.

## Overview

The enhanced hierarchy system supports multiple parser types at different levels:
- **List Parser**: For extracting arrays/lists from LLM responses
- **JSON Parser**: For structured JSON data extraction
- **YAML Parser**: For YAML configuration parsing
- **Pydantic Parser**: For type-safe structured data with validation
- **Custom Parsers**: User-defined parsers for specific formats

## Files Created

### 1. `combined_hierarchy_parser_example.py`
A comprehensive example that demonstrates:
- **Custom Parser Implementation**: `ProductSpecParser` for parsing product specifications
- **4-Level Hierarchy**: Using different parsers at each level
- **Pydantic Integration**: Type-safe data extraction with validation
- **Parallel Processing**: Efficient processing of multiple items

### 2. `parser_showcase_example.py` (Enhanced)
Individual parser demonstrations and hierarchy integration:
- JSON Parser for product information
- YAML Parser for service configurations  
- List Parser for recommendations
- Pydantic Parser for company information
- Hierarchy with mixed parsers

## Key Features Implemented

### 1. Custom Parser Support
```python
class ProductSpecParser(BaseParser):
    def parse(self, text: str) -> Dict[str, Any]:
        # Custom parsing logic for specific formats
        # Handles structured text like:
        # Name: Product Name
        # Price: $999.99
        # Rating: 4.5/5
        # Features: feature1, feature2, feature3
```

### 2. Enhanced HierarchyLevel Class
- Added `parser` parameter to accept custom parser instances
- Maintains backward compatibility with `parser_type`
- Automatic parser selection logic

### 3. Multi-Level Hierarchies
```python
# Level 1: List Parser
category_level = HierarchyLevel(
    name="category",
    parser_type="list",
    # ... other config
)

# Level 2: JSON Parser  
product_level = HierarchyLevel(
    name="product", 
    parser_type="json",
    # ... other config
)

# Level 3: Custom Parser
spec_level = HierarchyLevel(
    name="specification",
    parser=ProductSpecParser(),
    # ... other config
)

# Level 4: YAML Parser
config_level = HierarchyLevel(
    name="config",
    parser_type="yaml", 
    # ... other config
)
```

### 4. Pydantic Integration
```python
class TechSpecification(BaseModel):
    component: str
    specification: str
    importance_level: str
    technical_details: str

# Use in hierarchy
spec_level = HierarchyLevel(
    name="tech_spec",
    parser=PydanticParser(TechSpecification),
    # ... other config
)
```

## Example Hierarchies

### Enhanced Product Hierarchy
1. **Domain** → **Categories** (List Parser)
2. **Category** → **Products** (JSON Parser) 
3. **Product** → **Specifications** (Custom Parser)
4. **Specification** → **Config** (YAML Parser)

### Tech Component Hierarchy  
1. **Device Type** → **Components** (List Parser)
2. **Component** → **Tech Specs** (Pydantic Parser)

## Running the Examples

```bash
# Run the comprehensive combined example
python combined_hierarchy_parser_example.py

# Run individual parser showcase
python parser_showcase_example.py

# Run the original hierarchy example
python hierarchy_example.py
```

## Output Structure

The system generates:
- **Individual Level Results**: JSONL files for each hierarchy level
- **Combined Hierarchy**: Nested JSON structure showing relationships
- **Parallel Processing**: Efficient handling of multiple items
- **Type Safety**: Validated data structures with Pydantic

## Key Improvements Made

### 1. Async/Await Fix
- Fixed coroutine handling in `BatchProcessingAgent.process_and_save`
- Made `HierarchyProcessor.process` async
- Updated example files to use `asyncio.run()`

### 2. Input Key Resolution
- Fixed primary key detection using `input_keys` instead of level name
- Improved data flow between hierarchy levels
- Better error handling for missing keys

### 3. Custom Parser Support
- Extended `HierarchyLevel` to accept custom parser instances
- Maintained backward compatibility with `parser_type`
- Enhanced `create_agent` method for parser selection

### 4. Enhanced Examples
- Created comprehensive demonstration files
- Added custom parser implementations
- Integrated Pydantic models for type safety
- Demonstrated real-world use cases

## Benefits

1. **Flexibility**: Mix different parser types within a single hierarchy
2. **Type Safety**: Pydantic integration for validated data structures
3. **Performance**: Parallel processing at each level
4. **Extensibility**: Easy to add custom parsers for specific formats
5. **Maintainability**: Clear separation of concerns between levels
6. **Real-world Ready**: Handles complex data transformation scenarios

This enhanced system provides a powerful foundation for building complex data processing pipelines with multiple parsing strategies and hierarchical relationships.