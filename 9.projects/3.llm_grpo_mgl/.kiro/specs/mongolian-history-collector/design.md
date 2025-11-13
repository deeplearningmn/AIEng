# Design Document

## Overview

The Mongolian Modern History Dataset Generator is a Python-based application that leverages OpenAI's GPT-4o-mini model to generate structured, academic-quality historical datasets. The system processes predefined topics about Mongolia's modern history (1911–present) and produces detailed paragraph-level entries in JSON format.

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main Runner   │───▶│ Topic Processor │───▶│   GPT Client    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Config Manager  │    │   Validator     │    │ OpenAI API      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│    Logger       │    │ JSON Output     │
└─────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. Main Runner (`mongolian_history_generator.py`)
- **Purpose**: Orchestrates the entire data generation process
- **Responsibilities**:
  - Initialize configuration and logging
  - Load predefined topics
  - Coordinate topic processing
  - Generate final output and summary report

### 2. GPT Client (`gpt_client.py`)
- **Purpose**: Handles all interactions with OpenAI API
- **Key Methods**:
  - `generate_historical_data(topic: str) -> List[Dict]`
  - `_make_api_request(prompt: str) -> str`
  - `_retry_with_backoff(func, max_retries=3)`
- **Configuration**:
  - Model: gpt-4o-mini
  - Temperature: 0.25
  - Max tokens: 900

### 3. Topic Processor (`topic_processor.py`)
- **Purpose**: Processes individual historical topics and validates output
- **Key Methods**:
  - `process_topic(topic: str) -> List[Dict]`
  - `validate_entries(entries: List[Dict]) -> List[Dict]`
  - `_validate_single_entry(entry: Dict) -> bool`

### 4. Validator (`validator.py`)
- **Purpose**: Ensures data quality and format compliance
- **Validation Rules**:
  - Required fields: title, date, content
  - Date format: YYYY or YYYY-MM-DD
  - Content length: 80-150 words across 1-3 paragraphs
  - JSON format validation

### 5. Config Manager (`config.py`)
- **Purpose**: Manages application configuration and environment variables
- **Configuration Items**:
  - OpenAI API key
  - Output file paths
  - Logging levels
  - Default topics list

## Data Models

### Historical Entry
```python
@dataclass
class HistoricalEntry:
    title: str          # Short descriptive headline
    date: str          # YYYY or YYYY-MM-DD format
    content: str       # 1-3 paragraphs, 80-150 words total
    
    def validate(self) -> bool:
        """Validates entry format and content requirements"""
        pass
```

### Generation Result
```python
@dataclass
class GenerationResult:
    topic: str
    entries: List[HistoricalEntry]
    success: bool
    error_message: Optional[str]
    tokens_used: int
```

### Summary Report
```python
@dataclass
class SummaryReport:
    total_topics: int
    successful_generations: int
    failed_generations: int
    total_entries: int
    total_tokens_used: int
    errors: List[str]
```

## Error Handling

### API Error Handling
- **Connection Errors**: Retry with exponential backoff (max 3 attempts)
- **Rate Limiting**: Implement proper delays and respect API limits
- **Invalid Responses**: Log error and continue with next topic
- **Authentication Errors**: Fail fast with clear error message

### Data Validation Errors
- **Missing Fields**: Log warning and exclude invalid entries
- **Invalid Date Format**: Attempt to parse and correct, or exclude entry
- **Content Length Issues**: Log warning but include entry if close to requirements
- **JSON Parse Errors**: Log error and retry API call once

### System Errors
- **File I/O Errors**: Ensure proper permissions and disk space
- **Configuration Errors**: Validate all required settings at startup
- **Memory Issues**: Process topics in batches if needed

## Testing Strategy

### Unit Tests
- **GPT Client**: Mock OpenAI API responses for consistent testing
- **Validator**: Test all validation rules with edge cases
- **Topic Processor**: Test topic processing logic and error handling
- **Config Manager**: Test configuration loading and validation

### Integration Tests
- **End-to-End Flow**: Test complete generation process with sample topics
- **API Integration**: Test actual OpenAI API calls with test account
- **File Output**: Verify JSON output format and file creation

### Test Data
- **Sample Topics**: Use subset of actual historical topics
- **Mock Responses**: Create realistic GPT responses for testing
- **Edge Cases**: Test with malformed data and error conditions

## Performance Considerations

### API Efficiency
- **Batch Processing**: Process topics sequentially to avoid rate limits
- **Token Optimization**: Monitor token usage and optimize prompts
- **Caching**: Consider caching successful responses for development

### Memory Management
- **Streaming Output**: Write results incrementally to avoid memory issues
- **Garbage Collection**: Properly clean up large objects after processing
- **Resource Limits**: Set reasonable limits on concurrent operations

### Scalability
- **Topic Batching**: Support processing large numbers of topics
- **Parallel Processing**: Consider async processing for independent topics
- **Progress Tracking**: Provide progress indicators for long-running operations

## Security Considerations

### API Key Management
- **Environment Variables**: Store API key securely in environment
- **Key Rotation**: Support easy API key updates
- **Access Control**: Ensure API key is not logged or exposed

### Data Privacy
- **No PII**: Ensure generated content contains no personal information
- **Public Sources**: Verify all generated content references public information
- **Content Filtering**: Implement basic content appropriateness checks

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
LOG_LEVEL=INFO
OUTPUT_DIR=./data/generated
MAX_RETRIES=3
```

### Default Topics List
The system includes 20 predefined topics covering key periods of Mongolia's modern history:
1. 1911 revolution and the establishment of Bogd Khanate
2. 1915 Khiagta Treaty and Mongolian autonomy
3. 1921 Mongolian People's Revolution
4. Formation of the Mongolian People's Republic in 1924
5. Democratic Revolution of 1990 in Mongolia
6. Adoption of the 1992 Constitution and new governance structure
7. Economic transition and privatization in the 1990s
8. Mongolia's foreign policy between Russia, China, and the West after 1990
9. Mining boom: Oyu Tolgoi, Tavan Tolgoi and resource-based growth
10. COVID-19 pandemic impact on Mongolia (2020–2022)
[Additional topics to be defined...]

## Output Format

### JSON Structure
```json
[
  {
    "title": "Establishment of the Bogd Khanate",
    "date": "1911",
    "content": "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
  }
]
```

### File Organization
```
data/generated/
├── mongolian_history_dataset.json    # Complete dataset
├── generation_report.json            # Summary report
└── logs/
    └── generation_YYYYMMDD_HHMMSS.log # Detailed logs
```