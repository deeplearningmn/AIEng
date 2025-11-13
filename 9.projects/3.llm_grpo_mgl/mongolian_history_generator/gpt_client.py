"""
GPT Client for interacting with OpenAI API.

This module handles all interactions with OpenAI's GPT-4o-mini model
for generating historical data about Mongolia's modern history.
"""

import json
import time
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APITimeoutError, AuthenticationError
from .config import Config


class GPTClient:
    """Client for interacting with OpenAI GPT-4o-mini API."""
    
    def __init__(self, config: Config):
        """
        Initialize the GPT client with configuration.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        config.validate()
        
        self.logger.info(f"GPT Client initialized with model: {config.model_name}")
    
    def generate_historical_data(self, topic: str) -> List[Dict]:
        """
        Generate historical data entries for a given topic.
        
        Args:
            topic: The historical topic to generate data for
            
        Returns:
            List of dictionaries containing historical entries
            
        Raises:
            Exception: If API call fails after all retries
        """
        self.logger.info(f"Generating historical data for topic: {topic}")
        
        prompt = self._create_prompt(topic)
        
        try:
            response_text = self._make_api_request_with_retry(prompt)
            entries = self._parse_response(response_text)
            
            self.logger.info(f"Successfully generated {len(entries)} entries for topic: {topic}")
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to generate data for topic '{topic}': {str(e)}")
            raise
    
    def _make_api_request_with_retry(self, prompt: str) -> str:
        """
        Make API request with exponential backoff retry logic.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Response text from the API
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self._make_api_request(prompt)
                
            except AuthenticationError as e:
                # Don't retry authentication errors
                self.logger.error(f"Authentication failed: {str(e)}")
                raise
                
            except RateLimitError as e:
                last_exception = e
                self.logger.warning(f"Rate limit exceeded on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    # Longer wait for rate limiting: 5s, 10s, 20s
                    wait_time = 5 * (2 ** attempt)
                    self.logger.info(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                self.logger.warning(f"Connection/timeout error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    # Standard exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                    self.logger.info(f"Connection error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    # Standard exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        self.logger.error(f"All {self.config.max_retries} API request attempts failed")
        raise last_exception
    
    def _make_api_request(self, prompt: str) -> str:
        """
        Make a single API request to OpenAI with comprehensive error handling.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Response text from the API
            
        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            APIConnectionError: If connection fails
            APITimeoutError: If request times out
            ValueError: If response is empty or invalid
            Exception: For other API errors
        """
        tokens_used = 0
        error_type = None
        
        try:
            self.logger.debug(f"Making API request to {self.config.model_name}")
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            if not content:
                error_type = "empty_response"
                raise ValueError("Empty response from OpenAI API")
            
            # Extract token usage for monitoring
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.logger.debug(f"Tokens used: {tokens_used} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
            
            # Log successful API call if logger supports monitoring
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=True, tokens_used=tokens_used)
            
            return content.strip()
            
        except AuthenticationError as e:
            error_type = "authentication_error"
            self.logger.error(f"Authentication failed - check API key: {str(e)}")
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=False, tokens_used=tokens_used, error_type=error_type)
            raise
            
        except RateLimitError as e:
            error_type = "rate_limit_error"
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=False, tokens_used=tokens_used, error_type=error_type)
            raise
            
        except APIConnectionError as e:
            error_type = "connection_error"
            self.logger.warning(f"Connection error: {str(e)}")
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=False, tokens_used=tokens_used, error_type=error_type)
            raise
            
        except APITimeoutError as e:
            error_type = "timeout_error"
            self.logger.warning(f"Request timeout: {str(e)}")
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=False, tokens_used=tokens_used, error_type=error_type)
            raise
            
        except Exception as e:
            error_type = "unexpected_error"
            self.logger.error(f"Unexpected API error: {str(e)}")
            if hasattr(self.logger, 'log_api_call'):
                self.logger.log_api_call(success=False, tokens_used=tokens_used, error_type=error_type)
            raise
    
    def _get_system_prompt(self) -> str:
        """
        Get the comprehensive system prompt that defines the historian persona and requirements.
        
        Returns:
            System prompt string enforcing academic standards and format requirements
        """
        return """You are a distinguished academic historian with expertise in Mongolia's modern history (1911–present). 
Your role is to generate precise, factual historical entries that meet rigorous scholarly standards.

HISTORIAN PERSONA:
- Expert in Mongolian political, social, and economic history from 1911 onwards
- Committed to factual accuracy and neutral academic tone
- Focused on significant historical events, developments, and transformations
- Knowledgeable about key figures, dates, and historical context

CONTENT REQUIREMENTS:
- Generate 2-4 distinct historical entries per topic
- Each entry must contain exactly three fields: title, date, content
- Title: Concise, descriptive headline (5-12 words)
- Date: Use YYYY format for years or YYYY-MM-DD for specific dates
- Content: 1-3 well-structured paragraphs totaling 80-150 words

ACADEMIC STANDARDS:
- Maintain neutral, objective, scholarly tone throughout
- Use precise historical terminology and proper names
- Include specific dates, locations, and key figures when relevant
- Focus on factual information without interpretation or analysis
- Avoid subjective language, opinions, or value judgments

FORMATTING REQUIREMENTS:
- Output ONLY valid JSON array format - no additional text
- NO markdown formatting (**, *, #, etc.)
- NO sources, citations, URLs, or references
- NO commentary or explanatory text outside JSON
- Use \\n for paragraph breaks within content field
- Ensure proper JSON escaping for quotes and special characters

HISTORICAL SCOPE:
- Focus exclusively on Mongolia's modern period (1911–present)
- Cover political, economic, social, and cultural developments
- Include domestic events and international relations
- Emphasize transformative periods and significant milestones

EXAMPLE OUTPUT FORMAT:
[
  {
    "title": "Establishment of the Bogd Khanate",
    "date": "1911",
    "content": "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\\n\\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
  },
  {
    "title": "Autonomy Agreement with China",
    "date": "1913",
    "content": "The Sino-Mongolian Agreement of 1913 established Mongolia's autonomous status under Chinese suzerainty while maintaining the Bogd Khan's religious authority. This compromise arrangement reflected the complex geopolitical situation in Inner Asia, with Russia supporting Mongolian autonomy as a buffer against Chinese expansion.\\n\\nThe agreement granted Mongolia internal self-governance while acknowledging Chinese sovereignty, creating a framework that would influence subsequent negotiations. However, the arrangement proved unstable due to conflicting interpretations and changing regional power dynamics."
  }
]"""
    
    def _create_prompt(self, topic: str) -> str:
        """
        Create a comprehensive topic-specific prompt for historical data generation.
        
        Args:
            topic: The historical topic to generate entries for
            
        Returns:
            Detailed prompt string with specific instructions for the topic
        """
        return f"""Generate comprehensive historical entries about: {topic}

TOPIC FOCUS:
Provide detailed coverage of this specific aspect of Mongolia's modern history (1911–present). 
Include multiple perspectives and phases of development related to this topic.

CONTENT GUIDELINES:
- Cover key events, turning points, and significant developments
- Include specific dates, locations, and important historical figures
- Explain causes, processes, and consequences where relevant
- Provide context within broader Mongolian and regional history
- Ensure chronological accuracy and factual precision

ENTRY DIVERSITY:
Generate 2-4 distinct entries that cover different aspects or phases of this topic:
- Different time periods or stages of development
- Various perspectives (political, economic, social, cultural)
- Key events, policies, or transformations
- Important figures or institutions involved

ACADEMIC RIGOR:
- Use precise historical terminology and proper names
- Maintain chronological accuracy with specific dates
- Focus on documented historical facts and established scholarship
- Avoid speculation, interpretation, or contemporary political commentary
- Ensure content reflects current historical understanding

Remember: Output only the JSON array with no additional text or formatting."""
    
    def _parse_response(self, response_text: str) -> List[Dict]:
        """
        Parse the API response into a list of historical entries.
        
        Args:
            response_text: Raw response from the API
            
        Returns:
            List of parsed historical entries
            
        Raises:
            ValueError: If response cannot be parsed as JSON
        """
        try:
            # Try to parse as JSON directly
            entries = json.loads(response_text)
            
            if not isinstance(entries, list):
                raise ValueError("Response is not a JSON array")
            
            return entries
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            self.logger.debug(f"Raw response: {response_text}")
            
            # Try to extract JSON from response if it's wrapped in other text
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_text = response_text[start_idx:end_idx]
                    entries = json.loads(json_text)
                    
                    if isinstance(entries, list):
                        self.logger.info("Successfully extracted JSON from wrapped response")
                        return entries
                
            except (json.JSONDecodeError, ValueError):
                pass
            
            raise ValueError(f"Could not parse response as valid JSON: {str(e)}")