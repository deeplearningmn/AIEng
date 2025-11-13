"""
Unit tests for GPT Client module.

Tests cover API response mocking, retry logic, and response parsing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from openai import RateLimitError, APIConnectionError, APITimeoutError, AuthenticationError

from mongolian_history_generator.gpt_client import GPTClient
from mongolian_history_generator.config import Config


class TestGPTClient(unittest.TestCase):
    """Test cases for GPTClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            openai_api_key="test-api-key",
            model_name="gpt-4o-mini",
            temperature=0.25,
            max_tokens=900,
            max_retries=3
        )
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_init_success(self, mock_openai):
        """Test successful GPT client initialization."""
        client = GPTClient(self.config)
        
        self.assertEqual(client.config, self.config)
        mock_openai.assert_called_once_with(api_key="test-api-key")
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_generate_historical_data_success(self, mock_openai):
        """Test successful historical data generation."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                "title": "Test Historical Event",
                "date": "1911",
                "content": "This is a test historical entry about Mongolia's history."
            }
        ])
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        result = client.generate_historical_data("Test topic")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test Historical Event")
        self.assertEqual(result[0]["date"], "1911")
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_retry_logic_rate_limit_error(self, mock_openai):
        """Test retry logic with rate limit errors."""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            Mock(choices=[Mock(message=Mock(content='[{"title":"Success","date":"1911","content":"Test"}]'))], usage=Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50))
        ]
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        # Mock time.sleep to avoid actual delays in tests
        with patch('time.sleep'):
            result = client.generate_historical_data("Test topic")
            
        self.assertEqual(len(result), 1)
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 3)
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_retry_logic_connection_error(self, mock_openai):
        """Test retry logic with connection errors."""
        mock_client_instance = Mock()
        
        # Create proper exception instances
        connection_error = APIConnectionError(request=Mock())
        connection_error.message = "Connection failed"
        
        mock_client_instance.chat.completions.create.side_effect = [
            connection_error,
            Mock(choices=[Mock(message=Mock(content='[{"title":"Success","date":"1911","content":"Test"}]'))], usage=Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50))
        ]
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with patch('time.sleep'):
            result = client.generate_historical_data("Test topic")
            
        self.assertEqual(len(result), 1)
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 2)
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_retry_logic_timeout_error(self, mock_openai):
        """Test retry logic with timeout errors."""
        mock_client_instance = Mock()
        
        # Create proper exception instance
        timeout_error = APITimeoutError(request=Mock())
        timeout_error.message = "Request timed out."
        
        mock_client_instance.chat.completions.create.side_effect = [
            timeout_error,
            Mock(choices=[Mock(message=Mock(content='[{"title":"Success","date":"1911","content":"Test"}]'))], usage=Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50))
        ]
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with patch('time.sleep'):
            result = client.generate_historical_data("Test topic")
            
        self.assertEqual(len(result), 1)
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 2)
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_authentication_error_no_retry(self, mock_openai):
        """Test that authentication errors are not retried."""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = AuthenticationError("Invalid API key", response=Mock(), body=None)
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with self.assertRaises(AuthenticationError):
            client.generate_historical_data("Test topic")
            
        # Should only be called once, no retries
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 1)
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_max_retries_exceeded(self, mock_openai):
        """Test behavior when max retries are exceeded."""
        mock_client_instance = Mock()
        
        # Create proper exception instance
        connection_error = APIConnectionError(request=Mock())
        connection_error.message = "Connection failed"
        
        mock_client_instance.chat.completions.create.side_effect = connection_error
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with patch('time.sleep'):
            with self.assertRaises(APIConnectionError):
                client.generate_historical_data("Test topic")
                
        # Should be called max_retries times
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, self.config.max_retries)
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_prompt_generation(self, mock_openai):
        """Test that prompts are generated correctly."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '[{"title":"Test","date":"1911","content":"Test content"}]'
        mock_response.usage = Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        client.generate_historical_data("1911 revolution")
        
        # Verify API was called with correct parameters
        call_args = mock_client_instance.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], 'gpt-4o-mini')
        self.assertEqual(call_args[1]['temperature'], 0.25)
        self.assertEqual(call_args[1]['max_tokens'], 900)
        
        # Verify messages structure
        messages = call_args[1]['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIn('1911 revolution', messages[1]['content'])
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_response_parsing_valid_json(self, mock_openai):
        """Test parsing of valid JSON responses."""
        valid_response = [
            {
                "title": "Mongolian Revolution",
                "date": "1911",
                "content": "The revolution marked Mongolia's independence from Qing rule."
            },
            {
                "title": "Bogd Khanate Establishment", 
                "date": "1911-12",
                "content": "The theocratic monarchy was established under the Eighth Jebtsundamba."
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(valid_response)
        mock_response.usage = Mock(total_tokens=200, prompt_tokens=100, completion_tokens=100)
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        result = client.generate_historical_data("Test topic")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Mongolian Revolution")
        self.assertEqual(result[1]["date"], "1911-12")
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_response_parsing_wrapped_json(self, mock_openai):
        """Test parsing of JSON wrapped in additional text."""
        wrapped_response = """Here is the historical data:
        
        [
            {
                "title": "Test Event",
                "date": "1911",
                "content": "Test content about Mongolia."
            }
        ]
        
        This completes the historical entries."""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = wrapped_response
        mock_response.usage = Mock(total_tokens=150, prompt_tokens=75, completion_tokens=75)
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        result = client.generate_historical_data("Test topic")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test Event")
        
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_response_parsing_invalid_json(self, mock_openai):
        """Test handling of invalid JSON responses."""
        invalid_response = "This is not valid JSON at all"
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = invalid_response
        mock_response.usage = Mock(total_tokens=50, prompt_tokens=25, completion_tokens=25)
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with self.assertRaises(ValueError):
            client.generate_historical_data("Test topic")
            
    @patch('mongolian_history_generator.gpt_client.OpenAI')
    def test_empty_response_handling(self, mock_openai):
        """Test handling of empty API responses."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance
        
        client = GPTClient(self.config)
        
        with self.assertRaises(ValueError) as context:
            client.generate_historical_data("Test topic")
            
        self.assertIn("Empty response", str(context.exception))
        
    def test_system_prompt_content(self):
        """Test that system prompt contains required elements."""
        client = GPTClient(self.config)
        system_prompt = client._get_system_prompt()
        
        # Check for key requirements
        self.assertIn("historian", system_prompt.lower())
        self.assertIn("mongolia", system_prompt.lower())
        self.assertIn("1911", system_prompt)
        self.assertIn("json", system_prompt.lower())
        self.assertIn("title", system_prompt)
        self.assertIn("date", system_prompt)
        self.assertIn("content", system_prompt)
        
    def test_create_prompt_content(self):
        """Test that topic-specific prompts are created correctly."""
        client = GPTClient(self.config)
        topic = "1921 Mongolian People's Revolution"
        prompt = client._create_prompt(topic)
        
        # Check that topic is included
        self.assertIn(topic, prompt)
        self.assertIn("mongolia", prompt.lower())
        self.assertIn("json", prompt.lower())


if __name__ == '__main__':
    unittest.main()