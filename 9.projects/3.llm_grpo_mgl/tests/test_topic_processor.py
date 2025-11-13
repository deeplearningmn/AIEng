"""
Integration tests for topic processing engine.

Tests cover complete topic processing workflow, error handling with invalid GPT responses,
and validation integration with topic processor.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from openai import RateLimitError, APIConnectionError, AuthenticationError

from mongolian_history_generator.topic_processor import TopicProcessor
from mongolian_history_generator.config import Config
from mongolian_history_generator.models.data_models import HistoricalEntry, GenerationResult


class TestTopicProcessor(unittest.TestCase):
    """Test cases for TopicProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            openai_api_key="test-api-key",
            model_name="gpt-4o-mini",
            temperature=0.25,
            max_tokens=900,
            max_retries=3
        )
        
        # Valid test content that meets validation requirements
        self.valid_content = "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
        
        self.valid_gpt_response = [
            {
                "title": "Establishment of Bogd Khanate",
                "date": "1911",
                "content": self.valid_content
            },
            {
                "title": "Mongolian Autonomy Declaration",
                "date": "1911-12-29",
                "content": self.valid_content
            }
        ]
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    @patch('mongolian_history_generator.topic_processor.Validator')
    def test_init_success(self, mock_validator, mock_gpt_client):
        """Test successful topic processor initialization."""
        processor = TopicProcessor(self.config)
        
        self.assertEqual(processor.config, self.config)
        mock_gpt_client.assert_called_once_with(self.config)
        mock_validator.assert_called_once()
        self.assertEqual(len(processor._default_topics), 20)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_topic_success(self, mock_gpt_client):
        """Test successful topic processing with valid GPT response."""
        # Mock GPT client
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = self.valid_gpt_response
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        result = processor.process_topic("1911 revolution and the establishment of Bogd Khanate")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.entries), 2)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.topic, "1911 revolution and the establishment of Bogd Khanate")
        
        # Verify entries are HistoricalEntry objects
        for entry in result.entries:
            self.assertIsInstance(entry, HistoricalEntry)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_topic_gpt_failure(self, mock_gpt_client):
        """Test topic processing when GPT client fails."""
        # Mock GPT client to raise exception
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.side_effect = Exception("API Error")
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        result = processor.process_topic("Test topic")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertFalse(result.success)
        self.assertEqual(len(result.entries), 0)
        self.assertIn("Failed to process topic", result.error_message)
        self.assertIn("API Error", result.error_message)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_topic_invalid_gpt_response(self, mock_gpt_client):
        """Test topic processing with invalid GPT response structure."""
        # Mock GPT client with invalid response
        invalid_response = [
            {
                "title": "Valid Entry",
                "date": "1911",
                "content": self.valid_content
            },
            {
                "title": "",  # Invalid: empty title
                "date": "1911",
                "content": "Too short"  # Invalid: too short content
            },
            {
                # Invalid: missing required fields
                "title": "Missing Fields"
            }
        ]
        
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = invalid_response
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        result = processor.process_topic("Test topic")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.success)  # Should succeed with at least one valid entry
        self.assertEqual(len(result.entries), 1)  # Only the valid entry should be included
        self.assertEqual(result.entries[0].title, "Valid Entry")
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_topic_no_valid_entries(self, mock_gpt_client):
        """Test topic processing when no entries pass validation."""
        # Mock GPT client with all invalid entries
        invalid_response = [
            {
                "title": "",
                "date": "invalid-date",
                "content": "Too short"
            }
        ]
        
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = invalid_response
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        result = processor.process_topic("Test topic")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertFalse(result.success)
        self.assertEqual(len(result.entries), 0)
        self.assertEqual(result.error_message, "No valid entries generated")
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_entries_success(self, mock_gpt_client):
        """Test entry validation with valid entries."""
        processor = TopicProcessor(self.config)
        
        valid_entries, errors = processor.validate_entries(self.valid_gpt_response)
        
        self.assertEqual(len(valid_entries), 2)
        self.assertEqual(len(errors), 0)
        
        for entry in valid_entries:
            self.assertIsInstance(entry, HistoricalEntry)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_entries_empty_input(self, mock_gpt_client):
        """Test entry validation with empty input."""
        processor = TopicProcessor(self.config)
        
        valid_entries, errors = processor.validate_entries([])
        
        self.assertEqual(len(valid_entries), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("No entries received", errors[0])
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_entries_invalid_structure(self, mock_gpt_client):
        """Test entry validation with invalid entry structures."""
        processor = TopicProcessor(self.config)
        
        invalid_entries = [
            "not a dictionary",
            {"title": "Missing fields"},
            {
                "title": "Invalid content",
                "date": "1911",
                "content": "Too short"
            }
        ]
        
        valid_entries, errors = processor.validate_entries(invalid_entries)
        
        self.assertEqual(len(valid_entries), 0)
        self.assertEqual(len(errors), 3)
        self.assertIn("Not a dictionary", errors[0])
        self.assertIn("Missing fields", errors[1])
        self.assertIn("Content has", errors[2])  # Content validation error
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_multiple_topics_success(self, mock_gpt_client):
        """Test processing multiple topics successfully."""
        # Mock GPT client
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = self.valid_gpt_response
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        topics = ["Topic 1", "Topic 2", "Topic 3"]
        
        # Track progress callback calls
        progress_calls = []
        def progress_callback(current, total, topic, success):
            progress_calls.append((current, total, topic, success))
        
        results = processor.process_multiple_topics(topics, progress_callback)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(len(progress_calls), 3)
        
        for i, result in enumerate(results):
            self.assertIsInstance(result, GenerationResult)
            self.assertTrue(result.success)
            self.assertEqual(result.topic, topics[i])
            self.assertEqual(len(result.entries), 2)
        
        # Verify progress callback calls
        for i, (current, total, topic, success) in enumerate(progress_calls):
            self.assertEqual(current, i + 1)
            self.assertEqual(total, 3)
            self.assertEqual(topic, topics[i])
            self.assertTrue(success)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_multiple_topics_mixed_results(self, mock_gpt_client):
        """Test processing multiple topics with mixed success/failure."""
        # Mock GPT client with alternating success/failure
        mock_gpt_instance = Mock()
        
        def side_effect(topic):
            if "fail" in topic.lower():
                raise Exception("Simulated failure")
            return self.valid_gpt_response
        
        mock_gpt_instance.generate_historical_data.side_effect = side_effect
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        topics = ["Success Topic", "Fail Topic", "Another Success"]
        
        results = processor.process_multiple_topics(topics)
        
        self.assertEqual(len(results), 3)
        
        # First topic should succeed
        self.assertTrue(results[0].success)
        self.assertEqual(len(results[0].entries), 2)
        
        # Second topic should fail
        self.assertFalse(results[1].success)
        self.assertEqual(len(results[1].entries), 0)
        self.assertIn("Simulated failure", results[1].error_message)
        
        # Third topic should succeed
        self.assertTrue(results[2].success)
        self.assertEqual(len(results[2].entries), 2)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_get_default_topics(self, mock_gpt_client):
        """Test getting default topics list."""
        processor = TopicProcessor(self.config)
        topics = processor.get_default_topics()
        
        self.assertEqual(len(topics), 20)
        self.assertIn("1911 revolution and the establishment of Bogd Khanate", topics)
        self.assertIn("Democratic Revolution of 1990 in Mongolia", topics)
        self.assertIn("COVID-19 pandemic impact on Mongolia (2020–2022)", topics)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_load_topics_from_config_default(self, mock_gpt_client):
        """Test loading default topics from configuration."""
        processor = TopicProcessor(self.config)
        topics = processor.load_topics_from_config()
        
        self.assertEqual(len(topics), 20)
        self.assertEqual(topics, processor.get_default_topics())
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_load_topics_from_config_custom(self, mock_gpt_client):
        """Test loading custom topics from configuration."""
        processor = TopicProcessor(self.config)
        custom_topics = ["Custom Topic 1", "Custom Topic 2"]
        
        topics = processor.load_topics_from_config(custom_topics)
        
        self.assertEqual(len(topics), 2)
        self.assertEqual(topics, custom_topics)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_process_all_default_topics(self, mock_gpt_client):
        """Test processing all default topics."""
        # Mock GPT client
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = self.valid_gpt_response
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        
        # Track progress
        progress_calls = []
        def progress_callback(current, total, topic, success):
            progress_calls.append((current, total, topic, success))
        
        results = processor.process_all_default_topics(progress_callback)
        
        self.assertEqual(len(results), 20)
        self.assertEqual(len(progress_calls), 20)
        
        # All should succeed with our mock
        for result in results:
            self.assertTrue(result.success)
            self.assertEqual(len(result.entries), 2)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_single_entry_valid(self, mock_gpt_client):
        """Test validation of a single valid entry."""
        processor = TopicProcessor(self.config)
        
        entry_data = {
            "title": "Test Title",
            "date": "1911",
            "content": self.valid_content
        }
        
        is_valid, errors = processor._validate_single_entry(entry_data)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_single_entry_invalid_structure(self, mock_gpt_client):
        """Test validation of entry with invalid structure."""
        processor = TopicProcessor(self.config)
        
        # Test non-dictionary
        is_valid, errors = processor._validate_single_entry("not a dict")
        self.assertFalse(is_valid)
        self.assertIn("not a dictionary", errors[0])
        
        # Test missing fields
        is_valid, errors = processor._validate_single_entry({"title": "Only title"})
        self.assertFalse(is_valid)
        self.assertIn("Missing required field", errors[0])
        
        # Test wrong field types
        is_valid, errors = processor._validate_single_entry({
            "title": 123,
            "date": "1911",
            "content": "content"
        })
        self.assertFalse(is_valid)
        self.assertIn("must be a string", errors[0])
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_validate_single_entry_invalid_content(self, mock_gpt_client):
        """Test validation of entry with invalid content."""
        processor = TopicProcessor(self.config)
        
        entry_data = {
            "title": "Test Title",
            "date": "invalid-date",
            "content": "Too short content"
        }
        
        is_valid, errors = processor._validate_single_entry(entry_data)
        
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    @patch('mongolian_history_generator.topic_processor.GPTClient')
    def test_integration_with_real_validation(self, mock_gpt_client):
        """Test integration between topic processor and real validator."""
        # This test uses the real validator (not mocked) to ensure integration works
        
        # Mock only the GPT client
        mock_gpt_instance = Mock()
        mock_gpt_instance.generate_historical_data.return_value = [
            {
                "title": "Valid Historical Entry",
                "date": "1911",
                "content": self.valid_content
            },
            {
                "title": "",  # Invalid title
                "date": "1911",
                "content": "Short"  # Invalid content length
            }
        ]
        mock_gpt_client.return_value = mock_gpt_instance
        
        processor = TopicProcessor(self.config)
        result = processor.process_topic("Test integration")
        
        # Should succeed with one valid entry
        self.assertTrue(result.success)
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].title, "Valid Historical Entry")
        
        # Verify the entry passes real validation
        self.assertTrue(result.entries[0].validate())


if __name__ == '__main__':
    unittest.main()