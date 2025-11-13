"""
Topic processing engine for the Mongolian History Generator.

This module coordinates GPT client and validator to process historical topics
and generate validated historical entries with comprehensive error handling
and progress tracking.
"""

import logging
from typing import List, Dict, Optional, Tuple
from .gpt_client import GPTClient
from .validator import Validator
from .models.data_models import HistoricalEntry, GenerationResult
from .config import Config, get_default_topics


class TopicProcessor:
    """
    Processes individual historical topics and validates output.
    
    Coordinates between GPT client for data generation and validator
    for ensuring data quality and format compliance.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the topic processor with configuration.
        
        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.gpt_client = GPTClient(config)
        self.validator = Validator()
        self.logger = logging.getLogger(__name__)
        self._default_topics = get_default_topics()
        
        self.logger.info(f"Topic processor initialized with {len(self._default_topics)} default topics")
    
    def process_topic(self, topic: str) -> GenerationResult:
        """
        Process a single historical topic and return validated results.
        
        Args:
            topic: The historical topic to process
            
        Returns:
            GenerationResult containing processed entries and metadata
        """
        self.logger.info(f"Processing topic: {topic}")
        
        try:
            # Generate raw data from GPT
            raw_entries = self.gpt_client.generate_historical_data(topic)
            
            # Validate and convert entries
            valid_entries, validation_errors = self.validate_entries(raw_entries)
            
            # Log validation results
            if validation_errors:
                self.logger.warning(f"Validation issues for topic '{topic}': {len(validation_errors)} errors")
                for error in validation_errors:
                    self.logger.warning(f"  - {error}")
            
            # Create result
            result = GenerationResult(
                topic=topic,
                entries=valid_entries,
                success=len(valid_entries) > 0,
                error_message=None if len(valid_entries) > 0 else "No valid entries generated",
                tokens_used=0  # TODO: Track token usage from GPT client
            )
            
            self.logger.info(f"Topic '{topic}' processed: {len(valid_entries)} valid entries")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process topic '{topic}': {str(e)}"
            self.logger.error(error_msg)
            
            return GenerationResult(
                topic=topic,
                entries=[],
                success=False,
                error_message=error_msg,
                tokens_used=0
            )
    
    def validate_entries(self, raw_entries: List[Dict]) -> Tuple[List[HistoricalEntry], List[str]]:
        """
        Validate raw entries and convert to HistoricalEntry objects.
        
        Args:
            raw_entries: List of raw entry dictionaries from GPT
            
        Returns:
            Tuple of (valid_entries, error_messages)
        """
        valid_entries = []
        errors = []
        
        if not raw_entries:
            errors.append("No entries received from GPT client")
            return valid_entries, errors
        
        for i, raw_entry in enumerate(raw_entries):
            try:
                # Validate raw entry structure
                if not isinstance(raw_entry, dict):
                    errors.append(f"Entry {i}: Not a dictionary")
                    continue
                
                # Check required fields
                required_fields = ['title', 'date', 'content']
                missing_fields = [field for field in required_fields if field not in raw_entry]
                if missing_fields:
                    errors.append(f"Entry {i}: Missing fields: {', '.join(missing_fields)}")
                    continue
                
                # Create HistoricalEntry object
                entry = HistoricalEntry.from_dict(raw_entry)
                
                # Validate entry using validator
                is_valid, entry_errors = self.validator.validate_entry(entry)
                
                if is_valid:
                    valid_entries.append(entry)
                    self.logger.debug(f"Entry {i} validated successfully: {entry.title}")
                else:
                    error_msg = f"Entry {i} ({entry.title}): {'; '.join(entry_errors)}"
                    errors.append(error_msg)
                    
            except Exception as e:
                errors.append(f"Entry {i}: Processing error: {str(e)}")
        
        return valid_entries, errors
    
    def process_multiple_topics(self, topics: List[str], 
                              progress_callback: Optional[callable] = None) -> List[GenerationResult]:
        """
        Process multiple topics with progress tracking.
        
        Args:
            topics: List of historical topics to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of GenerationResult objects
        """
        self.logger.info(f"Processing {len(topics)} topics")
        results = []
        
        for i, topic in enumerate(topics):
            try:
                # Process single topic
                result = self.process_topic(topic)
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, len(topics), topic, result.success)
                
                # Log progress
                progress_pct = ((i + 1) / len(topics)) * 100
                self.logger.info(f"Progress: {i + 1}/{len(topics)} ({progress_pct:.1f}%) - "
                               f"Topic: {topic} - Success: {result.success}")
                
            except Exception as e:
                error_msg = f"Critical error processing topic '{topic}': {str(e)}"
                self.logger.error(error_msg)
                
                # Create failed result
                failed_result = GenerationResult(
                    topic=topic,
                    entries=[],
                    success=False,
                    error_message=error_msg,
                    tokens_used=0
                )
                results.append(failed_result)
        
        # Log final summary
        successful = sum(1 for r in results if r.success)
        total_entries = sum(len(r.entries) for r in results)
        
        self.logger.info(f"Batch processing complete: {successful}/{len(topics)} topics successful, "
                        f"{total_entries} total entries generated")
        
        return results
    
    def _validate_single_entry(self, entry_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single entry dictionary.
        
        Args:
            entry_data: Dictionary containing entry data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if it's a dictionary
        if not isinstance(entry_data, dict):
            errors.append("Entry is not a dictionary")
            return False, errors
        
        # Check required fields
        required_fields = ['title', 'date', 'content']
        for field in required_fields:
            if field not in entry_data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(entry_data[field], str):
                errors.append(f"Field '{field}' must be a string")
            elif not entry_data[field].strip():
                errors.append(f"Field '{field}' cannot be empty")
        
        if errors:
            return False, errors
        
        try:
            # Create and validate HistoricalEntry
            entry = HistoricalEntry.from_dict(entry_data)
            return self.validator.validate_entry(entry)
            
        except Exception as e:
            errors.append(f"Entry validation error: {str(e)}")
            return False, errors
    
    def get_default_topics(self) -> List[str]:
        """
        Get the default list of historical topics.
        
        Returns:
            List of default historical topics for Mongolia's modern history
        """
        return self._default_topics.copy()
    
    def load_topics_from_config(self, custom_topics: Optional[List[str]] = None) -> List[str]:
        """
        Load topics from configuration or use default topics.
        
        Args:
            custom_topics: Optional list of custom topics to use instead of defaults
            
        Returns:
            List of topics to process
        """
        if custom_topics:
            self.logger.info(f"Using {len(custom_topics)} custom topics")
            return custom_topics
        else:
            self.logger.info(f"Using {len(self._default_topics)} default topics")
            return self._default_topics.copy()
    
    def process_all_default_topics(self, 
                                 progress_callback: Optional[callable] = None) -> List[GenerationResult]:
        """
        Process all default topics for Mongolia's modern history.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of GenerationResult objects for all default topics
        """
        topics = self.get_default_topics()
        self.logger.info(f"Processing all {len(topics)} default topics")
        return self.process_multiple_topics(topics, progress_callback)