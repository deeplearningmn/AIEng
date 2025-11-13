"""
Data validation module for historical entries.

Provides comprehensive validation functions for historical data entries,
including field validation, date format validation, content length validation,
and JSON format validation.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .models.data_models import HistoricalEntry


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validator:
    """Comprehensive validator for historical entries and JSON data."""
    
    def __init__(self):
        """Initialize the validator with default settings."""
        self.min_words = 80
        self.max_words = 150
        self.min_paragraphs = 1
        self.max_paragraphs = 3
    
    def validate_entry(self, entry: HistoricalEntry) -> Tuple[bool, List[str]]:
        """
        Validate a complete historical entry.
        
        Args:
            entry: HistoricalEntry to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate title
        if not self.validate_title(entry.title):
            errors.append("Title is missing or empty")
        
        # Validate date
        date_valid, date_error = self.validate_date_format(entry.date)
        if not date_valid:
            errors.append(f"Date validation failed: {date_error}")
        
        # Validate content
        content_valid, content_errors = self.validate_content_length(entry.content)
        if not content_valid:
            errors.extend(content_errors)
        
        return len(errors) == 0, errors
    
    def validate_title(self, title: str) -> bool:
        """
        Validate title field.
        
        Args:
            title: Title string to validate
            
        Returns:
            bool: True if title is valid, False otherwise
        """
        return bool(title and title.strip())
    
    def validate_date_format(self, date: str) -> Tuple[bool, Optional[str]]:
        """
        Validate date format follows YYYY or YYYY-MM-DD pattern.
        
        Args:
            date: Date string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not date:
            return False, "Date is missing or empty"
        
        # Check YYYY format (4 digits)
        if re.match(r'^\d{4}$', date):
            year = int(date)
            if 1000 <= year <= 9999:
                return True, None
            else:
                return False, f"Year {year} is out of valid range"
        
        # Check YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            try:
                year, month, day = map(int, date.split('-'))
                
                # Basic range validation
                if not (1000 <= year <= 9999):
                    return False, f"Year {year} is out of valid range"
                if not (1 <= month <= 12):
                    return False, f"Month {month} is out of valid range (1-12)"
                if not (1 <= day <= 31):
                    return False, f"Day {day} is out of valid range (1-31)"
                
                return True, None
            except ValueError:
                return False, "Invalid date format - could not parse numbers"
        
        return False, "Date must be in YYYY or YYYY-MM-DD format"
    
    def validate_content_length(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validate content length (80-150 words) and paragraph structure (1-3 paragraphs).
        
        Args:
            content: Content string to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not content or not content.strip():
            errors.append("Content is missing or empty")
            return False, errors
        
        # Count words
        words = content.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            errors.append(f"Content has {word_count} words, minimum required is {self.min_words}")
        elif word_count > self.max_words:
            errors.append(f"Content has {word_count} words, maximum allowed is {self.max_words}")
        
        # Count paragraphs
        paragraph_count = self._count_paragraphs(content)
        
        if paragraph_count < self.min_paragraphs:
            errors.append(f"Content has {paragraph_count} paragraphs, minimum required is {self.min_paragraphs}")
        elif paragraph_count > self.max_paragraphs:
            errors.append(f"Content has {paragraph_count} paragraphs, maximum allowed is {self.max_paragraphs}")
        
        return len(errors) == 0, errors
    
    def _count_paragraphs(self, content: str) -> int:
        """
        Count paragraphs in content.
        
        Args:
            content: Content string to analyze
            
        Returns:
            int: Number of paragraphs
        """
        # Split by double newlines first (standard paragraph separation)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if paragraphs:
            return len(paragraphs)
        
        # If no double newlines, try single newlines
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        if paragraphs:
            return len(paragraphs)
        
        # If no newlines, treat as single paragraph if content exists
        return 1 if content.strip() else 0
    
    def validate_json_format(self, json_data: str) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON format and structure.
        
        Args:
            json_data: JSON string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}"
        
        # Check if it's a list (expected format for historical entries)
        if not isinstance(parsed_data, list):
            return False, "JSON data must be a list of entries"
        
        # Validate each entry in the list
        for i, entry_data in enumerate(parsed_data):
            if not isinstance(entry_data, dict):
                return False, f"Entry {i} is not a dictionary"
            
            # Check required fields
            required_fields = ['title', 'date', 'content']
            for field in required_fields:
                if field not in entry_data:
                    return False, f"Entry {i} is missing required field: {field}"
                if not isinstance(entry_data[field], str):
                    return False, f"Entry {i} field '{field}' must be a string"
        
        return True, None
    
    def validate_json_entries(self, json_data: str) -> Tuple[List[HistoricalEntry], List[str]]:
        """
        Validate JSON data and convert to HistoricalEntry objects.
        
        Args:
            json_data: JSON string containing historical entries
            
        Returns:
            Tuple of (valid_entries, error_messages)
        """
        # First validate JSON format
        json_valid, json_error = self.validate_json_format(json_data)
        if not json_valid:
            return [], [json_error]
        
        try:
            parsed_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            return [], [f"JSON parsing failed: {str(e)}"]
        
        valid_entries = []
        errors = []
        
        for i, entry_data in enumerate(parsed_data):
            try:
                # Create HistoricalEntry object
                entry = HistoricalEntry.from_dict(entry_data)
                
                # Validate the entry
                is_valid, entry_errors = self.validate_entry(entry)
                
                if is_valid:
                    valid_entries.append(entry)
                else:
                    errors.append(f"Entry {i} validation failed: {'; '.join(entry_errors)}")
                    
            except Exception as e:
                errors.append(f"Entry {i} processing failed: {str(e)}")
        
        return valid_entries, errors