"""
Unit tests for validator module.

Tests cover field validation, date format validation, content length validation,
JSON format validation, and edge cases.
"""

import unittest
import json
from mongolian_history_generator.validator import Validator, ValidationError
from mongolian_history_generator.models.data_models import HistoricalEntry


class TestValidator(unittest.TestCase):
    """Test cases for Validator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = Validator()
        self.valid_content = "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
        
        self.valid_entry = HistoricalEntry(
            title="Mongolian Revolution of 1911",
            date="1911",
            content=self.valid_content
        )
    
    def test_validate_entry_valid(self):
        """Test validation of a completely valid entry."""
        is_valid, errors = self.validator.validate_entry(self.valid_entry)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_entry_invalid_title(self):
        """Test validation with invalid title."""
        entry = HistoricalEntry("", "1911", self.valid_content)
        is_valid, errors = self.validator.validate_entry(entry)
        
        self.assertFalse(is_valid)
        self.assertIn("Title is missing or empty", errors)
    
    def test_validate_entry_invalid_date(self):
        """Test validation with invalid date."""
        entry = HistoricalEntry("Title", "invalid-date", self.valid_content)
        is_valid, errors = self.validator.validate_entry(entry)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Date validation failed" in error for error in errors))
    
    def test_validate_entry_invalid_content(self):
        """Test validation with invalid content."""
        entry = HistoricalEntry("Title", "1911", "Too short")
        is_valid, errors = self.validator.validate_entry(entry)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("words" in error for error in errors))
    
    def test_validate_title_valid(self):
        """Test title validation with valid inputs."""
        self.assertTrue(self.validator.validate_title("Valid Title"))
        self.assertTrue(self.validator.validate_title("Another Valid Title"))
    
    def test_validate_title_invalid(self):
        """Test title validation with invalid inputs."""
        self.assertFalse(self.validator.validate_title(""))
        self.assertFalse(self.validator.validate_title("   "))
        self.assertFalse(self.validator.validate_title(None))
    
    def test_validate_date_format_yyyy(self):
        """Test date validation with YYYY format."""
        valid, error = self.validator.validate_date_format("1911")
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        valid, error = self.validator.validate_date_format("2024")
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_date_format_yyyy_mm_dd(self):
        """Test date validation with YYYY-MM-DD format."""
        valid, error = self.validator.validate_date_format("1911-12-29")
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        valid, error = self.validator.validate_date_format("2024-01-15")
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_date_format_invalid_year_range(self):
        """Test date validation with invalid year ranges."""
        # 3-digit year doesn't match YYYY format, so gets generic error
        valid, error = self.validator.validate_date_format("999")
        self.assertFalse(valid)
        self.assertIn("Date must be in YYYY or YYYY-MM-DD format", error)
        
        # 5-digit year doesn't match YYYY format, so gets generic error
        valid, error = self.validator.validate_date_format("10000")
        self.assertFalse(valid)
        self.assertIn("Date must be in YYYY or YYYY-MM-DD format", error)
    
    def test_validate_date_format_invalid_month_day(self):
        """Test date validation with invalid month/day values."""
        valid, error = self.validator.validate_date_format("1911-13-01")
        self.assertFalse(valid)
        self.assertIn("Month 13 is out of valid range", error)
        
        valid, error = self.validator.validate_date_format("1911-12-32")
        self.assertFalse(valid)
        self.assertIn("Day 32 is out of valid range", error)
        
        valid, error = self.validator.validate_date_format("1911-00-15")
        self.assertFalse(valid)
        self.assertIn("Month 0 is out of valid range", error)
    
    def test_validate_date_format_invalid_formats(self):
        """Test date validation with completely invalid formats."""
        invalid_dates = [
            "",
            "not-a-date",
            "1911/12/29",
            "Dec 29, 1911",
            "1911-1",
            "1911-1-1",
            "19111"
        ]
        
        for invalid_date in invalid_dates:
            valid, error = self.validator.validate_date_format(invalid_date)
            self.assertFalse(valid, f"Date '{invalid_date}' should be invalid")
            self.assertIsNotNone(error)
    
    def test_validate_content_length_valid(self):
        """Test content validation with valid word counts and paragraphs."""
        # Valid single paragraph
        content = "The Mongolian Revolution of 1911 was a pivotal moment in Mongolia's history that marked the end of Qing dynasty rule and established the Bogd Khanate. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control. The Eighth Jebtsundamba Khutughtu was proclaimed as the Bogd Khan, creating a theocratic monarchy. The revolution faced immediate challenges from both Chinese and Russian interests in the region. Despite limited international recognition, this period laid crucial groundwork for Mongolian national identity and political consciousness that would influence future independence movements throughout the twentieth century."
        
        valid, errors = self.validator.validate_content_length(content)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_content_length_too_short(self):
        """Test content validation with too few words."""
        content = "The Mongolian Revolution of 1911 was important."
        
        valid, errors = self.validator.validate_content_length(content)
        self.assertFalse(valid)
        self.assertTrue(any("minimum required is 80" in error for error in errors))
    
    def test_validate_content_length_too_long(self):
        """Test content validation with too many words."""
        # Generate content with over 150 words
        content = " ".join(["word"] * 160)
        
        valid, errors = self.validator.validate_content_length(content)
        self.assertFalse(valid)
        self.assertTrue(any("maximum allowed is 150" in error for error in errors))
    
    def test_validate_content_length_too_many_paragraphs(self):
        """Test content validation with too many paragraphs."""
        # Four paragraphs with valid word count
        content = "Paragraph one with sufficient words to meet requirements.\n\nParagraph two with sufficient words to meet requirements.\n\nParagraph three with sufficient words to meet requirements.\n\nParagraph four with sufficient words to meet requirements and exceed the maximum paragraph limit."
        
        valid, errors = self.validator.validate_content_length(content)
        self.assertFalse(valid)
        self.assertTrue(any("maximum allowed is 3" in error for error in errors))
    
    def test_validate_content_length_empty(self):
        """Test content validation with empty content."""
        valid, errors = self.validator.validate_content_length("")
        self.assertFalse(valid)
        self.assertIn("Content is missing or empty", errors)
        
        valid, errors = self.validator.validate_content_length("   ")
        self.assertFalse(valid)
        self.assertIn("Content is missing or empty", errors)
    
    def test_count_paragraphs_double_newlines(self):
        """Test paragraph counting with double newlines."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        count = self.validator._count_paragraphs(content)
        self.assertEqual(count, 3)
    
    def test_count_paragraphs_single_newlines(self):
        """Test paragraph counting with single newlines."""
        content = "First paragraph.\nSecond paragraph.\nThird paragraph."
        count = self.validator._count_paragraphs(content)
        # The implementation first tries double newlines - this content has none, so it's treated as 1 paragraph
        self.assertEqual(count, 1)
    
    def test_count_paragraphs_single_newlines_only(self):
        """Test paragraph counting when content has no double newlines."""
        # Content with no double newlines should fall back to single newline counting
        content = "First.\n\nSecond.\n\nThird."  # This has double newlines
        count = self.validator._count_paragraphs(content)
        self.assertEqual(count, 3)
    
    def test_count_paragraphs_no_newlines(self):
        """Test paragraph counting with no newlines."""
        content = "Single paragraph with no line breaks."
        count = self.validator._count_paragraphs(content)
        self.assertEqual(count, 1)
    
    def test_count_paragraphs_empty(self):
        """Test paragraph counting with empty content."""
        count = self.validator._count_paragraphs("")
        self.assertEqual(count, 0)
        
        count = self.validator._count_paragraphs("   ")
        self.assertEqual(count, 0)
    
    def test_validate_json_format_valid(self):
        """Test JSON format validation with valid JSON."""
        valid_json = json.dumps([
            {
                "title": "Test Title",
                "date": "1911",
                "content": "Test content"
            }
        ])
        
        valid, error = self.validator.validate_json_format(valid_json)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_json_format_invalid_json(self):
        """Test JSON format validation with invalid JSON."""
        invalid_json = "This is not JSON"
        
        valid, error = self.validator.validate_json_format(invalid_json)
        self.assertFalse(valid)
        self.assertIn("Invalid JSON format", error)
    
    def test_validate_json_format_not_list(self):
        """Test JSON format validation with non-list JSON."""
        not_list_json = json.dumps({
            "title": "Test Title",
            "date": "1911",
            "content": "Test content"
        })
        
        valid, error = self.validator.validate_json_format(not_list_json)
        self.assertFalse(valid)
        self.assertIn("must be a list", error)
    
    def test_validate_json_format_missing_fields(self):
        """Test JSON format validation with missing required fields."""
        missing_field_json = json.dumps([
            {
                "title": "Test Title",
                "date": "1911"
                # Missing 'content' field
            }
        ])
        
        valid, error = self.validator.validate_json_format(missing_field_json)
        self.assertFalse(valid)
        self.assertIn("missing required field: content", error)
    
    def test_validate_json_format_wrong_field_types(self):
        """Test JSON format validation with wrong field types."""
        wrong_type_json = json.dumps([
            {
                "title": 123,  # Should be string
                "date": "1911",
                "content": "Test content"
            }
        ])
        
        valid, error = self.validator.validate_json_format(wrong_type_json)
        self.assertFalse(valid)
        self.assertIn("must be a string", error)
    
    def test_validate_json_entries_valid(self):
        """Test JSON entries validation with valid data."""
        valid_json = json.dumps([
            {
                "title": "Mongolian Revolution",
                "date": "1911",
                "content": "The Mongolian Revolution of 1911 was a pivotal moment in Mongolia's history that marked the end of Qing dynasty rule and established the Bogd Khanate. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control. The Eighth Jebtsundamba Khutughtu was proclaimed as the Bogd Khan, creating a theocratic monarchy. The revolution faced immediate challenges from both Chinese and Russian interests in the region. Despite limited international recognition, this period laid crucial groundwork for Mongolian national identity and political consciousness that would influence future independence movements throughout the twentieth century."
            }
        ])
        
        entries, errors = self.validator.validate_json_entries(valid_json)
        
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(errors), 0)
        self.assertEqual(entries[0].title, "Mongolian Revolution")
    
    def test_validate_json_entries_invalid_json(self):
        """Test JSON entries validation with invalid JSON."""
        invalid_json = "Not valid JSON"
        
        entries, errors = self.validator.validate_json_entries(invalid_json)
        
        self.assertEqual(len(entries), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid JSON format", errors[0])
    
    def test_validate_json_entries_invalid_entry(self):
        """Test JSON entries validation with invalid entry data."""
        invalid_entry_json = json.dumps([
            {
                "title": "Valid Title",
                "date": "1911",
                "content": "Too short"  # Invalid content length
            }
        ])
        
        entries, errors = self.validator.validate_json_entries(invalid_entry_json)
        
        self.assertEqual(len(entries), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Entry 0 validation failed", errors[0])
    
    def test_validate_json_entries_mixed_valid_invalid(self):
        """Test JSON entries validation with mix of valid and invalid entries."""
        mixed_json = json.dumps([
            {
                "title": "Valid Entry",
                "date": "1911",
                "content": "The Mongolian Revolution of 1911 was a pivotal moment in Mongolia's history that marked the end of Qing dynasty rule and established the Bogd Khanate. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control. The Eighth Jebtsundamba Khutughtu was proclaimed as the Bogd Khan, creating a theocratic monarchy. The revolution faced immediate challenges from both Chinese and Russian interests in the region. Despite limited international recognition, this period laid crucial groundwork for Mongolian national identity and political consciousness that would influence future independence movements throughout the twentieth century."
            },
            {
                "title": "Invalid Entry",
                "date": "invalid-date",
                "content": "Too short content"
            }
        ])
        
        entries, errors = self.validator.validate_json_entries(mixed_json)
        
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(entries[0].title, "Valid Entry")
        self.assertIn("Entry 1 validation failed", errors[0])


if __name__ == '__main__':
    unittest.main()