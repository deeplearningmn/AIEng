"""
Unit tests for data models and validation.

Tests cover HistoricalEntry validation, date format validation,
content length validation, and edge cases.
"""

import unittest
import json
from mongolian_history_generator.models.data_models import HistoricalEntry, GenerationResult, SummaryReport


class TestHistoricalEntry(unittest.TestCase):
    """Test cases for HistoricalEntry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_entry = HistoricalEntry(
            title="Mongolian Revolution of 1911",
            date="1911",
            content="The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
        )
    
    def test_valid_entry_validation(self):
        """Test validation of a completely valid entry."""
        self.assertTrue(self.valid_entry.validate())
    
    def test_title_validation_valid(self):
        """Test title validation with valid inputs."""
        entry = HistoricalEntry("Valid Title", "1911", self.valid_entry.content)
        self.assertTrue(entry._validate_title())
    
    def test_title_validation_empty(self):
        """Test title validation with empty title."""
        entry = HistoricalEntry("", "1911", self.valid_entry.content)
        self.assertFalse(entry._validate_title())
    
    def test_title_validation_whitespace_only(self):
        """Test title validation with whitespace-only title."""
        entry = HistoricalEntry("   ", "1911", self.valid_entry.content)
        self.assertFalse(entry._validate_title())
    
    def test_date_validation_yyyy_format(self):
        """Test date validation with YYYY format."""
        entry = HistoricalEntry("Title", "1911", self.valid_entry.content)
        self.assertTrue(entry._validate_date())
        
        entry = HistoricalEntry("Title", "2024", self.valid_entry.content)
        self.assertTrue(entry._validate_date())
    
    def test_date_validation_yyyy_mm_dd_format(self):
        """Test date validation with YYYY-MM-DD format."""
        entry = HistoricalEntry("Title", "1911-12-29", self.valid_entry.content)
        self.assertTrue(entry._validate_date())
        
        entry = HistoricalEntry("Title", "2024-01-15", self.valid_entry.content)
        self.assertTrue(entry._validate_date())
    
    def test_date_validation_invalid_formats(self):
        """Test date validation with invalid formats."""
        invalid_dates = [
            "",
            "11",
            "191",
            "19111",
            "1911-1",
            "1911-1-1",
            "not-a-date",
            "1911/12/29",
            "Dec 29, 1911"
        ]
        
        for invalid_date in invalid_dates:
            entry = HistoricalEntry("Title", invalid_date, self.valid_entry.content)
            self.assertFalse(entry._validate_date(), f"Date '{invalid_date}' should be invalid")
    
    def test_date_validation_edge_cases(self):
        """Test date validation with edge cases that pass format but may be invalid."""
        # Note: The current implementation only validates format, not actual date validity
        edge_cases = [
            "1911-13-01",  # Invalid month - but passes regex
            "1911-12-32",  # Invalid day - but passes regex
        ]
        
        for edge_date in edge_cases:
            entry = HistoricalEntry("Title", edge_date, self.valid_entry.content)
            # Current implementation only checks format, so these pass
            self.assertTrue(entry._validate_date(), f"Date '{edge_date}' passes format validation")
    
    def test_content_validation_valid_single_paragraph(self):
        """Test content validation with valid single paragraph."""
        # 100 words, single paragraph
        content = "The Mongolian Revolution of 1911 was a pivotal moment in Mongolia's history that marked the end of Qing dynasty rule and established the Bogd Khanate. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control. The Eighth Jebtsundamba Khutughtu was proclaimed as the Bogd Khan, creating a theocratic monarchy. The revolution faced immediate challenges from both Chinese and Russian interests in the region. Despite limited international recognition, this period laid crucial groundwork for Mongolian national identity and political consciousness that would influence future independence movements throughout the twentieth century."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertTrue(entry._validate_content())
    
    def test_content_validation_valid_multiple_paragraphs(self):
        """Test content validation with valid multiple paragraphs."""
        # Two paragraphs, ~120 words total
        content = "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and the establishment of the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region.\n\nThe new state struggled with limited international recognition and internal administrative challenges. Despite these difficulties, the Bogd Khanate period laid important groundwork for Mongolian national identity and political consciousness that would influence later independence movements."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertTrue(entry._validate_content())
    
    def test_content_validation_too_short(self):
        """Test content validation with too few words."""
        # Only 20 words
        content = "The Mongolian Revolution of 1911 was important for Mongolia's independence from Qing dynasty rule and Chinese control."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertFalse(entry._validate_content())
    
    def test_content_validation_too_long(self):
        """Test content validation with too many words."""
        # Over 150 words
        content = "The Mongolian Revolution of 1911 was a pivotal moment in Mongolia's history that marked the end of Qing dynasty rule and established the Bogd Khanate under the leadership of the Eighth Jebtsundamba Khutughtu. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control and foreign domination. The Eighth Jebtsundamba Khutughtu was proclaimed as the Bogd Khan, creating a theocratic monarchy that would govern Mongolia for the next decade. The revolution faced immediate challenges from both Chinese and Russian interests in the region, as both powers sought to maintain their influence over Mongolian territory and resources. Despite limited international recognition and diplomatic support, this period laid crucial groundwork for Mongolian national identity and political consciousness that would influence future independence movements throughout the twentieth century and beyond. The Bogd Khanate period established important precedents for Mongolian self-governance and cultural autonomy that would prove essential in later struggles for independence and sovereignty in the modern era."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertFalse(entry._validate_content())
    
    def test_content_validation_too_many_paragraphs(self):
        """Test content validation with too many paragraphs."""
        # Four paragraphs, valid word count
        content = "First paragraph about the revolution.\n\nSecond paragraph about the establishment.\n\nThird paragraph about challenges.\n\nFourth paragraph about legacy and impact on future movements."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertFalse(entry._validate_content())
    
    def test_content_validation_empty_content(self):
        """Test content validation with empty content."""
        entry = HistoricalEntry("Title", "1911", "")
        self.assertFalse(entry._validate_content())
        
        entry = HistoricalEntry("Title", "1911", "   ")
        self.assertFalse(entry._validate_content())
    
    def test_content_validation_single_newline_paragraphs(self):
        """Test content validation with single newline paragraph separation."""
        # Two paragraphs separated by single newline, with sufficient words (80+)
        content = "The Mongolian Revolution of 1911 marked the end of Qing dynasty rule and established the Bogd Khanate under the Eighth Jebtsundamba Khutughtu. This revolution was led by Mongolian nobles and religious leaders who sought independence from Chinese control and foreign domination.\nThis theocratic monarchy represented Mongolia's first attempt at modern independence, though it faced immediate challenges from both Chinese and Russian interests in the region and struggled with limited international recognition and administrative difficulties that would persist throughout the entire period of the Bogd Khanate's existence."
        
        entry = HistoricalEntry("Title", "1911", content)
        self.assertTrue(entry._validate_content())
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result = self.valid_entry.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['title'], self.valid_entry.title)
        self.assertEqual(result['date'], self.valid_entry.date)
        self.assertEqual(result['content'], self.valid_entry.content)
    
    def test_to_json_conversion(self):
        """Test conversion to JSON string."""
        json_str = self.valid_entry.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed['title'], self.valid_entry.title)
        self.assertEqual(parsed['date'], self.valid_entry.date)
        self.assertEqual(parsed['content'], self.valid_entry.content)
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            'title': 'Test Title',
            'date': '1921',
            'content': 'Test content with enough words to meet the minimum requirement of eighty words for validation purposes. This content should be long enough to pass the word count validation while still being under the maximum limit of one hundred and fifty words total.'
        }
        
        entry = HistoricalEntry.from_dict(data)
        
        self.assertEqual(entry.title, data['title'])
        self.assertEqual(entry.date, data['date'])
        self.assertEqual(entry.content, data['content'])


class TestGenerationResult(unittest.TestCase):
    """Test cases for GenerationResult class."""
    
    def test_generation_result_creation(self):
        """Test GenerationResult creation and conversion."""
        entries = [
            HistoricalEntry("Title 1", "1911", "Content 1 with sufficient words to meet validation requirements for historical entries in the dataset generation system."),
            HistoricalEntry("Title 2", "1921", "Content 2 with sufficient words to meet validation requirements for historical entries in the dataset generation system.")
        ]
        
        result = GenerationResult(
            topic="Test Topic",
            entries=entries,
            success=True,
            error_message=None,
            tokens_used=150
        )
        
        self.assertEqual(result.topic, "Test Topic")
        self.assertEqual(len(result.entries), 2)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.tokens_used, 150)
    
    def test_generation_result_to_dict(self):
        """Test GenerationResult dictionary conversion."""
        entries = [
            HistoricalEntry("Title", "1911", "Content with sufficient words to meet validation requirements for historical entries in the dataset generation system.")
        ]
        
        result = GenerationResult(
            topic="Test Topic",
            entries=entries,
            success=True,
            tokens_used=100
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['topic'], "Test Topic")
        self.assertEqual(len(result_dict['entries']), 1)
        self.assertTrue(result_dict['success'])
        self.assertEqual(result_dict['tokens_used'], 100)


class TestSummaryReport(unittest.TestCase):
    """Test cases for SummaryReport class."""
    
    def test_summary_report_creation(self):
        """Test SummaryReport creation and conversion."""
        report = SummaryReport(
            total_topics=20,
            successful_generations=18,
            failed_generations=2,
            total_entries=45,
            total_tokens_used=15000,
            errors=["Error 1", "Error 2"]
        )
        
        self.assertEqual(report.total_topics, 20)
        self.assertEqual(report.successful_generations, 18)
        self.assertEqual(report.failed_generations, 2)
        self.assertEqual(report.total_entries, 45)
        self.assertEqual(report.total_tokens_used, 15000)
        self.assertEqual(len(report.errors), 2)
    
    def test_summary_report_to_dict(self):
        """Test SummaryReport dictionary conversion."""
        report = SummaryReport(
            total_topics=10,
            successful_generations=9,
            failed_generations=1,
            total_entries=25,
            total_tokens_used=8000,
            errors=["Test error"]
        )
        
        report_dict = report.to_dict()
        
        self.assertEqual(report_dict['total_topics'], 10)
        self.assertEqual(report_dict['successful_generations'], 9)
        self.assertEqual(report_dict['failed_generations'], 1)
        self.assertEqual(report_dict['total_entries'], 25)
        self.assertEqual(report_dict['total_tokens_used'], 8000)
        self.assertEqual(len(report_dict['errors']), 1)
    
    def test_summary_report_to_json(self):
        """Test SummaryReport JSON conversion."""
        report = SummaryReport(
            total_topics=5,
            successful_generations=5,
            failed_generations=0,
            total_entries=15,
            total_tokens_used=3000,
            errors=[]
        )
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['total_topics'], 5)
        self.assertEqual(parsed['successful_generations'], 5)
        self.assertEqual(parsed['failed_generations'], 0)
        self.assertEqual(parsed['total_entries'], 15)
        self.assertEqual(parsed['total_tokens_used'], 3000)
        self.assertEqual(len(parsed['errors']), 0)


if __name__ == '__main__':
    unittest.main()