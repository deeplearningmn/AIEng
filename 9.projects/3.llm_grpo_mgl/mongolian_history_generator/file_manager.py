"""
File management utilities for the Mongolian History Generator.

Handles file operations, directory creation, and output formatting.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models.data_models import HistoricalEntry, GenerationResult, SummaryReport


class FileManager:
    """
    Manages file operations and output generation for the generator.
    
    Handles JSON output, summary reports, and directory management.
    """
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize file manager with output directory.
        
        Args:
            output_dir: Base directory for output files
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.ensure_directory_exists(self.output_dir)
    
    def ensure_directory_exists(self, directory: Path) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory: Path to directory to create
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ensured: {directory}")
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    def write_json_dataset(self, results: List[GenerationResult], filename: str) -> Path:
        """
        Write historical entries to JSON dataset file.
        
        Args:
            results: List of generation results containing entries
            filename: Output filename for the dataset
            
        Returns:
            Path to the written file
        """
        output_path = self.output_dir / filename
        
        # Collect all entries from successful results
        all_entries = []
        for result in results:
            if result.success and result.entries:
                for entry in result.entries:
                    all_entries.append(entry.to_dict())
        
        # Sort entries by date for better organization
        all_entries.sort(key=lambda x: x.get('date', ''))
        
        try:
            # Write JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_entries, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON dataset written to: {output_path} ({len(all_entries)} entries)")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to write JSON dataset to {output_path}: {e}")
            raise
    
    def write_summary_report(self, results: List[GenerationResult], 
                           filename: str, processing_time: float,
                           config_info: Dict[str, Any]) -> Path:
        """
        Write summary report with statistics and metadata.
        
        Args:
            results: List of generation results
            filename: Output filename for the report
            processing_time: Total processing time in seconds
            config_info: Configuration information to include
            
        Returns:
            Path to the written report file
        """
        report_path = self.output_dir / filename
        
        # Calculate statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_entries = sum(len(r.entries) for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        
        # Collect errors and warnings
        errors = []
        topics_processed = []
        
        for result in results:
            topic_info = {
                'topic': result.topic,
                'success': result.success,
                'entries_count': len(result.entries),
                'tokens_used': result.tokens_used
            }
            
            if not result.success and result.error_message:
                topic_info['error'] = result.error_message
                errors.append(f"{result.topic}: {result.error_message}")
            
            topics_processed.append(topic_info)
        
        # Create comprehensive report
        report_data = {
            'summary': {
                'generation_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'total_topics': len(results),
                'successful_generations': successful,
                'failed_generations': failed,
                'success_rate': round((successful / len(results)) * 100, 1) if results else 0,
                'total_entries': total_entries,
                'total_tokens_used': total_tokens,
                'average_entries_per_topic': round(total_entries / successful, 1) if successful > 0 else 0
            },
            'configuration': config_info,
            'topics_processed': topics_processed,
            'errors': errors,
            'file_info': {
                'dataset_entries': total_entries,
                'report_generated': datetime.now().isoformat()
            }
        }
        
        try:
            # Write report with proper formatting
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Summary report written to: {report_path}")
            self.logger.info(f"Statistics: {successful}/{len(results)} successful, "
                           f"{total_entries} entries, {total_tokens} tokens used")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to write summary report to {report_path}: {e}")
            raise
    
    def create_backup_filename(self, base_filename: str) -> str:
        """
        Create a backup filename with timestamp if file already exists.
        
        Args:
            base_filename: Original filename
            
        Returns:
            Backup filename with timestamp
        """
        file_path = self.output_dir / base_filename
        
        if not file_path.exists():
            return base_filename
        
        # Create timestamped backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = base_filename.rsplit('.', 1)
        
        if len(name_parts) == 2:
            backup_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            backup_name = f"{base_filename}_{timestamp}"
        
        return backup_name
    
    def backup_existing_file(self, filename: str) -> Optional[Path]:
        """
        Create a backup of an existing file before overwriting.
        
        Args:
            filename: Name of file to backup
            
        Returns:
            Path to backup file if created, None if no backup needed
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            return None
        
        backup_filename = self.create_backup_filename(filename)
        backup_path = self.output_dir / backup_filename
        
        try:
            # Copy existing file to backup location
            import shutil
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Existing file backed up: {file_path} -> {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup of {file_path}: {e}")
            return None
    
    def validate_json_output(self, filepath: Path) -> bool:
        """
        Validate that a JSON file is properly formatted and readable.
        
        Args:
            filepath: Path to JSON file to validate
            
        Returns:
            True if file is valid JSON, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            
            self.logger.debug(f"JSON validation successful: {filepath}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON validation failed for {filepath}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating JSON file {filepath}: {e}")
            return False
    
    def get_output_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about an output file.
        
        Args:
            filename: Name of file to get info for
            
        Returns:
            Dictionary with file information
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            return {'exists': False, 'path': str(file_path)}
        
        try:
            stat = file_path.stat()
            return {
                'exists': True,
                'path': str(file_path),
                'size_bytes': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'is_valid_json': self.validate_json_output(file_path) if file_path.suffix == '.json' else None
            }
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {e}")
            return {'exists': True, 'path': str(file_path), 'error': str(e)}
    
    def cleanup_old_files(self, pattern: str, keep_count: int = 5) -> List[Path]:
        """
        Clean up old files matching a pattern, keeping only the most recent ones.
        
        Args:
            pattern: File pattern to match (e.g., "*.log")
            keep_count: Number of most recent files to keep
            
        Returns:
            List of paths that were deleted
        """
        try:
            # Find all matching files
            matching_files = list(self.output_dir.glob(pattern))
            
            if len(matching_files) <= keep_count:
                return []
            
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Delete older files
            files_to_delete = matching_files[keep_count:]
            deleted_files = []
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_files.append(file_path)
                    self.logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {file_path}: {e}")
            
            if deleted_files:
                self.logger.info(f"Cleaned up {len(deleted_files)} old files matching '{pattern}'")
            
            return deleted_files
            
        except Exception as e:
            self.logger.error(f"Error during cleanup of files matching '{pattern}': {e}")
            return []