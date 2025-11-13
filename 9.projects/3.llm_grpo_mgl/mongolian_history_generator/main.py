"""
Main entry point for the Mongolian History Generator.

This module serves as the primary orchestrator for the data generation process.
"""

import os
import sys
import signal
import argparse
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .config import Config, get_default_topics
from .logger import setup_logging, get_logger
from .topic_processor import TopicProcessor
from .file_manager import FileManager
from .models.data_models import GenerationResult, SummaryReport


class MongolianHistoryGenerator:
    """
    Main application orchestrator for the Mongolian History Generator.
    
    Coordinates all components and manages the complete data generation workflow.
    """
    
    def __init__(self, config: Config):
        """Initialize the generator with configuration."""
        self.config = config
        self.logger = setup_logging(config.log_level, config.log_dir)
        self.topic_processor = TopicProcessor(config)
        self.file_manager = FileManager(config.output_dir, self.logger)
        self.shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Mongolian History Generator initialized")
        self.logger.info(f"Configuration: Model={config.model_name}, Temperature={config.temperature}")
        
        # Log system information
        self.logger.debug(f"Output directory: {config.output_dir}")
        self.logger.debug(f"Max retries: {config.max_retries}")
        self.logger.debug(f"Max tokens: {config.max_tokens}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def run(self, custom_topics: Optional[List[str]] = None) -> int:
        """
        Run the complete data generation process.
        
        Args:
            custom_topics: Optional list of custom topics to process
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        with self.logger.time_operation("Complete generation process"):
            try:
                # Create output directory
                os.makedirs(self.config.output_dir, exist_ok=True)
                self.logger.info(f"Output directory: {self.config.output_dir}")
                
                # Get topics to process
                topics = custom_topics if custom_topics else get_default_topics()
                self.logger.info(f"Processing {len(topics)} topics")
                
                # Process topics with enhanced progress tracking
                results = self._process_topics_with_monitoring(topics)
                
                if self.shutdown_requested:
                    self.logger.warning("Shutdown requested, stopping processing")
                    return 1
                
                # Generate outputs using file manager
                with self.logger.time_operation("Output generation"):
                    processing_time = time.time() - self.logger.metrics.start_time
                    self._generate_outputs(results, processing_time)
                
                # Log comprehensive metrics summary
                self.logger.log_metrics_summary()
                
                # Final status
                successful = sum(1 for r in results if r.success)
                total_entries = sum(len(r.entries) for r in results)
                
                self.logger.info(f"Generation completed: {successful}/{len(topics)} topics successful, "
                               f"{total_entries} total entries generated")
                
                return 0 if successful > 0 else 1
                
            except Exception as e:
                self.logger.error(f"Fatal error during generation: {e}")
                return 1
            finally:
                self._cleanup()
    
    def _process_topics_with_monitoring(self, topics: List[str]) -> List[GenerationResult]:
        """Process topics with enhanced monitoring and progress tracking."""
        results = []
        progress_tracker = self.logger.create_progress_tracker(len(topics), "Topic processing")
        
        for i, topic in enumerate(topics):
            if self.shutdown_requested:
                self.logger.warning(f"Shutdown requested, stopping at topic {i+1}/{len(topics)}")
                break
            
            try:
                # Time individual topic processing
                topic_start_time = time.time()
                
                with self.logger.time_operation(f"Processing topic: {topic}"):
                    result = self.topic_processor.process_topic(topic)
                
                processing_time = time.time() - topic_start_time
                
                # Log topic result with monitoring
                self.logger.log_topic_processing(
                    topic=topic,
                    success=result.success,
                    entries_count=len(result.entries),
                    processing_time=processing_time,
                    error_msg=result.error_message
                )
                
                results.append(result)
                progress_tracker.update(1, topic)
                
            except Exception as e:
                processing_time = time.time() - topic_start_time if 'topic_start_time' in locals() else 0
                error_msg = str(e)
                
                self.logger.log_topic_processing(
                    topic=topic,
                    success=False,
                    entries_count=0,
                    processing_time=processing_time,
                    error_msg=error_msg
                )
                
                # Create failed result
                failed_result = GenerationResult(
                    topic=topic,
                    entries=[],
                    success=False,
                    error_message=error_msg,
                    tokens_used=0
                )
                results.append(failed_result)
                progress_tracker.update(1, f"{topic} (FAILED)")
        
        progress_tracker.complete()
        return results
    
    def _generate_outputs(self, results: List[GenerationResult], processing_time: float) -> None:
        """Generate all output files using the file manager."""
        try:
            # Backup existing files if they exist
            self.file_manager.backup_existing_file(self.config.output_filename)
            self.file_manager.backup_existing_file(self.config.report_filename)
            
            # Generate JSON dataset
            dataset_path = self.file_manager.write_json_dataset(results, self.config.output_filename)
            
            # Validate JSON output
            if not self.file_manager.validate_json_output(dataset_path):
                self.logger.error("Generated JSON dataset failed validation")
            
            # Generate summary report with configuration info
            config_info = {
                'model': self.config.model_name,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'max_retries': self.config.max_retries
            }
            
            report_path = self.file_manager.write_summary_report(
                results, self.config.report_filename, processing_time, config_info
            )
            
            # Log file information
            dataset_info = self.file_manager.get_output_file_info(self.config.output_filename)
            report_info = self.file_manager.get_output_file_info(self.config.report_filename)
            
            self.logger.info(f"Dataset file: {dataset_info.get('size_bytes', 0)} bytes")
            self.logger.info(f"Report file: {report_info.get('size_bytes', 0)} bytes")
            
        except Exception as e:
            self.logger.error(f"Error generating output files: {e}")
            raise
    
    def _cleanup(self) -> None:
        """Perform cleanup operations."""
        self.logger.info("Performing cleanup...")
        
        try:
            # Clean up old log files (keep last 10)
            self.file_manager.cleanup_old_files("*.log", keep_count=10)
            
            # Clean up old backup files (keep last 5)
            self.file_manager.cleanup_old_files("*_backup_*.json", keep_count=5)
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
        
        self.logger.info("Cleanup completed")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate structured historical datasets about Mongolia's modern history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mongolian_history_generator
  python -m mongolian_history_generator --topics "1911 revolution" "1921 revolution"
  python -m mongolian_history_generator --output-dir ./custom_output
  python -m mongolian_history_generator --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--topics',
        nargs='*',
        help='Custom topics to process (default: use predefined 20 topics)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for generated files (default: ./data/generated)'
    )
    
    parser.add_argument(
        '--output-filename',
        help='Output JSON filename (default: mongolian_history_dataset.json)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        help='Maximum API retry attempts (default: 3)'
    )
    
    return parser


def main(custom_topics: Optional[List[str]] = None, config_override: Optional[Dict[str, Any]] = None) -> int:
    """
    Main function to run the Mongolian History Generator.
    
    Args:
        custom_topics: Optional list of custom topics to process
        config_override: Optional configuration overrides
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load configuration
        config = Config.from_environment()
        
        # Apply any configuration overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Validate configuration
        config.validate()
        
        # Create and run generator
        generator = MongolianHistoryGenerator(config)
        return generator.run(custom_topics)
        
    except Exception as e:
        print(f"Fatal error during initialization: {e}", file=sys.stderr)
        return 1


def cli_main() -> int:
    """Command-line interface main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Build configuration overrides from CLI arguments
    config_override = {}
    if args.output_dir:
        config_override['output_dir'] = args.output_dir
    if args.output_filename:
        config_override['output_filename'] = args.output_filename
    if args.log_level:
        config_override['log_level'] = args.log_level
    if args.max_retries:
        config_override['max_retries'] = args.max_retries
    
    # Run with CLI arguments
    return main(custom_topics=args.topics, config_override=config_override)


if __name__ == "__main__":
    sys.exit(cli_main())