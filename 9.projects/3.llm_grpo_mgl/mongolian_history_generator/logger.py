"""
Logging configuration for the Mongolian History Generator.

Provides structured logging with appropriate levels, formatting, and performance monitoring.
"""

import logging
import os
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> logging.Logger:
    """
    Set up logging configuration with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('mongolian_history_generator')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"generation_{timestamp}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial setup
    logger.info(f"Logging initialized - Level: {log_level}, Log file: {log_filename}")
    
    return logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics and monitoring data."""
    
    start_time: float = field(default_factory=time.time)
    api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    total_tokens_used: int = 0
    topics_processed: int = 0
    successful_topics: int = 0
    failed_topics: int = 0
    total_entries_generated: int = 0
    processing_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_api_call(self, success: bool, tokens_used: int = 0):
        """Record an API call with success status and token usage."""
        self.api_calls += 1
        self.total_tokens_used += tokens_used
        
        if success:
            self.successful_api_calls += 1
        else:
            self.failed_api_calls += 1
    
    def add_topic_result(self, success: bool, entries_count: int, processing_time: float):
        """Record topic processing result."""
        self.topics_processed += 1
        self.processing_times.append(processing_time)
        self.total_entries_generated += entries_count
        
        if success:
            self.successful_topics += 1
        else:
            self.failed_topics += 1
    
    def add_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        elapsed_time = time.time() - self.start_time
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'elapsed_time_seconds': round(elapsed_time, 2),
            'api_calls': {
                'total': self.api_calls,
                'successful': self.successful_api_calls,
                'failed': self.failed_api_calls,
                'success_rate': round((self.successful_api_calls / self.api_calls) * 100, 1) if self.api_calls > 0 else 0
            },
            'topics': {
                'total_processed': self.topics_processed,
                'successful': self.successful_topics,
                'failed': self.failed_topics,
                'success_rate': round((self.successful_topics / self.topics_processed) * 100, 1) if self.topics_processed > 0 else 0
            },
            'performance': {
                'total_entries_generated': self.total_entries_generated,
                'total_tokens_used': self.total_tokens_used,
                'average_processing_time_seconds': round(avg_processing_time, 2),
                'entries_per_minute': round((self.total_entries_generated / elapsed_time) * 60, 1) if elapsed_time > 0 else 0,
                'tokens_per_minute': round((self.total_tokens_used / elapsed_time) * 60, 1) if elapsed_time > 0 else 0
            },
            'errors': dict(self.error_counts)
        }


class ProgressTracker:
    """Tracks and logs progress for long-running operations."""
    
    def __init__(self, logger: logging.Logger, total_items: int, operation_name: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            logger: Logger instance for progress updates
            total_items: Total number of items to process
            operation_name: Name of the operation being tracked
        """
        self.logger = logger
        self.total_items = total_items
        self.operation_name = operation_name
        self.current_item = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 5.0  # Update every 5 seconds minimum
        
    def update(self, increment: int = 1, item_name: Optional[str] = None) -> None:
        """
        Update progress and log if enough time has passed.
        
        Args:
            increment: Number of items completed
            item_name: Optional name of current item
        """
        self.current_item += increment
        current_time = time.time()
        
        # Log progress if enough time has passed or if we're done
        if (current_time - self.last_update_time >= self.update_interval or 
            self.current_item >= self.total_items):
            
            self._log_progress(item_name)
            self.last_update_time = current_time
    
    def _log_progress(self, item_name: Optional[str] = None) -> None:
        """Log current progress with timing information."""
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.current_item / self.total_items) * 100
        
        # Calculate ETA
        if self.current_item > 0:
            avg_time_per_item = elapsed_time / self.current_item
            remaining_items = self.total_items - self.current_item
            eta_seconds = remaining_items * avg_time_per_item
            eta_str = f", ETA: {eta_seconds:.0f}s"
        else:
            eta_str = ""
        
        item_info = f" - {item_name}" if item_name else ""
        
        self.logger.info(f"{self.operation_name}: {self.current_item}/{self.total_items} "
                        f"({progress_pct:.1f}%) - Elapsed: {elapsed_time:.1f}s{eta_str}{item_info}")
    
    def complete(self) -> None:
        """Mark operation as complete and log final statistics."""
        total_time = time.time() - self.start_time
        avg_time = total_time / self.total_items if self.total_items > 0 else 0
        
        self.logger.info(f"{self.operation_name} completed: {self.total_items} items in "
                        f"{total_time:.1f}s (avg: {avg_time:.2f}s per item)")


class MonitoringLogger:
    """Enhanced logger with performance monitoring and progress tracking."""
    
    def __init__(self, base_logger: logging.Logger):
        """
        Initialize monitoring logger.
        
        Args:
            base_logger: Base logger instance to wrap
        """
        self.logger = base_logger
        self.metrics = PerformanceMetrics()
        self._lock = threading.Lock()
    
    def __getattr__(self, name):
        """Delegate attribute access to the base logger."""
        return getattr(self.logger, name)
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager to time an operation and log the duration.
        
        Args:
            operation_name: Name of the operation being timed
        """
        start_time = time.time()
        self.logger.debug(f"Starting {operation_name}")
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(f"{operation_name} completed in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
            raise
    
    def log_api_call(self, success: bool, tokens_used: int = 0, error_type: Optional[str] = None):
        """
        Log an API call with metrics tracking.
        
        Args:
            success: Whether the API call was successful
            tokens_used: Number of tokens used in the call
            error_type: Type of error if call failed
        """
        with self._lock:
            self.metrics.add_api_call(success, tokens_used)
            
            if success:
                self.logger.debug(f"API call successful, tokens used: {tokens_used}")
            else:
                self.logger.warning(f"API call failed: {error_type}")
                if error_type:
                    self.metrics.add_error(error_type)
    
    def log_topic_processing(self, topic: str, success: bool, entries_count: int, 
                           processing_time: float, error_msg: Optional[str] = None):
        """
        Log topic processing result with metrics.
        
        Args:
            topic: Name of the topic processed
            success: Whether processing was successful
            entries_count: Number of entries generated
            processing_time: Time taken to process
            error_msg: Error message if processing failed
        """
        with self._lock:
            self.metrics.add_topic_result(success, entries_count, processing_time)
            
            if success:
                self.logger.info(f"Topic '{topic}' processed successfully: "
                               f"{entries_count} entries in {processing_time:.2f}s")
            else:
                self.logger.error(f"Topic '{topic}' processing failed: {error_msg}")
                if error_msg:
                    self.metrics.add_error("topic_processing_error")
    
    def create_progress_tracker(self, total_items: int, operation_name: str = "Processing") -> ProgressTracker:
        """
        Create a progress tracker for long-running operations.
        
        Args:
            total_items: Total number of items to process
            operation_name: Name of the operation
            
        Returns:
            ProgressTracker instance
        """
        return ProgressTracker(self.logger, total_items, operation_name)
    
    def log_metrics_summary(self):
        """Log a comprehensive summary of all collected metrics."""
        summary = self.metrics.get_summary()
        
        self.logger.info("=== Performance Metrics Summary ===")
        self.logger.info(f"Total elapsed time: {summary['elapsed_time_seconds']}s")
        
        # API metrics
        api_metrics = summary['api_calls']
        self.logger.info(f"API calls: {api_metrics['total']} total, "
                        f"{api_metrics['successful']} successful ({api_metrics['success_rate']}%)")
        
        # Topic metrics
        topic_metrics = summary['topics']
        self.logger.info(f"Topics: {topic_metrics['total_processed']} processed, "
                        f"{topic_metrics['successful']} successful ({topic_metrics['success_rate']}%)")
        
        # Performance metrics
        perf_metrics = summary['performance']
        self.logger.info(f"Generated: {perf_metrics['total_entries_generated']} entries, "
                        f"{perf_metrics['total_tokens_used']} tokens used")
        self.logger.info(f"Rate: {perf_metrics['entries_per_minute']} entries/min, "
                        f"{perf_metrics['tokens_per_minute']} tokens/min")
        
        # Error summary
        if summary['errors']:
            self.logger.warning(f"Errors encountered: {summary['errors']}")
        
        self.logger.info("=== End Metrics Summary ===")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self._lock:
            return self.metrics.get_summary()


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> MonitoringLogger:
    """
    Set up logging configuration with file and console handlers, enhanced with monitoring.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        
    Returns:
        Configured MonitoringLogger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('mongolian_history_generator')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"generation_{timestamp}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create monitoring logger wrapper
    monitoring_logger = MonitoringLogger(logger)
    
    # Log initial setup
    monitoring_logger.info(f"Enhanced logging initialized - Level: {log_level}, Log file: {log_filename}")
    
    return monitoring_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (defaults to mongolian_history_generator)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'mongolian_history_generator.{name}')
    return logging.getLogger('mongolian_history_generator')