#!/usr/bin/env python3
"""
Command-line interface for the Mongolian History Generator.

This script provides a comprehensive CLI for generating structured historical datasets
about Mongolia's modern history (1911–present) using OpenAI's GPT-4o-mini model.

Usage Examples:
    # Generate dataset with default 20 topics
    python run_generator.py
    
    # Use custom output directory
    python run_generator.py --output-dir ./custom_output
    
    # Enable debug logging
    python run_generator.py --log-level DEBUG
    
    # Use custom topics from file
    python run_generator.py --topics-file my_topics.txt
    
    # Specify custom topics directly
    python run_generator.py --topics "1911 revolution" "1921 revolution"
    
    # Custom output filename
    python run_generator.py --output-filename my_dataset.json
    
    # Increase API retry attempts
    python run_generator.py --max-retries 5

Environment Variables:
    OPENAI_API_KEY     - Required: Your OpenAI API key
    OUTPUT_DIR         - Optional: Default output directory
    LOG_LEVEL          - Optional: Default logging level
    MAX_RETRIES        - Optional: Default retry attempts
"""

import argparse
import sys
import os
from typing import List, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mongolian_history_generator.main import main


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate structured historical datasets about Mongolia's modern history (1911–present)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default topics
  python run_generator.py
  
  # Custom output directory and debug logging
  python run_generator.py --output-dir ./my_output --log-level DEBUG
  
  # Use specific topics
  python run_generator.py --topics "1911 revolution" "Democratic Revolution of 1990"
  
  # Load topics from file
  python run_generator.py --topics-file custom_topics.txt
  
  # Custom configuration
  python run_generator.py --output-filename my_dataset.json --max-retries 5

Environment Setup:
  Set OPENAI_API_KEY environment variable with your OpenAI API key.
  
  Example:
    export OPENAI_API_KEY="sk-proj-your-api-key-here"
    python run_generator.py

Output Files:
  - mongolian_history_dataset.json: Generated historical entries
  - generation_report.json: Processing summary and statistics
  - logs/generation_YYYYMMDD_HHMMSS.log: Detailed execution logs
        """
    )
    
    # Topic selection options
    topic_group = parser.add_mutually_exclusive_group()
    topic_group.add_argument(
        '--topics',
        nargs='*',
        metavar='TOPIC',
        help='Custom topics to process (space-separated). If not specified, uses default 20 topics.'
    )
    
    topic_group.add_argument(
        '--topics-file',
        type=str,
        metavar='FILE',
        help='Path to file containing custom topics (one per line)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        metavar='DIR',
        help='Output directory for generated files (default: ./data/generated)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        metavar='FILE',
        help='Output JSON filename (default: mongolian_history_dataset.json)'
    )
    
    parser.add_argument(
        '--report-filename',
        type=str,
        metavar='FILE',
        help='Summary report filename (default: generation_report.json)'
    )
    
    # Logging configuration
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        metavar='DIR',
        help='Directory for log files (default: ./logs)'
    )
    
    # API configuration
    parser.add_argument(
        '--max-retries',
        type=int,
        metavar='N',
        help='Maximum number of API retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        metavar='MODEL',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        metavar='TEMP',
        help='Model temperature for generation (default: 0.25)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        metavar='N',
        help='Maximum tokens per API request (default: 900)'
    )
    
    # Utility options
    parser.add_argument(
        '--version',
        action='version',
        version='Mongolian History Generator 1.0.0'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and topics without making API calls'
    )
    
    return parser


def load_custom_topics(topics_file: str) -> List[str]:
    """Load custom topics from a file."""
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        if not topics:
            print(f"Warning: No valid topics found in '{topics_file}'", file=sys.stderr)
            return []
        
        print(f"Loaded {len(topics)} topics from '{topics_file}'")
        return topics
        
    except FileNotFoundError:
        print(f"Error: Topics file '{topics_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading topics file: {e}", file=sys.stderr)
        sys.exit(1)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Validate temperature range
    if args.temperature is not None and (args.temperature < 0 or args.temperature > 2):
        print("Error: Temperature must be between 0 and 2", file=sys.stderr)
        sys.exit(1)
    
    # Validate max_tokens
    if args.max_tokens is not None and args.max_tokens <= 0:
        print("Error: Max tokens must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Validate max_retries
    if args.max_retries is not None and args.max_retries < 0:
        print("Error: Max retries must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY') and not args.dry_run:
        print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'", file=sys.stderr)
        sys.exit(1)


def show_dry_run_info(args: argparse.Namespace, custom_topics: Optional[List[str]]) -> None:
    """Show configuration and topics for dry run."""
    from mongolian_history_generator.config import get_default_topics
    
    print("=== DRY RUN MODE ===")
    print("\nConfiguration:")
    print(f"  Output directory: {args.output_dir or './data/generated'}")
    print(f"  Output filename: {args.output_filename or 'mongolian_history_dataset.json'}")
    print(f"  Report filename: {args.report_filename or 'generation_report.json'}")
    print(f"  Log level: {args.log_level or 'INFO'}")
    print(f"  Log directory: {args.log_dir or './logs'}")
    print(f"  Model: {args.model or 'gpt-4o-mini'}")
    print(f"  Temperature: {args.temperature or 0.25}")
    print(f"  Max tokens: {args.max_tokens or 900}")
    print(f"  Max retries: {args.max_retries or 3}")
    
    topics = custom_topics if custom_topics else get_default_topics()
    print(f"\nTopics to process ({len(topics)}):")
    for i, topic in enumerate(topics, 1):
        print(f"  {i:2d}. {topic}")
    
    print(f"\nEstimated API calls: {len(topics)}")
    print("Note: Actual token usage will depend on response length")
    print("\nTo run for real, remove the --dry-run flag")


def main_cli():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Determine topics to use
    custom_topics = None
    if args.topics:
        custom_topics = args.topics
        print(f"Using {len(custom_topics)} custom topics from command line")
    elif args.topics_file:
        custom_topics = load_custom_topics(args.topics_file)
    
    # Handle dry run
    if args.dry_run:
        show_dry_run_info(args, custom_topics)
        return
    
    # Build configuration overrides
    config_override = {}
    if args.output_dir:
        config_override['output_dir'] = args.output_dir
    if args.output_filename:
        config_override['output_filename'] = args.output_filename
    if args.report_filename:
        config_override['report_filename'] = args.report_filename
    if args.log_level:
        config_override['log_level'] = args.log_level
    if args.log_dir:
        config_override['log_dir'] = args.log_dir
    if args.max_retries:
        config_override['max_retries'] = args.max_retries
    if args.model:
        config_override['model_name'] = args.model
    if args.temperature:
        config_override['temperature'] = args.temperature
    if args.max_tokens:
        config_override['max_tokens'] = args.max_tokens
    
    # Show startup information
    print("Mongolian History Generator v1.0.0")
    print("=" * 50)
    
    if custom_topics:
        print(f"Processing {len(custom_topics)} custom topics")
    else:
        print("Processing 20 default topics for Mongolia's modern history")
    
    print(f"Output directory: {config_override.get('output_dir', './data/generated')}")
    print(f"Log level: {config_override.get('log_level', 'INFO')}")
    print()
    
    # Run the generator
    try:
        exit_code = main(custom_topics=custom_topics, config_override=config_override)
        
        if exit_code == 0:
            print("\n" + "=" * 50)
            print("Generation completed successfully!")
            print("Check the output directory for generated files.")
        else:
            print("\n" + "=" * 50)
            print("Generation completed with errors.")
            print("Check the logs for details.")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()