"""
Entry point for running the Mongolian History Generator as a module.

Usage:
    python -m mongolian_history_generator
    python -m mongolian_history_generator --help
    python -m mongolian_history_generator --topics "1911 revolution" "1921 revolution"
"""

from .main import cli_main

if __name__ == "__main__":
    cli_main()