"""
Configuration management for the Mongolian History Generator.

Handles environment variables, API keys, and application settings.
"""

import os
import logging
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for the Mongolian History Generator."""
    
    # OpenAI API Configuration
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.25
    max_tokens: int = 900
    max_retries: int = 3
    
    # Output Configuration
    output_dir: str = "./data/generated"
    output_filename: str = "mongolian_history_dataset.json"
    report_filename: str = "generation_report.json"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    @classmethod
    def from_environment(cls) -> 'Config':
        """Create configuration from environment variables."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return cls(
            openai_api_key=api_key,
            model_name=os.getenv('OPENAI_MODEL', "gpt-4o-mini"),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.25')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '900')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            output_dir=os.getenv('OUTPUT_DIR', './data/generated'),
            output_filename=os.getenv('OUTPUT_FILENAME', 'mongolian_history_dataset.json'),
            report_filename=os.getenv('REPORT_FILENAME', 'generation_report.json'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_dir=os.getenv('LOG_DIR', './logs')
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")


def get_default_topics() -> List[str]:
    """Get the default list of 20 historical topics for Mongolia's modern history."""
    return [
        "1911 revolution and the establishment of Bogd Khanate",
        "1915 Khiagta Treaty and Mongolian autonomy",
        "1921 Mongolian People's Revolution",
        "Formation of the Mongolian People's Republic in 1924",
        "Choibalsan era and Soviet influence (1930s-1950s)",
        "Collectivization and agricultural reforms in Mongolia",
        "World War II and Mongolia's role in the Pacific theater",
        "Post-war reconstruction and industrialization (1950s-1960s)",
        "Tsedenbal era and socialist development (1960s-1980s)",
        "Cultural Revolution impact on Mongolia in the 1960s",
        "Economic development and mining industry growth (1970s-1980s)",
        "Perestroika influence and political changes in late 1980s",
        "Democratic Revolution of 1990 in Mongolia",
        "Adoption of the 1992 Constitution and new governance structure",
        "Economic transition and privatization in the 1990s",
        "Mongolia's foreign policy between Russia, China, and the West after 1990",
        "Mining boom: Oyu Tolgoi, Tavan Tolgoi and resource-based growth",
        "Democratic consolidation and political parties development (2000s-2010s)",
        "Environmental challenges and sustainable development efforts",
        "COVID-19 pandemic impact on Mongolia (2020–2022)"
    ]