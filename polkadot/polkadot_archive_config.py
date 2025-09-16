#!/usr/bin/env python3
"""
Polkadot Archive Data Collector Configuration
============================================

Configuration settings for the Polkadot archive data collector.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ArchiveCollectionConfig:
    """Configuration for archive data collection"""
    
    # QuickNode Configuration
    quicknode_url: str = "https://ancient-warmhearted-daylight.dot-mainnet.quiknode.pro/fc161dd4c4c279d2b0c5b3095ab2209673711fad/"
    
    # Parallel Processing
    max_workers: int = 30
    batch_size: int = 50  # Reduced batch size
    rate_limit_delay: float = 0.5  # Increased delay
    
    # Data Collection
    retry_attempts: int = 3
    database_path: str = "polkadot_archive_data.db"
    days_back: int = 365  # 1 year of historical data
    sample_rate: int = 10  # Sample every 10th block
    
    # Data Types to Collect
    collect_blocks: bool = True
    collect_staking: bool = True
    collect_parachains: bool = True
    collect_governance: bool = True
    collect_cross_chain: bool = True
    
    # Performance Settings
    connection_timeout: int = 60
    request_timeout: int = 30
    max_connections: int = 30
    
    # Data Validation
    validate_data: bool = True
    min_block_size: int = 100  # Minimum expected block size in bytes
    max_extrinsics_per_block: int = 1000  # Sanity check
    
    # Output Settings
    save_raw_data: bool = True
    save_processed_data: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'csv', 'parquet']

# Predefined configurations for different use cases
CONFIGURATIONS = {
    'quick_test': ArchiveCollectionConfig(
        days_back=7,
        sample_rate=1000,
        max_workers=5,
        batch_size=25,
        rate_limit_delay=1.0
    ),
    
    'monthly': ArchiveCollectionConfig(
        days_back=30,
        sample_rate=100,
        max_workers=10,
        batch_size=30,
        rate_limit_delay=0.8
    ),
    
    'quarterly': ArchiveCollectionConfig(
        days_back=90,
        sample_rate=50,
        max_workers=15,
        batch_size=40,
        rate_limit_delay=0.6
    ),
    
    'yearly': ArchiveCollectionConfig(
        days_back=365,
        sample_rate=20,
        max_workers=20,
        batch_size=50,
        rate_limit_delay=0.5
    ),
    
    'comprehensive': ArchiveCollectionConfig(
        days_back=365,
        sample_rate=10,  # Every 10th block (reduced from 5)
        max_workers=8,   # Reduced from 25 to 8
        batch_size=20,   # Reduced from 30 to 20
        rate_limit_delay=1.0  # Increased from 0.3 to 1.0
    ),
    
    'ultra_safe': ArchiveCollectionConfig(
        days_back=365,
        sample_rate=50,  # Every 50th block
        max_workers=3,   # Very conservative
        batch_size=10,   # Small batches
        rate_limit_delay=2.0  # Long delays
    )
}

def get_config(config_name: str = 'yearly') -> ArchiveCollectionConfig:
    """Get predefined configuration by name"""
    if config_name not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(CONFIGURATIONS.keys())}")
    
    return CONFIGURATIONS[config_name]

def create_custom_config(**kwargs) -> ArchiveCollectionConfig:
    """Create custom configuration with overrides"""
    base_config = ArchiveCollectionConfig()
    
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return base_config
