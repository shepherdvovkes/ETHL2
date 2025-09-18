#!/usr/bin/env python3
"""
Bitcoin Chain Collector Configuration
Configuration settings for the Bitcoin blockchain collector
"""

import os
from typing import Dict, Any

class CollectorConfig:
    """Configuration class for Bitcoin chain collector"""
    
    # QuickNode API Configuration
    QUICKNODE_ENDPOINT = "https://orbital-twilight-mansion.btc.quiknode.pro/a1280f4e959966b62d579978248263e3975e3b4d/"
    
    # Worker Configuration
    NUM_WORKERS = 10
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests
    REQUEST_TIMEOUT = 30  # seconds
    
    # Database Configuration
    DATABASE_PATH = "bitcoin_chain.db"
    BATCH_SIZE = 100  # Process in batches of 100 blocks
    
    # Collection Configuration
    START_HEIGHT = 0  # Genesis block
    END_HEIGHT = None  # None = current blockchain height
    RESUME_FROM_LAST = True  # Resume from last collected block
    
    # Progress Monitoring
    PROGRESS_UPDATE_INTERVAL = 30  # seconds
    LOG_LEVEL = "INFO"
    LOG_FILE = "bitcoin_collector.log"
    
    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    CONTINUE_ON_ERROR = True
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = 50
    CONNECTION_POOL_SIZE = 100
    
    # Data Storage
    SAVE_RAW_DATA = True  # Save raw JSON data
    COMPRESS_DATA = False  # Compress stored data
    
    @classmethod
    def from_env(cls) -> 'CollectorConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        config.QUICKNODE_ENDPOINT = os.getenv('QUICKNODE_ENDPOINT', config.QUICKNODE_ENDPOINT)
        config.NUM_WORKERS = int(os.getenv('NUM_WORKERS', config.NUM_WORKERS))
        config.DATABASE_PATH = os.getenv('DATABASE_PATH', config.DATABASE_PATH)
        config.START_HEIGHT = int(os.getenv('START_HEIGHT', config.START_HEIGHT))
        config.LOG_LEVEL = os.getenv('LOG_LEVEL', config.LOG_LEVEL)
        config.LOG_FILE = os.getenv('LOG_FILE', config.LOG_FILE)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'quicknode_endpoint': self.QUICKNODE_ENDPOINT,
            'num_workers': self.NUM_WORKERS,
            'rate_limit_delay': self.RATE_LIMIT_DELAY,
            'request_timeout': self.REQUEST_TIMEOUT,
            'database_path': self.DATABASE_PATH,
            'batch_size': self.BATCH_SIZE,
            'start_height': self.START_HEIGHT,
            'end_height': self.END_HEIGHT,
            'resume_from_last': self.RESUME_FROM_LAST,
            'progress_update_interval': self.PROGRESS_UPDATE_INTERVAL,
            'log_level': self.LOG_LEVEL,
            'log_file': self.LOG_FILE,
            'max_retries': self.MAX_RETRIES,
            'retry_delay': self.RETRY_DELAY,
            'continue_on_error': self.CONTINUE_ON_ERROR,
            'max_concurrent_requests': self.MAX_CONCURRENT_REQUESTS,
            'connection_pool_size': self.CONNECTION_POOL_SIZE,
            'save_raw_data': self.SAVE_RAW_DATA,
            'compress_data': self.COMPRESS_DATA
        }

# Default configuration instance
config = CollectorConfig.from_env()
