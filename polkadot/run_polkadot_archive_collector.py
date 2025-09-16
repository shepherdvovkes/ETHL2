#!/usr/bin/env python3
"""
Polkadot Archive Data Collector Runner
======================================

Runner script for the Polkadot archive data collector with various options.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
from loguru import logger

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from polkadot_archive_data_collector import PolkadotArchiveCollector, CollectionConfig
from polkadot_archive_config import get_config, create_custom_config, CONFIGURATIONS

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Also log to file
    logger.add(
        f"logs/polkadot_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="100 MB",
        retention="7 days"
    )

async def run_collection(config_name: str, custom_params: dict = None):
    """Run archive data collection with specified configuration"""
    try:
        # Get configuration
        if config_name in CONFIGURATIONS:
            config = get_config(config_name)
            logger.info(f"Using predefined configuration: {config_name}")
        else:
            logger.error(f"Unknown configuration: {config_name}")
            logger.info(f"Available configurations: {list(CONFIGURATIONS.keys())}")
            return
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"Override: {key} = {value}")
                else:
                    logger.warning(f"Unknown parameter: {key}")
        
        # Convert to CollectionConfig
        collection_config = CollectionConfig(
            quicknode_url=config.quicknode_url,
            max_workers=config.max_workers,
            batch_size=config.batch_size,
            rate_limit_delay=config.rate_limit_delay,
            retry_attempts=config.retry_attempts,
            database_path=config.database_path,
            days_back=config.days_back,
            sample_rate=config.sample_rate
        )
        
        # Initialize and run collector
        collector = PolkadotArchiveCollector(collection_config)
        await collector.run_comprehensive_collection()
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Polkadot Archive Data Collector")
    
    parser.add_argument(
        '--config', 
        choices=list(CONFIGURATIONS.keys()),
        default='yearly',
        help='Predefined configuration to use'
    )
    
    parser.add_argument(
        '--days', 
        type=int,
        help='Number of days back to collect (overrides config)'
    )
    
    parser.add_argument(
        '--workers', 
        type=int,
        help='Number of parallel workers (overrides config)'
    )
    
    parser.add_argument(
        '--sample-rate', 
        type=int,
        help='Block sampling rate (overrides config)'
    )
    
    parser.add_argument(
        '--database', 
        type=str,
        help='Database path (overrides config)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quicknode-url',
        type=str,
        help='QuickNode endpoint URL (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Prepare custom parameters
    custom_params = {}
    if args.days:
        custom_params['days_back'] = args.days
    if args.workers:
        custom_params['max_workers'] = args.workers
    if args.sample_rate:
        custom_params['sample_rate'] = args.sample_rate
    if args.database:
        custom_params['database_path'] = args.database
    if args.quicknode_url:
        custom_params['quicknode_url'] = args.quicknode_url
    
    # Display configuration
    logger.info("🚀 Polkadot Archive Data Collector")
    logger.info("=" * 50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Custom parameters: {custom_params}")
    logger.info(f"Log level: {args.log_level}")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run collection
    try:
        asyncio.run(run_collection(args.config, custom_params))
        logger.success("✅ Collection completed successfully!")
    except KeyboardInterrupt:
        logger.warning("⚠️ Collection interrupted by user")
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
