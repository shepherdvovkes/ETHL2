#!/usr/bin/env python3
"""
Comprehensive Data Collector Runner for PM2
Runs the data collection process continuously for PM2
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from collect_comprehensive_polkadot_data import ComprehensivePolkadotDataCollector

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/comprehensive-data-collector.log",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

class PM2DataCollectorRunner:
    """PM2-compatible data collector runner"""
    
    def __init__(self):
        self.collector = None
        self.running = True
        self.collection_interval = 600  # 10 minutes
    
    async def initialize(self):
        """Initialize the data collector"""
        try:
            self.collector = ComprehensivePolkadotDataCollector()
            await self.collector.initialize()
            logger.info("Data collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data collector: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.collector:
                await self.collector.cleanup()
                logger.info("Data collector cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def run_collection_cycle(self):
        """Run a single data collection cycle"""
        try:
            logger.info("Starting data collection cycle...")
            start_time = time.time()
            
            await self.collector.collect_all_metrics()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.success(f"Data collection cycle completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Data collection cycle failed: {e}")
            raise
    
    async def run_continuous(self):
        """Run data collection continuously"""
        logger.info("Starting continuous data collection...")
        logger.info(f"Collection interval: {self.collection_interval} seconds")
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                logger.info(f"Starting collection cycle #{cycle_count}")
                
                await self.run_collection_cycle()
                
                logger.info(f"Waiting {self.collection_interval} seconds until next collection...")
                await asyncio.sleep(self.collection_interval)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                logger.info("Waiting 60 seconds before retry...")
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop the data collector"""
        logger.info("Stopping data collector...")
        self.running = False

async def main():
    """Main function"""
    runner = PM2DataCollectorRunner()
    
    try:
        # Initialize
        await runner.initialize()
        
        # Run continuously
        await runner.run_continuous()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Data collector failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await runner.cleanup()
        logger.info("Data collector stopped")

if __name__ == "__main__":
    # Handle PM2 signals
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, stopping...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the main function
    asyncio.run(main())
