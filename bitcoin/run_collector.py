#!/usr/bin/env python3
"""
Bitcoin Chain Collector Runner
Simple script to run the Bitcoin blockchain collector
"""

import asyncio
import sys
import os
from bitcoin_chain_collector import BitcoinChainCollector
from collector_config import config

async def main():
    """Main function to run the collector"""
    print("🚀 Starting Bitcoin Chain Collector")
    print(f"📊 Configuration:")
    print(f"   - Workers: {config.NUM_WORKERS}")
    print(f"   - Database: {config.DATABASE_PATH}")
    print(f"   - Start Height: {config.START_HEIGHT}")
    print(f"   - End Height: {config.END_HEIGHT or 'Current blockchain height'}")
    print(f"   - QuickNode Endpoint: {config.QUICKNODE_ENDPOINT[:50]}...")
    print()
    
    # Create collector
    collector = BitcoinChainCollector(
        endpoint=config.QUICKNODE_ENDPOINT,
        num_workers=config.NUM_WORKERS
    )
    
    try:
        # Start collection
        await collector.start_collection(
            start_height=config.START_HEIGHT,
            end_height=config.END_HEIGHT
        )
        
        print("✅ Collection completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        sys.exit(1)
    finally:
        collector.stop()

if __name__ == "__main__":
    # Check if running in the correct directory
    if not os.path.exists("bitcoin_chain_collector.py"):
        print("❌ Error: Please run this script from the bitcoin directory")
        sys.exit(1)
    
    # Run the collector
    asyncio.run(main())
