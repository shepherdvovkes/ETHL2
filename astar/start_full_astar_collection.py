#!/usr/bin/env python3
"""
Start Full Astar Chain Collection
================================

Script to start full Astar blockchain collection with market data integration.
Uses 10 workers for optimal performance.
"""

import asyncio
import sys
import os
from datetime import datetime
from loguru import logger

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from astar_market_enhanced_multithreaded import AstarMarketEnhancedMultiThreadedCollector

async def get_current_block_info():
    """Get current block information"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                'jsonrpc': '2.0',
                'method': 'eth_blockNumber',
                'params': [],
                'id': 1
            }
            
            async with session.post('https://rpc.astar.network', json=payload) as response:
                result = await response.json()
                current_block = int(result.get('result', '0x0'), 16)
                
                # Get block info
                payload = {
                    'jsonrpc': '2.0',
                    'method': 'eth_getBlockByNumber',
                    'params': [hex(current_block), False],
                    'id': 1
                }
                
                async with session.post('https://rpc.astar.network', json=payload) as response:
                    block_result = await response.json()
                    block_data = block_result.get('result', {})
                    timestamp = int(block_data.get('timestamp', '0x0'), 16)
                    block_time = datetime.fromtimestamp(timestamp)
                    
                    return current_block, block_time
    except Exception as e:
        logger.error(f"Error getting current block info: {e}")
        return None, None

async def check_database_status():
    """Check current database status"""
    import sqlite3
    
    print("🔍 Checking Database Status")
    print("=" * 50)
    
    # Check existing database
    if os.path.exists('astar_multithreaded_data.db'):
        conn = sqlite3.connect('astar_multithreaded_data.db')
        cursor = conn.cursor()
        
        try:
            # Get block counts
            cursor.execute("SELECT COUNT(*) FROM astar_blocks")
            block_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_transactions")
            tx_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
            block_range = cursor.fetchone()
            
            print(f"📊 Existing Database:")
            print(f"   📦 Blocks: {block_count:,}")
            print(f"   💸 Transactions: {tx_count:,}")
            if block_range[0] and block_range[1]:
                print(f"   📈 Block Range: {block_range[0]:,} - {block_range[1]:,}")
            
            conn.close()
            return block_range[1] if block_range[1] else 0
        except Exception as e:
            print(f"❌ Error checking database: {e}")
            conn.close()
            return 0
    else:
        print("📊 No existing database found - starting fresh")
        return 0

async def start_full_collection():
    """Start full Astar chain collection"""
    print("🚀 Starting Full Astar Chain Collection")
    print("=" * 50)
    
    # Get current block info
    current_block, block_time = await get_current_block_info()
    if not current_block:
        print("❌ Failed to get current block information")
        return False
    
    print(f"📊 Current Astar Network Status:")
    print(f"   📦 Current Block: {current_block:,}")
    print(f"   📅 Block Time: {block_time}")
    
    # Check existing data
    last_collected = await check_database_status()
    
    if last_collected > 0:
        start_block = last_collected + 1
        blocks_to_collect = current_block - start_block + 1
        print(f"📊 Collection Plan:")
        print(f"   📦 Starting from block: {start_block:,}")
        print(f"   📦 Ending at block: {current_block:,}")
        print(f"   📦 Total blocks to collect: {blocks_to_collect:,}")
    else:
        # Start from a reasonable point (last 30 days)
        blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
        start_block = max(1, current_block - (30 * blocks_per_day))
        blocks_to_collect = current_block - start_block + 1
        print(f"📊 Collection Plan (Last 30 days):")
        print(f"   📦 Starting from block: {start_block:,}")
        print(f"   📦 Ending at block: {current_block:,}")
        print(f"   📦 Total blocks to collect: {blocks_to_collect:,}")
    
    # Calculate estimated time
    blocks_per_second = 10  # Conservative estimate with 10 workers
    estimated_time_hours = blocks_to_collect / (blocks_per_second * 3600)
    
    print(f"⏱️  Estimated Collection Time:")
    print(f"   🕐 {estimated_time_hours:.1f} hours")
    print(f"   🕐 {estimated_time_hours/24:.1f} days")
    
    # Confirm before starting
    print(f"\n⚠️  This will collect {blocks_to_collect:,} blocks with market data integration")
    print(f"   💾 Database size will grow significantly")
    print(f"   🌐 Network requests will be made continuously")
    print(f"   ⚡ Using 10 concurrent workers")
    
    # Start collection
    print(f"\n🚀 Starting collection at {datetime.now()}")
    print("=" * 50)
    
    try:
        # Initialize collector with 10 workers
        async with AstarMarketEnhancedMultiThreadedCollector(max_workers=10) as collector:
            # Calculate days to collect
            blocks_per_day = 24 * 60 * 60 // 6
            days_to_collect = (current_block - start_block) // blocks_per_day + 1
            
            print(f"🎯 Starting collection for {days_to_collect} days with market data integration")
            
            # Start collection
            await collector.collect_comprehensive_market_enhanced_data(days_back=days_to_collect)
            
            print(f"\n🎉 Full Astar chain collection completed at {datetime.now()}")
            return True
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Collection interrupted by user at {datetime.now()}")
        print("   💾 Partial data has been saved")
        return False
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return False

async def main():
    """Main function"""
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    print("🚀 Astar Full Chain Collection with Market Data")
    print("=" * 60)
    print("This script will collect the full Astar blockchain data")
    print("with integrated CoinGecko market data using 10 workers.")
    print("=" * 60)
    
    success = await start_full_collection()
    
    if success:
        print("\n✅ Full Astar chain collection completed successfully!")
        print("   📊 Database is ready for ML training")
        print("   💰 Market data integrated")
        print("   🚀 Ready for analysis and predictions")
    else:
        print("\n⚠️  Collection completed with issues")
        print("   💾 Check database for partial data")
        print("   🔍 Review logs for any errors")

if __name__ == "__main__":
    asyncio.run(main())
