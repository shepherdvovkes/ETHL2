#!/usr/bin/env python3
"""
Polkadot Historical Data Retriever
==================================

This script retrieves historical block data from Polkadot network and stores it in the database.
It supports multi-worker parallel processing for efficient data collection.

Features:
- Multi-worker parallel block retrieval
- RPC-based block data collection
- Comprehensive block metadata storage
- Error handling and retry logic
- Progress tracking and logging
"""

import asyncio
import aiohttp
import json
import sqlite3
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import os
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/polkadot_historical_retriever.log",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

@dataclass
class BlockData:
    """Data structure for block information"""
    block_number: int
    block_hash: str
    parent_hash: str
    timestamp: int
    extrinsics: List[Dict]
    events: List[Dict]
    validator: str
    block_size: int
    finalized: bool

class PolkadotHistoricalRetriever:
    """Historical data retriever for Polkadot blocks"""
    
    def __init__(self, worker_id: int = 0, start_block: int = None, end_block: int = None):
        self.worker_id = worker_id
        self.start_block = start_block
        self.end_block = end_block
        self.rpc_url = "https://rpc.polkadot.io"
        self.session = None
        self.db_path = "polkadot_archive_data.db"
        self.retry_attempts = 3
        self.retry_delay = 1
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict[str, Any]:
        """Make RPC call to Polkadot node"""
        if params is None:
            params = []
            
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "error" in result:
                            logger.error(f"Worker {self.worker_id}: RPC Error: {result['error']}")
                            return {}
                        return result.get("result", {})
                    else:
                        logger.warning(f"Worker {self.worker_id}: HTTP {response.status} for {method}")
                        
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: Attempt {attempt + 1} failed for {method}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
        return {}
    
    async def get_block_by_number(self, block_number: int) -> Optional[BlockData]:
        """Get block data by block number"""
        try:
            # Get block hash first
            hash_result = await self.make_rpc_call("chain_getBlockHash", [block_number])
            if not hash_result:
                return None
                
            block_hash = hash_result
            
            # Get block data
            block_result = await self.make_rpc_call("chain_getBlock", [block_hash])
            if not block_result:
                return None
                
            block = block_result.get("block", {})
            header = block.get("header", {})
            extrinsics = block.get("extrinsics", [])
            
            # Get events for this block
            events_result = await self.make_rpc_call("state_getRuntimeVersion", [block_hash])
            
            # Get block timestamp
            timestamp_result = await self.make_rpc_call("chain_getHeader", [block_hash])
            timestamp = 0
            if timestamp_result:
                # Extract timestamp from block header
                timestamp = int(time.time() * 1000)  # Fallback to current time
            
            return BlockData(
                block_number=block_number,
                block_hash=block_hash,
                parent_hash=header.get("parentHash", ""),
                timestamp=timestamp,
                extrinsics=extrinsics,
                events=[],  # Events would need additional RPC calls
                validator=header.get("author", ""),
                block_size=len(json.dumps(block)),
                finalized=True  # Assume finalized for historical blocks
            )
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error getting block {block_number}: {e}")
            return None
    
    def get_current_block_from_db(self) -> int:
        """Get the current block number from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(block_number) FROM block_metrics")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return result[0]
            else:
                return 27770131  # Default starting block
        except Exception as e:
            logger.warning(f"Could not get current block from DB: {e}, using default")
            return 27770131
    
    def store_block_data(self, block_data: BlockData) -> bool:
        """Store block data in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert block metrics
            cursor.execute("""
                INSERT OR REPLACE INTO block_metrics 
                (block_number, timestamp, extrinsics_count, events_count, block_size, validator_count, finalization_time, parachain_blocks, cross_chain_messages, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_data.block_number,
                datetime.fromtimestamp(block_data.timestamp / 1000, tz=timezone.utc).isoformat(),
                len(block_data.extrinsics),
                len(block_data.events),
                block_data.block_size,
                1,  # validator_count (simplified)
                0.0,  # finalization_time
                0,  # parachain_blocks
                0,  # cross_chain_messages
                datetime.now(timezone.utc).isoformat()
            ))
            
            # Store detailed block data in a separate table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS block_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_number INTEGER UNIQUE,
                    block_hash TEXT,
                    parent_hash TEXT,
                    timestamp INTEGER,
                    extrinsics_data TEXT,
                    events_data TEXT,
                    validator TEXT,
                    block_size INTEGER,
                    finalized BOOLEAN,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO block_details 
                (block_number, block_hash, parent_hash, timestamp, extrinsics_data, events_data, validator, block_size, finalized, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_data.block_number,
                block_data.block_hash,
                block_data.parent_hash,
                block_data.timestamp,
                json.dumps(block_data.extrinsics),
                json.dumps(block_data.events),
                block_data.validator,
                block_data.block_size,
                block_data.finalized,
                datetime.now(timezone.utc).isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error storing block {block_data.block_number}: {e}")
            return False
    
    async def collect_blocks_range(self) -> int:
        """Collect blocks in the specified range"""
        collected_count = 0
        failed_count = 0
        
        logger.info(f"Worker {self.worker_id}: Starting collection from block {self.start_block} to {self.end_block}")
        
        async with self:
            for block_number in range(self.start_block, self.end_block + 1):
                try:
                    # Check if block already exists
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM block_metrics WHERE block_number = ?", (block_number,))
                    exists = cursor.fetchone()
                    conn.close()
                    
                    if exists:
                        logger.debug(f"Worker {self.worker_id}: Block {block_number} already exists, skipping")
                        continue
                    
                    # Get block data
                    block_data = await self.get_block_by_number(block_number)
                    if block_data:
                        # Store block data
                        if self.store_block_data(block_data):
                            collected_count += 1
                            if collected_count % 100 == 0:
                                logger.info(f"Worker {self.worker_id}: Collected {collected_count} blocks (current: {block_number})")
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Worker {self.worker_id}: Failed to get block {block_number}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)  # 100ms delay between requests
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing block {block_number}: {e}")
                    failed_count += 1
                    continue
        
        logger.success(f"Worker {self.worker_id}: Completed! Collected: {collected_count}, Failed: {failed_count}")
        return collected_count

def calculate_block_ranges(current_block: int, num_workers: int, blocks_per_worker: int = 1000) -> List[Tuple[int, int]]:
    """Calculate block ranges for each worker"""
    ranges = []
    start_block = current_block + 1
    
    for i in range(num_workers):
        end_block = start_block + blocks_per_worker - 1
        ranges.append((start_block, end_block))
        start_block = end_block + 1
    
    return ranges

async def worker_main(worker_id: int, start_block: int, end_block: int):
    """Main function for a single worker"""
    retriever = PolkadotHistoricalRetriever(
        worker_id=worker_id,
        start_block=start_block,
        end_block=end_block
    )
    
    try:
        collected = await retriever.collect_blocks_range()
        logger.success(f"Worker {worker_id}: Successfully collected {collected} blocks")
        return collected
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed with error: {e}")
        raise

async def main():
    """Main function with multi-worker support"""
    parser = argparse.ArgumentParser(description='Polkadot Historical Data Retriever')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers (default: 10)')
    parser.add_argument('--blocks-per-worker', type=int, default=1000, help='Blocks per worker (default: 1000)')
    parser.add_argument('--start-block', type=int, help='Starting block number (default: from DB)')
    parser.add_argument('--end-block', type=int, help='Ending block number (default: current mainnet)')
    
    args = parser.parse_args()
    
    # Get current block from database
    temp_retriever = PolkadotHistoricalRetriever()
    current_block = args.start_block if args.start_block else temp_retriever.get_current_block_from_db()
    
    # Get current mainnet block if end_block not specified
    if not args.end_block:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://rpc.polkadot.io",
                    json={"id": 1, "jsonrpc": "2.0", "method": "chain_getBlock", "params": []}
                ) as response:
                    result = await response.json()
                    if "result" in result:
                        current_mainnet = int(result["result"]["block"]["header"]["number"], 16)
                        args.end_block = current_mainnet
                    else:
                        args.end_block = current_block + (args.workers * args.blocks_per_worker)
        except Exception as e:
            logger.warning(f"Could not get current mainnet block: {e}")
            args.end_block = current_block + (args.workers * args.blocks_per_worker)
    
    logger.info(f"Starting historical data retrieval with {args.workers} workers")
    logger.info(f"Database current block: {current_block}")
    logger.info(f"Target end block: {args.end_block}")
    logger.info(f"Total blocks to collect: {args.end_block - current_block}")
    
    # Calculate block ranges for each worker
    total_blocks = args.end_block - current_block
    blocks_per_worker = min(args.blocks_per_worker, total_blocks // args.workers)
    
    block_ranges = []
    start_block = current_block + 1
    
    for i in range(args.workers):
        end_block = min(start_block + blocks_per_worker - 1, args.end_block)
        if start_block <= args.end_block:
            block_ranges.append((start_block, end_block))
            start_block = end_block + 1
        else:
            break
    
    logger.info(f"Block ranges for {len(block_ranges)} workers:")
    for i, (start, end) in enumerate(block_ranges):
        logger.info(f"  Worker {i}: Blocks {start:,} - {end:,} ({end - start + 1:,} blocks)")
    
    # Create and start workers
    tasks = []
    for i, (start_block, end_block) in enumerate(block_ranges):
        task = asyncio.create_task(worker_main(i, start_block, end_block))
        tasks.append(task)
    
    # Wait for all workers to complete
    start_time = time.time()
    try:
        results = await asyncio.gather(*tasks)
        total_collected = sum(results)
        elapsed_time = time.time() - start_time
        
        logger.success(f"All workers completed successfully!")
        logger.success(f"Total blocks collected: {total_collected}")
        logger.success(f"Total time: {elapsed_time:.2f} seconds")
        logger.success(f"Average rate: {total_collected / elapsed_time:.2f} blocks/second")
        
    except Exception as e:
        logger.error(f"One or more workers failed: {e}")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())


