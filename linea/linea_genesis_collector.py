#!/usr/bin/env python3
"""
LINEA Genesis to Current Block Collector
Enhanced system to download full LINEA blockchain from genesis using 10 workers
Optimized for QuickNode API with high-performance parallel processing
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
import logging
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple
import signal
import sys
from pathlib import Path
from web3 import Web3
import hashlib
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linea_genesis_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LineaGenesisCollector:
    """Enhanced LINEA collector for full chain download from genesis"""
    
    def __init__(self, config_file: str = "linea_config.env"):
        self.config = self.load_config(config_file)
        self.archive_db_path = self.config.get('LINEA_ARCHIVE_DATABASE_PATH', 'linea_archive_data.db')
        self.rpc_url = self.config.get('LINEA_RPC_URL')
        
        # Enhanced worker configuration
        self.num_workers = 10  # Fixed to 10 workers for optimal performance
        self.batch_size = 100  # Smaller batches for better memory management
        self.max_retries = 5
        self.retry_delay = 2
        
        # QuickNode API optimization (conservative to avoid rate limits)
        self.rate_limit = 50  # Conservative rate limit for QuickNode
        self.concurrent_requests = 10  # Conservative concurrent requests
        
        # Collection settings
        self.start_block = 0  # Always start from genesis
        self.current_block = 0
        self.total_blocks = 0
        
        # Progress tracking
        self.progress_lock = Lock()
        self.is_running = False
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(self.rate_limit)
        
        # Database connections
        self.db_connections = []
        self.db_lock = Lock()
        
        # Web3 connection with connection pooling
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Enhanced statistics
        self.stats = {
            'blocks_collected': 0,
            'transactions_collected': 0,
            'accounts_collected': 0,
            'contracts_collected': 0,
            'tokens_collected': 0,
            'defi_collected': 0,
            'errors': 0,
            'start_time': None,
            'current_block': 0,
            'total_blocks': 0,
            'progress_percentage': 0.0,
            'blocks_per_second': 0.0,
            'estimated_completion': None,
            'worker_stats': {}
        }
        
        # Initialize database connections
        self.init_database_connections()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_file: str) -> Dict[str, str]:
        """Load configuration from environment file"""
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        return config
    
    def init_database_connections(self):
        """Initialize database connections for workers"""
        logger.info("🗄️ Initializing database connections...")
        for i in range(self.num_workers):
            conn = sqlite3.connect(self.archive_db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=20000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            self.db_connections.append(conn)
        logger.info(f"✅ Initialized {len(self.db_connections)} database connections")
    
    def get_database_connection(self, worker_id: int):
        """Get database connection for worker"""
        return self.db_connections[worker_id % len(self.db_connections)]
    
    async def get_current_block_number(self) -> int:
        """Get current block number from QuickNode"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "eth_blockNumber",
                        "params": [],
                        "id": 1
                    }
                    async with session.post(self.rpc_url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return int(data['result'], 16)
                        else:
                            logger.error(f"Failed to get current block: {response.status}")
                            return 0
        except Exception as e:
            logger.error(f"Error getting current block: {e}")
            return 0
    
    async def get_block_data(self, block_number: int, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Get block data from QuickNode API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "eth_getBlockByNumber",
                        "params": [hex(block_number), True],  # True for full transaction data
                        "id": block_number
                    }
                    async with session.post(self.rpc_url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'result' in data and data['result']:
                                return data['result']
                        elif response.status == 429:  # Rate limited
                            wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Rate limited for block {block_number}, waiting {wait_time}s (attempt {attempt + 1})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"Failed to get block {block_number}: {response.status}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            return None
            except Exception as e:
                logger.error(f"Error getting block {block_number} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                return None
        
        return None
    
    async def get_transaction_receipt(self, tx_hash: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Get transaction receipt from QuickNode API"""
        try:
            async with self.rate_limiter:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_getTransactionReceipt",
                    "params": [tx_hash],
                    "id": tx_hash
                }
                async with session.post(self.rpc_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'result' in data and data['result']:
                            return data['result']
                    return None
        except Exception as e:
            logger.error(f"Error getting transaction receipt {tx_hash}: {e}")
            return None
    
    def save_block_data(self, block_data: Dict, worker_id: int):
        """Save block data to database"""
        try:
            conn = self.get_database_connection(worker_id)
            cursor = conn.cursor()
            
            # Extract block information
            block_number = int(block_data['number'], 16)
            block_hash = block_data['hash']
            parent_hash = block_data['parentHash']
            timestamp = datetime.fromtimestamp(int(block_data['timestamp'], 16))
            gas_limit = int(block_data['gasLimit'], 16)
            gas_used = int(block_data['gasUsed'], 16)
            base_fee = int(block_data.get('baseFeePerGas', '0x0'), 16) if block_data.get('baseFeePerGas') else 0
            transaction_count = len(block_data.get('transactions', []))
            size = len(json.dumps(block_data))
            
            # Insert block data
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_blocks (
                    block_number, block_hash, parent_hash, timestamp, gas_limit, gas_used,
                    base_fee_per_gas, transaction_count, size, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_number, block_hash, parent_hash, timestamp, gas_limit, gas_used,
                base_fee, transaction_count, size, datetime.now(), datetime.now()
            ))
            
            # Save transactions
            for tx in block_data.get('transactions', []):
                self.save_transaction_data(tx, block_number, cursor)
            
            conn.commit()
            
            with self.progress_lock:
                self.stats['blocks_collected'] += 1
                self.stats['transactions_collected'] += transaction_count
                self.stats['current_block'] = block_number
                
                if self.total_blocks > 0:
                    self.stats['progress_percentage'] = (block_number / self.total_blocks) * 100
            
        except Exception as e:
            logger.error(f"Error saving block data: {e}")
            with self.progress_lock:
                self.stats['errors'] += 1
    
    def save_transaction_data(self, tx_data: Dict, block_number: int, cursor):
        """Save transaction data to database"""
        try:
            tx_hash = tx_data['hash']
            from_addr = tx_data.get('from', '')
            to_addr = tx_data.get('to', '')
            value = str(int(tx_data.get('value', '0x0'), 16))  # Store as text for large integers
            gas = int(tx_data.get('gas', '0x0'), 16)
            gas_price = str(int(tx_data.get('gasPrice', '0x0'), 16))  # Store as text for large integers
            nonce = int(tx_data.get('nonce', '0x0'), 16)
            input_data = tx_data.get('input', '')
            
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_transactions (
                    transaction_hash, block_number, from_address, to_address, value,
                    gas, gas_price, nonce, input_data, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_hash, block_number, from_addr, to_addr, value,
                gas, gas_price, nonce, input_data, datetime.now(), datetime.now()
            ))
            
        except Exception as e:
            logger.error(f"Error saving transaction data: {e}")
    
    async def worker_task(self, worker_id: int, block_range: Tuple[int, int]):
        """Worker task for processing block range"""
        start_block, end_block = block_range
        logger.info(f"🚀 Worker {worker_id}: Processing blocks {start_block} to {end_block}")
        
        worker_stats = {
            'blocks_processed': 0,
            'transactions_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                for block_num in range(start_block, end_block + 1):
                    if not self.is_running:
                        break
                    
                    # Get block data
                    block_data = await self.get_block_data(block_num, session)
                    if block_data:
                        # Save block data
                        self.save_block_data(block_data, worker_id)
                        worker_stats['blocks_processed'] += 1
                        worker_stats['transactions_processed'] += len(block_data.get('transactions', []))
                        
                        # Log progress every 100 blocks
                        if block_num % 100 == 0:
                            logger.info(f"Worker {worker_id}: Processed block {block_num}")
                    else:
                        worker_stats['errors'] += 1
                        logger.warning(f"Worker {worker_id}: Failed to get block {block_num}")
                    
                    # Longer delay to prevent overwhelming the API
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            worker_stats['errors'] += 1
        
        # Update worker statistics
        with self.progress_lock:
            self.stats['worker_stats'][worker_id] = worker_stats
        
        logger.info(f"✅ Worker {worker_id} completed: {worker_stats['blocks_processed']} blocks, {worker_stats['transactions_processed']} transactions")
    
    def calculate_block_ranges(self) -> List[Tuple[int, int]]:
        """Calculate block ranges for workers"""
        ranges = []
        blocks_per_worker = self.total_blocks // self.num_workers
        remainder = self.total_blocks % self.num_workers
        
        start = 0
        for i in range(self.num_workers):
            end = start + blocks_per_worker - 1
            if i < remainder:
                end += 1
            ranges.append((start, end))
            start = end + 1
        
        return ranges
    
    async def start_collection(self):
        """Start the full chain collection from genesis"""
        logger.info("🚀 Starting LINEA full chain collection from genesis...")
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Get current block number
        self.current_block = await self.get_current_block_number()
        self.total_blocks = self.current_block + 1  # Include block 0 (genesis)
        
        logger.info(f"📊 Total blocks to collect: {self.total_blocks:,}")
        logger.info(f"🎯 Target: Genesis (0) to Block {self.current_block:,}")
        
        # Calculate block ranges for workers
        block_ranges = self.calculate_block_ranges()
        
        logger.info("👥 Worker block ranges:")
        for i, (start, end) in enumerate(block_ranges):
            logger.info(f"   Worker {i+1}: Blocks {start:,} to {end:,} ({end-start+1:,} blocks)")
        
        # Start workers
        tasks = []
        for i, block_range in enumerate(block_ranges):
            task = asyncio.create_task(self.worker_task(i, block_range))
            tasks.append(task)
        
        # Start progress monitoring
        progress_task = asyncio.create_task(self.monitor_progress())
        
        try:
            # Wait for all workers to complete
            await asyncio.gather(*tasks)
            logger.info("✅ All workers completed successfully!")
            
        except Exception as e:
            logger.error(f"Collection error: {e}")
        finally:
            self.is_running = False
            progress_task.cancel()
    
    async def monitor_progress(self):
        """Monitor collection progress"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                with self.progress_lock:
                    elapsed_time = time.time() - self.stats['start_time']
                    if elapsed_time > 0:
                        self.stats['blocks_per_second'] = self.stats['blocks_collected'] / elapsed_time
                        
                        if self.stats['blocks_per_second'] > 0:
                            remaining_blocks = self.total_blocks - self.stats['current_block']
                            eta_seconds = remaining_blocks / self.stats['blocks_per_second']
                            self.stats['estimated_completion'] = datetime.now() + timedelta(seconds=eta_seconds)
                    
                    logger.info(f"📊 Progress: {self.stats['progress_percentage']:.2f}% "
                              f"({self.stats['blocks_collected']:,}/{self.total_blocks:,} blocks) "
                              f"| {self.stats['blocks_per_second']:.2f} blocks/sec "
                              f"| ETA: {self.stats['estimated_completion']}")
                    
                    # Log worker statistics
                    for worker_id, worker_stats in self.stats['worker_stats'].items():
                        logger.info(f"   Worker {worker_id}: {worker_stats['blocks_processed']:,} blocks, "
                                  f"{worker_stats['transactions_processed']:,} transactions, "
                                  f"{worker_stats['errors']} errors")
                
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                await asyncio.sleep(30)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    def cleanup(self):
        """Cleanup database connections"""
        logger.info("🧹 Cleaning up database connections...")
        for conn in self.db_connections:
            conn.close()
        logger.info("✅ Cleanup completed")
    
    async def run(self):
        """Main run function"""
        try:
            await self.start_collection()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()

def main():
    """Main function"""
    collector = LineaGenesisCollector()
    asyncio.run(collector.run())

if __name__ == "__main__":
    main()
