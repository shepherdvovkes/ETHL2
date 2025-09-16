#!/usr/bin/env python3
"""
Astar Multi-Threaded Data Collector
==================================

High-performance multi-threaded data collection pipeline for Astar (ASTR) network.
Uses concurrent processing to collect historical data much faster.

Features:
- Multi-threaded block collection
- Concurrent transaction processing
- Rate limiting and error handling
- Progress tracking and resumption
- Database optimization
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import multiprocessing as mp
warnings.filterwarnings('ignore')

class AstarMultiThreadedCollector:
    """Multi-threaded data collector for Astar network"""
    
    def __init__(self, rpc_url: str = None, max_workers: int = 10):
        # Astar RPC endpoints
        self.rpc_url = rpc_url or "https://rpc.astar.network"
        self.ws_url = "wss://rpc.astar.network"
        self.session = None
        
        # Multi-threading settings
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.rate_limit_delay = 0.05  # 50ms between requests per worker
        
        # Astar network details
        self.network_info = {
            "name": "Astar",
            "symbol": "ASTR",
            "chain_id": 592,
            "parachain_id": 2006,
            "category": "smart_contracts",
            "description": "Multi-VM smart contract platform supporting EVM and WASM"
        }
        
        # Database setup
        self.db_path = "astar_multithreaded_data.db"
        self.setup_database()
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.collected_blocks = 0
        self.total_blocks = 0
        self.start_time = None
        
        # Data collection settings
        self.batch_size = 50  # Process blocks in batches
        self.transaction_batch_size = 20  # Process transactions in batches
    
    def setup_database(self):
        """Setup SQLite database for Astar data with optimizations"""
        logger.info("Setting up optimized Astar database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Create tables with optimized indexes
        tables = {
            'astar_blocks': '''
                CREATE TABLE IF NOT EXISTS astar_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_number INTEGER UNIQUE NOT NULL,
                    block_hash TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    parent_hash TEXT,
                    gas_limit INTEGER,
                    gas_used INTEGER,
                    transaction_count INTEGER,
                    block_size INTEGER,
                    difficulty TEXT,
                    total_difficulty TEXT,
                    extra_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_transactions': '''
                CREATE TABLE IF NOT EXISTS astar_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_hash TEXT UNIQUE NOT NULL,
                    block_number INTEGER NOT NULL,
                    block_hash TEXT,
                    from_address TEXT,
                    to_address TEXT,
                    value TEXT,
                    gas INTEGER,
                    gas_price TEXT,
                    nonce INTEGER,
                    input_data TEXT,
                    transaction_index INTEGER,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_number) REFERENCES astar_blocks (block_number)
                )
            ''',
            'astar_metrics': '''
                CREATE TABLE IF NOT EXISTS astar_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    block_number INTEGER,
                    network_hash_rate TEXT,
                    network_difficulty TEXT,
                    active_addresses_24h INTEGER,
                    transaction_count_24h INTEGER,
                    gas_price_avg TEXT,
                    gas_used_avg INTEGER,
                    block_time_avg REAL,
                    network_utilization REAL,
                    total_supply TEXT,
                    circulating_supply TEXT,
                    market_cap_usd REAL,
                    price_usd REAL,
                    volume_24h_usd REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_blocks_number ON astar_blocks(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON astar_blocks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tx_hash ON astar_transactions(transaction_hash)",
            "CREATE INDEX IF NOT EXISTS idx_tx_block ON astar_transactions(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON astar_metrics(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Optimized database setup completed")
    
    async def __aenter__(self):
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make RPC call to Astar endpoint with rate limiting"""
        if params is None:
            params = []
        
        async with self.semaphore:  # Limit concurrent requests
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1
            }
            
            try:
                async with self.session.post(
                    self.rpc_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('result', {})
                    else:
                        logger.error(f"RPC call failed: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Error making RPC call: {e}")
                return {}
    
    async def get_current_block_number(self) -> int:
        """Get current block number"""
        result = await self.make_rpc_call('eth_blockNumber')
        if result:
            return int(result, 16)
        return 0
    
    async def get_block_data(self, block_number: int) -> Dict:
        """Get comprehensive block data"""
        block_hex = hex(block_number)
        block_data = await self.make_rpc_call('eth_getBlockByNumber', [block_hex, True])
        
        if not block_data:
            return {}
        
        # Extract block information
        block_info = {
            'block_number': block_number,
            'block_hash': block_data.get('hash', ''),
            'timestamp': datetime.fromtimestamp(int(block_data.get('timestamp', '0x0'), 16)),
            'parent_hash': block_data.get('parentHash', ''),
            'gas_limit': int(block_data.get('gasLimit', '0x0'), 16),
            'gas_used': int(block_data.get('gasUsed', '0x0'), 16),
            'transaction_count': len(block_data.get('transactions', [])),
            'block_size': len(str(block_data)),
            'difficulty': block_data.get('difficulty', '0x0'),
            'total_difficulty': block_data.get('totalDifficulty', '0x0'),
            'extra_data': block_data.get('extraData', ''),
            'transactions': block_data.get('transactions', [])
        }
        
        return block_info
    
    async def get_transaction_data(self, tx_hash: str) -> Dict:
        """Get detailed transaction data"""
        tx_data = await self.make_rpc_call('eth_getTransactionByHash', [tx_hash])
        if not tx_data:
            return {}
        
        # Get transaction receipt
        receipt = await self.make_rpc_call('eth_getTransactionReceipt', [tx_hash])
        
        tx_info = {
            'transaction_hash': tx_hash,
            'block_number': int(tx_data.get('blockNumber', '0x0'), 16),
            'block_hash': tx_data.get('blockHash', ''),
            'from_address': tx_data.get('from', ''),
            'to_address': tx_data.get('to', ''),
            'value': tx_data.get('value', '0x0'),
            'gas': int(tx_data.get('gas', '0x0'), 16),
            'gas_price': tx_data.get('gasPrice', '0x0'),
            'nonce': int(tx_data.get('nonce', '0x0'), 16),
            'input_data': tx_data.get('input', ''),
            'transaction_index': int(tx_data.get('transactionIndex', '0x0'), 16),
            'status': receipt.get('status', '0x0') if receipt else '0x0'
        }
        
        return tx_info
    
    def save_block_data_batch(self, block_data_list: List[Dict]):
        """Save multiple block data entries in batch"""
        if not block_data_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare batch insert
            block_values = []
            for block_data in block_data_list:
                block_values.append((
                    block_data['block_number'],
                    block_data['block_hash'],
                    block_data['timestamp'],
                    block_data['parent_hash'],
                    block_data['gas_limit'],
                    block_data['gas_used'],
                    block_data['transaction_count'],
                    block_data['block_size'],
                    block_data['difficulty'],
                    block_data['total_difficulty'],
                    block_data['extra_data']
                ))
            
            # Batch insert
            cursor.executemany('''
                INSERT OR REPLACE INTO astar_blocks 
                (block_number, block_hash, timestamp, parent_hash, gas_limit, gas_used, 
                 transaction_count, block_size, difficulty, total_difficulty, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', block_values)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving block data batch: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_transaction_data_batch(self, tx_data_list: List[Dict]):
        """Save multiple transaction data entries in batch"""
        if not tx_data_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare batch insert
            tx_values = []
            for tx_data in tx_data_list:
                tx_values.append((
                    tx_data['transaction_hash'],
                    tx_data['block_number'],
                    tx_data['block_hash'],
                    tx_data['from_address'],
                    tx_data['to_address'],
                    tx_data['value'],
                    tx_data['gas'],
                    tx_data['gas_price'],
                    tx_data['nonce'],
                    tx_data['input_data'],
                    tx_data['transaction_index'],
                    tx_data['status']
                ))
            
            # Batch insert
            cursor.executemany('''
                INSERT OR REPLACE INTO astar_transactions 
                (transaction_hash, block_number, block_hash, from_address, to_address, 
                 value, gas, gas_price, nonce, input_data, transaction_index, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tx_values)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving transaction data batch: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def collect_block_batch(self, block_numbers: List[int]) -> List[Dict]:
        """Collect data for a batch of blocks concurrently"""
        tasks = []
        for block_num in block_numbers:
            task = self.get_block_data(block_num)
            tasks.append(task)
        
        # Execute all block requests concurrently
        block_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty results
        valid_blocks = []
        for i, result in enumerate(block_results):
            if isinstance(result, dict) and result:
                valid_blocks.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting block {block_numbers[i]}: {result}")
        
        return valid_blocks
    
    async def collect_transaction_batch(self, tx_hashes: List[str]) -> List[Dict]:
        """Collect data for a batch of transactions concurrently"""
        tasks = []
        for tx_hash in tx_hashes:
            task = self.get_transaction_data(tx_hash)
            tasks.append(task)
        
        # Execute all transaction requests concurrently
        tx_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty results
        valid_transactions = []
        for i, result in enumerate(tx_results):
            if isinstance(result, dict) and result:
                valid_transactions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting transaction {tx_hashes[i]}: {result}")
        
        return valid_transactions
    
    def update_progress(self, blocks_collected: int):
        """Update progress with thread safety"""
        with self.progress_lock:
            self.collected_blocks += blocks_collected
            elapsed_time = time.time() - self.start_time
            rate = self.collected_blocks / elapsed_time if elapsed_time > 0 else 0
            remaining_blocks = self.total_blocks - self.collected_blocks
            eta = remaining_blocks / rate if rate > 0 else 0
            
            progress_percent = (self.collected_blocks / self.total_blocks) * 100
            
            logger.info(f"Progress: {self.collected_blocks}/{self.total_blocks} blocks "
                       f"({progress_percent:.1f}%) - Rate: {rate:.1f} blocks/sec - ETA: {eta/60:.1f} min")
    
    async def collect_historical_blocks_multithreaded(self, start_block: int, end_block: int):
        """Collect historical block data using multi-threading"""
        logger.info(f"🚀 Starting multi-threaded collection from block {start_block} to {end_block}")
        
        self.total_blocks = end_block - start_block + 1
        self.collected_blocks = 0
        self.start_time = time.time()
        
        # Create block number ranges for batch processing
        block_ranges = []
        for i in range(start_block, end_block + 1, self.batch_size):
            end_range = min(i + self.batch_size - 1, end_block)
            block_ranges.append(list(range(i, end_range + 1)))
        
        logger.info(f"Processing {len(block_ranges)} batches of {self.batch_size} blocks each")
        
        # Process batches concurrently
        for i, block_batch in enumerate(block_ranges):
            try:
                # Collect block data for this batch
                blocks_data = await self.collect_block_batch(block_batch)
                
                if blocks_data:
                    # Save block data
                    self.save_block_data_batch(blocks_data)
                    
                    # Collect transactions for these blocks
                    all_transactions = []
                    for block_data in blocks_data:
                        transactions = block_data.get('transactions', [])
                        for tx in transactions:
                            if isinstance(tx, dict) and 'hash' in tx:
                                all_transactions.append(tx['hash'])
                    
                    # Process transactions in batches
                    if all_transactions:
                        for j in range(0, len(all_transactions), self.transaction_batch_size):
                            tx_batch = all_transactions[j:j + self.transaction_batch_size]
                            tx_data = await self.collect_transaction_batch(tx_batch)
                            if tx_data:
                                self.save_transaction_data_batch(tx_data)
                    
                    # Update progress
                    self.update_progress(len(blocks_data))
                
                # Log batch completion
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(block_ranges)} batches")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
        
        logger.success(f"🎉 Multi-threaded collection completed! Collected {self.collected_blocks} blocks")
    
    async def collect_comprehensive_data_multithreaded(self, days_back: int = 7):
        """Collect comprehensive Astar data using multi-threading"""
        logger.info(f"🚀 Starting multi-threaded Astar data collection for {days_back} days")
        
        try:
            # Get current block
            current_block = await self.get_current_block_number()
            logger.info(f"Current Astar block: {current_block}")
            
            # Calculate start block
            blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
            start_block = max(1, current_block - (days_back * blocks_per_day))
            
            logger.info(f"Collecting data from block {start_block} to {current_block}")
            logger.info(f"Total blocks to collect: {current_block - start_block + 1}")
            logger.info(f"Using {self.max_workers} concurrent workers")
            
            # Collect historical blocks with multi-threading
            await self.collect_historical_blocks_multithreaded(start_block, current_block)
            
            # Generate summary
            await self.generate_data_summary()
            
            logger.success("🎉 Multi-threaded Astar data collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in multi-threaded data collection: {e}")
            raise
    
    async def generate_data_summary(self):
        """Generate summary of collected data"""
        logger.info("Generating data summary...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM astar_blocks")
            block_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_transactions")
            tx_count = cursor.fetchone()[0]
            
            # Get block range
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
            block_range = cursor.fetchone()
            
            # Get recent activity
            cursor.execute("SELECT AVG(transaction_count) FROM astar_blocks ORDER BY block_number DESC LIMIT 100")
            avg_tx_per_block = cursor.fetchone()[0] or 0
            
            summary = {
                "collection_timestamp": datetime.utcnow().isoformat(),
                "network": self.network_info,
                "collection_method": "multi_threaded",
                "max_workers": self.max_workers,
                "data_summary": {
                    "blocks_collected": block_count,
                    "transactions_collected": tx_count,
                    "block_range": {
                        "start": block_range[0],
                        "end": block_range[1]
                    },
                    "average_transactions_per_block": round(avg_tx_per_block, 2)
                }
            }
            
            # Save summary
            with open('astar_multithreaded_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Multi-threaded data summary generated:")
            logger.info(f"  Blocks: {block_count}")
            logger.info(f"  Transactions: {tx_count}")
            logger.info(f"  Block range: {block_range[0]} - {block_range[1]}")
            logger.info(f"  Avg TX per block: {avg_tx_per_block:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        finally:
            conn.close()

async def main():
    """Main function"""
    # Initialize multi-threaded collector with 15 workers
    async with AstarMultiThreadedCollector(max_workers=15) as collector:
        # Collect comprehensive data for last 7 days
        await collector.collect_comprehensive_data_multithreaded(days_back=7)

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run the multi-threaded collector
    asyncio.run(main())
