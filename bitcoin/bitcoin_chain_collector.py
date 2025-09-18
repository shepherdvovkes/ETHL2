#!/usr/bin/env python3
"""
Bitcoin Full Chain Collector
Collects complete Bitcoin blockchain data using QuickNode API with 10 workers
"""

import asyncio
import aiohttp
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
import signal
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BlockData:
    """Block data structure"""
    height: int
    hash: str
    previous_hash: str
    timestamp: int
    size: int
    weight: int
    version: int
    nonce: int
    bits: str
    difficulty: float
    merkle_root: str
    tx_count: int
    raw_data: dict

@dataclass
class TransactionData:
    """Transaction data structure"""
    txid: str
    block_height: int
    block_hash: str
    size: int
    weight: int
    fee: int
    input_count: int
    output_count: int
    raw_data: dict

class QuickNodeClient:
    """QuickNode API client for Bitcoin"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'Content-Type': 'application/json'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, params: list = None) -> dict:
        """Make RPC request to QuickNode"""
        if params is None:
            params = []
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with self.session.post(self.endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'error' in result and result['error']:
                        raise Exception(f"RPC Error: {result['error']}")
                    return result.get('result', {})
                else:
                    raise Exception(f"HTTP Error: {response.status}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
        
        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)
    
    async def get_blockchain_info(self) -> dict:
        """Get blockchain information"""
        return await self.make_request("getblockchaininfo")
    
    async def get_block_hash(self, height: int) -> str:
        """Get block hash by height"""
        return await self.make_request("getblockhash", [height])
    
    async def get_block(self, block_hash: str, verbosity: int = 2) -> dict:
        """Get block data by hash"""
        return await self.make_request("getblock", [block_hash, verbosity])
    
    async def get_block_header(self, block_hash: str) -> dict:
        """Get block header by hash"""
        return await self.make_request("getblockheader", [block_hash])
    
    async def get_raw_transaction(self, txid: str, block_hash: str = None) -> dict:
        """Get raw transaction data"""
        params = [txid, True]
        if block_hash:
            params.append(block_hash)
        return await self.make_request("getrawtransaction", params)

class BitcoinDatabase:
    """SQLite database for Bitcoin blockchain data"""
    
    def __init__(self, db_path: str = "bitcoin_chain.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Blocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    height INTEGER PRIMARY KEY,
                    hash TEXT UNIQUE NOT NULL,
                    previous_hash TEXT,
                    timestamp INTEGER,
                    size INTEGER,
                    weight INTEGER,
                    version INTEGER,
                    nonce INTEGER,
                    bits TEXT,
                    difficulty REAL,
                    merkle_root TEXT,
                    tx_count INTEGER,
                    raw_data TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    block_height INTEGER,
                    block_hash TEXT,
                    size INTEGER,
                    weight INTEGER,
                    fee INTEGER,
                    input_count INTEGER,
                    output_count INTEGER,
                    raw_data TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_height) REFERENCES blocks (height)
                )
            """)
            
            # Progress tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_progress (
                    id INTEGER PRIMARY KEY,
                    worker_id INTEGER,
                    start_height INTEGER,
                    end_height INTEGER,
                    status TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    blocks_collected INTEGER DEFAULT 0,
                    transactions_collected INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks (hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_block_height ON transactions (block_height)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_block_hash ON transactions (block_hash)")
            
            conn.commit()
    
    def save_block(self, block_data: BlockData):
        """Save block data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO blocks 
                (height, hash, previous_hash, timestamp, size, weight, version, 
                 nonce, bits, difficulty, merkle_root, tx_count, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_data.height, block_data.hash, block_data.previous_hash,
                block_data.timestamp, block_data.size, block_data.weight,
                block_data.version, block_data.nonce, block_data.bits,
                block_data.difficulty, block_data.merkle_root, block_data.tx_count,
                json.dumps(block_data.raw_data)
            ))
            conn.commit()
    
    def save_transaction(self, tx_data: TransactionData):
        """Save transaction data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transactions 
                (txid, block_height, block_hash, size, weight, fee, 
                 input_count, output_count, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_data.txid, tx_data.block_height, tx_data.block_hash,
                tx_data.size, tx_data.weight, tx_data.fee,
                tx_data.input_count, tx_data.output_count,
                json.dumps(tx_data.raw_data)
            ))
            conn.commit()
    
    def get_collection_progress(self) -> Dict:
        """Get collection progress statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total blocks collected
            cursor.execute("SELECT COUNT(*) FROM blocks")
            blocks_collected = cursor.fetchone()[0]
            
            # Get total transactions collected
            cursor.execute("SELECT COUNT(*) FROM transactions")
            transactions_collected = cursor.fetchone()[0]
            
            # Get latest block height
            cursor.execute("SELECT MAX(height) FROM blocks")
            latest_height = cursor.fetchone()[0] or 0
            
            # Get collection rate
            cursor.execute("""
                SELECT COUNT(*) FROM collection_progress 
                WHERE status = 'completed' AND completed_at > datetime('now', '-1 hour')
            """)
            recent_blocks = cursor.fetchone()[0]
            
            return {
                'blocks_collected': blocks_collected,
                'transactions_collected': transactions_collected,
                'latest_height': latest_height,
                'recent_rate': recent_blocks
            }
    
    def update_worker_progress(self, worker_id: int, start_height: int, 
                             end_height: int, status: str, blocks: int = 0, txs: int = 0):
        """Update worker progress"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if record exists
            cursor.execute("SELECT id FROM collection_progress WHERE worker_id = ?", (worker_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE collection_progress 
                    SET status = ?, 
                        completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END,
                        blocks_collected = ?,
                        transactions_collected = ?
                    WHERE worker_id = ?
                """, (status, status, blocks, txs, worker_id))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO collection_progress 
                    (worker_id, start_height, end_height, status, started_at, 
                     completed_at, blocks_collected, transactions_collected)
                    VALUES (?, ?, ?, ?, 
                            CASE WHEN ? = 'started' THEN CURRENT_TIMESTAMP ELSE NULL END,
                            CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END,
                            ?, ?)
                """, (worker_id, start_height, end_height, status, status, status, blocks, txs))
            
            conn.commit()

class BitcoinWorker:
    """Individual worker for collecting blockchain data"""
    
    def __init__(self, worker_id: int, endpoint: str, db: BitcoinDatabase, 
                 height_range: Tuple[int, int], progress_queue: Queue):
        self.worker_id = worker_id
        self.endpoint = endpoint
        self.db = db
        self.start_height, self.end_height = height_range
        self.progress_queue = progress_queue
        self.blocks_collected = 0
        self.transactions_collected = 0
        self.is_running = False
    
    async def collect_range(self):
        """Collect blockchain data for assigned height range"""
        self.is_running = True
        logger.info(f"Worker {self.worker_id} starting collection from {self.start_height} to {self.end_height}")
        
        # Update progress
        self.db.update_worker_progress(self.worker_id, self.start_height, 
                                     self.end_height, 'started')
        
        try:
            async with QuickNodeClient(self.endpoint) as client:
                for height in range(self.start_height, self.end_height + 1):
                    if not self.is_running:
                        break
                    
                    try:
                        # Get block hash
                        block_hash = await client.get_block_hash(height)
                        
                        # Get block data
                        block_data = await client.get_block(block_hash, 2)
                        
                        # Process block
                        block = self._process_block(block_data, height)
                        self.db.save_block(block)
                        self.blocks_collected += 1
                        
                        # Process transactions
                        if 'tx' in block_data:
                            for tx_data in block_data['tx']:
                                tx = self._process_transaction(tx_data, height, block_hash)
                                self.db.save_transaction(tx)
                                self.transactions_collected += 1
                        
                        # Update progress every 100 blocks
                        if height % 100 == 0:
                            self.progress_queue.put({
                                'worker_id': self.worker_id,
                                'height': height,
                                'blocks': self.blocks_collected,
                                'transactions': self.transactions_collected
                            })
                            logger.info(f"Worker {self.worker_id}: Collected block {height}")
                    
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id} error at height {height}: {e}")
                        continue
                
                # Final progress update
                self.db.update_worker_progress(self.worker_id, self.start_height, 
                                             self.end_height, 'completed', 
                                             self.blocks_collected, self.transactions_collected)
                
                logger.info(f"Worker {self.worker_id} completed: {self.blocks_collected} blocks, {self.transactions_collected} transactions")
        
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed: {e}")
            self.db.update_worker_progress(self.worker_id, self.start_height, 
                                         self.end_height, 'failed')
    
    def _process_block(self, block_data: dict, height: int) -> BlockData:
        """Process raw block data"""
        return BlockData(
            height=height,
            hash=block_data['hash'],
            previous_hash=block_data.get('previousblockhash', ''),
            timestamp=block_data['time'],
            size=block_data['size'],
            weight=block_data['weight'],
            version=block_data['version'],
            nonce=block_data['nonce'],
            bits=block_data['bits'],
            difficulty=block_data['difficulty'],
            merkle_root=block_data['merkleroot'],
            tx_count=len(block_data.get('tx', [])),
            raw_data=block_data
        )
    
    def _process_transaction(self, tx_data: dict, block_height: int, block_hash: str) -> TransactionData:
        """Process raw transaction data"""
        return TransactionData(
            txid=tx_data['txid'],
            block_height=block_height,
            block_hash=block_hash,
            size=tx_data.get('size', 0),
            weight=tx_data.get('weight', 0),
            fee=tx_data.get('fee', 0),
            input_count=len(tx_data.get('vin', [])),
            output_count=len(tx_data.get('vout', [])),
            raw_data=tx_data
        )
    
    def stop(self):
        """Stop the worker"""
        self.is_running = False

class BitcoinChainCollector:
    """Main Bitcoin blockchain collector with 10 workers"""
    
    def __init__(self, endpoint: str, num_workers: int = 10):
        self.endpoint = endpoint
        self.num_workers = num_workers
        self.db = BitcoinDatabase()
        self.workers = []
        self.progress_queue = Queue()
        self.is_running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    async def start_collection(self, start_height: int = 0, end_height: int = None):
        """Start blockchain collection with multiple workers"""
        logger.info(f"Starting Bitcoin chain collection with {self.num_workers} workers")
        
        # Get current blockchain height if not specified
        if end_height is None:
            async with QuickNodeClient(self.endpoint) as client:
                blockchain_info = await client.get_blockchain_info()
                end_height = blockchain_info['blocks']
                logger.info(f"Current blockchain height: {end_height}")
        
        # Calculate height ranges for each worker
        total_blocks = end_height - start_height + 1
        blocks_per_worker = total_blocks // self.num_workers
        remainder = total_blocks % self.num_workers
        
        height_ranges = []
        current_height = start_height
        
        for i in range(self.num_workers):
            worker_blocks = blocks_per_worker
            if i < remainder:  # Distribute remainder blocks
                worker_blocks += 1
            
            end_worker_height = current_height + worker_blocks - 1
            height_ranges.append((current_height, end_worker_height))
            current_height = end_worker_height + 1
        
        logger.info(f"Height ranges: {height_ranges}")
        
        # Start workers
        self.is_running = True
        tasks = []
        
        for i, (start, end) in enumerate(height_ranges):
            worker = BitcoinWorker(i, self.endpoint, self.db, (start, end), self.progress_queue)
            self.workers.append(worker)
            task = asyncio.create_task(worker.collect_range())
            tasks.append(task)
        
        # Start progress monitoring
        monitor_task = asyncio.create_task(self._monitor_progress())
        
        try:
            # Wait for all workers to complete
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Collection failed: {e}")
        finally:
            self.is_running = False
            monitor_task.cancel()
            
            # Final statistics
            progress = self.db.get_collection_progress()
            logger.info(f"Collection completed: {progress}")
    
    async def _monitor_progress(self):
        """Monitor collection progress"""
        while self.is_running:
            try:
                # Get progress updates from queue
                while not self.progress_queue.empty():
                    progress = self.progress_queue.get_nowait()
                    logger.info(f"Progress - Worker {progress['worker_id']}: "
                              f"Height {progress['height']}, "
                              f"Blocks: {progress['blocks']}, "
                              f"Txs: {progress['transactions']}")
                
                # Get overall progress
                progress = self.db.get_collection_progress()
                logger.info(f"Overall Progress: {progress['blocks_collected']} blocks, "
                          f"{progress['transactions_collected']} transactions, "
                          f"Latest: {progress['latest_height']}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                await asyncio.sleep(30)
    
    def stop(self):
        """Stop all workers"""
        self.is_running = False
        for worker in self.workers:
            worker.stop()
        logger.info("All workers stopped")

async def main():
    """Main function"""
    # QuickNode endpoint
    endpoint = "https://orbital-twilight-mansion.btc.quiknode.pro/a1280f4e959966b62d579978248263e3975e3b4d/"
    
    # Create collector with 10 workers
    collector = BitcoinChainCollector(endpoint, num_workers=10)
    
    try:
        # Start collection from genesis block
        await collector.start_collection(start_height=0)
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
    finally:
        collector.stop()

if __name__ == "__main__":
    asyncio.run(main())
