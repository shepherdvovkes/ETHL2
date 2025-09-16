#!/usr/bin/env python3
"""
LINEA Archive Data Collector
Collects complete LINEA blockchain history from genesis to current block
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linea_archive_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LineaArchiveCollector:
    """LINEA archive data collector for complete chain history"""
    
    def __init__(self, config_file: str = "linea_config.env"):
        self.config = self.load_config(config_file)
        self.archive_db_path = self.config.get('LINEA_ARCHIVE_DATABASE_PATH', 'linea_archive_data.db')
        self.rpc_url = self.config.get('LINEA_RPC_URL')
        
        # Worker configuration
        self.num_workers = int(self.config.get('ARCHIVE_CONCURRENT_WORKERS', 10))
        self.batch_size = int(self.config.get('ARCHIVE_BATCH_SIZE', 1000))
        self.max_retries = int(self.config.get('ARCHIVE_MAX_RETRIES', 3))
        self.retry_delay = int(self.config.get('ARCHIVE_RETRY_DELAY', 5))
        
        # Collection settings
        self.start_block = int(self.config.get('ARCHIVE_START_BLOCK', 0))
        self.archive_mode = self.config.get('ARCHIVE_MODE', 'full')
        
        # Progress tracking
        self.progress_lock = Lock()
        self.is_running = False
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(int(self.config.get('LINEA_RPC_RATE_LIMIT', 100)))
        
        # Database connections
        self.db_connections = []
        self.db_lock = Lock()
        
        # Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Statistics
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
            'progress_percentage': 0.0
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
        """Initialize database connection pool"""
        for _ in range(self.num_workers):
            conn = sqlite3.connect(self.archive_db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=20000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            self.db_connections.append(conn)
    
    def get_db_connection(self):
        """Get a database connection from the pool"""
        with self.db_lock:
            if self.db_connections:
                return self.db_connections.pop()
            else:
                conn = sqlite3.connect(self.archive_db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=20000")
                conn.execute("PRAGMA temp_store=MEMORY")
                return conn
    
    def return_db_connection(self, conn):
        """Return a database connection to the pool"""
        with self.db_lock:
            if len(self.db_connections) < self.num_workers:
                self.db_connections.append(conn)
            else:
                conn.close()
    
    async def make_rpc_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make RPC request with rate limiting and retry logic"""
        for attempt in range(self.max_retries):
            async with self.rate_limiter:
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or [],
                    "id": 1
                }
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(self.rpc_url, json=payload) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'result' in data:
                                    return data['result']
                                else:
                                    logger.error(f"RPC error: {data.get('error', 'Unknown error')}")
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                                        continue
                                    return None
                            else:
                                logger.error(f"HTTP error: {response.status}")
                                if attempt < self.max_retries - 1:
                                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                                    continue
                                return None
                    except Exception as e:
                        logger.error(f"RPC request failed (attempt {attempt + 1}): {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        return None
        
        return None
    
    async def get_latest_block_number(self) -> int:
        """Get the latest block number"""
        result = await self.make_rpc_request("eth_blockNumber")
        if result:
            return int(result, 16)
        return 0
    
    async def get_block_by_number(self, block_number: int, full_transactions: bool = True) -> Dict[str, Any]:
        """Get block by number"""
        hex_block = hex(block_number)
        result = await self.make_rpc_request("eth_getBlockByNumber", [hex_block, full_transactions])
        return result
    
    async def get_transaction_by_hash(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction by hash"""
        result = await self.make_rpc_request("eth_getTransactionByHash", [tx_hash])
        return result
    
    async def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction receipt"""
        result = await self.make_rpc_request("eth_getTransactionReceipt", [tx_hash])
        return result
    
    async def get_account_balance(self, address: str, block_number: str) -> str:
        """Get account balance at specific block"""
        result = await self.make_rpc_request("eth_getBalance", [address, block_number])
        return result or "0x0"
    
    async def get_account_nonce(self, address: str, block_number: str) -> int:
        """Get account nonce at specific block"""
        result = await self.make_rpc_request("eth_getTransactionCount", [address, block_number])
        if result:
            return int(result, 16)
        return 0
    
    async def get_account_code(self, address: str, block_number: str) -> str:
        """Get account code at specific block"""
        result = await self.make_rpc_request("eth_getCode", [address, block_number])
        return result or "0x"
    
    def store_block_archive(self, block_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store block data in archive database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_blocks 
                (block_number, block_hash, parent_hash, timestamp, gas_limit, gas_used, 
                 base_fee_per_gas, difficulty, total_difficulty, size, transaction_count, 
                 extra_data, mix_hash, nonce, receipts_root, sha3_uncles, state_root, 
                 transactions_root, withdrawals_root, withdrawals, blob_gas_used, 
                 excess_blob_gas, parent_beacon_block_root)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(block_data.get('number', 0), 16),
                block_data.get('hash', ''),
                block_data.get('parentHash', ''),
                datetime.fromtimestamp(int(block_data.get('timestamp', 0), 16)),
                int(block_data.get('gasLimit', 0), 16),
                int(block_data.get('gasUsed', 0), 16),
                int(block_data.get('baseFeePerGas', 0), 16) if block_data.get('baseFeePerGas') else 0,
                int(block_data.get('difficulty', 0), 16),
                block_data.get('totalDifficulty', '0x0'),
                int(block_data.get('size', 0), 16),
                len(block_data.get('transactions', [])),
                block_data.get('extraData', ''),
                block_data.get('mixHash', ''),
                block_data.get('nonce', ''),
                block_data.get('receiptsRoot', ''),
                block_data.get('sha3Uncles', ''),
                block_data.get('stateRoot', ''),
                block_data.get('transactionsRoot', ''),
                block_data.get('withdrawalsRoot', ''),
                json.dumps(block_data.get('withdrawals', [])),
                int(block_data.get('blobGasUsed', 0), 16) if block_data.get('blobGasUsed') else 0,
                int(block_data.get('excessBlobGas', 0), 16) if block_data.get('excessBlobGas') else 0,
                block_data.get('parentBeaconBlockRoot', '')
            ))
            conn.commit()
            self.stats['blocks_collected'] += 1
        except Exception as e:
            logger.error(f"Error storing archive block: {e}")
            conn.rollback()
    
    def store_transaction_archive(self, tx_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store transaction data in archive database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_transactions 
                (transaction_hash, block_number, block_hash, transaction_index, from_address, 
                 to_address, value, gas, gas_price, max_fee_per_gas, max_priority_fee_per_gas, 
                 nonce, input_data, v, r, s, type, access_list, chain_id, blob_versioned_hashes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_data.get('hash', ''),
                int(tx_data.get('blockNumber', 0), 16),
                tx_data.get('blockHash', ''),
                int(tx_data.get('transactionIndex', 0), 16),
                tx_data.get('from', ''),
                tx_data.get('to', ''),
                tx_data.get('value', '0x0'),
                int(tx_data.get('gas', 0), 16),
                int(tx_data.get('gasPrice', 0), 16),
                int(tx_data.get('maxFeePerGas', 0), 16) if tx_data.get('maxFeePerGas') else 0,
                int(tx_data.get('maxPriorityFeePerGas', 0), 16) if tx_data.get('maxPriorityFeePerGas') else 0,
                int(tx_data.get('nonce', 0), 16),
                tx_data.get('input', ''),
                tx_data.get('v', ''),
                tx_data.get('r', ''),
                tx_data.get('s', ''),
                int(tx_data.get('type', 2), 16),
                json.dumps(tx_data.get('accessList', [])),
                int(tx_data.get('chainId', 59144), 16),
                json.dumps(tx_data.get('blobVersionedHashes', []))
            ))
            conn.commit()
            self.stats['transactions_collected'] += 1
        except Exception as e:
            logger.error(f"Error storing archive transaction: {e}")
            conn.rollback()
    
    def store_transaction_receipt_archive(self, receipt_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store transaction receipt data in archive database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_transaction_receipts 
                (transaction_hash, block_number, block_hash, transaction_index, from_address, 
                 to_address, cumulative_gas_used, effective_gas_price, gas_used, 
                 contract_address, logs, logs_bloom, status, type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                receipt_data.get('transactionHash', ''),
                int(receipt_data.get('blockNumber', 0), 16),
                receipt_data.get('blockHash', ''),
                int(receipt_data.get('transactionIndex', 0), 16),
                receipt_data.get('from', ''),
                receipt_data.get('to', ''),
                int(receipt_data.get('cumulativeGasUsed', 0), 16),
                int(receipt_data.get('effectiveGasPrice', 0), 16),
                int(receipt_data.get('gasUsed', 0), 16),
                receipt_data.get('contractAddress', ''),
                json.dumps(receipt_data.get('logs', [])),
                receipt_data.get('logsBloom', ''),
                int(receipt_data.get('status', 1), 16),
                int(receipt_data.get('type', 2), 16)
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing archive transaction receipt: {e}")
            conn.rollback()
    
    def store_account_archive(self, address: str, account_data: Dict[str, Any], block_number: int, conn: sqlite3.Connection):
        """Store account data in archive database"""
        try:
            cursor = conn.cursor()
            
            # Check if it's a contract
            is_contract = len(account_data.get('code', '0x')) > 2
            
            cursor.execute("""
                INSERT OR REPLACE INTO linea_archive_accounts 
                (address, balance, nonce, code, storage_root, is_contract, 
                 first_seen_block, last_seen_block, transaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                address,
                account_data.get('balance', '0x0'),
                account_data.get('nonce', 0),
                account_data.get('code', '0x'),
                account_data.get('storageRoot', ''),
                is_contract,
                account_data.get('firstSeenBlock', block_number),
                account_data.get('lastSeenBlock', block_number),
                account_data.get('transactionCount', 0)
            ))
            
            # If it's a contract, store in contracts table
            if is_contract:
                cursor.execute("""
                    INSERT OR REPLACE INTO linea_archive_contracts 
                    (address, creator_address, creation_block, bytecode, is_verified)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    address,
                    account_data.get('creatorAddress', ''),
                    account_data.get('creationBlock', block_number),
                    account_data.get('code', '0x'),
                    False
                ))
                self.stats['contracts_collected'] += 1
            
            conn.commit()
            self.stats['accounts_collected'] += 1
        except Exception as e:
            logger.error(f"Error storing archive account: {e}")
            conn.rollback()
    
    def update_progress(self, collection_type: str, start_block: int, end_block: int, 
                       current_block: int, status: str, error_message: str = None):
        """Update collection progress"""
        with self.progress_lock:
            conn = self.get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO linea_archive_progress 
                    (collection_type, start_block, end_block, current_block, total_blocks, 
                     completed_blocks, status, progress_percentage, last_updated_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    collection_type,
                    start_block,
                    end_block,
                    current_block,
                    end_block - start_block + 1,
                    current_block - start_block + 1,
                    status,
                    ((current_block - start_block + 1) / (end_block - start_block + 1)) * 100,
                    datetime.now(),
                    error_message
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Error updating progress: {e}")
                conn.rollback()
            finally:
                self.return_db_connection(conn)
    
    async def collect_block_range(self, start_block: int, end_block: int, worker_id: int):
        """Collect data for a range of blocks"""
        logger.info(f"Worker {worker_id}: Collecting blocks {start_block} to {end_block}")
        
        conn = self.get_db_connection()
        try:
            current_block = start_block
            
            while current_block <= end_block and self.is_running:
                try:
                    # Get block data
                    block_data = await self.get_block_by_number(current_block, True)
                    if not block_data:
                        logger.error(f"Worker {worker_id}: Failed to get block {current_block}")
                        self.stats['errors'] += 1
                        current_block += 1
                        continue
                    
                    # Store block
                    self.store_block_archive(block_data, conn)
                    
                    # Process transactions
                    for tx in block_data.get('transactions', []):
                        if isinstance(tx, dict):
                            # Store transaction
                            self.store_transaction_archive(tx, conn)
                            
                            # Get and store transaction receipt
                            tx_receipt = await self.get_transaction_receipt(tx.get('hash', ''))
                            if tx_receipt:
                                self.store_transaction_receipt_archive(tx_receipt, conn)
                            
                            # Collect account data for from/to addresses
                            for address in [tx.get('from', ''), tx.get('to', '')]:
                                if address and address != '0x0000000000000000000000000000000000000000':
                                    try:
                                        balance = await self.get_account_balance(address, hex(current_block))
                                        nonce = await self.get_account_nonce(address, hex(current_block))
                                        code = await self.get_account_code(address, hex(current_block))
                                        
                                        account_data = {
                                            'balance': balance,
                                            'nonce': nonce,
                                            'code': code,
                                            'firstSeenBlock': current_block,
                                            'lastSeenBlock': current_block,
                                            'transactionCount': 0
                                        }
                                        
                                        self.store_account_archive(address, account_data, current_block, conn)
                                    except Exception as e:
                                        logger.error(f"Worker {worker_id}: Error collecting account {address}: {e}")
                    
                    # Update progress
                    self.stats['current_block'] = current_block
                    self.update_progress(f"worker_{worker_id}", start_block, end_block, 
                                       current_block, "running")
                    
                    current_block += 1
                    
                    # Small delay to avoid overwhelming the RPC
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing block {current_block}: {e}")
                    self.stats['errors'] += 1
                    current_block += 1
                    await asyncio.sleep(1)
            
            # Mark as completed
            self.update_progress(f"worker_{worker_id}", start_block, end_block, 
                               current_block - 1, "completed")
            
            logger.info(f"Worker {worker_id}: Completed blocks {start_block} to {end_block}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id}: Fatal error: {e}")
            self.update_progress(f"worker_{worker_id}", start_block, end_block, 
                               current_block, "failed", str(e))
        finally:
            self.return_db_connection(conn)
    
    def calculate_block_ranges(self, start_block: int, end_block: int) -> List[Tuple[int, int]]:
        """Calculate block ranges for workers"""
        total_blocks = end_block - start_block + 1
        blocks_per_worker = total_blocks // self.num_workers
        remainder = total_blocks % self.num_workers
        
        ranges = []
        current_start = start_block
        
        for i in range(self.num_workers):
            current_end = current_start + blocks_per_worker - 1
            
            # Add remainder blocks to the last worker
            if i == self.num_workers - 1:
                current_end += remainder
            
            ranges.append((current_start, current_end))
            current_start = current_end + 1
        
        return ranges
    
    async def collect_archive_data(self):
        """Main archive collection function"""
        logger.info("🚀 Starting LINEA archive data collection...")
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        try:
            # Get latest block number
            latest_block = await self.get_latest_block_number()
            if latest_block == 0:
                logger.error("Failed to get latest block number")
                return
            
            self.stats['total_blocks'] = latest_block - self.start_block + 1
            logger.info(f"Total blocks to collect: {self.stats['total_blocks']}")
            
            # Calculate block ranges for workers
            block_ranges = self.calculate_block_ranges(self.start_block, latest_block)
            
            # Create worker tasks
            tasks = []
            for i, (start, end) in enumerate(block_ranges):
                task = asyncio.create_task(
                    self.collect_block_range(start, end, i + 1)
                )
                tasks.append(task)
            
            # Progress monitoring task
            progress_task = asyncio.create_task(self.monitor_progress())
            tasks.append(progress_task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Archive collection error: {e}")
        finally:
            self.stop_collection()
    
    async def monitor_progress(self):
        """Monitor and log collection progress"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self.stats['total_blocks'] > 0:
                    progress = (self.stats['current_block'] - self.start_block + 1) / self.stats['total_blocks'] * 100
                    self.stats['progress_percentage'] = progress
                    
                    logger.info(f"""
📊 Archive Collection Progress:
   Current Block: {self.stats['current_block']:,}
   Total Blocks: {self.stats['total_blocks']:,}
   Progress: {progress:.2f}%
   Blocks Collected: {self.stats['blocks_collected']:,}
   Transactions Collected: {self.stats['transactions_collected']:,}
   Accounts Collected: {self.stats['accounts_collected']:,}
   Contracts Collected: {self.stats['contracts_collected']:,}
   Errors: {self.stats['errors']:,}
                    """)
                    
                    # Check if all workers are done
                    if self.stats['current_block'] >= self.stats['total_blocks'] + self.start_block - 1:
                        logger.info("✅ Archive collection completed!")
                        self.is_running = False
                        break
                        
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                await asyncio.sleep(30)
    
    def print_final_stats(self):
        """Print final collection statistics"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            logger.info(f"""
🎉 LINEA Archive Collection Completed!

📊 Final Statistics:
   Runtime: {runtime}
   Total Blocks: {self.stats['total_blocks']:,}
   Blocks Collected: {self.stats['blocks_collected']:,}
   Transactions Collected: {self.stats['transactions_collected']:,}
   Accounts Collected: {self.stats['accounts_collected']:,}
   Contracts Collected: {self.stats['contracts_collected']:,}
   Errors: {self.stats['errors']:,}
   Workers Used: {self.num_workers}
   Batch Size: {self.batch_size}
   
📁 Database: {self.archive_db_path}
            """)
    
    def stop_collection(self):
        """Stop archive collection"""
        logger.info("🛑 Stopping LINEA archive collection...")
        self.is_running = False
        
        # Close database connections
        with self.db_lock:
            for conn in self.db_connections:
                conn.close()
            self.db_connections.clear()
        
        self.print_final_stats()
        logger.info("✅ LINEA archive collection stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_collection()
        sys.exit(0)

async def main():
    """Main function"""
    collector = LineaArchiveCollector()
    
    try:
        await collector.collect_archive_data()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(main())
