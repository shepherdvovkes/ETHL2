#!/usr/bin/env python3
"""
Astar Resume Data Collector
==========================

Resume Astar data collection from the last collected block.
Continues the multi-threaded collection process.
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

class AstarResumeCollector:
    """Resume Astar data collection from existing progress"""
    
    def __init__(self, rpc_url: str = None, max_workers: int = 20):
        # Astar RPC endpoints (multiple fallbacks)
        self.rpc_endpoints = [
            "https://rpc.astar.network",
            "https://astar.api.onfinality.io/public",
            "https://astar-rpc.dwellir.com",
            "https://evm.astar.network"
        ]
        self.rpc_url = rpc_url or self.rpc_endpoints[0]
        self.current_endpoint_index = 0
        self.ws_url = "wss://rpc.astar.network"
        self.session = None
        
        # Multi-threading settings
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.rate_limit_delay = 0.1  # 100ms between requests per worker (increased for stability)
        
        # Database setup
        self.db_path = "astar_multithreaded_data.db"
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.collected_blocks = 0
        self.total_blocks = 0
        self.start_time = None
        
        # Data collection settings
        self.batch_size = 50  # Process blocks in batches
        self.transaction_batch_size = 20  # Process transactions in batches
    
    async def __aenter__(self):
        # Create session with connection pooling (increased for 50 workers)
        connector = aiohttp.TCPConnector(
            limit=50,  # Total connection pool size (reduced for stability)
            limit_per_host=25,  # Per-host connection limit (reduced for stability)
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=20)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_last_collected_block(self) -> int:
        """Get the last collected block number from database"""
        if not os.path.exists(self.db_path):
            logger.error("Database not found!")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT MAX(block_number) FROM astar_blocks")
            result = cursor.fetchone()
            last_block = result[0] if result[0] else 0
            logger.info(f"Last collected block: {last_block}")
            return last_block
        except Exception as e:
            logger.error(f"Error getting last block: {e}")
            return 0
        finally:
            conn.close()
    
    def get_collection_stats(self) -> Dict:
        """Get current collection statistics"""
        if not os.path.exists(self.db_path):
            return {"blocks": 0, "transactions": 0, "min_block": 0, "max_block": 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM astar_blocks")
            blocks_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_transactions")
            tx_count = cursor.fetchone()[0]
            
            # Get block range
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
            block_range = cursor.fetchone()
            
            return {
                "blocks": blocks_count,
                "transactions": tx_count,
                "min_block": block_range[0] or 0,
                "max_block": block_range[1] or 0
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"blocks": 0, "transactions": 0, "min_block": 0, "max_block": 0}
        finally:
            conn.close()
    
    def switch_endpoint(self):
        """Switch to next RPC endpoint"""
        self.current_endpoint_index = (self.current_endpoint_index + 1) % len(self.rpc_endpoints)
        self.rpc_url = self.rpc_endpoints[self.current_endpoint_index]
        logger.info(f"Switched to RPC endpoint: {self.rpc_url}")

    async def make_rpc_call(self, method: str, params: List = None, max_retries: int = 3) -> Dict:
        """Make RPC call to Astar endpoint with rate limiting, retry logic, and endpoint switching"""
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
            
            for attempt in range(max_retries):
                try:
                    async with self.session.post(
                        self.rpc_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get('result', {})
                        elif response.status == 429:  # Rate limited
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                                logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.error(f"RPC call failed after {max_retries} retries: 429")
                                return {}
                        else:
                            logger.error(f"RPC call failed: {response.status}")
                            return {}
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Try switching endpoint on connection errors
                        if "timeout" in str(e).lower() or "connection" in str(e).lower():
                            self.switch_endpoint()
                        wait_time = (2 ** attempt) * 1  # Exponential backoff: 1, 2, 4 seconds
                        logger.warning(f"RPC call error, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Error making RPC call after {max_retries} retries: {e}")
                        return {}
            
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
    
    async def resume_collection(self, target_days: int = 7):
        """Resume collection from the last collected block"""
        logger.info("🔄 Resuming Astar data collection...")
        
        # Get current stats
        stats = self.get_collection_stats()
        logger.info(f"Current collection stats: {stats}")
        
        # Get last collected block
        last_block = self.get_last_collected_block()
        if last_block == 0:
            logger.error("No previous collection found!")
            return
        
        # Get current block
        current_block = await self.get_current_block_number()
        logger.info(f"Current Astar block: {current_block}")
        
        # Calculate remaining blocks
        start_block = last_block + 1
        remaining_blocks = current_block - start_block + 1
        
        if remaining_blocks <= 0:
            logger.info("✅ Collection is up to date!")
            return
        
        logger.info(f"Resuming from block {start_block} to {current_block}")
        logger.info(f"Remaining blocks to collect: {remaining_blocks}")
        
        # Set up progress tracking
        self.total_blocks = remaining_blocks
        self.collected_blocks = 0
        self.start_time = time.time()
        
        # Create block number ranges for batch processing
        block_ranges = []
        for i in range(start_block, current_block + 1, self.batch_size):
            end_range = min(i + self.batch_size - 1, current_block)
            block_ranges.append(list(range(i, end_range + 1)))
        
        logger.info(f"Processing {len(block_ranges)} batches of {self.batch_size} blocks each")
        logger.info(f"Using {self.max_workers} concurrent workers")
        
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
        
        # Final stats
        final_stats = self.get_collection_stats()
        logger.success(f"🎉 Resume collection completed!")
        logger.info(f"Final stats: {final_stats}")

async def main():
    """Main function"""
    # Initialize resume collector with 50 workers
    async with AstarResumeCollector(max_workers=50) as collector:
        # Resume collection
        await collector.resume_collection(target_days=7)

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run the resume collector
    asyncio.run(main())
