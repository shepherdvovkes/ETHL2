#!/usr/bin/env python3
"""
Solana Archive Data Collector with 10 Concurrent Workers
Collects complete historical Solana blockchain data from genesis to present
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
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solana_archive_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaArchiveCollector:
    """Solana archive data collector with concurrent workers for complete chain data"""
    
    def __init__(self, config_file: str = "solana_config.env"):
        self.config = self.load_config(config_file)
        self.db_path = self.config.get('SOLANA_DATABASE_PATH', 'solana_data.db')
        self.archive_db_path = self.config.get('SOLANA_ARCHIVE_DATABASE_PATH', 'solana_archive_data.db')
        self.rpc_url = self.config.get('SOLANA_RPC_URL')
        self.wss_url = self.config.get('SOLANA_WSS_URL')
        
        # Worker configuration
        self.num_workers = int(self.config.get('ARCHIVE_CONCURRENT_WORKERS', 10))
        self.batch_size = int(self.config.get('ARCHIVE_BATCH_SIZE', 1000))
        self.max_retries = int(self.config.get('ARCHIVE_MAX_RETRIES', 3))
        self.retry_delay = int(self.config.get('ARCHIVE_RETRY_DELAY', 5))
        
        # Archive settings
        self.start_slot = int(self.config.get('ARCHIVE_START_SLOT', 0))
        self.archive_mode = self.config.get('ARCHIVE_MODE', 'full')
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(50)  # Conservative rate limiting for archive
        self.request_lock = Lock()
        
        # Database connection pool
        self.db_connections = []
        self.archive_db_connections = []
        self.db_lock = Lock()
        
        # Progress tracking
        self.progress = {
            'total_slots': 0,
            'completed_slots': 0,
            'failed_slots': 0,
            'current_slot': 0,
            'start_time': None,
            'last_updated': None
        }
        
        # Statistics
        self.stats = {
            'blocks_collected': 0,
            'transactions_collected': 0,
            'accounts_collected': 0,
            'tokens_collected': 0,
            'programs_collected': 0,
            'validators_collected': 0,
            'errors': 0,
            'start_time': None
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
        """Initialize database connection pools"""
        # Main database connections
        for _ in range(self.num_workers):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self.db_connections.append(conn)
        
        # Archive database connections
        for _ in range(self.num_workers):
            conn = sqlite3.connect(self.archive_db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self.archive_db_connections.append(conn)
    
    def get_db_connection(self, archive: bool = False):
        """Get database connection from pool"""
        with self.db_lock:
            if archive:
                if self.archive_db_connections:
                    return self.archive_db_connections.pop()
                else:
                    conn = sqlite3.connect(self.archive_db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    return conn
            else:
                if self.db_connections:
                    return self.db_connections.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")
                    return conn
    
    def return_db_connection(self, conn, archive: bool = False):
        """Return database connection to pool"""
        with self.db_lock:
            if archive:
                if len(self.archive_db_connections) < self.num_workers:
                    self.archive_db_connections.append(conn)
                else:
                    conn.close()
            else:
                if len(self.db_connections) < self.num_workers:
                    self.db_connections.append(conn)
                else:
                    conn.close()
    
    async def make_rpc_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make RPC request to Solana node with retry logic"""
        for attempt in range(self.max_retries):
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params or []
                    }
                    
                    try:
                        async with session.post(
                            self.rpc_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'error' in data:
                                    logger.error(f"RPC Error: {data['error']}")
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                                        continue
                                    return None
                                return data.get('result')
                            else:
                                logger.error(f"HTTP Error: {response.status}")
                                if attempt < self.max_retries - 1:
                                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                                    continue
                                return None
                                
                    except Exception as e:
                        logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        return None
        
        return None
    
    async def get_latest_slot(self) -> Optional[int]:
        """Get the latest slot number"""
        result = await self.make_rpc_request("getSlot")
        return result
    
    async def get_block(self, slot: int) -> Optional[Dict[str, Any]]:
        """Get block data for a specific slot"""
        result = await self.make_rpc_request("getBlock", [
            slot, 
            {
                "encoding": "jsonParsed", 
                "transactionDetails": "full",
                "rewards": True,
                "maxSupportedTransactionVersion": 0
            }
        ])
        return result
    
    async def get_blocks(self, start_slot: int, end_slot: int) -> List[Optional[Dict[str, Any]]]:
        """Get multiple blocks in a range"""
        tasks = []
        for slot in range(start_slot, end_slot + 1):
            tasks.append(self.get_block(slot))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def get_epoch_info(self) -> Optional[Dict[str, Any]]:
        """Get current epoch information"""
        result = await self.make_rpc_request("getEpochInfo")
        return result
    
    async def get_validators(self) -> Optional[Dict[str, Any]]:
        """Get validator information"""
        result = await self.make_rpc_request("getVoteAccounts")
        return result
    
    async def get_supply(self) -> Optional[Dict[str, Any]]:
        """Get token supply information"""
        result = await self.make_rpc_request("getSupply")
        return result
    
    def save_archive_block_data(self, block_data: Dict[str, Any], slot: int):
        """Save block data to archive database"""
        conn = self.get_db_connection(archive=True)
        try:
            cursor = conn.cursor()
            
            # Insert block data
            cursor.execute("""
                INSERT OR REPLACE INTO solana_archive_blocks 
                (slot, blockhash, parent_slot, parent_blockhash, timestamp, block_time, 
                 block_height, transaction_count, total_fee, reward, leader, leader_reward,
                 vote_accounts_count, vote_accounts_stake)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                slot,
                block_data.get('blockhash'),
                block_data.get('parentSlot'),
                block_data.get('previousBlockhash'),
                datetime.fromtimestamp(block_data.get('blockTime', 0)) if block_data.get('blockTime') else None,
                block_data.get('blockTime'),
                block_data.get('blockHeight'),
                len(block_data.get('transactions', [])),
                sum(tx.get('meta', {}).get('fee', 0) for tx in block_data.get('transactions', [])),
                block_data.get('rewards', [{}])[0].get('lamports', 0) if block_data.get('rewards') else 0,
                block_data.get('rewards', [{}])[0].get('pubkey') if block_data.get('rewards') else None,
                block_data.get('rewards', [{}])[0].get('lamports', 0) if block_data.get('rewards') else 0,
                len(block_data.get('rewards', [])),
                sum(r.get('lamports', 0) for r in block_data.get('rewards', []))
            ))
            
            # Insert transaction data
            for tx in block_data.get('transactions', []):
                if tx.get('transaction'):
                    signature = tx['transaction'].get('signatures', [''])[0]
                    meta = tx.get('meta', {})
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO solana_archive_transactions 
                        (signature, slot, block_time, fee, success, error, compute_units_consumed,
                         compute_units_price, accounts_count, instructions_count, log_messages,
                         inner_instructions, pre_balances, post_balances, pre_token_balances, post_token_balances)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        signature,
                        slot,
                        datetime.fromtimestamp(block_data.get('blockTime', 0)) if block_data.get('blockTime') else None,
                        meta.get('fee', 0),
                        meta.get('err') is None,
                        json.dumps(meta.get('err')) if meta.get('err') else None,
                        meta.get('computeUnitsConsumed'),
                        meta.get('computeUnitsPrice'),
                        len(tx['transaction'].get('message', {}).get('accountKeys', [])),
                        len(tx['transaction'].get('message', {}).get('instructions', [])),
                        json.dumps(meta.get('logMessages', [])),
                        json.dumps(meta.get('innerInstructions', [])),
                        json.dumps(meta.get('preBalances', [])),
                        json.dumps(meta.get('postBalances', [])),
                        json.dumps(meta.get('preTokenBalances', [])),
                        json.dumps(meta.get('postTokenBalances', []))
                    ))
            
            conn.commit()
            self.stats['blocks_collected'] += 1
            self.stats['transactions_collected'] += len(block_data.get('transactions', []))
            
        except Exception as e:
            logger.error(f"Error saving archive block data for slot {slot}: {e}")
            conn.rollback()
            self.stats['errors'] += 1
        finally:
            self.return_db_connection(conn, archive=True)
    
    def save_archive_network_metrics(self, epoch_info: Dict[str, Any], validators: Dict[str, Any], supply: Dict[str, Any]):
        """Save network metrics to archive database"""
        conn = self.get_db_connection(archive=True)
        try:
            cursor = conn.cursor()
            
            current_slot = epoch_info.get('slot', 0)
            current_epoch = epoch_info.get('epoch', 0)
            
            # Calculate metrics
            total_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []))
            active_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []) if not v.get('delinquent', False))
            delinquent_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []) if v.get('delinquent', False))
            
            cursor.execute("""
                INSERT INTO solana_archive_network_metrics 
                (timestamp, slot, epoch, tps, block_time_avg, block_time_std, slot_time_avg, slot_time_std,
                 skipped_slots, skipped_slots_percentage, vote_accounts_count, total_stake, active_stake, delinquent_stake,
                 total_supply, circulating_supply, inflation_rate, staking_ratio, avg_fee_per_transaction,
                 total_fees_24h, priority_fee_avg, validator_count, active_validator_count, delinquent_validator_count,
                 avg_commission, defi_tvl, defi_protocols_count, dex_volume_24h, nft_transactions_24h, nft_volume_24h, nft_marketplaces_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                current_slot,
                current_epoch,
                None,  # TPS calculation would need historical data
                400,   # Target slot time
                0,     # Standard deviation
                400,   # Target slot time
                0,     # Standard deviation
                0,     # Skipped slots
                0.0,   # Skipped slots percentage
                len(validators.get('current', [])),
                total_stake,
                active_stake,
                delinquent_stake,
                supply.get('value', {}).get('total', 0),
                supply.get('value', {}).get('circulating', 0),
                None,  # Inflation rate calculation
                active_stake / total_stake if total_stake > 0 else 0,
                None,  # Average fee per transaction
                None,  # Total fees 24h
                None,  # Priority fee average
                len(validators.get('current', [])),
                len([v for v in validators.get('current', []) if not v.get('delinquent', False)]),
                len([v for v in validators.get('current', []) if v.get('delinquent', False)]),
                None,  # Average commission
                None,  # DeFi TVL
                None,  # DeFi protocols count
                None,  # DEX volume 24h
                None,  # NFT transactions 24h
                None,  # NFT volume 24h
                None   # NFT marketplaces count
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving archive network metrics: {e}")
            conn.rollback()
            self.stats['errors'] += 1
        finally:
            self.return_db_connection(conn, archive=True)
    
    def update_archive_progress(self, collection_type: str, start_slot: int, end_slot: int, 
                              current_slot: int, total_slots: int, completed_slots: int, 
                              failed_slots: int, status: str):
        """Update archive collection progress"""
        conn = self.get_db_connection(archive=True)
        try:
            cursor = conn.cursor()
            
            progress_percentage = (completed_slots / total_slots * 100) if total_slots > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO solana_archive_progress 
                (collection_type, start_slot, end_slot, current_slot, total_slots, 
                 completed_slots, failed_slots, status, progress_percentage, 
                 started_at, last_updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                collection_type,
                start_slot,
                end_slot,
                current_slot,
                total_slots,
                completed_slots,
                failed_slots,
                status,
                progress_percentage,
                self.progress['start_time'],
                datetime.now(),
                datetime.now() if status == 'completed' else None
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating archive progress: {e}")
            conn.rollback()
        finally:
            self.return_db_connection(conn, archive=True)
    
    async def archive_worker(self, worker_id: int, slot_ranges: List[Tuple[int, int]]):
        """Worker for collecting archive data"""
        logger.info(f"Archive worker {worker_id} started with {len(slot_ranges)} ranges")
        
        for start_slot, end_slot in slot_ranges:
            try:
                logger.info(f"Worker {worker_id}: Processing slots {start_slot}-{end_slot}")
                
                # Process slots in batches
                for batch_start in range(start_slot, end_slot + 1, self.batch_size):
                    batch_end = min(batch_start + self.batch_size - 1, end_slot)
                    
                    # Get blocks in batch
                    blocks = await self.get_blocks(batch_start, batch_end)
                    
                    # Save blocks
                    for i, block_data in enumerate(blocks):
                        slot = batch_start + i
                        if block_data and isinstance(block_data, dict):
                            self.save_archive_block_data(block_data, slot)
                            self.progress['completed_slots'] += 1
                        else:
                            self.progress['failed_slots'] += 1
                            logger.warning(f"Worker {worker_id}: Failed to get block {slot}")
                    
                    # Update progress
                    self.progress['current_slot'] = batch_end
                    self.progress['last_updated'] = datetime.now()
                    
                    # Update database progress
                    self.update_archive_progress(
                        'blocks',
                        self.progress['current_slot'] - self.progress['completed_slots'],
                        self.progress['current_slot'],
                        self.progress['current_slot'],
                        self.progress['total_slots'],
                        self.progress['completed_slots'],
                        self.progress['failed_slots'],
                        'in_progress'
                    )
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
                
                logger.info(f"Worker {worker_id}: Completed range {start_slot}-{end_slot}")
                
            except Exception as e:
                logger.error(f"Archive worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    async def network_metrics_worker(self, worker_id: int):
        """Worker for collecting network metrics periodically"""
        logger.info(f"Network metrics worker {worker_id} started")
        
        while True:
            try:
                epoch_info = await self.get_epoch_info()
                validators = await self.get_validators()
                supply = await self.get_supply()
                
                if epoch_info and validators and supply:
                    self.save_archive_network_metrics(epoch_info, validators, supply)
                    self.stats['validators_collected'] += 1
                
                # Update every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Network metrics worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(60)
    
    async def stats_reporter(self):
        """Report collection statistics"""
        while True:
            try:
                runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                
                # Calculate progress percentage
                progress_pct = (self.progress['completed_slots'] / self.progress['total_slots'] * 100) if self.progress['total_slots'] > 0 else 0
                
                # Calculate ETA
                if self.progress['completed_slots'] > 0 and runtime > 0:
                    slots_per_second = self.progress['completed_slots'] / runtime
                    remaining_slots = self.progress['total_slots'] - self.progress['completed_slots']
                    eta_seconds = remaining_slots / slots_per_second if slots_per_second > 0 else 0
                    eta_hours = eta_seconds / 3600
                else:
                    eta_hours = 0
                
                logger.info(f"""
                📊 Solana Archive Collection Stats:
                Runtime: {runtime:.0f}s ({runtime/3600:.1f}h)
                Progress: {progress_pct:.2f}% ({self.progress['completed_slots']}/{self.progress['total_slots']} slots)
                Current Slot: {self.progress['current_slot']:,}
                Blocks: {self.stats['blocks_collected']:,}
                Transactions: {self.stats['transactions_collected']:,}
                Validators: {self.stats['validators_collected']:,}
                Errors: {self.stats['errors']:,}
                Failed Slots: {self.progress['failed_slots']:,}
                ETA: {eta_hours:.1f} hours
                """)
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
                await asyncio.sleep(60)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        # Update final progress
        self.update_archive_progress(
            'blocks',
            self.start_slot,
            self.progress['current_slot'],
            self.progress['current_slot'],
            self.progress['total_slots'],
            self.progress['completed_slots'],
            self.progress['failed_slots'],
            'interrupted'
        )
        sys.exit(0)
    
    def calculate_slot_ranges(self, start_slot: int, end_slot: int) -> List[Tuple[int, int]]:
        """Calculate slot ranges for workers"""
        total_slots = end_slot - start_slot + 1
        slots_per_worker = total_slots // self.num_workers
        remainder = total_slots % self.num_workers
        
        ranges = []
        current_start = start_slot
        
        for i in range(self.num_workers):
            # Add one extra slot to some workers to handle remainder
            worker_slots = slots_per_worker + (1 if i < remainder else 0)
            current_end = current_start + worker_slots - 1
            
            ranges.append((current_start, current_end))
            current_start = current_end + 1
        
        return ranges
    
    async def start_archive_collection(self):
        """Start the archive data collection"""
        logger.info("🚀 Starting Solana archive data collection...")
        self.stats['start_time'] = time.time()
        self.progress['start_time'] = datetime.now()
        
        try:
            # Get latest slot
            latest_slot = await self.get_latest_slot()
            if not latest_slot:
                logger.error("Failed to get latest slot")
                return
            
            logger.info(f"Latest slot: {latest_slot:,}")
            
            # Set total slots
            self.progress['total_slots'] = latest_slot - self.start_slot + 1
            self.progress['current_slot'] = self.start_slot
            
            logger.info(f"Total slots to collect: {self.progress['total_slots']:,}")
            
            # Calculate slot ranges for workers
            slot_ranges = self.calculate_slot_ranges(self.start_slot, latest_slot)
            
            logger.info(f"Distributing {len(slot_ranges)} ranges across {self.num_workers} workers")
            for i, (start, end) in enumerate(slot_ranges):
                logger.info(f"Worker {i}: slots {start:,} - {end:,} ({end-start+1:,} slots)")
            
            # Create worker tasks
            worker_tasks = []
            
            # Archive workers (9 workers for blocks)
            for i in range(min(9, len(slot_ranges))):
                task = asyncio.create_task(self.archive_worker(i, [slot_ranges[i]]))
                worker_tasks.append(task)
            
            # Network metrics worker (1 worker)
            task = asyncio.create_task(self.network_metrics_worker(0))
            worker_tasks.append(task)
            
            # Stats reporter
            stats_task = asyncio.create_task(self.stats_reporter())
            worker_tasks.append(stats_task)
            
            # Wait for all tasks
            await asyncio.gather(*worker_tasks)
            
        except Exception as e:
            logger.error(f"Error in archive collection: {e}")
        finally:
            # Update final progress
            self.update_archive_progress(
                'blocks',
                self.start_slot,
                self.progress['current_slot'],
                self.progress['current_slot'],
                self.progress['total_slots'],
                self.progress['completed_slots'],
                self.progress['failed_slots'],
                'completed'
            )
            
            # Cleanup
            for conn in self.db_connections + self.archive_db_connections:
                conn.close()
            logger.info("✅ Solana archive collection completed")

async def main():
    """Main function"""
    collector = SolanaArchiveCollector()
    await collector.start_archive_collection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Solana archive collector stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)