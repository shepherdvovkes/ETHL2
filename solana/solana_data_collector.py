#!/usr/bin/env python3
"""
Solana Data Collector with 10 Concurrent Workers
Collects comprehensive Solana blockchain data in real-time
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
from typing import Dict, List, Optional, Any
import signal
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solana_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaDataCollector:
    """Main Solana data collector with concurrent workers"""
    
    def __init__(self, config_file: str = "solana_config.env"):
        self.config = self.load_config(config_file)
        self.db_path = self.config.get('SOLANA_DATABASE_PATH', 'solana_data.db')
        self.rpc_url = self.config.get('SOLANA_RPC_URL')
        self.wss_url = self.config.get('SOLANA_WSS_URL')
        
        # Worker configuration
        self.num_workers = 10
        self.worker_tasks = []
        self.is_running = False
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(100)  # 100 requests per second
        self.request_lock = Lock()
        
        # Database connection pool
        self.db_connections = []
        self.db_lock = Lock()
        
        # Data collection intervals
        self.intervals = {
            'blocks': float(self.config.get('BLOCK_COLLECTION_INTERVAL', 1)),
            'transactions': float(self.config.get('TRANSACTION_COLLECTION_INTERVAL', 1)),
            'accounts': float(self.config.get('ACCOUNT_COLLECTION_INTERVAL', 5)),
            'tokens': float(self.config.get('TOKEN_COLLECTION_INTERVAL', 10)),
            'programs': float(self.config.get('PROGRAM_COLLECTION_INTERVAL', 30)),
            'validators': float(self.config.get('VALIDATOR_COLLECTION_INTERVAL', 60))
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
        """Initialize database connection pool"""
        for _ in range(self.num_workers):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self.db_connections.append(conn)
    
    def get_db_connection(self):
        """Get database connection from pool"""
        with self.db_lock:
            if self.db_connections:
                return self.db_connections.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                return conn
    
    def return_db_connection(self, conn):
        """Return database connection to pool"""
        with self.db_lock:
            if len(self.db_connections) < self.num_workers:
                self.db_connections.append(conn)
            else:
                conn.close()
    
    async def make_rpc_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make RPC request to Solana node"""
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
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'error' in data:
                                logger.error(f"RPC Error: {data['error']}")
                                return None
                            return data.get('result')
                        else:
                            logger.error(f"HTTP Error: {response.status}")
                            return None
                            
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    return None
    
    async def get_latest_slot(self) -> Optional[int]:
        """Get the latest slot number"""
        result = await self.make_rpc_request("getSlot")
        return result
    
    async def get_block(self, slot: int) -> Optional[Dict[str, Any]]:
        """Get block data for a specific slot"""
        result = await self.make_rpc_request("getBlock", [slot, {"encoding": "jsonParsed", "transactionDetails": "full"}])
        return result
    
    async def get_block_production(self, start_slot: int, end_slot: int) -> Optional[Dict[str, Any]]:
        """Get block production data"""
        result = await self.make_rpc_request("getBlockProduction", [{"startSlot": start_slot, "endSlot": end_slot}])
        return result
    
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
    
    async def get_token_accounts_by_owner(self, owner: str, program_id: str) -> Optional[Dict[str, Any]]:
        """Get token accounts for an owner"""
        result = await self.make_rpc_request("getTokenAccountsByOwner", [
            owner,
            {"programId": program_id},
            {"encoding": "jsonParsed"}
        ])
        return result
    
    async def get_program_accounts(self, program_id: str) -> Optional[Dict[str, Any]]:
        """Get all accounts for a program"""
        result = await self.make_rpc_request("getProgramAccounts", [
            program_id,
            {"encoding": "jsonParsed"}
        ])
        return result
    
    def save_block_data(self, block_data: Dict[str, Any], slot: int):
        """Save block data to database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Insert block data
            cursor.execute("""
                INSERT OR REPLACE INTO solana_blocks 
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
                        INSERT OR REPLACE INTO solana_transactions 
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
            logger.error(f"Error saving block data: {e}")
            conn.rollback()
            self.stats['errors'] += 1
        finally:
            self.return_db_connection(conn)
    
    def save_network_metrics(self, epoch_info: Dict[str, Any], validators: Dict[str, Any], supply: Dict[str, Any]):
        """Save network metrics to database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            current_slot = epoch_info.get('slot', 0)
            current_epoch = epoch_info.get('epoch', 0)
            
            # Calculate metrics
            total_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []))
            active_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []) if not v.get('delinquent', False))
            delinquent_stake = sum(v.get('activatedStake', 0) for v in validators.get('current', []) if v.get('delinquent', False))
            
            cursor.execute("""
                INSERT INTO solana_network_metrics 
                (timestamp, slot, epoch, tps, block_time_avg, slot_time_avg, 
                 vote_accounts_count, total_stake, active_stake, delinquent_stake,
                 total_supply, circulating_supply, inflation_rate, staking_ratio,
                 validator_count, active_validator_count, delinquent_validator_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                current_slot,
                current_epoch,
                None,  # TPS calculation would need historical data
                400,   # Target slot time
                400,   # Target slot time
                len(validators.get('current', [])),
                total_stake,
                active_stake,
                delinquent_stake,
                supply.get('value', {}).get('total', 0),
                supply.get('value', {}).get('circulating', 0),
                None,  # Inflation rate calculation
                active_stake / total_stake if total_stake > 0 else 0,
                len(validators.get('current', [])),
                len([v for v in validators.get('current', []) if not v.get('delinquent', False)]),
                len([v for v in validators.get('current', []) if v.get('delinquent', False)])
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving network metrics: {e}")
            conn.rollback()
            self.stats['errors'] += 1
        finally:
            self.return_db_connection(conn)
    
    async def block_collector_worker(self, worker_id: int):
        """Worker for collecting block data"""
        logger.info(f"Block collector worker {worker_id} started")
        
        while self.is_running:
            try:
                latest_slot = await self.get_latest_slot()
                if latest_slot:
                    # Collect recent blocks
                    for slot_offset in range(10):  # Collect last 10 blocks
                        slot = latest_slot - slot_offset
                        block_data = await self.get_block(slot)
                        if block_data:
                            self.save_block_data(block_data, slot)
                
                await asyncio.sleep(self.intervals['blocks'])
                
            except Exception as e:
                logger.error(f"Block collector worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    async def network_metrics_worker(self, worker_id: int):
        """Worker for collecting network metrics"""
        logger.info(f"Network metrics worker {worker_id} started")
        
        while self.is_running:
            try:
                epoch_info = await self.get_epoch_info()
                validators = await self.get_validators()
                supply = await self.get_supply()
                
                if epoch_info and validators and supply:
                    self.save_network_metrics(epoch_info, validators, supply)
                
                await asyncio.sleep(self.intervals['validators'])
                
            except Exception as e:
                logger.error(f"Network metrics worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    async def account_collector_worker(self, worker_id: int):
        """Worker for collecting account data"""
        logger.info(f"Account collector worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get system program accounts
                system_program = "11111111111111111111111111111111"
                accounts = await self.get_program_accounts(system_program)
                
                if accounts:
                    conn = self.get_db_connection()
                    try:
                        cursor = conn.cursor()
                        
                        for account in accounts[:100]:  # Limit to 100 accounts per cycle
                            account_data = account.get('account', {})
                            pubkey = account.get('pubkey')
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO solana_accounts 
                                (address, owner, lamports, data_length, executable, rent_epoch, account_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                pubkey,
                                account_data.get('owner'),
                                account_data.get('lamports', 0),
                                account_data.get('data', {}).get('length', 0),
                                account_data.get('executable', False),
                                account_data.get('rentEpoch'),
                                'system'
                            ))
                        
                        conn.commit()
                        self.stats['accounts_collected'] += len(accounts[:100])
                        
                    except Exception as e:
                        logger.error(f"Error saving account data: {e}")
                        conn.rollback()
                        self.stats['errors'] += 1
                    finally:
                        self.return_db_connection(conn)
                
                await asyncio.sleep(self.intervals['accounts'])
                
            except Exception as e:
                logger.error(f"Account collector worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    async def token_collector_worker(self, worker_id: int):
        """Worker for collecting token data"""
        logger.info(f"Token collector worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get token program accounts
                token_program = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                accounts = await self.get_program_accounts(token_program)
                
                if accounts:
                    conn = self.get_db_connection()
                    try:
                        cursor = conn.cursor()
                        
                        for account in accounts[:50]:  # Limit to 50 tokens per cycle
                            account_data = account.get('account', {})
                            pubkey = account.get('pubkey')
                            
                            # Parse token account data
                            data = account_data.get('data', [])
                            if len(data) >= 2:
                                parsed_data = data[1]  # Base64 decoded data
                                # This would need proper token account parsing
                                # For now, just store basic info
                                
                                cursor.execute("""
                                    INSERT OR REPLACE INTO solana_token_accounts 
                                    (address, mint, owner, amount, is_initialized, is_frozen)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    pubkey,
                                    None,  # Would need to parse from data
                                    None,  # Would need to parse from data
                                    0,     # Would need to parse from data
                                    True,  # Assume initialized
                                    False  # Assume not frozen
                                ))
                        
                        conn.commit()
                        self.stats['tokens_collected'] += len(accounts[:50])
                        
                    except Exception as e:
                        logger.error(f"Error saving token data: {e}")
                        conn.rollback()
                        self.stats['errors'] += 1
                    finally:
                        self.return_db_connection(conn)
                
                await asyncio.sleep(self.intervals['tokens'])
                
            except Exception as e:
                logger.error(f"Token collector worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    async def program_collector_worker(self, worker_id: int):
        """Worker for collecting program data"""
        logger.info(f"Program collector worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get program accounts for known programs
                programs = [
                    "11111111111111111111111111111111",  # System Program
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token Program
                    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated Token Program
                    "Rent111111111111111111111111111111111"  # Rent Program
                ]
                
                for program_id in programs:
                    accounts = await self.get_program_accounts(program_id)
                    
                    if accounts:
                        conn = self.get_db_connection()
                        try:
                            cursor = conn.cursor()
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO solana_programs 
                                (program_id, name, program_type, is_upgradeable, slot)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                program_id,
                                self.get_program_name(program_id),
                                'system',
                                False,
                                0  # Current slot
                            ))
                            
                            conn.commit()
                            self.stats['programs_collected'] += 1
                            
                        except Exception as e:
                            logger.error(f"Error saving program data: {e}")
                            conn.rollback()
                            self.stats['errors'] += 1
                        finally:
                            self.return_db_connection(conn)
                
                await asyncio.sleep(self.intervals['programs'])
                
            except Exception as e:
                logger.error(f"Program collector worker {worker_id} error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)
    
    def get_program_name(self, program_id: str) -> str:
        """Get human-readable program name"""
        program_names = {
            "11111111111111111111111111111111": "System Program",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "Token Program",
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": "Associated Token Program",
            "Rent111111111111111111111111111111111": "Rent Program"
        }
        return program_names.get(program_id, "Unknown Program")
    
    async def stats_reporter(self):
        """Report collection statistics"""
        while self.is_running:
            try:
                runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                
                logger.info(f"""
                📊 Solana Data Collection Stats:
                Runtime: {runtime:.0f}s
                Blocks: {self.stats['blocks_collected']}
                Transactions: {self.stats['transactions_collected']}
                Accounts: {self.stats['accounts_collected']}
                Tokens: {self.stats['tokens_collected']}
                Programs: {self.stats['programs_collected']}
                Validators: {self.stats['validators_collected']}
                Errors: {self.stats['errors']}
                """)
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
                await asyncio.sleep(60)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def start(self):
        """Start the data collector"""
        logger.info("🚀 Starting Solana data collector...")
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Create worker tasks
        worker_tasks = []
        
        # Block collectors (3 workers)
        for i in range(3):
            task = asyncio.create_task(self.block_collector_worker(i))
            worker_tasks.append(task)
        
        # Network metrics collector (1 worker)
        task = asyncio.create_task(self.network_metrics_worker(0))
        worker_tasks.append(task)
        
        # Account collectors (2 workers)
        for i in range(2):
            task = asyncio.create_task(self.account_collector_worker(i))
            worker_tasks.append(task)
        
        # Token collectors (2 workers)
        for i in range(2):
            task = asyncio.create_task(self.token_collector_worker(i))
            worker_tasks.append(task)
        
        # Program collectors (1 worker)
        task = asyncio.create_task(self.program_collector_worker(0))
        worker_tasks.append(task)
        
        # Validator collector (1 worker)
        task = asyncio.create_task(self.network_metrics_worker(1))
        worker_tasks.append(task)
        
        # Stats reporter
        stats_task = asyncio.create_task(self.stats_reporter())
        worker_tasks.append(stats_task)
        
        try:
            # Wait for all tasks
            await asyncio.gather(*worker_tasks)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Cleanup
            for conn in self.db_connections:
                conn.close()
            logger.info("✅ Solana data collector stopped")

async def main():
    """Main function"""
    collector = SolanaDataCollector()
    await collector.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Solana data collector stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
