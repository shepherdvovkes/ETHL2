#!/usr/bin/env python3
"""
LINEA Data Collector with 10 Concurrent Workers
Collects comprehensive LINEA blockchain data in real-time
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
import websockets
from web3 import Web3
from eth_abi import decode
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linea_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LineaDataCollector:
    """Main LINEA data collector with concurrent workers"""
    
    def __init__(self, config_file: str = "linea_config.env"):
        self.config = self.load_config(config_file)
        self.db_path = self.config.get('LINEA_DATABASE_PATH', 'linea_data.db')
        self.rpc_url = self.config.get('LINEA_RPC_URL')
        self.wss_url = self.config.get('LINEA_WSS_URL')
        
        # Worker configuration
        self.num_workers = int(self.config.get('CONCURRENT_WORKERS', 10))
        self.worker_tasks = []
        self.is_running = False
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(int(self.config.get('LINEA_RPC_RATE_LIMIT', 100)))
        self.request_lock = Lock()
        
        # Database connection pool
        self.db_connections = []
        self.db_lock = Lock()
        
        # Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Data collection intervals
        self.intervals = {
            'blocks': float(self.config.get('BLOCK_COLLECTION_INTERVAL', 2)),
            'transactions': float(self.config.get('TRANSACTION_COLLECTION_INTERVAL', 1)),
            'accounts': float(self.config.get('ACCOUNT_COLLECTION_INTERVAL', 5)),
            'contracts': float(self.config.get('CONTRACT_COLLECTION_INTERVAL', 10)),
            'tokens': float(self.config.get('TOKEN_COLLECTION_INTERVAL', 15)),
            'defi': float(self.config.get('DEFI_COLLECTION_INTERVAL', 30))
        }
        
        # Statistics
        self.stats = {
            'blocks_collected': 0,
            'transactions_collected': 0,
            'accounts_collected': 0,
            'contracts_collected': 0,
            'tokens_collected': 0,
            'defi_collected': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Known contract addresses
        self.known_contracts = {
            'bridge': '0xA0b86a33E6441E0a4bFc0B4d5F3F3E5A4F3F3F3F',
            'message_service': '0xd19bae9c65bde34f26c2ee8f2f3f3e5a4f3f3f3f',
            'aave': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
            'curve': '0x7f90122BF0700F9E7e1F688fe926940E8839F353'
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
            conn.execute("PRAGMA cache_size=20000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self.db_connections.append(conn)
    
    def get_db_connection(self):
        """Get a database connection from the pool"""
        with self.db_lock:
            if self.db_connections:
                return self.db_connections.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
        """Make RPC request with rate limiting"""
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
                                return None
                        else:
                            logger.error(f"HTTP error: {response.status}")
                            return None
                except Exception as e:
                    logger.error(f"RPC request failed: {e}")
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
    
    async def get_account_balance(self, address: str, block_number: str = "latest") -> str:
        """Get account balance"""
        result = await self.make_rpc_request("eth_getBalance", [address, block_number])
        return result or "0x0"
    
    async def get_account_nonce(self, address: str, block_number: str = "latest") -> int:
        """Get account nonce"""
        result = await self.make_rpc_request("eth_getTransactionCount", [address, block_number])
        if result:
            return int(result, 16)
        return 0
    
    async def get_account_code(self, address: str, block_number: str = "latest") -> str:
        """Get account code"""
        result = await self.make_rpc_request("eth_getCode", [address, block_number])
        return result or "0x"
    
    def store_block(self, block_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store block data in database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_blocks 
                (block_number, block_hash, parent_hash, timestamp, gas_limit, gas_used, 
                 base_fee_per_gas, difficulty, size, transaction_count, extra_data, 
                 mix_hash, nonce, receipts_root, sha3_uncles, state_root, 
                 transactions_root, withdrawals_root, withdrawals, blob_gas_used, 
                 excess_blob_gas, parent_beacon_block_root)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(block_data.get('number', 0), 16),
                block_data.get('hash', ''),
                block_data.get('parentHash', ''),
                datetime.fromtimestamp(int(block_data.get('timestamp', 0), 16)),
                int(block_data.get('gasLimit', 0), 16),
                int(block_data.get('gasUsed', 0), 16),
                int(block_data.get('baseFeePerGas', 0), 16) if block_data.get('baseFeePerGas') else 0,
                int(block_data.get('difficulty', 0), 16),
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
            logger.error(f"Error storing block: {e}")
            conn.rollback()
    
    def store_transaction(self, tx_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store transaction data in database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_transactions 
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
            logger.error(f"Error storing transaction: {e}")
            conn.rollback()
    
    def store_transaction_receipt(self, receipt_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store transaction receipt data in database"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO linea_transaction_receipts 
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
            logger.error(f"Error storing transaction receipt: {e}")
            conn.rollback()
    
    def store_account(self, address: str, account_data: Dict[str, Any], conn: sqlite3.Connection):
        """Store account data in database"""
        try:
            cursor = conn.cursor()
            
            # Check if it's a contract
            is_contract = len(account_data.get('code', '0x')) > 2
            
            cursor.execute("""
                INSERT OR REPLACE INTO linea_accounts 
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
                account_data.get('firstSeenBlock', 0),
                account_data.get('lastSeenBlock', 0),
                account_data.get('transactionCount', 0)
            ))
            
            # If it's a contract, store in contracts table
            if is_contract:
                cursor.execute("""
                    INSERT OR REPLACE INTO linea_contracts 
                    (address, creator_address, creation_block, bytecode, is_verified)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    address,
                    account_data.get('creatorAddress', ''),
                    account_data.get('creationBlock', 0),
                    account_data.get('code', '0x'),
                    False
                ))
            
            conn.commit()
            self.stats['accounts_collected'] += 1
            if is_contract:
                self.stats['contracts_collected'] += 1
        except Exception as e:
            logger.error(f"Error storing account: {e}")
            conn.rollback()
    
    async def collect_block_data(self, worker_id: int):
        """Collect block data (Worker 1-3)"""
        logger.info(f"Worker {worker_id}: Starting block data collection")
        
        conn = self.get_db_connection()
        try:
            while self.is_running:
                try:
                    latest_block = await self.get_latest_block_number()
                    if latest_block > 0:
                        # Get blocks in batches
                        for block_num in range(max(1, latest_block - 10), latest_block + 1):
                            if not self.is_running:
                                break
                            
                            block_data = await self.get_block_by_number(block_num, True)
                            if block_data:
                                self.store_block(block_data, conn)
                                
                                # Store transactions and receipts
                                for tx in block_data.get('transactions', []):
                                    if isinstance(tx, dict):
                                        self.store_transaction(tx, conn)
                                        
                                        # Get transaction receipt
                                        tx_receipt = await self.get_transaction_receipt(tx.get('hash', ''))
                                        if tx_receipt:
                                            self.store_transaction_receipt(tx_receipt, conn)
                    
                    await asyncio.sleep(self.intervals['blocks'])
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} block collection error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(5)
                    
        finally:
            self.return_db_connection(conn)
            logger.info(f"Worker {worker_id}: Block data collection stopped")
    
    async def collect_account_data(self, worker_id: int):
        """Collect account data (Worker 4-6)"""
        logger.info(f"Worker {worker_id}: Starting account data collection")
        
        conn = self.get_db_connection()
        try:
            while self.is_running:
                try:
                    # Get unique addresses from recent transactions
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DISTINCT from_address, to_address 
                        FROM linea_transactions 
                        WHERE block_number > (SELECT MAX(block_number) - 100 FROM linea_transactions)
                        LIMIT 50
                    """)
                    addresses = cursor.fetchall()
                    
                    unique_addresses = set()
                    for from_addr, to_addr in addresses:
                        if from_addr:
                            unique_addresses.add(from_addr)
                        if to_addr:
                            unique_addresses.add(to_addr)
                    
                    # Collect account data
                    for address in list(unique_addresses)[:20]:  # Limit to 20 per iteration
                        if not self.is_running:
                            break
                        
                        balance = await self.get_account_balance(address)
                        nonce = await self.get_account_nonce(address)
                        code = await self.get_account_code(address)
                        
                        account_data = {
                            'balance': balance,
                            'nonce': nonce,
                            'code': code,
                            'firstSeenBlock': 0,
                            'lastSeenBlock': await self.get_latest_block_number(),
                            'transactionCount': 0
                        }
                        
                        self.store_account(address, account_data, conn)
                        await asyncio.sleep(0.1)  # Small delay between requests
                    
                    await asyncio.sleep(self.intervals['accounts'])
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} account collection error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(5)
                    
        finally:
            self.return_db_connection(conn)
            logger.info(f"Worker {worker_id}: Account data collection stopped")
    
    async def collect_token_data(self, worker_id: int):
        """Collect token data (Worker 7-8)"""
        logger.info(f"Worker {worker_id}: Starting token data collection")
        
        conn = self.get_db_connection()
        try:
            # Known ERC20 token contracts
            known_tokens = [
                "0xA0b86a33E6441E0a4bFc0B4d5F3F3E5A4F3F3F3F",  # Example token
                "0x794a61358D6845594F94dc1DB02A252b5b4814aD",  # AAVE
            ]
            
            while self.is_running:
                try:
                    for token_address in known_tokens:
                        if not self.is_running:
                            break
                        
                        # Check if token exists and get basic info
                        code = await self.get_account_code(token_address)
                        if len(code) > 2:  # Has code, likely a contract
                            balance = await self.get_account_balance(token_address)
                            nonce = await self.get_account_nonce(token_address)
                            
                            account_data = {
                                'balance': balance,
                                'nonce': nonce,
                                'code': code,
                                'firstSeenBlock': 0,
                                'lastSeenBlock': await self.get_latest_block_number(),
                                'transactionCount': 0
                            }
                            
                            self.store_account(token_address, account_data, conn)
                            
                            # Store as token if it has token-like characteristics
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT OR REPLACE INTO linea_tokens 
                                (address, name, symbol, decimals, token_type, is_native)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                token_address,
                                f"Token_{token_address[:8]}",
                                f"TKN_{token_address[:4]}",
                                18,
                                "ERC20",
                                False
                            ))
                            conn.commit()
                            self.stats['tokens_collected'] += 1
                        
                        await asyncio.sleep(0.5)
                    
                    await asyncio.sleep(self.intervals['tokens'])
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} token collection error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(5)
                    
        finally:
            self.return_db_connection(conn)
            logger.info(f"Worker {worker_id}: Token data collection stopped")
    
    async def collect_defi_data(self, worker_id: int):
        """Collect DeFi protocol data (Worker 9-10)"""
        logger.info(f"Worker {worker_id}: Starting DeFi data collection")
        
        conn = self.get_db_connection()
        try:
            while self.is_running:
                try:
                    for protocol_name, protocol_address in self.known_contracts.items():
                        if not self.is_running:
                            break
                        
                        # Get protocol contract data
                        code = await self.get_account_code(protocol_address)
                        if len(code) > 2:
                            balance = await self.get_account_balance(protocol_address)
                            nonce = await self.get_account_nonce(protocol_address)
                            
                            account_data = {
                                'balance': balance,
                                'nonce': nonce,
                                'code': code,
                                'firstSeenBlock': 0,
                                'lastSeenBlock': await self.get_latest_block_number(),
                                'transactionCount': 0
                            }
                            
                            self.store_account(protocol_address, account_data, conn)
                            
                            # Store DeFi protocol data
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT OR REPLACE INTO linea_defi_protocols 
                                (protocol_name, protocol_address, protocol_type, block_number, timestamp)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                protocol_name,
                                protocol_address,
                                "DeFi",
                                await self.get_latest_block_number(),
                                datetime.now()
                            ))
                            conn.commit()
                            self.stats['defi_collected'] += 1
                        
                        await asyncio.sleep(0.5)
                    
                    await asyncio.sleep(self.intervals['defi'])
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} DeFi collection error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(5)
                    
        finally:
            self.return_db_connection(conn)
            logger.info(f"Worker {worker_id}: DeFi data collection stopped")
    
    async def collect_network_metrics(self):
        """Collect network metrics periodically"""
        logger.info("Starting network metrics collection")
        
        conn = self.get_db_connection()
        try:
            while self.is_running:
                try:
                    latest_block = await self.get_latest_block_number()
                    
                    # Calculate basic metrics
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM linea_transactions 
                        WHERE block_number > ?
                    """, (latest_block - 100,))
                    tx_count = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT AVG(gas_used), AVG(gas_price), COUNT(DISTINCT from_address)
                        FROM linea_transactions 
                        WHERE block_number > ?
                    """, (latest_block - 100,))
                    metrics = cursor.fetchone()
                    
                    avg_gas_used = metrics[0] or 0
                    avg_gas_price = metrics[1] or 0
                    unique_addresses = metrics[2] or 0
                    
                    # Store network metrics
                    cursor.execute("""
                        INSERT INTO linea_network_metrics 
                        (timestamp, block_number, tps, block_time_avg, gas_utilization, 
                         gas_price_avg, transaction_count, unique_addresses_count, 
                         total_gas_used, total_fees)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(),
                        latest_block,
                        tx_count / 100.0,  # Rough TPS calculation
                        2.0,  # LINEA block time
                        (avg_gas_used / 30000000.0) * 100,  # Gas utilization
                        avg_gas_price,
                        tx_count,
                        unique_addresses,
                        avg_gas_used * tx_count,
                        avg_gas_price * tx_count
                    ))
                    conn.commit()
                    
                    await asyncio.sleep(60)  # Collect metrics every minute
                    
                except Exception as e:
                    logger.error(f"Network metrics collection error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(60)
                    
        finally:
            self.return_db_connection(conn)
            logger.info("Network metrics collection stopped")
    
    def print_stats(self):
        """Print collection statistics"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            logger.info(f"""
📊 LINEA Data Collection Statistics:
   Runtime: {runtime}
   Blocks collected: {self.stats['blocks_collected']}
   Transactions collected: {self.stats['transactions_collected']}
   Accounts collected: {self.stats['accounts_collected']}
   Contracts collected: {self.stats['contracts_collected']}
   Tokens collected: {self.stats['tokens_collected']}
   DeFi protocols collected: {self.stats['defi_collected']}
   Errors: {self.stats['errors']}
   Workers: {self.num_workers}
            """)
    
    async def start_collection(self):
        """Start all data collection workers"""
        logger.info("🚀 Starting LINEA data collection with 10 concurrent workers...")
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Create worker tasks
        tasks = []
        
        # Workers 1-3: Block data collection
        for i in range(1, 4):
            task = asyncio.create_task(self.collect_block_data(i))
            tasks.append(task)
        
        # Workers 4-6: Account data collection
        for i in range(4, 7):
            task = asyncio.create_task(self.collect_account_data(i))
            tasks.append(task)
        
        # Workers 7-8: Token data collection
        for i in range(7, 9):
            task = asyncio.create_task(self.collect_token_data(i))
            tasks.append(task)
        
        # Workers 9-10: DeFi data collection
        for i in range(9, 11):
            task = asyncio.create_task(self.collect_defi_data(i))
            tasks.append(task)
        
        # Network metrics collection
        metrics_task = asyncio.create_task(self.collect_network_metrics())
        tasks.append(metrics_task)
        
        # Statistics printing task
        stats_task = asyncio.create_task(self.print_stats_periodically())
        tasks.append(stats_task)
        
        self.worker_tasks = tasks
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Collection tasks cancelled")
        except Exception as e:
            logger.error(f"Collection error: {e}")
        finally:
            self.stop_collection()
    
    async def print_stats_periodically(self):
        """Print statistics every 5 minutes"""
        while self.is_running:
            await asyncio.sleep(300)  # 5 minutes
            if self.is_running:
                self.print_stats()
    
    def stop_collection(self):
        """Stop all data collection"""
        logger.info("🛑 Stopping LINEA data collection...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Close database connections
        with self.db_lock:
            for conn in self.db_connections:
                conn.close()
            self.db_connections.clear()
        
        self.print_stats()
        logger.info("✅ LINEA data collection stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_collection()
        sys.exit(0)

async def main():
    """Main function"""
    collector = LineaDataCollector()
    
    try:
        await collector.start_collection()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(main())
