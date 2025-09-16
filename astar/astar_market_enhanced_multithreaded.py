#!/usr/bin/env python3
"""
Astar Market-Enhanced Multi-Threaded Data Collector
==================================================

Enhanced version of the multi-threaded collector with integrated CoinGecko market data.
Combines high-performance network data collection with real-time market data.

Features:
- Multi-threaded block collection (15+ workers)
- Real-time market data integration
- Combined dataset for ML training
- Enhanced database schema with market metrics
- Progress tracking and resumption
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
from astar_coingecko_integration import AstarCoinGeckoIntegration
warnings.filterwarnings('ignore')

class AstarMarketEnhancedMultiThreadedCollector:
    """Market-enhanced multi-threaded data collector for Astar network"""
    
    def __init__(self, rpc_url: str = None, max_workers: int = 15, coingecko_api_key: str = None):
        # Astar RPC endpoints
        self.rpc_url = rpc_url or "https://rpc.astar.network"
        self.ws_url = "wss://rpc.astar.network"
        self.session = None
        
        # Market data integration
        self.coingecko = AstarCoinGeckoIntegration(api_key=coingecko_api_key)
        
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
        self.db_path = "astar_market_enhanced_data.db"
        self.setup_database()
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.collected_blocks = 0
        self.total_blocks = 0
        self.start_time = None
        self.market_data_updates = 0
        
        # Data collection settings
        self.batch_size = 50
        self.transaction_batch_size = 20
        self.market_data_interval = 300  # Update market data every 5 minutes
    
    def setup_database(self):
        """Setup enhanced database with market data integration"""
        logger.info("Setting up market-enhanced Astar database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Create enhanced tables with market data
        tables = {
            'astar_market_enhanced_blocks': '''
                CREATE TABLE IF NOT EXISTS astar_market_enhanced_blocks (
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
            'astar_market_enhanced_transactions': '''
                CREATE TABLE IF NOT EXISTS astar_market_enhanced_transactions (
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
                    FOREIGN KEY (block_number) REFERENCES astar_market_enhanced_blocks (block_number)
                )
            ''',
            'astar_market_enhanced_market_data': '''
                CREATE TABLE IF NOT EXISTS astar_market_enhanced_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    price_usd REAL,
                    market_cap_usd REAL,
                    volume_24h_usd REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    price_change_14d REAL,
                    price_change_30d REAL,
                    price_volatility_24h REAL,
                    price_volatility REAL,
                    price_momentum REAL,
                    volume_trend REAL,
                    market_cap_rank INTEGER,
                    data_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_market_enhanced_combined': '''
                CREATE TABLE IF NOT EXISTS astar_market_enhanced_combined (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    block_number INTEGER,
                    
                    -- Network metrics
                    transaction_count INTEGER,
                    gas_used INTEGER,
                    gas_limit INTEGER,
                    gas_utilization REAL,
                    network_activity REAL,
                    block_efficiency REAL,
                    
                    -- Market metrics
                    price_usd REAL,
                    market_cap_usd REAL,
                    volume_24h_usd REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    price_volatility_24h REAL,
                    price_volatility REAL,
                    price_momentum REAL,
                    volume_trend REAL,
                    
                    -- Combined metrics
                    network_health REAL,
                    defi_activity REAL,
                    contract_activity REAL,
                    market_sentiment REAL,
                    correlation_network_price REAL,
                    
                    -- Time features
                    hour INTEGER,
                    day_of_week INTEGER,
                    is_weekend BOOLEAN,
                    
                    -- Rolling averages
                    tx_count_24h REAL,
                    gas_used_24h REAL,
                    gas_util_24h REAL,
                    tx_count_7d REAL,
                    gas_used_7d REAL,
                    network_activity_7d REAL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_market_enhanced_metrics': '''
                CREATE TABLE IF NOT EXISTS astar_market_enhanced_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    block_number INTEGER,
                    
                    -- Network metrics
                    network_hash_rate TEXT,
                    network_difficulty TEXT,
                    active_addresses_24h INTEGER,
                    transaction_count_24h INTEGER,
                    gas_price_avg TEXT,
                    gas_used_avg INTEGER,
                    gas_limit_avg INTEGER,
                    block_time_avg REAL,
                    network_utilization REAL,
                    
                    -- Market metrics
                    price_usd REAL,
                    market_cap_usd REAL,
                    volume_24h_usd REAL,
                    price_change_1h REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    price_volatility_24h REAL,
                    
                    -- Combined metrics
                    network_market_correlation REAL,
                    activity_price_correlation REAL,
                    volume_transaction_correlation REAL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_blocks_number ON astar_market_enhanced_blocks(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_blocks_timestamp ON astar_market_enhanced_blocks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_tx_hash ON astar_market_enhanced_transactions(transaction_hash)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_tx_block ON astar_market_enhanced_transactions(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_market_timestamp ON astar_market_enhanced_market_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_combined_timestamp ON astar_market_enhanced_combined(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_combined_block ON astar_market_enhanced_combined(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_market_enhanced_metrics_timestamp ON astar_market_enhanced_metrics(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Market-enhanced database setup completed")
    
    async def __aenter__(self):
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # Initialize CoinGecko integration
        await self.coingecko.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.coingecko.__aexit__(exc_type, exc_val, exc_tb)
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make RPC call to Astar endpoint with rate limiting"""
        if params is None:
            params = []
        
        async with self.semaphore:
            await asyncio.sleep(self.rate_limit_delay)
            
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
            
            cursor.executemany('''
                INSERT OR REPLACE INTO astar_market_enhanced_blocks 
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
            
            cursor.executemany('''
                INSERT OR REPLACE INTO astar_market_enhanced_transactions 
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
    
    def save_market_data(self, market_data: Dict):
        """Save market data to database"""
        if not market_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO astar_market_enhanced_market_data 
                (timestamp, price_usd, market_cap_usd, volume_24h_usd, price_change_24h,
                 price_change_7d, price_change_14d, price_change_30d, price_volatility_24h,
                 price_volatility, price_momentum, volume_trend, market_cap_rank, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.get("timestamp"),
                market_data.get("price_usd", 0.0),
                market_data.get("market_cap_usd", 0.0),
                market_data.get("volume_24h_usd", 0.0),
                market_data.get("price_change_24h", 0.0),
                market_data.get("price_change_7d", 0.0),
                market_data.get("price_change_14d", 0.0),
                market_data.get("price_change_30d", 0.0),
                market_data.get("price_volatility_24h", 0.0),
                market_data.get("price_volatility", 0.0),
                market_data.get("price_momentum", 0.0),
                market_data.get("volume_trend", 0.0),
                market_data.get("market_cap_rank", 0),
                market_data.get("data_source", "coingecko")
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_combined_data_batch(self, combined_data_list: List[Dict]):
        """Save combined network and market data in batch"""
        if not combined_data_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            combined_values = []
            for data in combined_data_list:
                combined_values.append((
                    data.get('timestamp'),
                    data.get('block_number'),
                    data.get('transaction_count'),
                    data.get('gas_used'),
                    data.get('gas_limit'),
                    data.get('gas_utilization'),
                    data.get('network_activity'),
                    data.get('block_efficiency'),
                    data.get('price_usd', 0.0),
                    data.get('market_cap_usd', 0.0),
                    data.get('volume_24h_usd', 0.0),
                    data.get('price_change_24h', 0.0),
                    data.get('price_change_7d', 0.0),
                    data.get('price_volatility_24h', 0.0),
                    data.get('price_volatility', 0.0),
                    data.get('price_momentum', 0.0),
                    data.get('volume_trend', 0.0),
                    data.get('network_health', 0.0),
                    data.get('defi_activity', 0.0),
                    data.get('contract_activity', 0.0),
                    data.get('market_sentiment', 0.0),
                    data.get('correlation_network_price', 0.0),
                    data.get('hour'),
                    data.get('day_of_week'),
                    data.get('is_weekend'),
                    data.get('tx_count_24h', 0.0),
                    data.get('gas_used_24h', 0.0),
                    data.get('gas_util_24h', 0.0),
                    data.get('tx_count_7d', 0.0),
                    data.get('gas_used_7d', 0.0),
                    data.get('network_activity_7d', 0.0)
                ))
            
            cursor.executemany('''
                INSERT OR REPLACE INTO astar_market_enhanced_combined 
                (timestamp, block_number, transaction_count, gas_used, gas_limit, gas_utilization,
                 network_activity, block_efficiency, price_usd, market_cap_usd, volume_24h_usd,
                 price_change_24h, price_change_7d, price_volatility_24h, price_volatility,
                 price_momentum, volume_trend, network_health, defi_activity, contract_activity,
                 market_sentiment, correlation_network_price, hour, day_of_week, is_weekend,
                 tx_count_24h, gas_used_24h, gas_util_24h, tx_count_7d, gas_used_7d, network_activity_7d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', combined_values)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving combined data batch: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def collect_block_batch(self, block_numbers: List[int]) -> List[Dict]:
        """Collect data for a batch of blocks concurrently"""
        tasks = []
        for block_num in block_numbers:
            task = self.get_block_data(block_num)
            tasks.append(task)
        
        block_results = await asyncio.gather(*tasks, return_exceptions=True)
        
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
        
        tx_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_transactions = []
        for i, result in enumerate(tx_results):
            if isinstance(result, dict) and result:
                valid_transactions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting transaction {tx_hashes[i]}: {result}")
        
        return valid_transactions
    
    def calculate_combined_metrics(self, network_data: List[Dict], market_data: Dict) -> List[Dict]:
        """Calculate combined metrics for a batch of network data"""
        if not network_data or not market_data:
            return []
        
        combined_data = []
        
        for block_data in network_data:
            # Network metrics
            gas_utilization = block_data['gas_used'] / block_data['gas_limit'] if block_data['gas_limit'] > 0 else 0
            block_efficiency = block_data['transaction_count'] / block_data['gas_limit'] * 1000000 if block_data['gas_limit'] > 0 else 0
            network_activity = block_data['gas_used'] * block_data['transaction_count']
            
            # Time features
            timestamp = block_data['timestamp']
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = day_of_week >= 5
            
            # Combined metrics
            network_health = min(1.0, (gas_utilization * 10 + min(1.0, block_data['transaction_count'] / 100) + min(1.0, network_activity / 1000000)) / 3)
            defi_activity = block_data['gas_used'] * block_data['transaction_count']
            contract_activity = min(1.0, block_data['transaction_count'] / 50)
            
            # Market sentiment
            price_change_24h = market_data.get('price_change_24h', 0)
            price_change_7d = market_data.get('price_change_7d', 0)
            market_sentiment = (price_change_24h + price_change_7d) / 2 / 100
            
            combined_data.append({
                'timestamp': timestamp,
                'block_number': block_data['block_number'],
                'transaction_count': block_data['transaction_count'],
                'gas_used': block_data['gas_used'],
                'gas_limit': block_data['gas_limit'],
                'gas_utilization': gas_utilization,
                'network_activity': network_activity,
                'block_efficiency': block_efficiency,
                'price_usd': market_data.get('price_usd', 0.0),
                'market_cap_usd': market_data.get('market_cap_usd', 0.0),
                'volume_24h_usd': market_data.get('volume_24h_usd', 0.0),
                'price_change_24h': price_change_24h,
                'price_change_7d': price_change_7d,
                'price_volatility_24h': market_data.get('price_volatility_24h', 0.0),
                'price_volatility': market_data.get('price_volatility', 0.0),
                'price_momentum': market_data.get('price_momentum', 0.0),
                'volume_trend': market_data.get('volume_trend', 0.0),
                'network_health': network_health,
                'defi_activity': defi_activity,
                'contract_activity': contract_activity,
                'market_sentiment': market_sentiment,
                'correlation_network_price': 0.5,  # Placeholder
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'tx_count_24h': 0.0,  # Would need historical calculation
                'gas_used_24h': 0.0,  # Would need historical calculation
                'gas_util_24h': 0.0,  # Would need historical calculation
                'tx_count_7d': 0.0,   # Would need historical calculation
                'gas_used_7d': 0.0,   # Would need historical calculation
                'network_activity_7d': 0.0  # Would need historical calculation
            })
        
        return combined_data
    
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
                       f"({progress_percent:.1f}%) - Rate: {rate:.1f} blocks/sec - ETA: {eta/60:.1f} min "
                       f"- Market updates: {self.market_data_updates}")
    
    async def collect_historical_blocks_market_enhanced(self, start_block: int, end_block: int):
        """Collect historical block data with market data integration"""
        logger.info(f"🚀 Starting market-enhanced multi-threaded collection from block {start_block} to {end_block}")
        
        self.total_blocks = end_block - start_block + 1
        self.collected_blocks = 0
        self.start_time = time.time()
        self.market_data_updates = 0
        
        # Get initial market data
        market_data = await self.coingecko.get_comprehensive_market_data()
        if market_data:
            self.save_market_data(market_data)
            self.market_data_updates += 1
            logger.success(f"Initial market data: ${market_data.get('price_usd', 0):.4f}")
        
        last_market_update = time.time()
        
        # Create block number ranges for batch processing
        block_ranges = []
        for i in range(start_block, end_block + 1, self.batch_size):
            end_range = min(i + self.batch_size - 1, end_block)
            block_ranges.append(list(range(i, end_range + 1)))
        
        logger.info(f"Processing {len(block_ranges)} batches of {self.batch_size} blocks each")
        
        # Process batches concurrently
        for i, block_batch in enumerate(block_ranges):
            try:
                # Update market data periodically
                current_time = time.time()
                if current_time - last_market_update > self.market_data_interval:
                    market_data = await self.coingecko.get_comprehensive_market_data()
                    if market_data:
                        self.save_market_data(market_data)
                        self.market_data_updates += 1
                        last_market_update = current_time
                        logger.info(f"Updated market data: ${market_data.get('price_usd', 0):.4f}")
                
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
                    
                    # Calculate and save combined data
                    if market_data:
                        combined_data = self.calculate_combined_metrics(blocks_data, market_data)
                        if combined_data:
                            self.save_combined_data_batch(combined_data)
                    
                    # Update progress
                    self.update_progress(len(blocks_data))
                
                # Log batch completion
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(block_ranges)} batches")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
        
        logger.success(f"🎉 Market-enhanced collection completed! Collected {self.collected_blocks} blocks with {self.market_data_updates} market updates")
    
    async def collect_comprehensive_market_enhanced_data(self, days_back: int = 7):
        """Collect comprehensive Astar data with market integration"""
        logger.info(f"🚀 Starting market-enhanced Astar data collection for {days_back} days")
        
        try:
            # Get current block
            current_block = await self.get_current_block_number()
            logger.info(f"Current Astar block: {current_block}")
            
            # Calculate start block
            blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
            start_block = max(1, current_block - (days_back * blocks_per_day))
            
            logger.info(f"Collecting data from block {start_block} to {current_block}")
            logger.info(f"Total blocks to collect: {current_block - start_block + 1}")
            logger.info(f"Using {self.max_workers} concurrent workers with market data integration")
            
            # Collect historical blocks with market data
            await self.collect_historical_blocks_market_enhanced(start_block, current_block)
            
            # Generate summary
            await self.generate_market_enhanced_summary()
            
            logger.success("🎉 Market-enhanced Astar data collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in market-enhanced data collection: {e}")
            raise
    
    async def generate_market_enhanced_summary(self):
        """Generate summary of market-enhanced data collection"""
        logger.info("Generating market-enhanced data summary...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_blocks")
            block_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_transactions")
            tx_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_market_data")
            market_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_combined")
            combined_count = cursor.fetchone()[0]
            
            # Get block range
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_market_enhanced_blocks")
            block_range = cursor.fetchone()
            
            # Get recent market data
            cursor.execute("SELECT AVG(price_usd), AVG(market_cap_usd), AVG(volume_24h_usd) FROM astar_market_enhanced_combined WHERE price_usd > 0")
            market_stats = cursor.fetchone()
            
            summary = {
                "collection_timestamp": datetime.utcnow().isoformat(),
                "network": self.network_info,
                "collection_method": "market_enhanced_multi_threaded",
                "max_workers": self.max_workers,
                "market_data_updates": self.market_data_updates,
                "data_summary": {
                    "blocks_collected": block_count,
                    "transactions_collected": tx_count,
                    "market_data_points": market_count,
                    "combined_data_points": combined_count,
                    "block_range": {
                        "start": block_range[0],
                        "end": block_range[1]
                    },
                    "market_statistics": {
                        "avg_price_usd": market_stats[0] or 0.0,
                        "avg_market_cap_usd": market_stats[1] or 0.0,
                        "avg_volume_24h_usd": market_stats[2] or 0.0
                    }
                }
            }
            
            # Save summary
            with open('astar_market_enhanced_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Market-enhanced data summary generated:")
            logger.info(f"  Blocks: {block_count}")
            logger.info(f"  Transactions: {tx_count}")
            logger.info(f"  Market data points: {market_count}")
            logger.info(f"  Combined data points: {combined_count}")
            logger.info(f"  Market updates: {self.market_data_updates}")
            logger.info(f"  Block range: {block_range[0]:,} - {block_range[1]:,}")
            if market_stats[0]:
                logger.info(f"  Avg price: ${market_stats[0]:.4f}")
                logger.info(f"  Avg market cap: ${market_stats[1]:,.0f}")
                logger.info(f"  Avg volume: ${market_stats[2]:,.0f}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        finally:
            conn.close()

async def main():
    """Main function"""
    # Initialize market-enhanced multi-threaded collector with 15 workers
    async with AstarMarketEnhancedMultiThreadedCollector(max_workers=15) as collector:
        # Collect comprehensive data for last 7 days with market data
        await collector.collect_comprehensive_market_enhanced_data(days_back=7)

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run the market-enhanced collector
    asyncio.run(main())
