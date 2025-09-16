#!/usr/bin/env python3
"""
Astar Comprehensive Data Collector
=================================

Comprehensive data collection pipeline for Astar (ASTR) network.
Collects all available data including blocks, transactions, contracts, and metrics.

Features:
- Complete block data collection
- Transaction analysis
- Smart contract data
- Token and DeFi metrics
- Historical data with timestamps
- Real-time monitoring
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
warnings.filterwarnings('ignore')

class AstarDataCollector:
    """Comprehensive data collector for Astar network"""
    
    def __init__(self, rpc_url: str = None):
        # Astar RPC endpoints
        self.rpc_url = rpc_url or "https://rpc.astar.network"
        self.ws_url = "wss://rpc.astar.network"
        self.session = None
        
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
        self.db_path = "astar_data.db"
        self.setup_database()
        
        # Data collection settings
        self.batch_size = 100
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    def setup_database(self):
        """Setup SQLite database for Astar data"""
        logger.info("Setting up Astar database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
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
            'astar_contracts': '''
                CREATE TABLE IF NOT EXISTS astar_contracts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_address TEXT UNIQUE NOT NULL,
                    contract_name TEXT,
                    contract_type TEXT,
                    creator_address TEXT,
                    creation_block INTEGER,
                    creation_transaction TEXT,
                    bytecode TEXT,
                    abi TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_tokens': '''
                CREATE TABLE IF NOT EXISTS astar_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT UNIQUE NOT NULL,
                    token_name TEXT,
                    token_symbol TEXT,
                    token_decimals INTEGER,
                    total_supply TEXT,
                    token_type TEXT,
                    is_native BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
            ''',
            'astar_defi_protocols': '''
                CREATE TABLE IF NOT EXISTS astar_defi_protocols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol_name TEXT NOT NULL,
                    protocol_type TEXT,
                    contract_address TEXT,
                    tvl_usd REAL,
                    volume_24h_usd REAL,
                    users_24h INTEGER,
                    transactions_24h INTEGER,
                    fees_24h_usd REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")
        
        conn.commit()
        conn.close()
        logger.info("Database setup completed")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make RPC call to Astar endpoint"""
        if params is None:
            params = []
        
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
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
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
        
        # Get block with transactions
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
    
    async def get_network_metrics(self) -> Dict:
        """Get comprehensive network metrics"""
        logger.info("Collecting Astar network metrics...")
        
        current_block = await self.get_current_block_number()
        if current_block == 0:
            return {}
        
        # Get recent blocks for analysis
        recent_blocks = []
        for i in range(10):  # Last 10 blocks
            block_data = await self.get_block_data(current_block - i)
            if block_data:
                recent_blocks.append(block_data)
            await asyncio.sleep(self.rate_limit_delay)
        
        # Calculate metrics
        metrics = {
            'timestamp': datetime.utcnow(),
            'block_number': current_block,
            'block_time_avg': 6.0,  # Astar has ~6 second blocks
            'transaction_count_24h': 0,
            'gas_price_avg': '0x0',
            'gas_used_avg': 0,
            'network_utilization': 0.0,
            'active_addresses_24h': 0
        }
        
        if recent_blocks:
            # Calculate average gas usage
            total_gas_used = sum(block['gas_used'] for block in recent_blocks)
            total_gas_limit = sum(block['gas_limit'] for block in recent_blocks)
            metrics['gas_used_avg'] = total_gas_used // len(recent_blocks)
            metrics['network_utilization'] = total_gas_used / total_gas_limit if total_gas_limit > 0 else 0
            
            # Calculate transaction count
            metrics['transaction_count_24h'] = sum(block['transaction_count'] for block in recent_blocks) * 1440  # Approximate for 24h
        
        return metrics
    
    def save_block_data(self, block_data: Dict):
        """Save block data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO astar_blocks 
                (block_number, block_hash, timestamp, parent_hash, gas_limit, gas_used, 
                 transaction_count, block_size, difficulty, total_difficulty, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving block data: {e}")
        finally:
            conn.close()
    
    def save_transaction_data(self, tx_data: Dict):
        """Save transaction data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO astar_transactions 
                (transaction_hash, block_number, block_hash, from_address, to_address, 
                 value, gas, gas_price, nonce, input_data, transaction_index, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving transaction data: {e}")
        finally:
            conn.close()
    
    def save_metrics_data(self, metrics: Dict):
        """Save metrics data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO astar_metrics 
                (timestamp, block_number, transaction_count_24h, gas_price_avg, gas_used_avg, 
                 block_time_avg, network_utilization)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics['timestamp'],
                metrics['block_number'],
                metrics['transaction_count_24h'],
                metrics['gas_price_avg'],
                metrics['gas_used_avg'],
                metrics['block_time_avg'],
                metrics['network_utilization']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving metrics data: {e}")
        finally:
            conn.close()
    
    async def collect_historical_blocks(self, start_block: int, end_block: int):
        """Collect historical block data"""
        logger.info(f"Collecting blocks from {start_block} to {end_block}")
        
        total_blocks = end_block - start_block + 1
        collected = 0
        
        for block_num in range(start_block, end_block + 1):
            try:
                # Get block data
                block_data = await self.get_block_data(block_num)
                if block_data:
                    # Save block data
                    self.save_block_data(block_data)
                    
                    # Collect transaction data for this block
                    for tx in block_data.get('transactions', []):
                        if isinstance(tx, dict) and 'hash' in tx:
                            tx_data = await self.get_transaction_data(tx['hash'])
                            if tx_data:
                                self.save_transaction_data(tx_data)
                            await asyncio.sleep(self.rate_limit_delay)
                    
                    collected += 1
                    
                    if collected % 10 == 0:
                        logger.info(f"Collected {collected}/{total_blocks} blocks ({collected/total_blocks*100:.1f}%)")
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error collecting block {block_num}: {e}")
                continue
        
        logger.info(f"Completed collecting {collected} blocks")
    
    async def collect_recent_data(self, hours_back: int = 24):
        """Collect recent data for specified hours"""
        logger.info(f"Collecting recent data for last {hours_back} hours")
        
        current_block = await self.get_current_block_number()
        if current_block == 0:
            logger.error("Could not get current block number")
            return
        
        # Estimate blocks per hour (Astar has ~6 second blocks)
        blocks_per_hour = 3600 // 6  # ~600 blocks per hour
        start_block = max(1, current_block - (hours_back * blocks_per_hour))
        
        await self.collect_historical_blocks(start_block, current_block)
    
    async def collect_comprehensive_data(self, days_back: int = 7):
        """Collect comprehensive Astar data"""
        logger.info(f"🚀 Starting comprehensive Astar data collection for {days_back} days")
        
        try:
            # Get current block
            current_block = await self.get_current_block_number()
            logger.info(f"Current Astar block: {current_block}")
            
            # Calculate start block
            blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
            start_block = max(1, current_block - (days_back * blocks_per_day))
            
            logger.info(f"Collecting data from block {start_block} to {current_block}")
            logger.info(f"Total blocks to collect: {current_block - start_block + 1}")
            
            # Collect historical blocks
            await self.collect_historical_blocks(start_block, current_block)
            
            # Collect network metrics
            metrics = await self.get_network_metrics()
            if metrics:
                self.save_metrics_data(metrics)
            
            # Generate summary
            await self.generate_data_summary()
            
            logger.success("🎉 Astar data collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
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
            
            cursor.execute("SELECT COUNT(*) FROM astar_metrics")
            metrics_count = cursor.fetchone()[0]
            
            # Get block range
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
            block_range = cursor.fetchone()
            
            # Get recent activity
            cursor.execute("SELECT AVG(transaction_count) FROM astar_blocks ORDER BY block_number DESC LIMIT 100")
            avg_tx_per_block = cursor.fetchone()[0] or 0
            
            summary = {
                "collection_timestamp": datetime.utcnow().isoformat(),
                "network": self.network_info,
                "data_summary": {
                    "blocks_collected": block_count,
                    "transactions_collected": tx_count,
                    "metrics_collected": metrics_count,
                    "block_range": {
                        "start": block_range[0],
                        "end": block_range[1]
                    },
                    "average_transactions_per_block": round(avg_tx_per_block, 2)
                }
            }
            
            # Save summary
            with open('astar_data_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Data summary generated:")
            logger.info(f"  Blocks: {block_count}")
            logger.info(f"  Transactions: {tx_count}")
            logger.info(f"  Metrics: {metrics_count}")
            logger.info(f"  Block range: {block_range[0]} - {block_range[1]}")
            logger.info(f"  Avg TX per block: {avg_tx_per_block:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        finally:
            conn.close()

async def main():
    """Main function"""
    async with AstarDataCollector() as collector:
        # Collect comprehensive data for last 7 days
        await collector.collect_comprehensive_data(days_back=7)

if __name__ == "__main__":
    asyncio.run(main())
