#!/usr/bin/env python3
"""
Astar Enhanced Market Data Collector
===================================

Enhanced data collector that combines network data with real-time market data
from CoinGecko API for comprehensive Astar analysis.

Features:
- Network data collection (blocks, transactions, metrics)
- Real-time market data integration (price, volume, market cap)
- Combined dataset for ML training
- Historical market data
- Data synchronization and validation
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
from astar_coingecko_integration import AstarCoinGeckoIntegration
warnings.filterwarnings('ignore')

class AstarEnhancedMarketCollector:
    """Enhanced collector combining network and market data"""
    
    def __init__(self, rpc_url: str = None, coingecko_api_key: str = None):
        # Network data collection
        self.rpc_url = rpc_url or "https://rpc.astar.network"
        self.session = None
        
        # Market data integration
        self.coingecko = AstarCoinGeckoIntegration(api_key=coingecko_api_key)
        
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
        self.db_path = "astar_enhanced_market_data.db"
        self.setup_database()
        
        # Data collection settings
        self.batch_size = 50
        self.rate_limit_delay = 0.1
        self.market_data_interval = 300  # Update market data every 5 minutes
    
    def setup_database(self):
        """Setup enhanced database with both network and market data"""
        logger.info("Setting up enhanced Astar database with market data...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Create tables
        tables = {
            'astar_enhanced_blocks': '''
                CREATE TABLE IF NOT EXISTS astar_enhanced_blocks (
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
                    miner TEXT,
                    extra_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_enhanced_transactions': '''
                CREATE TABLE IF NOT EXISTS astar_enhanced_transactions (
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
                    FOREIGN KEY (block_number) REFERENCES astar_enhanced_blocks (block_number)
                )
            ''',
            'astar_enhanced_market_data': '''
                CREATE TABLE IF NOT EXISTS astar_enhanced_market_data (
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
            'astar_enhanced_combined_data': '''
                CREATE TABLE IF NOT EXISTS astar_enhanced_combined_data (
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
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_enhanced_metrics': '''
                CREATE TABLE IF NOT EXISTS astar_enhanced_metrics (
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
            "CREATE INDEX IF NOT EXISTS idx_enhanced_blocks_number ON astar_enhanced_blocks(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_blocks_timestamp ON astar_enhanced_blocks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_tx_hash ON astar_enhanced_transactions(transaction_hash)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_tx_block ON astar_enhanced_transactions(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_market_timestamp ON astar_enhanced_market_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_combined_timestamp ON astar_enhanced_combined_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_metrics_timestamp ON astar_enhanced_metrics(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Enhanced database setup completed")
    
    async def __aenter__(self):
        # Create session for network data
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
            'miner': block_data.get('miner', ''),
            'extra_data': block_data.get('extraData', ''),
            'transactions': block_data.get('transactions', [])
        }
        
        return block_info
    
    async def get_network_metrics(self, block_data: Dict) -> Dict:
        """Calculate network metrics from block data"""
        if not block_data:
            return {}
        
        # Basic network metrics
        gas_utilization = block_data['gas_used'] / block_data['gas_limit'] if block_data['gas_limit'] > 0 else 0
        block_efficiency = block_data['transaction_count'] / block_data['gas_limit'] * 1000000 if block_data['gas_limit'] > 0 else 0
        network_activity = block_data['gas_used'] * block_data['transaction_count']
        
        # Time-based features
        timestamp = block_data['timestamp']
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        return {
            'block_number': block_data['block_number'],
            'timestamp': timestamp,
            'transaction_count': block_data['transaction_count'],
            'gas_used': block_data['gas_used'],
            'gas_limit': block_data['gas_limit'],
            'gas_utilization': gas_utilization,
            'network_activity': network_activity,
            'block_efficiency': block_efficiency,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend
        }
    
    async def get_market_data(self) -> Dict:
        """Get current market data from CoinGecko"""
        return await self.coingecko.get_comprehensive_market_data()
    
    def calculate_combined_metrics(self, network_data: Dict, market_data: Dict) -> Dict:
        """Calculate combined network and market metrics"""
        if not network_data or not market_data:
            return {}
        
        # Network health score (0-1)
        gas_utilization = network_data.get('gas_utilization', 0)
        transaction_count = network_data.get('transaction_count', 0)
        network_activity = network_data.get('network_activity', 0)
        
        # Normalize network health (higher is better)
        network_health = min(1.0, (gas_utilization * 10 + min(1.0, transaction_count / 100) + min(1.0, network_activity / 1000000)) / 3)
        
        # DeFi activity (simplified - based on gas usage and transaction count)
        defi_activity = network_data.get('gas_used', 0) * network_data.get('transaction_count', 0)
        
        # Contract activity (simplified - based on transaction count)
        contract_activity = min(1.0, network_data.get('transaction_count', 0) / 50)
        
        # Market sentiment (based on price changes)
        price_change_24h = market_data.get('price_change_24h', 0)
        price_change_7d = market_data.get('price_change_7d', 0)
        market_sentiment = (price_change_24h + price_change_7d) / 2 / 100  # Normalize to -1 to 1
        
        # Correlation metrics (simplified)
        correlation_network_price = 0.5  # Placeholder - would need historical analysis
        
        return {
            'network_health': network_health,
            'defi_activity': defi_activity,
            'contract_activity': contract_activity,
            'market_sentiment': market_sentiment,
            'correlation_network_price': correlation_network_price
        }
    
    def save_combined_data(self, network_data: Dict, market_data: Dict, combined_metrics: Dict):
        """Save combined network and market data"""
        if not network_data or not market_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO astar_enhanced_combined_data 
                (timestamp, block_number, transaction_count, gas_used, gas_limit, gas_utilization,
                 network_activity, block_efficiency, price_usd, market_cap_usd, volume_24h_usd,
                 price_change_24h, price_change_7d, price_volatility_24h, price_volatility,
                 price_momentum, volume_trend, network_health, defi_activity, contract_activity,
                 market_sentiment, correlation_network_price, hour, day_of_week, is_weekend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                network_data.get('timestamp'),
                network_data.get('block_number'),
                network_data.get('transaction_count'),
                network_data.get('gas_used'),
                network_data.get('gas_limit'),
                network_data.get('gas_utilization'),
                network_data.get('network_activity'),
                network_data.get('block_efficiency'),
                market_data.get('price_usd', 0.0),
                market_data.get('market_cap_usd', 0.0),
                market_data.get('volume_24h_usd', 0.0),
                market_data.get('price_change_24h', 0.0),
                market_data.get('price_change_7d', 0.0),
                market_data.get('price_volatility_24h', 0.0),
                market_data.get('price_volatility', 0.0),
                market_data.get('price_momentum', 0.0),
                market_data.get('volume_trend', 0.0),
                combined_metrics.get('network_health', 0.0),
                combined_metrics.get('defi_activity', 0.0),
                combined_metrics.get('contract_activity', 0.0),
                combined_metrics.get('market_sentiment', 0.0),
                combined_metrics.get('correlation_network_price', 0.0),
                network_data.get('hour'),
                network_data.get('day_of_week'),
                network_data.get('is_weekend')
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def collect_enhanced_data(self, num_blocks: int = 100):
        """Collect enhanced data combining network and market data"""
        logger.info(f"🚀 Starting enhanced Astar data collection for {num_blocks} blocks")
        
        try:
            # Get current block number
            current_block = await self.get_current_block_number()
            if current_block == 0:
                logger.error("Could not get current block number")
                return
            
            start_block = max(1, current_block - num_blocks + 1)
            
            # Get initial market data
            market_data = await self.get_market_data()
            if not market_data:
                logger.warning("Could not fetch market data, continuing with network data only")
            
            collected_blocks = 0
            market_data_updates = 0
            last_market_update = time.time()
            
            for block_num in range(start_block, current_block + 1):
                try:
                    # Get block data
                    block_data = await self.get_block_data(block_num)
                    if not block_data:
                        continue
                    
                    # Calculate network metrics
                    network_metrics = await self.get_network_metrics(block_data)
                    
                    # Update market data periodically
                    current_time = time.time()
                    if current_time - last_market_update > self.market_data_interval:
                        market_data = await self.get_market_data()
                        if market_data:
                            market_data_updates += 1
                            last_market_update = current_time
                    
                    # Calculate combined metrics
                    combined_metrics = self.calculate_combined_metrics(network_metrics, market_data)
                    
                    # Save combined data
                    self.save_combined_data(network_metrics, market_data, combined_metrics)
                    
                    collected_blocks += 1
                    
                    if collected_blocks % 10 == 0:
                        logger.info(f"Collected {collected_blocks}/{num_blocks} blocks, {market_data_updates} market updates")
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error collecting block {block_num}: {e}")
                    continue
            
            logger.success(f"🎉 Enhanced data collection completed!")
            logger.info(f"Collected {collected_blocks} blocks with {market_data_updates} market data updates")
            
            # Generate summary
            await self.generate_enhanced_summary(collected_blocks, market_data_updates)
            
        except Exception as e:
            logger.error(f"Error in enhanced data collection: {e}")
            raise
    
    async def generate_enhanced_summary(self, blocks_collected: int, market_updates: int):
        """Generate summary of enhanced data collection"""
        logger.info("Generating enhanced data summary...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM astar_enhanced_combined_data")
            combined_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_enhanced_market_data")
            market_count = cursor.fetchone()[0]
            
            # Get data ranges
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM astar_enhanced_combined_data")
            time_range = cursor.fetchone()
            
            # Get recent market data
            cursor.execute("SELECT AVG(price_usd), AVG(market_cap_usd), AVG(volume_24h_usd) FROM astar_enhanced_combined_data WHERE price_usd > 0")
            market_stats = cursor.fetchone()
            
            summary = {
                "collection_timestamp": datetime.utcnow().isoformat(),
                "network": self.network_info,
                "collection_type": "enhanced_with_market_data",
                "data_summary": {
                    "blocks_collected": blocks_collected,
                    "combined_data_points": combined_count,
                    "market_data_points": market_count,
                    "market_data_updates": market_updates,
                    "time_range": {
                        "start": time_range[0],
                        "end": time_range[1]
                    },
                    "market_statistics": {
                        "avg_price_usd": market_stats[0] or 0.0,
                        "avg_market_cap_usd": market_stats[1] or 0.0,
                        "avg_volume_24h_usd": market_stats[2] or 0.0
                    }
                }
            }
            
            # Save summary
            with open('astar_enhanced_market_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Enhanced data summary generated:")
            logger.info(f"  Blocks: {blocks_collected}")
            logger.info(f"  Combined data points: {combined_count}")
            logger.info(f"  Market data points: {market_count}")
            logger.info(f"  Market updates: {market_updates}")
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
    async with AstarEnhancedMarketCollector() as collector:
        # Collect enhanced data for last 100 blocks
        await collector.collect_enhanced_data(num_blocks=100)

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run the enhanced collector
    asyncio.run(main())
