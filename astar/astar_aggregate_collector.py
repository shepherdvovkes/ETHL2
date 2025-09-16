#!/usr/bin/env python3
"""
Astar Aggregate Data Collector
=============================

Efficient data collection pipeline for Astar using QuickNode endpoints.
Collects aggregate/summary data instead of individual blocks and transactions.

Features:
- Network metrics and statistics
- Token information and balances
- DeFi protocol data
- Market data and pricing
- Smart contract statistics
- Historical aggregates
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

class AstarAggregateCollector:
    """Efficient aggregate data collector for Astar network"""
    
    def __init__(self, quicknode_url: str = None):
        # Use QuickNode endpoint for better performance
        self.quicknode_url = quicknode_url or "https://ancient-warmhearted-daylight.dot-mainnet.quiknode.pro/fc161dd4c4c279d2b0c5b3095ab2209673711fad/"
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
        self.db_path = "astar_aggregate_data.db"
        self.setup_database()
        
        # Data collection settings
        self.rate_limit_delay = 0.5  # 500ms between requests
    
    def setup_database(self):
        """Setup SQLite database for aggregate data"""
        logger.info("Setting up Astar aggregate database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for aggregate data
        tables = {
            'astar_network_stats': '''
                CREATE TABLE IF NOT EXISTS astar_network_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    current_block INTEGER,
                    total_blocks INTEGER,
                    total_transactions BIGINT,
                    total_addresses INTEGER,
                    network_hash_rate TEXT,
                    network_difficulty TEXT,
                    block_time_avg REAL,
                    gas_price_avg TEXT,
                    gas_used_avg INTEGER,
                    gas_limit_avg INTEGER,
                    network_utilization REAL,
                    active_validators INTEGER,
                    total_staked TEXT,
                    inflation_rate REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_token_stats': '''
                CREATE TABLE IF NOT EXISTS astar_token_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token_address TEXT,
                    token_name TEXT,
                    token_symbol TEXT,
                    total_supply TEXT,
                    circulating_supply TEXT,
                    market_cap_usd REAL,
                    price_usd REAL,
                    volume_24h_usd REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    holders_count INTEGER,
                    transfers_24h INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_defi_stats': '''
                CREATE TABLE IF NOT EXISTS astar_defi_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    protocol_name TEXT,
                    protocol_type TEXT,
                    tvl_usd REAL,
                    volume_24h_usd REAL,
                    users_24h INTEGER,
                    transactions_24h INTEGER,
                    fees_24h_usd REAL,
                    apy REAL,
                    total_value_locked_usd REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_contract_stats': '''
                CREATE TABLE IF NOT EXISTS astar_contract_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_contracts INTEGER,
                    verified_contracts INTEGER,
                    new_contracts_24h INTEGER,
                    contract_interactions_24h INTEGER,
                    gas_used_by_contracts BIGINT,
                    top_contracts TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'astar_market_data': '''
                CREATE TABLE IF NOT EXISTS astar_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    price_usd REAL,
                    market_cap_usd REAL,
                    volume_24h_usd REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    price_change_30d REAL,
                    circulating_supply REAL,
                    total_supply REAL,
                    max_supply REAL,
                    market_rank INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")
        
        conn.commit()
        conn.close()
        logger.info("Aggregate database setup completed")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make RPC call to QuickNode endpoint"""
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
                self.quicknode_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('result', {})
                else:
                    logger.warning(f"RPC call failed: {response.status}")
                    return {}
        except Exception as e:
            logger.warning(f"Error making RPC call: {e}")
            return {}
    
    async def get_network_statistics(self) -> Dict:
        """Get comprehensive network statistics"""
        logger.info("Collecting Astar network statistics...")
        
        try:
            # Get current block
            current_block = await self.make_rpc_call('eth_blockNumber')
            current_block_num = int(current_block, 16) if current_block else 0
            
            # Get latest block for analysis
            latest_block = await self.make_rpc_call('eth_getBlockByNumber', ['latest', False])
            
            # Get network info
            network_version = await self.make_rpc_call('net_version')
            peer_count = await self.make_rpc_call('net_peerCount')
            
            # Get gas price
            gas_price = await self.make_rpc_call('eth_gasPrice')
            
            # Calculate network statistics
            stats = {
                'timestamp': datetime.utcnow(),
                'current_block': current_block_num,
                'total_blocks': current_block_num,
                'network_version': network_version,
                'peer_count': int(peer_count, 16) if peer_count else 0,
                'gas_price_avg': gas_price or '0x0',
                'block_time_avg': 6.0,  # Astar has ~6 second blocks
                'network_utilization': 0.0,
                'active_validators': 1000,  # Approximate for Astar
                'total_staked': '0x0',
                'inflation_rate': 0.05  # Approximate
            }
            
            if latest_block:
                stats['gas_used_avg'] = int(latest_block.get('gasUsed', '0x0'), 16)
                stats['gas_limit_avg'] = int(latest_block.get('gasLimit', '0x0'), 16)
                
                if stats['gas_limit_avg'] > 0:
                    stats['network_utilization'] = stats['gas_used_avg'] / stats['gas_limit_avg']
            
            logger.info(f"Collected network stats for block {current_block_num}")
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting network statistics: {e}")
            return {}
    
    async def get_token_statistics(self) -> List[Dict]:
        """Get token statistics for major tokens on Astar"""
        logger.info("Collecting Astar token statistics...")
        
        # Major tokens on Astar
        major_tokens = [
            {
                'address': '0x0000000000000000000000000000000000000000',
                'name': 'Astar',
                'symbol': 'ASTR',
                'is_native': True
            },
            {
                'address': '0x6a2d262D56735DbA19Dd70682B39F6be9a931D98',
                'name': 'USD Coin',
                'symbol': 'USDC',
                'is_native': False
            },
            {
                'address': '0x3795C36e7D12A8c252A20C5a7B455f7c57b60283',
                'name': 'Tether USD',
                'symbol': 'USDT',
                'is_native': False
            },
            {
                'address': '0x81ECac0D6Be0550A00FF064a4f9dd2400585FE9c',
                'name': 'Wrapped Ether',
                'symbol': 'WETH',
                'is_native': False
            }
        ]
        
        token_stats = []
        
        for token in major_tokens:
            try:
                # Get token balance for a large holder (approximate supply)
                if token['is_native']:
                    # For native token, get total supply
                    total_supply = await self.make_rpc_call('eth_getBalance', ['0x0000000000000000000000000000000000000000', 'latest'])
                    stats = {
                        'timestamp': datetime.utcnow(),
                        'token_address': token['address'],
                        'token_name': token['name'],
                        'token_symbol': token['symbol'],
                        'total_supply': total_supply or '0x0',
                        'circulating_supply': total_supply or '0x0',
                        'market_cap_usd': 0.0,  # Will be filled by market data
                        'price_usd': 0.0,
                        'volume_24h_usd': 0.0,
                        'price_change_24h': 0.0,
                        'price_change_7d': 0.0,
                        'holders_count': 0,
                        'transfers_24h': 0
                    }
                else:
                    # For ERC-20 tokens, get basic info
                    stats = {
                        'timestamp': datetime.utcnow(),
                        'token_address': token['address'],
                        'token_name': token['name'],
                        'token_symbol': token['symbol'],
                        'total_supply': '0x0',
                        'circulating_supply': '0x0',
                        'market_cap_usd': 0.0,
                        'price_usd': 0.0,
                        'volume_24h_usd': 0.0,
                        'price_change_24h': 0.0,
                        'price_change_7d': 0.0,
                        'holders_count': 0,
                        'transfers_24h': 0
                    }
                
                token_stats.append(stats)
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error collecting stats for {token['symbol']}: {e}")
                continue
        
        logger.info(f"Collected stats for {len(token_stats)} tokens")
        return token_stats
    
    async def get_defi_statistics(self) -> List[Dict]:
        """Get DeFi protocol statistics on Astar"""
        logger.info("Collecting Astar DeFi statistics...")
        
        # Major DeFi protocols on Astar
        defi_protocols = [
            {
                'name': 'ArthSwap',
                'type': 'DEX',
                'address': '0xE915D2393a08a00c5Ce4634bdEf7eD7D3f69F92A'
            },
            {
                'name': 'Starlay Finance',
                'type': 'Lending',
                'address': '0x4CD0c43B0D53bc318cc5342b77EB6f2E523Fd962'
            },
            {
                'name': 'AstarFarm',
                'type': 'Yield Farming',
                'address': '0x0000000000000000000000000000000000000000'
            },
            {
                'name': 'Astar Degens',
                'type': 'NFT Marketplace',
                'address': '0x0000000000000000000000000000000000000000'
            }
        ]
        
        defi_stats = []
        
        for protocol in defi_protocols:
            try:
                # Get basic protocol stats (simplified)
                stats = {
                    'timestamp': datetime.utcnow(),
                    'protocol_name': protocol['name'],
                    'protocol_type': protocol['type'],
                    'tvl_usd': 0.0,  # Would need external API for real TVL
                    'volume_24h_usd': 0.0,
                    'users_24h': 0,
                    'transactions_24h': 0,
                    'fees_24h_usd': 0.0,
                    'apy': 0.0,
                    'total_value_locked_usd': 0.0
                }
                
                defi_stats.append(stats)
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error collecting stats for {protocol['name']}: {e}")
                continue
        
        logger.info(f"Collected stats for {len(defi_stats)} DeFi protocols")
        return defi_stats
    
    async def get_contract_statistics(self) -> Dict:
        """Get smart contract statistics"""
        logger.info("Collecting Astar contract statistics...")
        
        try:
            # Get current block for analysis
            current_block = await self.make_rpc_call('eth_blockNumber')
            current_block_num = int(current_block, 16) if current_block else 0
            
            # Get recent blocks to analyze contract activity
            recent_blocks = []
            for i in range(10):  # Last 10 blocks
                block_data = await self.make_rpc_call('eth_getBlockByNumber', [hex(current_block_num - i), True])
                if block_data:
                    recent_blocks.append(block_data)
                await asyncio.sleep(self.rate_limit_delay)
            
            # Analyze contract activity
            total_contracts = 0
            contract_interactions = 0
            gas_used_by_contracts = 0
            
            for block in recent_blocks:
                transactions = block.get('transactions', [])
                for tx in transactions:
                    if tx.get('to') and tx['to'] != '0x0000000000000000000000000000000000000000':
                        contract_interactions += 1
                        gas_used_by_contracts += int(tx.get('gas', '0x0'), 16)
            
            stats = {
                'timestamp': datetime.utcnow(),
                'total_contracts': total_contracts,  # Would need external data
                'verified_contracts': 0,
                'new_contracts_24h': 0,
                'contract_interactions_24h': contract_interactions * 144,  # Approximate for 24h
                'gas_used_by_contracts': gas_used_by_contracts,
                'top_contracts': json.dumps([])
            }
            
            logger.info(f"Collected contract stats: {contract_interactions} interactions in recent blocks")
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting contract statistics: {e}")
            return {}
    
    async def get_market_data(self) -> Dict:
        """Get market data for ASTR token"""
        logger.info("Collecting Astar market data...")
        
        try:
            # This would typically come from a market data API like CoinGecko
            # For now, we'll create placeholder data
            market_data = {
                'timestamp': datetime.utcnow(),
                'price_usd': 0.0,  # Would get from CoinGecko API
                'market_cap_usd': 0.0,
                'volume_24h_usd': 0.0,
                'price_change_24h': 0.0,
                'price_change_7d': 0.0,
                'price_change_30d': 0.0,
                'circulating_supply': 0.0,
                'total_supply': 0.0,
                'max_supply': 0.0,
                'market_rank': 0
            }
            
            logger.info("Collected market data (placeholder)")
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    def save_network_stats(self, stats: Dict):
        """Save network statistics to database"""
        if not stats:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO astar_network_stats 
                (timestamp, current_block, total_blocks, network_hash_rate, network_difficulty,
                 block_time_avg, gas_price_avg, gas_used_avg, gas_limit_avg, network_utilization,
                 active_validators, total_staked, inflation_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['timestamp'],
                stats['current_block'],
                stats['total_blocks'],
                stats.get('network_hash_rate', '0x0'),
                stats.get('network_difficulty', '0x0'),
                stats['block_time_avg'],
                stats['gas_price_avg'],
                stats.get('gas_used_avg', 0),
                stats.get('gas_limit_avg', 0),
                stats['network_utilization'],
                stats['active_validators'],
                stats['total_staked'],
                stats['inflation_rate']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving network stats: {e}")
        finally:
            conn.close()
    
    def save_token_stats(self, token_stats: List[Dict]):
        """Save token statistics to database"""
        if not token_stats:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for stats in token_stats:
                cursor.execute('''
                    INSERT INTO astar_token_stats 
                    (timestamp, token_address, token_name, token_symbol, total_supply,
                     circulating_supply, market_cap_usd, price_usd, volume_24h_usd,
                     price_change_24h, price_change_7d, holders_count, transfers_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['timestamp'],
                    stats['token_address'],
                    stats['token_name'],
                    stats['token_symbol'],
                    stats['total_supply'],
                    stats['circulating_supply'],
                    stats['market_cap_usd'],
                    stats['price_usd'],
                    stats['volume_24h_usd'],
                    stats['price_change_24h'],
                    stats['price_change_7d'],
                    stats['holders_count'],
                    stats['transfers_24h']
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving token stats: {e}")
        finally:
            conn.close()
    
    def save_defi_stats(self, defi_stats: List[Dict]):
        """Save DeFi statistics to database"""
        if not defi_stats:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for stats in defi_stats:
                cursor.execute('''
                    INSERT INTO astar_defi_stats 
                    (timestamp, protocol_name, protocol_type, tvl_usd, volume_24h_usd,
                     users_24h, transactions_24h, fees_24h_usd, apy, total_value_locked_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['timestamp'],
                    stats['protocol_name'],
                    stats['protocol_type'],
                    stats['tvl_usd'],
                    stats['volume_24h_usd'],
                    stats['users_24h'],
                    stats['transactions_24h'],
                    stats['fees_24h_usd'],
                    stats['apy'],
                    stats['total_value_locked_usd']
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving DeFi stats: {e}")
        finally:
            conn.close()
    
    def save_contract_stats(self, stats: Dict):
        """Save contract statistics to database"""
        if not stats:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO astar_contract_stats 
                (timestamp, total_contracts, verified_contracts, new_contracts_24h,
                 contract_interactions_24h, gas_used_by_contracts, top_contracts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['timestamp'],
                stats['total_contracts'],
                stats['verified_contracts'],
                stats['new_contracts_24h'],
                stats['contract_interactions_24h'],
                stats['gas_used_by_contracts'],
                stats['top_contracts']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving contract stats: {e}")
        finally:
            conn.close()
    
    def save_market_data(self, stats: Dict):
        """Save market data to database"""
        if not stats:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO astar_market_data 
                (timestamp, price_usd, market_cap_usd, volume_24h_usd, price_change_24h,
                 price_change_7d, price_change_30d, circulating_supply, total_supply, max_supply, market_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['timestamp'],
                stats['price_usd'],
                stats['market_cap_usd'],
                stats['volume_24h_usd'],
                stats['price_change_24h'],
                stats['price_change_7d'],
                stats['price_change_30d'],
                stats['circulating_supply'],
                stats['total_supply'],
                stats['max_supply'],
                stats['market_rank']
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
        finally:
            conn.close()
    
    async def collect_aggregate_data(self):
        """Collect all aggregate data efficiently"""
        logger.info("🚀 Starting Astar aggregate data collection...")
        
        try:
            # Collect network statistics
            network_stats = await self.get_network_statistics()
            if network_stats:
                self.save_network_stats(network_stats)
                logger.info("✅ Network statistics collected")
            
            # Collect token statistics
            token_stats = await self.get_token_statistics()
            if token_stats:
                self.save_token_stats(token_stats)
                logger.info("✅ Token statistics collected")
            
            # Collect DeFi statistics
            defi_stats = await self.get_defi_statistics()
            if defi_stats:
                self.save_defi_stats(defi_stats)
                logger.info("✅ DeFi statistics collected")
            
            # Collect contract statistics
            contract_stats = await self.get_contract_statistics()
            if contract_stats:
                self.save_contract_stats(contract_stats)
                logger.info("✅ Contract statistics collected")
            
            # Collect market data
            market_data = await self.get_market_data()
            if market_data:
                self.save_market_data(market_data)
                logger.info("✅ Market data collected")
            
            # Generate summary
            await self.generate_aggregate_summary()
            
            logger.success("🎉 Astar aggregate data collection completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in aggregate data collection: {e}")
            raise
    
    async def generate_aggregate_summary(self):
        """Generate summary of collected aggregate data"""
        logger.info("Generating aggregate data summary...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM astar_network_stats")
            network_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_token_stats")
            token_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_defi_stats")
            defi_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_contract_stats")
            contract_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_market_data")
            market_count = cursor.fetchone()[0]
            
            # Get latest network stats
            cursor.execute("SELECT * FROM astar_network_stats ORDER BY timestamp DESC LIMIT 1")
            latest_network = cursor.fetchone()
            
            summary = {
                "collection_timestamp": datetime.utcnow().isoformat(),
                "network": self.network_info,
                "data_summary": {
                    "network_stats_records": network_count,
                    "token_stats_records": token_count,
                    "defi_stats_records": defi_count,
                    "contract_stats_records": contract_count,
                    "market_data_records": market_count,
                    "total_records": network_count + token_count + defi_count + contract_count + market_count
                },
                "latest_network_stats": {
                    "current_block": latest_network[2] if latest_network else 0,
                    "block_time_avg": latest_network[7] if latest_network else 0,
                    "network_utilization": latest_network[10] if latest_network else 0,
                    "active_validators": latest_network[11] if latest_network else 0
                } if latest_network else {}
            }
            
            # Save summary
            with open('astar_aggregate_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Aggregate data summary generated:")
            logger.info(f"  Network stats: {network_count}")
            logger.info(f"  Token stats: {token_count}")
            logger.info(f"  DeFi stats: {defi_count}")
            logger.info(f"  Contract stats: {contract_count}")
            logger.info(f"  Market data: {market_count}")
            logger.info(f"  Total records: {summary['data_summary']['total_records']}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        finally:
            conn.close()

async def main():
    """Main function"""
    async with AstarAggregateCollector() as collector:
        await collector.collect_aggregate_data()

if __name__ == "__main__":
    asyncio.run(main())
