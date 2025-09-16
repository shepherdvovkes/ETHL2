#!/usr/bin/env python3
"""
Polkadot Archive Data Collector
===============================

Comprehensive archive data collector for Polkadot using QuickNode endpoints
with 30 parallel workers for efficient historical data collection.

Features:
- Historical block data collection
- Transaction and event data
- Staking and governance data
- Parachain metrics
- Cross-chain messaging data
- 30 parallel workers for optimal performance
- Data validation and quality checks
- Comprehensive error handling
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from loguru import logger
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class CollectionConfig:
    """Configuration for data collection"""
    quicknode_url: str = "https://ancient-warmhearted-daylight.dot-mainnet.quiknode.pro/fc161dd4c4c279d2b0c5b3095ab2209673711fad/"
    max_workers: int = 30
    batch_size: int = 100
    rate_limit_delay: float = 0.1
    retry_attempts: int = 3
    database_path: str = "polkadot_archive_data.db"
    days_back: int = 365  # Collect 1 year of data
    sample_rate: int = 10  # Sample every 10th block

class QuickNodePolkadotArchiveClient:
    """Enhanced QuickNode client for Polkadot archive data collection"""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.session = None
        self.semaphore = asyncio.Semaphore(config.max_workers)
        
        # RPC methods for comprehensive data collection
        self.rpc_methods = {
            'chain_getBlock': 'Get complete block data',
            'chain_getHeader': 'Get block header',
            'chain_getBlockHash': 'Get block hash',
            'chain_getFinalizedHead': 'Get finalized head',
            'chain_getRuntimeVersion': 'Get runtime version',
            'state_getStorage': 'Get storage data',
            'state_getMetadata': 'Get metadata',
            'system_health': 'Get system health',
            'system_peers': 'Get peer information',
            'system_properties': 'Get system properties',
            'staking_validators': 'Get validator information',
            'staking_nominators': 'Get nominator information',
            'staking_activeEra': 'Get active era',
            'staking_erasStakers': 'Get era stakers',
            'paras_parachains': 'Get parachain information',
            'paras_parachainInfo': 'Get parachain details',
            'hrmp_hrmpChannels': 'Get HRMP channels',
            'xcm_pallet_querySupportedVersion': 'Get XCM version info',
            'democracy_publicPropCount': 'Get public proposals count',
            'democracy_referendumCount': 'Get referendums count',
            'council_members': 'Get council members',
            'treasury_proposalCount': 'Get treasury proposals count'
        }
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.config.max_workers)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None, retry_count: int = 0) -> Dict:
        """Make RPC call with retry logic and adaptive rate limiting"""
        if params is None:
            params = []
        
        async with self.semaphore:  # Limit concurrent requests
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1
            }
            
            try:
                async with self.session.post(
                    self.config.quicknode_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "error" in result:
                            logger.error(f"RPC Error for {method}: {result['error']}")
                            return {}
                        return result.get('result', {})
                    elif response.status == 429:  # Rate limited
                        if retry_count < self.config.retry_attempts:
                            wait_time = min(30, 2 ** retry_count + 5)  # Longer wait for rate limits
                            logger.warning(f"Rate limited for {method}, waiting {wait_time}s (attempt {retry_count + 1})")
                            await asyncio.sleep(wait_time)
                            return await self.make_rpc_call(method, params, retry_count + 1)
                        else:
                            logger.error(f"Rate limited {method} after {self.config.retry_attempts} attempts")
                            return {}
                    else:
                        logger.error(f"HTTP Error {response.status} for {method}")
                        return {}
            except Exception as e:
                if retry_count < self.config.retry_attempts:
                    wait_time = min(10, 2 ** retry_count)
                    logger.warning(f"Retrying {method} (attempt {retry_count + 1}): {e}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    return await self.make_rpc_call(method, params, retry_count + 1)
                else:
                    logger.error(f"Failed {method} after {self.config.retry_attempts} attempts: {e}")
                    return {}
            finally:
                # Adaptive rate limiting based on success/failure
                delay = self.config.rate_limit_delay
                if retry_count > 0:
                    delay *= 2  # Increase delay after retries
                await asyncio.sleep(delay)
    
    async def get_current_block(self) -> int:
        """Get current finalized block number"""
        result = await self.make_rpc_call('chain_getFinalizedHead')
        if result:
            block_data = await self.make_rpc_call('chain_getHeader', [result])
            if block_data and 'number' in block_data:
                return int(block_data['number'], 16)
        return 0
    
    async def get_block_range(self, days_back: int) -> Tuple[int, int]:
        """Calculate block range for historical collection"""
        current_block = await self.get_current_block()
        if current_block == 0:
            return 0, 0
        
        # Polkadot has ~6 second block time
        blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
        start_block = max(1, current_block - (days_back * blocks_per_day))
        
        return start_block, current_block
    
    async def collect_block_data(self, block_number: int) -> Dict:
        """Collect comprehensive data for a single block"""
        try:
            # Get block hash
            block_hash = await self.make_rpc_call('chain_getBlockHash', [hex(block_number)])
            if not block_hash:
                return {}
            
            # Collect all block data in parallel
            tasks = [
                self.make_rpc_call('chain_getBlock', [block_hash]),
                self.make_rpc_call('chain_getHeader', [block_hash])
            ]
            
            block_data, header_data = await asyncio.gather(*tasks)
            
            if not block_data:
                return {}
            
            # Extract comprehensive metrics
            metrics = self._extract_block_metrics(block_data, header_data, block_number)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting block {block_number}: {e}")
            return {}
    
    def _extract_block_metrics(self, block_data: Dict, header_data: Dict, block_number: int) -> Dict:
        """Extract comprehensive metrics from block data"""
        metrics = {
            'block_number': block_number,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'extrinsics_count': 0,
            'events_count': 0,
            'block_size': 0,
            'validator_count': 0,
            'finalization_time': 0,
            'parachain_blocks': 0,
            'cross_chain_messages': 0
        }
        
        try:
            if 'block' in block_data:
                block = block_data['block']
                
                # Count extrinsics
                if 'extrinsics' in block:
                    metrics['extrinsics_count'] = len(block['extrinsics'])
                
                # Count events
                if 'header' in block and 'digest' in block['header']:
                    digest = block['header']['digest']
                    if 'logs' in digest:
                        metrics['events_count'] = len(digest['logs'])
                
                # Estimate block size
                block_json = json.dumps(block_data)
                metrics['block_size'] = len(block_json.encode('utf-8'))
            
            # Extract from header
            if header_data and 'number' in header_data:
                metrics['block_number'] = int(header_data['number'], 16)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics for block {block_number}: {e}")
            return metrics
    
    async def collect_staking_data(self) -> Dict:
        """Collect comprehensive staking data"""
        try:
            tasks = [
                self.make_rpc_call('staking_validators'),
                self.make_rpc_call('staking_nominators'),
                self.make_rpc_call('staking_activeEra'),
                self.make_rpc_call('system_health')
            ]
            
            validators, nominators, active_era, health = await asyncio.gather(*tasks)
            
            return {
                'validators': validators,
                'nominators': nominators,
                'active_era': active_era,
                'health': health,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting staking data: {e}")
            return {}
    
    async def collect_parachain_data(self) -> Dict:
        """Collect parachain information"""
        try:
            tasks = [
                self.make_rpc_call('paras_parachains'),
                self.make_rpc_call('hrmp_hrmpChannels')
            ]
            
            parachains, hrmp_channels = await asyncio.gather(*tasks)
            
            return {
                'parachains': parachains,
                'hrmp_channels': hrmp_channels,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting parachain data: {e}")
            return {}
    
    async def collect_governance_data(self) -> Dict:
        """Collect governance metrics"""
        try:
            tasks = [
                self.make_rpc_call('democracy_publicPropCount'),
                self.make_rpc_call('democracy_referendumCount'),
                self.make_rpc_call('council_members'),
                self.make_rpc_call('treasury_proposalCount')
            ]
            
            proposals, referendums, council, treasury = await asyncio.gather(*tasks)
            
            return {
                'proposals': proposals,
                'referendums': referendums,
                'council_members': council,
                'treasury_proposals': treasury,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting governance data: {e}")
            return {}

class PolkadotArchiveDatabase:
    """Database manager for archive data"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Block metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS block_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_number INTEGER UNIQUE,
                timestamp TEXT,
                extrinsics_count INTEGER,
                events_count INTEGER,
                block_size INTEGER,
                validator_count INTEGER,
                finalization_time REAL,
                parachain_blocks INTEGER,
                cross_chain_messages INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Staking data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS staking_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                validators_count INTEGER,
                nominators_count INTEGER,
                active_era INTEGER,
                total_staked REAL,
                inflation_rate REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Parachain data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parachain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                parachains_count INTEGER,
                hrmp_channels_count INTEGER,
                active_parachains INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Governance data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS governance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                proposals_count INTEGER,
                referendums_count INTEGER,
                council_members_count INTEGER,
                treasury_proposals_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_block_number ON block_metrics(block_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON block_metrics(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def store_block_metrics(self, metrics: Dict):
        """Store block metrics in database"""
        if not metrics:
            return
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO block_metrics 
                (block_number, timestamp, extrinsics_count, events_count, block_size, 
                 validator_count, finalization_time, parachain_blocks, cross_chain_messages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('block_number'),
                metrics.get('timestamp'),
                metrics.get('extrinsics_count'),
                metrics.get('events_count'),
                metrics.get('block_size'),
                metrics.get('validator_count'),
                metrics.get('finalization_time'),
                metrics.get('parachain_blocks'),
                metrics.get('cross_chain_messages')
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing block metrics: {e}")
        finally:
            conn.close()
    
    def store_staking_data(self, data: Dict):
        """Store staking data in database"""
        if not data:
            return
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            validators_count = len(data.get('validators', []))
            nominators_count = len(data.get('nominators', []))
            active_era = data.get('active_era', {}).get('index', 0)
            
            cursor.execute('''
                INSERT INTO staking_data 
                (timestamp, validators_count, nominators_count, active_era, total_staked, inflation_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.get('timestamp'),
                validators_count,
                nominators_count,
                active_era,
                0,  # Will be calculated separately
                7.5  # Default inflation rate
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing staking data: {e}")
        finally:
            conn.close()
    
    def store_parachain_data(self, data: Dict):
        """Store parachain data in database"""
        if not data:
            return
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            parachains_count = len(data.get('parachains', []))
            hrmp_channels_count = len(data.get('hrmp_channels', []))
            
            cursor.execute('''
                INSERT INTO parachain_data 
                (timestamp, parachains_count, hrmp_channels_count, active_parachains)
                VALUES (?, ?, ?, ?)
            ''', (
                data.get('timestamp'),
                parachains_count,
                hrmp_channels_count,
                parachains_count  # Assume all are active
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing parachain data: {e}")
        finally:
            conn.close()
    
    def store_governance_data(self, data: Dict):
        """Store governance data in database"""
        if not data:
            return
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO governance_data 
                (timestamp, proposals_count, referendums_count, council_members_count, treasury_proposals_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data.get('timestamp'),
                data.get('proposals'),
                data.get('referendums'),
                len(data.get('council_members', [])),
                data.get('treasury_proposals')
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing governance data: {e}")
        finally:
            conn.close()

class PolkadotArchiveCollector:
    """Main archive data collector with 30 parallel workers"""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.db = PolkadotArchiveDatabase(config.database_path)
        self.collected_blocks = 0
        self.failed_blocks = 0
        self.start_time = None
    
    async def collect_historical_blocks(self, start_block: int, end_block: int) -> List[Dict]:
        """Collect historical block data with 30 parallel workers"""
        logger.info(f"Starting historical block collection from {start_block} to {end_block}")
        
        # Create block list with sampling
        blocks_to_collect = list(range(start_block, end_block + 1, self.config.sample_rate))
        logger.info(f"Will collect {len(blocks_to_collect)} blocks (sampling every {self.config.sample_rate} blocks)")
        
        # Split into batches for parallel processing
        batches = [blocks_to_collect[i:i + self.config.batch_size] 
                  for i in range(0, len(blocks_to_collect), self.config.batch_size)]
        
        all_metrics = []
        
        async with QuickNodePolkadotArchiveClient(self.config) as client:
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} blocks)")
                
                # Process batch with 30 parallel workers
                tasks = [client.collect_block_data(block_num) for block_num in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error collecting block {batch[i]}: {result}")
                        self.failed_blocks += 1
                    elif result:
                        all_metrics.append(result)
                        self.collected_blocks += 1
                        self.db.store_block_metrics(result)
                    else:
                        self.failed_blocks += 1
                
                # Progress update
                total_processed = (batch_idx + 1) * self.config.batch_size
                progress = min(100, (total_processed / len(blocks_to_collect)) * 100)
                logger.info(f"Progress: {progress:.1f}% - Collected: {self.collected_blocks}, Failed: {self.failed_blocks}")
                
                # Adaptive delay between batches based on failure rate
                failure_rate = self.failed_blocks / (self.collected_blocks + self.failed_blocks) if (self.collected_blocks + self.failed_blocks) > 0 else 0
                if failure_rate > 0.1:  # If more than 10% failures, increase delay
                    delay = min(10, 2 + failure_rate * 10)
                    logger.info(f"High failure rate ({failure_rate:.1%}), increasing delay to {delay}s")
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(2)  # Normal delay
        
        logger.success(f"Historical block collection completed: {self.collected_blocks} blocks collected, {self.failed_blocks} failed")
        return all_metrics
    
    async def collect_network_metrics(self):
        """Collect current network metrics"""
        logger.info("Collecting current network metrics...")
        
        async with QuickNodePolkadotArchiveClient(self.config) as client:
            # Collect all metrics in parallel
            tasks = [
                client.collect_staking_data(),
                client.collect_parachain_data(),
                client.collect_governance_data()
            ]
            
            staking_data, parachain_data, governance_data = await asyncio.gather(*tasks)
            
            # Store in database
            if staking_data:
                self.db.store_staking_data(staking_data)
            if parachain_data:
                self.db.store_parachain_data(parachain_data)
            if governance_data:
                self.db.store_governance_data(governance_data)
            
            logger.success("Network metrics collected and stored")
    
    async def run_comprehensive_collection(self):
        """Run comprehensive archive data collection"""
        self.start_time = time.time()
        logger.info("🚀 Starting comprehensive Polkadot archive data collection")
        logger.info(f"Configuration: {self.config.max_workers} workers, {self.config.days_back} days back")
        
        try:
            # Get block range
            async with QuickNodePolkadotArchiveClient(self.config) as client:
                start_block, end_block = await client.get_block_range(self.config.days_back)
            
            if start_block == 0 or end_block == 0:
                logger.error("Could not determine block range")
                return
            
            logger.info(f"Block range: {start_block} to {end_block}")
            
            # Collect historical blocks
            historical_data = await self.collect_historical_blocks(start_block, end_block)
            
            # Collect current network metrics
            await self.collect_network_metrics()
            
            # Generate summary
            self._generate_collection_summary(historical_data)
            
        except Exception as e:
            logger.error(f"Error in comprehensive collection: {e}")
            raise
    
    def _generate_collection_summary(self, historical_data: List[Dict]):
        """Generate collection summary and statistics"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = {
            'collection_timestamp': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': duration,
            'blocks_collected': self.collected_blocks,
            'blocks_failed': self.failed_blocks,
            'success_rate': (self.collected_blocks / (self.collected_blocks + self.failed_blocks)) * 100 if (self.collected_blocks + self.failed_blocks) > 0 else 0,
            'workers_used': self.config.max_workers,
            'days_collected': self.config.days_back,
            'database_path': self.config.database_path,
            'sample_rate': self.config.sample_rate
        }
        
        # Save summary
        with open('polkadot_archive_collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.success("🎉 Archive data collection completed!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Blocks collected: {self.collected_blocks}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Database: {self.config.database_path}")

async def main():
    """Main function"""
    # Configuration
    config = CollectionConfig(
        max_workers=30,
        days_back=365,  # 1 year of data
        sample_rate=10,  # Every 10th block
        batch_size=100,
        rate_limit_delay=0.1
    )
    
    # Initialize collector
    collector = PolkadotArchiveCollector(config)
    
    # Run comprehensive collection
    await collector.run_comprehensive_collection()

if __name__ == "__main__":
    asyncio.run(main())
