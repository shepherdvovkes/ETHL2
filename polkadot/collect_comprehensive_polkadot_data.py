#!/usr/bin/env python3
"""
Comprehensive Polkadot Data Collection Script
Collects and stores comprehensive metrics for Polkadot and all parachains
"""

import asyncio
import sys
import os
import argparse
import multiprocessing
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal, engine
from database.polkadot_comprehensive_models import (
    PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
    PolkadotStakingMetrics, PolkadotGovernanceMetrics,
    PolkadotEconomicMetrics, ParachainMetrics,
    ParachainCrossChainMetrics, PolkadotEcosystemMetrics,
    PolkadotPerformanceMetrics, PolkadotSecurityMetrics,
    PolkadotDeveloperMetrics, ParachainDeFiMetrics,
    ParachainPerformanceMetrics, ParachainSecurityMetrics,
    ParachainDeveloperMetrics, TokenMarketData, ValidatorInfo
)
from api.polkadot_comprehensive_client import PolkadotComprehensiveClient

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

class ComprehensivePolkadotDataCollector:
    """Comprehensive data collector for Polkadot metrics"""
    
    def __init__(self, worker_id: int = 0, start_block: int = None, end_block: int = None):
        self.client = None
        self.db = None
        self.worker_id = worker_id
        self.start_block = start_block
        self.end_block = end_block
    
    async def initialize(self):
        """Initialize the data collector"""
        try:
            # Initialize Polkadot client
            self.client = PolkadotComprehensiveClient()
            logger.info("Polkadot comprehensive client initialized")
            
            # Initialize database session
            self.db = SessionLocal()
            logger.info("Database session initialized")
            
            # Create tables if they don't exist
            from database.polkadot_comprehensive_models import Base
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db:
                self.db.close()
                logger.info(f"Worker {self.worker_id}: Database session closed")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Cleanup error: {e}")
    
    def get_current_block_from_db(self) -> int:
        """Get the current block number from the database"""
        try:
            import sqlite3
            conn = sqlite3.connect('polkadot_archive_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(block_number) FROM block_metrics")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return result[0]
            else:
                return 27770131  # Default starting block
        except Exception as e:
            logger.warning(f"Could not get current block from DB: {e}, using default")
            return 27770131
    
    def get_or_create_network(self) -> PolkadotNetwork:
        """Get or create Polkadot network"""
        network = self.db.query(PolkadotNetwork).filter(
            PolkadotNetwork.name == "Polkadot"
        ).first()
        
        if not network:
            network = PolkadotNetwork(
                name="Polkadot",
                chain_id="polkadot",
                rpc_endpoint="https://rpc.polkadot.io",
                ws_endpoint="wss://rpc.polkadot.io",
                is_mainnet=True,
                spec_version=1000,
                transaction_version=1
            )
            self.db.add(network)
            self.db.commit()
            self.db.refresh(network)
            logger.info("Created Polkadot network record")
        
        return network
    
    def get_or_create_parachain(self, parachain_info: Dict[str, Any], network: PolkadotNetwork) -> Parachain:
        """Get or create parachain"""
        parachain = self.db.query(Parachain).filter(
            Parachain.parachain_id == parachain_info["id"]
        ).first()
        
        if not parachain:
            parachain = Parachain(
                parachain_id=parachain_info["id"],
                name=parachain_info["name"],
                symbol=parachain_info["symbol"],
                network_id=network.id,
                status="active",
                category=parachain_info.get("category", "general"),
                rpc_endpoint=parachain_info.get("rpc"),
                ws_endpoint=parachain_info.get("ws")
            )
            self.db.add(parachain)
            self.db.commit()
            self.db.refresh(parachain)
            logger.info(f"Created parachain record for {parachain_info['name']}")
        
        return parachain
    
    async def collect_and_store_network_metrics(self):
        """Collect and store network metrics"""
        try:
            logger.info("Collecting network metrics...")
            
            async with self.client:
                # Get comprehensive network metrics
                network_metrics = await self.client.get_comprehensive_network_metrics()
                
                if not network_metrics:
                    logger.warning("No network metrics collected")
                    return
                
                network = self.get_or_create_network()
                
                # Store network metrics
                network_metrics_record = PolkadotNetworkMetrics(
                    network_id=network.id,
                    current_block=network_metrics.get("network_info", {}).get("latest_block", {}).get("number"),
                    validator_count=network_metrics.get("network_info", {}).get("validator_count", 0),
                    runtime_version=network_metrics.get("network_info", {}).get("runtime_version", {}).get("specName"),
                    spec_version=network_metrics.get("network_info", {}).get("runtime_version", {}).get("specVersion"),
                    peer_count=network_metrics.get("network_info", {}).get("peer_count", 0),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(network_metrics_record)
                
                # Store staking metrics
                staking_data = network_metrics.get("staking_metrics", {})
                staking_metrics_record = PolkadotStakingMetrics(
                    network_id=network.id,
                    total_staked=staking_data.get("total_staked"),
                    validator_count=staking_data.get("validator_count", 0),
                    waiting_validators=staking_data.get("waiting_validators", 0),
                    nomination_pools_count=staking_data.get("nomination_pools_count", 0),
                    inflation_rate=staking_data.get("inflation"),
                    active_era=staking_data.get("active_era", {}).get("index"),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(staking_metrics_record)
                
                # Store governance metrics
                governance_data = network_metrics.get("governance_metrics", {})
                governance_metrics_record = PolkadotGovernanceMetrics(
                    network_id=network.id,
                    active_proposals=governance_data.get("active_proposals", 0),
                    referendum_count=governance_data.get("referendums", 0),
                    council_members=governance_data.get("council_members", 0),
                    treasury_proposals=governance_data.get("treasury_proposals", 0),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(governance_metrics_record)
                
                # Store economic metrics
                economic_data = network_metrics.get("economic_metrics", {})
                economic_metrics_record = PolkadotEconomicMetrics(
                    network_id=network.id,
                    treasury_balance=economic_data.get("treasury_balance"),
                    inflation_rate=economic_data.get("inflation"),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(economic_metrics_record)
                
                # Store performance metrics
                performance_data = network_metrics.get("performance_metrics", {})
                performance_metrics_record = PolkadotPerformanceMetrics(
                    network_id=network.id,
                    service_availability=99.9,  # Placeholder
                    uptime=99.9,  # Placeholder
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(performance_metrics_record)
                
                # Store security metrics
                security_data = network_metrics.get("security_metrics", {})
                security_metrics_record = PolkadotSecurityMetrics(
                    network_id=network.id,
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(security_metrics_record)
                
                # Store developer metrics
                developer_data = network_metrics.get("developer_metrics", {})
                developer_metrics_record = PolkadotDeveloperMetrics(
                    network_id=network.id,
                    total_developers=developer_data.get("total_developers", 0),
                    github_commits_24h=developer_data.get("github_commits_24h", 0),
                    active_projects=developer_data.get("active_projects", 0),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(developer_metrics_record)
                
                self.db.commit()
                logger.success("Network metrics stored successfully")
                
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            self.db.rollback()
    
    async def collect_and_store_parachain_metrics(self):
        """Collect and store parachain metrics"""
        try:
            logger.info("Collecting parachain metrics...")
            
            async with self.client:
                # Get comprehensive parachain metrics
                parachain_metrics = await self.client.get_comprehensive_parachain_metrics()
                
                if not parachain_metrics:
                    logger.warning("No parachain metrics collected")
                    return
                
                network = self.get_or_create_network()
                stored_count = 0
                
                for parachain_name, info in parachain_metrics.items():
                    try:
                        # Get or create parachain
                        parachain = self.get_or_create_parachain(info, network)
                        
                        # Store parachain metrics
                        parachain_metrics_record = ParachainMetrics(
                            parachain_id=parachain.id,
                            current_block=info.get("head", {}).get("number"),
                            active_addresses_24h=info.get("metrics", {}).get("active_addresses", 0),
                            daily_transactions=info.get("metrics", {}).get("transactions_24h", 0),
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        self.db.add(parachain_metrics_record)
                        
                        # Store DeFi metrics if applicable
                        if info.get("category") == "defi":
                            defi_metrics = info.get("metrics", {})
                            defi_metrics_record = ParachainDeFiMetrics(
                                parachain_id=parachain.id,
                                total_tvl=defi_metrics.get("tvl"),
                                dex_volume_24h=defi_metrics.get("dex_volume_24h"),
                                lending_tvl=defi_metrics.get("lending_tvl"),
                                timestamp=datetime.now(timezone.utc)
                            )
                            
                            self.db.add(defi_metrics_record)
                        
                        # Store performance metrics
                        performance_metrics_record = ParachainPerformanceMetrics(
                            parachain_id=parachain.id,
                            uptime=99.9,  # Placeholder
                            availability=99.9,  # Placeholder
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        self.db.add(performance_metrics_record)
                        
                        # Store security metrics
                        security_metrics_record = ParachainSecurityMetrics(
                            parachain_id=parachain.id,
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        self.db.add(security_metrics_record)
                        
                        # Store developer metrics
                        developer_metrics_record = ParachainDeveloperMetrics(
                            parachain_id=parachain.id,
                            active_developers=50,  # Placeholder
                            github_commits_24h=10,  # Placeholder
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        self.db.add(developer_metrics_record)
                        
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing metrics for {parachain_name}: {e}")
                        continue
                
                self.db.commit()
                logger.success(f"Parachain metrics stored successfully: {stored_count} parachains")
                
        except Exception as e:
            logger.error(f"Error collecting parachain metrics: {e}")
            self.db.rollback()
    
    async def collect_and_store_cross_chain_metrics(self):
        """Collect and store cross-chain metrics"""
        try:
            logger.info("Collecting cross-chain metrics...")
            
            async with self.client:
                # Get cross-chain metrics
                cross_chain_metrics = await self.client.get_cross_chain_metrics()
                
                if not cross_chain_metrics:
                    logger.warning("No cross-chain metrics collected")
                    return
                
                # Store cross-chain metrics for each parachain
                parachains = self.db.query(Parachain).all()
                stored_count = 0
                
                for parachain in parachains:
                    cross_chain_metrics_record = ParachainCrossChainMetrics(
                        parachain_id=parachain.id,
                        hrmp_channels_count=cross_chain_metrics.get("hrmp_channels_count", 0),
                        xcmp_channels_count=cross_chain_metrics.get("xcmp_channels_count", 0),
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    self.db.add(cross_chain_metrics_record)
                    stored_count += 1
                
                self.db.commit()
                logger.success(f"Cross-chain metrics stored successfully: {stored_count} parachains")
                
        except Exception as e:
            logger.error(f"Error collecting cross-chain metrics: {e}")
            self.db.rollback()
    
    async def collect_and_store_ecosystem_metrics(self):
        """Collect and store ecosystem metrics"""
        try:
            logger.info("Collecting ecosystem metrics...")
            
            async with self.client:
                # Get ecosystem metrics
                ecosystem_metrics = await self.client.get_ecosystem_metrics()
                
                if not ecosystem_metrics:
                    logger.warning("No ecosystem metrics collected")
                    return
                
                # Store ecosystem metrics
                ecosystem_metrics_record = PolkadotEcosystemMetrics(
                    total_parachains=ecosystem_metrics.get("total_parachains", 0),
                    active_parachains=ecosystem_metrics.get("active_parachains", 0),
                    active_cross_chain_channels=ecosystem_metrics.get("cross_chain_channels", 0),
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.db.add(ecosystem_metrics_record)
                self.db.commit()
                logger.success("Ecosystem metrics stored successfully")
                
        except Exception as e:
            logger.error(f"Error collecting ecosystem metrics: {e}")
            self.db.rollback()
    
    async def collect_and_store_token_market_data(self):
        """Collect and store token market data"""
        try:
            logger.info("Collecting token market data...")
            
            async with self.client:
                # Get token market data
                token_data = await self.client.get_token_market_data()
                
                if not token_data:
                    logger.warning("No token market data collected")
                    return
                
                stored_count = 0
                
                for token_symbol, data in token_data.items():
                    if token_symbol == "timestamp":
                        continue
                    
                    # Find parachain if it exists
                    parachain = None
                    if token_symbol != "DOT":
                        parachain = self.db.query(Parachain).filter(
                            Parachain.symbol == token_symbol
                        ).first()
                    
                    token_market_record = TokenMarketData(
                        token_symbol=token_symbol,
                        token_name=token_symbol,
                        parachain_id=parachain.id if parachain else None,
                        price_usd=data.get("price_usd"),
                        price_change_24h=data.get("price_change_24h"),
                        market_cap=data.get("market_cap"),
                        volume_24h=data.get("volume_24h"),
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    self.db.add(token_market_record)
                    stored_count += 1
                
                self.db.commit()
                logger.success(f"Token market data stored successfully: {stored_count} tokens")
                
        except Exception as e:
            logger.error(f"Error collecting token market data: {e}")
            self.db.rollback()
    
    async def collect_and_store_validator_info(self):
        """Collect and store validator information"""
        try:
            logger.info("Collecting validator information...")
            
            async with self.client:
                # Get validator info
                validator_data = await self.client.get_validator_info()
                
                if not validator_data:
                    logger.warning("No validator data collected")
                    return
                
                network = self.get_or_create_network()
                stored_count = 0
                
                validators = validator_data.get("validators", [])
                for validator_address in validators:
                    # Check if validator already exists
                    existing_validator = self.db.query(ValidatorInfo).filter(
                        ValidatorInfo.validator_address == validator_address
                    ).first()
                    
                    if not existing_validator:
                        validator_record = ValidatorInfo(
                            validator_address=validator_address,
                            network_id=network.id,
                            is_active=True,
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        self.db.add(validator_record)
                        stored_count += 1
                
                self.db.commit()
                logger.success(f"Validator information stored successfully: {stored_count} validators")
                
        except Exception as e:
            logger.error(f"Error collecting validator information: {e}")
            self.db.rollback()
    
    async def collect_all_metrics(self):
        """Collect all metrics"""
        try:
            logger.info("Starting comprehensive data collection...")
            
            # Collect network metrics
            await self.collect_and_store_network_metrics()
            
            # Collect parachain metrics
            await self.collect_and_store_parachain_metrics()
            
            # Collect cross-chain metrics
            await self.collect_and_store_cross_chain_metrics()
            
            # Collect ecosystem metrics
            await self.collect_and_store_ecosystem_metrics()
            
            # Collect token market data
            await self.collect_and_store_token_market_data()
            
            # Collect validator information
            await self.collect_and_store_validator_info()
            
            logger.success("All metrics collected and stored successfully!")
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            raise

def calculate_block_ranges(current_block: int, num_workers: int, blocks_per_worker: int = 1000) -> List[tuple]:
    """Calculate block ranges for each worker"""
    ranges = []
    start_block = current_block + 1
    
    for i in range(num_workers):
        end_block = start_block + blocks_per_worker - 1
        ranges.append((start_block, end_block))
        start_block = end_block + 1
    
    return ranges

async def worker_main(worker_id: int, start_block: int, end_block: int):
    """Main function for a single worker"""
    logger.info(f"Worker {worker_id}: Starting collection from block {start_block} to {end_block}")
    
    collector = ComprehensivePolkadotDataCollector(
        worker_id=worker_id,
        start_block=start_block,
        end_block=end_block
    )
    
    try:
        # Initialize collector
        await collector.initialize()
        
        # Collect all metrics
        await collector.collect_all_metrics()
        
        logger.success(f"Worker {worker_id}: Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Data collection failed: {e}")
        raise
    
    finally:
        # Cleanup
        await collector.cleanup()

async def main():
    """Main function with multi-worker support"""
    parser = argparse.ArgumentParser(description='Polkadot Data Collector')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers (default: 10)')
    parser.add_argument('--blocks-per-worker', type=int, default=1000, help='Blocks per worker (default: 1000)')
    parser.add_argument('--start-block', type=int, help='Starting block number (default: from DB)')
    
    args = parser.parse_args()
    
    # Get current block from database
    temp_collector = ComprehensivePolkadotDataCollector()
    current_block = args.start_block if args.start_block else temp_collector.get_current_block_from_db()
    
    logger.info(f"Starting multi-worker collection with {args.workers} workers")
    logger.info(f"Current block in DB: {current_block}")
    
    # Calculate block ranges for each worker
    block_ranges = calculate_block_ranges(current_block, args.workers, args.blocks_per_worker)
    
    # Create and start workers
    tasks = []
    for i, (start_block, end_block) in enumerate(block_ranges):
        logger.info(f"Worker {i}: Block range {start_block} - {end_block}")
        task = asyncio.create_task(worker_main(i, start_block, end_block))
        tasks.append(task)
    
    # Wait for all workers to complete
    try:
        await asyncio.gather(*tasks)
        logger.success("All workers completed successfully!")
    except Exception as e:
        logger.error(f"One or more workers failed: {e}")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
