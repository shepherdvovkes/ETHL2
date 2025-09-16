#!/usr/bin/env python3
"""
Polkadot Data Collection Script
Independent script for collecting and storing Polkadot metrics
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal, engine
from database.polkadot_models import (
    PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
    PolkadotStakingMetrics, PolkadotGovernanceMetrics,
    PolkadotEconomicMetrics, ParachainMetrics,
    ParachainCrossChainMetrics, PolkadotEcosystemMetrics,
    PolkadotPerformanceMetrics
)
from api.polkadot_client import PolkadotClient

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

class PolkadotDataCollector:
    """Data collector for Polkadot metrics"""
    
    def __init__(self):
        self.client = None
        self.db = None
    
    async def initialize(self):
        """Initialize the data collector"""
        try:
            # Initialize Polkadot client
            self.client = PolkadotClient()
            logger.info("Polkadot client initialized")
            
            # Initialize database session
            self.db = SessionLocal()
            logger.info("Database session initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db:
                self.db.close()
                logger.info("Database session closed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_network_metrics()
                logger.info("Network metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {}
    
    async def collect_staking_metrics(self) -> Dict[str, Any]:
        """Collect staking metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_staking_metrics()
                logger.info("Staking metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting staking metrics: {e}")
            return {}
    
    async def collect_governance_metrics(self) -> Dict[str, Any]:
        """Collect governance metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_governance_metrics()
                logger.info("Governance metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting governance metrics: {e}")
            return {}
    
    async def collect_economic_metrics(self) -> Dict[str, Any]:
        """Collect economic metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_economic_metrics()
                logger.info("Economic metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting economic metrics: {e}")
            return {}
    
    async def collect_parachain_metrics(self) -> Dict[str, Any]:
        """Collect parachain metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_all_parachains_info()
                logger.info("Parachain metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting parachain metrics: {e}")
            return {}
    
    async def collect_cross_chain_metrics(self) -> Dict[str, Any]:
        """Collect cross-chain metrics"""
        try:
            async with self.client:
                metrics = await self.client.get_cross_chain_metrics()
                logger.info("Cross-chain metrics collected")
                return metrics
        except Exception as e:
            logger.error(f"Error collecting cross-chain metrics: {e}")
            return {}
    
    def store_network_metrics(self, metrics: Dict[str, Any]):
        """Store network metrics in database"""
        try:
            # Get or create Polkadot network
            network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if not network:
                logger.warning("Polkadot network not found in database")
                return
            
            # Store network metrics
            network_metrics = PolkadotNetworkMetrics(
                network_id=network.id,
                current_block=metrics.get("network_info", {}).get("latest_block", {}).get("number"),
                validator_count=metrics.get("network_info", {}).get("validator_count", 0),
                runtime_version=metrics.get("runtime_version", {}).get("specName"),
                spec_version=metrics.get("runtime_version", {}).get("specVersion"),
                timestamp=datetime.utcnow()
            )
            
            self.db.add(network_metrics)
            self.db.commit()
            
            logger.success("Network metrics stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing network metrics: {e}")
            self.db.rollback()
    
    def store_staking_metrics(self, metrics: Dict[str, Any]):
        """Store staking metrics in database"""
        try:
            network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if not network:
                logger.warning("Polkadot network not found in database")
                return
            
            staking_metrics = PolkadotStakingMetrics(
                network_id=network.id,
                total_staked=metrics.get("total_staked"),
                validator_count=metrics.get("validator_count", 0),
                nominator_count=metrics.get("nominator_count", 0),
                active_era=metrics.get("active_era", {}).get("index"),
                inflation_rate=metrics.get("inflation"),
                timestamp=datetime.utcnow()
            )
            
            self.db.add(staking_metrics)
            self.db.commit()
            
            logger.success("Staking metrics stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing staking metrics: {e}")
            self.db.rollback()
    
    def store_governance_metrics(self, metrics: Dict[str, Any]):
        """Store governance metrics in database"""
        try:
            network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if not network:
                logger.warning("Polkadot network not found in database")
                return
            
            governance_metrics = PolkadotGovernanceMetrics(
                network_id=network.id,
                active_proposals=metrics.get("active_proposals", 0),
                referendum_count=metrics.get("referendums", 0),
                council_members=metrics.get("council_members", 0),
                timestamp=datetime.utcnow()
            )
            
            self.db.add(governance_metrics)
            self.db.commit()
            
            logger.success("Governance metrics stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing governance metrics: {e}")
            self.db.rollback()
    
    def store_economic_metrics(self, metrics: Dict[str, Any]):
        """Store economic metrics in database"""
        try:
            network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if not network:
                logger.warning("Polkadot network not found in database")
                return
            
            economic_metrics = PolkadotEconomicMetrics(
                network_id=network.id,
                treasury_balance=metrics.get("treasury_balance"),
                inflation_rate=metrics.get("inflation"),
                block_reward=metrics.get("block_reward"),
                timestamp=datetime.utcnow()
            )
            
            self.db.add(economic_metrics)
            self.db.commit()
            
            logger.success("Economic metrics stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing economic metrics: {e}")
            self.db.rollback()
    
    def store_parachain_metrics(self, parachains_info: Dict[str, Any]):
        """Store parachain metrics in database"""
        try:
            network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if not network:
                logger.warning("Polkadot network not found in database")
                return
            
            stored_count = 0
            
            for parachain_name, info in parachains_info.items():
                # Get parachain from database
                parachain = self.db.query(Parachain).filter(
                    Parachain.parachain_id == info.get("id")
                ).first()
                
                if parachain:
                    # Store parachain metrics
                    parachain_metrics = ParachainMetrics(
                        parachain_id=parachain.id,
                        current_block=info.get("head", {}).get("number"),
                        timestamp=datetime.utcnow()
                    )
                    
                    self.db.add(parachain_metrics)
                    stored_count += 1
            
            self.db.commit()
            logger.success(f"Parachain metrics stored successfully: {stored_count} parachains")
            
        except Exception as e:
            logger.error(f"Error storing parachain metrics: {e}")
            self.db.rollback()
    
    def store_cross_chain_metrics(self, metrics: Dict[str, Any]):
        """Store cross-chain metrics in database"""
        try:
            # Store cross-chain metrics for each parachain
            parachains = self.db.query(Parachain).all()
            stored_count = 0
            
            for parachain in parachains:
                cross_chain_metrics = ParachainCrossChainMetrics(
                    parachain_id=parachain.id,
                    hrmp_channels_count=len(metrics.get("hrmp_channels", [])),
                    xcmp_channels_count=len(metrics.get("xcmp_channels", [])),
                    timestamp=datetime.utcnow()
                )
                
                self.db.add(cross_chain_metrics)
                stored_count += 1
            
            self.db.commit()
            logger.success(f"Cross-chain metrics stored successfully: {stored_count} parachains")
            
        except Exception as e:
            logger.error(f"Error storing cross-chain metrics: {e}")
            self.db.rollback()
    
    async def collect_all_metrics(self):
        """Collect all metrics"""
        try:
            logger.info("Starting comprehensive data collection...")
            
            # Collect network metrics
            network_metrics = await self.collect_network_metrics()
            if network_metrics:
                self.store_network_metrics(network_metrics)
            
            # Collect staking metrics
            staking_metrics = await self.collect_staking_metrics()
            if staking_metrics:
                self.store_staking_metrics(staking_metrics)
            
            # Collect governance metrics
            governance_metrics = await self.collect_governance_metrics()
            if governance_metrics:
                self.store_governance_metrics(governance_metrics)
            
            # Collect economic metrics
            economic_metrics = await self.collect_economic_metrics()
            if economic_metrics:
                self.store_economic_metrics(economic_metrics)
            
            # Collect parachain metrics
            parachain_metrics = await self.collect_parachain_metrics()
            if parachain_metrics:
                self.store_parachain_metrics(parachain_metrics)
            
            # Collect cross-chain metrics
            cross_chain_metrics = await self.collect_cross_chain_metrics()
            if cross_chain_metrics:
                self.store_cross_chain_metrics(cross_chain_metrics)
            
            logger.success("All metrics collected and stored successfully!")
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            raise

async def main():
    """Main function"""
    collector = PolkadotDataCollector()
    
    try:
        # Initialize collector
        await collector.initialize()
        
        # Collect all metrics
        await collector.collect_all_metrics()
        
        logger.success("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        await collector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
