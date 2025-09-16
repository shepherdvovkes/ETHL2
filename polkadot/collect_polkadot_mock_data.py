#!/usr/bin/env python3
"""
Polkadot Mock Data Collection Script
Creates realistic mock data for testing the system
"""

import sys
import os
from datetime import datetime, timezone
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

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

class PolkadotMockDataCollector:
    """Mock data collector for Polkadot metrics"""
    
    def __init__(self):
        self.db = None
    
    async def initialize(self):
        """Initialize the data collector"""
        try:
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
                logger.info("Database session closed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
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
    
    async def collect_mock_data(self):
        """Collect mock data for all metrics"""
        try:
            logger.info("Collecting mock data...")
            
            network = self.get_or_create_network()
            
            # Mock parachain data
            parachains_data = [
                {"id": 2004, "name": "Moonbeam", "symbol": "GLMR", "category": "defi"},
                {"id": 2006, "name": "Astar", "symbol": "ASTR", "category": "defi"},
                {"id": 2000, "name": "Acala", "symbol": "ACA", "category": "defi"},
                {"id": 2034, "name": "HydraDX", "symbol": "HDX", "category": "defi"},
                {"id": 2030, "name": "Bifrost", "symbol": "BNC", "category": "defi"},
                {"id": 1000, "name": "AssetHub", "symbol": "DOT", "category": "infrastructure"},
                {"id": 2026, "name": "Nodle", "symbol": "NODL", "category": "iot"},
                {"id": 2035, "name": "Phala Network", "symbol": "PHA", "category": "computing"},
                {"id": 2091, "name": "Frequency", "symbol": "FRQCY", "category": "social"},
                {"id": 2046, "name": "NeuroWeb", "symbol": "NEURO", "category": "ai"},
            ]
            
            # Store parachains
            parachains = []
            for parachain_info in parachains_data:
                parachain = self.get_or_create_parachain(parachain_info, network)
                parachains.append(parachain)
            
            # Store network metrics
            network_metrics = PolkadotNetworkMetrics(
                network_id=network.id,
                current_block=18500000,
                validator_count=1000,
                runtime_version="polkadot",
                spec_version=1000,
                peer_count=500,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(network_metrics)
            
            # Store staking metrics
            staking_metrics = PolkadotStakingMetrics(
                network_id=network.id,
                total_staked=8900000000000000000,  # 8.9B DOT (smaller value)
                validator_count=1000,
                waiting_validators=50,
                nomination_pools_count=100,
                inflation_rate=7.5,
                active_era=1234,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(staking_metrics)
            
            # Store governance metrics
            governance_metrics = PolkadotGovernanceMetrics(
                network_id=network.id,
                active_proposals=3,
                referendum_count=1,
                council_members=13,
                treasury_proposals=5,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(governance_metrics)
            
            # Store economic metrics
            economic_metrics = PolkadotEconomicMetrics(
                network_id=network.id,
                treasury_balance=5000000000000000000,  # 5B DOT (smaller value)
                inflation_rate=7.5,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(economic_metrics)
            
            # Store parachain metrics
            for parachain in parachains:
                parachain_metrics = ParachainMetrics(
                    parachain_id=parachain.id,
                    current_block=1000000 + parachain.parachain_id * 1000,
                    active_addresses_24h=5000 + parachain.parachain_id * 100,
                    daily_transactions=10000 + parachain.parachain_id * 500,
                    timestamp=datetime.now(timezone.utc)
                )
                self.db.add(parachain_metrics)
                
                # Store DeFi metrics for DeFi parachains
                if parachain.category == "defi":
                    defi_metrics = ParachainDeFiMetrics(
                        parachain_id=parachain.id,
                        total_tvl=1000000000 + parachain.parachain_id * 100000000,  # $1B+ TVL
                        dex_volume_24h=50000000 + parachain.parachain_id * 1000000,  # $50M+ volume
                        lending_tvl=300000000 + parachain.parachain_id * 50000000,  # $300M+ lending
                        timestamp=datetime.now(timezone.utc)
                    )
                    self.db.add(defi_metrics)
                
                # Store cross-chain metrics
                cross_chain_metrics = ParachainCrossChainMetrics(
                    parachain_id=parachain.id,
                    hrmp_channels_count=5 + parachain.parachain_id % 10,
                    xcmp_channels_count=2 + parachain.parachain_id % 5,
                    timestamp=datetime.now(timezone.utc)
                )
                self.db.add(cross_chain_metrics)
            
            # Store ecosystem metrics
            ecosystem_metrics = PolkadotEcosystemMetrics(
                total_parachains=len(parachains),
                active_parachains=len(parachains),
                active_cross_chain_channels=45,
                total_active_developers=500,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(ecosystem_metrics)
            
            # Store token market data
            token_data = [
                {"symbol": "DOT", "name": "Polkadot", "price": 7.50, "market_cap": 10000000000},
                {"symbol": "GLMR", "name": "Moonbeam", "price": 0.25, "market_cap": 500000000},
                {"symbol": "ASTR", "name": "Astar", "price": 0.15, "market_cap": 300000000},
                {"symbol": "ACA", "name": "Acala", "price": 0.08, "market_cap": 200000000},
            ]
            
            for token in token_data:
                # Find parachain for token
                parachain = None
                if token["symbol"] != "DOT":
                    parachain = self.db.query(Parachain).filter(
                        Parachain.symbol == token["symbol"]
                    ).first()
                
                token_market = TokenMarketData(
                    token_symbol=token["symbol"],
                    token_name=token["name"],
                    parachain_id=parachain.id if parachain else None,
                    price_usd=token["price"],
                    market_cap=token["market_cap"],
                    volume_24h=token["market_cap"] * 0.1,  # 10% of market cap
                    price_change_24h=2.5,
                    timestamp=datetime.now(timezone.utc)
                )
                self.db.add(token_market)
            
            self.db.commit()
            logger.success("Mock data collected and stored successfully!")
            
        except Exception as e:
            logger.error(f"Error collecting mock data: {e}")
            self.db.rollback()
            raise

async def main():
    """Main function"""
    collector = PolkadotMockDataCollector()
    
    try:
        # Initialize collector
        await collector.initialize()
        
        # Collect mock data
        await collector.collect_mock_data()
        
        logger.success("Mock data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Mock data collection failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        await collector.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
