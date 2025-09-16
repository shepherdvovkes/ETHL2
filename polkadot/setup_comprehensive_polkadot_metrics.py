#!/usr/bin/env python3
"""
Setup script for Comprehensive Polkadot Metrics System
Initializes database tables and populates initial data
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal, engine
from database.polkadot_comprehensive_models import (
    Base, PolkadotNetwork, Parachain
)
from api.polkadot_comprehensive_client import PolkadotComprehensiveClient

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

class ComprehensivePolkadotSetup:
    """Setup class for comprehensive Polkadot metrics system"""
    
    def __init__(self):
        self.db = None
        self.client = None
    
    async def initialize(self):
        """Initialize the setup"""
        try:
            # Initialize database session
            self.db = SessionLocal()
            logger.info("Database session initialized")
            
            # Initialize Polkadot client
            self.client = PolkadotComprehensiveClient()
            logger.info("Polkadot client initialized")
            
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
    
    def create_database_tables(self):
        """Create all database tables"""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            logger.success("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def setup_polkadot_network(self):
        """Setup Polkadot network record"""
        try:
            logger.info("Setting up Polkadot network...")
            
            # Check if network already exists
            existing_network = self.db.query(PolkadotNetwork).filter(
                PolkadotNetwork.name == "Polkadot"
            ).first()
            
            if existing_network:
                logger.info("Polkadot network already exists")
                return existing_network
            
            # Create Polkadot network
            network = PolkadotNetwork(
                name="Polkadot",
                chain_id="polkadot",
                rpc_endpoint="https://rpc.polkadot.io",
                ws_endpoint="wss://rpc.polkadot.io",
                is_mainnet=True,
                spec_version=1000,
                transaction_version=1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(network)
            self.db.commit()
            self.db.refresh(network)
            
            logger.success("Polkadot network created successfully")
            return network
            
        except Exception as e:
            logger.error(f"Error setting up Polkadot network: {e}")
            self.db.rollback()
            raise
    
    def setup_parachains(self, network):
        """Setup parachain records"""
        try:
            logger.info("Setting up parachains...")
            
            async with self.client:
                parachains_info = self.client.get_supported_parachains()
            
            created_count = 0
            updated_count = 0
            
            for parachain_name, info in parachains_info.items():
                try:
                    # Check if parachain already exists
                    existing_parachain = self.db.query(Parachain).filter(
                        Parachain.parachain_id == info["id"]
                    ).first()
                    
                    if existing_parachain:
                        # Update existing parachain
                        existing_parachain.name = info["name"]
                        existing_parachain.symbol = info["symbol"]
                        existing_parachain.category = info.get("category", "general")
                        existing_parachain.rpc_endpoint = info.get("rpc")
                        existing_parachain.ws_endpoint = info.get("ws")
                        existing_parachain.updated_at = datetime.utcnow()
                        updated_count += 1
                    else:
                        # Create new parachain
                        parachain = Parachain(
                            parachain_id=info["id"],
                            name=info["name"],
                            symbol=info["symbol"],
                            network_id=network.id,
                            status="active",
                            category=info.get("category", "general"),
                            rpc_endpoint=info.get("rpc"),
                            ws_endpoint=info.get("ws"),
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        
                        self.db.add(parachain)
                        created_count += 1
                    
                except Exception as e:
                    logger.error(f"Error setting up parachain {parachain_name}: {e}")
                    continue
            
            self.db.commit()
            logger.success(f"Parachains setup completed: {created_count} created, {updated_count} updated")
            
        except Exception as e:
            logger.error(f"Error setting up parachains: {e}")
            self.db.rollback()
            raise
    
    async def test_data_collection(self):
        """Test data collection functionality"""
        try:
            logger.info("Testing data collection...")
            
            async with self.client:
                # Test network info
                network_info = await self.client.get_network_info()
                if network_info:
                    logger.success("Network info collection test passed")
                else:
                    logger.warning("Network info collection test failed")
                
                # Test parachain info
                parachain_info = await self.client.get_parachain_info(2004)  # Moonbeam
                if parachain_info:
                    logger.success("Parachain info collection test passed")
                else:
                    logger.warning("Parachain info collection test failed")
                
                # Test health check
                health = await self.client.health_check()
                if health.get("status") == "healthy":
                    logger.success("Health check test passed")
                else:
                    logger.warning("Health check test failed")
            
        except Exception as e:
            logger.error(f"Data collection test failed: {e}")
            raise
    
    def verify_setup(self):
        """Verify the setup is complete"""
        try:
            logger.info("Verifying setup...")
            
            # Check network
            network_count = self.db.query(PolkadotNetwork).count()
            if network_count > 0:
                logger.success(f"Network records: {network_count}")
            else:
                logger.error("No network records found")
                return False
            
            # Check parachains
            parachain_count = self.db.query(Parachain).count()
            if parachain_count > 0:
                logger.success(f"Parachain records: {parachain_count}")
            else:
                logger.error("No parachain records found")
                return False
            
            # Check categories
            categories = self.db.query(Parachain.category).distinct().all()
            category_list = [cat[0] for cat in categories if cat[0]]
            logger.success(f"Parachain categories: {', '.join(category_list)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Setup verification failed: {e}")
            return False
    
    async def run_setup(self):
        """Run the complete setup process"""
        try:
            logger.info("Starting comprehensive Polkadot metrics setup...")
            
            # Initialize
            await self.initialize()
            
            # Create database tables
            self.create_database_tables()
            
            # Setup Polkadot network
            network = self.setup_polkadot_network()
            
            # Setup parachains
            self.setup_parachains(network)
            
            # Test data collection
            await self.test_data_collection()
            
            # Verify setup
            if self.verify_setup():
                logger.success("Comprehensive Polkadot metrics setup completed successfully!")
                return True
            else:
                logger.error("Setup verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    setup = ComprehensivePolkadotSetup()
    
    try:
        success = await setup.run_setup()
        
        if success:
            logger.success("Setup completed successfully!")
            print("\n" + "="*60)
            print("🚀 COMPREHENSIVE POLKADOT METRICS SYSTEM READY!")
            print("="*60)
            print("\nNext steps:")
            print("1. Run data collection: python collect_comprehensive_polkadot_data.py")
            print("2. Start metrics server: python polkadot_comprehensive_metrics_server.py")
            print("3. Open dashboard: http://localhost:8008")
            print("\nAvailable endpoints:")
            print("- /api/network/overview - Network overview")
            print("- /api/parachains - All parachains")
            print("- /api/tokens/market-data - Token market data")
            print("- /api/validators - Validator information")
            print("- /api/ecosystem/metrics - Ecosystem metrics")
            print("="*60)
        else:
            logger.error("Setup failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
