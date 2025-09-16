#!/usr/bin/env python3
"""
Test script for Comprehensive Polkadot Metrics System
Tests the setup, data collection, and API functionality
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
    PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
    PolkadotStakingMetrics, ParachainMetrics
)
from api.polkadot_comprehensive_client import PolkadotComprehensiveClient

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

class ComprehensiveMetricsTester:
    """Test class for comprehensive metrics system"""
    
    def __init__(self):
        self.db = None
        self.client = None
    
    async def initialize(self):
        """Initialize the tester"""
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
    
    def test_database_connection(self):
        """Test database connection and tables"""
        try:
            logger.info("Testing database connection...")
            
            # Test network table
            network_count = self.db.query(PolkadotNetwork).count()
            logger.success(f"Network records: {network_count}")
            
            # Test parachain table
            parachain_count = self.db.query(Parachain).count()
            logger.success(f"Parachain records: {parachain_count}")
            
            # Test metrics tables
            network_metrics_count = self.db.query(PolkadotNetworkMetrics).count()
            logger.success(f"Network metrics records: {network_metrics_count}")
            
            staking_metrics_count = self.db.query(PolkadotStakingMetrics).count()
            logger.success(f"Staking metrics records: {staking_metrics_count}")
            
            parachain_metrics_count = self.db.query(ParachainMetrics).count()
            logger.success(f"Parachain metrics records: {parachain_metrics_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_polkadot_client(self):
        """Test Polkadot client functionality"""
        try:
            logger.info("Testing Polkadot client...")
            
            async with self.client:
                # Test health check
                health = await self.client.health_check()
                if health.get("status") == "healthy":
                    logger.success("Health check passed")
                else:
                    logger.warning(f"Health check failed: {health}")
                
                # Test network info
                network_info = await self.client.get_network_info()
                if network_info:
                    logger.success("Network info collection passed")
                    logger.info(f"Chain: {network_info.get('chain', 'Unknown')}")
                    logger.info(f"Validators: {network_info.get('validator_count', 0)}")
                else:
                    logger.warning("Network info collection failed")
                
                # Test parachain info
                parachain_info = await self.client.get_parachain_info(2004)  # Moonbeam
                if parachain_info:
                    logger.success("Parachain info collection passed")
                    logger.info(f"Parachain ID: {parachain_info.get('parachain_id', 'Unknown')}")
                else:
                    logger.warning("Parachain info collection failed")
                
                # Test supported parachains
                parachains = self.client.get_supported_parachains()
                logger.success(f"Supported parachains: {len(parachains)}")
                
                # List some parachains
                for name, info in list(parachains.items())[:5]:
                    logger.info(f"  - {info['name']} (ID: {info['id']}, Category: {info.get('category', 'general')})")
            
            return True
            
        except Exception as e:
            logger.error(f"Polkadot client test failed: {e}")
            return False
    
    def test_data_models(self):
        """Test data models and relationships"""
        try:
            logger.info("Testing data models...")
            
            # Test network-parachain relationship
            network = self.db.query(PolkadotNetwork).first()
            if network:
                parachains = self.db.query(Parachain).filter(Parachain.network_id == network.id).all()
                logger.success(f"Network '{network.name}' has {len(parachains)} parachains")
                
                # Show parachain categories
                categories = {}
                for parachain in parachains:
                    category = parachain.category or 'general'
                    categories[category] = categories.get(category, 0) + 1
                
                logger.info("Parachain categories:")
                for category, count in categories.items():
                    logger.info(f"  - {category}: {count}")
            else:
                logger.warning("No network found")
            
            return True
            
        except Exception as e:
            logger.error(f"Data models test failed: {e}")
            return False
    
    def test_metrics_data(self):
        """Test metrics data availability"""
        try:
            logger.info("Testing metrics data...")
            
            # Test network metrics
            latest_network_metrics = self.db.query(PolkadotNetworkMetrics).order_by(
                PolkadotNetworkMetrics.timestamp.desc()
            ).first()
            
            if latest_network_metrics:
                logger.success("Latest network metrics found")
                logger.info(f"  - Block: {latest_network_metrics.current_block}")
                logger.info(f"  - Validators: {latest_network_metrics.validator_count}")
                logger.info(f"  - Timestamp: {latest_network_metrics.timestamp}")
            else:
                logger.warning("No network metrics found")
            
            # Test staking metrics
            latest_staking_metrics = self.db.query(PolkadotStakingMetrics).order_by(
                PolkadotStakingMetrics.timestamp.desc()
            ).first()
            
            if latest_staking_metrics:
                logger.success("Latest staking metrics found")
                logger.info(f"  - Total staked: {latest_staking_metrics.total_staked}")
                logger.info(f"  - Validators: {latest_staking_metrics.validator_count}")
                logger.info(f"  - Inflation: {latest_staking_metrics.inflation_rate}%")
            else:
                logger.warning("No staking metrics found")
            
            # Test parachain metrics
            parachain_metrics_count = self.db.query(ParachainMetrics).count()
            if parachain_metrics_count > 0:
                logger.success(f"Parachain metrics records: {parachain_metrics_count}")
                
                # Show latest parachain metrics
                latest_parachain_metrics = self.db.query(ParachainMetrics).order_by(
                    ParachainMetrics.timestamp.desc()
                ).limit(5).all()
                
                logger.info("Latest parachain metrics:")
                for metrics in latest_parachain_metrics:
                    parachain = self.db.query(Parachain).filter(Parachain.id == metrics.parachain_id).first()
                    if parachain:
                        logger.info(f"  - {parachain.name}: Block {metrics.current_block}")
            else:
                logger.warning("No parachain metrics found")
            
            return True
            
        except Exception as e:
            logger.error(f"Metrics data test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            logger.info("Starting comprehensive metrics system tests...")
            
            # Initialize
            await self.initialize()
            
            # Run tests
            tests = [
                ("Database Connection", self.test_database_connection),
                ("Polkadot Client", self.test_polkadot_client),
                ("Data Models", self.test_data_models),
                ("Metrics Data", self.test_metrics_data)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'='*50}")
                
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func()
                    else:
                        result = test_func()
                    
                    if result:
                        logger.success(f"✅ {test_name} test passed")
                        passed += 1
                    else:
                        logger.error(f"❌ {test_name} test failed")
                        
                except Exception as e:
                    logger.error(f"❌ {test_name} test failed with error: {e}")
            
            # Summary
            logger.info(f"\n{'='*50}")
            logger.info(f"TEST SUMMARY")
            logger.info(f"{'='*50}")
            logger.info(f"Tests passed: {passed}/{total}")
            
            if passed == total:
                logger.success("🎉 All tests passed! System is ready.")
                return True
            else:
                logger.error(f"⚠️  {total - passed} tests failed. Please check the issues above.")
                return False
                
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    tester = ComprehensiveMetricsTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            logger.success("\n🚀 Comprehensive Polkadot Metrics System is working correctly!")
            print("\n" + "="*60)
            print("✅ SYSTEM READY FOR USE")
            print("="*60)
            print("\nNext steps:")
            print("1. Start the metrics server: python polkadot_comprehensive_metrics_server.py")
            print("2. Open the dashboard: http://localhost:8008")
            print("3. Run data collection: python collect_comprehensive_polkadot_data.py")
            print("="*60)
        else:
            logger.error("\n❌ System tests failed. Please fix the issues before proceeding.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
