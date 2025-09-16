#!/usr/bin/env python3
"""
Test Script for Polkadot Archive Data Collector
===============================================

Simple test script to verify the archive collector functionality.
"""

import asyncio
import sys
import os
from loguru import logger

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from polkadot_archive_data_collector import QuickNodePolkadotArchiveClient, CollectionConfig
from polkadot_archive_config import get_config

async def test_quicknode_connection():
    """Test QuickNode connection and basic functionality"""
    logger.info("Testing QuickNode connection...")
    
    config = CollectionConfig(
        max_workers=5,  # Use fewer workers for testing
        days_back=1,    # Just 1 day for testing
        sample_rate=1000,  # Sample every 1000th block
        batch_size=10,
        rate_limit_delay=0.2
    )
    
    try:
        async with QuickNodePolkadotArchiveClient(config) as client:
            # Test basic connection
            current_block = await client.get_current_block()
            logger.info(f"Current block: {current_block}")
            
            if current_block == 0:
                logger.error("Failed to get current block - check QuickNode connection")
                return False
            
            # Test block range calculation
            start_block, end_block = await client.get_block_range(1)
            logger.info(f"Block range for 1 day: {start_block} to {end_block}")
            
            # Test collecting a single block
            logger.info("Testing single block collection...")
            block_data = await client.collect_block_data(current_block - 1000)  # Get a recent block
            if block_data:
                logger.success(f"Successfully collected block data: {block_data.get('block_number')}")
            else:
                logger.error("Failed to collect block data")
                return False
            
            # Test network metrics collection
            logger.info("Testing network metrics collection...")
            staking_data = await client.collect_staking_data()
            if staking_data:
                logger.success("Successfully collected staking data")
            else:
                logger.warning("Failed to collect staking data")
            
            parachain_data = await client.collect_parachain_data()
            if parachain_data:
                logger.success("Successfully collected parachain data")
            else:
                logger.warning("Failed to collect parachain data")
            
            governance_data = await client.collect_governance_data()
            if governance_data:
                logger.success("Successfully collected governance data")
            else:
                logger.warning("Failed to collect governance data")
            
            return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

async def test_database_functionality():
    """Test database functionality"""
    logger.info("Testing database functionality...")
    
    from polkadot_archive_data_collector import PolkadotArchiveDatabase
    
    try:
        # Test database initialization
        db = PolkadotArchiveDatabase("test_polkadot_archive.db")
        logger.success("Database initialized successfully")
        
        # Test storing sample data
        sample_block_metrics = {
            'block_number': 12345678,
            'timestamp': '2024-01-01T00:00:00Z',
            'extrinsics_count': 50,
            'events_count': 100,
            'block_size': 1024,
            'validator_count': 1000,
            'finalization_time': 6.0,
            'parachain_blocks': 5,
            'cross_chain_messages': 10
        }
        
        db.store_block_metrics(sample_block_metrics)
        logger.success("Block metrics stored successfully")
        
        sample_staking_data = {
            'timestamp': '2024-01-01T00:00:00Z',
            'validators': ['validator1', 'validator2'],
            'nominators': ['nominator1', 'nominator2'],
            'active_era': {'index': 1234}
        }
        
        db.store_staking_data(sample_staking_data)
        logger.success("Staking data stored successfully")
        
        # Clean up test database
        os.remove("test_polkadot_archive.db")
        logger.info("Test database cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

async def test_configuration_system():
    """Test configuration system"""
    logger.info("Testing configuration system...")
    
    try:
        # Test predefined configurations
        configs_to_test = ['quick_test', 'monthly', 'yearly']
        
        for config_name in configs_to_test:
            config = get_config(config_name)
            logger.info(f"Configuration '{config_name}': {config.days_back} days, {config.max_workers} workers")
        
        # Test custom configuration
        from polkadot_archive_config import create_custom_config
        custom_config = create_custom_config(
            days_back=30,
            max_workers=15,
            sample_rate=5
        )
        logger.info(f"Custom configuration: {custom_config.days_back} days, {custom_config.max_workers} workers")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Starting Polkadot Archive Collector Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Database Functionality", test_database_functionality),
        ("QuickNode Connection", test_quicknode_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.success(f"✅ {test_name} passed")
            else:
                logger.error(f"❌ {test_name} failed")
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary")
    logger.info("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("🎉 All tests passed! The archive collector is ready to use.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the issues above.")
        return False

def main():
    """Main function"""
    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.warning("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
