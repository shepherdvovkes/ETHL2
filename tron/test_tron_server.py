#!/usr/bin/env python3
"""
TRON Metrics Server Test Script
Tests the TRON monitoring system functionality
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.tron_quicknode_client import TronQuickNodeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_tron_client():
    """Test TRON QuickNode client functionality"""
    logger.info("🧪 Testing TRON QuickNode Client...")
    
    try:
        async with TronQuickNodeClient() as client:
            logger.info("✅ Client initialized successfully")
            
            # Test network performance metrics
            logger.info("📊 Testing network performance metrics...")
            network_metrics = await client.get_network_performance_metrics()
            if network_metrics:
                logger.info(f"✅ Network metrics collected: {len(network_metrics)} fields")
                logger.info(f"   Current block: {network_metrics.get('current_block', 'N/A')}")
                logger.info(f"   TPS: {network_metrics.get('transaction_throughput', 'N/A')}")
            else:
                logger.warning("⚠️ No network metrics returned")
            
            # Test economic metrics
            logger.info("💰 Testing economic metrics...")
            economic_metrics = await client.get_economic_metrics()
            if economic_metrics:
                logger.info(f"✅ Economic metrics collected: {len(economic_metrics)} fields")
                logger.info(f"   Total supply: {economic_metrics.get('total_supply', 'N/A')}")
                logger.info(f"   Transaction fees 24h: {economic_metrics.get('transaction_fees_24h', 'N/A')}")
            else:
                logger.warning("⚠️ No economic metrics returned")
            
            # Test DeFi metrics
            logger.info("🏦 Testing DeFi metrics...")
            defi_metrics = await client.get_defi_metrics()
            if defi_metrics:
                logger.info(f"✅ DeFi metrics collected: {len(defi_metrics)} fields")
                logger.info(f"   TVL: ${defi_metrics.get('total_value_locked', 'N/A'):,.0f}")
                logger.info(f"   Protocols: {defi_metrics.get('defi_protocols_count', 'N/A')}")
            else:
                logger.warning("⚠️ No DeFi metrics returned")
            
            # Test smart contract metrics
            logger.info("📜 Testing smart contract metrics...")
            contract_metrics = await client.get_smart_contract_metrics()
            if contract_metrics:
                logger.info(f"✅ Smart contract metrics collected: {len(contract_metrics)} fields")
                logger.info(f"   New contracts 24h: {contract_metrics.get('new_contracts_24h', 'N/A')}")
                logger.info(f"   TRC-20 tokens: {contract_metrics.get('trc20_tokens_count', 'N/A')}")
            else:
                logger.warning("⚠️ No smart contract metrics returned")
            
            # Test staking metrics
            logger.info("🔒 Testing staking metrics...")
            staking_metrics = await client.get_staking_metrics()
            if staking_metrics:
                logger.info(f"✅ Staking metrics collected: {len(staking_metrics)} fields")
                logger.info(f"   Total staked: {staking_metrics.get('total_staked', 'N/A')} TRX")
                logger.info(f"   Staking ratio: {staking_metrics.get('staking_ratio', 'N/A')}%")
            else:
                logger.warning("⚠️ No staking metrics returned")
            
            # Test user activity metrics
            logger.info("👥 Testing user activity metrics...")
            user_metrics = await client.get_user_activity_metrics()
            if user_metrics:
                logger.info(f"✅ User activity metrics collected: {len(user_metrics)} fields")
                logger.info(f"   Active addresses 24h: {user_metrics.get('active_addresses_24h', 'N/A'):,}")
                logger.info(f"   New addresses 24h: {user_metrics.get('new_addresses_24h', 'N/A'):,}")
            else:
                logger.warning("⚠️ No user activity metrics returned")
            
            # Test network health metrics
            logger.info("🛡️ Testing network health metrics...")
            health_metrics = await client.get_network_health_metrics()
            if health_metrics:
                logger.info(f"✅ Network health metrics collected: {len(health_metrics)} fields")
                logger.info(f"   Security score: {health_metrics.get('security_score', 'N/A')}/100")
                logger.info(f"   Decentralization: {health_metrics.get('decentralization_index', 'N/A')}/100")
            else:
                logger.warning("⚠️ No network health metrics returned")
            
            # Test comprehensive metrics
            logger.info("📈 Testing comprehensive metrics...")
            comprehensive_metrics = await client.get_comprehensive_metrics()
            if comprehensive_metrics:
                logger.info(f"✅ Comprehensive metrics collected")
                overall_scores = comprehensive_metrics.get('overall_scores', {})
                if overall_scores:
                    logger.info(f"   Overall score: {overall_scores.get('overall_score', 'N/A')}/100")
                    logger.info(f"   Risk level: {overall_scores.get('risk_level', 'N/A')}")
                logger.info(f"   Categories: {len(comprehensive_metrics)}")
            else:
                logger.warning("⚠️ No comprehensive metrics returned")
            
            logger.info("🎉 TRON client tests completed successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ TRON client test failed: {str(e)}")
        return False

async def test_server_endpoints():
    """Test server endpoints (if server is running)"""
    logger.info("🌐 Testing server endpoints...")
    
    import aiohttp
    
    base_url = "http://localhost:8008"
    endpoints = [
        "/health",
        "/metrics",
        "/metrics/network",
        "/metrics/economic",
        "/metrics/defi",
        "/metrics/smart-contracts",
        "/metrics/staking",
        "/metrics/user-activity",
        "/metrics/network-health",
        "/metrics/comprehensive"
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(f"{base_url}{endpoint}") as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"✅ {endpoint}: {response.status} - {len(str(data))} bytes")
                        else:
                            logger.warning(f"⚠️ {endpoint}: {response.status}")
                except Exception as e:
                    logger.warning(f"⚠️ {endpoint}: Connection failed - {str(e)}")
            
            logger.info("🎉 Server endpoint tests completed!")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ Server endpoint tests failed: {str(e)}")
        logger.info("💡 Make sure the TRON server is running on port 8008")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting TRON Metrics Server Tests...")
    logger.info(f"📅 Test started at: {datetime.now().isoformat()}")
    
    # Test client functionality
    client_success = await test_tron_client()
    
    # Test server endpoints
    server_success = await test_server_endpoints()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"TRON Client Tests: {'✅ PASSED' if client_success else '❌ FAILED'}")
    logger.info(f"Server Endpoint Tests: {'✅ PASSED' if server_success else '⚠️ SKIPPED/Failed'}")
    
    if client_success:
        logger.info("\n🎉 TRON Metrics Server is ready to use!")
        logger.info("📋 Next steps:")
        logger.info("   1. Start the server: python tron_metrics_server.py")
        logger.info("   2. Access dashboard: http://localhost:8008/dashboard")
        logger.info("   3. View API docs: http://localhost:8008/docs")
    else:
        logger.error("\n❌ TRON Metrics Server setup needs attention")
        logger.info("📋 Troubleshooting:")
        logger.info("   1. Check tron_config.env configuration")
        logger.info("   2. Verify QuickNode endpoint is accessible")
        logger.info("   3. Ensure all dependencies are installed")
    
    return client_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
