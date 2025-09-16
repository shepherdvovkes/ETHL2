#!/usr/bin/env python3
"""
Test Solana System
Comprehensive test suite for the Solana data collection and metrics system
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SolanaSystemTester:
    """Test suite for Solana system components"""
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.metrics_url = "http://localhost:9091"
        self.test_results = {}
    
    async def test_database_connection(self) -> bool:
        """Test database connections"""
        logger.info("🔍 Testing database connections...")
        
        try:
            # Test main database
            conn = sqlite3.connect('solana_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM solana_blocks")
            main_count = cursor.fetchone()[0]
            conn.close()
            
            # Test archive database
            conn = sqlite3.connect('solana_archive_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM solana_archive_blocks")
            archive_count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"✅ Main database: {main_count} blocks")
            logger.info(f"✅ Archive database: {archive_count} blocks")
            
            self.test_results['database_connection'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            self.test_results['database_connection'] = False
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test API endpoints"""
        logger.info("🔍 Testing API endpoints...")
        
        endpoints = [
            '/',
            '/api/blocks',
            '/api/transactions',
            '/api/network_metrics',
            '/api/validators',
            '/api/programs',
            '/api/archive',
            '/api/stats'
        ]
        
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"✅ {endpoint}: {response.status}")
                            success_count += 1
                        else:
                            logger.error(f"❌ {endpoint}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"❌ {endpoint}: {e}")
        
        success_rate = success_count / len(endpoints)
        self.test_results['api_endpoints'] = success_rate > 0.8
        
        logger.info(f"📊 API endpoints test: {success_count}/{len(endpoints)} passed")
        return success_rate > 0.8
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection"""
        logger.info("🔍 Testing WebSocket connection...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"{self.base_url.replace('http', 'ws')}/ws") as ws:
                    # Send test command
                    await ws.send_str(json.dumps({
                        'command': 'get_stats'
                    }))
                    
                    # Wait for response
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get('type') == 'stats':
                                logger.info("✅ WebSocket connection successful")
                                self.test_results['websocket'] = True
                                return True
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"❌ WebSocket error: {ws.exception()}")
                            self.test_results['websocket'] = False
                            return False
                            
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            self.test_results['websocket'] = False
            return False
    
    async def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint"""
        logger.info("🔍 Testing metrics endpoint...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.metrics_url}/metrics") as response:
                    if response.status == 200:
                        text = await response.text()
                        if 'solana_blocks_total' in text:
                            logger.info("✅ Metrics endpoint working")
                            self.test_results['metrics'] = True
                            return True
                        else:
                            logger.error("❌ Metrics endpoint missing expected data")
                            self.test_results['metrics'] = False
                            return False
                    else:
                        logger.error(f"❌ Metrics endpoint returned {response.status}")
                        self.test_results['metrics'] = False
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Metrics test failed: {e}")
            self.test_results['metrics'] = False
            return False
    
    async def test_archive_data_access(self) -> bool:
        """Test archive data access"""
        logger.info("🔍 Testing archive data access...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test archive blocks
                url = f"{self.base_url}/api/archive?type=blocks&start_slot=1&end_slot=100"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Archive blocks: {len(data)} blocks retrieved")
                        
                        # Test archive progress
                        url = f"{self.base_url}/api/archive?type=progress"
                        async with session.get(url) as response:
                            if response.status == 200:
                                progress_data = await response.json()
                                logger.info(f"✅ Archive progress: {len(progress_data)} progress records")
                                self.test_results['archive_data'] = True
                                return True
                            else:
                                logger.error(f"❌ Archive progress: {response.status}")
                                self.test_results['archive_data'] = False
                                return False
                    else:
                        logger.error(f"❌ Archive blocks: {response.status}")
                        self.test_results['archive_data'] = False
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Archive data test failed: {e}")
            self.test_results['archive_data'] = False
            return False
    
    async def test_data_collection_simulation(self) -> bool:
        """Test data collection simulation"""
        logger.info("🔍 Testing data collection simulation...")
        
        try:
            # Test if we can get current stats
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        
                        # Check if we have some data
                        main_blocks = stats.get('main_database', {}).get('block_count', 0)
                        archive_blocks = stats.get('archive_database', {}).get('block_count', 0)
                        
                        logger.info(f"✅ Main database: {main_blocks} blocks")
                        logger.info(f"✅ Archive database: {archive_blocks} blocks")
                        
                        # If we have any data, consider it successful
                        if main_blocks > 0 or archive_blocks > 0:
                            self.test_results['data_collection'] = True
                            return True
                        else:
                            logger.warning("⚠️ No data found in databases")
                            self.test_results['data_collection'] = False
                            return False
                    else:
                        logger.error(f"❌ Stats endpoint: {response.status}")
                        self.test_results['data_collection'] = False
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Data collection test failed: {e}")
            self.test_results['data_collection'] = False
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        logger.info("🚀 Starting Solana system tests...")
        
        tests = [
            self.test_database_connection,
            self.test_api_endpoints,
            self.test_websocket_connection,
            self.test_metrics_endpoint,
            self.test_archive_data_access,
            self.test_data_collection_simulation
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"❌ Test {test.__name__} failed with exception: {e}")
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"""
        📊 Test Results Summary:
        Total Tests: {total_tests}
        Passed: {passed_tests}
        Failed: {total_tests - passed_tests}
        Success Rate: {success_rate:.1%}
        
        Detailed Results:
        """)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        return self.test_results

async def main():
    """Main test function"""
    tester = SolanaSystemTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    success_rate = sum(1 for result in results.values() if result) / len(results)
    if success_rate >= 0.8:
        logger.info("🎉 System tests passed!")
        exit(0)
    else:
        logger.error("❌ System tests failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())