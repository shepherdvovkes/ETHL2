#!/usr/bin/env python3
"""
Test Runner for Avalanche Network Real-Time Metrics Server
Runs comprehensive tests to validate real data collection and functionality
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from loguru import logger

def setup_logging():
    """Setup logging for test runner"""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/test_runner_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

def check_requirements():
    """Check if test requirements are installed"""
    logger.info("🔍 Checking test requirements...")
    
    required_packages = [
        "pytest",
        "pytest-asyncio",
        "aiohttp",
        "fastapi",
        "httpx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All test requirements satisfied")
    return True

def check_environment():
    """Check environment for testing"""
    logger.info("🔧 Checking test environment...")
    
    # Check if config.env exists
    if not os.path.exists("config.env"):
        logger.warning("⚠️  config.env not found. Some tests may fail without API keys.")
        logger.info("Create config.env from avalanche_config.env template for full testing")
    
    # Check if logs directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
        logger.info("✅ Created logs directory")
    
    # Check if src directory exists
    if not os.path.exists("src"):
        logger.error("❌ src directory not found. Please run from project root.")
        return False
    
    logger.info("✅ Test environment check complete")
    return True

def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """Run tests based on type"""
    logger.info(f"🧪 Running {test_type} tests...")
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test file
    if test_type == "all":
        cmd.append("test_avalanche_server.py")
    elif test_type == "data_collection":
        cmd.extend(["-k", "test_avalanche_data_collection or test_external_api_connectivity"])
    elif test_type == "monitoring":
        cmd.extend(["-k", "test_monitoring_system"])
    elif test_type == "api":
        cmd.extend(["-k", "test_api_server"])
    elif test_type == "performance":
        cmd.extend(["-k", "test_performance"])
    elif test_type == "validation":
        cmd.extend(["-k", "test_data_validation"])
    else:
        logger.error(f"Unknown test type: {test_type}")
        return False
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add markers for real data tests
    cmd.extend(["-m", "real_data"])
    
    # Add timeout
    cmd.extend(["--timeout=300"])
    
    # Run tests
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"✅ Tests completed in {duration:.2f} seconds")
        
        return result.returncode == 0
    
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

def run_specific_test_class(test_class):
    """Run specific test class"""
    logger.info(f"🧪 Running {test_class} tests...")
    
    cmd = [
        "python", "-m", "pytest",
        f"test_avalanche_server.py::{test_class}",
        "-v",
        "--timeout=300"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"✅ {test_class} tests completed in {duration:.2f} seconds")
        
        return result.returncode == 0
    
    except Exception as e:
        logger.error(f"❌ {test_class} test execution failed: {e}")
        return False

def run_individual_test(test_name):
    """Run individual test"""
    logger.info(f"🧪 Running {test_name} test...")
    
    cmd = [
        "python", "-m", "pytest",
        f"test_avalanche_server.py::{test_name}",
        "-v",
        "--timeout=300"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"✅ {test_name} test completed in {duration:.2f} seconds")
        
        return result.returncode == 0
    
    except Exception as e:
        logger.error(f"❌ {test_name} test execution failed: {e}")
        return False

def run_connectivity_tests():
    """Run basic connectivity tests"""
    logger.info("🌐 Running connectivity tests...")
    
    tests = [
        "TestExternalAPIConnectivity::test_coingecko_api_connectivity",
        "TestExternalAPIConnectivity::test_avalanche_rpc_connectivity",
        "TestExternalAPIConnectivity::test_defillama_api_connectivity"
    ]
    
    results = []
    for test in tests:
        success = run_individual_test(test)
        results.append((test, success))
    
    # Print summary
    logger.info("\n📊 Connectivity Test Results:")
    for test, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status} {test}")
    
    return all(success for _, success in results)

def run_data_collection_tests():
    """Run data collection tests"""
    logger.info("📊 Running data collection tests...")
    
    tests = [
        "TestAvalancheDataCollection::test_network_performance_collection",
        "TestAvalancheDataCollection::test_economic_data_collection",
        "TestAvalancheDataCollection::test_defi_metrics_collection",
        "TestAvalancheDataCollection::test_subnet_data_collection",
        "TestAvalancheDataCollection::test_security_status_collection",
        "TestAvalancheDataCollection::test_user_behavior_collection",
        "TestAvalancheDataCollection::test_competitive_position_collection",
        "TestAvalancheDataCollection::test_technical_health_collection",
        "TestAvalancheDataCollection::test_risk_indicators_collection",
        "TestAvalancheDataCollection::test_macro_environment_collection",
        "TestAvalancheDataCollection::test_ecosystem_health_collection"
    ]
    
    results = []
    for test in tests:
        success = run_individual_test(test)
        results.append((test, success))
    
    # Print summary
    logger.info("\n📊 Data Collection Test Results:")
    for test, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status} {test}")
    
    return all(success for _, success in results)

def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger.info("🚀 Running comprehensive test suite...")
    
    test_suites = [
        ("Connectivity Tests", run_connectivity_tests),
        ("Data Collection Tests", run_data_collection_tests),
        ("Monitoring System Tests", lambda: run_specific_test_class("TestMonitoringSystem")),
        ("API Server Tests", lambda: run_specific_test_class("TestAPIServer")),
        ("Data Validation Tests", lambda: run_specific_test_class("TestDataValidation")),
        ("Performance Tests", lambda: run_specific_test_class("TestPerformance"))
    ]
    
    results = []
    for suite_name, suite_func in test_suites:
        logger.info(f"\n🧪 Running {suite_name}...")
        start_time = time.time()
        success = suite_func()
        end_time = time.time()
        duration = end_time - start_time
        
        results.append((suite_name, success, duration))
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {suite_name} completed in {duration:.2f} seconds")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("📊 COMPREHENSIVE TEST RESULTS")
    logger.info("="*80)
    
    total_duration = sum(duration for _, _, duration in results)
    passed_suites = sum(1 for _, success, _ in results if success)
    total_suites = len(results)
    
    for suite_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {suite_name:<30} {duration:>8.2f}s")
    
    logger.info("-" * 80)
    logger.info(f"Total Suites: {total_suites}")
    logger.info(f"Passed: {passed_suites}")
    logger.info(f"Failed: {total_suites - passed_suites}")
    logger.info(f"Total Duration: {total_duration:.2f}s")
    logger.info(f"Success Rate: {(passed_suites/total_suites)*100:.1f}%")
    logger.info("="*80)
    
    return passed_suites == total_suites

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Avalanche Server Test Runner")
    parser.add_argument("--type", choices=[
        "all", "connectivity", "data_collection", "monitoring", 
        "api", "performance", "validation", "comprehensive"
    ], default="comprehensive", help="Type of tests to run")
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    parser.add_argument("--test", help="Run specific test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 Starting Avalanche Server Test Runner")
    logger.info(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements and environment
    if not check_requirements():
        logger.error("❌ Requirements check failed")
        return False
    
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Run tests based on arguments
    success = False
    
    if args.test:
        success = run_individual_test(args.test)
    elif args.test_class:
        success = run_specific_test_class(args.test_class)
    elif args.type == "connectivity":
        success = run_connectivity_tests()
    elif args.type == "data_collection":
        success = run_data_collection_tests()
    elif args.type == "comprehensive":
        success = run_comprehensive_test()
    else:
        success = run_tests(args.type, args.verbose, args.coverage, args.parallel)
    
    # Final result
    if success:
        logger.info("🎉 All tests passed successfully!")
        return True
    else:
        logger.error("❌ Some tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
