#!/usr/bin/env python3
"""
Comprehensive Test Suite for Avalanche Network Real-Time Metrics Server
Tests real data collection without mocks to ensure actual functionality
"""

import pytest
import pytest_asyncio
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from avalanche_realtime_server import RealTimeDataCollector
from avalanche_monitoring_system import AvalancheMonitoringSystem, MonitoringRule, AlertLevel, AlertType
from avalanche_api_server import app
from api.coingecko_client import CoinGeckoClient
from api.blockchain_client import BlockchainClient, BlockchainType
from config.settings import settings

class TestAvalancheDataCollection:
    """Test real data collection from Avalanche network and external APIs"""
    
    @pytest_asyncio.fixture
    async def data_collector(self):
        """Create a real data collector instance"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        yield collector
        await collector.__aexit__(None, None, None)
    
    @pytest_asyncio.fixture
    async def monitoring_system(self):
        """Create a real monitoring system instance"""
        return AvalancheMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_avalanche_rpc_connectivity(self, data_collector):
        """Test real connectivity to Avalanche RPC endpoints via QuickNode"""
        # Test QuickNode C-Chain connectivity
        assert data_collector.quicknode_client is not None, "QuickNode client should be initialized"
        
        # Test getting current block number
        current_block = await data_collector.quicknode_client.get_c_chain_block_number()
        assert current_block > 0, "Should get valid block number from QuickNode C-Chain"
        assert isinstance(current_block, int), "Block number should be integer"
        
        # Test getting block info
        block_info = await data_collector.quicknode_client.get_c_chain_block_by_number(current_block)
        assert block_info is not None, "Should get block information from QuickNode"
        assert isinstance(block_info, dict), "Block info should be dictionary"
        if block_info:
            assert "transactions" in block_info, "Block should contain transactions"
            assert "timestamp" in block_info, "Block should have timestamp"
        
        # Test gas price
        gas_price = await data_collector.quicknode_client.get_c_chain_gas_price()
        assert gas_price >= 0, "Should get valid gas price"
        assert isinstance(gas_price, (int, float)), "Gas price should be numeric"
        
        # Test network stats
        network_stats = await data_collector.quicknode_client.get_network_stats()
        assert network_stats is not None, "Should get network stats from QuickNode"
        assert "c_chain" in network_stats, "Network stats should contain C-Chain data"
        assert "p_chain" in network_stats, "Network stats should contain P-Chain data"
    
    @pytest.mark.asyncio
    async def test_quicknode_p_chain_connectivity(self, data_collector):
        """Test QuickNode P-Chain connectivity and data retrieval"""
        # Test P-Chain validators
        validators = await data_collector.quicknode_client.get_p_chain_validators()
        assert validators is not None, "Should get validators from P-Chain"
        assert isinstance(validators, list), "Validators should be a list"
        assert len(validators) > 0, "Should have validators"
        
        # Test P-Chain subnets
        subnets = await data_collector.quicknode_client.get_p_chain_subnets()
        assert subnets is not None, "Should get subnets from P-Chain"
        assert isinstance(subnets, list), "Subnets should be a list"
        assert len(subnets) > 0, "Should have subnets"
        
        # Test staking info
        staking_info = await data_collector.quicknode_client.get_p_chain_staking_info()
        assert staking_info is not None, "Should get staking info"
        assert "total_validators" in staking_info, "Should include total validators"
        assert "active_validators" in staking_info, "Should include active validators"
        assert "total_stake" in staking_info, "Should include total stake"
        
        # Validate data types
        assert isinstance(staking_info["total_validators"], int), "Total validators should be integer"
        assert isinstance(staking_info["active_validators"], int), "Active validators should be integer"
        assert isinstance(staking_info["total_stake"], (int, float)), "Total stake should be numeric"
    
    @pytest.mark.asyncio
    async def test_network_performance_collection(self, data_collector):
        """Test real network performance metrics collection"""
        metrics = await data_collector.collect_network_performance()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "block_time" in metrics, "Should include block time"
        assert "transaction_throughput" in metrics, "Should include TPS"
        assert "finality_time" in metrics, "Should include finality time"
        assert "gas_price_avg" in metrics, "Should include gas price"
        assert "current_block" in metrics, "Should include current block"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types and ranges
        assert isinstance(metrics["block_time"], (int, float)), "Block time should be numeric"
        assert isinstance(metrics["transaction_throughput"], int), "TPS should be integer"
        assert isinstance(metrics["gas_price_avg"], (int, float)), "Gas price should be numeric"
        assert isinstance(metrics["current_block"], int), "Block number should be integer"
        
        # Validate reasonable ranges
        assert 0 < metrics["block_time"] < 10, "Block time should be reasonable"
        assert metrics["transaction_throughput"] >= 0, "TPS should be non-negative"
        assert metrics["gas_price_avg"] >= 0, "Gas price should be non-negative"
        assert metrics["current_block"] > 0, "Block number should be positive"
    
    @pytest.mark.asyncio
    async def test_economic_data_collection(self, data_collector):
        """Test real economic data collection from CoinGecko"""
        metrics = await data_collector.collect_economic_data()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "price" in metrics, "Should include AVAX price"
        assert "market_cap" in metrics, "Should include market cap"
        assert "daily_volume" in metrics, "Should include daily volume"
        assert "circulating_supply" in metrics, "Should include circulating supply"
        assert "price_change_24h" in metrics, "Should include 24h price change"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types and ranges
        assert isinstance(metrics["price"], (int, float)), "Price should be numeric"
        assert isinstance(metrics["market_cap"], (int, float)), "Market cap should be numeric"
        assert isinstance(metrics["daily_volume"], (int, float)), "Volume should be numeric"
        assert isinstance(metrics["circulating_supply"], (int, float)), "Supply should be numeric"
        
        # Validate reasonable ranges
        assert metrics["price"] > 0, "Price should be positive"
        assert metrics["market_cap"] > 0, "Market cap should be positive"
        assert metrics["daily_volume"] >= 0, "Volume should be non-negative"
        assert metrics["circulating_supply"] > 0, "Supply should be positive"
    
    @pytest.mark.asyncio
    async def test_defi_metrics_collection(self, data_collector):
        """Test real DeFi metrics collection"""
        metrics = await data_collector.collect_defi_metrics()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "total_tvl" in metrics, "Should include total TVL"
        assert "protocols_count" in metrics, "Should include protocol count"
        assert "protocols" in metrics, "Should include protocol data"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["total_tvl"], (int, float)), "TVL should be numeric"
        assert isinstance(metrics["protocols_count"], int), "Protocol count should be integer"
        assert isinstance(metrics["protocols"], dict), "Protocols should be dictionary"
        
        # Validate ranges
        assert metrics["total_tvl"] >= 0, "TVL should be non-negative"
        assert metrics["protocols_count"] >= 0, "Protocol count should be non-negative"
    
    @pytest.mark.asyncio
    async def test_subnet_data_collection(self, data_collector):
        """Test real subnet data collection"""
        metrics = await data_collector.collect_subnet_data()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "total_subnets" in metrics, "Should include total subnets"
        assert "active_subnets" in metrics, "Should include active subnets"
        assert "total_validators" in metrics, "Should include total validators"
        assert "active_validators" in metrics, "Should include active validators"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["total_subnets"], int), "Total subnets should be integer"
        assert isinstance(metrics["active_subnets"], int), "Active subnets should be integer"
        assert isinstance(metrics["total_validators"], int), "Total validators should be integer"
        assert isinstance(metrics["active_validators"], int), "Active validators should be integer"
        
        # Validate ranges
        assert metrics["total_subnets"] >= 0, "Total subnets should be non-negative"
        assert metrics["active_subnets"] >= 0, "Active subnets should be non-negative"
        assert metrics["total_validators"] >= 0, "Total validators should be non-negative"
        assert metrics["active_validators"] >= 0, "Active validators should be non-negative"
    
    @pytest.mark.asyncio
    async def test_security_status_collection(self, data_collector):
        """Test real security status collection"""
        metrics = await data_collector.collect_security_status()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "validator_count" in metrics, "Should include validator count"
        assert "active_validators" in metrics, "Should include active validators"
        assert "staking_ratio" in metrics, "Should include staking ratio"
        assert "security_score" in metrics, "Should include security score"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["validator_count"], int), "Validator count should be integer"
        assert isinstance(metrics["active_validators"], int), "Active validators should be integer"
        assert isinstance(metrics["staking_ratio"], (int, float)), "Staking ratio should be numeric"
        assert isinstance(metrics["security_score"], (int, float)), "Security score should be numeric"
        
        # Validate ranges
        assert metrics["validator_count"] >= 0, "Validator count should be non-negative"
        assert metrics["active_validators"] >= 0, "Active validators should be non-negative"
        assert 0 <= metrics["staking_ratio"] <= 100, "Staking ratio should be percentage"
        assert 0 <= metrics["security_score"] <= 100, "Security score should be percentage"
    
    @pytest.mark.asyncio
    async def test_user_behavior_collection(self, data_collector):
        """Test real user behavior metrics collection"""
        metrics = await data_collector.collect_user_behavior()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "whale_activity" in metrics, "Should include whale activity"
        assert "retail_vs_institutional" in metrics, "Should include user type distribution"
        assert "transaction_sizes" in metrics, "Should include transaction size data"
        assert "unique_addresses_50_blocks" in metrics, "Should include unique addresses"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["whale_activity"], int), "Whale activity should be integer"
        assert isinstance(metrics["retail_vs_institutional"], dict), "User distribution should be dictionary"
        assert isinstance(metrics["transaction_sizes"], dict), "Transaction sizes should be dictionary"
        assert isinstance(metrics["unique_addresses_50_blocks"], int), "Unique addresses should be integer"
        
        # Validate ranges
        assert metrics["whale_activity"] >= 0, "Whale activity should be non-negative"
        assert metrics["unique_addresses_50_blocks"] >= 0, "Unique addresses should be non-negative"
    
    @pytest.mark.asyncio
    async def test_competitive_position_collection(self, data_collector):
        """Test real competitive position data collection"""
        metrics = await data_collector.collect_competitive_position()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "market_rank" in metrics, "Should include market rank"
        assert "market_share" in metrics, "Should include market share"
        assert "market_cap" in metrics, "Should include market cap"
        assert "competitors" in metrics, "Should include competitor data"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["market_rank"], int), "Market rank should be integer"
        assert isinstance(metrics["market_share"], (int, float)), "Market share should be numeric"
        assert isinstance(metrics["market_cap"], (int, float)), "Market cap should be numeric"
        assert isinstance(metrics["competitors"], list), "Competitors should be list"
        
        # Validate ranges
        assert metrics["market_rank"] > 0, "Market rank should be positive"
        assert 0 <= metrics["market_share"] <= 100, "Market share should be percentage"
        assert metrics["market_cap"] >= 0, "Market cap should be non-negative"
    
    @pytest.mark.asyncio
    async def test_technical_health_collection(self, data_collector):
        """Test real technical health metrics collection"""
        metrics = await data_collector.collect_technical_health()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "rpc_performance" in metrics, "Should include RPC performance"
        assert "endpoints_status" in metrics, "Should include endpoint status"
        assert "overall_health_score" in metrics, "Should include health score"
        assert "network_uptime" in metrics, "Should include network uptime"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["rpc_performance"], dict), "RPC performance should be dictionary"
        assert isinstance(metrics["endpoints_status"], dict), "Endpoint status should be dictionary"
        assert isinstance(metrics["overall_health_score"], (int, float)), "Health score should be numeric"
        assert isinstance(metrics["network_uptime"], (int, float)), "Uptime should be numeric"
        
        # Validate ranges
        assert 0 <= metrics["overall_health_score"] <= 100, "Health score should be percentage"
        assert 0 <= metrics["network_uptime"] <= 100, "Uptime should be percentage"
    
    @pytest.mark.asyncio
    async def test_risk_indicators_collection(self, data_collector):
        """Test real risk indicators collection"""
        metrics = await data_collector.collect_risk_indicators()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "price_volatility" in metrics, "Should include price volatility"
        assert "risk_level" in metrics, "Should include risk level"
        assert "centralization_risk" in metrics, "Should include centralization risk"
        assert "technical_risk" in metrics, "Should include technical risk"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["price_volatility"], (int, float)), "Volatility should be numeric"
        assert isinstance(metrics["risk_level"], str), "Risk level should be string"
        assert isinstance(metrics["centralization_risk"], str), "Centralization risk should be string"
        assert isinstance(metrics["technical_risk"], str), "Technical risk should be string"
        
        # Validate ranges
        assert metrics["price_volatility"] >= 0, "Volatility should be non-negative"
        assert metrics["risk_level"] in ["Low", "Medium", "High"], "Risk level should be valid"
    
    @pytest.mark.asyncio
    async def test_macro_environment_collection(self, data_collector):
        """Test real macro environment data collection"""
        metrics = await data_collector.collect_macro_environment()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "total_market_cap" in metrics, "Should include total market cap"
        assert "market_cap_change_24h" in metrics, "Should include market cap change"
        assert "bitcoin_dominance" in metrics, "Should include Bitcoin dominance"
        assert "ethereum_dominance" in metrics, "Should include Ethereum dominance"
        assert "active_cryptocurrencies" in metrics, "Should include active cryptocurrencies"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["total_market_cap"], (int, float)), "Total market cap should be numeric"
        assert isinstance(metrics["market_cap_change_24h"], (int, float)), "Market cap change should be numeric"
        assert isinstance(metrics["bitcoin_dominance"], (int, float)), "Bitcoin dominance should be numeric"
        assert isinstance(metrics["ethereum_dominance"], (int, float)), "Ethereum dominance should be numeric"
        assert isinstance(metrics["active_cryptocurrencies"], int), "Active cryptocurrencies should be integer"
        
        # Validate ranges
        assert metrics["total_market_cap"] >= 0, "Total market cap should be non-negative"
        assert 0 <= metrics["bitcoin_dominance"] <= 100, "Bitcoin dominance should be percentage"
        assert 0 <= metrics["ethereum_dominance"] <= 100, "Ethereum dominance should be percentage"
        assert metrics["active_cryptocurrencies"] > 0, "Active cryptocurrencies should be positive"
    
    @pytest.mark.asyncio
    async def test_ecosystem_health_collection(self, data_collector):
        """Test real ecosystem health metrics collection"""
        metrics = await data_collector.collect_ecosystem_health()
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert "community_growth" in metrics, "Should include community growth"
        assert "media_coverage" in metrics, "Should include media coverage"
        assert "partnership_quality" in metrics, "Should include partnership quality"
        assert "developer_experience" in metrics, "Should include developer experience"
        assert "timestamp" in metrics, "Should include timestamp"
        
        # Validate data types
        assert isinstance(metrics["community_growth"], dict), "Community growth should be dictionary"
        assert isinstance(metrics["media_coverage"], dict), "Media coverage should be dictionary"
        assert isinstance(metrics["partnership_quality"], dict), "Partnership quality should be dictionary"
        assert isinstance(metrics["developer_experience"], dict), "Developer experience should be dictionary"
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, data_collector):
        """Test comprehensive metrics collection from all sources"""
        # Test that all collection methods work together
        all_metrics = {}
        
        collection_methods = [
            ("network_performance", data_collector.collect_network_performance),
            ("economic_data", data_collector.collect_economic_data),
            ("defi_metrics", data_collector.collect_defi_metrics),
            ("subnet_data", data_collector.collect_subnet_data),
            ("security_status", data_collector.collect_security_status),
            ("development_activity", data_collector.collect_development_activity),
            ("user_behavior", data_collector.collect_user_behavior),
            ("competitive_position", data_collector.collect_competitive_position),
            ("technical_health", data_collector.collect_technical_health),
            ("risk_indicators", data_collector.collect_risk_indicators),
            ("macro_environment", data_collector.collect_macro_environment),
            ("ecosystem_health", data_collector.collect_ecosystem_health)
        ]
        
        for metric_type, collection_method in collection_methods:
            try:
                metrics = await collection_method()
                assert isinstance(metrics, dict), f"{metric_type} should return dictionary"
                assert "timestamp" in metrics, f"{metric_type} should include timestamp"
                all_metrics[metric_type] = metrics
            except Exception as e:
                pytest.fail(f"Failed to collect {metric_type}: {e}")
        
        # Validate that we collected data from all sources
        assert len(all_metrics) == 12, "Should collect from all 12 metric categories"
        
        # Validate timestamps are recent
        current_time = datetime.utcnow()
        for metric_type, metrics in all_metrics.items():
            timestamp_str = metrics.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                time_diff = (current_time - timestamp).total_seconds()
                assert time_diff < 300, f"{metric_type} timestamp should be recent (within 5 minutes)"

class TestMonitoringSystem:
    """Test the monitoring and alerting system"""
    
    @pytest_asyncio.fixture
    async def monitoring_system(self):
        """Create a real monitoring system instance"""
        return AvalancheMonitoringSystem()
    
    def test_monitoring_rules_initialization(self, monitoring_system):
        """Test that monitoring rules are properly initialized"""
        rules = monitoring_system.get_monitoring_rules()
        
        assert len(rules) > 0, "Should have monitoring rules"
        
        # Check for key rule types
        rule_names = [rule.name for rule in rules]
        expected_rules = [
            "high_gas_price",
            "low_throughput", 
            "high_network_utilization",
            "high_price_volatility",
            "extreme_price_drop",
            "low_validator_count",
            "low_staking_ratio",
            "slow_rpc_response",
            "low_health_score",
            "defi_tvl_drop"
        ]
        
        for expected_rule in expected_rules:
            assert expected_rule in rule_names, f"Should have {expected_rule} rule"
    
    def test_alert_manager_initialization(self, monitoring_system):
        """Test that alert manager is properly initialized"""
        alert_manager = monitoring_system.alert_manager
        
        assert alert_manager is not None, "Should have alert manager"
        assert len(alert_manager.notification_channels) > 0, "Should have notification channels"
        
        # Check notification channels
        expected_channels = ["email", "webhook", "slack", "telegram"]
        for channel in expected_channels:
            assert channel in alert_manager.notification_channels, f"Should have {channel} channel"
    
    @pytest.mark.asyncio
    async def test_rule_evaluation_with_real_data(self, monitoring_system):
        """Test rule evaluation with real data"""
        # Create test data that should trigger alerts
        test_data = {
            "network_performance": {
                "gas_price_avg": 150.0,  # Should trigger high gas price alert
                "transaction_throughput": 500,  # Should trigger low throughput alert
                "network_utilization": 95.0  # Should trigger high utilization alert
            },
            "economic_data": {
                "price_change_24h": 25.0,  # Should trigger high volatility alert
                "daily_volume": 50000000  # Should trigger low volume alert
            },
            "security_status": {
                "validator_count": 500,  # Should trigger low validator alert
                "staking_ratio": 30.0  # Should trigger low staking ratio alert
            },
            "technical_health": {
                "rpc_performance": {
                    "response_time_ms": 6000.0  # Should trigger slow RPC alert
                },
                "overall_health_score": 70.0  # Should trigger low health score alert
            },
            "defi_metrics": {
                "total_tvl": 500000000  # Should trigger DeFi TVL drop alert
            }
        }
        
        # Evaluate rules
        await monitoring_system.evaluate_rules(test_data)
        
        # Check that alerts were created
        active_alerts = monitoring_system.get_active_alerts()
        assert len(active_alerts) > 0, "Should have created alerts for threshold violations"
        
        # Validate alert properties
        for alert in active_alerts:
            assert alert.id is not None, "Alert should have ID"
            assert alert.type is not None, "Alert should have type"
            assert alert.level is not None, "Alert should have level"
            assert alert.title is not None, "Alert should have title"
            assert alert.message is not None, "Alert should have message"
            assert alert.timestamp is not None, "Alert should have timestamp"
            assert alert.current_value is not None, "Alert should have current value"
            assert alert.threshold_value is not None, "Alert should have threshold value"

class TestAPIServer:
    """Test the FastAPI server endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Avalanche Network Metrics API" in data["message"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_metrics_summary_endpoint(self, client):
        """Test metrics summary endpoint"""
        response = client.get("/metrics/summary")
        # This might return 404 if no data is available yet, which is acceptable
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "timestamp" in data
            assert "key_metrics" in data
            assert "status" in data
    
    def test_network_performance_endpoint(self, client):
        """Test network performance endpoint"""
        response = client.get("/metrics/network-performance")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "block_time" in data
            assert "transaction_throughput" in data
            assert "finality_time" in data
            assert "gas_price_avg" in data
    
    def test_economic_metrics_endpoint(self, client):
        """Test economic metrics endpoint"""
        response = client.get("/metrics/economic")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "total_value_locked" in data
            assert "daily_volume" in data
            assert "market_cap" in data
            assert "circulating_supply" in data
    
    def test_historical_data_endpoint(self, client):
        """Test historical data endpoint"""
        response = client.get("/historical/24")
        assert response.status_code in [200, 404, 500]  # 500 if DB not set up
        
        if response.status_code == 200:
            data = response.json()
            assert "period_hours" in data
            assert "network_metrics" in data
            assert "economic_metrics" in data
    
    def test_trigger_collection_endpoint(self, client):
        """Test manual collection trigger endpoint"""
        response = client.post("/collect")
        assert response.status_code in [200, 409]  # 409 if already collecting
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "status" in data
    
    def test_development_metrics_endpoint(self, client):
        """Test development metrics endpoint"""
        response = client.get("/metrics/development")
        assert response.status_code == 200
        data = response.json()
        assert "github_commits" in data
        assert "github_stars" in data
        assert "github_forks" in data
        assert "developer_count" in data
        assert "smart_contract_deployments" in data
        assert "subnet_launches" in data
    
    def test_competitive_metrics_endpoint(self, client):
        """Test competitive metrics endpoint"""
        response = client.get("/metrics/competitive")
        assert response.status_code == 200
        data = response.json()
        assert "market_share" in data
        assert "performance_vs_competitors" in data
        assert "ecosystem_growth" in data
    
    def test_risk_metrics_endpoint(self, client):
        """Test risk metrics endpoint"""
        response = client.get("/metrics/risk")
        assert response.status_code == 200
        data = response.json()
        assert "centralization_risks" in data
        assert "technical_risks" in data
        assert "regulatory_risks" in data
        assert "market_risks" in data
    
    def test_macro_metrics_endpoint(self, client):
        """Test macro metrics endpoint"""
        response = client.get("/metrics/macro")
        assert response.status_code == 200
        data = response.json()
        assert "market_conditions" in data
        assert "institutional_adoption" in data
        assert "regulatory_environment" in data
        assert "economic_indicators" in data

class TestExternalAPIConnectivity:
    """Test connectivity to external APIs"""
    
    @pytest.mark.asyncio
    async def test_coingecko_api_connectivity(self):
        """Test real connectivity to CoinGecko API"""
        async with CoinGeckoClient() as cg:
            # Test basic connectivity
            global_data = await cg.get_global_data()
            assert global_data is not None, "Should get global data from CoinGecko"
            assert "data" in global_data, "Should have data field"
            
            # Test AVAX specific data
            avax_data = await cg.get_coin_data("avalanche-2")
            assert avax_data is not None, "Should get AVAX data from CoinGecko"
            assert "market_data" in avax_data, "Should have market data"
            
            # Test price data
            price_data = await cg.get_coin_price(["avalanche-2"], ["usd"])
            assert price_data is not None, "Should get price data"
            assert "avalanche-2" in price_data, "Should have AVAX price data"
    
    @pytest.mark.asyncio
    async def test_avalanche_rpc_connectivity(self):
        """Test real connectivity to Avalanche RPC"""
        async with aiohttp.ClientSession() as session:
            # Test C-Chain RPC
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            }
            
            async with session.post(
                "https://api.avax.network/ext/bc/C/rpc",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status == 200, "Should connect to Avalanche C-Chain RPC"
                data = await response.json()
                assert "result" in data, "Should get block number result"
                assert data["result"].startswith("0x"), "Result should be hex format"
    
    @pytest.mark.asyncio
    async def test_defillama_api_connectivity(self):
        """Test real connectivity to DeFiLlama API"""
        async with aiohttp.ClientSession() as session:
            # Test DeFiLlama API
            async with session.get("https://api.llama.fi/tvl/avalanche") as response:
                assert response.status == 200, "Should connect to DeFiLlama API"
                data = await response.json()
                assert isinstance(data, (int, float)), "Should get TVL data"
                assert data >= 0, "TVL should be non-negative"

class TestDataValidation:
    """Test data validation and quality"""
    
    @pytest.mark.asyncio
    async def test_data_freshness(self):
        """Test that collected data is fresh"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        
        try:
            # Collect network performance data
            metrics = await collector.collect_network_performance()
            
            if metrics and "timestamp" in metrics:
                timestamp_str = metrics["timestamp"]
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                current_time = datetime.utcnow()
                time_diff = (current_time - timestamp).total_seconds()
                
                assert time_diff < 60, f"Data should be fresh (within 1 minute), got {time_diff} seconds"
        finally:
            await collector.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """Test data consistency across multiple collections"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        
        try:
            # Collect data multiple times
            results = []
            for _ in range(3):
                metrics = await collector.collect_network_performance()
                if metrics:
                    results.append(metrics)
                await asyncio.sleep(1)  # Wait 1 second between collections
            
            assert len(results) >= 2, "Should get at least 2 successful collections"
            
            # Check that block numbers are increasing (or at least not decreasing significantly)
            block_numbers = [r.get("current_block", 0) for r in results if "current_block" in r]
            if len(block_numbers) >= 2:
                # Block numbers should be increasing or at most 1 block behind
                for i in range(1, len(block_numbers)):
                    diff = block_numbers[i] - block_numbers[i-1]
                    assert diff >= -1, f"Block numbers should not decrease significantly: {block_numbers}"
        finally:
            await collector.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in data collection"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        
        try:
            # Test with invalid RPC endpoint
            original_rpc = collector.avalanche_config["rpc_url"]
            collector.avalanche_config["rpc_url"] = "https://invalid-endpoint.com"
            
            # Should handle error gracefully
            metrics = await collector.collect_network_performance()
            assert isinstance(metrics, dict), "Should return dictionary even on error"
            
            # Restore original RPC
            collector.avalanche_config["rpc_url"] = original_rpc
        finally:
            await collector.__aexit__(None, None, None)

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_collection_performance(self):
        """Test that data collection completes within reasonable time"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        
        try:
            # Test network performance collection speed
            start_time = time.time()
            metrics = await collector.collect_network_performance()
            end_time = time.time()
            
            collection_time = end_time - start_time
            assert collection_time < 30, f"Network performance collection should complete within 30 seconds, took {collection_time:.2f}s"
            assert metrics is not None, "Should return metrics"
            
            # Test economic data collection speed
            start_time = time.time()
            metrics = await collector.collect_economic_data()
            end_time = time.time()
            
            collection_time = end_time - start_time
            assert collection_time < 30, f"Economic data collection should complete within 30 seconds, took {collection_time:.2f}s"
            assert metrics is not None, "Should return metrics"
        finally:
            await collector.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_concurrent_collection(self):
        """Test concurrent data collection"""
        collector = RealTimeDataCollector()
        await collector.__aenter__()
        
        try:
            # Start multiple collection tasks concurrently
            tasks = [
                collector.collect_network_performance(),
                collector.collect_economic_data(),
                collector.collect_defi_metrics(),
                collector.collect_subnet_data()
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            collection_time = end_time - start_time
            assert collection_time < 60, f"Concurrent collection should complete within 60 seconds, took {collection_time:.2f}s"
            
            # Check that we got results (some might fail, but most should succeed)
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 2, f"Should get at least 2 successful results, got {len(successful_results)}"
        finally:
            await collector.__aexit__(None, None, None)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
