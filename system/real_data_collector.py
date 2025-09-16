#!/usr/bin/env python3
"""
Real Data Collector for Polkadot Metrics
========================================

This script implements a comprehensive data collection pipeline that gathers
real data from multiple external sources to replace the current mock data system.

Data Sources:
- Polkadot RPC (primary blockchain data)
- CoinGecko API (price and market data)
- Subscan API (detailed blockchain analytics)
- GitHub API (developer metrics)
- Social media APIs (community metrics)
"""

import asyncio
import aiohttp
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
import os
from dataclasses import dataclass
import time

# Configure logging
logger.add("logs/real_data_collector.log", rotation="1 day", retention="30 days")

@dataclass
class DataSource:
    """Configuration for external data sources"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    last_request: float = 0

class RealDataCollector:
    """Collects real data from external sources"""
    
    def __init__(self):
        self.sources = {
            "polkadot_rpc": DataSource(
                name="Polkadot RPC",
                base_url="https://rpc.polkadot.io",
                rate_limit=100
            ),
            "coingecko": DataSource(
                name="CoinGecko",
                base_url="https://api.coingecko.com/api/v3",
                api_key=os.getenv("COINGECKO_API_KEY"),
                rate_limit=50
            ),
            "subscan": DataSource(
                name="Subscan",
                base_url="https://polkadot.api.subscan.io/api",
                api_key=os.getenv("SUBSCAN_API_KEY"),
                rate_limit=30
            ),
            "github": DataSource(
                name="GitHub",
                base_url="https://api.github.com",
                api_key=os.getenv("GITHUB_API_KEY"),
                rate_limit=60
            )
        }
        
        self.session = None
        self.db_path = "polkadot_metrics.db"
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self, source_name: str):
        """Check and enforce rate limits"""
        source = self.sources[source_name]
        current_time = time.time()
        
        if current_time - source.last_request < (60 / source.rate_limit):
            sleep_time = (60 / source.rate_limit) - (current_time - source.last_request)
            await asyncio.sleep(sleep_time)
        
        source.last_request = time.time()
    
    async def _make_rpc_call(self, method: str, params: List = None) -> Dict[str, Any]:
        """Make RPC call to Polkadot node"""
        if params is None:
            params = []
            
        await self._rate_limit_check("polkadot_rpc")
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            async with self.session.post(
                self.sources["polkadot_rpc"].base_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result")
                else:
                    logger.error(f"RPC call failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"RPC call error: {e}")
            return None
    
    async def _make_api_call(self, source_name: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make API call to external service"""
        if params is None:
            params = {}
            
        await self._rate_limit_check(source_name)
        source = self.sources[source_name]
        
        url = f"{source.base_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if source.api_key:
            headers["Authorization"] = f"Bearer {source.api_key}"
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API call failed: {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"API call error: {e}")
            return None
    
    async def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect real network metrics from Polkadot RPC"""
        logger.info("Collecting network metrics...")
        
        try:
            # Get chain info
            chain_info = await self._make_rpc_call("system_chain")
            version = await self._make_rpc_call("system_version")
            
            # Get current block
            latest_block = await self._make_rpc_call("chain_getBlock")
            block_number = latest_block.get("block", {}).get("header", {}).get("number") if latest_block else 0
            
            # Get validator count
            validators = await self._make_rpc_call("session_validators")
            validator_count = len(validators) if validators else 0
            
            # Get runtime version
            runtime_version = await self._make_rpc_call("state_getRuntimeVersion")
            
            # Get peer count
            peer_count = await self._make_rpc_call("system_peers")
            peer_count = len(peer_count) if peer_count else 0
            
            return {
                "chain_name": chain_info,
                "version": version,
                "block_number": int(block_number) if block_number else 0,
                "validator_count": validator_count,
                "runtime_version": runtime_version.get("specVersion") if runtime_version else 0,
                "peer_count": peer_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to collect network metrics: {e}")
            return {}
    
    async def collect_staking_metrics(self) -> Dict[str, Any]:
        """Collect real staking metrics from Polkadot RPC"""
        logger.info("Collecting staking metrics...")
        
        try:
            # Get active era
            active_era = await self._make_rpc_call("staking_activeEra")
            era_index = active_era.get("index") if active_era else 0
            
            # Get validator count
            validators = await self._make_rpc_call("session_validators")
            validator_count = len(validators) if validators else 0
            
            # Get nominator count (approximate)
            nominators = await self._make_rpc_call("staking_nominators")
            nominator_count = len(nominators) if nominators else 0
            
            # Get total issuance
            total_issuance = await self._make_rpc_call("balances_totalIssuance")
            total_issuance = int(total_issuance) / 1e10 if total_issuance else 0  # Convert from Planck to DOT
            
            # Get staking info
            staking_info = await self._make_rpc_call("staking_stakingInfo")
            
            return {
                "active_era": era_index,
                "validator_count": validator_count,
                "nominator_count": nominator_count,
                "total_issuance": total_issuance,
                "staking_info": staking_info,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to collect staking metrics: {e}")
            return {}
    
    async def collect_price_data(self) -> Dict[str, Any]:
        """Collect real price data from CoinGecko"""
        logger.info("Collecting price data...")
        
        try:
            # Get Polkadot price data
            price_data = await self._make_api_call(
                "coingecko",
                "simple/price",
                {
                    "ids": "polkadot",
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true"
                }
            )
            
            if price_data and "polkadot" in price_data:
                dot_data = price_data["polkadot"]
                return {
                    "price_usd": dot_data.get("usd", 0),
                    "market_cap_usd": dot_data.get("usd_market_cap", 0),
                    "volume_24h_usd": dot_data.get("usd_24h_vol", 0),
                    "price_change_24h": dot_data.get("usd_24h_change", 0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.warning("No price data received from CoinGecko")
                return {}
        except Exception as e:
            logger.error(f"Failed to collect price data: {e}")
            return {}
    
    async def collect_governance_metrics(self) -> Dict[str, Any]:
        """Collect real governance metrics from Polkadot RPC"""
        logger.info("Collecting governance metrics...")
        
        try:
            # Get referendum count
            referendum_count = await self._make_rpc_call("referenda_referendumCount")
            
            # Get council members
            council_members = await self._make_rpc_call("council_members")
            council_count = len(council_members) if council_members else 0
            
            # Get treasury info
            treasury_balance = await self._make_rpc_call("treasury_proposalBond")
            
            return {
                "referendum_count": referendum_count if referendum_count else 0,
                "council_members": council_count,
                "treasury_balance": treasury_balance,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to collect governance metrics: {e}")
            return {}
    
    async def collect_developer_metrics(self) -> Dict[str, Any]:
        """Collect real developer metrics from GitHub"""
        logger.info("Collecting developer metrics...")
        
        try:
            # Get Polkadot repository stats
            repo_data = await self._make_api_call(
                "github",
                "repos/paritytech/polkadot"
            )
            
            if repo_data:
                return {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "last_updated": repo_data.get("updated_at"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.warning("No repository data received from GitHub")
                return {}
        except Exception as e:
            logger.error(f"Failed to collect developer metrics: {e}")
            return {}
    
    async def store_metrics_in_db(self, metrics: Dict[str, Any]):
        """Store collected metrics in database"""
        logger.info("Storing metrics in database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store network metrics
            if "network_metrics" in metrics:
                network_data = metrics["network_metrics"]
                cursor.execute("""
                    INSERT INTO polkadot_network_metrics 
                    (block_number, validator_count, peer_count, runtime_version, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    network_data.get("block_number", 0),
                    network_data.get("validator_count", 0),
                    network_data.get("peer_count", 0),
                    network_data.get("runtime_version", 0),
                    network_data.get("timestamp")
                ))
            
            # Store staking metrics
            if "staking_metrics" in metrics:
                staking_data = metrics["staking_metrics"]
                cursor.execute("""
                    INSERT INTO polkadot_staking_metrics 
                    (active_era, validator_count, nominator_count, total_issuance, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    staking_data.get("active_era", 0),
                    staking_data.get("validator_count", 0),
                    staking_data.get("nominator_count", 0),
                    staking_data.get("total_issuance", 0),
                    staking_data.get("timestamp")
                ))
            
            # Store economic metrics
            if "price_data" in metrics:
                price_data = metrics["price_data"]
                cursor.execute("""
                    INSERT INTO polkadot_economic_metrics 
                    (price_usd, market_cap_usd, volume_24h_usd, price_change_24h, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    price_data.get("price_usd", 0),
                    price_data.get("market_cap_usd", 0),
                    price_data.get("volume_24h_usd", 0),
                    price_data.get("price_change_24h", 0),
                    price_data.get("timestamp")
                ))
            
            # Store governance metrics
            if "governance_metrics" in metrics:
                gov_data = metrics["governance_metrics"]
                cursor.execute("""
                    INSERT INTO polkadot_governance_metrics 
                    (referendum_count, council_members, treasury_balance, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    gov_data.get("referendum_count", 0),
                    gov_data.get("council_members", 0),
                    gov_data.get("treasury_balance", 0),
                    gov_data.get("timestamp")
                ))
            
            # Store developer metrics
            if "developer_metrics" in metrics:
                dev_data = metrics["developer_metrics"]
                cursor.execute("""
                    INSERT INTO polkadot_developer_metrics 
                    (github_stars_total, github_forks_total, github_contributors, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    dev_data.get("stars", 0),
                    dev_data.get("forks", 0),
                    dev_data.get("watchers", 0),
                    dev_data.get("timestamp")
                ))
            
            conn.commit()
            conn.close()
            logger.info("Metrics stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics"""
        logger.info("Starting comprehensive data collection...")
        
        metrics = {}
        
        # Collect all metrics concurrently
        tasks = [
            ("network_metrics", self.collect_network_metrics()),
            ("staking_metrics", self.collect_staking_metrics()),
            ("price_data", self.collect_price_data()),
            ("governance_metrics", self.collect_governance_metrics()),
            ("developer_metrics", self.collect_developer_metrics())
        ]
        
        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        
        for i, (name, _) in enumerate(tasks):
            if isinstance(results[i], Exception):
                logger.error(f"Failed to collect {name}: {results[i]}")
                metrics[name] = {}
            else:
                metrics[name] = results[i]
        
        return metrics

async def main():
    """Main data collection function"""
    logger.info("Starting real data collection pipeline...")
    
    async with RealDataCollector() as collector:
        # Collect all metrics
        metrics = await collector.collect_all_metrics()
        
        # Store in database
        await collector.store_metrics_in_db(metrics)
        
        # Log summary
        logger.info("Data collection completed:")
        for name, data in metrics.items():
            if data:
                logger.info(f"  ✅ {name}: {len(data)} fields collected")
            else:
                logger.warning(f"  ❌ {name}: No data collected")
    
    logger.info("Real data collection pipeline completed")

if __name__ == "__main__":
    asyncio.run(main())
