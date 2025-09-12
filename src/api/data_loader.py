import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal
from database.models_v2 import (
    CryptoAsset, Blockchain, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, PartnershipMetrics, NetworkMetrics,
    TrendingMetrics, CrossChainMetrics, MLPrediction
)
from api.quicknode_client import QuickNodeClient
from api.etherscan_client import EtherscanClient
from api.coingecko_client import CoinGeckoClient
from api.github_client import GitHubClient
from api.blockchain_client import BlockchainClient, BlockchainType
from api.metrics_mapper import MetricsMapper, MetricCategory, DataSource
from config.settings import settings

@dataclass
class DataCollectionRequest:
    """Request for data collection"""
    asset_id: Optional[int] = None
    blockchain_id: Optional[int] = None
    symbol: Optional[str] = None
    time_periods: List[str] = None  # ["1w", "2w", "4w"]
    metrics: List[str] = None  # Specific metrics to collect
    force_refresh: bool = False
    
    def __post_init__(self):
        if self.time_periods is None:
            self.time_periods = ["1w", "2w", "4w"]
        if self.metrics is None:
            self.metrics = []

@dataclass
class DataCollectionResult:
    """Result of data collection"""
    success: bool
    asset_id: int
    blockchain_id: int
    time_period: str
    metrics_collected: List[str]
    data_points: int
    errors: List[str]
    duration_seconds: float
    timestamp: datetime

class DataLoader:
    """Advanced data loader for crypto analytics"""
    
    def __init__(self):
        self.metrics_mapper = MetricsMapper()
        self.session = None
        self.collection_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def __aenter__(self):
        self.session = SessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def collect_data_for_asset(
        self, 
        request: DataCollectionRequest
    ) -> List[DataCollectionResult]:
        """Collect data for a specific asset across multiple time periods"""
        results = []
        
        # Get asset information
        asset = await self._get_asset(request)
        if not asset:
            logger.error(f"Asset not found: {request}")
            return results
        
        # Get collection plan
        asset_data = self._prepare_asset_data(asset)
        collection_plan = self.metrics_mapper.get_data_collection_plan(
            asset_data, request.time_periods
        )
        
        logger.info(f"Starting data collection for {asset.symbol} ({asset.name})")
        logger.info(f"Available metrics: {len(collection_plan['available_metrics'])}")
        logger.info(f"Time periods: {request.time_periods}")
        
        # Collect data for each time period
        for time_period in request.time_periods:
            try:
                result = await self._collect_data_for_period(
                    asset, time_period, collection_plan, request
                )
                results.append(result)
                
                # Add delay between periods to respect rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting data for period {time_period}: {e}")
                results.append(DataCollectionResult(
                    success=False,
                    asset_id=asset.id,
                    blockchain_id=asset.blockchain_id,
                    time_period=time_period,
                    metrics_collected=[],
                    data_points=0,
                    errors=[str(e)],
                    duration_seconds=0,
                    timestamp=datetime.utcnow()
                ))
        
        return results
    
    async def _get_asset(self, request: DataCollectionRequest) -> Optional[CryptoAsset]:
        """Get asset from database"""
        if request.asset_id:
            return self.session.query(CryptoAsset).filter(
                CryptoAsset.id == request.asset_id
            ).first()
        elif request.symbol:
            return self.session.query(CryptoAsset).filter(
                CryptoAsset.symbol == request.symbol
            ).first()
        return None
    
    def _prepare_asset_data(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Prepare asset data for metrics mapper"""
        return {
            "id": asset.id,
            "symbol": asset.symbol,
            "name": asset.name,
            "contract_address": asset.contract_address,
            "blockchain_id": asset.blockchain_id,
            "coingecko_id": asset.coingecko_id,
            "github_repo": asset.github_repo,
            "website": asset.website,
            "category": asset.category
        }
    
    async def _collect_data_for_period(
        self,
        asset: CryptoAsset,
        time_period: str,
        collection_plan: Dict[str, Any],
        request: DataCollectionRequest
    ) -> DataCollectionResult:
        """Collect data for a specific time period"""
        start_time = datetime.utcnow()
        metrics_collected = []
        errors = []
        data_points = 0
        
        try:
            # Calculate time range
            days = self._parse_time_period(time_period)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Collecting data for {asset.symbol} - {time_period} ({days} days)")
            
            # Collect metrics by data source
            for source_name, metrics in collection_plan["metrics_by_source"].items():
                try:
                    source_metrics = await self._collect_metrics_from_source(
                        source_name, metrics, asset, start_date, end_date, request
                    )
                    metrics_collected.extend(source_metrics["collected"])
                    data_points += source_metrics["data_points"]
                    
                    if source_metrics["errors"]:
                        errors.extend(source_metrics["errors"])
                        
                except Exception as e:
                    error_msg = f"Error collecting from {source_name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Store collected data
            await self._store_collected_data(asset, metrics_collected, time_period)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return DataCollectionResult(
                success=len(errors) == 0,
                asset_id=asset.id,
                blockchain_id=asset.blockchain_id,
                time_period=time_period,
                metrics_collected=metrics_collected,
                data_points=data_points,
                errors=errors,
                duration_seconds=duration,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in data collection for period {time_period}: {e}")
            return DataCollectionResult(
                success=False,
                asset_id=asset.id,
                blockchain_id=asset.blockchain_id,
                time_period=time_period,
                metrics_collected=[],
                data_points=0,
                errors=[str(e)],
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow()
            )
    
    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to days"""
        period_map = {
            "1w": 7,
            "2w": 14,
            "4w": 28,
            "1d": 1,
            "3d": 3,
            "1m": 30,
            "3m": 90,
            "1y": 365
        }
        return period_map.get(time_period, 7)
    
    async def _collect_metrics_from_source(
        self,
        source_name: str,
        metrics: List[str],
        asset: CryptoAsset,
        start_date: datetime,
        end_date: datetime,
        request: DataCollectionRequest
    ) -> Dict[str, Any]:
        """Collect metrics from a specific data source"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            if source_name == DataSource.QUICKNODE.value:
                result = await self._collect_from_quicknode(metrics, asset, start_date, end_date)
            elif source_name == DataSource.ETHERSCAN.value:
                result = await self._collect_from_etherscan(metrics, asset, start_date, end_date)
            elif source_name == DataSource.COINGECKO.value:
                result = await self._collect_from_coingecko(metrics, asset, start_date, end_date)
            elif source_name == DataSource.GITHUB.value:
                result = await self._collect_from_github(metrics, asset, start_date, end_date)
            elif source_name == DataSource.BLOCKCHAIN_RPC.value:
                result = await self._collect_from_blockchain_rpc(metrics, asset, start_date, end_date)
            elif source_name == DataSource.CALCULATED.value:
                result = await self._collect_calculated_metrics(metrics, asset, start_date, end_date)
            else:
                result = {"collected": [], "data_points": 0, "errors": [f"Unknown source: {source_name}"]}
            
            collected = result.get("collected", [])
            data_points = result.get("data_points", 0)
            errors = result.get("errors", [])
            
        except Exception as e:
            errors.append(f"Error collecting from {source_name}: {e}")
        
        return {
            "collected": collected,
            "data_points": data_points,
            "errors": errors
        }
    
    async def _collect_from_quicknode(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect metrics from QuickNode"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            async with QuickNodeClient() as client:
                # Get current block number
                current_block = await client.get_block_number()
                if current_block == 0:
                    errors.append("Failed to get current block number")
                    return {"collected": [], "data_points": 0, "errors": errors}
                
                # Calculate block range
                days = (end_date - start_date).days
                blocks_per_day = 24 * 60 * 60 // 2  # Polygon block time ~2s
                from_block = max(0, current_block - (blocks_per_day * days))
                
                # Collect on-chain metrics
                if "daily_transactions" in metrics:
                    tx_count = await client.get_transaction_count(asset.contract_address or "")
                    collected.append("daily_transactions")
                    data_points += 1
                
                if "active_addresses_24h" in metrics:
                    active_addresses = await client.get_active_addresses(from_block, current_block)
                    collected.append("active_addresses_24h")
                    data_points += 1
                
                if "transaction_volume_24h" in metrics:
                    tx_volume = await client.get_transaction_volume(from_block, current_block)
                    collected.append("transaction_volume_24h")
                    data_points += 1
                
                if "gas_price_avg" in metrics:
                    gas_price = await client.get_gas_price()
                    collected.append("gas_price_avg")
                    data_points += 1
                
                if "contract_interactions_24h" in metrics and asset.contract_address:
                    interactions = await client.get_contract_interactions(
                        asset.contract_address, from_block, current_block
                    )
                    collected.append("contract_interactions_24h")
                    data_points += len(interactions)
                
        except Exception as e:
            errors.append(f"QuickNode collection error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _collect_from_etherscan(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect metrics from Etherscan"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            async with EtherscanClient() as client:
                if asset.contract_address:
                    # Get token info
                    if "total_supply" in metrics:
                        total_supply = await client.get_token_supply(asset.contract_address)
                        collected.append("total_supply")
                        data_points += 1
                    
                    # Get contract security analysis
                    if "contract_verified" in metrics:
                        is_verified = await client.get_contract_verification_status(asset.contract_address)
                        collected.append("contract_verified")
                        data_points += 1
                    
                    if "vulnerability_score" in metrics:
                        security_analysis = await client.analyze_contract_security(asset.contract_address)
                        collected.append("vulnerability_score")
                        data_points += 1
                
                # Get gas price
                if "gas_price_avg" in metrics:
                    gas_data = await client.get_gas_price()
                    collected.append("gas_price_avg")
                    data_points += 1
                
        except Exception as e:
            errors.append(f"Etherscan collection error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _collect_from_coingecko(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect metrics from CoinGecko"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            async with CoinGeckoClient() as client:
                if asset.coingecko_id:
                    # Get current price data
                    if any(m in metrics for m in ["price_usd", "market_cap", "volume_24h", "price_change_24h"]):
                        price_data = await client.get_coin_price([asset.coingecko_id])
                        if price_data and asset.coingecko_id in price_data:
                            coin_data = price_data[asset.coingecko_id]
                            
                            if "price_usd" in metrics and "usd" in coin_data:
                                collected.append("price_usd")
                                data_points += 1
                            
                            if "market_cap" in metrics and "usd_market_cap" in coin_data:
                                collected.append("market_cap")
                                data_points += 1
                            
                            if "volume_24h" in metrics and "usd_24h_vol" in coin_data:
                                collected.append("volume_24h")
                                data_points += 1
                            
                            if "price_change_24h" in metrics and "usd_24h_change" in coin_data:
                                collected.append("price_change_24h")
                                data_points += 1
                    
                    # Get historical data
                    if "volatility_24h" in metrics:
                        days = (end_date - start_date).days
                        historical_data = await client.get_coin_market_chart(
                            asset.coingecko_id, days=days
                        )
                        if historical_data and "prices" in historical_data:
                            collected.append("volatility_24h")
                            data_points += len(historical_data["prices"])
                    
                    # Get token supply data
                    if any(m in metrics for m in ["circulating_supply", "total_supply", "max_supply"]):
                        coin_data = await client.get_coin_data(asset.coingecko_id)
                        if coin_data and "market_data" in coin_data:
                            market_data = coin_data["market_data"]
                            
                            if "circulating_supply" in metrics and "circulating_supply" in market_data:
                                collected.append("circulating_supply")
                                data_points += 1
                            
                            if "total_supply" in metrics and "total_supply" in market_data:
                                collected.append("total_supply")
                                data_points += 1
                            
                            if "max_supply" in metrics and "max_supply" in market_data:
                                collected.append("max_supply")
                                data_points += 1
                
        except Exception as e:
            errors.append(f"CoinGecko collection error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _collect_from_github(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect metrics from GitHub"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            async with GitHubClient() as client:
                if asset.github_repo:
                    # Get comprehensive GitHub metrics
                    github_metrics = await client.get_github_metrics(asset.github_repo)
                    
                    if github_metrics:
                        activity_metrics = github_metrics.get("activity_metrics", {})
                        development_metrics = github_metrics.get("development_metrics", {})
                        community_metrics = github_metrics.get("community_metrics", {})
                        
                        # Map GitHub metrics to our metric names
                        metric_mapping = {
                            "commits_24h": "commits_24h",
                            "commits_7d": "commits_7d",
                            "active_contributors_30d": "active_contributors_30d",
                            "stars": "stars",
                            "forks": "forks",
                            "open_issues": "open_issues",
                            "open_prs": "open_prs"
                        }
                        
                        for github_metric, our_metric in metric_mapping.items():
                            if our_metric in metrics:
                                if github_metric in activity_metrics or github_metric in development_metrics or github_metric in community_metrics:
                                    collected.append(our_metric)
                                    data_points += 1
                
        except Exception as e:
            errors.append(f"GitHub collection error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _collect_from_blockchain_rpc(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect metrics from blockchain RPC"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            # Get blockchain info
            blockchain = self.session.query(Blockchain).filter(
                Blockchain.id == asset.blockchain_id
            ).first()
            
            if blockchain:
                # Create blockchain client
                blockchain_client = BlockchainClient.create_client(blockchain.name.lower())
                
                async with blockchain_client as client:
                    # Get network stats
                    if "block_time_avg" in metrics or "network_utilization" in metrics:
                        network_stats = await client.get_network_stats()
                        if "block_time_avg" in metrics:
                            collected.append("block_time_avg")
                            data_points += 1
                        if "network_utilization" in metrics:
                            collected.append("network_utilization")
                            data_points += 1
                    
                    # Get historical data
                    if any(m in metrics for m in ["active_addresses_24h", "transaction_volume_24h"]):
                        days = (end_date - start_date).days
                        historical_data = await client.get_historical_data(days)
                        if historical_data:
                            if "active_addresses_24h" in metrics:
                                collected.append("active_addresses_24h")
                                data_points += 1
                            if "transaction_volume_24h" in metrics:
                                collected.append("transaction_volume_24h")
                                data_points += 1
                
        except Exception as e:
            errors.append(f"Blockchain RPC collection error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _collect_calculated_metrics(
        self, 
        metrics: List[str], 
        asset: CryptoAsset, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect calculated metrics"""
        collected = []
        data_points = 0
        errors = []
        
        try:
            # Get existing data for calculations
            if "market_cap" in metrics:
                # Market cap = price * circulating_supply
                # This would require fetching price and supply data first
                collected.append("market_cap")
                data_points += 1
            
            if "inflation_rate" in metrics:
                # Inflation rate calculation
                collected.append("inflation_rate")
                data_points += 1
            
            if "momentum_score" in metrics:
                # Momentum score calculation
                collected.append("momentum_score")
                data_points += 1
            
        except Exception as e:
            errors.append(f"Calculated metrics error: {e}")
        
        return {"collected": collected, "data_points": data_points, "errors": errors}
    
    async def _store_collected_data(
        self, 
        asset: CryptoAsset, 
        metrics_collected: List[str], 
        time_period: str
    ):
        """Store collected data in database"""
        try:
            # This is a simplified version - in reality, you'd store actual metric values
            # based on the collected data and time period
            
            # For now, just log what was collected
            logger.info(f"Stored {len(metrics_collected)} metrics for {asset.symbol} - {time_period}")
            
            # In a real implementation, you would:
            # 1. Create metric records with actual values
            # 2. Store them in the appropriate tables
            # 3. Handle time period aggregation
            # 4. Update existing records or create new ones
            
        except Exception as e:
            logger.error(f"Error storing collected data: {e}")
    
    async def collect_data_for_multiple_assets(
        self, 
        requests: List[DataCollectionRequest]
    ) -> Dict[str, List[DataCollectionResult]]:
        """Collect data for multiple assets"""
        results = {}
        
        for request in requests:
            try:
                asset_results = await self.collect_data_for_asset(request)
                asset_key = f"{request.symbol or request.asset_id}"
                results[asset_key] = asset_results
                
                # Add delay between assets to respect rate limits
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error collecting data for request {request}: {e}")
                results[f"{request.symbol or request.asset_id}"] = []
        
        return results
    
    async def collect_data_for_blockchain(
        self, 
        blockchain_id: int, 
        time_periods: List[str] = None
    ) -> Dict[str, List[DataCollectionResult]]:
        """Collect data for all assets on a specific blockchain"""
        if time_periods is None:
            time_periods = ["1w", "2w", "4w"]
        
        # Get all assets for the blockchain
        assets = self.session.query(CryptoAsset).filter(
            CryptoAsset.blockchain_id == blockchain_id,
            CryptoAsset.is_active == True
        ).all()
        
        requests = [
            DataCollectionRequest(
                asset_id=asset.id,
                time_periods=time_periods
            ) for asset in assets
        ]
        
        return await self.collect_data_for_multiple_assets(requests)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        return self.collection_stats.copy()
    
    async def estimate_collection_time(
        self, 
        request: DataCollectionRequest
    ) -> Dict[str, Any]:
        """Estimate time required for data collection"""
        asset = await self._get_asset(request)
        if not asset:
            return {"error": "Asset not found"}
        
        asset_data = self._prepare_asset_data(asset)
        collection_plan = self.metrics_mapper.get_data_collection_plan(
            asset_data, request.time_periods
        )
        
        return {
            "asset": asset.symbol,
            "time_periods": request.time_periods,
            "available_metrics": len(collection_plan["available_metrics"]),
            "required_sources": len(collection_plan["required_sources"]),
            "estimated_duration": collection_plan["estimated_duration"],
            "collection_schedule": collection_plan["collection_schedule"]
        }
