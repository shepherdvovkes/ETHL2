from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from loguru import logger

class MetricCategory(str, Enum):
    """Categories of metrics"""
    ONCHAIN = "onchain"
    FINANCIAL = "financial"
    GITHUB = "github"
    TOKENOMICS = "tokenomics"
    SECURITY = "security"
    COMMUNITY = "community"
    PARTNERSHIP = "partnership"
    NETWORK = "network"
    TRENDING = "trending"
    CROSS_CHAIN = "cross_chain"

class DataSource(str, Enum):
    """Data sources for metrics"""
    QUICKNODE = "quicknode"
    ETHERSCAN = "etherscan"
    COINGECKO = "coingecko"
    GITHUB = "github"
    BLOCKCHAIN_RPC = "blockchain_rpc"
    EXTERNAL_API = "external_api"
    CALCULATED = "calculated"

@dataclass
class MetricDefinition:
    """Definition of a metric and its data requirements"""
    name: str
    category: MetricCategory
    description: str
    data_sources: List[DataSource]
    required_fields: List[str]
    calculation_method: Optional[str] = None
    update_frequency: str = "1h"  # How often to update
    priority: int = 1  # 1=high, 2=medium, 3=low
    dependencies: List[str] = None  # Other metrics this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class MetricsMapper:
    """Maps metrics to their data sources and requirements"""
    
    def __init__(self):
        self.metrics_definitions = self._initialize_metrics_definitions()
        self.data_source_mappings = self._initialize_data_source_mappings()
    
    def _initialize_metrics_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize all metric definitions"""
        return {
            # On-chain Metrics
            "tvl": MetricDefinition(
                name="tvl",
                category=MetricCategory.ONCHAIN,
                description="Total Value Locked in the protocol",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.QUICKNODE],
                required_fields=["contract_address", "blockchain_id"],
                calculation_method="sum_of_locked_tokens",
                update_frequency="1h",
                priority=1
            ),
            "daily_transactions": MetricDefinition(
                name="daily_transactions",
                category=MetricCategory.ONCHAIN,
                description="Number of daily transactions",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.QUICKNODE],
                required_fields=["contract_address", "blockchain_id"],
                calculation_method="count_transactions_24h",
                update_frequency="1h",
                priority=1
            ),
            "active_addresses_24h": MetricDefinition(
                name="active_addresses_24h",
                category=MetricCategory.ONCHAIN,
                description="Number of unique active addresses in 24h",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.QUICKNODE],
                required_fields=["blockchain_id"],
                calculation_method="count_unique_addresses",
                update_frequency="1h",
                priority=1
            ),
            "transaction_volume_24h": MetricDefinition(
                name="transaction_volume_24h",
                category=MetricCategory.ONCHAIN,
                description="Total transaction volume in 24h",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.QUICKNODE],
                required_fields=["contract_address", "blockchain_id"],
                calculation_method="sum_transaction_values",
                update_frequency="1h",
                priority=1
            ),
            "gas_price_avg": MetricDefinition(
                name="gas_price_avg",
                category=MetricCategory.ONCHAIN,
                description="Average gas price",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.ETHERSCAN],
                required_fields=["blockchain_id"],
                calculation_method="average_gas_price",
                update_frequency="15m",
                priority=2
            ),
            "contract_interactions_24h": MetricDefinition(
                name="contract_interactions_24h",
                category=MetricCategory.ONCHAIN,
                description="Number of contract interactions in 24h",
                data_sources=[DataSource.BLOCKCHAIN_RPC, DataSource.QUICKNODE],
                required_fields=["contract_address", "blockchain_id"],
                calculation_method="count_contract_calls",
                update_frequency="1h",
                priority=2
            ),
            
            # Financial Metrics
            "price_usd": MetricDefinition(
                name="price_usd",
                category=MetricCategory.FINANCIAL,
                description="Current price in USD",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id", "symbol"],
                calculation_method="direct_fetch",
                update_frequency="5m",
                priority=1
            ),
            "market_cap": MetricDefinition(
                name="market_cap",
                category=MetricCategory.FINANCIAL,
                description="Market capitalization",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id", "circulating_supply"],
                calculation_method="price * circulating_supply",
                update_frequency="5m",
                priority=1,
                dependencies=["price_usd", "circulating_supply"]
            ),
            "volume_24h": MetricDefinition(
                name="volume_24h",
                category=MetricCategory.FINANCIAL,
                description="24h trading volume",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id"],
                calculation_method="direct_fetch",
                update_frequency="15m",
                priority=1
            ),
            "volatility_24h": MetricDefinition(
                name="volatility_24h",
                category=MetricCategory.FINANCIAL,
                description="24h price volatility",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id"],
                calculation_method="calculate_volatility",
                update_frequency="1h",
                priority=2
            ),
            "price_change_24h": MetricDefinition(
                name="price_change_24h",
                category=MetricCategory.FINANCIAL,
                description="24h price change percentage",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id"],
                calculation_method="direct_fetch",
                update_frequency="15m",
                priority=1
            ),
            
            # GitHub Metrics
            "commits_24h": MetricDefinition(
                name="commits_24h",
                category=MetricCategory.GITHUB,
                description="Number of commits in 24h",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="count_commits_since",
                update_frequency="6h",
                priority=2
            ),
            "commits_7d": MetricDefinition(
                name="commits_7d",
                category=MetricCategory.GITHUB,
                description="Number of commits in 7 days",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="count_commits_since",
                update_frequency="6h",
                priority=2
            ),
            "active_contributors_30d": MetricDefinition(
                name="active_contributors_30d",
                category=MetricCategory.GITHUB,
                description="Number of active contributors in 30 days",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="count_unique_contributors",
                update_frequency="24h",
                priority=2
            ),
            "stars": MetricDefinition(
                name="stars",
                category=MetricCategory.GITHUB,
                description="Number of GitHub stars",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="direct_fetch",
                update_frequency="24h",
                priority=3
            ),
            "forks": MetricDefinition(
                name="forks",
                category=MetricCategory.GITHUB,
                description="Number of GitHub forks",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="direct_fetch",
                update_frequency="24h",
                priority=3
            ),
            "open_issues": MetricDefinition(
                name="open_issues",
                category=MetricCategory.GITHUB,
                description="Number of open issues",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="count_open_issues",
                update_frequency="6h",
                priority=2
            ),
            "open_prs": MetricDefinition(
                name="open_prs",
                category=MetricCategory.GITHUB,
                description="Number of open pull requests",
                data_sources=[DataSource.GITHUB],
                required_fields=["github_repo"],
                calculation_method="count_open_prs",
                update_frequency="6h",
                priority=2
            ),
            
            # Tokenomics Metrics
            "circulating_supply": MetricDefinition(
                name="circulating_supply",
                category=MetricCategory.TOKENOMICS,
                description="Circulating supply of tokens",
                data_sources=[DataSource.COINGECKO, DataSource.BLOCKCHAIN_RPC],
                required_fields=["coingecko_id", "contract_address"],
                calculation_method="direct_fetch_or_calculate",
                update_frequency="1h",
                priority=1
            ),
            "total_supply": MetricDefinition(
                name="total_supply",
                category=MetricCategory.TOKENOMICS,
                description="Total supply of tokens",
                data_sources=[DataSource.COINGECKO, DataSource.BLOCKCHAIN_RPC],
                required_fields=["coingecko_id", "contract_address"],
                calculation_method="direct_fetch_or_calculate",
                update_frequency="1h",
                priority=1
            ),
            "max_supply": MetricDefinition(
                name="max_supply",
                category=MetricCategory.TOKENOMICS,
                description="Maximum supply of tokens",
                data_sources=[DataSource.COINGECKO],
                required_fields=["coingecko_id"],
                calculation_method="direct_fetch",
                update_frequency="24h",
                priority=2
            ),
            "inflation_rate": MetricDefinition(
                name="inflation_rate",
                category=MetricCategory.TOKENOMICS,
                description="Token inflation rate",
                data_sources=[DataSource.CALCULATED],
                required_fields=["circulating_supply", "total_supply"],
                calculation_method="calculate_inflation_rate",
                update_frequency="24h",
                priority=3,
                dependencies=["circulating_supply", "total_supply"]
            ),
            
            # Security Metrics
            "audit_status": MetricDefinition(
                name="audit_status",
                category=MetricCategory.SECURITY,
                description="Smart contract audit status",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["contract_address"],
                calculation_method="check_audit_databases",
                update_frequency="168h",  # Weekly
                priority=2
            ),
            "contract_verified": MetricDefinition(
                name="contract_verified",
                category=MetricCategory.SECURITY,
                description="Whether contract is verified on explorer",
                data_sources=[DataSource.ETHERSCAN],
                required_fields=["contract_address", "blockchain_id"],
                calculation_method="check_verification_status",
                update_frequency="24h",
                priority=2
            ),
            "vulnerability_score": MetricDefinition(
                name="vulnerability_score",
                category=MetricCategory.SECURITY,
                description="Security vulnerability score",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["contract_address"],
                calculation_method="analyze_contract_security",
                update_frequency="168h",  # Weekly
                priority=2
            ),
            
            # Network Metrics
            "block_time_avg": MetricDefinition(
                name="block_time_avg",
                category=MetricCategory.NETWORK,
                description="Average block time",
                data_sources=[DataSource.BLOCKCHAIN_RPC],
                required_fields=["blockchain_id"],
                calculation_method="calculate_average_block_time",
                update_frequency="1h",
                priority=2
            ),
            "network_utilization": MetricDefinition(
                name="network_utilization",
                category=MetricCategory.NETWORK,
                description="Network utilization percentage",
                data_sources=[DataSource.BLOCKCHAIN_RPC],
                required_fields=["blockchain_id"],
                calculation_method="calculate_network_utilization",
                update_frequency="1h",
                priority=2
            ),
            "validator_count": MetricDefinition(
                name="validator_count",
                category=MetricCategory.NETWORK,
                description="Number of validators",
                data_sources=[DataSource.BLOCKCHAIN_RPC],
                required_fields=["blockchain_id"],
                calculation_method="count_validators",
                update_frequency="24h",
                priority=3
            ),
            
            # Community Metrics
            "twitter_followers": MetricDefinition(
                name="twitter_followers",
                category=MetricCategory.COMMUNITY,
                description="Number of Twitter followers",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["twitter_handle"],
                calculation_method="fetch_social_metrics",
                update_frequency="24h",
                priority=3
            ),
            "telegram_members": MetricDefinition(
                name="telegram_members",
                category=MetricCategory.COMMUNITY,
                description="Number of Telegram members",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["telegram_handle"],
                calculation_method="fetch_social_metrics",
                update_frequency="24h",
                priority=3
            ),
            "discord_members": MetricDefinition(
                name="discord_members",
                category=MetricCategory.COMMUNITY,
                description="Number of Discord members",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["discord_invite"],
                calculation_method="fetch_social_metrics",
                update_frequency="24h",
                priority=3
            ),
            
            # Trending Metrics
            "momentum_score": MetricDefinition(
                name="momentum_score",
                category=MetricCategory.TRENDING,
                description="Price momentum score",
                data_sources=[DataSource.CALCULATED],
                required_fields=["price_usd"],
                calculation_method="calculate_momentum",
                update_frequency="1h",
                priority=2,
                dependencies=["price_usd"]
            ),
            "fear_greed_index": MetricDefinition(
                name="fear_greed_index",
                category=MetricCategory.TRENDING,
                description="Market fear and greed index",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=[],
                calculation_method="fetch_fear_greed_index",
                update_frequency="24h",
                priority=3
            ),
            "social_sentiment": MetricDefinition(
                name="social_sentiment",
                category=MetricCategory.TRENDING,
                description="Social media sentiment score",
                data_sources=[DataSource.EXTERNAL_API],
                required_fields=["symbol"],
                calculation_method="analyze_social_sentiment",
                update_frequency="6h",
                priority=3
            )
        }
    
    def _initialize_data_source_mappings(self) -> Dict[DataSource, Dict[str, Any]]:
        """Initialize data source configurations"""
        return {
            DataSource.QUICKNODE: {
                "client_class": "QuickNodeClient",
                "rate_limit": 100,  # requests per minute
                "priority": 1,
                "supports": ["ethereum", "polygon", "arbitrum", "optimism"]
            },
            DataSource.ETHERSCAN: {
                "client_class": "EtherscanClient",
                "rate_limit": 5,  # requests per second
                "priority": 2,
                "supports": ["ethereum", "polygon", "bsc", "arbitrum", "optimism"]
            },
            DataSource.COINGECKO: {
                "client_class": "CoinGeckoClient",
                "rate_limit": 50,  # requests per minute
                "priority": 1,
                "supports": ["all"]
            },
            DataSource.GITHUB: {
                "client_class": "GitHubClient",
                "rate_limit": 5000,  # requests per hour
                "priority": 2,
                "supports": ["all"]
            },
            DataSource.BLOCKCHAIN_RPC: {
                "client_class": "BlockchainClient",
                "rate_limit": 1000,  # requests per minute
                "priority": 1,
                "supports": ["all"]
            },
            DataSource.EXTERNAL_API: {
                "client_class": "ExternalAPIClient",
                "rate_limit": 100,  # requests per minute
                "priority": 3,
                "supports": ["all"]
            },
            DataSource.CALCULATED: {
                "client_class": "CalculatedMetrics",
                "rate_limit": 0,  # No rate limit for calculations
                "priority": 1,
                "supports": ["all"]
            }
        }
    
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name"""
        return self.metrics_definitions.get(metric_name)
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[MetricDefinition]:
        """Get all metrics in a category"""
        return [
            metric for metric in self.metrics_definitions.values()
            if metric.category == category
        ]
    
    def get_metrics_by_priority(self, priority: int) -> List[MetricDefinition]:
        """Get metrics by priority level"""
        return [
            metric for metric in self.metrics_definitions.values()
            if metric.priority == priority
        ]
    
    def get_required_data_sources(self, metric_names: List[str]) -> Set[DataSource]:
        """Get all required data sources for given metrics"""
        data_sources = set()
        for metric_name in metric_names:
            metric = self.get_metric_definition(metric_name)
            if metric:
                data_sources.update(metric.data_sources)
        return data_sources
    
    def get_metrics_for_asset(self, asset_data: Dict[str, Any]) -> List[str]:
        """Get available metrics for a specific asset based on its data"""
        available_metrics = []
        
        for metric_name, metric_def in self.metrics_definitions.items():
            # Check if asset has required fields for this metric
            has_required_fields = all(
                field in asset_data and asset_data[field] is not None
                for field in metric_def.required_fields
            )
            
            if has_required_fields:
                available_metrics.append(metric_name)
        
        return available_metrics
    
    def get_data_collection_plan(
        self, 
        asset_data: Dict[str, Any], 
        time_periods: List[str] = ["1w", "2w", "4w"]
    ) -> Dict[str, Any]:
        """Get data collection plan for an asset"""
        available_metrics = self.get_metrics_for_asset(asset_data)
        required_sources = self.get_required_data_sources(available_metrics)
        
        # Group metrics by data source
        metrics_by_source = {}
        for metric_name in available_metrics:
            metric_def = self.get_metric_definition(metric_name)
            for source in metric_def.data_sources:
                if source not in metrics_by_source:
                    metrics_by_source[source] = []
                metrics_by_source[source].append(metric_name)
        
        # Create collection plan
        collection_plan = {
            "asset": asset_data,
            "time_periods": time_periods,
            "available_metrics": available_metrics,
            "required_sources": list(required_sources),
            "metrics_by_source": metrics_by_source,
            "collection_schedule": self._create_collection_schedule(available_metrics),
            "estimated_duration": self._estimate_collection_duration(required_sources, time_periods)
        }
        
        return collection_plan
    
    def _create_collection_schedule(self, metrics: List[str]) -> Dict[str, List[str]]:
        """Create collection schedule based on update frequencies"""
        schedule = {
            "5m": [],
            "15m": [],
            "1h": [],
            "6h": [],
            "24h": [],
            "168h": []  # Weekly
        }
        
        for metric_name in metrics:
            metric_def = self.get_metric_definition(metric_name)
            if metric_def and metric_def.update_frequency in schedule:
                schedule[metric_def.update_frequency].append(metric_name)
        
        # Remove empty schedules
        return {k: v for k, v in schedule.items() if v}
    
    def _estimate_collection_duration(
        self, 
        sources: Set[DataSource], 
        time_periods: List[str]
    ) -> Dict[str, Any]:
        """Estimate data collection duration"""
        total_requests = 0
        rate_limited_sources = []
        
        for source in sources:
            source_config = self.data_source_mappings.get(source, {})
            rate_limit = source_config.get("rate_limit", 100)
            
            # Estimate requests needed (rough calculation)
            requests_per_period = 10  # Average requests per time period
            total_requests += len(time_periods) * requests_per_period
            
            if rate_limit < 1000:  # Consider rate limited if less than 1000/min
                rate_limited_sources.append({
                    "source": source,
                    "rate_limit": rate_limit,
                    "estimated_delay": (len(time_periods) * requests_per_period) / rate_limit * 60
                })
        
        return {
            "total_requests": total_requests,
            "rate_limited_sources": rate_limited_sources,
            "estimated_duration_minutes": max(
                [s["estimated_delay"] for s in rate_limited_sources] + [5]
            )
        }
    
    def get_metric_dependencies(self, metric_name: str) -> List[str]:
        """Get all dependencies for a metric"""
        metric = self.get_metric_definition(metric_name)
        if not metric:
            return []
        
        dependencies = set(metric.dependencies)
        
        # Recursively get dependencies of dependencies
        for dep in metric.dependencies:
            dependencies.update(self.get_metric_dependencies(dep))
        
        return list(dependencies)
    
    def validate_asset_data(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate asset data and suggest missing fields"""
        validation_result = {
            "valid": True,
            "missing_fields": [],
            "suggestions": [],
            "available_metrics": [],
            "unavailable_metrics": []
        }
        
        all_metrics = list(self.metrics_definitions.keys())
        available_metrics = self.get_metrics_for_asset(asset_data)
        unavailable_metrics = [m for m in all_metrics if m not in available_metrics]
        
        validation_result["available_metrics"] = available_metrics
        validation_result["unavailable_metrics"] = unavailable_metrics
        
        # Check for missing critical fields
        critical_fields = ["symbol", "name", "blockchain_id"]
        missing_critical = [f for f in critical_fields if f not in asset_data or not asset_data[f]]
        
        if missing_critical:
            validation_result["valid"] = False
            validation_result["missing_fields"] = missing_critical
        
        # Suggest additional fields for more metrics
        suggested_fields = []
        for metric_name in unavailable_metrics:
            metric_def = self.get_metric_definition(metric_name)
            if metric_def:
                missing_fields = [
                    f for f in metric_def.required_fields 
                    if f not in asset_data or not asset_data[f]
                ]
                suggested_fields.extend(missing_fields)
        
        validation_result["suggestions"] = list(set(suggested_fields))
        
        return validation_result
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all available metrics"""
        summary = {
            "total_metrics": len(self.metrics_definitions),
            "by_category": {},
            "by_priority": {},
            "by_data_source": {},
            "most_common_sources": []
        }
        
        # Count by category
        for metric in self.metrics_definitions.values():
            category = metric.category.value
            if category not in summary["by_category"]:
                summary["by_category"][category] = 0
            summary["by_category"][category] += 1
        
        # Count by priority
        for metric in self.metrics_definitions.values():
            priority = metric.priority
            if priority not in summary["by_priority"]:
                summary["by_priority"][priority] = 0
            summary["by_priority"][priority] += 1
        
        # Count by data source
        for metric in self.metrics_definitions.values():
            for source in metric.data_sources:
                source_name = source.value
                if source_name not in summary["by_data_source"]:
                    summary["by_data_source"][source_name] = 0
                summary["by_data_source"][source_name] += 1
        
        # Get most common sources
        sorted_sources = sorted(
            summary["by_data_source"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        summary["most_common_sources"] = sorted_sources[:5]
        
        return summary
