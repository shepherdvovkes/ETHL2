from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Numeric, Index, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

Base = declarative_base()

class TronResourceType(str, Enum):
    """TRON resource types"""
    ENERGY = "energy"
    BANDWIDTH = "bandwidth"

class TronContractType(str, Enum):
    """TRON smart contract types"""
    TRC20 = "trc20"
    TRC721 = "trc721"
    TRC1155 = "trc1155"
    SMART_CONTRACT = "smart_contract"

class TronDeFiProtocol(str, Enum):
    """Major TRON DeFi protocols"""
    SUNSWAP = "sunswap"
    JUSTSWAP = "justswap"
    TRONLINK = "tronlink"
    DEFI = "defi"
    TRONFI = "tronfi"
    COMPOUND = "compound"
    VENUS = "venus"

class TronNetworkMetrics(Base):
    """TRON network performance metrics"""
    __tablename__ = 'tron_network_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Block Metrics
    current_block_number = Column(Integer, nullable=False)
    block_time = Column(Float, nullable=False)  # seconds
    block_size = Column(Float, nullable=False)  # bytes
    transaction_throughput = Column(Integer, nullable=False)  # TPS
    finality_time = Column(Float, nullable=False)  # seconds
    
    # Network Utilization
    network_utilization = Column(Float, nullable=False)  # percentage
    energy_consumption = Column(Float, nullable=False)
    bandwidth_consumption = Column(Float, nullable=False)
    
    # Transaction Metrics
    transaction_count_24h = Column(Integer, nullable=False)
    transaction_fees_24h = Column(Float, nullable=False)  # TRX
    average_transaction_fee = Column(Float, nullable=False)  # TRX
    
    # Network Health
    active_nodes = Column(Integer, nullable=False)
    network_uptime = Column(Float, nullable=False)  # percentage
    consensus_participation = Column(Float, nullable=False)  # percentage
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_network_timestamp', 'timestamp'),
        Index('idx_tron_network_block', 'current_block_number'),
    )

class TronEconomicMetrics(Base):
    """TRON economic and market metrics"""
    __tablename__ = 'tron_economic_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Token Economics
    trx_price_usd = Column(Float, nullable=False)
    trx_price_btc = Column(Float, nullable=False)
    market_cap = Column(Float, nullable=False)  # USD
    fully_diluted_market_cap = Column(Float, nullable=False)  # USD
    volume_24h = Column(Float, nullable=False)  # USD
    price_change_24h = Column(Float, nullable=False)  # percentage
    price_change_7d = Column(Float, nullable=False)  # percentage
    
    # Supply Metrics
    total_supply = Column(Float, nullable=False)  # TRX
    circulating_supply = Column(Float, nullable=False)  # TRX
    burned_tokens = Column(Float, nullable=False)  # TRX
    
    # Network Revenue
    transaction_fees_24h = Column(Float, nullable=False)  # USD
    network_revenue_24h = Column(Float, nullable=False)  # USD
    revenue_per_transaction = Column(Float, nullable=False)  # USD
    
    # Market Indicators
    market_dominance = Column(Float, nullable=False)  # percentage
    trading_volume_ratio = Column(Float, nullable=False)
    liquidity_score = Column(Float, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_economic_timestamp', 'timestamp'),
        Index('idx_tron_economic_price', 'trx_price_usd'),
    )

class TronDeFiMetrics(Base):
    """TRON DeFi ecosystem metrics"""
    __tablename__ = 'tron_defi_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # TVL Metrics
    total_value_locked = Column(Float, nullable=False)  # USD
    tvl_change_24h = Column(Float, nullable=False)  # percentage
    tvl_change_7d = Column(Float, nullable=False)  # percentage
    
    # Protocol Metrics
    defi_protocols_count = Column(Integer, nullable=False)
    active_protocols_count = Column(Integer, nullable=False)
    new_protocols_30d = Column(Integer, nullable=False)
    
    # DEX Metrics
    dex_volume_24h = Column(Float, nullable=False)  # USD
    dex_trades_24h = Column(Integer, nullable=False)
    dex_liquidity = Column(Float, nullable=False)  # USD
    
    # Lending Metrics
    lending_tvl = Column(Float, nullable=False)  # USD
    total_borrowed = Column(Float, nullable=False)  # USD
    lending_utilization_rate = Column(Float, nullable=False)  # percentage
    
    # Yield Farming
    yield_farming_tvl = Column(Float, nullable=False)  # USD
    average_apy = Column(Float, nullable=False)  # percentage
    top_apy = Column(Float, nullable=False)  # percentage
    
    # Bridge Metrics
    bridge_volume_24h = Column(Float, nullable=False)  # USD
    bridge_transactions_24h = Column(Integer, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_defi_timestamp', 'timestamp'),
        Index('idx_tron_defi_tvl', 'total_value_locked'),
    )

class TronSmartContractMetrics(Base):
    """TRON smart contract and token metrics"""
    __tablename__ = 'tron_smart_contract_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Contract Deployment
    new_contracts_24h = Column(Integer, nullable=False)
    new_contracts_7d = Column(Integer, nullable=False)
    total_contracts = Column(Integer, nullable=False)
    
    # TRC-20 Tokens
    trc20_tokens_count = Column(Integer, nullable=False)
    trc20_volume_24h = Column(Float, nullable=False)  # USD
    trc20_transactions_24h = Column(Integer, nullable=False)
    
    # Major Tokens
    usdt_supply = Column(Float, nullable=False)  # USD
    usdc_supply = Column(Float, nullable=False)  # USD
    btt_supply = Column(Float, nullable=False)  # USD
    
    # NFT Metrics (TRC-721/1155)
    nft_collections_count = Column(Integer, nullable=False)
    nft_transactions_24h = Column(Integer, nullable=False)
    nft_volume_24h = Column(Float, nullable=False)  # USD
    
    # Contract Interactions
    contract_calls_24h = Column(Integer, nullable=False)
    contract_gas_consumed = Column(Float, nullable=False)  # energy units
    average_contract_complexity = Column(Float, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_contract_timestamp', 'timestamp'),
        Index('idx_tron_contract_new', 'new_contracts_24h'),
    )

class TronStakingMetrics(Base):
    """TRON staking and governance metrics"""
    __tablename__ = 'tron_staking_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Staking Metrics
    total_staked = Column(Float, nullable=False)  # TRX
    staking_ratio = Column(Float, nullable=False)  # percentage
    staking_apy = Column(Float, nullable=False)  # percentage
    
    # Validator Metrics
    active_validators = Column(Integer, nullable=False)
    total_validators = Column(Integer, nullable=False)
    validator_participation_rate = Column(Float, nullable=False)  # percentage
    
    # Governance
    governance_proposals = Column(Integer, nullable=False)
    active_proposals = Column(Integer, nullable=False)
    voting_participation = Column(Float, nullable=False)  # percentage
    
    # Resource Management
    energy_frozen = Column(Float, nullable=False)  # TRX
    bandwidth_frozen = Column(Float, nullable=False)  # TRX
    resource_utilization = Column(Float, nullable=False)  # percentage
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_staking_timestamp', 'timestamp'),
        Index('idx_tron_staking_ratio', 'staking_ratio'),
    )

class TronUserActivityMetrics(Base):
    """TRON user activity and adoption metrics"""
    __tablename__ = 'tron_user_activity_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Address Metrics
    active_addresses_24h = Column(Integer, nullable=False)
    new_addresses_24h = Column(Integer, nullable=False)
    total_addresses = Column(Integer, nullable=False)
    
    # User Behavior
    average_transactions_per_user = Column(Float, nullable=False)
    user_retention_rate = Column(Float, nullable=False)  # percentage
    whale_activity = Column(Integer, nullable=False)  # large transactions
    
    # Adoption Metrics
    dapp_users_24h = Column(Integer, nullable=False)
    defi_users_24h = Column(Integer, nullable=False)
    nft_users_24h = Column(Integer, nullable=False)
    
    # Geographic Distribution
    top_countries = Column(JSON, nullable=True)  # JSON array
    regional_activity = Column(JSON, nullable=True)  # JSON object
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_user_timestamp', 'timestamp'),
        Index('idx_tron_user_active', 'active_addresses_24h'),
    )

class TronNetworkHealthMetrics(Base):
    """TRON network health and security metrics"""
    __tablename__ = 'tron_network_health_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Network Performance
    average_latency = Column(Float, nullable=False)  # milliseconds
    network_congestion = Column(Float, nullable=False)  # percentage
    block_production_rate = Column(Float, nullable=False)  # blocks per second
    
    # Security Metrics
    security_score = Column(Float, nullable=False)  # 0-100
    decentralization_index = Column(Float, nullable=False)  # 0-100
    validator_distribution = Column(JSON, nullable=True)  # JSON object
    
    # Risk Assessment
    centralization_risk = Column(Float, nullable=False)  # 0-100
    technical_risk = Column(Float, nullable=False)  # 0-100
    economic_risk = Column(Float, nullable=False)  # 0-100
    
    # Incident Tracking
    security_incidents_24h = Column(Integer, nullable=False)
    failed_transactions_24h = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=False)  # percentage
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_health_timestamp', 'timestamp'),
        Index('idx_tron_health_security', 'security_score'),
    )

class TronProtocolMetrics(Base):
    """Individual TRON protocol metrics"""
    __tablename__ = 'tron_protocol_metrics'
    
    id = Column(Integer, primary_key=True)
    protocol_name = Column(String(100), nullable=False)
    protocol_type = Column(String(50), nullable=False)  # dex, lending, staking, etc.
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Protocol Specific Metrics
    tvl = Column(Float, nullable=False)  # USD
    volume_24h = Column(Float, nullable=False)  # USD
    users_24h = Column(Integer, nullable=False)
    transactions_24h = Column(Integer, nullable=False)
    
    # Performance Metrics
    apy = Column(Float, nullable=True)  # percentage
    fees_24h = Column(Float, nullable=False)  # USD
    market_share = Column(Float, nullable=False)  # percentage
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_protocol_name_timestamp', 'protocol_name', 'timestamp'),
        Index('idx_tron_protocol_tvl', 'tvl'),
        Index('idx_tron_protocol_type', 'protocol_type'),
    )

class TronTokenMetrics(Base):
    """Individual TRON token metrics"""
    __tablename__ = 'tron_token_metrics'
    
    id = Column(Integer, primary_key=True)
    token_symbol = Column(String(20), nullable=False)
    token_address = Column(String(42), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Token Economics
    price_usd = Column(Float, nullable=False)
    market_cap = Column(Float, nullable=False)  # USD
    volume_24h = Column(Float, nullable=False)  # USD
    price_change_24h = Column(Float, nullable=False)  # percentage
    
    # Supply Metrics
    total_supply = Column(Float, nullable=False)
    circulating_supply = Column(Float, nullable=False)
    
    # Activity Metrics
    holders_count = Column(Integer, nullable=False)
    transactions_24h = Column(Integer, nullable=False)
    transfers_24h = Column(Integer, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_token_symbol_timestamp', 'token_symbol', 'timestamp'),
        Index('idx_tron_token_address', 'token_address'),
        Index('idx_tron_token_price', 'price_usd'),
    )

class TronComprehensiveMetrics(Base):
    """Comprehensive TRON metrics aggregation"""
    __tablename__ = 'tron_comprehensive_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Network Performance Score (0-100)
    network_performance_score = Column(Float, nullable=False)
    
    # Economic Health Score (0-100)
    economic_health_score = Column(Float, nullable=False)
    
    # DeFi Ecosystem Score (0-100)
    defi_ecosystem_score = Column(Float, nullable=False)
    
    # Adoption Score (0-100)
    adoption_score = Column(Float, nullable=False)
    
    # Security Score (0-100)
    security_score = Column(Float, nullable=False)
    
    # Overall TRON Score (0-100)
    overall_score = Column(Float, nullable=False)
    
    # Trend Indicators
    network_trend = Column(String(20), nullable=False)  # up, down, stable
    economic_trend = Column(String(20), nullable=False)
    defi_trend = Column(String(20), nullable=False)
    adoption_trend = Column(String(20), nullable=False)
    security_trend = Column(String(20), nullable=False)
    
    # Risk Assessment
    risk_level = Column(String(20), nullable=False)  # low, medium, high
    risk_factors = Column(JSON, nullable=True)  # JSON array
    
    # Recommendations
    recommendations = Column(JSON, nullable=True)  # JSON array
    
    # Indexes
    __table_args__ = (
        Index('idx_tron_comprehensive_timestamp', 'timestamp'),
        Index('idx_tron_comprehensive_score', 'overall_score'),
        Index('idx_tron_comprehensive_risk', 'risk_level'),
    )
