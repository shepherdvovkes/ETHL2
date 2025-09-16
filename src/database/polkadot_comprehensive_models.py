from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Numeric, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()

class ParachainStatus(str, Enum):
    """Parachain status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LEASE_EXPIRED = "lease_expired"
    ONBOARDING = "onboarding"
    CROWDLOAN = "crowdloan"

class MetricCategory(str, Enum):
    """Metric category enumeration"""
    NETWORK = "network"
    STAKING = "staking"
    GOVERNANCE = "governance"
    ECONOMIC = "economic"
    CROSS_CHAIN = "cross_chain"
    DEVELOPER = "developer"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEFI = "defi"
    NFT = "nft"
    GAMING = "gaming"

class PolkadotNetwork(Base):
    """Polkadot main network information"""
    __tablename__ = 'polkadot_networks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    chain_id = Column(String(50), nullable=False)
    rpc_endpoint = Column(String(500), nullable=True)
    ws_endpoint = Column(String(500), nullable=True)
    is_mainnet = Column(Boolean, default=False)
    genesis_hash = Column(String(66), nullable=True)
    spec_version = Column(Integer, nullable=True)
    transaction_version = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parachains = relationship("Parachain", back_populates="network")
    network_metrics = relationship("PolkadotNetworkMetrics", back_populates="network")
    staking_metrics = relationship("PolkadotStakingMetrics", back_populates="network")
    governance_metrics = relationship("PolkadotGovernanceMetrics", back_populates="network")
    economic_metrics = relationship("PolkadotEconomicMetrics", back_populates="network")
    performance_metrics = relationship("PolkadotPerformanceMetrics", back_populates="network")
    security_metrics = relationship("PolkadotSecurityMetrics", back_populates="network")
    developer_metrics = relationship("PolkadotDeveloperMetrics", back_populates="network")

class Parachain(Base):
    """Parachain information"""
    __tablename__ = 'parachains'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(10), nullable=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    status = Column(String(20), nullable=False, default=ParachainStatus.ACTIVE)
    lease_period_start = Column(Integer, nullable=True)
    lease_period_end = Column(Integer, nullable=True)
    website = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)  # DeFi, Gaming, NFT, etc.
    rpc_endpoint = Column(String(500), nullable=True)
    ws_endpoint = Column(String(500), nullable=True)
    genesis_hash = Column(String(66), nullable=True)
    spec_version = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="parachains")
    parachain_metrics = relationship("ParachainMetrics", back_populates="parachain")
    cross_chain_metrics = relationship("ParachainCrossChainMetrics", back_populates="parachain")
    defi_metrics = relationship("ParachainDeFiMetrics", back_populates="parachain")
    performance_metrics = relationship("ParachainPerformanceMetrics", back_populates="parachain")
    security_metrics = relationship("ParachainSecurityMetrics", back_populates="parachain")
    developer_metrics = relationship("ParachainDeveloperMetrics", back_populates="parachain")

# Enhanced Network Metrics
class PolkadotNetworkMetrics(Base):
    """Enhanced Polkadot main network metrics"""
    __tablename__ = 'polkadot_network_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Block metrics
    current_block = Column(Integer, nullable=True)
    block_time_avg = Column(Float, nullable=True)
    block_size_avg = Column(Float, nullable=True)
    transaction_throughput = Column(Float, nullable=True)
    blocks_per_hour = Column(Float, nullable=True)
    
    # Network performance
    network_utilization = Column(Float, nullable=True)
    finalization_time = Column(Float, nullable=True)
    consensus_latency = Column(Float, nullable=True)
    peer_count = Column(Integer, nullable=True)
    sync_status = Column(Boolean, nullable=True)
    
    # Transaction metrics
    total_transactions = Column(Integer, nullable=True)
    daily_transactions = Column(Integer, nullable=True)
    transaction_success_rate = Column(Float, nullable=True)
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    total_fees_24h = Column(Numeric(20, 8), nullable=True)
    
    # Account metrics
    total_accounts = Column(Integer, nullable=True)
    active_accounts_24h = Column(Integer, nullable=True)
    new_accounts_24h = Column(Integer, nullable=True)
    unique_addresses_7d = Column(Integer, nullable=True)
    
    # Runtime metrics
    runtime_version = Column(String(50), nullable=True)
    spec_version = Column(Integer, nullable=True)
    transaction_version = Column(Integer, nullable=True)
    
    # Validator metrics
    validator_count = Column(Integer, nullable=True)
    active_validators = Column(Integer, nullable=True)
    validator_set_size = Column(Integer, nullable=True)
    waiting_validators = Column(Integer, nullable=True)
    
    # Cross-chain metrics
    xcm_messages_24h = Column(Integer, nullable=True)
    hrmp_channels_active = Column(Integer, nullable=True)
    xcmp_channels_active = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="network_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_network_metrics_timestamp', 'timestamp'),
        Index('idx_network_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# Enhanced Staking Metrics
class PolkadotStakingMetrics(Base):
    """Enhanced Polkadot staking metrics"""
    __tablename__ = 'polkadot_staking_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Staking amounts
    total_staked = Column(Numeric(30, 8), nullable=True)
    total_staked_usd = Column(Numeric(30, 8), nullable=True)
    staking_ratio = Column(Float, nullable=True)
    staking_ratio_ideal = Column(Float, nullable=True)
    
    # Era information
    active_era = Column(Integer, nullable=True)
    current_era = Column(Integer, nullable=True)
    era_progress = Column(Float, nullable=True)
    era_length = Column(Integer, nullable=True)
    
    # Validator metrics
    validator_count = Column(Integer, nullable=True)
    active_validators = Column(Integer, nullable=True)
    waiting_validators = Column(Integer, nullable=True)
    nominator_count = Column(Integer, nullable=True)
    min_validator_stake = Column(Numeric(30, 8), nullable=True)
    max_validator_stake = Column(Numeric(30, 8), nullable=True)
    avg_validator_stake = Column(Numeric(30, 8), nullable=True)
    
    # Nomination pools
    nomination_pools_count = Column(Integer, nullable=True)
    nomination_pools_members = Column(Integer, nullable=True)
    nomination_pools_staked = Column(Numeric(30, 8), nullable=True)
    
    # Rewards
    block_reward = Column(Numeric(30, 8), nullable=True)
    validator_reward = Column(Numeric(30, 8), nullable=True)
    nominator_reward = Column(Numeric(30, 8), nullable=True)
    era_reward = Column(Numeric(30, 8), nullable=True)
    
    # Inflation
    inflation_rate = Column(Float, nullable=True)
    ideal_staking_rate = Column(Float, nullable=True)
    inflation_annual = Column(Float, nullable=True)
    
    # Slashing
    slashing_events_24h = Column(Integer, nullable=True)
    total_slashed_24h = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="staking_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_staking_metrics_timestamp', 'timestamp'),
        Index('idx_staking_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# Enhanced Governance Metrics
class PolkadotGovernanceMetrics(Base):
    """Enhanced Polkadot governance metrics"""
    __tablename__ = 'polkadot_governance_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Democracy metrics
    active_proposals = Column(Integer, nullable=True)
    total_proposals = Column(Integer, nullable=True)
    referendum_count = Column(Integer, nullable=True)
    active_referendums = Column(Integer, nullable=True)
    passed_referendums = Column(Integer, nullable=True)
    rejected_referendums = Column(Integer, nullable=True)
    
    # Council metrics
    council_members = Column(Integer, nullable=True)
    council_motions = Column(Integer, nullable=True)
    council_votes = Column(Integer, nullable=True)
    council_threshold = Column(Float, nullable=True)
    
    # Treasury metrics
    treasury_proposals = Column(Integer, nullable=True)
    treasury_spend_proposals = Column(Integer, nullable=True)
    treasury_bounty_proposals = Column(Integer, nullable=True)
    treasury_tip_proposals = Column(Integer, nullable=True)
    treasury_balance = Column(Numeric(30, 8), nullable=True)
    treasury_spend_rate = Column(Float, nullable=True)
    
    # Voting participation
    voter_participation_rate = Column(Float, nullable=True)
    total_votes_cast = Column(Integer, nullable=True)
    direct_voters = Column(Integer, nullable=True)
    delegated_voters = Column(Integer, nullable=True)
    delegation_count = Column(Integer, nullable=True)
    
    # Conviction voting
    conviction_voting_proposals = Column(Integer, nullable=True)
    conviction_voting_votes = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="governance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_governance_metrics_timestamp', 'timestamp'),
        Index('idx_governance_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# Enhanced Economic Metrics
class PolkadotEconomicMetrics(Base):
    """Enhanced Polkadot economic metrics"""
    __tablename__ = 'polkadot_economic_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Treasury
    treasury_balance = Column(Numeric(30, 8), nullable=True)
    treasury_balance_usd = Column(Numeric(30, 8), nullable=True)
    treasury_spend_rate = Column(Float, nullable=True)
    treasury_burn_rate = Column(Float, nullable=True)
    
    # Tokenomics
    total_supply = Column(Numeric(30, 8), nullable=True)
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    locked_supply = Column(Numeric(30, 8), nullable=True)
    inflation_rate = Column(Float, nullable=True)
    deflation_rate = Column(Float, nullable=True)
    burn_rate = Column(Float, nullable=True)
    
    # Market metrics
    market_cap = Column(Numeric(30, 8), nullable=True)
    market_cap_usd = Column(Numeric(30, 8), nullable=True)
    price_usd = Column(Numeric(20, 8), nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    volume_24h = Column(Numeric(30, 8), nullable=True)
    volume_7d = Column(Numeric(30, 8), nullable=True)
    
    # Transaction fees
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    total_fees_24h = Column(Numeric(20, 8), nullable=True)
    total_fees_7d = Column(Numeric(20, 8), nullable=True)
    fee_burn_24h = Column(Numeric(20, 8), nullable=True)
    
    # Economic activity
    gdp_24h = Column(Numeric(30, 8), nullable=True)  # Gross Domestic Product
    velocity = Column(Float, nullable=True)  # Money velocity
    nvt_ratio = Column(Float, nullable=True)  # Network Value to Transactions
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="economic_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_economic_metrics_timestamp', 'timestamp'),
        Index('idx_economic_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# New Performance Metrics
class PolkadotPerformanceMetrics(Base):
    """Polkadot performance and technical metrics"""
    __tablename__ = 'polkadot_performance_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Network performance
    network_latency_avg = Column(Float, nullable=True)
    network_throughput = Column(Float, nullable=True)
    consensus_time = Column(Float, nullable=True)
    finalization_time = Column(Float, nullable=True)
    block_propagation_time = Column(Float, nullable=True)
    
    # Scalability metrics
    transactions_per_second = Column(Float, nullable=True)
    blocks_per_minute = Column(Float, nullable=True)
    parachain_slot_utilization = Column(Float, nullable=True)
    max_throughput = Column(Float, nullable=True)
    
    # Resource utilization
    cpu_usage_avg = Column(Float, nullable=True)
    memory_usage_avg = Column(Float, nullable=True)
    storage_growth_rate = Column(Float, nullable=True)
    bandwidth_usage = Column(Float, nullable=True)
    
    # Error rates
    block_production_error_rate = Column(Float, nullable=True)
    transaction_failure_rate = Column(Float, nullable=True)
    network_partition_events = Column(Integer, nullable=True)
    sync_errors = Column(Integer, nullable=True)
    
    # Quality of service
    service_availability = Column(Float, nullable=True)
    response_time_p95 = Column(Float, nullable=True)
    response_time_p99 = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    uptime = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_metrics_timestamp', 'timestamp'),
        Index('idx_performance_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# New Security Metrics
class PolkadotSecurityMetrics(Base):
    """Polkadot security metrics"""
    __tablename__ = 'polkadot_security_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Validator security
    validator_uptime_avg = Column(Float, nullable=True)
    validator_commission_avg = Column(Float, nullable=True)
    slashing_events = Column(Integer, nullable=True)
    total_slashed = Column(Numeric(30, 8), nullable=True)
    
    # Network security
    nakamoto_coefficient = Column(Integer, nullable=True)
    gini_coefficient = Column(Float, nullable=True)
    decentralization_index = Column(Float, nullable=True)
    validator_geographic_distribution = Column(Integer, nullable=True)
    
    # Security incidents
    security_incidents_24h = Column(Integer, nullable=True)
    security_incidents_7d = Column(Integer, nullable=True)
    critical_vulnerabilities = Column(Integer, nullable=True)
    resolved_vulnerabilities = Column(Integer, nullable=True)
    
    # Audit metrics
    security_audits_completed = Column(Integer, nullable=True)
    bug_bounty_rewards_paid = Column(Numeric(20, 8), nullable=True)
    bug_bounty_reports = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="security_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_security_metrics_timestamp', 'timestamp'),
        Index('idx_security_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# New Developer Metrics
class PolkadotDeveloperMetrics(Base):
    """Polkadot developer activity metrics"""
    __tablename__ = 'polkadot_developer_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Developer activity
    total_developers = Column(Integer, nullable=True)
    full_time_developers = Column(Integer, nullable=True)
    part_time_developers = Column(Integer, nullable=True)
    new_developers_30d = Column(Integer, nullable=True)
    
    # Code activity
    github_commits_24h = Column(Integer, nullable=True)
    github_commits_7d = Column(Integer, nullable=True)
    github_commits_30d = Column(Integer, nullable=True)
    github_stars = Column(Integer, nullable=True)
    github_forks = Column(Integer, nullable=True)
    
    # Project metrics
    active_projects = Column(Integer, nullable=True)
    new_projects_launched = Column(Integer, nullable=True)
    projects_funded = Column(Integer, nullable=True)
    total_funding = Column(Numeric(30, 8), nullable=True)
    
    # Documentation
    documentation_updates = Column(Integer, nullable=True)
    tutorial_views = Column(Integer, nullable=True)
    community_questions = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="developer_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_developer_metrics_timestamp', 'timestamp'),
        Index('idx_developer_metrics_network_timestamp', 'network_id', 'timestamp'),
    )

# Enhanced Parachain Metrics
class ParachainMetrics(Base):
    """Enhanced parachain-specific metrics"""
    __tablename__ = 'parachain_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Block metrics
    current_block = Column(Integer, nullable=True)
    block_time_avg = Column(Float, nullable=True)
    block_production_rate = Column(Float, nullable=True)
    blocks_per_hour = Column(Float, nullable=True)
    
    # Transaction metrics
    daily_transactions = Column(Integer, nullable=True)
    total_transactions = Column(Integer, nullable=True)
    transaction_volume_24h = Column(Numeric(30, 8), nullable=True)
    transaction_volume_7d = Column(Numeric(30, 8), nullable=True)
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    transaction_success_rate = Column(Float, nullable=True)
    transaction_throughput = Column(Float, nullable=True)
    
    # User activity
    active_addresses_24h = Column(Integer, nullable=True)
    active_addresses_7d = Column(Integer, nullable=True)
    new_addresses_24h = Column(Integer, nullable=True)
    unique_users_7d = Column(Integer, nullable=True)
    total_accounts = Column(Integer, nullable=True)
    
    # Token metrics
    token_supply = Column(Numeric(30, 8), nullable=True)
    token_circulation = Column(Numeric(30, 8), nullable=True)
    token_price = Column(Numeric(20, 8), nullable=True)
    market_cap = Column(Numeric(30, 8), nullable=True)
    
    # Network health
    validator_count = Column(Integer, nullable=True)
    collator_count = Column(Integer, nullable=True)
    network_utilization = Column(Float, nullable=True)
    uptime = Column(Float, nullable=True)
    
    # Smart contracts
    new_contracts_deployed = Column(Integer, nullable=True)
    total_contracts = Column(Integer, nullable=True)
    contract_interactions_24h = Column(Integer, nullable=True)
    contract_gas_used = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="parachain_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_parachain_metrics_timestamp', 'timestamp'),
        Index('idx_parachain_metrics_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# Enhanced Cross-Chain Metrics
class ParachainCrossChainMetrics(Base):
    """Enhanced parachain cross-chain messaging metrics"""
    __tablename__ = 'parachain_cross_chain_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # HRMP (Horizontal Relay-routed Message Passing)
    hrmp_channels_count = Column(Integer, nullable=True)
    hrmp_messages_sent_24h = Column(Integer, nullable=True)
    hrmp_messages_received_24h = Column(Integer, nullable=True)
    hrmp_volume_24h = Column(Numeric(30, 8), nullable=True)
    hrmp_fees_24h = Column(Numeric(20, 8), nullable=True)
    
    # XCMP (Cross-Chain Message Passing)
    xcmp_channels_count = Column(Integer, nullable=True)
    xcmp_messages_sent_24h = Column(Integer, nullable=True)
    xcmp_messages_received_24h = Column(Integer, nullable=True)
    xcmp_volume_24h = Column(Numeric(30, 8), nullable=True)
    xcmp_fees_24h = Column(Numeric(20, 8), nullable=True)
    
    # XCM (Cross-Consensus Message)
    xcm_messages_sent_24h = Column(Integer, nullable=True)
    xcm_messages_received_24h = Column(Integer, nullable=True)
    xcm_volume_24h = Column(Numeric(30, 8), nullable=True)
    xcm_fees_24h = Column(Numeric(20, 8), nullable=True)
    
    # Bridge metrics (if applicable)
    bridge_volume_24h = Column(Numeric(30, 8), nullable=True)
    bridge_transactions_24h = Column(Integer, nullable=True)
    bridge_fees_24h = Column(Numeric(20, 8), nullable=True)
    bridge_tvl = Column(Numeric(30, 8), nullable=True)
    
    # Cross-chain liquidity
    cross_chain_liquidity = Column(Numeric(30, 8), nullable=True)
    liquidity_imbalance = Column(Float, nullable=True)
    arbitrage_opportunities = Column(Integer, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="cross_chain_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_cross_chain_metrics_timestamp', 'timestamp'),
        Index('idx_cross_chain_metrics_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# New DeFi Metrics for Parachains
class ParachainDeFiMetrics(Base):
    """Parachain DeFi-specific metrics"""
    __tablename__ = 'parachain_defi_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # TVL metrics
    total_tvl = Column(Numeric(30, 8), nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    tvl_change_30d = Column(Float, nullable=True)
    
    # DEX metrics
    dex_volume_24h = Column(Numeric(30, 8), nullable=True)
    dex_volume_7d = Column(Numeric(30, 8), nullable=True)
    dex_trades_24h = Column(Integer, nullable=True)
    dex_liquidity_pools = Column(Integer, nullable=True)
    dex_liquidity_tvl = Column(Numeric(30, 8), nullable=True)
    
    # Lending metrics
    lending_tvl = Column(Numeric(30, 8), nullable=True)
    total_borrowed = Column(Numeric(30, 8), nullable=True)
    lending_apy_avg = Column(Float, nullable=True)
    borrowing_apy_avg = Column(Float, nullable=True)
    liquidation_events_24h = Column(Integer, nullable=True)
    
    # Staking metrics
    liquid_staking_tvl = Column(Numeric(30, 8), nullable=True)
    staking_apy_avg = Column(Float, nullable=True)
    staking_participation_rate = Column(Float, nullable=True)
    
    # Yield farming
    yield_farming_tvl = Column(Numeric(30, 8), nullable=True)
    yield_farming_apy_avg = Column(Float, nullable=True)
    active_farms = Column(Integer, nullable=True)
    
    # Derivatives
    derivatives_tvl = Column(Numeric(30, 8), nullable=True)
    derivatives_volume_24h = Column(Numeric(30, 8), nullable=True)
    options_volume_24h = Column(Numeric(30, 8), nullable=True)
    futures_volume_24h = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="defi_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_defi_metrics_timestamp', 'timestamp'),
        Index('idx_defi_metrics_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# New Performance Metrics for Parachains
class ParachainPerformanceMetrics(Base):
    """Parachain performance metrics"""
    __tablename__ = 'parachain_performance_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    tps = Column(Float, nullable=True)  # Transactions per second
    block_time = Column(Float, nullable=True)
    finalization_time = Column(Float, nullable=True)
    latency_avg = Column(Float, nullable=True)
    
    # Resource utilization
    cpu_usage = Column(Float, nullable=True)
    memory_usage = Column(Float, nullable=True)
    storage_usage = Column(Float, nullable=True)
    bandwidth_usage = Column(Float, nullable=True)
    
    # Error rates
    error_rate = Column(Float, nullable=True)
    transaction_failure_rate = Column(Float, nullable=True)
    block_production_failures = Column(Integer, nullable=True)
    
    # Quality metrics
    uptime = Column(Float, nullable=True)
    availability = Column(Float, nullable=True)
    response_time_p95 = Column(Float, nullable=True)
    response_time_p99 = Column(Float, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_parachain_performance_timestamp', 'timestamp'),
        Index('idx_parachain_performance_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# New Security Metrics for Parachains
class ParachainSecurityMetrics(Base):
    """Parachain security metrics"""
    __tablename__ = 'parachain_security_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Security incidents
    security_incidents_24h = Column(Integer, nullable=True)
    security_incidents_7d = Column(Integer, nullable=True)
    critical_vulnerabilities = Column(Integer, nullable=True)
    resolved_vulnerabilities = Column(Integer, nullable=True)
    
    # Audit metrics
    security_audits_completed = Column(Integer, nullable=True)
    bug_bounty_rewards_paid = Column(Numeric(20, 8), nullable=True)
    bug_bounty_reports = Column(Integer, nullable=True)
    
    # Smart contract security
    contract_vulnerabilities = Column(Integer, nullable=True)
    contract_audits = Column(Integer, nullable=True)
    contract_upgrades = Column(Integer, nullable=True)
    
    # Network security
    validator_uptime = Column(Float, nullable=True)
    collator_uptime = Column(Float, nullable=True)
    network_attacks_blocked = Column(Integer, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="security_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_parachain_security_timestamp', 'timestamp'),
        Index('idx_parachain_security_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# New Developer Metrics for Parachains
class ParachainDeveloperMetrics(Base):
    """Parachain developer activity metrics"""
    __tablename__ = 'parachain_developer_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Developer activity
    active_developers = Column(Integer, nullable=True)
    new_developers_30d = Column(Integer, nullable=True)
    github_commits_24h = Column(Integer, nullable=True)
    github_commits_7d = Column(Integer, nullable=True)
    github_stars = Column(Integer, nullable=True)
    github_forks = Column(Integer, nullable=True)
    
    # Project metrics
    active_projects = Column(Integer, nullable=True)
    new_projects_launched = Column(Integer, nullable=True)
    projects_funded = Column(Integer, nullable=True)
    total_funding = Column(Numeric(30, 8), nullable=True)
    
    # Documentation
    documentation_updates = Column(Integer, nullable=True)
    tutorial_views = Column(Integer, nullable=True)
    community_questions = Column(Integer, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="developer_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_parachain_developer_timestamp', 'timestamp'),
        Index('idx_parachain_developer_parachain_timestamp', 'parachain_id', 'timestamp'),
    )

# Ecosystem-wide metrics
class PolkadotEcosystemMetrics(Base):
    """Overall Polkadot ecosystem metrics"""
    __tablename__ = 'polkadot_ecosystem_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Ecosystem overview
    total_parachains = Column(Integer, nullable=True)
    active_parachains = Column(Integer, nullable=True)
    parachain_slots_available = Column(Integer, nullable=True)
    parachain_slots_occupied = Column(Integer, nullable=True)
    
    # Total ecosystem TVL
    total_ecosystem_tvl = Column(Numeric(30, 8), nullable=True)
    total_ecosystem_tvl_usd = Column(Numeric(30, 8), nullable=True)
    tvl_growth_rate = Column(Float, nullable=True)
    
    # Cross-chain activity
    total_cross_chain_messages_24h = Column(Integer, nullable=True)
    total_cross_chain_volume_24h = Column(Numeric(30, 8), nullable=True)
    active_cross_chain_channels = Column(Integer, nullable=True)
    
    # Developer activity
    total_active_developers = Column(Integer, nullable=True)
    total_new_projects_launched = Column(Integer, nullable=True)
    total_github_commits_24h = Column(Integer, nullable=True)
    total_github_stars = Column(Integer, nullable=True)
    
    # Community metrics
    social_media_mentions_24h = Column(Integer, nullable=True)
    community_growth_rate = Column(Float, nullable=True)
    telegram_members = Column(Integer, nullable=True)
    discord_members = Column(Integer, nullable=True)
    twitter_followers = Column(Integer, nullable=True)
    
    # Security metrics
    total_security_audits_completed = Column(Integer, nullable=True)
    total_bug_bounty_rewards_paid = Column(Numeric(20, 8), nullable=True)
    total_security_incidents_24h = Column(Integer, nullable=True)
    
    # Innovation metrics
    total_new_features_released = Column(Integer, nullable=True)
    total_protocol_upgrades = Column(Integer, nullable=True)
    total_parachain_upgrades = Column(Integer, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_ecosystem_metrics_timestamp', 'timestamp'),
    )

# Market data for tokens
class TokenMarketData(Base):
    """Market data for Polkadot and parachain tokens"""
    __tablename__ = 'token_market_data'
    
    id = Column(Integer, primary_key=True)
    token_symbol = Column(String(20), nullable=False)
    token_name = Column(String(100), nullable=False)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Price data
    price_usd = Column(Numeric(20, 8), nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    price_change_90d = Column(Float, nullable=True)
    
    # Market data
    market_cap = Column(Numeric(30, 8), nullable=True)
    market_cap_rank = Column(Integer, nullable=True)
    volume_24h = Column(Numeric(30, 8), nullable=True)
    volume_7d = Column(Numeric(30, 8), nullable=True)
    
    # Supply data
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    total_supply = Column(Numeric(30, 8), nullable=True)
    max_supply = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    parachain = relationship("Parachain")
    
    # Indexes
    __table_args__ = (
        Index('idx_token_market_timestamp', 'timestamp'),
        Index('idx_token_market_symbol_timestamp', 'token_symbol', 'timestamp'),
    )

# Validator information
class ValidatorInfo(Base):
    """Validator information and metrics"""
    __tablename__ = 'validator_info'
    
    id = Column(Integer, primary_key=True)
    validator_address = Column(String(100), nullable=False, unique=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Validator details
    name = Column(String(200), nullable=True)
    commission = Column(Float, nullable=True)
    total_stake = Column(Numeric(30, 8), nullable=True)
    own_stake = Column(Numeric(30, 8), nullable=True)
    nominator_count = Column(Integer, nullable=True)
    
    # Performance metrics
    era_points = Column(Integer, nullable=True)
    era_rewards = Column(Numeric(30, 8), nullable=True)
    slashing_events = Column(Integer, nullable=True)
    uptime = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_waiting = Column(Boolean, default=False)
    is_chilling = Column(Boolean, default=False)
    
    # Relationships
    network = relationship("PolkadotNetwork")
    
    # Indexes
    __table_args__ = (
        Index('idx_validator_info_timestamp', 'timestamp'),
        Index('idx_validator_info_network_timestamp', 'network_id', 'timestamp'),
    )
