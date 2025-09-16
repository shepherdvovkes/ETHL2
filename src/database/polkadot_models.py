from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Numeric, JSON, BigInteger
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

class PolkadotNetwork(Base):
    """Polkadot main network information"""
    __tablename__ = 'polkadot_networks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    chain_id = Column(String(50), nullable=False)
    rpc_endpoint = Column(String(500), nullable=True)
    ws_endpoint = Column(String(500), nullable=True)
    is_mainnet = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parachains = relationship("Parachain", back_populates="network")
    network_metrics = relationship("PolkadotNetworkMetrics", back_populates="network")
    staking_metrics = relationship("PolkadotStakingMetrics", back_populates="network")
    governance_metrics = relationship("PolkadotGovernanceMetrics", back_populates="network")
    economic_metrics = relationship("PolkadotEconomicMetrics", back_populates="network")

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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="parachains")
    parachain_metrics = relationship("ParachainMetrics", back_populates="parachain")
    cross_chain_metrics = relationship("ParachainCrossChainMetrics", back_populates="parachain")
    defi_metrics = relationship("PolkadotDeFiMetrics", back_populates="parachain")

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
    block_production_rate = Column(Float, nullable=True)
    block_finalization_rate = Column(Float, nullable=True)
    
    # Network performance
    network_utilization = Column(Float, nullable=True)
    finalization_time = Column(Float, nullable=True)
    consensus_latency = Column(Float, nullable=True)
    peer_count = Column(Integer, nullable=True)
    sync_status = Column(Boolean, nullable=True)
    network_bandwidth_usage = Column(Float, nullable=True)
    network_latency_p95 = Column(Float, nullable=True)
    
    # Transaction metrics
    total_transactions = Column(Integer, nullable=True)
    daily_transactions = Column(Integer, nullable=True)
    transaction_success_rate = Column(Float, nullable=True)
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    total_fees_24h = Column(Numeric(20, 8), nullable=True)
    transaction_queue_size = Column(Integer, nullable=True)
    
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
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="network_metrics")

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
    staking_ratio_change_24h = Column(Float, nullable=True)
    staking_ratio_change_7d = Column(Float, nullable=True)
    
    # Era information
    active_era = Column(Integer, nullable=True)
    current_era = Column(Integer, nullable=True)
    era_progress = Column(Float, nullable=True)
    era_length = Column(Integer, nullable=True)
    era_remaining_blocks = Column(Integer, nullable=True)
    
    # Validator metrics
    validator_count = Column(Integer, nullable=True)
    nominator_count = Column(Integer, nullable=True)
    min_validator_stake = Column(Numeric(30, 8), nullable=True)
    max_validator_stake = Column(Numeric(30, 8), nullable=True)
    avg_validator_stake = Column(Numeric(30, 8), nullable=True)
    median_validator_stake = Column(Numeric(30, 8), nullable=True)
    
    # Nomination pools
    nomination_pool_count = Column(Integer, nullable=True)
    nomination_pool_members = Column(Integer, nullable=True)
    nomination_pool_tvl = Column(Numeric(30, 8), nullable=True)
    nomination_pool_apy = Column(Float, nullable=True)
    
    # Rewards
    block_reward = Column(Numeric(30, 8), nullable=True)
    validator_reward = Column(Numeric(30, 8), nullable=True)
    nominator_reward = Column(Numeric(30, 8), nullable=True)
    era_reward = Column(Numeric(30, 8), nullable=True)
    total_rewards_24h = Column(Numeric(30, 8), nullable=True)
    
    # Inflation
    inflation_rate = Column(Float, nullable=True)
    ideal_staking_rate = Column(Float, nullable=True)
    annual_inflation = Column(Float, nullable=True)
    deflation_rate = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="staking_metrics")

class PolkadotGovernanceMetrics(Base):
    """Enhanced Polkadot governance metrics"""
    __tablename__ = 'polkadot_governance_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Democracy metrics
    active_proposals = Column(Integer, nullable=True)
    referendum_count = Column(Integer, nullable=True)
    active_referendums = Column(Integer, nullable=True)
    referendum_success_rate = Column(Float, nullable=True)
    referendum_turnout_rate = Column(Float, nullable=True)
    
    # Council metrics
    council_members = Column(Integer, nullable=True)
    council_motions = Column(Integer, nullable=True)
    council_votes = Column(Integer, nullable=True)
    council_motion_approval_rate = Column(Float, nullable=True)
    council_activity_score = Column(Float, nullable=True)
    
    # Treasury metrics
    treasury_proposals = Column(Integer, nullable=True)
    treasury_spend_proposals = Column(Integer, nullable=True)
    treasury_bounty_proposals = Column(Integer, nullable=True)
    treasury_proposal_approval_rate = Column(Float, nullable=True)
    treasury_spend_rate = Column(Float, nullable=True)
    
    # Voting participation
    voter_participation_rate = Column(Float, nullable=True)
    total_votes_cast = Column(Integer, nullable=True)
    direct_voters = Column(Integer, nullable=True)
    delegated_voters = Column(Integer, nullable=True)
    conviction_voting_usage = Column(Float, nullable=True)
    
    # Governance efficiency
    proposal_implementation_time = Column(Float, nullable=True)
    governance_activity_score = Column(Float, nullable=True)
    community_engagement_score = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="governance_metrics")

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
    treasury_proposal_funding_rate = Column(Float, nullable=True)
    
    # Tokenomics
    total_supply = Column(Numeric(30, 8), nullable=True)
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    inflation_rate = Column(Float, nullable=True)
    deflation_rate = Column(Float, nullable=True)
    token_velocity = Column(Float, nullable=True)
    token_holder_count = Column(Integer, nullable=True)
    
    # Market metrics
    market_cap = Column(Numeric(30, 8), nullable=True)
    price_usd = Column(Numeric(20, 8), nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    volume_24h = Column(Numeric(30, 8), nullable=True)
    volume_7d = Column(Numeric(30, 8), nullable=True)
    
    # Transaction fees
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    total_fees_24h = Column(Numeric(20, 8), nullable=True)
    fee_burn_24h = Column(Numeric(20, 8), nullable=True)
    fee_burn_rate = Column(Float, nullable=True)
    
    # Economic indicators
    gdp_equivalent = Column(Numeric(30, 8), nullable=True)
    economic_activity_score = Column(Float, nullable=True)
    liquidity_score = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork", back_populates="economic_metrics")

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
    block_finalization_rate = Column(Float, nullable=True)
    blocks_per_hour = Column(Float, nullable=True)
    
    # Transaction metrics
    daily_transactions = Column(Integer, nullable=True)
    transaction_volume_24h = Column(Numeric(30, 8), nullable=True)
    avg_transaction_fee = Column(Numeric(20, 8), nullable=True)
    transaction_success_rate = Column(Float, nullable=True)
    transaction_throughput = Column(Float, nullable=True)
    transaction_queue_size = Column(Integer, nullable=True)
    
    # User activity
    active_addresses_24h = Column(Integer, nullable=True)
    new_addresses_24h = Column(Integer, nullable=True)
    unique_users_7d = Column(Integer, nullable=True)
    total_addresses = Column(Integer, nullable=True)
    address_growth_rate = Column(Float, nullable=True)
    
    # DeFi metrics (if applicable)
    tvl = Column(Numeric(30, 8), nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    tvl_change_30d = Column(Float, nullable=True)
    tvl_rank = Column(Integer, nullable=True)
    
    # Smart contracts
    new_contracts_deployed = Column(Integer, nullable=True)
    contract_interactions_24h = Column(Integer, nullable=True)
    total_contracts = Column(Integer, nullable=True)
    contract_deployment_rate = Column(Float, nullable=True)
    
    # Network health
    validator_count = Column(Integer, nullable=True)
    collator_count = Column(Integer, nullable=True)
    network_utilization = Column(Float, nullable=True)
    uptime_percentage = Column(Float, nullable=True)
    sync_status = Column(Boolean, nullable=True)
    
    # Performance metrics
    latency_avg = Column(Float, nullable=True)
    throughput_avg = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="parachain_metrics")

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
    hrmp_message_success_rate = Column(Float, nullable=True)
    hrmp_channel_utilization = Column(Float, nullable=True)
    
    # XCMP (Cross-Chain Message Passing)
    xcmp_channels_count = Column(Integer, nullable=True)
    xcmp_messages_sent_24h = Column(Integer, nullable=True)
    xcmp_messages_received_24h = Column(Integer, nullable=True)
    xcmp_volume_24h = Column(Numeric(30, 8), nullable=True)
    xcmp_message_success_rate = Column(Float, nullable=True)
    xcmp_channel_utilization = Column(Float, nullable=True)
    
    # Bridge metrics (if applicable)
    bridge_volume_24h = Column(Numeric(30, 8), nullable=True)
    bridge_transactions_24h = Column(Integer, nullable=True)
    bridge_fees_24h = Column(Numeric(20, 8), nullable=True)
    bridge_success_rate = Column(Float, nullable=True)
    bridge_latency_avg = Column(Float, nullable=True)
    
    # Cross-chain liquidity
    cross_chain_liquidity = Column(Numeric(30, 8), nullable=True)
    liquidity_imbalance = Column(Float, nullable=True)
    arbitrage_opportunities = Column(Integer, nullable=True)
    cross_chain_arbitrage_volume = Column(Numeric(30, 8), nullable=True)
    
    # Message analysis
    message_type_distribution = Column(JSON, nullable=True)
    message_size_analysis = Column(JSON, nullable=True)
    message_priority_analysis = Column(JSON, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="cross_chain_metrics")

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

class PolkadotPerformanceMetrics(Base):
    """Polkadot performance and technical metrics"""
    __tablename__ = 'polkadot_performance_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Network performance
    network_latency_avg = Column(Float, nullable=True)
    network_throughput = Column(Float, nullable=True)
    consensus_time = Column(Float, nullable=True)
    finalization_time = Column(Float, nullable=True)
    
    # Scalability metrics
    transactions_per_second = Column(Float, nullable=True)
    blocks_per_minute = Column(Float, nullable=True)
    parachain_slot_utilization = Column(Float, nullable=True)
    
    # Resource utilization
    cpu_usage_avg = Column(Float, nullable=True)
    memory_usage_avg = Column(Float, nullable=True)
    storage_growth_rate = Column(Float, nullable=True)
    
    # Error rates
    block_production_error_rate = Column(Float, nullable=True)
    transaction_failure_rate = Column(Float, nullable=True)
    network_partition_events = Column(Integer, nullable=True)
    
    # Quality of service
    service_availability = Column(Float, nullable=True)
    response_time_p95 = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)

# ===== COMPREHENSIVE METRICS MODELS =====

class PolkadotSecurityMetrics(Base):
    """Critical security metrics for Polkadot"""
    __tablename__ = 'polkadot_security_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Slashing events
    slash_events_count_24h = Column(Integer, nullable=True)
    slash_events_total_amount = Column(Numeric(30, 8), nullable=True)
    validator_slash_events = Column(Integer, nullable=True)
    nominator_slash_events = Column(Integer, nullable=True)
    
    # Slash reasons
    unjustified_slash_events = Column(Integer, nullable=True)
    justified_slash_events = Column(Integer, nullable=True)
    equivocation_slash_events = Column(Integer, nullable=True)
    offline_slash_events = Column(Integer, nullable=True)
    grandpa_equivocation_events = Column(Integer, nullable=True)
    babe_equivocation_events = Column(Integer, nullable=True)
    
    # Security incidents
    security_incidents_count = Column(Integer, nullable=True)
    network_attacks_detected = Column(Integer, nullable=True)
    validator_compromise_events = Column(Integer, nullable=True)
    
    # Network health
    fork_events_count = Column(Integer, nullable=True)
    chain_reorganization_events = Column(Integer, nullable=True)
    consensus_failure_events = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotValidatorMetrics(Base):
    """Advanced validator performance metrics"""
    __tablename__ = 'polkadot_validator_metrics'
    
    id = Column(Integer, primary_key=True)
    validator_id = Column(String(100), nullable=False)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    uptime_percentage = Column(Float, nullable=True)
    block_production_rate = Column(Float, nullable=True)
    era_points_earned = Column(Integer, nullable=True)
    commission_rate = Column(Float, nullable=True)
    
    # Staking metrics
    self_stake_amount = Column(Numeric(30, 8), nullable=True)
    total_stake_amount = Column(Numeric(30, 8), nullable=True)
    nominator_count = Column(Integer, nullable=True)
    
    # Infrastructure metrics
    geographic_location = Column(String(100), nullable=True)
    hosting_provider = Column(String(100), nullable=True)
    hardware_specs = Column(JSON, nullable=True)
    
    # Network metrics
    peer_connections = Column(Integer, nullable=True)
    sync_status = Column(Boolean, nullable=True)
    chain_data_size = Column(BigInteger, nullable=True)
    
    # Resource usage
    cpu_usage_percentage = Column(Float, nullable=True)
    memory_usage_percentage = Column(Float, nullable=True)
    disk_usage_percentage = Column(Float, nullable=True)
    network_bandwidth_usage = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotParachainSlotMetrics(Base):
    """Parachain slot auction and lease metrics"""
    __tablename__ = 'polkadot_parachain_slot_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Slot auction metrics
    slot_auction_id = Column(Integer, nullable=True)
    slot_auction_status = Column(String(50), nullable=True)
    winning_bid_amount = Column(Numeric(30, 8), nullable=True)
    crowdloan_total_amount = Column(Numeric(30, 8), nullable=True)
    crowdloan_participant_count = Column(Integer, nullable=True)
    
    # Lease metrics
    lease_period_start = Column(Integer, nullable=True)
    lease_period_end = Column(Integer, nullable=True)
    lease_periods_remaining = Column(Integer, nullable=True)
    lease_renewal_probability = Column(Float, nullable=True)
    
    # Market metrics
    slot_competition_ratio = Column(Float, nullable=True)
    slot_price_trend = Column(Float, nullable=True)
    slot_utilization_rate = Column(Float, nullable=True)
    
    # Historical data
    previous_lease_periods = Column(JSON, nullable=True)
    historical_bid_data = Column(JSON, nullable=True)
    crowdloan_history = Column(JSON, nullable=True)
    
    # Relationships
    parachain = relationship("Parachain")

class PolkadotCrossChainAdvancedMetrics(Base):
    """Advanced cross-chain messaging metrics"""
    __tablename__ = 'polkadot_cross_chain_advanced_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # XCMP metrics
    xcmp_message_failure_rate = Column(Float, nullable=True)
    xcmp_message_retry_count = Column(Integer, nullable=True)
    xcmp_message_processing_time = Column(Float, nullable=True)
    xcmp_channel_capacity_utilization = Column(Float, nullable=True)
    xcmp_channel_fee_analysis = Column(JSON, nullable=True)
    
    # HRMP metrics
    hrmp_channel_opening_requests = Column(Integer, nullable=True)
    hrmp_channel_closing_requests = Column(Integer, nullable=True)
    hrmp_channel_deposit_requirements = Column(Numeric(30, 8), nullable=True)
    hrmp_channel_utilization_rate = Column(Float, nullable=True)
    
    # Bridge metrics
    cross_chain_bridge_volume = Column(Numeric(30, 8), nullable=True)
    cross_chain_bridge_fees = Column(Numeric(30, 8), nullable=True)
    cross_chain_bridge_success_rate = Column(Float, nullable=True)
    cross_chain_bridge_latency = Column(Float, nullable=True)
    
    # Message analysis
    message_type_distribution = Column(JSON, nullable=True)
    message_size_analysis = Column(JSON, nullable=True)
    message_priority_analysis = Column(JSON, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotGovernanceAdvancedMetrics(Base):
    """Advanced governance analytics"""
    __tablename__ = 'polkadot_governance_advanced_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Referendum analytics
    referendum_turnout_by_proposal_type = Column(JSON, nullable=True)
    referendum_success_rate_by_category = Column(JSON, nullable=True)
    governance_proposal_implementation_time = Column(Float, nullable=True)
    
    # Voter analytics
    governance_voter_demographics = Column(JSON, nullable=True)
    governance_delegation_patterns = Column(JSON, nullable=True)
    governance_conviction_voting_analysis = Column(JSON, nullable=True)
    
    # Committee activity
    governance_technical_committee_activity = Column(JSON, nullable=True)
    governance_fellowship_activity = Column(JSON, nullable=True)
    
    # Treasury analytics
    governance_treasury_proposal_approval_rate = Column(Float, nullable=True)
    governance_community_sentiment_analysis = Column(JSON, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotEconomicAdvancedMetrics(Base):
    """Advanced economic analysis metrics"""
    __tablename__ = 'polkadot_economic_advanced_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Token analysis
    token_velocity_analysis = Column(JSON, nullable=True)
    token_holder_distribution = Column(JSON, nullable=True)
    token_whale_movement_tracking = Column(JSON, nullable=True)
    token_institutional_holdings = Column(JSON, nullable=True)
    
    # Economic pressure
    token_deflationary_pressure = Column(Float, nullable=True)
    token_burn_rate_analysis = Column(JSON, nullable=True)
    token_staking_yield_analysis = Column(JSON, nullable=True)
    
    # Market analysis
    token_liquidity_analysis = Column(JSON, nullable=True)
    token_correlation_analysis = Column(JSON, nullable=True)
    token_volatility_metrics = Column(JSON, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotInfrastructureMetrics(Base):
    """Network infrastructure diversity metrics"""
    __tablename__ = 'polkadot_infrastructure_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Geographic distribution
    node_geographic_distribution = Column(JSON, nullable=True)
    validator_geographic_distribution = Column(JSON, nullable=True)
    
    # Infrastructure diversity
    node_hosting_provider_diversity = Column(JSON, nullable=True)
    node_hardware_diversity = Column(JSON, nullable=True)
    node_network_topology_analysis = Column(JSON, nullable=True)
    
    # Network quality
    node_peer_connection_quality = Column(JSON, nullable=True)
    network_decentralization_index = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotDeveloperMetrics(Base):
    """Developer ecosystem and activity metrics"""
    __tablename__ = 'polkadot_developer_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # GitHub activity
    github_commits_24h = Column(Integer, nullable=True)
    github_commits_7d = Column(Integer, nullable=True)
    github_commits_30d = Column(Integer, nullable=True)
    github_stars_total = Column(Integer, nullable=True)
    github_forks_total = Column(Integer, nullable=True)
    github_contributors = Column(Integer, nullable=True)
    
    # Project metrics
    active_projects = Column(Integer, nullable=True)
    new_projects_launched = Column(Integer, nullable=True)
    projects_funded = Column(Integer, nullable=True)
    total_funding_amount = Column(Numeric(30, 8), nullable=True)
    
    # Developer engagement
    active_developers = Column(Integer, nullable=True)
    new_developers = Column(Integer, nullable=True)
    developer_retention_rate = Column(Float, nullable=True)
    developer_satisfaction_score = Column(Float, nullable=True)
    
    # Documentation metrics
    documentation_updates = Column(Integer, nullable=True)
    tutorial_views = Column(Integer, nullable=True)
    community_questions = Column(Integer, nullable=True)
    support_tickets = Column(Integer, nullable=True)
    
    # Code quality
    code_review_activity = Column(Integer, nullable=True)
    test_coverage = Column(Float, nullable=True)
    bug_reports = Column(Integer, nullable=True)
    security_audits = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotCommunityMetrics(Base):
    """Community engagement and social metrics"""
    __tablename__ = 'polkadot_community_metrics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Social media metrics
    twitter_followers = Column(Integer, nullable=True)
    twitter_mentions_24h = Column(Integer, nullable=True)
    telegram_members = Column(Integer, nullable=True)
    discord_members = Column(Integer, nullable=True)
    reddit_subscribers = Column(Integer, nullable=True)
    
    # Community growth
    community_growth_rate = Column(Float, nullable=True)
    new_members_24h = Column(Integer, nullable=True)
    active_members_7d = Column(Integer, nullable=True)
    community_engagement_score = Column(Float, nullable=True)
    
    # Event metrics
    events_held = Column(Integer, nullable=True)
    event_attendees = Column(Integer, nullable=True)
    conference_participants = Column(Integer, nullable=True)
    meetup_attendance = Column(Integer, nullable=True)
    
    # Content metrics
    blog_posts = Column(Integer, nullable=True)
    video_views = Column(Integer, nullable=True)
    podcast_downloads = Column(Integer, nullable=True)
    newsletter_subscribers = Column(Integer, nullable=True)
    
    # Sentiment analysis
    sentiment_score = Column(Float, nullable=True)
    positive_mentions = Column(Integer, nullable=True)
    negative_mentions = Column(Integer, nullable=True)
    neutral_mentions = Column(Integer, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")

class PolkadotDeFiMetrics(Base):
    """DeFi ecosystem metrics across parachains"""
    __tablename__ = 'polkadot_defi_metrics'
    
    id = Column(Integer, primary_key=True)
    parachain_id = Column(Integer, ForeignKey('parachains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # TVL metrics
    total_tvl = Column(Numeric(30, 8), nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    tvl_change_30d = Column(Float, nullable=True)
    tvl_rank = Column(Integer, nullable=True)
    
    # DEX metrics
    dex_volume_24h = Column(Numeric(30, 8), nullable=True)
    dex_trades_24h = Column(Integer, nullable=True)
    dex_liquidity_pools = Column(Integer, nullable=True)
    dex_trading_pairs = Column(Integer, nullable=True)
    dex_apy_avg = Column(Float, nullable=True)
    
    # Lending metrics
    lending_tvl = Column(Numeric(30, 8), nullable=True)
    total_borrowed = Column(Numeric(30, 8), nullable=True)
    lending_apy_avg = Column(Float, nullable=True)
    borrowing_apy_avg = Column(Float, nullable=True)
    liquidation_events = Column(Integer, nullable=True)
    
    # Staking metrics
    liquid_staking_tvl = Column(Numeric(30, 8), nullable=True)
    staking_apy_avg = Column(Float, nullable=True)
    staking_pools = Column(Integer, nullable=True)
    staking_participants = Column(Integer, nullable=True)
    
    # Yield farming
    yield_farming_tvl = Column(Numeric(30, 8), nullable=True)
    active_farms = Column(Integer, nullable=True)
    farm_apy_avg = Column(Float, nullable=True)
    farm_participants = Column(Integer, nullable=True)
    
    # Derivatives
    derivatives_tvl = Column(Numeric(30, 8), nullable=True)
    options_volume = Column(Numeric(30, 8), nullable=True)
    futures_volume = Column(Numeric(30, 8), nullable=True)
    perpetual_volume = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    parachain = relationship("Parachain", back_populates="defi_metrics")

class PolkadotAdvancedAnalytics(Base):
    """Advanced analytics and predictive metrics"""
    __tablename__ = 'polkadot_advanced_analytics'
    
    id = Column(Integer, primary_key=True)
    network_id = Column(Integer, ForeignKey('polkadot_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Predictive analytics
    price_prediction_7d = Column(Numeric(20, 8), nullable=True)
    price_prediction_30d = Column(Numeric(20, 8), nullable=True)
    tvl_prediction_7d = Column(Numeric(30, 8), nullable=True)
    tvl_prediction_30d = Column(Numeric(30, 8), nullable=True)
    
    # Trend analysis
    network_growth_trend = Column(Float, nullable=True)
    adoption_trend = Column(Float, nullable=True)
    innovation_trend = Column(Float, nullable=True)
    competition_trend = Column(Float, nullable=True)
    
    # Risk metrics
    network_risk_score = Column(Float, nullable=True)
    security_risk_score = Column(Float, nullable=True)
    economic_risk_score = Column(Float, nullable=True)
    regulatory_risk_score = Column(Float, nullable=True)
    
    # Performance benchmarks
    performance_vs_ethereum = Column(Float, nullable=True)
    performance_vs_bitcoin = Column(Float, nullable=True)
    performance_vs_competitors = Column(JSON, nullable=True)
    market_share = Column(Float, nullable=True)
    
    # Innovation metrics
    new_features_adoption_rate = Column(Float, nullable=True)
    protocol_upgrade_success_rate = Column(Float, nullable=True)
    developer_innovation_score = Column(Float, nullable=True)
    community_innovation_score = Column(Float, nullable=True)
    
    # Relationships
    network = relationship("PolkadotNetwork")