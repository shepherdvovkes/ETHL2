#!/usr/bin/env python3
"""
Модели данных для Layer 2 сетей Ethereum
Интеграция с существующей схемой базы данных
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Numeric, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

# Import Base from existing models
from .models_v2 import Base

class L2Type(str, Enum):
    """Типы Layer 2 решений"""
    OPTIMISTIC_ROLLUP = "Optimistic Rollup"
    ZK_ROLLUP = "ZK Rollup"
    VALIDIUM = "Validium"
    PLASMA = "Plasma"
    SIDECHAIN = "Sidechain"
    STATE_CHANNEL = "State Channel"
    HYBRID = "Hybrid"

class SecurityModel(str, Enum):
    """Модели безопасности"""
    ETHEREUM_SECURITY = "Ethereum Security"
    OWN_VALIDATORS = "Own Validators"
    HYBRID = "Hybrid"

class L2Network(Base):
    """Основная таблица Layer 2 сетей"""
    __tablename__ = 'l2_networks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # Arbitrum One, Optimism, etc.
    symbol = Column(String(20), nullable=True)  # ARB, OP, etc.
    l2_type = Column(String(50), nullable=False)  # Optimistic Rollup, ZK Rollup, etc.
    security_model = Column(String(50), nullable=False)  # Ethereum Security, Own Validators
    
    # Основная информация
    launch_date = Column(DateTime, nullable=True)
    website = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), default='Active')  # Active, Inactive, Testnet
    
    # Технические характеристики
    consensus_mechanism = Column(String(100), nullable=True)
    block_time = Column(String(50), nullable=True)
    gas_limit = Column(Integer, nullable=True)
    evm_compatibility = Column(Boolean, default=True)
    programming_language = Column(String(100), nullable=True)
    virtual_machine = Column(String(100), nullable=True)
    data_availability = Column(String(100), nullable=True)
    fraud_proofs = Column(Boolean, default=False)
    zero_knowledge_proofs = Column(Boolean, default=False)
    
    # Связь с блокчейном
    parent_blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent_blockchain = relationship("Blockchain")
    performance_metrics = relationship("L2PerformanceMetrics", back_populates="l2_network")
    economic_metrics = relationship("L2EconomicMetrics", back_populates="l2_network")
    security_metrics = relationship("L2SecurityMetrics", back_populates="l2_network")
    ecosystem_metrics = relationship("L2EcosystemMetrics", back_populates="l2_network")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_networks_type', 'l2_type'),
        Index('idx_l2_networks_security', 'security_model'),
        Index('idx_l2_networks_status', 'status'),
    )

class L2PerformanceMetrics(Base):
    """Метрики производительности L2 сетей"""
    __tablename__ = 'l2_performance_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Производительность
    transactions_per_second = Column(Integer, nullable=True)
    finality_time = Column(String(50), nullable=True)  # ~1 minute, ~10 minutes
    withdrawal_time = Column(String(50), nullable=True)  # 7 days, ~10 minutes
    gas_fee_reduction = Column(Float, nullable=True)  # Процент снижения комиссий
    throughput_improvement = Column(Float, nullable=True)  # Улучшение пропускной способности
    latency = Column(String(50), nullable=True)
    
    # Сетевые метрики
    block_time_avg = Column(Float, nullable=True)
    block_size_avg = Column(Float, nullable=True)
    network_utilization = Column(Float, nullable=True)
    gas_price_avg = Column(Float, nullable=True)
    gas_used_avg = Column(Float, nullable=True)
    
    # Активность
    daily_transactions = Column(Integer, nullable=True)
    active_addresses_24h = Column(Integer, nullable=True)
    new_addresses_24h = Column(Integer, nullable=True)
    unique_users_7d = Column(Integer, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_performance_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2EconomicMetrics(Base):
    """Экономические метрики L2 сетей"""
    __tablename__ = 'l2_economic_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # TVL и капитализация
    total_value_locked = Column(Numeric(30, 8), nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    tvl_change_30d = Column(Float, nullable=True)
    tvl_rank = Column(Integer, nullable=True)
    
    # Объемы и активность
    daily_volume = Column(Numeric(30, 8), nullable=True)
    volume_change_24h = Column(Float, nullable=True)
    volume_change_7d = Column(Float, nullable=True)
    active_users_24h = Column(Integer, nullable=True)
    
    # Комиссии и доходы
    transaction_fees_24h = Column(Numeric(20, 8), nullable=True)
    revenue_24h = Column(Numeric(20, 8), nullable=True)
    avg_transaction_fee = Column(Float, nullable=True)
    fee_revenue_7d = Column(Numeric(20, 8), nullable=True)
    
    # Токеномика
    native_token_price = Column(Numeric(20, 8), nullable=True)
    market_cap = Column(Numeric(30, 8), nullable=True)
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    total_supply = Column(Numeric(30, 8), nullable=True)
    inflation_rate = Column(Float, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network", back_populates="economic_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_economic_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2SecurityMetrics(Base):
    """Метрики безопасности L2 сетей"""
    __tablename__ = 'l2_security_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Валидаторы и консенсус
    validator_count = Column(Integer, nullable=True)
    slashing_mechanism = Column(Boolean, default=False)
    multisig_required = Column(Boolean, default=False)
    upgrade_mechanism = Column(String(100), nullable=True)
    time_to_finality = Column(String(50), nullable=True)
    
    # Аудит и безопасность
    audit_count = Column(Integer, nullable=True)
    audit_firms = Column(JSON, nullable=True)  # Список аудиторских фирм
    bug_bounty_program = Column(Boolean, default=False)
    security_score = Column(Float, nullable=True)
    vulnerability_count = Column(Integer, nullable=True)
    
    # Децентрализация
    decentralization_score = Column(Float, nullable=True)
    sequencer_decentralization = Column(Float, nullable=True)
    governance_decentralization = Column(Float, nullable=True)
    treasury_control = Column(Float, nullable=True)
    
    # Безопасность контрактов
    contract_verified = Column(Boolean, default=False)
    source_code_available = Column(Boolean, default=False)
    reentrancy_protection = Column(Boolean, default=False)
    overflow_protection = Column(Boolean, default=False)
    access_control = Column(Boolean, default=False)
    emergency_pause = Column(Boolean, default=False)
    
    # Relationships
    l2_network = relationship("L2Network", back_populates="security_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_security_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2EcosystemMetrics(Base):
    """Метрики экосистемы L2 сетей"""
    __tablename__ = 'l2_ecosystem_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # DeFi протоколы
    defi_protocols_count = Column(Integer, nullable=True)
    defi_tvl = Column(Numeric(30, 8), nullable=True)
    defi_tvl_change_24h = Column(Float, nullable=True)
    lending_protocols = Column(Integer, nullable=True)
    dex_protocols = Column(Integer, nullable=True)
    yield_farming_protocols = Column(Integer, nullable=True)
    
    # NFT и игры
    nft_marketplaces = Column(Integer, nullable=True)
    nft_volume_24h = Column(Numeric(30, 8), nullable=True)
    gaming_protocols = Column(Integer, nullable=True)
    gaming_volume_24h = Column(Numeric(30, 8), nullable=True)
    
    # Инфраструктура
    bridges_count = Column(Integer, nullable=True)
    bridge_volume_24h = Column(Numeric(30, 8), nullable=True)
    wallets_support = Column(Integer, nullable=True)
    rpc_providers = Column(Integer, nullable=True)
    indexers = Column(Integer, nullable=True)
    
    # Разработка
    new_contracts_deployed_24h = Column(Integer, nullable=True)
    contract_interactions_24h = Column(Integer, nullable=True)
    developer_activity_score = Column(Float, nullable=True)
    github_repos = Column(Integer, nullable=True)
    
    # Сообщество
    social_media_followers = Column(JSON, nullable=True)  # Twitter, Discord, etc.
    community_activity_score = Column(Float, nullable=True)
    documentation_quality = Column(Float, nullable=True)
    tutorial_availability = Column(Float, nullable=True)
    
    # Партнерства
    partnership_count = Column(Integer, nullable=True)
    tier1_partnerships = Column(Integer, nullable=True)
    strategic_partnerships = Column(Integer, nullable=True)
    integration_count = Column(Integer, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network", back_populates="ecosystem_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_ecosystem_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2ComparisonMetrics(Base):
    """Сравнительные метрики L2 сетей"""
    __tablename__ = 'l2_comparison_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Рейтинги
    overall_rank = Column(Integer, nullable=True)
    tps_rank = Column(Integer, nullable=True)
    tvl_rank = Column(Integer, nullable=True)
    security_rank = Column(Integer, nullable=True)
    adoption_rank = Column(Integer, nullable=True)
    innovation_rank = Column(Integer, nullable=True)
    
    # Сравнительные показатели
    tps_vs_ethereum = Column(Float, nullable=True)  # Во сколько раз быстрее Ethereum
    fee_reduction_vs_ethereum = Column(Float, nullable=True)  # Процент снижения комиссий
    finality_vs_ethereum = Column(Float, nullable=True)  # Во сколько раз быстрее финализация
    
    # Рыночная доля
    market_share_tvl = Column(Float, nullable=True)  # Доля в общем TVL L2
    market_share_volume = Column(Float, nullable=True)  # Доля в общем объеме L2
    market_share_transactions = Column(Float, nullable=True)  # Доля в общих транзакциях L2
    
    # Тренды
    growth_rate_30d = Column(Float, nullable=True)
    adoption_rate_30d = Column(Float, nullable=True)
    innovation_score = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_comparison_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2RiskAssessment(Base):
    """Оценка рисков L2 сетей"""
    __tablename__ = 'l2_risk_assessment'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Категории рисков
    centralization_risk = Column(Float, nullable=True)  # 0-10
    security_risk = Column(Float, nullable=True)  # 0-10
    liquidity_risk = Column(Float, nullable=True)  # 0-10
    technical_risk = Column(Float, nullable=True)  # 0-10
    regulatory_risk = Column(Float, nullable=True)  # 0-10
    adoption_risk = Column(Float, nullable=True)  # 0-10
    
    # Общий риск
    overall_risk_score = Column(Float, nullable=True)  # 0-10
    risk_level = Column(String(20), nullable=True)  # Low, Medium, High, Critical
    
    # Детализация рисков
    risk_factors = Column(JSON, nullable=True)  # Список конкретных рисков
    mitigation_measures = Column(JSON, nullable=True)  # Меры по снижению рисков
    
    # Временные риски
    short_term_risks = Column(JSON, nullable=True)  # Риски на 1-3 месяца
    medium_term_risks = Column(JSON, nullable=True)  # Риски на 3-12 месяцев
    long_term_risks = Column(JSON, nullable=True)  # Риски на 1+ год
    
    # Relationships
    l2_network = relationship("L2Network")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_risk_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2TrendingMetrics(Base):
    """Трендовые метрики L2 сетей"""
    __tablename__ = 'l2_trending_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Momentum indicators
    momentum_score = Column(Float, nullable=True)
    trend_direction = Column(String(20), nullable=True)  # bullish, bearish, sideways
    trend_strength = Column(Float, nullable=True)
    
    # Seasonal patterns
    seasonality_score = Column(Float, nullable=True)
    cyclical_patterns = Column(JSON, nullable=True)
    
    # Anomaly detection
    anomaly_score = Column(Float, nullable=True)
    anomaly_type = Column(String(50), nullable=True)
    anomaly_severity = Column(String(20), nullable=True)  # low, medium, high
    
    # Market sentiment
    fear_greed_index = Column(Float, nullable=True)
    social_sentiment = Column(Float, nullable=True)
    news_sentiment = Column(Float, nullable=True)
    developer_sentiment = Column(Float, nullable=True)
    
    # Growth indicators
    user_growth_rate = Column(Float, nullable=True)
    volume_growth_rate = Column(Float, nullable=True)
    tvl_growth_rate = Column(Float, nullable=True)
    adoption_growth_rate = Column(Float, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_trending_network_timestamp', 'l2_network_id', 'timestamp'),
    )

class L2CrossChainMetrics(Base):
    """Кроссчейн метрики L2 сетей"""
    __tablename__ = 'l2_cross_chain_metrics'
    
    id = Column(Integer, primary_key=True)
    l2_network_id = Column(Integer, ForeignKey('l2_networks.id'), nullable=False)
    target_blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Bridge metrics
    bridge_volume_24h = Column(Numeric(30, 8), nullable=True)
    bridge_transactions_24h = Column(Integer, nullable=True)
    bridge_fees_24h = Column(Numeric(20, 8), nullable=True)
    bridge_volume_7d = Column(Numeric(30, 8), nullable=True)
    
    # Liquidity metrics
    cross_chain_liquidity = Column(Numeric(30, 8), nullable=True)
    liquidity_imbalance = Column(Float, nullable=True)
    bridge_efficiency = Column(Float, nullable=True)
    
    # User metrics
    cross_chain_users_24h = Column(Integer, nullable=True)
    cross_chain_retention_rate = Column(Float, nullable=True)
    
    # Relationships
    l2_network = relationship("L2Network")
    target_blockchain = relationship("Blockchain")
    
    # Indexes
    __table_args__ = (
        Index('idx_l2_cross_chain_network_timestamp', 'l2_network_id', 'timestamp'),
        Index('idx_l2_cross_chain_target_timestamp', 'target_blockchain_id', 'timestamp'),
    )
