from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Numeric, Index, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()

class BlockchainType(str, Enum):
    """Типы блокчейнов"""
    MAINNET = "mainnet"
    LAYER2 = "layer2"
    SIDECHAIN = "sidechain"
    TESTNET = "testnet"

class AssetCategory(str, Enum):
    """Категории активов"""
    DEFI = "DeFi"
    LAYER1 = "Layer1"
    LAYER2 = "Layer2"
    STABLECOIN = "Stablecoin"
    WRAPPED = "Wrapped"
    GOVERNANCE = "Governance"
    NFT = "NFT"
    GAMING = "Gaming"
    METAVERSE = "Metaverse"
    INFRASTRUCTURE = "Infrastructure"
    PRIVACY = "Privacy"
    MEME = "Meme"
    AI = "AI"
    RWA = "RWA"  # Real World Assets

class Blockchain(Base):
    """Таблица блокчейнов"""
    __tablename__ = 'blockchains'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # Ethereum, Polygon, BSC, etc.
    symbol = Column(String(20), unique=True, nullable=False)  # ETH, MATIC, BNB, etc.
    chain_id = Column(Integer, unique=True, nullable=False)  # 1, 137, 56, etc.
    blockchain_type = Column(String(20), nullable=False)  # mainnet, layer2, sidechain
    rpc_url = Column(String(500), nullable=True)
    explorer_url = Column(String(200), nullable=True)
    native_token = Column(String(20), nullable=False)  # ETH, MATIC, BNB
    is_active = Column(Boolean, default=True)
    launch_date = Column(DateTime, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    network_metrics = relationship("NetworkMetrics", back_populates="blockchain")

class CryptoAsset(Base):
    """Основная таблица крипто-активов"""
    __tablename__ = 'crypto_assets'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    contract_address = Column(String(42), nullable=True)
    blockchain = Column(String(20), nullable=False)
    category = Column(String(50), nullable=False, default="DeFi")
    github_repo = Column(String(200), nullable=True)
    website = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for symbol + blockchain combination
    __table_args__ = (
        UniqueConstraint('symbol', 'blockchain', name='unique_symbol_blockchain'),
        Index('idx_symbol_blockchain', 'symbol', 'blockchain'),
        Index('idx_contract_address', 'contract_address'),
        Index('idx_category', 'category'),
    )
    
    # Relationships
    onchain_metrics = relationship("OnChainMetrics", back_populates="asset")
    github_metrics = relationship("GitHubMetrics", back_populates="asset")
    financial_metrics = relationship("FinancialMetrics", back_populates="asset")
    tokenomics_metrics = relationship("TokenomicsMetrics", back_populates="asset")
    security_metrics = relationship("SecurityMetrics", back_populates="asset")
    community_metrics = relationship("CommunityMetrics", back_populates="asset")
    partnership_metrics = relationship("PartnershipMetrics", back_populates="asset")
    predictions = relationship("MLPrediction", back_populates="asset")
    smart_contracts = relationship("SmartContract", back_populates="asset")

class NetworkMetrics(Base):
    """Метрики сети блокчейна"""
    __tablename__ = 'network_metrics'
    
    id = Column(Integer, primary_key=True)
    blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Network performance
    block_time_avg = Column(Float, nullable=True)
    block_size_avg = Column(Float, nullable=True)
    transaction_throughput = Column(Float, nullable=True)
    network_utilization = Column(Float, nullable=True)
    
    # Security metrics
    hash_rate = Column(Float, nullable=True)
    difficulty = Column(Float, nullable=True)
    validator_count = Column(Integer, nullable=True)
    staking_ratio = Column(Float, nullable=True)
    
    # Economic metrics
    total_supply = Column(Numeric(30, 8), nullable=True)
    inflation_rate = Column(Float, nullable=True)
    deflation_rate = Column(Float, nullable=True)
    burn_rate = Column(Float, nullable=True)
    
    # Gas metrics
    gas_price_avg = Column(Float, nullable=True)
    gas_price_median = Column(Float, nullable=True)
    gas_limit = Column(Float, nullable=True)
    gas_used_avg = Column(Float, nullable=True)
    
    # Relationships
    blockchain = relationship("Blockchain", back_populates="network_metrics")

class EconomicMetrics(Base):
    """Экономические метрики блокчейна"""
    __tablename__ = 'economic_metrics'
    
    id = Column(Integer, primary_key=True)
    blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Economic indicators
    total_value_locked = Column(Numeric(30, 8), nullable=True)
    daily_volume = Column(Numeric(30, 8), nullable=True)
    active_users_24h = Column(Integer, nullable=True)
    transaction_fees_24h = Column(Numeric(20, 8), nullable=True)
    revenue_24h = Column(Numeric(20, 8), nullable=True)
    market_cap = Column(Numeric(30, 8), nullable=True)
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    total_supply = Column(Numeric(30, 8), nullable=True)
    
    # Price metrics
    price_usd = Column(Numeric(20, 8), nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    
    # Relationships
    blockchain = relationship("Blockchain")

class EcosystemMetrics(Base):
    """Экосистемные метрики блокчейна"""
    __tablename__ = 'ecosystem_metrics'
    
    id = Column(Integer, primary_key=True)
    blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Ecosystem indicators
    defi_protocols_count = Column(Integer, nullable=True)
    nft_marketplaces = Column(Integer, nullable=True)
    bridges_count = Column(Integer, nullable=True)
    dapp_count = Column(Integer, nullable=True)
    
    # Community metrics
    developer_count = Column(Integer, nullable=True)
    github_repos = Column(Integer, nullable=True)
    social_media_followers = Column(Integer, nullable=True)
    
    # Partnership metrics
    partnership_count = Column(Integer, nullable=True)
    integration_count = Column(Integer, nullable=True)
    
    # Relationships
    blockchain = relationship("Blockchain")

class OnChainMetrics(Base):
    """On-chain метрики для активов"""
    __tablename__ = 'onchain_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # TVL и ликвидность
    tvl = Column(Numeric(30, 8), nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    tvl_change_30d = Column(Float, nullable=True)
    tvl_rank = Column(Integer, nullable=True)
    
    # Транзакционная активность
    daily_transactions = Column(Integer, nullable=True)
    transaction_volume_24h = Column(Numeric(30, 8), nullable=True)
    transaction_volume_7d = Column(Numeric(30, 8), nullable=True)
    avg_transaction_fee = Column(Float, nullable=True)
    transaction_success_rate = Column(Float, nullable=True)
    gas_usage_efficiency = Column(Float, nullable=True)
    
    # Активность пользователей
    active_addresses_24h = Column(Integer, nullable=True)
    new_addresses_24h = Column(Integer, nullable=True)
    unique_users_7d = Column(Integer, nullable=True)
    user_retention_rate = Column(Float, nullable=True)
    whale_activity = Column(Float, nullable=True)
    
    # Смарт-контракты
    new_contracts_deployed = Column(Integer, nullable=True)
    contract_interactions_24h = Column(Integer, nullable=True)
    contract_complexity_score = Column(Float, nullable=True)
    
    # DeFi специфичные метрики
    liquidity_pools_count = Column(Integer, nullable=True)
    liquidity_pools_tvl = Column(Numeric(30, 8), nullable=True)
    yield_farming_apy = Column(Float, nullable=True)
    lending_volume = Column(Numeric(30, 8), nullable=True)
    borrowing_volume = Column(Numeric(30, 8), nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="onchain_metrics")

class FinancialMetrics(Base):
    """Финансовые метрики"""
    __tablename__ = 'financial_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Цена и капитализация
    price_usd = Column(Numeric(20, 8), nullable=True)
    market_cap = Column(Numeric(30, 8), nullable=True)
    fully_diluted_valuation = Column(Numeric(30, 8), nullable=True)
    market_cap_rank = Column(Integer, nullable=True)
    market_cap_dominance = Column(Float, nullable=True)
    
    # Объемы торгов
    volume_24h = Column(Numeric(30, 8), nullable=True)
    volume_7d = Column(Numeric(30, 8), nullable=True)
    volume_change_24h = Column(Float, nullable=True)
    volume_market_cap_ratio = Column(Float, nullable=True)
    liquidity_score = Column(Float, nullable=True)
    
    # Волатильность
    volatility_24h = Column(Float, nullable=True)
    volatility_7d = Column(Float, nullable=True)
    volatility_30d = Column(Float, nullable=True)
    beta_coefficient = Column(Float, nullable=True)
    
    # Ликвидность
    bid_ask_spread = Column(Float, nullable=True)
    order_book_depth = Column(Float, nullable=True)
    slippage_analysis = Column(Float, nullable=True)
    
    # Price changes
    price_change_1h = Column(Float, nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    price_change_90d = Column(Float, nullable=True)
    price_change_1y = Column(Float, nullable=True)
    
    # Historical data
    all_time_high = Column(Numeric(20, 8), nullable=True)
    all_time_low = Column(Numeric(20, 8), nullable=True)
    ath_date = Column(DateTime, nullable=True)
    atl_date = Column(DateTime, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="financial_metrics")

class TokenomicsMetrics(Base):
    """Токеномика"""
    __tablename__ = 'tokenomics_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Предложение токенов
    circulating_supply = Column(Numeric(30, 8), nullable=True)
    total_supply = Column(Numeric(30, 8), nullable=True)
    max_supply = Column(Numeric(30, 8), nullable=True)
    inflation_rate = Column(Float, nullable=True)
    burn_rate = Column(Float, nullable=True)
    
    # Распределение токенов
    team_allocation = Column(Float, nullable=True)
    investor_allocation = Column(Float, nullable=True)
    community_allocation = Column(Float, nullable=True)
    treasury_allocation = Column(Float, nullable=True)
    public_sale_allocation = Column(Float, nullable=True)
    
    # Vesting schedule
    vesting_schedule = Column(JSON, nullable=True)
    unlocked_percentage = Column(Float, nullable=True)
    next_unlock_date = Column(DateTime, nullable=True)
    next_unlock_amount = Column(Numeric(30, 8), nullable=True)
    
    # Утилитарность токена
    utility_score = Column(Float, nullable=True)
    governance_power = Column(Float, nullable=True)
    staking_rewards = Column(Float, nullable=True)
    fee_burn_mechanism = Column(Boolean, default=False)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="tokenomics_metrics")

class GitHubMetrics(Base):
    """GitHub активность"""
    __tablename__ = 'github_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Коммиты
    commits_24h = Column(Integer, nullable=True)
    commits_7d = Column(Integer, nullable=True)
    commits_30d = Column(Integer, nullable=True)
    commits_90d = Column(Integer, nullable=True)
    code_quality_score = Column(Float, nullable=True)
    test_coverage = Column(Float, nullable=True)
    
    # Pull Requests
    open_prs = Column(Integer, nullable=True)
    merged_prs_7d = Column(Integer, nullable=True)
    closed_prs_7d = Column(Integer, nullable=True)
    pr_merge_rate = Column(Float, nullable=True)
    avg_pr_lifetime = Column(Float, nullable=True)
    
    # Issues
    open_issues = Column(Integer, nullable=True)
    closed_issues_7d = Column(Integer, nullable=True)
    issue_resolution_time = Column(Float, nullable=True)
    bug_report_ratio = Column(Float, nullable=True)
    
    # Участники
    active_contributors_30d = Column(Integer, nullable=True)
    total_contributors = Column(Integer, nullable=True)
    external_contributors = Column(Integer, nullable=True)
    core_team_activity = Column(Float, nullable=True)
    
    # Популярность
    stars = Column(Integer, nullable=True)
    forks = Column(Integer, nullable=True)
    stars_change_7d = Column(Integer, nullable=True)
    watch_count = Column(Integer, nullable=True)
    primary_language = Column(String(50), nullable=True)
    languages_distribution = Column(JSON, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="github_metrics")

class SecurityMetrics(Base):
    """Метрики безопасности"""
    __tablename__ = 'security_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Аудит и верификация
    audit_status = Column(String(50), nullable=True)  # audited, unaudited, pending
    audit_firm = Column(String(100), nullable=True)
    audit_date = Column(DateTime, nullable=True)
    audit_score = Column(Float, nullable=True)
    contract_verified = Column(Boolean, default=False)
    source_code_available = Column(Boolean, default=False)
    
    # Безопасность контрактов
    vulnerability_score = Column(Float, nullable=True)
    multisig_wallets = Column(Boolean, default=False)
    timelock_mechanisms = Column(Boolean, default=False)
    upgrade_mechanisms = Column(String(50), nullable=True)
    emergency_pause = Column(Boolean, default=False)
    
    # Децентрализация
    governance_decentralization = Column(Float, nullable=True)
    validator_distribution = Column(Float, nullable=True)
    node_distribution = Column(Float, nullable=True)
    treasury_control = Column(Float, nullable=True)
    
    # Smart contract security
    reentrancy_protection = Column(Boolean, default=False)
    overflow_protection = Column(Boolean, default=False)
    access_control = Column(Boolean, default=False)
    pause_functionality = Column(Boolean, default=False)
    
    # Composite security score (calculated from individual metrics)
    security_score = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="security_metrics")

class CommunityMetrics(Base):
    """Метрики сообщества"""
    __tablename__ = 'community_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Социальные сети
    twitter_followers = Column(Integer, nullable=True)
    telegram_members = Column(Integer, nullable=True)
    discord_members = Column(Integer, nullable=True)
    reddit_subscribers = Column(Integer, nullable=True)
    facebook_likes = Column(Integer, nullable=True)
    instagram_followers = Column(Integer, nullable=True)
    youtube_subscribers = Column(Integer, nullable=True)
    tiktok_followers = Column(Integer, nullable=True)
    
    # Engagement metrics
    social_engagement_rate = Column(Float, nullable=True)
    twitter_engagement_rate = Column(Float, nullable=True)
    telegram_activity_score = Column(Float, nullable=True)
    discord_activity_score = Column(Float, nullable=True)
    
    # Content metrics
    blog_posts_30d = Column(Integer, nullable=True)
    youtube_videos_30d = Column(Integer, nullable=True)
    podcast_appearances_30d = Column(Integer, nullable=True)
    media_mentions_30d = Column(Integer, nullable=True)
    brand_awareness_score = Column(Float, nullable=True)
    
    # Educational resources
    documentation_quality = Column(Float, nullable=True)
    tutorial_availability = Column(Float, nullable=True)
    community_guides_count = Column(Integer, nullable=True)
    support_responsiveness = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="community_metrics")

class PartnershipMetrics(Base):
    """Метрики партнерств и интеграций"""
    __tablename__ = 'partnership_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Партнерства
    partnership_count = Column(Integer, nullable=True)
    tier1_partnerships = Column(Integer, nullable=True)
    strategic_partnerships = Column(Integer, nullable=True)
    partnership_quality_score = Column(Float, nullable=True)
    
    # Интеграции
    integration_count = Column(Integer, nullable=True)
    exchange_listings = Column(Integer, nullable=True)
    wallet_support = Column(Integer, nullable=True)
    defi_protocol_integrations = Column(Integer, nullable=True)
    cefi_integrations = Column(Integer, nullable=True)
    nft_marketplace_support = Column(Integer, nullable=True)
    cross_chain_bridges = Column(Integer, nullable=True)
    
    # Ecosystem metrics
    ecosystem_connections = Column(Integer, nullable=True)
    interoperability_score = Column(Float, nullable=True)
    adoption_rate = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="partnership_metrics")

class CompetitorAnalysis(Base):
    """Анализ конкурентов"""
    __tablename__ = 'competitor_analysis'
    
    id = Column(Integer, primary_key=True)
    competitor_name = Column(String(100), nullable=False)
    competitor_type = Column(String(50), nullable=False)  # analytics, aggregator, etc.
    website = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    
    # Метрики конкурента
    monthly_visitors = Column(Integer, nullable=True)
    alexa_rank = Column(Integer, nullable=True)
    social_media_followers = Column(JSON, nullable=True)
    
    # Функциональность
    features = Column(JSON, nullable=True)
    supported_chains = Column(JSON, nullable=True)
    pricing_model = Column(String(50), nullable=True)
    
    # Оценка
    market_share = Column(Float, nullable=True)
    user_rating = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class MLPrediction(Base):
    """ML предсказания"""
    __tablename__ = 'ml_predictions'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    prediction_type = Column(String(50), nullable=False)  # investment_score, price_prediction, etc.
    
    # Предсказание
    prediction_value = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    prediction_horizon = Column(String(20), nullable=False)  # 1d, 7d, 30d, etc.
    
    # Метаданные
    features_used = Column(JSON, nullable=True)
    model_version = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="predictions")

class SmartContract(Base):
    """Смарт-контракты для анализа"""
    __tablename__ = 'smart_contracts'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=True)
    contract_address = Column(String(42), nullable=False, unique=True)
    blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    contract_name = Column(String(100), nullable=True)
    contract_type = Column(String(50), nullable=True)  # ERC20, ERC721, DeFi, etc.
    
    # Contract details
    bytecode = Column(Text, nullable=True)
    abi = Column(JSON, nullable=True)
    verified = Column(Boolean, default=False)
    
    # Deployment info
    deployer_address = Column(String(42), nullable=True)
    deployment_tx = Column(String(66), nullable=True)
    deployment_block = Column(Integer, nullable=True)
    deployment_timestamp = Column(DateTime, nullable=True)
    
    # Security
    audit_status = Column(String(50), nullable=True)  # audited, unaudited, pending
    audit_firm = Column(String(100), nullable=True)
    audit_date = Column(DateTime, nullable=True)
    security_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="smart_contracts")
    blockchain = relationship("Blockchain")

class DataSource(Base):
    """Источники данных"""
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    source_type = Column(String(50), nullable=False)  # api, scraper, manual
    endpoint = Column(String(200), nullable=True)
    api_key_required = Column(Boolean, default=False)
    rate_limit = Column(Integer, nullable=True)  # requests per minute
    last_successful_fetch = Column(DateTime, nullable=True)
    status = Column(String(20), default='active')  # active, inactive, error
    created_at = Column(DateTime, default=datetime.utcnow)

class TrendingMetrics(Base):
    """Трендовые метрики"""
    __tablename__ = 'trending_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
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
    
    # Relationships
    asset = relationship("CryptoAsset")

class CrossChainMetrics(Base):
    """Кроссчейн метрики"""
    __tablename__ = 'cross_chain_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    source_blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    target_blockchain_id = Column(Integer, ForeignKey('blockchains.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Bridge metrics
    bridge_volume_24h = Column(Numeric(30, 8), nullable=True)
    bridge_transactions_24h = Column(Integer, nullable=True)
    bridge_fees_24h = Column(Numeric(20, 8), nullable=True)
    
    # Liquidity metrics
    cross_chain_liquidity = Column(Numeric(30, 8), nullable=True)
    liquidity_imbalance = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset")
    source_blockchain = relationship("Blockchain", foreign_keys=[source_blockchain_id])
    target_blockchain = relationship("Blockchain", foreign_keys=[target_blockchain_id])
