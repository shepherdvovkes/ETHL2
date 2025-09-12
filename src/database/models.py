from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class CryptoAsset(Base):
    """Основная таблица крипто-активов"""
    __tablename__ = 'crypto_assets'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    contract_address = Column(String(42), nullable=True)
    blockchain = Column(String(20), nullable=False, default="polygon")
    category = Column(String(50), nullable=False, default="DeFi")
    github_repo = Column(String(200), nullable=True)
    website = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    onchain_metrics = relationship("OnChainMetrics", back_populates="asset")
    github_metrics = relationship("GitHubMetrics", back_populates="asset")
    financial_metrics = relationship("FinancialMetrics", back_populates="asset")
    predictions = relationship("MLPrediction", back_populates="asset")

class OnChainMetrics(Base):
    """On-chain метрики для Polygon"""
    __tablename__ = 'onchain_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # TVL и ликвидность
    tvl = Column(Float, nullable=True)
    tvl_change_24h = Column(Float, nullable=True)
    tvl_change_7d = Column(Float, nullable=True)
    
    # Транзакции
    daily_transactions = Column(Integer, nullable=True)
    transaction_volume_24h = Column(Float, nullable=True)
    avg_transaction_fee = Column(Float, nullable=True)
    
    # Активность пользователей
    active_addresses_24h = Column(Integer, nullable=True)
    new_addresses_24h = Column(Integer, nullable=True)
    unique_users_7d = Column(Integer, nullable=True)
    
    # Смарт-контракты
    new_contracts_deployed = Column(Integer, nullable=True)
    contract_interactions_24h = Column(Integer, nullable=True)
    
    # Сеть Polygon
    block_time_avg = Column(Float, nullable=True)
    gas_price_avg = Column(Float, nullable=True)
    network_utilization = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="onchain_metrics")

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
    
    # Pull Requests
    open_prs = Column(Integer, nullable=True)
    merged_prs_7d = Column(Integer, nullable=True)
    closed_prs_7d = Column(Integer, nullable=True)
    
    # Issues
    open_issues = Column(Integer, nullable=True)
    closed_issues_7d = Column(Integer, nullable=True)
    
    # Участники
    active_contributors_30d = Column(Integer, nullable=True)
    total_contributors = Column(Integer, nullable=True)
    
    # Звезды и форки
    stars = Column(Integer, nullable=True)
    forks = Column(Integer, nullable=True)
    stars_change_7d = Column(Integer, nullable=True)
    
    # Languages
    primary_language = Column(String(50), nullable=True)
    languages_distribution = Column(JSON, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="github_metrics")

class FinancialMetrics(Base):
    """Финансовые метрики"""
    __tablename__ = 'financial_metrics'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Цена и капитализация
    price_usd = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    fully_diluted_valuation = Column(Float, nullable=True)
    
    # Объемы торгов
    volume_24h = Column(Float, nullable=True)
    volume_7d = Column(Float, nullable=True)
    volume_change_24h = Column(Float, nullable=True)
    
    # Волатильность
    volatility_24h = Column(Float, nullable=True)
    volatility_7d = Column(Float, nullable=True)
    volatility_30d = Column(Float, nullable=True)
    
    # Ликвидность
    bid_ask_spread = Column(Float, nullable=True)
    order_book_depth = Column(Float, nullable=True)
    
    # Циркуляция
    circulating_supply = Column(Float, nullable=True)
    total_supply = Column(Float, nullable=True)
    max_supply = Column(Float, nullable=True)
    
    # Price changes
    price_change_1h = Column(Float, nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    price_change_30d = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("CryptoAsset", back_populates="financial_metrics")

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

class SmartContract(Base):
    """Смарт-контракты для анализа"""
    __tablename__ = 'smart_contracts'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'), nullable=True)
    contract_address = Column(String(42), nullable=False, unique=True)
    blockchain = Column(String(20), nullable=False, default="polygon")
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
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    asset = relationship("CryptoAsset")
