# 🗄️ DEFIMON Database Schema v2

## Обзор

DEFIMON v2 поддерживает **50+ блокчейнов** и **10 категорий метрик** для комплексного анализа крипто-активов.

## 🏗️ Архитектура базы данных

### Основные таблицы

#### 1. `blockchains` - Блокчейны
```sql
CREATE TABLE blockchains (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,           -- Ethereum, Polygon, BSC
    symbol VARCHAR(20) UNIQUE NOT NULL,          -- ETH, MATIC, BNB
    chain_id INTEGER UNIQUE NOT NULL,            -- 1, 137, 56
    blockchain_type VARCHAR(20) NOT NULL,        -- mainnet, layer2, sidechain
    rpc_url VARCHAR(500),                        -- RPC endpoint
    explorer_url VARCHAR(200),                   -- Block explorer
    native_token VARCHAR(20) NOT NULL,           -- Native token symbol
    is_active BOOLEAN DEFAULT TRUE,
    launch_date TIMESTAMP,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### 2. `crypto_assets` - Крипто-активы
```sql
CREATE TABLE crypto_assets (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                 -- Token symbol
    name VARCHAR(100) NOT NULL,                  -- Token name
    contract_address VARCHAR(42),                -- Contract address
    blockchain_id INTEGER REFERENCES blockchains(id),
    category VARCHAR(50) DEFAULT 'DeFi',         -- DeFi, Layer1, Stablecoin, etc.
    github_repo VARCHAR(200),                    -- GitHub repository
    website VARCHAR(200),                        -- Official website
    description TEXT,                            -- Project description
    logo_url VARCHAR(500),                       -- Logo URL
    coingecko_id VARCHAR(100),                   -- CoinGecko ID
    coinmarketcap_id VARCHAR(100),               -- CoinMarketCap ID
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, blockchain_id)
);
```

### 📊 Метрики

#### 3. `network_metrics` - Метрики сети
```sql
CREATE TABLE network_metrics (
    id SERIAL PRIMARY KEY,
    blockchain_id INTEGER REFERENCES blockchains(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Network performance
    block_time_avg FLOAT,                        -- Average block time
    block_size_avg FLOAT,                        -- Average block size
    transaction_throughput FLOAT,                -- TPS
    network_utilization FLOAT,                   -- Network utilization %
    
    -- Security metrics
    hash_rate FLOAT,                             -- Network hash rate
    difficulty FLOAT,                            -- Mining difficulty
    validator_count INTEGER,                     -- Number of validators
    staking_ratio FLOAT,                         -- Staking ratio %
    
    -- Economic metrics
    total_supply NUMERIC(30,8),                  -- Total supply
    inflation_rate FLOAT,                        -- Inflation rate %
    deflation_rate FLOAT,                        -- Deflation rate %
    burn_rate FLOAT,                             -- Token burn rate %
    
    -- Gas metrics
    gas_price_avg FLOAT,                         -- Average gas price
    gas_price_median FLOAT,                      -- Median gas price
    gas_limit FLOAT,                             -- Gas limit
    gas_used_avg FLOAT                           -- Average gas used
);
```

#### 4. `onchain_metrics` - On-chain метрики
```sql
CREATE TABLE onchain_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- TVL и ликвидность
    tvl NUMERIC(30,8),                           -- Total Value Locked
    tvl_change_24h FLOAT,                        -- TVL change 24h %
    tvl_change_7d FLOAT,                         -- TVL change 7d %
    tvl_change_30d FLOAT,                        -- TVL change 30d %
    tvl_rank INTEGER,                            -- TVL ranking
    
    -- Транзакционная активность
    daily_transactions INTEGER,                  -- Daily transactions
    transaction_volume_24h NUMERIC(30,8),        -- Transaction volume 24h
    transaction_volume_7d NUMERIC(30,8),         -- Transaction volume 7d
    avg_transaction_fee FLOAT,                   -- Average transaction fee
    transaction_success_rate FLOAT,              -- Transaction success rate %
    gas_usage_efficiency FLOAT,                  -- Gas usage efficiency
    
    -- Активность пользователей
    active_addresses_24h INTEGER,                -- Active addresses 24h
    new_addresses_24h INTEGER,                   -- New addresses 24h
    unique_users_7d INTEGER,                     -- Unique users 7d
    user_retention_rate FLOAT,                   -- User retention rate %
    whale_activity FLOAT,                        -- Whale activity score
    
    -- Смарт-контракты
    new_contracts_deployed INTEGER,              -- New contracts deployed
    contract_interactions_24h INTEGER,           -- Contract interactions 24h
    contract_complexity_score FLOAT,             -- Contract complexity score
    
    -- DeFi специфичные метрики
    liquidity_pools_count INTEGER,               -- Number of liquidity pools
    liquidity_pools_tvl NUMERIC(30,8),           -- Liquidity pools TVL
    yield_farming_apy FLOAT,                     -- Yield farming APY %
    lending_volume NUMERIC(30,8),                -- Lending volume
    borrowing_volume NUMERIC(30,8)               -- Borrowing volume
);
```

#### 5. `financial_metrics` - Финансовые метрики
```sql
CREATE TABLE financial_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Цена и капитализация
    price_usd NUMERIC(20,8),                     -- Price in USD
    market_cap NUMERIC(30,8),                    -- Market capitalization
    fully_diluted_valuation NUMERIC(30,8),       -- Fully diluted valuation
    market_cap_rank INTEGER,                     -- Market cap ranking
    market_cap_dominance FLOAT,                  -- Market cap dominance %
    
    -- Объемы торгов
    volume_24h NUMERIC(30,8),                    -- Volume 24h
    volume_7d NUMERIC(30,8),                     -- Volume 7d
    volume_change_24h FLOAT,                     -- Volume change 24h %
    volume_market_cap_ratio FLOAT,               -- Volume/Market cap ratio
    liquidity_score FLOAT,                       -- Liquidity score
    
    -- Волатильность
    volatility_24h FLOAT,                        -- Volatility 24h
    volatility_7d FLOAT,                         -- Volatility 7d
    volatility_30d FLOAT,                        -- Volatility 30d
    beta_coefficient FLOAT,                      -- Beta coefficient
    
    -- Ликвидность
    bid_ask_spread FLOAT,                        -- Bid-ask spread
    order_book_depth FLOAT,                      -- Order book depth
    slippage_analysis FLOAT,                     -- Slippage analysis
    
    -- Price changes
    price_change_1h FLOAT,                       -- Price change 1h %
    price_change_24h FLOAT,                      -- Price change 24h %
    price_change_7d FLOAT,                       -- Price change 7d %
    price_change_30d FLOAT,                      -- Price change 30d %
    price_change_90d FLOAT,                      -- Price change 90d %
    price_change_1y FLOAT,                       -- Price change 1y %
    
    -- Historical data
    all_time_high NUMERIC(20,8),                 -- All-time high
    all_time_low NUMERIC(20,8),                  -- All-time low
    ath_date TIMESTAMP,                          -- ATH date
    atl_date TIMESTAMP                           -- ATL date
);
```

#### 6. `tokenomics_metrics` - Токеномика
```sql
CREATE TABLE tokenomics_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Предложение токенов
    circulating_supply NUMERIC(30,8),            -- Circulating supply
    total_supply NUMERIC(30,8),                  -- Total supply
    max_supply NUMERIC(30,8),                    -- Max supply
    inflation_rate FLOAT,                        -- Inflation rate %
    burn_rate FLOAT,                             -- Burn rate %
    
    -- Распределение токенов
    team_allocation FLOAT,                       -- Team allocation %
    investor_allocation FLOAT,                   -- Investor allocation %
    community_allocation FLOAT,                  -- Community allocation %
    treasury_allocation FLOAT,                   -- Treasury allocation %
    public_sale_allocation FLOAT,                -- Public sale allocation %
    
    -- Vesting schedule
    vesting_schedule JSON,                       -- Vesting schedule data
    unlocked_percentage FLOAT,                   -- Unlocked percentage %
    next_unlock_date TIMESTAMP,                  -- Next unlock date
    next_unlock_amount NUMERIC(30,8),            -- Next unlock amount
    
    -- Утилитарность токена
    utility_score FLOAT,                         -- Utility score
    governance_power FLOAT,                      -- Governance power
    staking_rewards FLOAT,                       -- Staking rewards %
    fee_burn_mechanism BOOLEAN DEFAULT FALSE     -- Fee burn mechanism
);
```

#### 7. `github_metrics` - GitHub активность
```sql
CREATE TABLE github_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Коммиты
    commits_24h INTEGER,                         -- Commits 24h
    commits_7d INTEGER,                          -- Commits 7d
    commits_30d INTEGER,                         -- Commits 30d
    commits_90d INTEGER,                         -- Commits 90d
    code_quality_score FLOAT,                    -- Code quality score
    test_coverage FLOAT,                         -- Test coverage %
    
    -- Pull Requests
    open_prs INTEGER,                            -- Open PRs
    merged_prs_7d INTEGER,                       -- Merged PRs 7d
    closed_prs_7d INTEGER,                       -- Closed PRs 7d
    pr_merge_rate FLOAT,                         -- PR merge rate %
    avg_pr_lifetime FLOAT,                       -- Average PR lifetime
    
    -- Issues
    open_issues INTEGER,                         -- Open issues
    closed_issues_7d INTEGER,                    -- Closed issues 7d
    issue_resolution_time FLOAT,                 -- Issue resolution time
    bug_report_ratio FLOAT,                      -- Bug report ratio
    
    -- Участники
    active_contributors_30d INTEGER,             -- Active contributors 30d
    total_contributors INTEGER,                  -- Total contributors
    external_contributors INTEGER,               -- External contributors
    core_team_activity FLOAT,                    -- Core team activity
    
    -- Популярность
    stars INTEGER,                               -- GitHub stars
    forks INTEGER,                               -- GitHub forks
    stars_change_7d INTEGER,                     -- Stars change 7d
    watch_count INTEGER,                         -- Watch count
    primary_language VARCHAR(50),                -- Primary language
    languages_distribution JSON                  -- Languages distribution
);
```

#### 8. `security_metrics` - Метрики безопасности
```sql
CREATE TABLE security_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Аудит и верификация
    audit_status VARCHAR(50),                    -- audited, unaudited, pending
    audit_firm VARCHAR(100),                     -- Audit firm
    audit_date TIMESTAMP,                        -- Audit date
    audit_score FLOAT,                           -- Audit score
    contract_verified BOOLEAN DEFAULT FALSE,     -- Contract verified
    source_code_available BOOLEAN DEFAULT FALSE, -- Source code available
    
    -- Безопасность контрактов
    vulnerability_score FLOAT,                   -- Vulnerability score
    multisig_wallets BOOLEAN DEFAULT FALSE,      -- Multisig wallets
    timelock_mechanisms BOOLEAN DEFAULT FALSE,   -- Timelock mechanisms
    upgrade_mechanisms VARCHAR(50),              -- Upgrade mechanisms
    emergency_pause BOOLEAN DEFAULT FALSE,       -- Emergency pause
    
    -- Децентрализация
    governance_decentralization FLOAT,           -- Governance decentralization
    validator_distribution FLOAT,                -- Validator distribution
    node_distribution FLOAT,                     -- Node distribution
    treasury_control FLOAT,                      -- Treasury control
    
    -- Smart contract security
    reentrancy_protection BOOLEAN DEFAULT FALSE, -- Reentrancy protection
    overflow_protection BOOLEAN DEFAULT FALSE,   -- Overflow protection
    access_control BOOLEAN DEFAULT FALSE,        -- Access control
    pause_functionality BOOLEAN DEFAULT FALSE    -- Pause functionality
);
```

#### 9. `community_metrics` - Метрики сообщества
```sql
CREATE TABLE community_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Социальные сети
    twitter_followers INTEGER,                   -- Twitter followers
    telegram_members INTEGER,                    -- Telegram members
    discord_members INTEGER,                     -- Discord members
    reddit_subscribers INTEGER,                  -- Reddit subscribers
    facebook_likes INTEGER,                      -- Facebook likes
    instagram_followers INTEGER,                 -- Instagram followers
    youtube_subscribers INTEGER,                 -- YouTube subscribers
    tiktok_followers INTEGER,                    -- TikTok followers
    
    -- Engagement metrics
    social_engagement_rate FLOAT,                -- Social engagement rate
    twitter_engagement_rate FLOAT,               -- Twitter engagement rate
    telegram_activity_score FLOAT,               -- Telegram activity score
    discord_activity_score FLOAT,                -- Discord activity score
    
    -- Content metrics
    blog_posts_30d INTEGER,                      -- Blog posts 30d
    youtube_videos_30d INTEGER,                  -- YouTube videos 30d
    podcast_appearances_30d INTEGER,             -- Podcast appearances 30d
    media_mentions_30d INTEGER,                  -- Media mentions 30d
    brand_awareness_score FLOAT,                 -- Brand awareness score
    
    -- Educational resources
    documentation_quality FLOAT,                 -- Documentation quality
    tutorial_availability FLOAT,                 -- Tutorial availability
    community_guides_count INTEGER,              -- Community guides count
    support_responsiveness FLOAT                 -- Support responsiveness
);
```

#### 10. `partnership_metrics` - Метрики партнерств
```sql
CREATE TABLE partnership_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Партнерства
    partnership_count INTEGER,                   -- Partnership count
    tier1_partnerships INTEGER,                  -- Tier 1 partnerships
    strategic_partnerships INTEGER,              -- Strategic partnerships
    partnership_quality_score FLOAT,             -- Partnership quality score
    
    -- Интеграции
    integration_count INTEGER,                   -- Integration count
    exchange_listings INTEGER,                   -- Exchange listings
    wallet_support INTEGER,                      -- Wallet support
    defi_protocol_integrations INTEGER,          -- DeFi protocol integrations
    cefi_integrations INTEGER,                   -- CeFi integrations
    nft_marketplace_support INTEGER,             -- NFT marketplace support
    cross_chain_bridges INTEGER,                 -- Cross-chain bridges
    
    -- Ecosystem metrics
    ecosystem_connections INTEGER,               -- Ecosystem connections
    interoperability_score FLOAT,                -- Interoperability score
    adoption_rate FLOAT                          -- Adoption rate
);
```

#### 11. `ml_predictions` - ML предсказания
```sql
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    model_name VARCHAR(100) NOT NULL,            -- Model name
    prediction_type VARCHAR(50) NOT NULL,        -- investment_score, price_prediction
    prediction_value FLOAT NOT NULL,             -- Prediction value
    confidence_score FLOAT NOT NULL,             -- Confidence score
    prediction_horizon VARCHAR(20) NOT NULL,     -- 1d, 7d, 30d
    features_used JSON,                          -- Features used
    model_version VARCHAR(20),                   -- Model version
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 12. `smart_contracts` - Смарт-контракты
```sql
CREATE TABLE smart_contracts (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    contract_address VARCHAR(42) UNIQUE NOT NULL,
    blockchain_id INTEGER REFERENCES blockchains(id),
    contract_name VARCHAR(100),
    contract_type VARCHAR(50),                   -- ERC20, ERC721, DeFi
    bytecode TEXT,
    abi JSON,
    verified BOOLEAN DEFAULT FALSE,
    deployer_address VARCHAR(42),
    deployment_tx VARCHAR(66),
    deployment_block INTEGER,
    deployment_timestamp TIMESTAMP,
    audit_status VARCHAR(50),                    -- audited, unaudited, pending
    audit_firm VARCHAR(100),
    audit_date TIMESTAMP,
    security_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### 13. `trending_metrics` - Трендовые метрики
```sql
CREATE TABLE trending_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Momentum indicators
    momentum_score FLOAT,                        -- Momentum score
    trend_direction VARCHAR(20),                 -- bullish, bearish, sideways
    trend_strength FLOAT,                        -- Trend strength
    
    -- Seasonal patterns
    seasonality_score FLOAT,                     -- Seasonality score
    cyclical_patterns JSON,                      -- Cyclical patterns
    
    -- Anomaly detection
    anomaly_score FLOAT,                         -- Anomaly score
    anomaly_type VARCHAR(50),                    -- Anomaly type
    anomaly_severity VARCHAR(20),                -- low, medium, high
    
    -- Market sentiment
    fear_greed_index FLOAT,                      -- Fear & Greed index
    social_sentiment FLOAT,                      -- Social sentiment
    news_sentiment FLOAT                         -- News sentiment
);
```

#### 14. `cross_chain_metrics` - Кроссчейн метрики
```sql
CREATE TABLE cross_chain_metrics (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES crypto_assets(id),
    source_blockchain_id INTEGER REFERENCES blockchains(id),
    target_blockchain_id INTEGER REFERENCES blockchains(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Bridge metrics
    bridge_volume_24h NUMERIC(30,8),             -- Bridge volume 24h
    bridge_transactions_24h INTEGER,             -- Bridge transactions 24h
    bridge_fees_24h NUMERIC(20,8),               -- Bridge fees 24h
    
    -- Liquidity metrics
    cross_chain_liquidity NUMERIC(30,8),         -- Cross-chain liquidity
    liquidity_imbalance FLOAT                    -- Liquidity imbalance
);
```

## 🌐 Поддерживаемые блокчейны (50+)

### Layer 1 Blockchains
- **Ethereum** (ETH) - Smart contract platform
- **Bitcoin** (BTC) - Digital currency
- **Binance Smart Chain** (BNB) - BSC ecosystem
- **Polygon** (MATIC) - Ethereum scaling
- **Avalanche** (AVAX) - High-performance platform
- **Solana** (SOL) - High-speed blockchain
- **Cardano** (ADA) - Research-driven platform
- **Polkadot** (DOT) - Multi-chain platform
- **Cosmos** (ATOM) - Internet of blockchains
- **Fantom** (FTM) - Fast smart contracts

### Layer 2 Solutions
- **Arbitrum One** - Optimistic rollup
- **Optimism** - Optimistic rollup
- **Polygon zkEVM** - Zero-knowledge rollup
- **zkSync Era** - Zero-knowledge rollup
- **StarkNet** - Zero-knowledge rollup
- **Base** - Coinbase Layer 2
- **Linea** - ConsenSys Layer 2
- **Scroll** - Native zkEVM
- **Mantle** - Modular Layer 2
- **Blast** - Native yield Layer 2

### DeFi Protocols
- **Uniswap** (UNI) - DEX protocol
- **Aave** (AAVE) - Lending protocol
- **Compound** (COMP) - Lending protocol
- **Curve** (CRV) - Stablecoin exchange
- **SushiSwap** (SUSHI) - DEX protocol

### Gaming & NFT
- **Immutable X** (IMX) - NFT Layer 2
- **Axie Infinity** (AXS) - Play-to-earn
- **The Sandbox** (SAND) - Virtual world
- **Decentraland** (MANA) - VR platform

### Privacy Coins
- **Monero** (XMR) - Privacy-focused
- **Zcash** (ZEC) - Privacy-focused
- **Dash** (DASH) - Digital cash

### Enterprise & Others
- **Tron** (TRX) - Content platform
- **Litecoin** (LTC) - Digital silver
- **Chainlink** (LINK) - Oracle network
- **Near Protocol** (NEAR) - Developer-friendly
- **Algorand** (ALGO) - Pure PoS
- **Tezos** (XTZ) - Self-amending
- **EOS** (EOS) - High-performance
- **Waves** (WAVES) - Custom platform
- **NEO** (NEO) - Smart economy
- **VeChain** (VET) - Enterprise-focused
- **Hedera** (HBAR) - Enterprise-grade
- **Elrond** (EGLD) - High-throughput
- **Harmony** (ONE) - Fast blockchain
- **Klaytn** (KLAY) - Metaverse blockchain
- **Cronos** (CRO) - Crypto.com blockchain
- **Gnosis Chain** (GNO) - Ethereum sidechain
- **Celo** (CELO) - Mobile-first
- **Moonbeam** (GLMR) - Ethereum-compatible parachain
- **Aurora** (ETH) - NEAR-based Ethereum
- **Evmos** (EVMOS) - Cosmos Ethereum
- **Kava** (KAVA) - DeFi hub
- **Injective** (INJ) - DeFi-focused
- **Osmosis** (OSMO) - Cosmos DEX
- **Juno** (JUNO) - Smart contracts
- **Secret Network** (SCRT) - Privacy contracts
- **Terra Classic** (LUNC) - Algorithmic stablecoins
- **Terra 2.0** (LUNA) - Rebuilt Terra
- **Aptos** (APT) - Move-based
- **Sui** (SUI) - Move-based
- **Sei** (SEI) - Trading-focused

## 📊 Индексы для производительности

```sql
-- Blockchain indexes
CREATE INDEX idx_blockchains_name ON blockchains(name);
CREATE INDEX idx_blockchains_symbol ON blockchains(symbol);
CREATE INDEX idx_blockchains_chain_id ON blockchains(chain_id);

-- Asset indexes
CREATE INDEX idx_crypto_assets_symbol ON crypto_assets(symbol);
CREATE INDEX idx_crypto_assets_blockchain_id ON crypto_assets(blockchain_id);
CREATE INDEX idx_crypto_assets_category ON crypto_assets(category);
CREATE INDEX idx_crypto_assets_contract_address ON crypto_assets(contract_address);

-- Metrics indexes
CREATE INDEX idx_onchain_metrics_asset_timestamp ON onchain_metrics(asset_id, timestamp);
CREATE INDEX idx_financial_metrics_asset_timestamp ON financial_metrics(asset_id, timestamp);
CREATE INDEX idx_github_metrics_asset_timestamp ON github_metrics(asset_id, timestamp);
CREATE INDEX idx_tokenomics_metrics_asset_timestamp ON tokenomics_metrics(asset_id, timestamp);
CREATE INDEX idx_security_metrics_asset_timestamp ON security_metrics(asset_id, timestamp);
CREATE INDEX idx_community_metrics_asset_timestamp ON community_metrics(asset_id, timestamp);
CREATE INDEX idx_partnership_metrics_asset_timestamp ON partnership_metrics(asset_id, timestamp);

-- Network metrics indexes
CREATE INDEX idx_network_metrics_blockchain_timestamp ON network_metrics(blockchain_id, timestamp);

-- ML predictions indexes
CREATE INDEX idx_ml_predictions_asset_model ON ml_predictions(asset_id, model_name);
CREATE INDEX idx_ml_predictions_type_horizon ON ml_predictions(prediction_type, prediction_horizon);

-- Smart contracts indexes
CREATE INDEX idx_smart_contracts_address ON smart_contracts(contract_address);
CREATE INDEX idx_smart_contracts_blockchain ON smart_contracts(blockchain_id);
CREATE INDEX idx_smart_contracts_asset ON smart_contracts(asset_id);

-- Trending metrics indexes
CREATE INDEX idx_trending_metrics_asset_timestamp ON trending_metrics(asset_id, timestamp);

-- Cross-chain metrics indexes
CREATE INDEX idx_cross_chain_metrics_asset_timestamp ON cross_chain_metrics(asset_id, timestamp);
CREATE INDEX idx_cross_chain_metrics_source_target ON cross_chain_metrics(source_blockchain_id, target_blockchain_id);
```

## 🚀 Установка и миграция

### 1. Новая установка
```bash
python setup_database_v2.py
```

### 2. Миграция с существующей БД
```bash
python migrate_database.py
```

### 3. Инициализация блокчейнов
```bash
python src/database/blockchain_init.py
```

## 📈 Преимущества новой схемы

1. **Масштабируемость** - Поддержка 50+ блокчейнов
2. **Полнота данных** - 10 категорий метрик
3. **Производительность** - Оптимизированные индексы
4. **Гибкость** - JSON поля для сложных данных
5. **Аналитика** - Кроссчейн метрики
6. **ML готовность** - Структурированные данные для ML
7. **Безопасность** - Метрики аудита и безопасности
8. **Сообщество** - Социальные метрики
9. **Партнерства** - Экосистемные связи
10. **Тренды** - Временные паттерны

## 🔄 Обновления

Система поддерживает автоматические обновления схемы через миграции. Все изменения обратно совместимы.
