-- DEFIMON Database Initialization Script
-- Creates the database schema for the Java 8 microservices platform

-- Create database if not exists
CREATE DATABASE defimon_db;

-- Connect to the database
\c defimon_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS defimon;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO defimon, public;

-- Create assets table
CREATE TABLE IF NOT EXISTS assets (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    contract_address VARCHAR(42),
    blockchain VARCHAR(50) NOT NULL,
    category VARCHAR(50),
    decimals INTEGER,
    total_supply NUMERIC(36, 18),
    circulating_supply NUMERIC(36, 18),
    max_supply NUMERIC(36, 18),
    github_repo VARCHAR(200),
    website VARCHAR(200),
    description TEXT,
    logo_url VARCHAR(500),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    security_score NUMERIC(10, 4),
    risk_level VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_data_collection TIMESTAMP
);

-- Create indexes for assets
CREATE INDEX IF NOT EXISTS idx_asset_symbol ON assets(symbol);
CREATE INDEX IF NOT EXISTS idx_asset_blockchain ON assets(blockchain);
CREATE INDEX IF NOT EXISTS idx_asset_contract ON assets(contract_address);
CREATE INDEX IF NOT EXISTS idx_asset_active ON assets(is_active);
CREATE INDEX IF NOT EXISTS idx_asset_updated ON assets(updated_at);

-- Create onchain_metrics table
CREATE TABLE IF NOT EXISTS onchain_metrics (
    id BIGSERIAL PRIMARY KEY,
    asset_id BIGINT NOT NULL REFERENCES assets(id),
    timestamp TIMESTAMP NOT NULL,
    price_usd NUMERIC(20, 8),
    price_btc NUMERIC(20, 8),
    price_eth NUMERIC(20, 8),
    market_cap NUMERIC(20, 2),
    fully_diluted_valuation NUMERIC(20, 2),
    volume_24h NUMERIC(20, 2),
    volume_change_24h NUMERIC(10, 4),
    price_change_1h NUMERIC(10, 4),
    price_change_24h NUMERIC(10, 4),
    price_change_7d NUMERIC(10, 4),
    price_change_30d NUMERIC(10, 4),
    high_24h NUMERIC(20, 8),
    low_24h NUMERIC(20, 8),
    high_7d NUMERIC(20, 8),
    low_7d NUMERIC(20, 8),
    high_30d NUMERIC(20, 8),
    low_30d NUMERIC(20, 8),
    active_addresses_24h BIGINT,
    active_addresses_7d BIGINT,
    active_addresses_30d BIGINT,
    transaction_count_24h BIGINT,
    transaction_volume_24h NUMERIC(20, 8),
    average_transaction_value NUMERIC(20, 8),
    network_hash_rate NUMERIC(20, 2),
    difficulty NUMERIC(20, 2),
    block_height BIGINT,
    block_time_avg NUMERIC(10, 4),
    gas_price_avg NUMERIC(20, 8),
    gas_used_avg BIGINT,
    tvl NUMERIC(20, 2),
    tvl_change_24h NUMERIC(10, 4),
    yield_farming_apy NUMERIC(10, 4),
    liquidity_pools_count INTEGER,
    dex_volume_24h NUMERIC(20, 2),
    supply_inflation_rate NUMERIC(10, 6),
    burn_rate_24h NUMERIC(20, 8),
    staked_amount NUMERIC(20, 8),
    staking_apy NUMERIC(10, 4),
    whale_transactions_24h INTEGER,
    exchange_inflows_24h NUMERIC(20, 8),
    exchange_outflows_24h NUMERIC(20, 8),
    exchange_netflow_24h NUMERIC(20, 8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for onchain_metrics
CREATE INDEX IF NOT EXISTS idx_onchain_asset_timestamp ON onchain_metrics(asset_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_onchain_timestamp ON onchain_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_onchain_price ON onchain_metrics(price_usd);
CREATE INDEX IF NOT EXISTS idx_onchain_market_cap ON onchain_metrics(market_cap);

-- Create financial_metrics table
CREATE TABLE IF NOT EXISTS financial_metrics (
    id BIGSERIAL PRIMARY KEY,
    asset_id BIGINT NOT NULL REFERENCES assets(id),
    timestamp TIMESTAMP NOT NULL,
    price_to_earnings_ratio NUMERIC(10, 4),
    price_to_sales_ratio NUMERIC(10, 4),
    price_to_book_ratio NUMERIC(10, 4),
    debt_to_equity_ratio NUMERIC(10, 4),
    current_ratio NUMERIC(10, 4),
    quick_ratio NUMERIC(10, 4),
    revenue_24h NUMERIC(20, 2),
    revenue_7d NUMERIC(20, 2),
    revenue_30d NUMERIC(20, 2),
    revenue_growth_rate NUMERIC(10, 4),
    gross_profit_margin NUMERIC(10, 4),
    net_profit_margin NUMERIC(10, 4),
    operating_margin NUMERIC(10, 4),
    operating_cash_flow NUMERIC(20, 2),
    investing_cash_flow NUMERIC(20, 2),
    financing_cash_flow NUMERIC(20, 2),
    free_cash_flow NUMERIC(20, 2),
    total_assets NUMERIC(20, 2),
    total_liabilities NUMERIC(20, 2),
    total_equity NUMERIC(20, 2),
    working_capital NUMERIC(20, 2),
    cash_and_equivalents NUMERIC(20, 2),
    short_term_debt NUMERIC(20, 2),
    long_term_debt NUMERIC(20, 2),
    enterprise_value NUMERIC(20, 2),
    ev_to_revenue NUMERIC(10, 4),
    ev_to_ebitda NUMERIC(10, 4),
    price_to_cash_flow NUMERIC(10, 4),
    price_to_free_cash_flow NUMERIC(10, 4),
    earnings_growth_rate NUMERIC(10, 4),
    book_value_growth_rate NUMERIC(10, 4),
    dividend_yield NUMERIC(10, 4),
    payout_ratio NUMERIC(10, 4),
    beta NUMERIC(10, 4),
    volatility_30d NUMERIC(10, 4),
    volatility_90d NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    sortino_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    bid_ask_spread NUMERIC(10, 6),
    market_depth NUMERIC(20, 2),
    liquidity_score NUMERIC(10, 4),
    return_on_assets NUMERIC(10, 4),
    return_on_equity NUMERIC(10, 4),
    return_on_invested_capital NUMERIC(10, 4),
    asset_turnover NUMERIC(10, 4),
    inventory_turnover NUMERIC(10, 4),
    receivables_turnover NUMERIC(10, 4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for financial_metrics
CREATE INDEX IF NOT EXISTS idx_financial_asset_timestamp ON financial_metrics(asset_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_financial_timestamp ON financial_metrics(timestamp);

-- Create ml_predictions table
CREATE TABLE IF NOT EXISTS ml_predictions (
    id BIGSERIAL PRIMARY KEY,
    asset_id BIGINT NOT NULL REFERENCES assets(id),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_horizon VARCHAR(20) NOT NULL,
    prediction_value NUMERIC(20, 8) NOT NULL,
    confidence_score NUMERIC(5, 4) NOT NULL,
    prediction_interval_lower NUMERIC(20, 8),
    prediction_interval_upper NUMERIC(20, 8),
    feature_importance TEXT,
    model_metadata TEXT,
    training_data_size BIGINT,
    training_accuracy NUMERIC(5, 4),
    validation_accuracy NUMERIC(5, 4),
    test_accuracy NUMERIC(5, 4),
    rmse NUMERIC(20, 8),
    mae NUMERIC(20, 8),
    mape NUMERIC(10, 4),
    r_squared NUMERIC(5, 4),
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    volatility NUMERIC(10, 4),
    skewness NUMERIC(10, 4),
    kurtosis NUMERIC(10, 4),
    var_95 NUMERIC(20, 8),
    var_99 NUMERIC(20, 8),
    expected_shortfall NUMERIC(20, 8),
    calmar_ratio NUMERIC(10, 4),
    sortino_ratio NUMERIC(10, 4),
    information_ratio NUMERIC(10, 4),
    treynor_ratio NUMERIC(10, 4),
    jensen_alpha NUMERIC(10, 4),
    tracking_error NUMERIC(10, 4),
    beta NUMERIC(10, 4),
    correlation_market NUMERIC(5, 4),
    correlation_btc NUMERIC(5, 4),
    correlation_eth NUMERIC(5, 4),
    sentiment_score NUMERIC(5, 4),
    technical_score NUMERIC(5, 4),
    fundamental_score NUMERIC(5, 4),
    social_score NUMERIC(5, 4),
    overall_score NUMERIC(5, 4),
    risk_score NUMERIC(5, 4),
    investment_recommendation VARCHAR(20),
    target_price NUMERIC(20, 8),
    stop_loss NUMERIC(20, 8),
    take_profit NUMERIC(20, 8),
    position_size NUMERIC(10, 4),
    expected_return NUMERIC(10, 4),
    expected_volatility NUMERIC(10, 4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create indexes for ml_predictions
CREATE INDEX IF NOT EXISTS idx_ml_asset_timestamp ON ml_predictions(asset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ml_model ON ml_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_ml_horizon ON ml_predictions(prediction_horizon);
CREATE INDEX IF NOT EXISTS idx_ml_confidence ON ml_predictions(confidence_score);
CREATE INDEX IF NOT EXISTS idx_ml_expires ON ml_predictions(expires_at);

-- Create social_metrics table
CREATE TABLE IF NOT EXISTS social_metrics (
    id BIGSERIAL PRIMARY KEY,
    asset_id BIGINT NOT NULL REFERENCES assets(id),
    timestamp TIMESTAMP NOT NULL,
    github_stars BIGINT,
    github_forks BIGINT,
    github_watchers BIGINT,
    github_commits_24h BIGINT,
    github_commits_7d BIGINT,
    github_commits_30d BIGINT,
    github_contributors BIGINT,
    github_issues_open BIGINT,
    github_issues_closed BIGINT,
    github_pull_requests_open BIGINT,
    github_pull_requests_merged BIGINT,
    github_languages TEXT,
    github_last_commit TIMESTAMP,
    twitter_followers BIGINT,
    twitter_following BIGINT,
    twitter_tweets_24h BIGINT,
    twitter_tweets_7d BIGINT,
    twitter_mentions_24h BIGINT,
    twitter_mentions_7d BIGINT,
    twitter_retweets_24h BIGINT,
    twitter_likes_24h BIGINT,
    twitter_sentiment_score NUMERIC(5, 4),
    twitter_sentiment_positive NUMERIC(5, 4),
    twitter_sentiment_negative NUMERIC(5, 4),
    twitter_sentiment_neutral NUMERIC(5, 4),
    reddit_subscribers BIGINT,
    reddit_active_users BIGINT,
    reddit_posts_24h BIGINT,
    reddit_comments_24h BIGINT,
    reddit_upvotes_24h BIGINT,
    reddit_downvotes_24h BIGINT,
    reddit_sentiment_score NUMERIC(5, 4),
    telegram_members BIGINT,
    telegram_messages_24h BIGINT,
    telegram_messages_7d BIGINT,
    telegram_sentiment_score NUMERIC(5, 4),
    discord_members BIGINT,
    discord_online_members BIGINT,
    discord_messages_24h BIGINT,
    discord_sentiment_score NUMERIC(5, 4),
    youtube_subscribers BIGINT,
    youtube_videos_30d BIGINT,
    youtube_views_30d BIGINT,
    youtube_likes_30d BIGINT,
    youtube_dislikes_30d BIGINT,
    youtube_sentiment_score NUMERIC(5, 4),
    news_mentions_24h BIGINT,
    news_mentions_7d BIGINT,
    news_sentiment_score NUMERIC(5, 4),
    news_sentiment_positive NUMERIC(5, 4),
    news_sentiment_negative NUMERIC(5, 4),
    news_sentiment_neutral NUMERIC(5, 4),
    google_trends_score NUMERIC(5, 4),
    google_search_volume BIGINT,
    bing_search_volume BIGINT,
    yahoo_search_volume BIGINT,
    influencer_mentions_24h BIGINT,
    influencer_sentiment_score NUMERIC(5, 4),
    influencer_reach BIGINT,
    community_growth_rate NUMERIC(10, 4),
    engagement_rate NUMERIC(5, 4),
    activity_score NUMERIC(5, 4),
    buzz_score NUMERIC(5, 4),
    fear_greed_index NUMERIC(5, 4),
    social_volume_score NUMERIC(5, 4),
    social_sentiment_score NUMERIC(5, 4),
    social_engagement_score NUMERIC(5, 4),
    social_influence_score NUMERIC(5, 4),
    overall_social_score NUMERIC(5, 4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for social_metrics
CREATE INDEX IF NOT EXISTS idx_social_asset_timestamp ON social_metrics(asset_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_social_timestamp ON social_metrics(timestamp);

-- Create collection_stats table for monitoring
CREATE TABLE IF NOT EXISTS collection_stats (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    total_success NUMERIC(20, 2) DEFAULT 0,
    total_errors NUMERIC(20, 2) DEFAULT 0,
    total_assets_processed NUMERIC(20, 2) DEFAULT 0,
    average_collection_time NUMERIC(10, 4) DEFAULT 0,
    max_concurrent_collections INTEGER DEFAULT 0,
    last_collection_time TIMESTAMP,
    success_rate NUMERIC(5, 4) DEFAULT 0,
    error_rate NUMERIC(5, 4) DEFAULT 0,
    throughput NUMERIC(10, 4) DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for collection_stats
CREATE INDEX IF NOT EXISTS idx_collection_stats_service ON collection_stats(service_name);
CREATE INDEX IF NOT EXISTS idx_collection_stats_time ON collection_stats(last_collection_time);

-- Insert default assets
INSERT INTO assets (symbol, name, blockchain, category, is_active, is_verified, created_at, updated_at) VALUES
('BTC', 'Bitcoin', 'bitcoin', 'cryptocurrency', true, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('ETH', 'Ethereum', 'ethereum', 'cryptocurrency', true, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('MATIC', 'Polygon', 'polygon', 'cryptocurrency', true, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('USDC', 'USD Coin', 'ethereum', 'stablecoin', true, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('USDT', 'Tether', 'ethereum', 'stablecoin', true, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (symbol) DO NOTHING;

-- Create functions for data collection monitoring
CREATE OR REPLACE FUNCTION update_asset_last_collection()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE assets 
    SET last_data_collection = NEW.timestamp, updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.asset_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_asset_collection_onchain
    AFTER INSERT ON onchain_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_asset_last_collection();

CREATE TRIGGER trigger_update_asset_collection_financial
    AFTER INSERT ON financial_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_asset_last_collection();

CREATE TRIGGER trigger_update_asset_collection_social
    AFTER INSERT ON social_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_asset_last_collection();

-- Create views for analytics
CREATE OR REPLACE VIEW asset_summary AS
SELECT 
    a.id,
    a.symbol,
    a.name,
    a.blockchain,
    a.is_active,
    a.is_verified,
    a.last_data_collection,
    o.price_usd,
    o.market_cap,
    o.volume_24h,
    o.price_change_24h,
    s.overall_social_score,
    m.overall_score as ml_score,
    m.investment_recommendation
FROM assets a
LEFT JOIN LATERAL (
    SELECT price_usd, market_cap, volume_24h, price_change_24h
    FROM onchain_metrics 
    WHERE asset_id = a.id 
    ORDER BY timestamp DESC 
    LIMIT 1
) o ON true
LEFT JOIN LATERAL (
    SELECT overall_social_score
    FROM social_metrics 
    WHERE asset_id = a.id 
    ORDER BY timestamp DESC 
    LIMIT 1
) s ON true
LEFT JOIN LATERAL (
    SELECT overall_score, investment_recommendation
    FROM ml_predictions 
    WHERE asset_id = a.id 
    ORDER BY created_at DESC 
    LIMIT 1
) m ON true;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA defimon TO defimon;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA defimon TO defimon;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA defimon TO defimon;

-- Create monitoring user
CREATE USER IF NOT EXISTS defimon_monitor WITH PASSWORD 'monitor_password';
GRANT SELECT ON ALL TABLES IN SCHEMA defimon TO defimon_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO defimon_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO defimon_monitor;

-- Log completion
INSERT INTO collection_stats (service_name, total_success, last_collection_time) 
VALUES ('database_init', 1, CURRENT_TIMESTAMP);

COMMIT;
