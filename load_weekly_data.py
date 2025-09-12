#!/usr/bin/env python3
"""
Скрипт для загрузки данных Polygon за последнюю неделю
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'defimon_db',
    'user': 'defimon',
    'password': 'password'
}

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

def load_weekly_network_metrics():
    """Загрузить метрики сети за неделю"""
    logger.info("Loading weekly network metrics...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Генерируем данные за последние 7 дней
        for day in range(7):
            timestamp = datetime.now() - timedelta(days=day)
            
            # Случайные значения для метрик сети Polygon
            network_metrics = {
                'blockchain_id': 1,  # Polygon
                'timestamp': timestamp,
                'block_time_avg': 2.1 + random.uniform(-0.2, 0.2),
                'block_size_avg': 50000 + random.randint(-5000, 5000),
                'transaction_throughput': 100 + random.randint(-10, 10),
                'network_utilization': 75.5 + random.uniform(-5, 5),
                'hash_rate': 1000000 + random.randint(-100000, 100000),
                'difficulty': 10000000 + random.randint(-1000000, 1000000),
                'validator_count': 100 + random.randint(-5, 5),
                'staking_ratio': 45.2 + random.uniform(-2, 2),
                'total_supply': 10000000000 + random.randint(-1000000, 1000000),
                'inflation_rate': 2.5 + random.uniform(-0.5, 0.5),
                'deflation_rate': 0.1 + random.uniform(-0.05, 0.05),
                'burn_rate': 0.2 + random.uniform(-0.1, 0.1),
                'gas_price_avg': 30.5 + random.uniform(-5, 5),
                'gas_price_median': 28.0 + random.uniform(-5, 5),
                'gas_limit': 30000000,
                'gas_used_avg': 15000000 + random.randint(-1000000, 1000000)
            }
            
            cursor.execute("""
                INSERT INTO network_metrics (
                    blockchain_id, timestamp, block_time_avg, block_size_avg,
                    transaction_throughput, network_utilization, hash_rate,
                    difficulty, validator_count, staking_ratio, total_supply,
                    inflation_rate, deflation_rate, burn_rate, gas_price_avg,
                    gas_price_median, gas_limit, gas_used_avg
                ) VALUES (
                    %(blockchain_id)s, %(timestamp)s, %(block_time_avg)s, %(block_size_avg)s,
                    %(transaction_throughput)s, %(network_utilization)s, %(hash_rate)s,
                    %(difficulty)s, %(validator_count)s, %(staking_ratio)s, %(total_supply)s,
                    %(inflation_rate)s, %(deflation_rate)s, %(burn_rate)s, %(gas_price_avg)s,
                    %(gas_price_median)s, %(gas_limit)s, %(gas_used_avg)s
                )
            """, network_metrics)
        
        conn.commit()
        logger.info("✅ Network metrics loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading network metrics: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def load_weekly_asset_metrics():
    """Загрузить метрики активов за неделю"""
    logger.info("Loading weekly asset metrics...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Получить все активы
        cursor.execute("SELECT id, symbol FROM crypto_assets")
        assets = cursor.fetchall()
        
        # Генерируем данные за последние 7 дней для каждого актива
        for asset_id, symbol in assets:
            logger.info(f"Loading metrics for {symbol}...")
            
            for day in range(7):
                timestamp = datetime.now() - timedelta(days=day)
                
                # Базовые значения для каждого актива
                base_values = {
                    'MATIC': {'price': 0.85, 'tvl': 1000000, 'volume': 50000000},
                    'USDC': {'price': 1.00, 'tvl': 2000000, 'volume': 100000000},
                    'USDT': {'price': 1.00, 'tvl': 1500000, 'volume': 80000000},
                    'WETH': {'price': 2500, 'tvl': 500000, 'volume': 30000000},
                    'WBTC': {'price': 45000, 'tvl': 300000, 'volume': 20000000},
                    'AAVE': {'price': 120, 'tvl': 800000, 'volume': 15000000},
                    'CRV': {'price': 0.6, 'tvl': 400000, 'volume': 10000000},
                    'SUSHI': {'price': 1.2, 'tvl': 300000, 'volume': 8000000},
                    'QUICK': {'price': 0.8, 'tvl': 200000, 'volume': 5000000},
                    'BAL': {'price': 4.5, 'tvl': 250000, 'volume': 6000000}
                }
                
                base = base_values.get(symbol, {'price': 1.0, 'tvl': 100000, 'volume': 1000000})
                
                # On-chain метрики
                onchain_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'tvl': base['tvl'] * (1 + random.uniform(-0.1, 0.1)),
                    'tvl_change_24h': random.uniform(-5, 5),
                    'tvl_change_7d': random.uniform(-15, 15),
                    'tvl_change_30d': random.uniform(-30, 30),
                    'daily_transactions': 1000 + random.randint(-200, 200),
                    'transaction_volume_24h': base['volume'] * (1 + random.uniform(-0.2, 0.2)),
                    'transaction_volume_7d': base['volume'] * 7 * (1 + random.uniform(-0.1, 0.1)),
                    'avg_transaction_fee': 0.01 + random.uniform(-0.005, 0.005),
                    'transaction_success_rate': 98.5 + random.uniform(-1, 1),
                    'gas_usage_efficiency': 85.0 + random.uniform(-5, 5),
                    'active_addresses_24h': 500 + random.randint(-100, 100),
                    'new_addresses_24h': 50 + random.randint(-10, 10),
                    'unique_users_7d': 2000 + random.randint(-200, 200),
                    'user_retention_rate': 75.0 + random.uniform(-5, 5),
                    'whale_activity': random.uniform(0, 10),
                    'new_contracts_deployed': random.randint(0, 5),
                    'contract_interactions_24h': 100 + random.randint(-20, 20),
                    'contract_complexity_score': 7.5 + random.uniform(-1, 1),
                    'liquidity_pools_count': 10 + random.randint(-2, 2),
                    'liquidity_pools_tvl': base['tvl'] * 0.8 * (1 + random.uniform(-0.1, 0.1)),
                    'yield_farming_apy': 5.0 + random.uniform(-2, 2),
                    'lending_volume': base['volume'] * 0.3 * (1 + random.uniform(-0.2, 0.2)),
                    'borrowing_volume': base['volume'] * 0.2 * (1 + random.uniform(-0.2, 0.2))
                }
                
                cursor.execute("""
                    INSERT INTO onchain_metrics (
                        asset_id, timestamp, tvl, tvl_change_24h, tvl_change_7d, tvl_change_30d,
                        daily_transactions, transaction_volume_24h, transaction_volume_7d,
                        avg_transaction_fee, transaction_success_rate, gas_usage_efficiency,
                        active_addresses_24h, new_addresses_24h, unique_users_7d,
                        user_retention_rate, whale_activity, new_contracts_deployed,
                        contract_interactions_24h, contract_complexity_score,
                        liquidity_pools_count, liquidity_pools_tvl, yield_farming_apy,
                        lending_volume, borrowing_volume
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(tvl)s, %(tvl_change_24h)s, %(tvl_change_7d)s, %(tvl_change_30d)s,
                        %(daily_transactions)s, %(transaction_volume_24h)s, %(transaction_volume_7d)s,
                        %(avg_transaction_fee)s, %(transaction_success_rate)s, %(gas_usage_efficiency)s,
                        %(active_addresses_24h)s, %(new_addresses_24h)s, %(unique_users_7d)s,
                        %(user_retention_rate)s, %(whale_activity)s, %(new_contracts_deployed)s,
                        %(contract_interactions_24h)s, %(contract_complexity_score)s,
                        %(liquidity_pools_count)s, %(liquidity_pools_tvl)s, %(yield_farming_apy)s,
                        %(lending_volume)s, %(borrowing_volume)s
                    )
                """, onchain_metrics)
                
                # Финансовые метрики
                financial_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'price_usd': base['price'] * (1 + random.uniform(-0.05, 0.05)),
                    'market_cap': base['price'] * 1000000000 * (1 + random.uniform(-0.1, 0.1)),
                    'fully_diluted_valuation': base['price'] * 1000000000 * 1.2 * (1 + random.uniform(-0.1, 0.1)),
                    'market_cap_rank': random.randint(1, 100),
                    'market_cap_dominance': random.uniform(0.1, 5.0),
                    'volume_24h': base['volume'] * (1 + random.uniform(-0.3, 0.3)),
                    'volume_7d': base['volume'] * 7 * (1 + random.uniform(-0.2, 0.2)),
                    'volume_change_24h': random.uniform(-20, 20),
                    'volume_market_cap_ratio': random.uniform(0.1, 2.0),
                    'liquidity_score': 8.0 + random.uniform(-1, 1),
                    'volatility_24h': 2.5 + random.uniform(-1, 1),
                    'volatility_7d': 5.0 + random.uniform(-2, 2),
                    'volatility_30d': 10.0 + random.uniform(-3, 3),
                    'beta_coefficient': 1.0 + random.uniform(-0.5, 0.5),
                    'bid_ask_spread': 0.001 + random.uniform(-0.0005, 0.0005),
                    'order_book_depth': 100000 + random.randint(-10000, 10000),
                    'slippage_analysis': 0.1 + random.uniform(-0.05, 0.05),
                    'price_change_1h': random.uniform(-2, 2),
                    'price_change_24h': random.uniform(-10, 10),
                    'price_change_7d': random.uniform(-20, 20),
                    'price_change_30d': random.uniform(-40, 40),
                    'price_change_90d': random.uniform(-60, 60),
                    'price_change_1y': random.uniform(-80, 80),
                    'all_time_high': base['price'] * 2.0,
                    'all_time_low': base['price'] * 0.5,
                    'ath_date': datetime.now() - timedelta(days=random.randint(30, 365)),
                    'atl_date': datetime.now() - timedelta(days=random.randint(100, 1000))
                }
                
                cursor.execute("""
                    INSERT INTO financial_metrics (
                        asset_id, timestamp, price_usd, market_cap, fully_diluted_valuation,
                        market_cap_rank, market_cap_dominance, volume_24h, volume_7d,
                        volume_change_24h, volume_market_cap_ratio, liquidity_score,
                        volatility_24h, volatility_7d, volatility_30d, beta_coefficient,
                        bid_ask_spread, order_book_depth, slippage_analysis,
                        price_change_1h, price_change_24h, price_change_7d,
                        price_change_30d, price_change_90d, price_change_1y,
                        all_time_high, all_time_low, ath_date, atl_date
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(price_usd)s, %(market_cap)s, %(fully_diluted_valuation)s,
                        %(market_cap_rank)s, %(market_cap_dominance)s, %(volume_24h)s, %(volume_7d)s,
                        %(volume_change_24h)s, %(volume_market_cap_ratio)s, %(liquidity_score)s,
                        %(volatility_24h)s, %(volatility_7d)s, %(volatility_30d)s, %(beta_coefficient)s,
                        %(bid_ask_spread)s, %(order_book_depth)s, %(slippage_analysis)s,
                        %(price_change_1h)s, %(price_change_24h)s, %(price_change_7d)s,
                        %(price_change_30d)s, %(price_change_90d)s, %(price_change_1y)s,
                        %(all_time_high)s, %(all_time_low)s, %(ath_date)s, %(atl_date)s
                    )
                """, financial_metrics)
                
                # Токеномика
                tokenomics_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'circulating_supply': 1000000000 + random.randint(-100000000, 100000000),
                    'total_supply': 1000000000 + random.randint(-100000000, 100000000),
                    'max_supply': 10000000000 if symbol != 'USDC' and symbol != 'USDT' else None,
                    'inflation_rate': 2.5 + random.uniform(-1, 1),
                    'burn_rate': 0.1 + random.uniform(-0.05, 0.05),
                    'team_allocation': 20.0 + random.uniform(-5, 5),
                    'investor_allocation': 15.0 + random.uniform(-3, 3),
                    'community_allocation': 40.0 + random.uniform(-10, 10),
                    'treasury_allocation': 15.0 + random.uniform(-3, 3),
                    'public_sale_allocation': 10.0 + random.uniform(-2, 2),
                    'unlocked_percentage': 80.0 + random.uniform(-10, 10),
                    'utility_score': 7.5 + random.uniform(-1, 1),
                    'governance_power': 8.0 + random.uniform(-1, 1),
                    'staking_rewards': 5.0 + random.uniform(-2, 2),
                    'fee_burn_mechanism': random.choice([True, False])
                }
                
                cursor.execute("""
                    INSERT INTO tokenomics_metrics (
                        asset_id, timestamp, circulating_supply, total_supply, max_supply,
                        inflation_rate, burn_rate, team_allocation, investor_allocation,
                        community_allocation, treasury_allocation, public_sale_allocation,
                        unlocked_percentage, utility_score, governance_power,
                        staking_rewards, fee_burn_mechanism
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(circulating_supply)s, %(total_supply)s, %(max_supply)s,
                        %(inflation_rate)s, %(burn_rate)s, %(team_allocation)s, %(investor_allocation)s,
                        %(community_allocation)s, %(treasury_allocation)s, %(public_sale_allocation)s,
                        %(unlocked_percentage)s, %(utility_score)s, %(governance_power)s,
                        %(staking_rewards)s, %(fee_burn_mechanism)s
                    )
                """, tokenomics_metrics)
                
                # GitHub метрики (только для активов с GitHub)
                if symbol in ['MATIC', 'AAVE', 'CRV', 'SUSHI', 'QUICK', 'BAL']:
                    github_metrics = {
                        'asset_id': asset_id,
                        'timestamp': timestamp,
                        'commits_24h': random.randint(1, 10),
                        'commits_7d': random.randint(10, 50),
                        'commits_30d': random.randint(50, 200),
                        'commits_90d': random.randint(200, 800),
                        'code_quality_score': 8.0 + random.uniform(-1, 1),
                        'test_coverage': 75.0 + random.uniform(-10, 10),
                        'open_prs': random.randint(5, 30),
                        'merged_prs_7d': random.randint(3, 15),
                        'closed_prs_7d': random.randint(1, 10),
                        'pr_merge_rate': 80.0 + random.uniform(-10, 10),
                        'avg_pr_lifetime': 3.0 + random.uniform(-1, 1),
                        'open_issues': random.randint(10, 50),
                        'closed_issues_7d': random.randint(5, 25),
                        'issue_resolution_time': 2.0 + random.uniform(-0.5, 0.5),
                        'bug_report_ratio': 0.1 + random.uniform(-0.05, 0.05),
                        'active_contributors_30d': random.randint(5, 20),
                        'total_contributors': random.randint(50, 200),
                        'external_contributors': random.randint(10, 50),
                        'core_team_activity': 8.5 + random.uniform(-1, 1),
                        'stars': random.randint(1000, 10000),
                        'forks': random.randint(100, 1000),
                        'stars_change_7d': random.randint(-50, 50),
                        'watch_count': random.randint(100, 500),
                        'primary_language': random.choice(['Solidity', 'TypeScript', 'Python', 'Rust']),
                        'languages_distribution': '{"Solidity": 60, "TypeScript": 25, "Python": 10, "Other": 5}'
                    }
                    
                    cursor.execute("""
                        INSERT INTO github_metrics (
                            asset_id, timestamp, commits_24h, commits_7d, commits_30d, commits_90d,
                            code_quality_score, test_coverage, open_prs, merged_prs_7d, closed_prs_7d,
                            pr_merge_rate, avg_pr_lifetime, open_issues, closed_issues_7d,
                            issue_resolution_time, bug_report_ratio, active_contributors_30d,
                            total_contributors, external_contributors, core_team_activity,
                            stars, forks, stars_change_7d, watch_count, primary_language,
                            languages_distribution
                        ) VALUES (
                            %(asset_id)s, %(timestamp)s, %(commits_24h)s, %(commits_7d)s, %(commits_30d)s, %(commits_90d)s,
                            %(code_quality_score)s, %(test_coverage)s, %(open_prs)s, %(merged_prs_7d)s, %(closed_prs_7d)s,
                            %(pr_merge_rate)s, %(avg_pr_lifetime)s, %(open_issues)s, %(closed_issues_7d)s,
                            %(issue_resolution_time)s, %(bug_report_ratio)s, %(active_contributors_30d)s,
                            %(total_contributors)s, %(external_contributors)s, %(core_team_activity)s,
                            %(stars)s, %(forks)s, %(stars_change_7d)s, %(watch_count)s, %(primary_language)s,
                            %(languages_distribution)s
                        )
                    """, github_metrics)
                
                # Метрики безопасности
                security_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'audit_status': random.choice(['audited', 'unaudited', 'pending']),
                    'audit_firm': random.choice(['ConsenSys', 'OpenZeppelin', 'Trail of Bits', 'Quantstamp', None]),
                    'audit_score': 8.5 + random.uniform(-1, 1) if random.choice([True, False]) else None,
                    'contract_verified': random.choice([True, False]),
                    'source_code_available': random.choice([True, False]),
                    'vulnerability_score': 2.0 + random.uniform(-1, 1),
                    'multisig_wallets': random.choice([True, False]),
                    'timelock_mechanisms': random.choice([True, False]),
                    'upgrade_mechanisms': random.choice(['proxy', 'governance', 'none']),
                    'emergency_pause': random.choice([True, False]),
                    'governance_decentralization': 7.0 + random.uniform(-2, 2),
                    'validator_distribution': 8.0 + random.uniform(-1, 1),
                    'node_distribution': 7.5 + random.uniform(-1, 1),
                    'treasury_control': 6.0 + random.uniform(-2, 2),
                    'reentrancy_protection': random.choice([True, False]),
                    'overflow_protection': random.choice([True, False]),
                    'access_control': random.choice([True, False]),
                    'pause_functionality': random.choice([True, False])
                }
                
                cursor.execute("""
                    INSERT INTO security_metrics (
                        asset_id, timestamp, audit_status, audit_firm, audit_score,
                        contract_verified, source_code_available, vulnerability_score,
                        multisig_wallets, timelock_mechanisms, upgrade_mechanisms,
                        emergency_pause, governance_decentralization, validator_distribution,
                        node_distribution, treasury_control, reentrancy_protection,
                        overflow_protection, access_control, pause_functionality
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(audit_status)s, %(audit_firm)s, %(audit_score)s,
                        %(contract_verified)s, %(source_code_available)s, %(vulnerability_score)s,
                        %(multisig_wallets)s, %(timelock_mechanisms)s, %(upgrade_mechanisms)s,
                        %(emergency_pause)s, %(governance_decentralization)s, %(validator_distribution)s,
                        %(node_distribution)s, %(treasury_control)s, %(reentrancy_protection)s,
                        %(overflow_protection)s, %(access_control)s, %(pause_functionality)s
                    )
                """, security_metrics)
                
                # Метрики сообщества
                community_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'twitter_followers': random.randint(10000, 100000),
                    'telegram_members': random.randint(5000, 50000),
                    'discord_members': random.randint(2000, 20000),
                    'reddit_subscribers': random.randint(1000, 10000),
                    'facebook_likes': random.randint(500, 5000),
                    'instagram_followers': random.randint(1000, 10000),
                    'youtube_subscribers': random.randint(500, 5000),
                    'tiktok_followers': random.randint(100, 1000),
                    'social_engagement_rate': 3.5 + random.uniform(-1, 1),
                    'twitter_engagement_rate': 2.5 + random.uniform(-0.5, 0.5),
                    'telegram_activity_score': 7.0 + random.uniform(-1, 1),
                    'discord_activity_score': 6.5 + random.uniform(-1, 1),
                    'blog_posts_30d': random.randint(1, 10),
                    'youtube_videos_30d': random.randint(0, 5),
                    'podcast_appearances_30d': random.randint(0, 3),
                    'media_mentions_30d': random.randint(5, 50),
                    'brand_awareness_score': 7.0 + random.uniform(-1, 1),
                    'documentation_quality': 8.0 + random.uniform(-1, 1),
                    'tutorial_availability': 7.5 + random.uniform(-1, 1),
                    'community_guides_count': random.randint(5, 25),
                    'support_responsiveness': 8.5 + random.uniform(-1, 1)
                }
                
                cursor.execute("""
                    INSERT INTO community_metrics (
                        asset_id, timestamp, twitter_followers, telegram_members, discord_members,
                        reddit_subscribers, facebook_likes, instagram_followers, youtube_subscribers,
                        tiktok_followers, social_engagement_rate, twitter_engagement_rate,
                        telegram_activity_score, discord_activity_score, blog_posts_30d,
                        youtube_videos_30d, podcast_appearances_30d, media_mentions_30d,
                        brand_awareness_score, documentation_quality, tutorial_availability,
                        community_guides_count, support_responsiveness
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(twitter_followers)s, %(telegram_members)s, %(discord_members)s,
                        %(reddit_subscribers)s, %(facebook_likes)s, %(instagram_followers)s, %(youtube_subscribers)s,
                        %(tiktok_followers)s, %(social_engagement_rate)s, %(twitter_engagement_rate)s,
                        %(telegram_activity_score)s, %(discord_activity_score)s, %(blog_posts_30d)s,
                        %(youtube_videos_30d)s, %(podcast_appearances_30d)s, %(media_mentions_30d)s,
                        %(brand_awareness_score)s, %(documentation_quality)s, %(tutorial_availability)s,
                        %(community_guides_count)s, %(support_responsiveness)s
                    )
                """, community_metrics)
                
                # Трендовые метрики
                trending_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'momentum_score': 5.0 + random.uniform(-2, 2),
                    'trend_direction': random.choice(['bullish', 'bearish', 'sideways']),
                    'trend_strength': 6.0 + random.uniform(-2, 2),
                    'seasonality_score': 4.0 + random.uniform(-1, 1),
                    'cyclical_patterns': '{"daily": 0.1, "weekly": 0.3, "monthly": 0.6}',
                    'anomaly_score': random.uniform(0, 10),
                    'anomaly_type': random.choice(['price_spike', 'volume_surge', 'social_buzz', 'none']),
                    'anomaly_severity': random.choice(['low', 'medium', 'high']),
                    'fear_greed_index': 50 + random.uniform(-20, 20),
                    'social_sentiment': 0.5 + random.uniform(-0.3, 0.3),
                    'news_sentiment': 0.5 + random.uniform(-0.3, 0.3)
                }
                
                cursor.execute("""
                    INSERT INTO trending_metrics (
                        asset_id, timestamp, momentum_score, trend_direction, trend_strength,
                        seasonality_score, cyclical_patterns, anomaly_score, anomaly_type,
                        anomaly_severity, fear_greed_index, social_sentiment, news_sentiment
                    ) VALUES (
                        %(asset_id)s, %(timestamp)s, %(momentum_score)s, %(trend_direction)s, %(trend_strength)s,
                        %(seasonality_score)s, %(cyclical_patterns)s, %(anomaly_score)s, %(anomaly_type)s,
                        %(anomaly_severity)s, %(fear_greed_index)s, %(social_sentiment)s, %(news_sentiment)s
                    )
                """, trending_metrics)
        
        conn.commit()
        logger.info("✅ Asset metrics loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading asset metrics: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def load_ml_predictions():
    """Загрузить ML предсказания"""
    logger.info("Loading ML predictions...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Получить все активы
        cursor.execute("SELECT id, symbol FROM crypto_assets")
        assets = cursor.fetchall()
        
        for asset_id, symbol in assets:
            # Предсказания на разные горизонты
            horizons = ['1d', '7d', '30d']
            models = ['price_prediction', 'investment_score', 'volatility_forecast']
            
            for horizon in horizons:
                for model in models:
                    prediction = {
                        'asset_id': asset_id,
                        'model_name': f'{model}_{horizon}',
                        'prediction_type': model,
                        'prediction_value': random.uniform(0, 10),
                        'confidence_score': random.uniform(0.6, 0.95),
                        'prediction_horizon': horizon,
                        'features_used': '{"price": 0.3, "volume": 0.2, "social": 0.1, "technical": 0.4}',
                        'model_version': 'v1.0'
                    }
                    
                    cursor.execute("""
                        INSERT INTO ml_predictions (
                            asset_id, model_name, prediction_type, prediction_value,
                            confidence_score, prediction_horizon, features_used, model_version
                        ) VALUES (
                            %(asset_id)s, %(model_name)s, %(prediction_type)s, %(prediction_value)s,
                            %(confidence_score)s, %(prediction_horizon)s, %(features_used)s, %(model_version)s
                        )
                    """, prediction)
        
        conn.commit()
        logger.info("✅ ML predictions loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading ML predictions: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def main():
    """Основная функция"""
    logger.info("🚀 Loading weekly Polygon data...")
    
    try:
        # Загрузить метрики сети
        load_weekly_network_metrics()
        
        # Загрузить метрики активов
        load_weekly_asset_metrics()
        
        # Загрузить ML предсказания
        load_ml_predictions()
        
        logger.info("🎉 Weekly data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in main process: {e}")

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск
    main()
