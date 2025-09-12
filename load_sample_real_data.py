#!/usr/bin/env python3
"""
Скрипт для загрузки примеров реальных данных
"""

import psycopg2
from datetime import datetime, timedelta
import random

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

def load_sample_real_data():
    """Загрузить примеры реальных данных"""
    print("🚀 Loading sample real data...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Получить активы
        cursor.execute("SELECT id, symbol, coingecko_id FROM crypto_assets WHERE coingecko_id IS NOT NULL")
        assets = cursor.fetchall()
        
        print(f"Found {len(assets)} assets")
        
        # Реальные данные для основных токенов (примерные, но основанные на реальных значениях)
        real_data = {
            'MATIC': {
                'price': 0.85,
                'market_cap': 8500000000,
                'volume_24h': 500000000,
                'price_change_24h': 2.5,
                'price_change_7d': -5.2
            },
            'USDC': {
                'price': 1.00,
                'market_cap': 32000000000,
                'volume_24h': 8000000000,
                'price_change_24h': 0.01,
                'price_change_7d': 0.02
            },
            'USDT': {
                'price': 1.00,
                'market_cap': 95000000000,
                'volume_24h': 25000000000,
                'price_change_24h': 0.01,
                'price_change_7d': 0.01
            },
            'WETH': {
                'price': 2500.0,
                'market_cap': 3000000000,
                'volume_24h': 2000000000,
                'price_change_24h': 1.8,
                'price_change_7d': 3.2
            },
            'WBTC': {
                'price': 45000.0,
                'market_cap': 8500000000,
                'volume_24h': 1500000000,
                'price_change_24h': 0.5,
                'price_change_7d': -2.1
            },
            'AAVE': {
                'price': 120.0,
                'market_cap': 1800000000,
                'volume_24h': 150000000,
                'price_change_24h': 3.2,
                'price_change_7d': 8.5
            },
            'CRV': {
                'price': 0.6,
                'market_cap': 500000000,
                'volume_24h': 80000000,
                'price_change_24h': -1.5,
                'price_change_7d': -12.3
            },
            'SUSHI': {
                'price': 1.2,
                'market_cap': 300000000,
                'volume_24h': 50000000,
                'price_change_24h': 2.1,
                'price_change_7d': 5.8
            },
            'QUICK': {
                'price': 0.8,
                'market_cap': 150000000,
                'volume_24h': 20000000,
                'price_change_24h': -0.8,
                'price_change_7d': -3.2
            },
            'BAL': {
                'price': 4.5,
                'market_cap': 250000000,
                'volume_24h': 30000000,
                'price_change_24h': 1.2,
                'price_change_7d': 2.5
            }
        }
        
        # Генерируем данные за последние 7 дней
        for day in range(7):
            timestamp = datetime.now() - timedelta(days=day)
            
            for asset_id, symbol, coingecko_id in assets:
                if symbol not in real_data:
                    continue
                
                base_data = real_data[symbol]
                
                # Добавить небольшую вариацию для каждого дня
                variation = 1 + (random.uniform(-0.05, 0.05) * day)
                
                price = base_data['price'] * variation
                market_cap = base_data['market_cap'] * variation
                volume = base_data['volume_24h'] * variation
                price_change_24h = base_data['price_change_24h'] + random.uniform(-1, 1)
                price_change_7d = base_data['price_change_7d'] + random.uniform(-2, 2)
                
                # Финансовые метрики
                financial_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'price_usd': price,
                    'market_cap': market_cap,
                    'fully_diluted_valuation': market_cap * 1.2,
                    'market_cap_rank': random.randint(1, 100),
                    'market_cap_dominance': random.uniform(0.1, 5.0),
                    'volume_24h': volume,
                    'volume_7d': volume * 7,
                    'volume_change_24h': random.uniform(-20, 20),
                    'volume_market_cap_ratio': volume / market_cap if market_cap > 0 else 0,
                    'liquidity_score': 8.0 + random.uniform(-1, 1),
                    'volatility_24h': abs(price_change_24h),
                    'volatility_7d': abs(price_change_7d),
                    'volatility_30d': abs(price_change_7d) * 1.5,
                    'beta_coefficient': 1.0 + random.uniform(-0.5, 0.5),
                    'bid_ask_spread': 0.001 + random.uniform(-0.0005, 0.0005),
                    'order_book_depth': 100000 + random.randint(-10000, 10000),
                    'slippage_analysis': 0.1 + random.uniform(-0.05, 0.05),
                    'price_change_1h': price_change_24h / 24,
                    'price_change_24h': price_change_24h,
                    'price_change_7d': price_change_7d,
                    'price_change_30d': price_change_7d * 1.5,
                    'price_change_90d': price_change_7d * 2.0,
                    'price_change_1y': price_change_7d * 3.0,
                    'all_time_high': price * 2.0,
                    'all_time_low': price * 0.5,
                    'ath_date': timestamp - timedelta(days=30),
                    'atl_date': timestamp - timedelta(days=365)
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
                
                # On-chain метрики
                onchain_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'tvl': market_cap * 0.1,
                    'tvl_change_24h': price_change_24h * 0.8,
                    'tvl_change_7d': price_change_7d * 0.8,
                    'tvl_change_30d': price_change_7d * 1.2,
                    'daily_transactions': int(volume / price / 1000) if price > 0 else 0,
                    'transaction_volume_24h': volume,
                    'transaction_volume_7d': volume * 7,
                    'avg_transaction_fee': 0.01 + random.uniform(-0.005, 0.005),
                    'transaction_success_rate': 98.5 + random.uniform(-1, 1),
                    'gas_usage_efficiency': 85.0 + random.uniform(-5, 5),
                    'active_addresses_24h': int(volume / price / 10000) if price > 0 else 0,
                    'new_addresses_24h': int(volume / price / 100000) if price > 0 else 0,
                    'unique_users_7d': int(volume / price / 1000) if price > 0 else 0,
                    'user_retention_rate': 75.0 + random.uniform(-5, 5),
                    'whale_activity': 5.0 + random.uniform(-2, 2),
                    'new_contracts_deployed': random.randint(0, 5),
                    'contract_interactions_24h': int(volume / price / 1000) if price > 0 else 0,
                    'contract_complexity_score': 7.5 + random.uniform(-1, 1),
                    'liquidity_pools_count': 10 + random.randint(-2, 2),
                    'liquidity_pools_tvl': market_cap * 0.08,
                    'yield_farming_apy': 5.0 + random.uniform(-2, 2),
                    'lending_volume': volume * 0.3,
                    'borrowing_volume': volume * 0.2
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
                
                # Токеномика
                tokenomics_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'circulating_supply': market_cap / price if price > 0 else 0,
                    'total_supply': market_cap / price * 1.1 if price > 0 else 0,
                    'max_supply': market_cap / price * 1.2 if price > 0 else 0,
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
                
                # Трендовые метрики
                trending_metrics = {
                    'asset_id': asset_id,
                    'timestamp': timestamp,
                    'momentum_score': 5.0 + (price_change_24h / 10),
                    'trend_direction': 'bullish' if price_change_24h > 0 else 'bearish' if price_change_24h < -2 else 'sideways',
                    'trend_strength': abs(price_change_24h) / 10,
                    'seasonality_score': 4.0 + random.uniform(-1, 1),
                    'cyclical_patterns': '{"daily": 0.1, "weekly": 0.3, "monthly": 0.6}',
                    'anomaly_score': abs(price_change_24h) if abs(price_change_24h) > 10 else 0,
                    'anomaly_type': 'price_spike' if abs(price_change_24h) > 10 else 'none',
                    'anomaly_severity': 'high' if abs(price_change_24h) > 20 else 'medium' if abs(price_change_24h) > 10 else 'low',
                    'fear_greed_index': 50 + (price_change_24h * 2),
                    'social_sentiment': 0.5 + (price_change_24h / 100),
                    'news_sentiment': 0.5 + (price_change_24h / 100)
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
        print("✅ Sample real data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_sample_real_data()
