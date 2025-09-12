#!/usr/bin/env python3
"""
Скрипт для заполнения базы данных демонстрационными данными по Polygon
"""

import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    Blockchain, CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics
)
from loguru import logger

def create_polygon_blockchain():
    """Создать Polygon блокчейн"""
    db = SessionLocal()
    try:
        # Проверить, существует ли уже Polygon
        existing = db.query(Blockchain).filter(Blockchain.name == "Polygon").first()
        if existing:
            logger.info(f"Polygon blockchain already exists with ID: {existing.id}")
            return existing.id
        
        # Создать Polygon блокчейн
        polygon = Blockchain(
            name="Polygon",
            symbol="MATIC",
            chain_id=137,
            blockchain_type="sidechain",
            rpc_url="https://polygon-rpc.com",
            explorer_url="https://polygonscan.com",
            native_token="MATIC",
            is_active=True,
            description="Polygon is a Layer 2 scaling solution for Ethereum"
        )
        
        db.add(polygon)
        db.commit()
        db.refresh(polygon)
        
        logger.info(f"Created Polygon blockchain with ID: {polygon.id}")
        return polygon.id
        
    except Exception as e:
        logger.error(f"Error creating Polygon blockchain: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def create_polygon_assets(blockchain_id: int):
    """Создать активы Polygon"""
    db = SessionLocal()
    try:
        # Основные токены Polygon
        polygon_assets = [
            {
                "symbol": "MATIC",
                "name": "Polygon",
                "contract_address": "0x0000000000000000000000000000000000001010",
                "category": "Layer2",
                "coingecko_id": "matic-network",
                "github_repo": "maticnetwork/polygon-sdk",
                "website": "https://polygon.technology",
                "description": "Native token of Polygon network"
            },
            {
                "symbol": "USDC",
                "name": "USD Coin",
                "contract_address": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                "category": "Stablecoin",
                "coingecko_id": "usd-coin",
                "website": "https://www.centre.io",
                "description": "USD-pegged stablecoin"
            },
            {
                "symbol": "USDT",
                "name": "Tether USD",
                "contract_address": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
                "category": "Stablecoin",
                "coingecko_id": "tether",
                "website": "https://tether.to",
                "description": "USD-pegged stablecoin"
            },
            {
                "symbol": "WETH",
                "name": "Wrapped Ether",
                "contract_address": "0x7ceb23fd6fc0ad59923861afc8967b5e6d6c4e",
                "category": "Wrapped",
                "coingecko_id": "weth",
                "website": "https://weth.io",
                "description": "Wrapped Ethereum token"
            },
            {
                "symbol": "AAVE",
                "name": "Aave Token",
                "contract_address": "0xd6df932a45c0f255f85145f286ea0b292b21c90b",
                "category": "DeFi",
                "coingecko_id": "aave",
                "github_repo": "aave/aave-v3-core",
                "website": "https://aave.com",
                "description": "Aave protocol governance token"
            },
            {
                "symbol": "CRV",
                "name": "Curve DAO Token",
                "contract_address": "0x172370d5cd63279efa6d502dab29171933a610af",
                "category": "DeFi",
                "coingecko_id": "curve-dao-token",
                "github_repo": "curvefi/curve-contract",
                "website": "https://curve.fi",
                "description": "Curve protocol governance token"
            }
        ]
        
        created_assets = []
        for asset_data in polygon_assets:
            # Проверить, существует ли уже актив
            existing = db.query(CryptoAsset).filter(
                CryptoAsset.symbol == asset_data["symbol"],
                CryptoAsset.blockchain_id == blockchain_id
            ).first()
            
            if existing:
                logger.info(f"Asset {asset_data['symbol']} already exists with ID: {existing.id}")
                created_assets.append(existing)
                continue
            
            # Создать новый актив
            asset = CryptoAsset(
                blockchain_id=blockchain_id,
                **asset_data
            )
            
            db.add(asset)
            created_assets.append(asset)
            logger.info(f"Created asset: {asset_data['symbol']} - {asset_data['name']}")
        
        db.commit()
        
        # Обновить ID для созданных активов
        for asset in created_assets:
            db.refresh(asset)
        
        logger.info(f"Created {len(created_assets)} Polygon assets")
        return created_assets
        
    except Exception as e:
        logger.error(f"Error creating Polygon assets: {e}")
        db.rollback()
        return []
    finally:
        db.close()

def generate_weekly_metrics(assets):
    """Генерировать метрики за неделю для каждого актива"""
    db = SessionLocal()
    try:
        logger.info(f"Generating weekly metrics for {len(assets)} assets...")
        
        # Генерировать данные за последние 7 дней
        end_date = datetime.utcnow()
        
        for asset in assets:
            logger.info(f"Generating metrics for {asset.symbol}...")
            
            # Генерировать данные за каждый день недели
            for day in range(7):
                current_date = end_date - timedelta(days=day)
                
                # Базовые значения для каждого актива
                base_values = {
                    "MATIC": {"price": 0.8, "tvl": 1000000, "volume": 50000000},
                    "USDC": {"price": 1.0, "tvl": 2000000, "volume": 100000000},
                    "USDT": {"price": 1.0, "tvl": 1500000, "volume": 80000000},
                    "WETH": {"price": 2000, "tvl": 500000, "volume": 30000000},
                    "AAVE": {"price": 100, "tvl": 300000, "volume": 20000000},
                    "CRV": {"price": 0.5, "tvl": 200000, "volume": 15000000}
                }
                
                base = base_values.get(asset.symbol, {"price": 1.0, "tvl": 100000, "volume": 1000000})
                
                # Добавить случайные колебания
                price_variation = random.uniform(0.95, 1.05)
                tvl_variation = random.uniform(0.9, 1.1)
                volume_variation = random.uniform(0.8, 1.2)
                
                # On-chain метрики
                onchain_metrics = OnChainMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    tvl=base["tvl"] * tvl_variation,
                    tvl_change_24h=random.uniform(-5, 5),
                    tvl_change_7d=random.uniform(-10, 10),
                    daily_transactions=random.randint(1000, 10000),
                    active_addresses_24h=random.randint(500, 5000),
                    transaction_volume_24h=base["volume"] * volume_variation,
                    gas_price_avg=random.uniform(20, 50),
                    contract_interactions_24h=random.randint(100, 1000)
                )
                db.add(onchain_metrics)
                
                # Финансовые метрики
                financial_metrics = FinancialMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    price_usd=base["price"] * price_variation,
                    market_cap=base["tvl"] * 10 * price_variation,
                    volume_24h=base["volume"] * volume_variation,
                    volatility_24h=random.uniform(1, 5),
                    price_change_24h=random.uniform(-5, 5),
                    price_change_7d=random.uniform(-15, 15),
                    price_change_30d=random.uniform(-30, 30)
                )
                db.add(financial_metrics)
                
                # GitHub метрики (только для активов с GitHub репозиторием)
                if asset.github_repo:
                    github_metrics = GitHubMetrics(
                        asset_id=asset.id,
                        timestamp=current_date,
                        commits_24h=random.randint(1, 20),
                        commits_7d=random.randint(10, 100),
                        commits_30d=random.randint(50, 500),
                        active_contributors_30d=random.randint(2, 20),
                        stars=random.randint(1000, 50000),
                        forks=random.randint(100, 5000),
                        open_issues=random.randint(5, 100),
                        open_prs=random.randint(2, 50),
                        code_quality_score=random.uniform(7, 9)
                    )
                    db.add(github_metrics)
                
                # Токеномика
                tokenomics_metrics = TokenomicsMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    circulating_supply=random.uniform(1000000000, 10000000000),
                    total_supply=random.uniform(1000000000, 10000000000),
                    max_supply=random.uniform(10000000000, 100000000000),
                    inflation_rate=random.uniform(1, 5),
                    burn_rate=random.uniform(0.1, 2)
                )
                db.add(tokenomics_metrics)
                
                # Метрики безопасности
                security_metrics = SecurityMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    audit_status="audited" if random.random() > 0.3 else "unaudited",
                    audit_score=random.uniform(7, 9.5),
                    contract_verified=random.random() > 0.2,
                    vulnerability_score=random.uniform(1, 3),
                    multisig_wallets=random.random() > 0.5
                )
                db.add(security_metrics)
                
                # Метрики сообщества
                community_metrics = CommunityMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    twitter_followers=random.randint(10000, 500000),
                    telegram_members=random.randint(5000, 100000),
                    discord_members=random.randint(2000, 50000),
                    social_engagement_rate=random.uniform(2, 5)
                )
                db.add(community_metrics)
                
                # Трендовые метрики
                trending_metrics = TrendingMetrics(
                    asset_id=asset.id,
                    timestamp=current_date,
                    momentum_score=random.uniform(3, 8),
                    trend_direction=random.choice(["bullish", "bearish", "sideways"]),
                    fear_greed_index=random.uniform(20, 80),
                    social_sentiment=random.uniform(0.3, 0.8)
                )
                db.add(trending_metrics)
        
        db.commit()
        logger.info("✅ Weekly metrics generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating weekly metrics: {e}")
        db.rollback()
    finally:
        db.close()

def generate_network_metrics(blockchain_id: int):
    """Генерировать метрики сети Polygon"""
    db = SessionLocal()
    try:
        logger.info("Generating Polygon network metrics...")
        
        # Генерировать данные за последние 7 дней
        end_date = datetime.utcnow()
        
        for day in range(7):
            current_date = end_date - timedelta(days=day)
            
            network_metrics = NetworkMetrics(
                blockchain_id=blockchain_id,
                timestamp=current_date,
                block_time_avg=random.uniform(1.8, 2.2),
                block_size_avg=random.uniform(20000, 50000),
                transaction_throughput=random.uniform(50, 100),
                network_utilization=random.uniform(60, 90),
                hash_rate=random.uniform(1000000, 2000000),
                validator_count=random.randint(100, 200),
                staking_ratio=random.uniform(20, 40),
                total_supply=random.uniform(10000000000, 10000000000),
                inflation_rate=random.uniform(1, 3),
                gas_price_avg=random.uniform(20, 50),
                gas_price_median=random.uniform(15, 45),
                gas_limit=random.uniform(30000000, 50000000),
                gas_used_avg=random.uniform(20000000, 40000000)
            )
            db.add(network_metrics)
        
        db.commit()
        logger.info("✅ Network metrics generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating network metrics: {e}")
        db.rollback()
    finally:
        db.close()

def main():
    """Основная функция"""
    logger.info("🚀 Starting Polygon demo data generation...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("Database initialized")
        
        # Создать Polygon блокчейн
        blockchain_id = create_polygon_blockchain()
        if not blockchain_id:
            logger.error("Failed to create Polygon blockchain")
            return
        
        # Создать активы Polygon
        assets = create_polygon_assets(blockchain_id)
        if not assets:
            logger.error("Failed to create Polygon assets")
            return
        
        # Генерировать метрики за неделю
        generate_weekly_metrics(assets)
        
        # Генерировать метрики сети
        generate_network_metrics(blockchain_id)
        
        logger.info("🎉 Demo data generation completed successfully!")
        
        # Показать статистику
        db = SessionLocal()
        try:
            total_assets = db.query(CryptoAsset).count()
            total_onchain = db.query(OnChainMetrics).count()
            total_financial = db.query(FinancialMetrics).count()
            total_github = db.query(GitHubMetrics).count()
            total_network = db.query(NetworkMetrics).count()
            
            logger.info(f"📊 Database Statistics:")
            logger.info(f"   - Assets: {total_assets}")
            logger.info(f"   - On-chain metrics: {total_onchain}")
            logger.info(f"   - Financial metrics: {total_financial}")
            logger.info(f"   - GitHub metrics: {total_github}")
            logger.info(f"   - Network metrics: {total_network}")
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"❌ Error in demo data generation: {e}")

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
