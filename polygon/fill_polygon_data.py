#!/usr/bin/env python3
"""
Скрипт для заполнения базы данных данными по Polygon за одну неделю
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    Blockchain, CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics
)
from api.data_loader import DataLoader, DataCollectionRequest
from api.metrics_mapper import MetricsMapper
from config.settings import settings

async def setup_polygon_blockchain():
    """Создать запись о Polygon блокчейне"""
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
            rpc_url=settings.QUICKNODE_HTTP_ENDPOINT,
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

async def setup_polygon_assets(blockchain_id: int):
    """Создать основные активы Polygon"""
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
                "symbol": "WBTC",
                "name": "Wrapped Bitcoin",
                "contract_address": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6",
                "category": "Wrapped",
                "coingecko_id": "wrapped-bitcoin",
                "website": "https://wbtc.network",
                "description": "Wrapped Bitcoin token"
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
            },
            {
                "symbol": "SUSHI",
                "name": "SushiToken",
                "contract_address": "0x0b3f868e0be5597d5db7feb59e1cadb0c3da0f9c",
                "category": "DeFi",
                "coingecko_id": "sushi",
                "github_repo": "sushiswap/sushiswap",
                "website": "https://sushi.com",
                "description": "SushiSwap protocol token"
            },
            {
                "symbol": "QUICK",
                "name": "Quickswap",
                "contract_address": "0x831753dd7087cac61ab5644b308642cc1c33dc13",
                "category": "DeFi",
                "coingecko_id": "quick",
                "github_repo": "QuickSwap/QuickSwap-core",
                "website": "https://quickswap.exchange",
                "description": "QuickSwap DEX token"
            },
            {
                "symbol": "BAL",
                "name": "Balancer",
                "contract_address": "0x9a71012b13ca4d3d0cdc72a177df3ef03b0e76a3",
                "category": "DeFi",
                "coingecko_id": "balancer",
                "github_repo": "balancer-labs/balancer-v2-monorepo",
                "website": "https://balancer.fi",
                "description": "Balancer protocol token"
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

async def collect_polygon_data():
    """Собрать данные по Polygon за одну неделю"""
    logger.info("Starting Polygon data collection...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("Database initialized")
        
        # Создать Polygon блокчейн
        blockchain_id = await setup_polygon_blockchain()
        if not blockchain_id:
            logger.error("Failed to create Polygon blockchain")
            return
        
        # Создать активы Polygon
        assets = await setup_polygon_assets(blockchain_id)
        if not assets:
            logger.error("Failed to create Polygon assets")
            return
        
        # Собрать данные для каждого актива
        async with DataLoader() as data_loader:
            for asset in assets:
                try:
                    logger.info(f"Collecting data for {asset.symbol}...")
                    
                    # Создать запрос на сбор данных за 1 неделю
                    request = DataCollectionRequest(
                        asset_id=asset.id,
                        time_periods=["1w"],
                        force_refresh=True
                    )
                    
                    # Собрать данные
                    results = await data_loader.collect_data_for_asset(request)
                    
                    # Логировать результаты
                    for result in results:
                        if result.success:
                            logger.info(f"✅ Successfully collected {result.data_points} data points for {asset.symbol} - {result.time_period}")
                        else:
                            logger.error(f"❌ Failed to collect data for {asset.symbol} - {result.time_period}: {result.errors}")
                    
                    # Небольшая задержка между активами
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {asset.symbol}: {e}")
                    continue
        
        logger.info("🎉 Polygon data collection completed!")
        
    except Exception as e:
        logger.error(f"Error in Polygon data collection: {e}")

async def generate_sample_metrics():
    """Генерировать примеры метрик для демонстрации"""
    db = SessionLocal()
    try:
        # Получить активы Polygon
        polygon_assets = db.query(CryptoAsset).join(Blockchain).filter(
            Blockchain.name == "Polygon"
        ).all()
        
        if not polygon_assets:
            logger.warning("No Polygon assets found")
            return
        
        logger.info(f"Generating sample metrics for {len(polygon_assets)} assets...")
        
        # Генерировать примеры метрик для каждого актива
        for asset in polygon_assets:
            try:
                # On-chain метрики
                onchain_metrics = OnChainMetrics(
                    asset_id=asset.id,
                    tvl=1000000 + (hash(asset.symbol) % 1000000),  # Примерное значение
                    tvl_change_24h=5.2 + (hash(asset.symbol) % 10 - 5),
                    tvl_change_7d=12.8 + (hash(asset.symbol) % 20 - 10),
                    daily_transactions=1000 + (hash(asset.symbol) % 5000),
                    active_addresses_24h=500 + (hash(asset.symbol) % 2000),
                    transaction_volume_24h=500000 + (hash(asset.symbol) % 2000000),
                    gas_price_avg=30.5 + (hash(asset.symbol) % 20),
                    contract_interactions_24h=100 + (hash(asset.symbol) % 500)
                )
                db.add(onchain_metrics)
                
                # Финансовые метрики
                financial_metrics = FinancialMetrics(
                    asset_id=asset.id,
                    price_usd=1.0 + (hash(asset.symbol) % 100) / 100,
                    market_cap=1000000000 + (hash(asset.symbol) % 10000000000),
                    volume_24h=10000000 + (hash(asset.symbol) % 100000000),
                    volatility_24h=2.5 + (hash(asset.symbol) % 5),
                    price_change_24h=1.2 + (hash(asset.symbol) % 10 - 5),
                    price_change_7d=5.8 + (hash(asset.symbol) % 20 - 10),
                    price_change_30d=15.3 + (hash(asset.symbol) % 30 - 15)
                )
                db.add(financial_metrics)
                
                # GitHub метрики (только для активов с GitHub репозиторием)
                if asset.github_repo:
                    github_metrics = GitHubMetrics(
                        asset_id=asset.id,
                        commits_24h=5 + (hash(asset.symbol) % 20),
                        commits_7d=25 + (hash(asset.symbol) % 100),
                        commits_30d=100 + (hash(asset.symbol) % 500),
                        active_contributors_30d=3 + (hash(asset.symbol) % 15),
                        stars=1000 + (hash(asset.symbol) % 10000),
                        forks=100 + (hash(asset.symbol) % 1000),
                        open_issues=10 + (hash(asset.symbol) % 50),
                        open_prs=5 + (hash(asset.symbol) % 25),
                        code_quality_score=7.5 + (hash(asset.symbol) % 2.5)
                    )
                    db.add(github_metrics)
                
                # Токеномика
                tokenomics_metrics = TokenomicsMetrics(
                    asset_id=asset.id,
                    circulating_supply=1000000000 + (hash(asset.symbol) % 10000000000),
                    total_supply=1000000000 + (hash(asset.symbol) % 10000000000),
                    max_supply=10000000000 + (hash(asset.symbol) % 100000000000),
                    inflation_rate=2.5 + (hash(asset.symbol) % 5),
                    burn_rate=0.1 + (hash(asset.symbol) % 2)
                )
                db.add(tokenomics_metrics)
                
                # Метрики безопасности
                security_metrics = SecurityMetrics(
                    asset_id=asset.id,
                    audit_status="audited" if hash(asset.symbol) % 2 == 0 else "unaudited",
                    audit_score=8.5 + (hash(asset.symbol) % 1.5),
                    contract_verified=True if hash(asset.symbol) % 3 != 0 else False,
                    vulnerability_score=2.0 + (hash(asset.symbol) % 3),
                    multisig_wallets=True if hash(asset.symbol) % 2 == 0 else False
                )
                db.add(security_metrics)
                
                # Метрики сообщества
                community_metrics = CommunityMetrics(
                    asset_id=asset.id,
                    twitter_followers=10000 + (hash(asset.symbol) % 100000),
                    telegram_members=5000 + (hash(asset.symbol) % 50000),
                    discord_members=2000 + (hash(asset.symbol) % 20000),
                    social_engagement_rate=3.5 + (hash(asset.symbol) % 2)
                )
                db.add(community_metrics)
                
                # Трендовые метрики
                trending_metrics = TrendingMetrics(
                    asset_id=asset.id,
                    momentum_score=5.0 + (hash(asset.symbol) % 5),
                    trend_direction="bullish" if hash(asset.symbol) % 3 == 0 else "bearish" if hash(asset.symbol) % 3 == 1 else "sideways",
                    fear_greed_index=50 + (hash(asset.symbol) % 50),
                    social_sentiment=0.5 + (hash(asset.symbol) % 1)
                )
                db.add(trending_metrics)
                
                logger.info(f"Generated sample metrics for {asset.symbol}")
                
            except Exception as e:
                logger.error(f"Error generating metrics for {asset.symbol}: {e}")
                continue
        
        db.commit()
        logger.info("✅ Sample metrics generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating sample metrics: {e}")
        db.rollback()
    finally:
        db.close()

async def main():
    """Основная функция"""
    logger.info("🚀 Starting Polygon data collection and database population...")
    
    try:
        # Собрать реальные данные
        await collect_polygon_data()
        
        # Генерировать примеры метрик для демонстрации
        await generate_sample_metrics()
        
        logger.info("🎉 Database population completed successfully!")
        
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
    asyncio.run(main())
