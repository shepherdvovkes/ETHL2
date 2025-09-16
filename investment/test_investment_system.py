#!/usr/bin/env python3
"""
Тестовый скрипт для проверки системы анализа инвестиционных метрик
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import init_db
from database.models_v2 import CryptoAsset
from collect_investment_metrics import InvestmentMetricsCollector

async def test_system():
    """Тестировать систему сбора метрик"""
    logger.info("🧪 Testing MATIC Investment Analysis System...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("✅ Database initialized")
        
        # Проверить наличие MATIC в базе
        from database.database import SessionLocal
        session = SessionLocal()
        
        matic_asset = session.query(CryptoAsset).filter(
            CryptoAsset.symbol == "MATIC"
        ).first()
        
        if not matic_asset:
            logger.warning("⚠️ MATIC asset not found in database")
            logger.info("💡 Run 'python fill_polygon_data.py' to populate the database")
            return False
        
        logger.info(f"✅ Found MATIC asset: {matic_asset.name} (ID: {matic_asset.id})")
        
        # Тестировать сбор метрик
        async with InvestmentMetricsCollector() as collector:
            logger.info("📊 Testing metrics collection...")
            
            # Собрать только финансовые метрики для теста
            try:
                financial_metrics = await collector._collect_financial_metrics(matic_asset)
                if financial_metrics:
                    logger.info("✅ Financial metrics collection successful")
                    logger.info(f"   Current price: ${financial_metrics.get('price_usd', 0):.4f}")
                    logger.info(f"   Market cap: ${financial_metrics.get('market_cap', 0):,.0f}")
                else:
                    logger.warning("⚠️ No financial metrics collected")
            except Exception as e:
                logger.error(f"❌ Financial metrics collection failed: {e}")
            
            # Тестировать GitHub метрики
            try:
                github_metrics = await collector._collect_github_metrics(matic_asset)
                if github_metrics:
                    logger.info("✅ GitHub metrics collection successful")
                    logger.info(f"   Stars: {github_metrics.get('stars', 0):,}")
                    logger.info(f"   Commits (30d): {github_metrics.get('commits_30d', 0)}")
                else:
                    logger.warning("⚠️ No GitHub metrics collected")
            except Exception as e:
                logger.error(f"❌ GitHub metrics collection failed: {e}")
        
        session.close()
        
        logger.info("🎉 System test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

async def main():
    """Основная функция тестирования"""
    logger.info("🚀 MATIC Investment System Test")
    logger.info("=" * 40)
    
    success = await test_system()
    
    if success:
        logger.info("✅ All tests passed! System is ready to use.")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("   1. Run 'python run_investment_analysis.py' for full analysis")
        logger.info("   2. Open 'matic_investment_dashboard.html' to view results")
        logger.info("   3. Check 'matic_investment_report.json' for detailed analysis")
    else:
        logger.error("❌ Tests failed! Please check the configuration and try again.")
        logger.info("")
        logger.info("🔧 Troubleshooting:")
        logger.info("   1. Check API keys in .env file")
        logger.info("   2. Ensure database is initialized")
        logger.info("   3. Run 'python fill_polygon_data.py' to populate data")

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск тестов
    asyncio.run(main())
