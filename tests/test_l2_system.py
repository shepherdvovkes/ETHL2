#!/usr/bin/env python3
"""
Тестовый скрипт для проверки системы Layer 2 мониторинга
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_l2_data():
    """Тест данных L2 сетей"""
    logger.info("🧪 Testing L2 data...")
    
    try:
        from ethereum_l2_networks_complete_list import ETHEREUM_L2_NETWORKS, get_network_stats
        
        # Проверить данные
        stats = get_network_stats()
        logger.info(f"✅ Found {stats['total_networks']} L2 networks")
        logger.info(f"✅ Total TVL: ${stats['total_tvl']/1e9:.2f}B")
        
        # Показать топ-5
        print("\n📊 TOP-5 L2 NETWORKS BY TVL:")
        sorted_networks = sorted([n for n in ETHEREUM_L2_NETWORKS if n.tvl_usd], 
                               key=lambda x: x.tvl_usd, reverse=True)[:5]
        
        for i, network in enumerate(sorted_networks, 1):
            print(f"  {i}. {network.name}: ${network.tvl_usd/1e9:.2f}B ({network.type.value})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing L2 data: {e}")
        return False

async def test_detailed_analysis():
    """Тест детального анализа"""
    logger.info("🧪 Testing detailed analysis...")
    
    try:
        from l2_networks_detailed_analysis import DETAILED_L2_NETWORKS, analyze_network_performance
        
        # Проверить детальный анализ
        perf_analysis = analyze_network_performance()
        logger.info(f"✅ Fastest TPS: {perf_analysis['fastest_tps'].basic_info.name}")
        logger.info(f"✅ Lowest fees: {perf_analysis['lowest_fees'].basic_info.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing detailed analysis: {e}")
        return False

async def test_database_connection():
    """Тест подключения к базе данных"""
    logger.info("🧪 Testing database connection...")
    
    try:
        import psycopg2
        
        DB_CONFIG = {
            'host': 'localhost',
            'port': 5432,
            'database': 'defimon_db',
            'user': 'defimon',
            'password': 'password'
        }
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Проверить подключение
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✅ Connected to PostgreSQL: {version[0]}")
        
        # Проверить существующие таблицы
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        logger.info(f"✅ Found {len(tables)} tables in database")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing database connection: {e}")
        return False

async def test_models():
    """Тест моделей данных"""
    logger.info("🧪 Testing data models...")
    
    try:
        from src.database.l2_models import L2Network, L2Type, SecurityModel
        
        # Создать тестовую модель
        test_network = L2Network(
            name="Test Network",
            l2_type=L2Type.OPTIMISTIC_ROLLUP.value,
            security_model=SecurityModel.ETHEREUM_SECURITY.value,
            parent_blockchain_id=1
        )
        
        logger.info(f"✅ Created test L2 network: {test_network.name}")
        logger.info(f"✅ Type: {test_network.l2_type}")
        logger.info(f"✅ Security: {test_network.security_model}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing models: {e}")
        return False

async def main():
    """Основная функция тестирования"""
    logger.info("🚀 Starting L2 system tests...")
    
    tests = [
        ("L2 Data", test_l2_data),
        ("Detailed Analysis", test_detailed_analysis),
        ("Database Connection", test_database_connection),
        ("Data Models", test_models)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    logger.info(f"\n{'='*50}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! L2 system is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the issues above.")

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
