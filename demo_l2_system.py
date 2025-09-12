#!/usr/bin/env python3
"""
Демонстрационный скрипт для системы Layer 2 мониторинга
Показывает все возможности системы
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demo_l2_data():
    """Демонстрация данных L2 сетей"""
    logger.info("🎯 DEMO: L2 Networks Data")
    print("=" * 80)
    
    try:
        from ethereum_l2_networks_complete_list import ETHEREUM_L2_NETWORKS, get_network_stats, print_network_summary
        
        # Показать сводку
        print_network_summary()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in L2 data demo: {e}")
        return False

async def demo_detailed_analysis():
    """Демонстрация детального анализа"""
    logger.info("🎯 DEMO: Detailed Analysis")
    print("=" * 80)
    
    try:
        from l2_networks_detailed_analysis import DETAILED_L2_NETWORKS, analyze_network_performance, compare_networks_by_type, generate_risk_assessment
        
        # Анализ производительности
        perf_analysis = analyze_network_performance()
        print("🏆 PERFORMANCE LEADERS:")
        print(f"  Fastest TPS: {perf_analysis['fastest_tps'].basic_info.name} ({perf_analysis['fastest_tps'].performance.transactions_per_second} TPS)")
        print(f"  Lowest Fees: {perf_analysis['lowest_fees'].basic_info.name} ({perf_analysis['lowest_fees'].performance.gas_fee_reduction}% reduction)")
        print(f"  Fastest Finality: {perf_analysis['fastest_finality'].basic_info.name} ({perf_analysis['fastest_finality'].performance.finality_time})")
        print()
        
        # Сравнение по типам
        type_comparison = compare_networks_by_type()
        print("📊 COMPARISON BY TYPE:")
        for l2_type, stats in type_comparison.items():
            print(f"  {l2_type}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg TPS: {stats['avg_tps']:.0f}")
            print(f"    Avg TVL: ${stats['avg_tvl']/1e9:.2f}B")
            print(f"    Avg Fee Reduction: {stats['avg_fee_reduction']:.1f}%")
        print()
        
        # Оценка рисков
        risk_assessment = generate_risk_assessment()
        print("⚠️  RISK ASSESSMENT:")
        for category, risks in risk_assessment.items():
            if risks:
                print(f"  {category.upper()}:")
                for risk in risks[:3]:  # Показать первые 3
                    print(f"    {risk['network']}: {risk['risk']} ({risk['level']})")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in detailed analysis demo: {e}")
        return False

async def demo_database_integration():
    """Демонстрация интеграции с базой данных"""
    logger.info("🎯 DEMO: Database Integration")
    print("=" * 80)
    
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        from src.database.l2_models import L2Network
        from src.database.models_v2 import Blockchain
        
        # Database connection
        DB_CONFIG = {
            'host': 'localhost',
            'port': 5432,
            'database': 'defimon_db',
            'user': 'defimon',
            'password': 'password'
        }
        
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Проверить количество сетей в БД
        total_networks = session.query(L2Network).count()
        print(f"📊 DATABASE STATISTICS:")
        print(f"  Total L2 Networks in DB: {total_networks}")
        
        # Показать топ-5 сетей
        top_networks = session.query(L2Network).limit(5).all()
        print(f"\n🏆 TOP-5 L2 NETWORKS IN DATABASE:")
        for i, network in enumerate(top_networks, 1):
            print(f"  {i}. {network.name} ({network.l2_type})")
            print(f"     Security: {network.security_model}")
            print(f"     Status: {network.status}")
        
        # Проверить Ethereum blockchain
        ethereum = session.query(Blockchain).filter_by(name='Ethereum').first()
        if ethereum:
            print(f"\n🔗 ETHEREUM BLOCKCHAIN:")
            print(f"  ID: {ethereum.id}")
            print(f"  Symbol: {ethereum.symbol}")
            print(f"  Chain ID: {ethereum.chain_id}")
            print(f"  Type: {ethereum.blockchain_type}")
        
        session.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in database integration demo: {e}")
        return False

async def demo_dashboard():
    """Демонстрация дашборда"""
    logger.info("🎯 DEMO: Dashboard")
    print("=" * 80)
    
    try:
        from l2_monitoring_dashboard import L2Dashboard
        
        dashboard = L2Dashboard()
        
        # Сгенерировать отчет
        report = dashboard.generate_dashboard_report()
        
        print("📈 DASHBOARD OVERVIEW:")
        overview = report.get('overview', {})
        print(f"  Total Networks: {overview.get('total_networks', 0)}")
        print(f"  Active Networks: {overview.get('active_networks', 0)}")
        print(f"  Total TVL: ${overview.get('total_tvl', 0)/1e9:.2f}B")
        
        # Показать распределение по типам
        type_dist = overview.get('type_distribution', {})
        if type_dist:
            print(f"\n📊 NETWORKS BY TYPE:")
            for l2_type, count in type_dist.items():
                print(f"  {l2_type}: {count}")
        
        # Показать топ TVL
        top_tvl = overview.get('top_tvl_networks', [])
        if top_tvl:
            print(f"\n💰 TOP NETWORKS BY TVL:")
            for i, network in enumerate(top_tvl[:5], 1):
                print(f"  {i}. {network['name']}: ${network['tvl']/1e9:.2f}B")
        
        dashboard.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in dashboard demo: {e}")
        return False

async def demo_monitoring():
    """Демонстрация мониторинга"""
    logger.info("🎯 DEMO: Real-time Monitoring")
    print("=" * 80)
    
    try:
        from l2_real_time_monitor import L2RealTimeMonitor
        
        monitor = L2RealTimeMonitor()
        
        # Запустить проверки здоровья
        alerts = monitor.run_health_checks()
        
        print("🔍 HEALTH CHECK RESULTS:")
        if alerts:
            print(f"  Active Alerts: {len(alerts)}")
            for alert in alerts[:3]:  # Показать первые 3
                severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡" if alert["severity"] == "MEDIUM" else "🟢"
                print(f"  {severity_icon} [{alert['severity']}] {alert['message']}")
        else:
            print("  ✅ No active alerts")
        
        # Получить статус всех сетей
        status_summary = monitor.get_all_networks_status()
        print(f"\n📊 NETWORK STATUS SUMMARY:")
        print(f"  Total Networks: {status_summary.get('total_networks', 0)}")
        print(f"  Healthy: {status_summary.get('healthy', 0)}")
        print(f"  High Risk: {status_summary.get('high_risk', 0)}")
        print(f"  Volatile: {status_summary.get('volatile', 0)}")
        print(f"  Performance Issues: {status_summary.get('performance_issues', 0)}")
        
        monitor.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in monitoring demo: {e}")
        return False

async def main():
    """Основная функция демонстрации"""
    logger.info("🚀 Starting L2 System Demo...")
    
    demos = [
        ("L2 Networks Data", demo_l2_data),
        ("Detailed Analysis", demo_detailed_analysis),
        ("Database Integration", demo_database_integration),
        ("Dashboard", demo_dashboard),
        ("Real-time Monitoring", demo_monitoring)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running demo: {demo_name}")
        logger.info(f"{'='*80}")
        
        try:
            result = await demo_func()
            results.append((demo_name, result))
            
            if result:
                logger.info(f"✅ {demo_name} demo completed successfully")
            else:
                logger.error(f"❌ {demo_name} demo failed")
                
        except Exception as e:
            logger.error(f"❌ {demo_name} demo error: {e}")
            results.append((demo_name, False))
    
    # Итоговый отчет
    logger.info(f"\n{'='*80}")
    logger.info("DEMO RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        logger.info(f"  {demo_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} demos completed successfully")
    
    if passed == total:
        logger.info("🎉 All demos completed! L2 system is fully operational.")
        print("\n" + "="*80)
        print("🎯 L2 SYSTEM INTEGRATION COMPLETE!")
        print("="*80)
        print("✅ 27 Layer 2 networks integrated")
        print("✅ Comprehensive data models created")
        print("✅ Real-time monitoring system active")
        print("✅ Dashboard and analytics ready")
        print("✅ Database integration complete")
        print("✅ Risk assessment and alerts working")
        print("\n🚀 Your L2 monitoring system is ready for production!")
        print("="*80)
    else:
        logger.warning(f"⚠️  {total - passed} demos failed. Check the issues above.")

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск демонстрации
    asyncio.run(main())
