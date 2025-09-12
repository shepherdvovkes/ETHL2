#!/usr/bin/env python3
"""
Интеграционный скрипт для полной интеграции Layer 2 системы
Объединяет все компоненты: загрузку данных, дашборд и мониторинг
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our L2 components
from load_l2_networks_data import main as load_l2_data
from l2_monitoring_dashboard import L2Dashboard
from l2_real_time_monitor import L2RealTimeMonitor

class L2SystemIntegrator:
    """Интегратор системы Layer 2 мониторинга"""
    
    def __init__(self):
        self.dashboard = None
        self.monitor = None
    
    async def setup_system(self):
        """Настройка системы"""
        logger.info("🔧 Setting up L2 monitoring system...")
        
        try:
            # Инициализировать компоненты
            self.dashboard = L2Dashboard()
            self.monitor = L2RealTimeMonitor()
            
            logger.info("✅ L2 system setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up L2 system: {e}")
            return False
    
    async def load_initial_data(self):
        """Загрузить начальные данные"""
        logger.info("📊 Loading initial L2 data...")
        
        try:
            # Загрузить данные L2 сетей
            await load_l2_data()
            logger.info("✅ Initial L2 data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading initial L2 data: {e}")
            return False
    
    async def run_dashboard(self):
        """Запустить дашборд"""
        logger.info("📈 Running L2 dashboard...")
        
        try:
            if not self.dashboard:
                self.dashboard = L2Dashboard()
            
            # Сгенерировать отчет
            report = self.dashboard.generate_dashboard_report()
            
            # Вывести сводку
            self.dashboard.print_dashboard_summary()
            
            # Экспортировать данные
            self.dashboard.export_dashboard_data("l2_dashboard_report.json")
            
            logger.info("✅ L2 dashboard completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error running L2 dashboard: {e}")
            return False
    
    async def run_monitoring(self, duration: int = 3600):  # 1 час по умолчанию
        """Запустить мониторинг"""
        logger.info(f"🔍 Running L2 monitoring for {duration} seconds...")
        
        try:
            if not self.monitor:
                self.monitor = L2RealTimeMonitor()
            
            # Запустить мониторинг на указанное время
            monitoring_task = asyncio.create_task(
                self.monitor.start_monitoring(interval=300)  # 5 минут
            )
            
            # Ждать указанное время
            await asyncio.sleep(duration)
            
            # Остановить мониторинг
            self.monitor.monitoring_active = False
            monitoring_task.cancel()
            
            logger.info("✅ L2 monitoring completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error running L2 monitoring: {e}")
            return False
    
    async def run_health_check(self):
        """Запустить проверку здоровья системы"""
        logger.info("🏥 Running L2 system health check...")
        
        try:
            if not self.monitor:
                self.monitor = L2RealTimeMonitor()
            
            # Запустить проверки
            alerts = self.monitor.run_health_checks()
            
            # Получить статус всех сетей
            status_summary = self.monitor.get_all_networks_status()
            
            # Вывести результаты
            print("=" * 80)
            print("🏥 L2 SYSTEM HEALTH CHECK")
            print("=" * 80)
            print(f"📅 Check Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            print("📊 SYSTEM STATUS:")
            print(f"  Total Networks: {status_summary.get('total_networks', 0)}")
            print(f"  Healthy: {status_summary.get('healthy', 0)}")
            print(f"  High Risk: {status_summary.get('high_risk', 0)}")
            print(f"  Volatile: {status_summary.get('volatile', 0)}")
            print(f"  Performance Issues: {status_summary.get('performance_issues', 0)}")
            print()
            
            if alerts:
                print("⚠️  ACTIVE ALERTS:")
                for alert in alerts:
                    severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡" if alert["severity"] == "MEDIUM" else "🟢"
                    print(f"  {severity_icon} [{alert['severity']}] {alert['message']}")
                print()
            else:
                print("✅ NO ACTIVE ALERTS")
                print()
            
            # Экспортировать результаты
            health_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "status_summary": status_summary,
                "alerts": alerts,
                "health_score": self.calculate_health_score(status_summary, alerts)
            }
            
            with open("l2_health_check.json", 'w', encoding='utf-8') as f:
                json.dump(health_report, f, indent=2, ensure_ascii=False, default=str)
            
            print("=" * 80)
            logger.info("✅ L2 system health check completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error running L2 health check: {e}")
            return False
    
    def calculate_health_score(self, status_summary: Dict, alerts: List[Dict]) -> float:
        """Рассчитать общий балл здоровья системы"""
        try:
            total_networks = status_summary.get('total_networks', 1)
            healthy_networks = status_summary.get('healthy', 0)
            
            # Базовый балл на основе здоровых сетей
            base_score = (healthy_networks / total_networks) * 100
            
            # Штрафы за алерты
            high_risk_penalty = len([a for a in alerts if a['severity'] == 'HIGH']) * 10
            medium_risk_penalty = len([a for a in alerts if a['severity'] == 'MEDIUM']) * 5
            low_risk_penalty = len([a for a in alerts if a['severity'] == 'LOW']) * 2
            
            # Итоговый балл
            final_score = max(0, base_score - high_risk_penalty - medium_risk_penalty - low_risk_penalty)
            
            return round(final_score, 1)
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.0
    
    async def generate_comprehensive_report(self):
        """Сгенерировать комплексный отчет"""
        logger.info("📋 Generating comprehensive L2 report...")
        
        try:
            # Собрать данные из всех компонентов
            dashboard_report = None
            health_status = None
            
            if self.dashboard:
                dashboard_report = self.dashboard.generate_dashboard_report()
            
            if self.monitor:
                health_status = self.monitor.get_all_networks_status()
                alerts = self.monitor.run_health_checks()
            else:
                alerts = []
            
            # Создать комплексный отчет
            comprehensive_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_info": {
                    "version": "1.0.0",
                    "components": ["data_loader", "dashboard", "monitor"],
                    "status": "operational"
                },
                "dashboard": dashboard_report,
                "health_status": health_status,
                "alerts": alerts,
                "health_score": self.calculate_health_score(health_status or {}, alerts),
                "recommendations": self.generate_recommendations(health_status or {}, alerts)
            }
            
            # Экспортировать отчет
            with open("l2_comprehensive_report.json", 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("✅ Comprehensive L2 report generated successfully")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive report: {e}")
            return None
    
    def generate_recommendations(self, status_summary: Dict, alerts: List[Dict]) -> List[str]:
        """Сгенерировать рекомендации на основе анализа"""
        recommendations = []
        
        try:
            # Рекомендации на основе статуса сетей
            high_risk_count = status_summary.get('high_risk', 0)
            if high_risk_count > 0:
                recommendations.append(f"Monitor {high_risk_count} high-risk networks closely")
            
            volatile_count = status_summary.get('volatile', 0)
            if volatile_count > 0:
                recommendations.append(f"Investigate volatility in {volatile_count} networks")
            
            performance_issues = status_summary.get('performance_issues', 0)
            if performance_issues > 0:
                recommendations.append(f"Address performance issues in {performance_issues} networks")
            
            # Рекомендации на основе алертов
            high_severity_alerts = [a for a in alerts if a['severity'] == 'HIGH']
            if high_severity_alerts:
                recommendations.append(f"Immediate attention required for {len(high_severity_alerts)} high-severity alerts")
            
            centralization_alerts = [a for a in alerts if a['type'] == 'CENTRALIZATION']
            if centralization_alerts:
                recommendations.append("Consider decentralization strategies for affected networks")
            
            low_audit_alerts = [a for a in alerts if a['type'] == 'LOW_AUDITS']
            if low_audit_alerts:
                recommendations.append("Encourage more security audits for affected networks")
            
            # Общие рекомендации
            if not recommendations:
                recommendations.append("System is operating normally - continue regular monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Cleaning up L2 system resources...")
        
        try:
            if self.dashboard:
                self.dashboard.close()
            
            if self.monitor:
                self.monitor.close()
            
            logger.info("✅ L2 system cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="L2 System Integrator")
    parser.add_argument("--mode", choices=["setup", "dashboard", "monitor", "health", "full"], 
                       default="full", help="Operation mode")
    parser.add_argument("--duration", type=int, default=3600, 
                       help="Monitoring duration in seconds (default: 3600)")
    parser.add_argument("--load-data", action="store_true", 
                       help="Load initial L2 data")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting L2 System Integrator in {args.mode} mode...")
    
    integrator = L2SystemIntegrator()
    
    try:
        # Настройка системы
        if not await integrator.setup_system():
            return
        
        # Загрузка данных
        if args.load_data or args.mode in ["setup", "full"]:
            if not await integrator.load_initial_data():
                return
        
        # Выполнение операций в зависимости от режима
        if args.mode == "setup":
            logger.info("✅ L2 system setup completed")
            
        elif args.mode == "dashboard":
            await integrator.run_dashboard()
            
        elif args.mode == "monitor":
            await integrator.run_monitoring(args.duration)
            
        elif args.mode == "health":
            await integrator.run_health_check()
            
        elif args.mode == "full":
            # Полный цикл
            await integrator.run_dashboard()
            await integrator.run_health_check()
            await integrator.generate_comprehensive_report()
            
            # Запустить мониторинг на 10 минут
            await integrator.run_monitoring(600)
        
        logger.info("🎉 L2 System Integrator completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in L2 System Integrator: {e}")
    finally:
        integrator.cleanup()

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
