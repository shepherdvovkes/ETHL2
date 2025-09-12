#!/usr/bin/env python3
"""
Мониторинг Layer 2 сетей в реальном времени
Интеграция с существующей системой мониторинга
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text, func, desc, asc
from sqlalchemy.orm import sessionmaker

# Import models
from src.database.l2_models import (
    L2Network, L2PerformanceMetrics, L2EconomicMetrics, 
    L2SecurityMetrics, L2EcosystemMetrics, L2ComparisonMetrics,
    L2RiskAssessment, L2TrendingMetrics, L2CrossChainMetrics
)
from src.database.models_v2 import Blockchain

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'defimon_db',
    'user': 'defimon',
    'password': 'password'
}

class L2RealTimeMonitor:
    """Мониторинг L2 сетей в реальном времени"""
    
    def __init__(self):
        self.session = self.get_sqlalchemy_session()
        self.monitoring_active = False
        self.alerts = []
        self.metrics_history = {}
    
    def get_sqlalchemy_session(self):
        """Get SQLAlchemy session"""
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        Session = sessionmaker(bind=engine)
        return Session()
    
    def check_tvl_anomalies(self) -> List[Dict]:
        """Проверить аномалии в TVL"""
        alerts = []
        
        try:
            # Получить текущие и предыдущие метрики TVL
            current_metrics = self.session.query(
                L2Network.name,
                L2EconomicMetrics.total_value_locked,
                L2EconomicMetrics.tvl_change_24h
            ).join(L2EconomicMetrics).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            for name, tvl, change_24h in current_metrics:
                if tvl and change_24h:
                    # Проверить на резкие изменения
                    if abs(change_24h) > 20:  # Изменение более 20%
                        alert_type = "TVL_SPIKE" if change_24h > 0 else "TVL_DROP"
                        severity = "HIGH" if abs(change_24h) > 50 else "MEDIUM"
                        
                        alerts.append({
                            "type": alert_type,
                            "severity": severity,
                            "network": name,
                            "value": float(tvl),
                            "change": change_24h,
                            "message": f"TVL {change_24h:+.1f}% in {name}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking TVL anomalies: {e}")
            return []
    
    def check_performance_issues(self) -> List[Dict]:
        """Проверить проблемы с производительностью"""
        alerts = []
        
        try:
            # Получить метрики производительности
            perf_metrics = self.session.query(
                L2Network.name,
                L2PerformanceMetrics.transactions_per_second,
                L2PerformanceMetrics.gas_fee_reduction,
                L2PerformanceMetrics.finality_time
            ).join(L2PerformanceMetrics).filter(
                L2PerformanceMetrics.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            for name, tps, fee_reduction, finality in perf_metrics:
                # Проверить низкую производительность
                if tps and tps < 100:  # Менее 100 TPS
                    alerts.append({
                        "type": "LOW_TPS",
                        "severity": "MEDIUM",
                        "network": name,
                        "value": tps,
                        "message": f"Low TPS detected in {name}: {tps}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Проверить высокие комиссии
                if fee_reduction and fee_reduction < 50:  # Снижение комиссий менее 50%
                    alerts.append({
                        "type": "HIGH_FEES",
                        "severity": "LOW",
                        "network": name,
                        "value": fee_reduction,
                        "message": f"High fees in {name}: only {fee_reduction:.1f}% reduction",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking performance issues: {e}")
            return []
    
    def check_security_risks(self) -> List[Dict]:
        """Проверить риски безопасности"""
        alerts = []
        
        try:
            # Получить метрики безопасности
            security_metrics = self.session.query(
                L2Network.name,
                L2RiskAssessment.overall_risk_score,
                L2RiskAssessment.risk_level,
                L2SecurityMetrics.validator_count,
                L2SecurityMetrics.audit_count
            ).join(L2RiskAssessment).join(L2SecurityMetrics).filter(
                L2RiskAssessment.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            for name, risk_score, risk_level, validators, audits in security_metrics:
                # Проверить высокий риск
                if risk_score and risk_score > 8.0:
                    alerts.append({
                        "type": "HIGH_RISK",
                        "severity": "HIGH",
                        "network": name,
                        "value": risk_score,
                        "message": f"High risk detected in {name}: {risk_score:.1f}/10",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Проверить централизацию
                if validators and validators == 1:
                    alerts.append({
                        "type": "CENTRALIZATION",
                        "severity": "MEDIUM",
                        "network": name,
                        "value": validators,
                        "message": f"Centralization risk in {name}: only {validators} validator",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Проверить недостаток аудитов
                if audits and audits < 2:
                    alerts.append({
                        "type": "LOW_AUDITS",
                        "severity": "MEDIUM",
                        "network": name,
                        "value": audits,
                        "message": f"Low audit count in {name}: {audits} audits",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking security risks: {e}")
            return []
    
    def check_ecosystem_health(self) -> List[Dict]:
        """Проверить здоровье экосистемы"""
        alerts = []
        
        try:
            # Получить метрики экосистемы
            ecosystem_metrics = self.session.query(
                L2Network.name,
                L2EcosystemMetrics.defi_protocols_count,
                L2EcosystemMetrics.nft_marketplaces,
                L2EcosystemMetrics.bridges_count
            ).join(L2EcosystemMetrics).filter(
                L2EcosystemMetrics.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            for name, defi_protocols, nft_marketplaces, bridges in ecosystem_metrics:
                # Проверить слабую экосистему
                total_protocols = (defi_protocols or 0) + (nft_marketplaces or 0) + (bridges or 0)
                if total_protocols < 5:
                    alerts.append({
                        "type": "WEAK_ECOSYSTEM",
                        "severity": "LOW",
                        "network": name,
                        "value": total_protocols,
                        "message": f"Weak ecosystem in {name}: only {total_protocols} protocols",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking ecosystem health: {e}")
            return []
    
    def check_trending_anomalies(self) -> List[Dict]:
        """Проверить аномалии в трендах"""
        alerts = []
        
        try:
            # Получить трендовые метрики
            trending_metrics = self.session.query(
                L2Network.name,
                L2TrendingMetrics.momentum_score,
                L2TrendingMetrics.trend_direction,
                L2TrendingMetrics.anomaly_score,
                L2TrendingMetrics.anomaly_type
            ).join(L2TrendingMetrics).filter(
                L2TrendingMetrics.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            for name, momentum, trend, anomaly_score, anomaly_type in trending_metrics:
                # Проверить аномалии
                if anomaly_score and anomaly_score > 5.0:
                    severity = "HIGH" if anomaly_score > 8.0 else "MEDIUM"
                    alerts.append({
                        "type": "TREND_ANOMALY",
                        "severity": severity,
                        "network": name,
                        "value": anomaly_score,
                        "message": f"Trend anomaly in {name}: {anomaly_type} (score: {anomaly_score:.1f})",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Проверить экстремальный momentum
                if momentum and (momentum > 9.0 or momentum < 1.0):
                    alerts.append({
                        "type": "EXTREME_MOMENTUM",
                        "severity": "MEDIUM",
                        "network": name,
                        "value": momentum,
                        "message": f"Extreme momentum in {name}: {momentum:.1f}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking trending anomalies: {e}")
            return []
    
    def run_health_checks(self) -> List[Dict]:
        """Запустить все проверки здоровья"""
        all_alerts = []
        
        logger.info("Running L2 health checks...")
        
        # Запустить все проверки
        all_alerts.extend(self.check_tvl_anomalies())
        all_alerts.extend(self.check_performance_issues())
        all_alerts.extend(self.check_security_risks())
        all_alerts.extend(self.check_ecosystem_health())
        all_alerts.extend(self.check_trending_anomalies())
        
        # Обновить историю алертов
        self.alerts.extend(all_alerts)
        
        # Ограничить историю последними 100 алертами
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return all_alerts
    
    def get_network_status(self, network_name: str) -> Dict:
        """Получить статус конкретной сети"""
        try:
            network = self.session.query(L2Network).filter_by(name=network_name).first()
            if not network:
                return {"status": "NOT_FOUND"}
            
            # Получить последние метрики
            latest_perf = self.session.query(L2PerformanceMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2PerformanceMetrics.timestamp)).first()
            
            latest_econ = self.session.query(L2EconomicMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2EconomicMetrics.timestamp)).first()
            
            latest_risk = self.session.query(L2RiskAssessment).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2RiskAssessment.timestamp)).first()
            
            # Определить общий статус
            status = "HEALTHY"
            issues = []
            
            if latest_risk and latest_risk.overall_risk_score > 7.0:
                status = "HIGH_RISK"
                issues.append("High risk score")
            
            if latest_econ and latest_econ.tvl_change_24h and abs(latest_econ.tvl_change_24h) > 20:
                status = "VOLATILE"
                issues.append("High TVL volatility")
            
            if latest_perf and latest_perf.transactions_per_second and latest_perf.transactions_per_second < 100:
                status = "PERFORMANCE_ISSUES"
                issues.append("Low TPS")
            
            return {
                "status": status,
                "issues": issues,
                "last_updated": datetime.utcnow().isoformat(),
                "performance": {
                    "tps": latest_perf.transactions_per_second if latest_perf else None,
                    "finality_time": latest_perf.finality_time if latest_perf else None
                },
                "economics": {
                    "tvl": float(latest_econ.total_value_locked) if latest_econ and latest_econ.total_value_locked else None,
                    "tvl_change_24h": latest_econ.tvl_change_24h if latest_econ else None
                },
                "risk": {
                    "overall_score": latest_risk.overall_risk_score if latest_risk else None,
                    "risk_level": latest_risk.risk_level if latest_risk else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting network status for {network_name}: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def get_all_networks_status(self) -> Dict:
        """Получить статус всех сетей"""
        try:
            networks = self.session.query(L2Network).all()
            status_summary = {
                "total_networks": len(networks),
                "healthy": 0,
                "high_risk": 0,
                "volatile": 0,
                "performance_issues": 0,
                "networks": {}
            }
            
            for network in networks:
                network_status = self.get_network_status(network.name)
                status_summary["networks"][network.name] = network_status
                
                # Подсчитать статусы
                if network_status["status"] == "HEALTHY":
                    status_summary["healthy"] += 1
                elif network_status["status"] == "HIGH_RISK":
                    status_summary["high_risk"] += 1
                elif network_status["status"] == "VOLATILE":
                    status_summary["volatile"] += 1
                elif network_status["status"] == "PERFORMANCE_ISSUES":
                    status_summary["performance_issues"] += 1
            
            return status_summary
        except Exception as e:
            logger.error(f"Error getting all networks status: {e}")
            return {}
    
    def print_monitoring_summary(self):
        """Вывести сводку мониторинга"""
        print("=" * 80)
        print("🔍 L2 NETWORKS REAL-TIME MONITORING")
        print("=" * 80)
        print(f"📅 Last Check: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Запустить проверки
        alerts = self.run_health_checks()
        
        # Статус всех сетей
        status_summary = self.get_all_networks_status()
        
        print("📊 NETWORK STATUS SUMMARY:")
        print(f"  Total Networks: {status_summary.get('total_networks', 0)}")
        print(f"  Healthy: {status_summary.get('healthy', 0)}")
        print(f"  High Risk: {status_summary.get('high_risk', 0)}")
        print(f"  Volatile: {status_summary.get('volatile', 0)}")
        print(f"  Performance Issues: {status_summary.get('performance_issues', 0)}")
        print()
        
        # Алерты
        if alerts:
            print("⚠️  ACTIVE ALERTS:")
            for alert in alerts:
                severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡" if alert["severity"] == "MEDIUM" else "🟢"
                print(f"  {severity_icon} [{alert['severity']}] {alert['message']}")
            print()
        else:
            print("✅ NO ACTIVE ALERTS")
            print()
        
        # Топ проблемных сетей
        problematic_networks = []
        for name, status in status_summary.get("networks", {}).items():
            if status["status"] != "HEALTHY":
                problematic_networks.append((name, status))
        
        if problematic_networks:
            print("🚨 PROBLEMATIC NETWORKS:")
            for name, status in problematic_networks[:5]:  # Показать топ-5
                print(f"  • {name}: {status['status']}")
                if status.get("issues"):
                    for issue in status["issues"]:
                        print(f"    - {issue}")
            print()
        
        print("=" * 80)
    
    def start_monitoring(self, interval: int = 300):  # 5 минут по умолчанию
        """Запустить мониторинг в реальном времени"""
        logger.info(f"Starting L2 real-time monitoring (interval: {interval}s)")
        self.monitoring_active = True
        
        try:
            while self.monitoring_active:
                # Вывести сводку
                self.print_monitoring_summary()
                
                # Экспортировать алерты
                if self.alerts:
                    self.export_alerts()
                
                # Ждать до следующей проверки
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.monitoring_active = False
    
    def export_alerts(self, filename: str = "l2_alerts.json"):
        """Экспортировать алерты в JSON"""
        alerts_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_alerts": len(self.alerts),
            "alerts": self.alerts
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(alerts_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Alerts exported to {filename}")
    
    def close(self):
        """Закрыть сессию"""
        self.session.close()

async def main():
    """Основная функция"""
    logger.info("🚀 Starting L2 real-time monitoring...")
    
    monitor = L2RealTimeMonitor()
    
    try:
        # Запустить мониторинг
        await monitor.start_monitoring(interval=300)  # 5 минут
        
    except Exception as e:
        logger.error(f"❌ Error in monitoring: {e}")
    finally:
        monitor.close()

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
