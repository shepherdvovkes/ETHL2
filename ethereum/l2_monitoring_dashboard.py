#!/usr/bin/env python3
"""
Дашборд для мониторинга Layer 2 сетей Ethereum
Интеграция с существующей системой аналитики
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

def get_sqlalchemy_session():
    """Get SQLAlchemy session"""
    engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    Session = sessionmaker(bind=engine)
    return Session()

class L2Dashboard:
    """Дашборд для мониторинга L2 сетей"""
    
    def __init__(self):
        self.session = get_sqlalchemy_session()
    
    def get_l2_overview(self) -> Dict:
        """Получить общий обзор L2 сетей"""
        try:
            # Общая статистика
            total_networks = self.session.query(L2Network).count()
            active_networks = self.session.query(L2Network).filter_by(status='Active').count()
            
            # Статистика по типам
            type_stats = self.session.query(
                L2Network.l2_type,
                func.count(L2Network.id).label('count')
            ).group_by(L2Network.l2_type).all()
            
            # Общий TVL
            total_tvl = self.session.query(
                func.sum(L2EconomicMetrics.total_value_locked)
            ).join(L2Network).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            # Топ-5 по TVL
            top_tvl = self.session.query(
                L2Network.name,
                L2EconomicMetrics.total_value_locked
            ).join(L2EconomicMetrics).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).order_by(desc(L2EconomicMetrics.total_value_locked)).limit(5).all()
            
            return {
                "total_networks": total_networks,
                "active_networks": active_networks,
                "type_distribution": {stat.l2_type: stat.count for stat in type_stats},
                "total_tvl": float(total_tvl),
                "top_tvl_networks": [
                    {"name": network.name, "tvl": float(tvl)} 
                    for network, tvl in top_tvl
                ]
            }
        except Exception as e:
            logger.error(f"Error getting L2 overview: {e}")
            return {}
    
    def get_performance_leaderboard(self) -> List[Dict]:
        """Получить рейтинг производительности"""
        try:
            # Получить последние метрики производительности
            latest_perf = self.session.query(
                L2Network.name,
                L2Network.l2_type,
                L2PerformanceMetrics.transactions_per_second,
                L2PerformanceMetrics.gas_fee_reduction,
                L2PerformanceMetrics.finality_time,
                L2PerformanceMetrics.withdrawal_time
            ).join(L2PerformanceMetrics).filter(
                L2PerformanceMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).order_by(desc(L2PerformanceMetrics.transactions_per_second)).all()
            
            leaderboard = []
            for i, (name, l2_type, tps, fee_reduction, finality, withdrawal) in enumerate(latest_perf, 1):
                leaderboard.append({
                    "rank": i,
                    "name": name,
                    "type": l2_type,
                    "tps": tps,
                    "fee_reduction": fee_reduction,
                    "finality_time": finality,
                    "withdrawal_time": withdrawal
                })
            
            return leaderboard
        except Exception as e:
            logger.error(f"Error getting performance leaderboard: {e}")
            return []
    
    def get_economic_metrics(self) -> Dict:
        """Получить экономические метрики"""
        try:
            # Общий TVL и изменения
            total_tvl = self.session.query(
                func.sum(L2EconomicMetrics.total_value_locked)
            ).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            total_tvl_7d_ago = self.session.query(
                func.sum(L2EconomicMetrics.total_value_locked)
            ).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=8),
                L2EconomicMetrics.timestamp < datetime.utcnow() - timedelta(days=7)
            ).scalar() or 0
            
            tvl_change_7d = ((total_tvl - total_tvl_7d_ago) / total_tvl_7d_ago * 100) if total_tvl_7d_ago > 0 else 0
            
            # Общий объем торгов
            total_volume = self.session.query(
                func.sum(L2EconomicMetrics.daily_volume)
            ).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            # Топ сети по росту TVL
            top_growth = self.session.query(
                L2Network.name,
                L2EconomicMetrics.tvl_change_24h
            ).join(L2EconomicMetrics).filter(
                L2EconomicMetrics.timestamp >= datetime.utcnow() - timedelta(days=1),
                L2EconomicMetrics.tvl_change_24h.isnot(None)
            ).order_by(desc(L2EconomicMetrics.tvl_change_24h)).limit(5).all()
            
            return {
                "total_tvl": float(total_tvl),
                "tvl_change_7d": tvl_change_7d,
                "total_volume_24h": float(total_volume),
                "top_growth_networks": [
                    {"name": name, "growth": float(growth)} 
                    for name, growth in top_growth
                ]
            }
        except Exception as e:
            logger.error(f"Error getting economic metrics: {e}")
            return {}
    
    def get_security_analysis(self) -> Dict:
        """Получить анализ безопасности"""
        try:
            # Статистика по аудитам
            audit_stats = self.session.query(
                func.avg(L2SecurityMetrics.audit_count).label('avg_audits'),
                func.count(L2SecurityMetrics.id).label('networks_with_audits')
            ).filter(
                L2SecurityMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).first()
            
            # Сети с высоким риском
            high_risk_networks = self.session.query(
                L2Network.name,
                L2RiskAssessment.overall_risk_score,
                L2RiskAssessment.risk_level
            ).join(L2RiskAssessment).filter(
                L2RiskAssessment.timestamp >= datetime.utcnow() - timedelta(days=1),
                L2RiskAssessment.overall_risk_score > 7.0
            ).order_by(desc(L2RiskAssessment.overall_risk_score)).all()
            
            # Статистика по децентрализации
            decentralization_stats = self.session.query(
                func.avg(L2SecurityMetrics.validator_count).label('avg_validators'),
                func.count(L2SecurityMetrics.id).filter(L2SecurityMetrics.validator_count == 1).label('centralized_networks')
            ).filter(
                L2SecurityMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).first()
            
            return {
                "avg_audits": float(audit_stats.avg_audits) if audit_stats.avg_audits else 0,
                "networks_with_audits": audit_stats.networks_with_audits,
                "high_risk_networks": [
                    {"name": name, "risk_score": float(score), "risk_level": level}
                    for name, score, level in high_risk_networks
                ],
                "avg_validators": float(decentralization_stats.avg_validators) if decentralization_stats.avg_validators else 0,
                "centralized_networks": decentralization_stats.centralized_networks
            }
        except Exception as e:
            logger.error(f"Error getting security analysis: {e}")
            return {}
    
    def get_ecosystem_health(self) -> Dict:
        """Получить здоровье экосистемы"""
        try:
            # Общая статистика экосистемы
            total_defi_protocols = self.session.query(
                func.sum(L2EcosystemMetrics.defi_protocols_count)
            ).filter(
                L2EcosystemMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            total_nft_marketplaces = self.session.query(
                func.sum(L2EcosystemMetrics.nft_marketplaces)
            ).filter(
                L2EcosystemMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            total_bridges = self.session.query(
                func.sum(L2EcosystemMetrics.bridges_count)
            ).filter(
                L2EcosystemMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar() or 0
            
            # Топ сети по количеству протоколов
            top_ecosystems = self.session.query(
                L2Network.name,
                L2EcosystemMetrics.defi_protocols_count,
                L2EcosystemMetrics.nft_marketplaces,
                L2EcosystemMetrics.bridges_count
            ).join(L2EcosystemMetrics).filter(
                L2EcosystemMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).order_by(desc(L2EcosystemMetrics.defi_protocols_count)).limit(5).all()
            
            return {
                "total_defi_protocols": total_defi_protocols,
                "total_nft_marketplaces": total_nft_marketplaces,
                "total_bridges": total_bridges,
                "top_ecosystems": [
                    {
                        "name": name,
                        "defi_protocols": defi_count,
                        "nft_marketplaces": nft_count,
                        "bridges": bridges_count
                    }
                    for name, defi_count, nft_count, bridges_count in top_ecosystems
                ]
            }
        except Exception as e:
            logger.error(f"Error getting ecosystem health: {e}")
            return {}
    
    def get_trending_analysis(self) -> Dict:
        """Получить анализ трендов"""
        try:
            # Общие тренды
            bullish_networks = self.session.query(L2Network.name).join(L2TrendingMetrics).filter(
                L2TrendingMetrics.timestamp >= datetime.utcnow() - timedelta(days=1),
                L2TrendingMetrics.trend_direction == 'bullish'
            ).count()
            
            bearish_networks = self.session.query(L2Network.name).join(L2TrendingMetrics).filter(
                L2TrendingMetrics.timestamp >= datetime.utcnow() - timedelta(days=1),
                L2TrendingMetrics.trend_direction == 'bearish'
            ).count()
            
            # Топ сети по momentum
            top_momentum = self.session.query(
                L2Network.name,
                L2TrendingMetrics.momentum_score,
                L2TrendingMetrics.trend_direction
            ).join(L2TrendingMetrics).filter(
                L2TrendingMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).order_by(desc(L2TrendingMetrics.momentum_score)).limit(5).all()
            
            # Средние показатели роста
            avg_growth = self.session.query(
                func.avg(L2TrendingMetrics.user_growth_rate).label('avg_user_growth'),
                func.avg(L2TrendingMetrics.volume_growth_rate).label('avg_volume_growth'),
                func.avg(L2TrendingMetrics.tvl_growth_rate).label('avg_tvl_growth')
            ).filter(
                L2TrendingMetrics.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).first()
            
            return {
                "bullish_networks": bullish_networks,
                "bearish_networks": bearish_networks,
                "top_momentum_networks": [
                    {"name": name, "momentum": float(momentum), "trend": trend}
                    for name, momentum, trend in top_momentum
                ],
                "avg_user_growth": float(avg_growth.avg_user_growth) if avg_growth.avg_user_growth else 0,
                "avg_volume_growth": float(avg_growth.avg_volume_growth) if avg_growth.avg_volume_growth else 0,
                "avg_tvl_growth": float(avg_growth.avg_tvl_growth) if avg_growth.avg_tvl_growth else 0
            }
        except Exception as e:
            logger.error(f"Error getting trending analysis: {e}")
            return {}
    
    def get_network_details(self, network_name: str) -> Dict:
        """Получить детальную информацию о сети"""
        try:
            network = self.session.query(L2Network).filter_by(name=network_name).first()
            if not network:
                return {}
            
            # Получить последние метрики
            latest_perf = self.session.query(L2PerformanceMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2PerformanceMetrics.timestamp)).first()
            
            latest_econ = self.session.query(L2EconomicMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2EconomicMetrics.timestamp)).first()
            
            latest_sec = self.session.query(L2SecurityMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2SecurityMetrics.timestamp)).first()
            
            latest_eco = self.session.query(L2EcosystemMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2EcosystemMetrics.timestamp)).first()
            
            latest_risk = self.session.query(L2RiskAssessment).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2RiskAssessment.timestamp)).first()
            
            latest_trend = self.session.query(L2TrendingMetrics).filter_by(
                l2_network_id=network.id
            ).order_by(desc(L2TrendingMetrics.timestamp)).first()
            
            return {
                "basic_info": {
                    "name": network.name,
                    "symbol": network.symbol,
                    "type": network.l2_type,
                    "security_model": network.security_model,
                    "launch_date": network.launch_date.isoformat() if network.launch_date else None,
                    "website": network.website,
                    "description": network.description,
                    "status": network.status
                },
                "performance": {
                    "tps": latest_perf.transactions_per_second if latest_perf else None,
                    "finality_time": latest_perf.finality_time if latest_perf else None,
                    "withdrawal_time": latest_perf.withdrawal_time if latest_perf else None,
                    "gas_fee_reduction": latest_perf.gas_fee_reduction if latest_perf else None,
                    "latency": latest_perf.latency if latest_perf else None
                },
                "economics": {
                    "tvl": float(latest_econ.total_value_locked) if latest_econ and latest_econ.total_value_locked else None,
                    "tvl_change_24h": latest_econ.tvl_change_24h if latest_econ else None,
                    "daily_volume": float(latest_econ.daily_volume) if latest_econ and latest_econ.daily_volume else None,
                    "active_users_24h": latest_econ.active_users_24h if latest_econ else None,
                    "market_cap": float(latest_econ.market_cap) if latest_econ and latest_econ.market_cap else None
                },
                "security": {
                    "validator_count": latest_sec.validator_count if latest_sec else None,
                    "audit_count": latest_sec.audit_count if latest_sec else None,
                    "security_score": latest_sec.security_score if latest_sec else None,
                    "multisig_required": latest_sec.multisig_required if latest_sec else None,
                    "bug_bounty_program": latest_sec.bug_bounty_program if latest_sec else None
                },
                "ecosystem": {
                    "defi_protocols": latest_eco.defi_protocols_count if latest_eco else None,
                    "nft_marketplaces": latest_eco.nft_marketplaces if latest_eco else None,
                    "bridges": latest_eco.bridges_count if latest_eco else None,
                    "wallets_support": latest_eco.wallets_support if latest_eco else None
                },
                "risk": {
                    "overall_risk_score": latest_risk.overall_risk_score if latest_risk else None,
                    "risk_level": latest_risk.risk_level if latest_risk else None,
                    "centralization_risk": latest_risk.centralization_risk if latest_risk else None,
                    "security_risk": latest_risk.security_risk if latest_risk else None,
                    "liquidity_risk": latest_risk.liquidity_risk if latest_risk else None
                },
                "trends": {
                    "momentum_score": latest_trend.momentum_score if latest_trend else None,
                    "trend_direction": latest_trend.trend_direction if latest_trend else None,
                    "trend_strength": latest_trend.trend_strength if latest_trend else None,
                    "user_growth_rate": latest_trend.user_growth_rate if latest_trend else None,
                    "volume_growth_rate": latest_trend.volume_growth_rate if latest_trend else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting network details for {network_name}: {e}")
            return {}
    
    def generate_dashboard_report(self) -> Dict:
        """Сгенерировать полный отчет дашборда"""
        logger.info("Generating L2 dashboard report...")
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overview": self.get_l2_overview(),
            "performance_leaderboard": self.get_performance_leaderboard(),
            "economic_metrics": self.get_economic_metrics(),
            "security_analysis": self.get_security_analysis(),
            "ecosystem_health": self.get_ecosystem_health(),
            "trending_analysis": self.get_trending_analysis()
        }
        
        return report
    
    def print_dashboard_summary(self):
        """Вывести краткую сводку дашборда"""
        report = self.generate_dashboard_report()
        
        print("=" * 80)
        print("🚀 LAYER 2 NETWORKS DASHBOARD")
        print("=" * 80)
        print(f"📅 Last Updated: {report['timestamp']}")
        print()
        
        # Обзор
        overview = report['overview']
        print("📊 OVERVIEW:")
        print(f"  Total Networks: {overview.get('total_networks', 0)}")
        print(f"  Active Networks: {overview.get('active_networks', 0)}")
        print(f"  Total TVL: ${overview.get('total_tvl', 0)/1e9:.2f}B")
        print()
        
        # Топ TVL
        print("💰 TOP NETWORKS BY TVL:")
        for i, network in enumerate(overview.get('top_tvl_networks', [])[:5], 1):
            print(f"  {i}. {network['name']}: ${network['tvl']/1e9:.2f}B")
        print()
        
        # Производительность
        print("⚡ PERFORMANCE LEADERBOARD:")
        for network in report['performance_leaderboard'][:5]:
            print(f"  {network['rank']}. {network['name']} ({network['type']})")
            print(f"     TPS: {network['tps']:,} | Fee Reduction: {network['fee_reduction']:.1f}%")
        print()
        
        # Экономика
        econ = report['economic_metrics']
        print("💹 ECONOMIC METRICS:")
        print(f"  Total TVL: ${econ.get('total_tvl', 0)/1e9:.2f}B")
        print(f"  TVL Change 7d: {econ.get('tvl_change_7d', 0):+.1f}%")
        print(f"  Total Volume 24h: ${econ.get('total_volume_24h', 0)/1e6:.1f}M")
        print()
        
        # Безопасность
        security = report['security_analysis']
        print("🔒 SECURITY ANALYSIS:")
        print(f"  Average Audits: {security.get('avg_audits', 0):.1f}")
        print(f"  High Risk Networks: {len(security.get('high_risk_networks', []))}")
        print(f"  Centralized Networks: {security.get('centralized_networks', 0)}")
        print()
        
        # Экосистема
        ecosystem = report['ecosystem_health']
        print("🌐 ECOSYSTEM HEALTH:")
        print(f"  Total DeFi Protocols: {ecosystem.get('total_defi_protocols', 0)}")
        print(f"  Total NFT Marketplaces: {ecosystem.get('total_nft_marketplaces', 0)}")
        print(f"  Total Bridges: {ecosystem.get('total_bridges', 0)}")
        print()
        
        # Тренды
        trends = report['trending_analysis']
        print("📈 TRENDING ANALYSIS:")
        print(f"  Bullish Networks: {trends.get('bullish_networks', 0)}")
        print(f"  Bearish Networks: {trends.get('bearish_networks', 0)}")
        print(f"  Avg User Growth: {trends.get('avg_user_growth', 0):.1f}%")
        print()
        
        print("=" * 80)
    
    def export_dashboard_data(self, filename: str = "l2_dashboard_data.json"):
        """Экспортировать данные дашборда в JSON"""
        report = self.generate_dashboard_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard data exported to {filename}")
    
    def close(self):
        """Закрыть сессию"""
        self.session.close()

async def main():
    """Основная функция"""
    logger.info("🚀 Starting L2 monitoring dashboard...")
    
    dashboard = L2Dashboard()
    
    try:
        # Вывести сводку
        dashboard.print_dashboard_summary()
        
        # Экспортировать данные
        dashboard.export_dashboard_data()
        
        logger.info("🎉 L2 dashboard completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in dashboard: {e}")
    finally:
        dashboard.close()

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
