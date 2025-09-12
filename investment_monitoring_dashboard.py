#!/usr/bin/env python3
"""
Дашборд для мониторинга инвестиционных метрик MATIC
Отображает ключевые показатели и сигналы для принятия решений
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, TrendingMetrics, MLPrediction
)
from api.metrics_mapper import MetricsMapper

class InvestmentMonitoringDashboard:
    """Дашборд для мониторинга инвестиционных метрик"""
    
    def __init__(self):
        self.session = None
        self.metrics_mapper = MetricsMapper()
        
    async def __aenter__(self):
        self.session = SessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def get_matic_dashboard_data(self) -> Dict[str, Any]:
        """Получить данные для дашборда MATIC"""
        try:
            # Найти MATIC актив
            matic_asset = self.session.query(CryptoAsset).filter(
                CryptoAsset.symbol == "MATIC"
            ).first()
            
            if not matic_asset:
                logger.error("MATIC asset not found")
                return {}
            
            # Получить последние метрики
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == matic_asset.id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_onchain = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == matic_asset.id
            ).order_by(OnChainMetrics.timestamp.desc()).first()
            
            latest_github = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == matic_asset.id
            ).order_by(GitHubMetrics.timestamp.desc()).first()
            
            latest_security = self.session.query(SecurityMetrics).filter(
                SecurityMetrics.asset_id == matic_asset.id
            ).order_by(SecurityMetrics.timestamp.desc()).first()
            
            latest_ml = self.session.query(MLPrediction).filter(
                MLPrediction.asset_id == matic_asset.id
            ).order_by(MLPrediction.created_at.desc()).first()
            
            # Создать дашборд данные
            dashboard_data = {
                "asset_info": {
                    "symbol": matic_asset.symbol,
                    "name": matic_asset.name,
                    "last_updated": datetime.utcnow().isoformat()
                },
                "financial_overview": self._format_financial_data(latest_financial),
                "onchain_overview": self._format_onchain_data(latest_onchain),
                "development_overview": self._format_github_data(latest_github),
                "security_overview": self._format_security_data(latest_security),
                "ml_predictions": self._format_ml_data(latest_ml),
                "investment_signals": await self._generate_investment_signals(matic_asset.id),
                "risk_assessment": await self._assess_risks(matic_asset.id),
                "recommendations": await self._generate_recommendations(matic_asset.id)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _format_financial_data(self, financial: Optional[FinancialMetrics]) -> Dict[str, Any]:
        """Форматировать финансовые данные"""
        if not financial:
            return {"status": "no_data"}
        
        return {
            "current_price": financial.price_usd,
            "market_cap": financial.market_cap,
            "volume_24h": financial.volume_24h,
            "price_changes": {
                "1h": financial.price_change_1h,
                "24h": financial.price_change_24h,
                "7d": financial.price_change_7d,
                "30d": financial.price_change_30d
            },
            "volatility": {
                "24h": financial.volatility_24h,
                "7d": financial.volatility_7d,
                "30d": financial.volatility_30d
            },
            "supply": {
                "circulating": financial.circulating_supply,
                "total": financial.total_supply,
                "max": financial.max_supply
            },
            "status": "active",
            "last_updated": financial.timestamp.isoformat() if financial.timestamp else None
        }
    
    def _format_onchain_data(self, onchain: Optional[OnChainMetrics]) -> Dict[str, Any]:
        """Форматировать on-chain данные"""
        if not onchain:
            return {"status": "no_data"}
        
        return {
            "tvl": onchain.tvl,
            "tvl_changes": {
                "24h": onchain.tvl_change_24h,
                "7d": onchain.tvl_change_7d
            },
            "activity": {
                "daily_transactions": onchain.daily_transactions,
                "active_addresses_24h": onchain.active_addresses_24h,
                "transaction_volume_24h": onchain.transaction_volume_24h
            },
            "network": {
                "gas_price_avg": onchain.gas_price_avg,
                "block_time_avg": onchain.block_time_avg,
                "network_utilization": onchain.network_utilization
            },
            "contracts": {
                "interactions_24h": onchain.contract_interactions_24h,
                "new_contracts_deployed": onchain.new_contracts_deployed
            },
            "status": "active",
            "last_updated": onchain.timestamp.isoformat() if onchain.timestamp else None
        }
    
    def _format_github_data(self, github: Optional[GitHubMetrics]) -> Dict[str, Any]:
        """Форматировать GitHub данные"""
        if not github:
            return {"status": "no_data"}
        
        return {
            "activity": {
                "commits_24h": github.commits_24h,
                "commits_7d": github.commits_7d,
                "commits_30d": github.commits_30d
            },
            "contributors": {
                "active_30d": github.active_contributors_30d,
                "total": github.total_contributors
            },
            "community": {
                "stars": github.stars,
                "forks": github.forks,
                "stars_change_7d": github.stars_change_7d
            },
            "development": {
                "open_issues": github.open_issues,
                "open_prs": github.open_prs,
                "merged_prs_7d": github.merged_prs_7d,
                "closed_issues_7d": github.closed_issues_7d
            },
            "quality": {
                "code_quality_score": github.code_quality_score,
                "primary_language": github.primary_language
            },
            "status": "active",
            "last_updated": github.timestamp.isoformat() if github.timestamp else None
        }
    
    def _format_security_data(self, security: Optional[SecurityMetrics]) -> Dict[str, Any]:
        """Форматировать данные безопасности"""
        if not security:
            return {"status": "no_data"}
        
        return {
            "audit": {
                "status": security.audit_status,
                "score": security.audit_score
            },
            "contract": {
                "verified": security.contract_verified,
                "vulnerability_score": security.vulnerability_score
            },
            "security_features": {
                "multisig_wallets": security.multisig_wallets,
                "timelock_contracts": security.timelock_contracts,
                "proxy_contracts": security.proxy_contracts,
                "upgradeable_contracts": security.upgradeable_contracts
            },
            "status": "active",
            "last_updated": security.timestamp.isoformat() if security.timestamp else None
        }
    
    def _format_ml_data(self, ml: Optional[MLPrediction]) -> Dict[str, Any]:
        """Форматировать ML данные"""
        if not ml:
            return {"status": "no_data"}
        
        return {
            "investment_score": ml.prediction_value,
            "confidence_score": ml.confidence_score,
            "model_name": ml.model_name,
            "prediction_horizon": ml.prediction_horizon,
            "features_used": ml.features_used,
            "model_version": ml.model_version,
            "status": "active",
            "last_updated": ml.created_at.isoformat() if ml.created_at else None
        }
    
    async def _generate_investment_signals(self, asset_id: int) -> Dict[str, Any]:
        """Генерировать инвестиционные сигналы"""
        try:
            signals = {
                "buy_signals": [],
                "sell_signals": [],
                "hold_signals": [],
                "warning_signals": []
            }
            
            # Получить последние метрики
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == asset_id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_onchain = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == asset_id
            ).order_by(OnChainMetrics.timestamp.desc()).first()
            
            latest_github = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == asset_id
            ).order_by(GitHubMetrics.timestamp.desc()).first()
            
            # Анализ финансовых сигналов
            if latest_financial:
                if latest_financial.price_change_7d > 15:
                    signals["buy_signals"].append({
                        "type": "price_momentum",
                        "message": f"Strong 7-day price increase: {latest_financial.price_change_7d:.2f}%",
                        "strength": "high"
                    })
                elif latest_financial.price_change_7d < -15:
                    signals["sell_signals"].append({
                        "type": "price_decline",
                        "message": f"Significant 7-day price decline: {latest_financial.price_change_7d:.2f}%",
                        "strength": "high"
                    })
                
                if latest_financial.volatility_24h > 10:
                    signals["warning_signals"].append({
                        "type": "high_volatility",
                        "message": f"High volatility detected: {latest_financial.volatility_24h:.2f}%",
                        "strength": "medium"
                    })
            
            # Анализ on-chain сигналов
            if latest_onchain:
                if latest_onchain.active_addresses_24h > 100000:
                    signals["buy_signals"].append({
                        "type": "high_activity",
                        "message": f"High network activity: {latest_onchain.active_addresses_24h:,} active addresses",
                        "strength": "medium"
                    })
                
                if latest_onchain.tvl_change_7d > 10:
                    signals["buy_signals"].append({
                        "type": "tvl_growth",
                        "message": f"Strong TVL growth: {latest_onchain.tvl_change_7d:.2f}%",
                        "strength": "high"
                    })
                elif latest_onchain.tvl_change_7d < -10:
                    signals["sell_signals"].append({
                        "type": "tvl_decline",
                        "message": f"TVL decline: {latest_onchain.tvl_change_7d:.2f}%",
                        "strength": "medium"
                    })
            
            # Анализ разработки
            if latest_github:
                if latest_github.commits_30d > 100:
                    signals["buy_signals"].append({
                        "type": "active_development",
                        "message": f"Active development: {latest_github.commits_30d} commits in 30 days",
                        "strength": "medium"
                    })
                elif latest_github.commits_30d < 10:
                    signals["warning_signals"].append({
                        "type": "low_development",
                        "message": f"Low development activity: {latest_github.commits_30d} commits in 30 days",
                        "strength": "medium"
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating investment signals: {e}")
            return {"buy_signals": [], "sell_signals": [], "hold_signals": [], "warning_signals": []}
    
    async def _assess_risks(self, asset_id: int) -> Dict[str, Any]:
        """Оценить риски"""
        try:
            risks = {
                "technical_risks": [],
                "market_risks": [],
                "security_risks": [],
                "development_risks": [],
                "overall_risk_score": 0
            }
            
            # Получить последние метрики
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == asset_id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_security = self.session.query(SecurityMetrics).filter(
                SecurityMetrics.asset_id == asset_id
            ).order_by(SecurityMetrics.timestamp.desc()).first()
            
            latest_github = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == asset_id
            ).order_by(GitHubMetrics.timestamp.desc()).first()
            
            risk_score = 0
            
            # Технические риски
            if latest_financial:
                if latest_financial.volatility_24h > 15:
                    risks["technical_risks"].append({
                        "type": "high_volatility",
                        "severity": "high",
                        "description": f"Extremely high volatility: {latest_financial.volatility_24h:.2f}%"
                    })
                    risk_score += 30
                elif latest_financial.volatility_24h > 10:
                    risks["technical_risks"].append({
                        "type": "moderate_volatility",
                        "severity": "medium",
                        "description": f"High volatility: {latest_financial.volatility_24h:.2f}%"
                    })
                    risk_score += 20
                
                if latest_financial.volume_24h < latest_financial.market_cap * 0.01:
                    risks["market_risks"].append({
                        "type": "low_liquidity",
                        "severity": "high",
                        "description": "Low trading volume relative to market cap"
                    })
                    risk_score += 25
            
            # Риски безопасности
            if latest_security:
                if not latest_security.contract_verified:
                    risks["security_risks"].append({
                        "type": "unverified_contract",
                        "severity": "high",
                        "description": "Smart contract is not verified"
                    })
                    risk_score += 20
                
                if latest_security.audit_status != "audited":
                    risks["security_risks"].append({
                        "type": "unaudited_contract",
                        "severity": "medium",
                        "description": "Smart contract has not been audited"
                    })
                    risk_score += 15
                
                if latest_security.vulnerability_score > 5:
                    risks["security_risks"].append({
                        "type": "vulnerability_detected",
                        "severity": "high",
                        "description": f"High vulnerability score: {latest_security.vulnerability_score}"
                    })
                    risk_score += 25
            
            # Риски разработки
            if latest_github:
                if latest_github.commits_30d < 5:
                    risks["development_risks"].append({
                        "type": "inactive_development",
                        "severity": "medium",
                        "description": "Very low development activity"
                    })
                    risk_score += 15
                
                if latest_github.active_contributors_30d < 2:
                    risks["development_risks"].append({
                        "type": "few_contributors",
                        "severity": "medium",
                        "description": "Very few active contributors"
                    })
                    risk_score += 10
            
            risks["overall_risk_score"] = min(risk_score, 100)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return {"overall_risk_score": 50}
    
    async def _generate_recommendations(self, asset_id: int) -> Dict[str, Any]:
        """Генерировать рекомендации"""
        try:
            recommendations = {
                "investment_recommendation": "HOLD",
                "confidence_level": "medium",
                "reasoning": [],
                "action_items": [],
                "price_targets": {},
                "time_horizons": {}
            }
            
            # Получить последние метрики
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == asset_id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_ml = self.session.query(MLPrediction).filter(
                MLPrediction.asset_id == asset_id
            ).order_by(MLPrediction.created_at.desc()).first()
            
            # Анализ для рекомендации
            buy_signals = 0
            sell_signals = 0
            
            if latest_financial:
                if latest_financial.price_change_7d > 10:
                    buy_signals += 1
                    recommendations["reasoning"].append("Positive price momentum")
                elif latest_financial.price_change_7d < -10:
                    sell_signals += 1
                    recommendations["reasoning"].append("Negative price momentum")
                
                if latest_financial.volume_24h > latest_financial.market_cap * 0.05:
                    buy_signals += 1
                    recommendations["reasoning"].append("High trading volume")
                
                # Целевые цены
                current_price = latest_financial.price_usd
                recommendations["price_targets"] = {
                    "conservative": current_price * 1.2,
                    "moderate": current_price * 1.5,
                    "optimistic": current_price * 2.0
                }
            
            if latest_ml:
                if latest_ml.prediction_value > 0.7:
                    buy_signals += 1
                    recommendations["reasoning"].append("Strong ML prediction")
                elif latest_ml.prediction_value < 0.3:
                    sell_signals += 1
                    recommendations["reasoning"].append("Weak ML prediction")
                
                recommendations["confidence_level"] = "high" if latest_ml.confidence_score > 0.8 else "medium"
            
            # Определить рекомендацию
            if buy_signals > sell_signals + 1:
                recommendations["investment_recommendation"] = "BUY"
            elif sell_signals > buy_signals + 1:
                recommendations["investment_recommendation"] = "SELL"
            else:
                recommendations["investment_recommendation"] = "HOLD"
            
            # Временные горизонты
            recommendations["time_horizons"] = {
                "short_term": "1-3 months",
                "medium_term": "3-12 months",
                "long_term": "1-3 years"
            }
            
            # Пункты действий
            if recommendations["investment_recommendation"] == "BUY":
                recommendations["action_items"] = [
                    "Consider dollar-cost averaging",
                    "Set stop-loss at 20% below entry",
                    "Monitor for profit-taking opportunities"
                ]
            elif recommendations["investment_recommendation"] == "SELL":
                recommendations["action_items"] = [
                    "Consider reducing position size",
                    "Set tight stop-loss",
                    "Monitor for re-entry opportunities"
                ]
            else:
                recommendations["action_items"] = [
                    "Monitor key metrics closely",
                    "Prepare for potential volatility",
                    "Consider position sizing"
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"investment_recommendation": "HOLD", "confidence_level": "low"}
    
    async def generate_html_dashboard(self) -> str:
        """Генерировать HTML дашборд"""
        try:
            dashboard_data = await self.get_matic_dashboard_data()
            
            html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>MATIC Investment Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                    .card { background: white; padding: 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 150px; text-align: center; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
                    .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
                    .positive { color: #28a745; }
                    .negative { color: #dc3545; }
                    .neutral { color: #6c757d; }
                    .signal { padding: 10px; margin: 5px 0; border-radius: 5px; }
                    .buy-signal { background-color: #d4edda; border-left: 4px solid #28a745; }
                    .sell-signal { background-color: #f8d7da; border-left: 4px solid #dc3545; }
                    .warning-signal { background-color: #fff3cd; border-left: 4px solid #ffc107; }
                    .recommendation { font-size: 20px; font-weight: bold; padding: 15px; border-radius: 8px; text-align: center; }
                    .recommendation.buy { background-color: #d4edda; color: #155724; }
                    .recommendation.sell { background-color: #f8d7da; color: #721c24; }
                    .recommendation.hold { background-color: #d1ecf1; color: #0c5460; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 MATIC Investment Dashboard</h1>
                        <p>Real-time investment metrics and analysis</p>
                        <p>Last updated: {last_updated}</p>
                    </div>
                    
                    <div class="card">
                        <h2>📊 Financial Overview</h2>
                        <div class="metric">
                            <div class="metric-value">${current_price:.4f}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value {price_change_class}">{price_change_24h:+.2f}%</div>
                            <div class="metric-label">24h Change</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${market_cap:,.0f}</div>
                            <div class="metric-label">Market Cap</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${volume_24h:,.0f}</div>
                            <div class="metric-label">24h Volume</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🔗 On-Chain Metrics</h2>
                        <div class="metric">
                            <div class="metric-value">{active_addresses:,}</div>
                            <div class="metric-label">Active Addresses (24h)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{daily_transactions:,}</div>
                            <div class="metric-label">Daily Transactions</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value {tvl_change_class}">{tvl_change_7d:+.2f}%</div>
                            <div class="metric-label">TVL Change (7d)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{gas_price:.2f} Gwei</div>
                            <div class="metric-label">Avg Gas Price</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>💻 Development Activity</h2>
                        <div class="metric">
                            <div class="metric-value">{commits_30d}</div>
                            <div class="metric-label">Commits (30d)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{active_contributors}</div>
                            <div class="metric-label">Active Contributors</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{stars:,}</div>
                            <div class="metric-label">GitHub Stars</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{code_quality_score}/100</div>
                            <div class="metric-label">Code Quality</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🔒 Security Status</h2>
                        <div class="metric">
                            <div class="metric-value {contract_status_class}">{contract_verified}</div>
                            <div class="metric-label">Contract Verified</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{audit_status}</div>
                            <div class="metric-label">Audit Status</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{audit_score}/10</div>
                            <div class="metric-label">Audit Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{vulnerability_score}/10</div>
                            <div class="metric-label">Vulnerability Score</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🤖 ML Predictions</h2>
                        <div class="metric">
                            <div class="metric-value">{investment_score:.2f}</div>
                            <div class="metric-label">Investment Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{confidence_score:.2f}</div>
                            <div class="metric-label">Confidence Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{model_name}</div>
                            <div class="metric-label">Model</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{prediction_horizon}</div>
                            <div class="metric-label">Horizon</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>📈 Investment Signals</h2>
                        {signals_html}
                    </div>
                    
                    <div class="card">
                        <h2>⚠️ Risk Assessment</h2>
                        <div class="metric">
                            <div class="metric-value {risk_class}">{overall_risk_score}/100</div>
                            <div class="metric-label">Overall Risk Score</div>
                        </div>
                        {risks_html}
                    </div>
                    
                    <div class="card">
                        <h2>💡 Investment Recommendation</h2>
                        <div class="recommendation {recommendation_class}">
                            {investment_recommendation}
                        </div>
                        <p><strong>Confidence:</strong> {confidence_level}</p>
                        <p><strong>Reasoning:</strong> {reasoning}</p>
                        <h3>Action Items:</h3>
                        <ul>{action_items_html}</ul>
                        <h3>Price Targets:</h3>
                        <ul>
                            <li>Conservative: ${conservative_price:.4f}</li>
                            <li>Moderate: ${moderate_price:.4f}</li>
                            <li>Optimistic: ${optimistic_price:.4f}</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Подготовить данные для шаблона
            financial = dashboard_data.get("financial_overview", {})
            onchain = dashboard_data.get("onchain_overview", {})
            github = dashboard_data.get("development_overview", {})
            security = dashboard_data.get("security_overview", {})
            ml = dashboard_data.get("ml_predictions", {})
            signals = dashboard_data.get("investment_signals", {})
            risks = dashboard_data.get("risk_assessment", {})
            recommendations = dashboard_data.get("recommendations", {})
            
            # Генерировать HTML для сигналов
            signals_html = ""
            for signal_type, signal_list in signals.items():
                if signal_list:
                    signals_html += f"<h3>{signal_type.replace('_', ' ').title()}</h3>"
                    for signal in signal_list:
                        signals_html += f'<div class="signal {signal_type.replace("_", "-")}-signal">{signal.get("message", "")}</div>'
            
            # Генерировать HTML для рисков
            risks_html = ""
            for risk_type, risk_list in risks.items():
                if isinstance(risk_list, list) and risk_list:
                    risks_html += f"<h3>{risk_type.replace('_', ' ').title()}</h3>"
                    for risk in risk_list:
                        risks_html += f'<div class="signal warning-signal">{risk.get("description", "")}</div>'
            
            # Генерировать HTML для пунктов действий
            action_items_html = ""
            for item in recommendations.get("action_items", []):
                action_items_html += f"<li>{item}</li>"
            
            # Определить CSS классы
            price_change_class = "positive" if financial.get("price_changes", {}).get("24h", 0) > 0 else "negative"
            tvl_change_class = "positive" if onchain.get("tvl_changes", {}).get("7d", 0) > 0 else "negative"
            contract_status_class = "positive" if security.get("contract", {}).get("verified", False) else "negative"
            risk_class = "negative" if risks.get("overall_risk_score", 50) > 70 else "neutral" if risks.get("overall_risk_score", 50) > 40 else "positive"
            recommendation_class = recommendations.get("investment_recommendation", "HOLD").lower()
            
            # Заполнить шаблон
            html_content = html_template.format(
                last_updated=dashboard_data.get("asset_info", {}).get("last_updated", ""),
                current_price=financial.get("current_price", 0),
                price_change_24h=financial.get("price_changes", {}).get("24h", 0),
                price_change_class=price_change_class,
                market_cap=financial.get("market_cap", 0),
                volume_24h=financial.get("volume_24h", 0),
                active_addresses=onchain.get("activity", {}).get("active_addresses_24h", 0),
                daily_transactions=onchain.get("activity", {}).get("daily_transactions", 0),
                tvl_change_7d=onchain.get("tvl_changes", {}).get("7d", 0),
                tvl_change_class=tvl_change_class,
                gas_price=onchain.get("network", {}).get("gas_price_avg", 0),
                commits_30d=github.get("activity", {}).get("commits_30d", 0),
                active_contributors=github.get("contributors", {}).get("active_30d", 0),
                stars=github.get("community", {}).get("stars", 0),
                code_quality_score=github.get("quality", {}).get("code_quality_score", 0),
                contract_verified="Yes" if security.get("contract", {}).get("verified", False) else "No",
                contract_status_class=contract_status_class,
                audit_status=security.get("audit", {}).get("status", "Unknown"),
                audit_score=security.get("audit", {}).get("score", 0),
                vulnerability_score=security.get("contract", {}).get("vulnerability_score", 0),
                investment_score=ml.get("investment_score", 0),
                confidence_score=ml.get("confidence_score", 0),
                model_name=ml.get("model_name", "Unknown"),
                prediction_horizon=ml.get("prediction_horizon", "Unknown"),
                signals_html=signals_html,
                overall_risk_score=risks.get("overall_risk_score", 50),
                risk_class=risk_class,
                risks_html=risks_html,
                investment_recommendation=recommendations.get("investment_recommendation", "HOLD"),
                recommendation_class=recommendation_class,
                confidence_level=recommendations.get("confidence_level", "medium"),
                reasoning=", ".join(recommendations.get("reasoning", [])),
                action_items_html=action_items_html,
                conservative_price=recommendations.get("price_targets", {}).get("conservative", 0),
                moderate_price=recommendations.get("price_targets", {}).get("moderate", 0),
                optimistic_price=recommendations.get("price_targets", {}).get("optimistic", 0)
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML dashboard: {e}")
            return f"<html><body><h1>Error generating dashboard: {e}</h1></body></html>"
    
    async def save_dashboard_to_file(self, filename: str = "matic_dashboard.html"):
        """Сохранить дашборд в файл"""
        try:
            html_content = await self.generate_html_dashboard()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Dashboard saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
            return None

async def main():
    """Основная функция"""
    logger.info("🚀 Starting MATIC Investment Dashboard...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("Database initialized")
        
        # Создать дашборд
        async with InvestmentMonitoringDashboard() as dashboard:
            # Получить данные дашборда
            dashboard_data = await dashboard.get_matic_dashboard_data()
            
            if dashboard_data:
                logger.info("✅ Successfully generated dashboard data")
                
                # Сохранить данные в JSON
                with open("matic_dashboard_data.json", "w") as f:
                    json.dump(dashboard_data, f, indent=2, default=str)
                
                # Создать HTML дашборд
                dashboard_file = await dashboard.save_dashboard_to_file()
                
                if dashboard_file:
                    logger.info(f"📊 Dashboard saved to {dashboard_file}")
                    logger.info("🌐 Open the HTML file in your browser to view the dashboard")
                else:
                    logger.error("❌ Failed to save dashboard")
            else:
                logger.error("❌ Failed to generate dashboard data")
        
        logger.info("🎉 Dashboard generation completed!")
        
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
