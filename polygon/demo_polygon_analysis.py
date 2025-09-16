#!/usr/bin/env python3
"""
Демонстрационная версия анализа Polygon с синтетическими данными
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import json
import random

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics, MLPrediction
)

class DemoPolygonAnalysis:
    """Демонстрационный анализ Polygon с синтетическими данными"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.polygon_asset_id = 1  # MATIC asset ID
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def generate_synthetic_data(self, days_back: int = 90):
        """Генерировать синтетические данные для демонстрации"""
        logger.info("🎭 Generating synthetic data for demonstration...")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Базовые значения
        base_price = 0.85
        base_tvl = 1000000000  # 1B
        base_volume = 50000000  # 50M
        
        # Генерировать данные за каждый день
        for i in range(days_back):
            current_date = start_date + timedelta(days=i)
            
            # Добавить тренд и волатильность
            trend_factor = 1 + (i / days_back) * 0.1  # 10% рост за период
            volatility = random.uniform(-0.05, 0.05)  # ±5% волатильность
            price = base_price * trend_factor * (1 + volatility)
            
            # Финансовые метрики
            financial_metrics = FinancialMetrics(
                asset_id=self.polygon_asset_id,
                price_usd=price,
                market_cap=price * 10000000000,  # 10B supply
                volume_24h=base_volume * random.uniform(0.5, 2.0),
                volatility_24h=abs(volatility) * 100,
                price_change_24h=volatility * 100,
                price_change_7d=random.uniform(-10, 15),
                price_change_30d=random.uniform(-20, 30),
                volume_market_cap_ratio=random.uniform(0.01, 0.05),
                liquidity_score=random.uniform(7, 9),
                all_time_high=price * 1.5,
                all_time_low=price * 0.6
            )
            self.session.add(financial_metrics)
            
            # On-chain метрики
            tvl_growth = random.uniform(0.95, 1.05)
            onchain_metrics = OnChainMetrics(
                asset_id=self.polygon_asset_id,
                tvl=base_tvl * tvl_growth,
                tvl_change_24h=random.uniform(-5, 8),
                tvl_change_7d=random.uniform(-10, 15),
                daily_transactions=int(random.uniform(2000000, 4000000)),
                active_addresses_24h=int(random.uniform(500000, 800000)),
                transaction_volume_24h=base_volume * random.uniform(0.8, 1.5),
                avg_transaction_fee=random.uniform(0.001, 0.01),
                transaction_success_rate=random.uniform(98, 99.5),
                gas_usage_efficiency=random.uniform(80, 90),
                new_addresses_24h=int(random.uniform(10000, 50000)),
                unique_users_7d=int(random.uniform(1000000, 2000000)),
                user_retention_rate=random.uniform(70, 80),
                whale_activity=random.uniform(3, 8),
                new_contracts_deployed=int(random.uniform(50, 200)),
                contract_interactions_24h=int(random.uniform(100000, 500000)),
                contract_complexity_score=random.uniform(7, 9),
                liquidity_pools_count=int(random.uniform(100, 500)),
                liquidity_pools_tvl=base_tvl * 0.1 * random.uniform(0.8, 1.2),
                yield_farming_apy=random.uniform(3, 12),
                lending_volume=base_volume * random.uniform(0.2, 0.4),
                borrowing_volume=base_volume * random.uniform(0.1, 0.3)
            )
            self.session.add(onchain_metrics)
            
            # GitHub метрики (еженедельно)
            if i % 7 == 0:
                github_metrics = GitHubMetrics(
                    asset_id=self.polygon_asset_id,
                    commits_24h=int(random.uniform(5, 25)),
                    commits_7d=int(random.uniform(30, 100)),
                    commits_30d=int(random.uniform(100, 400)),
                    active_contributors_30d=int(random.uniform(20, 50)),
                    stars=int(random.uniform(8000, 9000)),
                    forks=int(random.uniform(1100, 1300)),
                    open_issues=int(random.uniform(20, 40)),
                    open_prs=int(random.uniform(8, 20)),
                    code_quality_score=random.uniform(8, 9)
                )
                self.session.add(github_metrics)
            
            # Community метрики (еженедельно)
            if i % 7 == 0:
                community_metrics = CommunityMetrics(
                    asset_id=self.polygon_asset_id,
                    twitter_followers=int(random.uniform(840000, 860000)),
                    telegram_members=int(random.uniform(120000, 130000)),
                    discord_members=int(random.uniform(44000, 46000)),
                    social_engagement_rate=random.uniform(4, 5),
                    brand_awareness_score=random.uniform(8, 9)
                )
                self.session.add(community_metrics)
            
            # Trending метрики
            trending_metrics = TrendingMetrics(
                asset_id=self.polygon_asset_id,
                momentum_score=random.uniform(4, 7),
                trend_direction=random.choice(["bullish", "bearish", "sideways"]),
                trend_strength=random.uniform(0.3, 0.8),
                fear_greed_index=random.uniform(30, 70),
                social_sentiment=random.uniform(0.4, 0.7),
                anomaly_score=random.uniform(0, 5),
                anomaly_type=random.choice(["none", "price_spike", "volume_surge"])
            )
            self.session.add(trending_metrics)
        
        self.session.commit()
        logger.info(f"✅ Generated synthetic data for {days_back} days")
    
    async def analyze_trends(self):
        """Анализ трендов на основе синтетических данных"""
        logger.info("📈 Analyzing trends...")
        
        # Получить последние данные
        latest_financial = self.session.query(FinancialMetrics).filter(
            FinancialMetrics.asset_id == self.polygon_asset_id
        ).order_by(FinancialMetrics.timestamp.desc()).first()
        
        latest_onchain = self.session.query(OnChainMetrics).filter(
            OnChainMetrics.asset_id == self.polygon_asset_id
        ).order_by(OnChainMetrics.timestamp.desc()).first()
        
        if not latest_financial or not latest_onchain:
            return {"error": "No data available"}
        
        # Анализ трендов
        trends = {
            "price_trend": "bullish" if latest_financial.price_change_7d > 5 else "bearish" if latest_financial.price_change_7d < -5 else "sideways",
            "tvl_trend": "bullish" if latest_onchain.tvl_change_7d > 5 else "bearish" if latest_onchain.tvl_change_7d < -5 else "sideways",
            "volume_trend": "high" if latest_financial.volume_24h > latest_financial.volume_24h * 1.5 else "normal",
            "network_activity": "high" if latest_onchain.daily_transactions > 3000000 else "normal"
        }
        
        # Определить общий тренд
        bullish_signals = sum(1 for trend in trends.values() if trend in ["bullish", "high"])
        bearish_signals = sum(1 for trend in trends.values() if trend in ["bearish"])
        
        if bullish_signals > bearish_signals:
            overall_trend = "bullish"
        elif bearish_signals > bullish_signals:
            overall_trend = "bearish"
        else:
            overall_trend = "sideways"
        
        return {
            "overall_trend": overall_trend,
            "trends": trends,
            "current_price": latest_financial.price_usd,
            "price_change_7d": latest_financial.price_change_7d,
            "tvl": latest_onchain.tvl,
            "daily_transactions": latest_onchain.daily_transactions
        }
    
    async def predict_price(self):
        """Простое предсказание цены на основе трендов"""
        logger.info("🔮 Making price prediction...")
        
        # Получить последние данные
        latest_financial = self.session.query(FinancialMetrics).filter(
            FinancialMetrics.asset_id == self.polygon_asset_id
        ).order_by(FinancialMetrics.timestamp.desc()).first()
        
        if not latest_financial:
            return {"error": "No financial data available"}
        
        current_price = latest_financial.price_usd
        price_change_7d = latest_financial.price_change_7d
        volatility = latest_financial.volatility_24h
        
        # Простая модель предсказания
        # Базовое предсказание на основе последнего тренда
        base_prediction = price_change_7d * 0.7  # Консервативная оценка
        
        # Добавить волатильность
        volatility_factor = random.uniform(-volatility/2, volatility/2)
        
        # Добавить случайный фактор
        random_factor = random.uniform(-2, 2)
        
        predicted_change = base_prediction + volatility_factor + random_factor
        
        # Ограничить экстремальные значения
        predicted_change = max(-20, min(20, predicted_change))
        
        predicted_price = current_price * (1 + predicted_change / 100)
        
        # Рассчитать уверенность
        confidence = max(0.3, 1.0 - abs(volatility) / 20)
        
        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_7d": predicted_change,
            "confidence": confidence,
            "model": "trend_based_simple"
        }
    
    async def generate_recommendations(self, trend_analysis, price_prediction):
        """Генерация рекомендаций"""
        recommendations = []
        
        overall_trend = trend_analysis.get("overall_trend", "sideways")
        predicted_change = price_prediction.get("predicted_change_7d", 0)
        confidence = price_prediction.get("confidence", 0.5)
        
        # Рекомендации на основе тренда
        if overall_trend == "bullish":
            recommendations.append("🟢 Bullish trend detected. Consider holding or accumulating MATIC.")
        elif overall_trend == "bearish":
            recommendations.append("🔴 Bearish trend detected. Consider reducing position or waiting for better entry.")
        else:
            recommendations.append("🟡 Mixed signals. Wait for clearer directional movement.")
        
        # Рекомендации на основе предсказания
        if confidence > 0.7:
            if predicted_change > 5:
                recommendations.append("📈 High confidence bullish prediction. Consider accumulating.")
            elif predicted_change < -5:
                recommendations.append("📉 High confidence bearish prediction. Consider reducing position.")
            else:
                recommendations.append("📊 High confidence sideways prediction. Consider range trading.")
        else:
            recommendations.append("⚠️ Low confidence prediction. Use smaller position sizes.")
        
        # Общие рекомендации
        recommendations.extend([
            "🔍 Monitor on-chain metrics for early signals.",
            "💧 Watch for DeFi TVL growth as positive indicator.",
            "⛽ Track network activity and gas prices.",
            "💰 Consider dollar-cost averaging for long-term positions.",
            "🛡️ Set stop-losses based on risk tolerance."
        ])
        
        return recommendations
    
    async def run_demo_analysis(self):
        """Запустить демонстрационный анализ"""
        try:
            logger.info("🚀 Starting Demo Polygon Analysis...")
            
            # Генерировать синтетические данные
            await self.generate_synthetic_data(days_back=90)
            
            # Анализ трендов
            trend_analysis = await self.analyze_trends()
            
            # Предсказание цены
            price_prediction = await self.predict_price()
            
            # Генерация рекомендаций
            recommendations = await self.generate_recommendations(trend_analysis, price_prediction)
            
            # Создать итоговый отчет
            report = {
                "analysis_timestamp": datetime.utcnow(),
                "analysis_type": "demo_with_synthetic_data",
                "trend_analysis": trend_analysis,
                "price_prediction": price_prediction,
                "recommendations": recommendations,
                "data_quality": "synthetic_demo_data",
                "disclaimer": "This is a demonstration with synthetic data. Not financial advice."
            }
            
            # Сохранить отчет
            with open("demo_polygon_analysis_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            # Вывести результаты
            self._print_demo_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in demo analysis: {e}")
            return {"error": str(e)}
    
    def _print_demo_results(self, report):
        """Вывести результаты демонстрации"""
        logger.info("=" * 80)
        logger.info("🎯 DEMO POLYGON ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        trend_analysis = report["trend_analysis"]
        price_prediction = report["price_prediction"]
        
        logger.info(f"📊 Analysis Type: {report['analysis_type']}")
        logger.info(f"📅 Data Quality: {report['data_quality']}")
        
        logger.info(f"📈 Overall Trend: {trend_analysis.get('overall_trend', 'unknown').upper()}")
        logger.info(f"💰 Current MATIC Price: ${trend_analysis.get('current_price', 0):.4f}")
        logger.info(f"📊 7-day Price Change: {trend_analysis.get('price_change_7d', 0):+.2f}%")
        logger.info(f"🏦 TVL: ${trend_analysis.get('tvl', 0):,.0f}")
        logger.info(f"🔄 Daily Transactions: {trend_analysis.get('daily_transactions', 0):,}")
        
        logger.info(f"🔮 Predicted 7-day Change: {price_prediction.get('predicted_change_7d', 0):+.2f}%")
        logger.info(f"🎯 Predicted Price: ${price_prediction.get('predicted_price', 0):.4f}")
        logger.info(f"🎲 Confidence: {price_prediction.get('confidence', 0):.1%}")
        logger.info(f"🤖 Model: {price_prediction.get('model', 'unknown')}")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info(f"\n⚠️ DISCLAIMER: {report['disclaimer']}")
        logger.info("\n📁 Full report saved to: demo_polygon_analysis_report.json")
        logger.info("=" * 80)

async def main():
    """Основная функция"""
    logger.info("🎭 Starting Demo Polygon Analysis...")
    
    try:
        # Инициализировать базу данных
        init_db()
        
        async with DemoPolygonAnalysis() as demo:
            await demo.run_demo_analysis()
        
        logger.info("🎉 Demo analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in demo analysis: {e}")

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
