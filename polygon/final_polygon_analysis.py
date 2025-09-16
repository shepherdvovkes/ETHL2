#!/usr/bin/env python3
"""
Финальная система анализа Polygon - объединяет все компоненты
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import json
import pandas as pd
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import init_db
from quicknode_data_mapper import QuickNodeDataMapper
from smart_polygon_predictor import SmartPolygonPredictor
from demo_polygon_analysis import DemoPolygonAnalysis

class FinalPolygonAnalysis:
    """Финальная система анализа Polygon"""
    
    def __init__(self):
        self.results = {}
        
    async def run_complete_analysis(self, use_real_data: bool = True):
        """Запустить полный анализ"""
        try:
            logger.info("🚀 Starting Final Polygon Analysis System...")
            
            if use_real_data:
                await self._run_real_data_analysis()
            else:
                await self._run_demo_analysis()
            
            # Генерировать итоговый отчет
            await self._generate_final_report()
            
            logger.info("✅ Final analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in final analysis: {e}")
            raise
    
    async def _run_real_data_analysis(self):
        """Анализ с реальными данными"""
        logger.info("📊 Running analysis with real data...")
        
        try:
            # 1. Создать карту данных QuickNode
            logger.info("🗺️ Step 1: Creating QuickNode data map...")
            mapper = QuickNodeDataMapper()
            data_map = await mapper.map_available_data()
            await mapper.save_data_map()
            
            # 2. Запустить умный предсказатель
            logger.info("🧠 Step 2: Running smart predictor...")
            async with SmartPolygonPredictor() as predictor:
                await predictor.load_data_map()
                analysis_result = await predictor.run_smart_analysis()
            
            self.results["real_data_analysis"] = {
                "data_map": data_map,
                "prediction": analysis_result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in real data analysis: {e}")
            self.results["real_data_analysis"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_demo_analysis(self):
        """Демонстрационный анализ"""
        logger.info("🎭 Running demo analysis...")
        
        try:
            async with DemoPolygonAnalysis() as demo:
                demo_result = await demo.run_demo_analysis()
            
            self.results["demo_analysis"] = {
                "result": demo_result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in demo analysis: {e}")
            self.results["demo_analysis"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _generate_final_report(self):
        """Генерировать итоговый отчет"""
        logger.info("📋 Generating final report...")
        
        report = {
            "report_timestamp": datetime.utcnow(),
            "analysis_type": "comprehensive_polygon_analysis",
            "system_version": "2.0",
            "components": {
                "data_mapping": "QuickNode API data mapping",
                "smart_prediction": "ML-based price prediction",
                "trend_analysis": "Comprehensive trend analysis",
                "demo_mode": "Synthetic data demonstration"
            },
            "results": self.results,
            "summary": self._generate_summary(),
            "recommendations": self._generate_final_recommendations(),
            "data_sources": {
                "quicknode_api": "Polygon blockchain data",
                "coingecko_api": "Price and market data",
                "synthetic_data": "ML training data generation"
            },
            "disclaimer": "This analysis is for educational purposes only. Not financial advice."
        }
        
        # Сохранить отчет
        with open("final_polygon_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Вывести результаты
        self._print_final_results(report)
        
        self.results["final_report"] = report
    
    def _generate_summary(self) -> dict:
        """Генерировать сводку"""
        summary = {
            "analysis_status": "completed",
            "data_quality": "mixed_real_and_synthetic",
            "prediction_available": False,
            "trend_analysis_available": False
        }
        
        # Проверить результаты реального анализа
        if "real_data_analysis" in self.results:
            real_analysis = self.results["real_data_analysis"]
            if real_analysis.get("status") == "completed":
                summary["prediction_available"] = True
                if "prediction" in real_analysis:
                    prediction = real_analysis["prediction"]
                    if "prediction" in prediction:
                        pred_data = prediction["prediction"]
                        summary.update({
                            "current_price": pred_data.get("current_price", 0),
                            "predicted_change_7d": pred_data.get("predicted_change_7d", 0),
                            "prediction_confidence": pred_data.get("confidence", 0)
                        })
        
        # Проверить результаты демо анализа
        if "demo_analysis" in self.results:
            demo_analysis = self.results["demo_analysis"]
            if demo_analysis.get("status") == "completed":
                summary["trend_analysis_available"] = True
                if "result" in demo_analysis:
                    demo_result = demo_analysis["result"]
                    if "trend_analysis" in demo_result:
                        trend_data = demo_result["trend_analysis"]
                        summary.update({
                            "overall_trend": trend_data.get("overall_trend", "unknown"),
                            "current_price_demo": trend_data.get("current_price", 0),
                            "price_change_7d_demo": trend_data.get("price_change_7d", 0)
                        })
        
        return summary
    
    def _generate_final_recommendations(self) -> list:
        """Генерировать финальные рекомендации"""
        recommendations = []
        
        # Рекомендации на основе анализа
        if "real_data_analysis" in self.results:
            real_analysis = self.results["real_data_analysis"]
            if real_analysis.get("status") == "completed":
                recommendations.append("✅ Real data analysis completed successfully")
                recommendations.append("📊 Use QuickNode data map for targeted data collection")
                recommendations.append("🤖 ML models trained on synthetic data with real-time features")
            else:
                recommendations.append("⚠️ Real data analysis failed - using fallback methods")
        
        if "demo_analysis" in self.results:
            demo_analysis = self.results["demo_analysis"]
            if demo_analysis.get("status") == "completed":
                recommendations.append("🎭 Demo analysis completed with synthetic data")
                recommendations.append("📈 Trend analysis available for demonstration")
        
        # Общие рекомендации
        recommendations.extend([
            "🔍 Monitor QuickNode API connectivity for real-time data",
            "📊 Use data map to optimize API calls and reduce costs",
            "🤖 Train ML models with more historical data when available",
            "💡 Combine multiple data sources for better predictions",
            "🛡️ Always use proper risk management in trading decisions",
            "📚 This system demonstrates the architecture for crypto analysis"
        ])
        
        return recommendations
    
    def _print_final_results(self, report: dict):
        """Вывести финальные результаты"""
        logger.info("=" * 80)
        logger.info("🎯 FINAL POLYGON ANALYSIS SYSTEM RESULTS")
        logger.info("=" * 80)
        
        summary = report["summary"]
        
        logger.info(f"📊 Analysis Status: {summary.get('analysis_status', 'unknown').upper()}")
        logger.info(f"📅 System Version: {report.get('system_version', 'unknown')}")
        logger.info(f"🗂️ Data Quality: {summary.get('data_quality', 'unknown')}")
        
        # Результаты предсказания
        if summary.get("prediction_available"):
            logger.info(f"💰 Current Price: ${summary.get('current_price', 0):.4f}")
            logger.info(f"🔮 Predicted 7-day Change: {summary.get('predicted_change_7d', 0):+.2f}%")
            logger.info(f"🎲 Prediction Confidence: {summary.get('prediction_confidence', 0):.1%}")
        
        # Результаты трендового анализа
        if summary.get("trend_analysis_available"):
            logger.info(f"📈 Overall Trend: {summary.get('overall_trend', 'unknown').upper()}")
            logger.info(f"📊 Demo Price: ${summary.get('current_price_demo', 0):.4f}")
            logger.info(f"📈 Demo 7-day Change: {summary.get('price_change_7d_demo', 0):+.2f}%")
        
        # Компоненты системы
        logger.info("\n🛠️ SYSTEM COMPONENTS:")
        components = report.get("components", {})
        for component, description in components.items():
            logger.info(f"   • {component}: {description}")
        
        # Источники данных
        logger.info("\n📡 DATA SOURCES:")
        data_sources = report.get("data_sources", {})
        for source, description in data_sources.items():
            logger.info(f"   • {source}: {description}")
        
        # Рекомендации
        logger.info("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info(f"\n⚠️ DISCLAIMER: {report['disclaimer']}")
        logger.info("\n📁 Full report saved to: final_polygon_analysis_report.json")
        logger.info("=" * 80)

async def main():
    """Основная функция"""
    logger.info("🚀 Starting Final Polygon Analysis System...")
    
    try:
        # Инициализировать базу данных
        init_db()
        
        # Создать систему анализа
        analysis_system = FinalPolygonAnalysis()
        
        # Запустить анализ (сначала попробуем реальные данные, потом демо)
        try:
            await analysis_system.run_complete_analysis(use_real_data=True)
        except Exception as e:
            logger.warning(f"Real data analysis failed: {e}")
            logger.info("🔄 Falling back to demo analysis...")
            await analysis_system.run_complete_analysis(use_real_data=False)
        
        logger.info("🎉 Final analysis system completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in final analysis system: {e}")

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
