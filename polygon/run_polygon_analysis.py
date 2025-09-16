#!/usr/bin/env python3
"""
Главный скрипт для полного анализа Polygon и предсказания цены на следующую неделю
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import json

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import init_db
from polygon_price_prediction_data import PolygonDataCollector
from polygon_price_predictor import PolygonPricePredictor
from polygon_trend_analyzer import PolygonTrendAnalyzer

class PolygonAnalysisPipeline:
    """Полный пайплайн анализа Polygon"""
    
    def __init__(self):
        self.results = {}
        
    async def run_complete_analysis(self):
        """Запустить полный анализ Polygon"""
        try:
            logger.info("🚀 Starting complete Polygon analysis pipeline...")
            
            # 1. Сбор данных
            logger.info("📊 Step 1: Collecting Polygon data...")
            await self._collect_data()
            
            # 2. Анализ трендов
            logger.info("📈 Step 2: Analyzing trends and patterns...")
            await self._analyze_trends()
            
            # 3. Предсказание цены
            logger.info("🔮 Step 3: Generating price predictions...")
            await self._predict_price()
            
            # 4. Генерация отчета
            logger.info("📋 Step 4: Generating comprehensive report...")
            await self._generate_report()
            
            logger.info("✅ Complete Polygon analysis pipeline finished!")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis pipeline: {e}")
            raise
    
    async def _collect_data(self):
        """Собрать данные Polygon"""
        try:
            async with PolygonDataCollector() as collector:
                await collector.collect_all_data()
            
            self.results["data_collection"] = {
                "status": "completed",
                "timestamp": datetime.utcnow(),
                "message": "All Polygon data collected successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            self.results["data_collection"] = {
                "status": "failed",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _analyze_trends(self):
        """Анализ трендов и паттернов"""
        try:
            async with PolygonTrendAnalyzer() as analyzer:
                await analyzer.setup_polygon_asset()
                trend_analysis = await analyzer.generate_comprehensive_analysis(days_back=90)
            
            self.results["trend_analysis"] = trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            self.results["trend_analysis"] = {
                "status": "failed",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _predict_price(self):
        """Предсказание цены"""
        try:
            async with PolygonPricePredictor() as predictor:
                await predictor.setup_polygon_asset()
                
                # Подготовить данные для обучения
                features_df, targets_series = await predictor.prepare_training_data(days_back=180)
                
                if not features_df.empty:
                    # Обучить модели
                    await predictor.train_models(features_df, targets_series)
                    
                    # Сделать предсказание
                    prediction = await predictor.predict_price_change_7d()
                    
                    # Получить анализ предсказаний
                    analysis = await predictor.get_prediction_analysis()
                    
                    # Сохранить модели
                    await predictor.save_models()
                    
                    self.results["price_prediction"] = {
                        "prediction": prediction,
                        "analysis": analysis,
                        "status": "completed"
                    }
                else:
                    self.results["price_prediction"] = {
                        "status": "failed",
                        "error": "No training data available"
                    }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            self.results["price_prediction"] = {
                "status": "failed",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _generate_report(self):
        """Генерация итогового отчета"""
        try:
            report = {
                "report_timestamp": datetime.utcnow(),
                "analysis_period": "90 days",
                "prediction_horizon": "7 days",
                "summary": self._generate_summary(),
                "detailed_results": self.results,
                "recommendations": self._generate_final_recommendations()
            }
            
            # Сохранить отчет
            with open("polygon_analysis_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            self.results["final_report"] = report
            
            # Вывести краткий отчет
            self._print_summary_report(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def _generate_summary(self) -> dict:
        """Генерация краткой сводки"""
        summary = {
            "overall_status": "completed",
            "data_collection_status": self.results.get("data_collection", {}).get("status", "unknown"),
            "trend_analysis_status": "completed" if "trend_analysis" in self.results else "failed",
            "price_prediction_status": self.results.get("price_prediction", {}).get("status", "unknown")
        }
        
        # Добавить информацию о трендах
        if "trend_analysis" in self.results and "overall_trend" in self.results["trend_analysis"]:
            summary["overall_trend"] = self.results["trend_analysis"]["overall_trend"]
            summary["trend_distribution"] = self.results["trend_analysis"]["trend_distribution"]
        
        # Добавить информацию о предсказании
        if "price_prediction" in self.results and "prediction" in self.results["price_prediction"]:
            prediction = self.results["price_prediction"]["prediction"]
            summary["predicted_price_change"] = prediction.get("price_change_7d_percent", 0)
            summary["prediction_confidence"] = prediction.get("confidence_score", 0)
            summary["current_price"] = prediction.get("current_price", 0)
            summary["predicted_price"] = prediction.get("predicted_price", 0)
        
        return summary
    
    def _generate_final_recommendations(self) -> list:
        """Генерация финальных рекомендаций"""
        recommendations = []
        
        # Рекомендации на основе трендов
        if "trend_analysis" in self.results and "recommendations" in self.results["trend_analysis"]:
            recommendations.extend(self.results["trend_analysis"]["recommendations"])
        
        # Рекомендации на основе предсказания
        if "price_prediction" in self.results and "prediction" in self.results["price_prediction"]:
            prediction = self.results["price_prediction"]["prediction"]
            price_change = prediction.get("price_change_7d_percent", 0)
            confidence = prediction.get("confidence_score", 0)
            
            if confidence > 0.7:  # Высокая уверенность
                if price_change > 5:
                    recommendations.append("High confidence bullish prediction. Consider accumulating MATIC.")
                elif price_change < -5:
                    recommendations.append("High confidence bearish prediction. Consider reducing position.")
                else:
                    recommendations.append("High confidence sideways prediction. Consider range trading.")
            else:
                recommendations.append("Low confidence prediction. Wait for more data or use smaller position sizes.")
        
        # Общие рекомендации
        recommendations.extend([
            "Monitor on-chain metrics for early signals of trend changes.",
            "Watch for DeFi TVL growth as a positive indicator for MATIC price.",
            "Track network activity and gas prices for adoption signals.",
            "Consider dollar-cost averaging for long-term positions.",
            "Set stop-losses based on technical analysis and risk tolerance."
        ])
        
        return recommendations
    
    def _print_summary_report(self, report: dict):
        """Вывести краткий отчет в консоль"""
        logger.info("=" * 80)
        logger.info("🎯 POLYGON ANALYSIS SUMMARY REPORT")
        logger.info("=" * 80)
        
        summary = report["summary"]
        
        # Статус анализа
        logger.info(f"📊 Analysis Status: {summary.get('overall_status', 'Unknown').upper()}")
        logger.info(f"📅 Analysis Period: {report['analysis_period']}")
        logger.info(f"🔮 Prediction Horizon: {report['prediction_horizon']}")
        
        # Тренды
        if "overall_trend" in summary:
            logger.info(f"📈 Overall Trend: {summary['overall_trend'].upper()}")
            if "trend_distribution" in summary:
                dist = summary["trend_distribution"]
                logger.info(f"   • Bullish: {dist.get('bullish_percentage', 0):.1f}%")
                logger.info(f"   • Bearish: {dist.get('bearish_percentage', 0):.1f}%")
                logger.info(f"   • Sideways: {dist.get('sideways_percentage', 0):.1f}%")
        
        # Предсказание цены
        if "current_price" in summary:
            logger.info(f"💰 Current MATIC Price: ${summary['current_price']:.4f}")
        
        if "predicted_price_change" in summary:
            change = summary["predicted_price_change"]
            confidence = summary.get("prediction_confidence", 0)
            logger.info(f"🔮 Predicted 7-day Change: {change:+.2f}% (Confidence: {confidence:.1%})")
            
            if "predicted_price" in summary:
                logger.info(f"🎯 Predicted Price: ${summary['predicted_price']:.4f}")
        
        # Рекомендации
        logger.info("\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"][:5], 1):  # Показать топ-5 рекомендаций
            logger.info(f"   {i}. {rec}")
        
        logger.info("\n📁 Full report saved to: polygon_analysis_report.json")
        logger.info("=" * 80)

async def main():
    """Основная функция"""
    logger.info("🚀 Starting Polygon Analysis Pipeline...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("Database initialized")
        
        # Запустить полный анализ
        pipeline = PolygonAnalysisPipeline()
        await pipeline.run_complete_analysis()
        
        logger.info("🎉 Polygon analysis pipeline completed successfully!")
        
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
