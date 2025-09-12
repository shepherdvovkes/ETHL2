#!/usr/bin/env python3
"""
Автоматический запуск полного анализа инвестиционных метрик для MATIC
Объединяет сбор данных, ML анализ и создание дашборда
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from collect_investment_metrics import InvestmentMetricsCollector
from investment_monitoring_dashboard import InvestmentMonitoringDashboard

async def run_complete_analysis():
    """Запустить полный анализ инвестиционных метрик"""
    logger.info("🚀 Starting complete MATIC investment analysis...")
    
    try:
        # Этап 1: Сбор всех метрик
        logger.info("📊 Phase 1: Collecting investment metrics...")
        async with InvestmentMetricsCollector() as collector:
            matic_results = await collector.collect_metrics_for_matic()
            
            if matic_results:
                logger.info("✅ Successfully collected all metrics")
                
                # Сохранить результаты сбора
                import json
                with open("matic_metrics_collection.json", "w") as f:
                    json.dump(matic_results, f, indent=2, default=str)
                
                logger.info("💾 Metrics collection results saved to matic_metrics_collection.json")
            else:
                logger.error("❌ Failed to collect metrics")
                return False
        
        # Этап 2: Создание дашборда
        logger.info("📈 Phase 2: Generating investment dashboard...")
        async with InvestmentMonitoringDashboard() as dashboard:
            dashboard_data = await dashboard.get_matic_dashboard_data()
            
            if dashboard_data:
                logger.info("✅ Successfully generated dashboard data")
                
                # Сохранить данные дашборда
                with open("matic_dashboard_data.json", "w") as f:
                    json.dump(dashboard_data, f, indent=2, default=str)
                
                # Создать HTML дашборд
                dashboard_file = await dashboard.save_dashboard_to_file("matic_investment_dashboard.html")
                
                if dashboard_file:
                    logger.info(f"🌐 Dashboard saved to {dashboard_file}")
                else:
                    logger.error("❌ Failed to save dashboard")
            else:
                logger.error("❌ Failed to generate dashboard data")
                return False
        
        # Этап 3: Генерация отчета
        logger.info("📋 Phase 3: Generating investment report...")
        await generate_investment_report()
        
        logger.info("🎉 Complete MATIC investment analysis finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in complete analysis: {e}")
        return False

async def generate_investment_report():
    """Генерировать итоговый инвестиционный отчет"""
    try:
        # Загрузить данные
        import json
        
        try:
            with open("matic_metrics_collection.json", "r") as f:
                metrics_data = json.load(f)
        except FileNotFoundError:
            logger.warning("Metrics collection file not found, skipping report generation")
            return
        
        try:
            with open("matic_dashboard_data.json", "r") as f:
                dashboard_data = json.load(f)
        except FileNotFoundError:
            logger.warning("Dashboard data file not found, skipping report generation")
            return
        
        # Извлечь ключевые данные
        investment_report = metrics_data.get("investment_report", {})
        financial = dashboard_data.get("financial_overview", {})
        onchain = dashboard_data.get("onchain_overview", {})
        github = dashboard_data.get("development_overview", {})
        security = dashboard_data.get("security_overview", {})
        ml = dashboard_data.get("ml_predictions", {})
        recommendations = dashboard_data.get("recommendations", {})
        
        # Создать отчет
        report = {
            "executive_summary": {
                "asset": "MATIC (Polygon)",
                "analysis_date": datetime.utcnow().isoformat(),
                "investment_recommendation": recommendations.get("investment_recommendation", "HOLD"),
                "confidence_level": recommendations.get("confidence_level", "medium"),
                "investment_score": investment_report.get("investment_score", 0),
                "risk_score": investment_report.get("risk_score", 0),
                "key_takeaways": [
                    f"Current price: ${financial.get('current_price', 0):.4f}",
                    f"24h change: {financial.get('price_changes', {}).get('24h', 0):+.2f}%",
                    f"Active addresses: {onchain.get('activity', {}).get('active_addresses_24h', 0):,}",
                    f"Development activity: {github.get('activity', {}).get('commits_30d', 0)} commits (30d)",
                    f"ML prediction: {ml.get('investment_score', 0):.2f} (confidence: {ml.get('confidence_score', 0):.2f})"
                ]
            },
            "financial_analysis": {
                "current_price": financial.get("current_price", 0),
                "market_cap": financial.get("market_cap", 0),
                "volume_24h": financial.get("volume_24h", 0),
                "price_changes": financial.get("price_changes", {}),
                "volatility": financial.get("volatility", {}),
                "supply_metrics": financial.get("supply", {})
            },
            "onchain_analysis": {
                "tvl": onchain.get("tvl", 0),
                "tvl_changes": onchain.get("tvl_changes", {}),
                "activity_metrics": onchain.get("activity", {}),
                "network_metrics": onchain.get("network", {}),
                "contract_metrics": onchain.get("contracts", {})
            },
            "development_analysis": {
                "activity_metrics": github.get("activity", {}),
                "contributor_metrics": github.get("contributors", {}),
                "community_metrics": github.get("community", {}),
                "development_metrics": github.get("development", {}),
                "quality_metrics": github.get("quality", {})
            },
            "security_analysis": {
                "audit_status": security.get("audit", {}),
                "contract_verification": security.get("contract", {}),
                "security_features": security.get("security_features", {})
            },
            "ml_analysis": {
                "investment_score": ml.get("investment_score", 0),
                "confidence_score": ml.get("confidence_score", 0),
                "model_details": {
                    "model_name": ml.get("model_name", "Unknown"),
                    "prediction_horizon": ml.get("prediction_horizon", "Unknown"),
                    "features_used": ml.get("features_used", [])
                }
            },
            "investment_recommendations": {
                "primary_recommendation": recommendations.get("investment_recommendation", "HOLD"),
                "confidence_level": recommendations.get("confidence_level", "medium"),
                "reasoning": recommendations.get("reasoning", []),
                "action_items": recommendations.get("action_items", []),
                "price_targets": recommendations.get("price_targets", {}),
                "time_horizons": recommendations.get("time_horizons", {})
            },
            "risk_assessment": {
                "overall_risk_score": dashboard_data.get("risk_assessment", {}).get("overall_risk_score", 50),
                "risk_categories": dashboard_data.get("risk_assessment", {})
            },
            "investment_signals": dashboard_data.get("investment_signals", {}),
            "next_steps": [
                "Monitor key metrics daily",
                "Set up price alerts",
                "Review weekly reports",
                "Adjust position based on new data",
                "Consider dollar-cost averaging strategy"
            ]
        }
        
        # Сохранить отчет
        with open("matic_investment_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📄 Investment report saved to matic_investment_report.json")
        
        # Вывести краткое резюме
        logger.info("📊 INVESTMENT ANALYSIS SUMMARY:")
        logger.info(f"   💰 Current Price: ${financial.get('current_price', 0):.4f}")
        logger.info(f"   📈 24h Change: {financial.get('price_changes', {}).get('24h', 0):+.2f}%")
        logger.info(f"   🎯 Investment Score: {investment_report.get('investment_score', 0)}/100")
        logger.info(f"   ⚠️ Risk Score: {investment_report.get('risk_score', 0)}/100")
        logger.info(f"   💡 Recommendation: {recommendations.get('investment_recommendation', 'HOLD')}")
        logger.info(f"   🤖 ML Prediction: {ml.get('investment_score', 0):.2f} (confidence: {ml.get('confidence_score', 0):.2f})")
        
    except Exception as e:
        logger.error(f"Error generating investment report: {e}")

async def main():
    """Основная функция"""
    logger.info("🚀 MATIC Investment Analysis System")
    logger.info("=" * 50)
    
    start_time = datetime.utcnow()
    
    try:
        success = await run_complete_analysis()
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            logger.info("✅ Analysis completed successfully!")
            logger.info(f"⏱️ Total execution time: {duration:.2f} seconds")
            logger.info("📁 Generated files:")
            logger.info("   - matic_metrics_collection.json (Raw metrics data)")
            logger.info("   - matic_dashboard_data.json (Dashboard data)")
            logger.info("   - matic_investment_dashboard.html (Interactive dashboard)")
            logger.info("   - matic_investment_report.json (Executive report)")
            logger.info("")
            logger.info("🌐 Open matic_investment_dashboard.html in your browser to view the dashboard")
        else:
            logger.error("❌ Analysis failed!")
            
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")

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
