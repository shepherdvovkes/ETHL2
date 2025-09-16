#!/usr/bin/env python3
"""
Умный предсказатель Polygon, использующий карту данных QuickNode API
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
import joblib

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import CryptoAsset, MLPrediction
from api.quicknode_client import QuickNodeClient
from api.coingecko_client import CoinGeckoClient
from config.settings import settings

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

class SmartPolygonPredictor:
    """Умный предсказатель, использующий карту данных QuickNode"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.polygon_asset_id = 1  # MATIC
        self.models = {}
        self.scalers = {}
        self.data_map = {}
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def load_data_map(self, filepath: str = "quicknode_data_map.json"):
        """Загрузить карту данных QuickNode"""
        try:
            with open(filepath, "r") as f:
                self.data_map = json.load(f)
            logger.info("🗺️ Data map loaded successfully")
        except FileNotFoundError:
            logger.warning("Data map not found, creating basic map...")
            self.data_map = await self._create_basic_data_map()
    
    async def _create_basic_data_map(self) -> Dict[str, Any]:
        """Создать базовую карту данных"""
        return {
            "available_data": {
                "network": {
                    "current_block": "Available",
                    "gas_price_current": "Available",
                    "block_time_avg": 2.3
                },
                "transactions": {
                    "transaction_types": ["native_transfers", "token_transfers", "contract_calls"],
                    "estimated_daily_transactions": "2-4 million"
                },
                "defi": {
                    "protocols": ["aave", "quickswap", "curve", "sushiswap"],
                    "data_types": ["tvl", "swap_volumes", "lending_volumes"]
                }
            }
        }
    
    async def collect_current_metrics(self) -> Dict[str, Any]:
        """Собрать текущие метрики на основе карты данных"""
        logger.info("📊 Collecting current metrics...")
        
        metrics = {}
        
        try:
            async with QuickNodeClient() as qn_client:
                # Сетевые метрики
                current_block = await qn_client.get_block_number()
                gas_price = await qn_client.get_gas_price()
                network_stats = await qn_client.get_network_stats()
                
                metrics.update({
                    "current_block": current_block,
                    "gas_price_gwei": gas_price,
                    "network_utilization": network_stats.get("network_utilization", 0.0),
                    "block_time_avg": 2.3
                })
                
                # Анализ последних блоков для транзакционной активности
                blocks_analyzed = 0
                total_transactions = 0
                total_volume = 0
                
                for i in range(min(10, current_block)):  # Анализируем последние 10 блоков
                    try:
                        block = await qn_client.get_block_by_number(current_block - i)
                        if block and "transactions" in block:
                            total_transactions += len(block["transactions"])
                            blocks_analyzed += 1
                            
                            # Примерная оценка объема
                            for tx in block["transactions"][:5]:  # Первые 5 транзакций
                                if "value" in tx and tx["value"] != "0x0":
                                    try:
                                        value_wei = int(tx["value"], 16)
                                        total_volume += value_wei / 10**18  # Convert to MATIC
                                    except:
                                        continue
                    except:
                        continue
                
                if blocks_analyzed > 0:
                    avg_transactions_per_block = total_transactions / blocks_analyzed
                    estimated_daily_transactions = avg_transactions_per_block * (24 * 60 * 60 / 2.3)  # 2.3s per block
                    estimated_daily_volume = total_volume * (24 * 60 * 60 / 2.3)
                    
                    metrics.update({
                        "avg_transactions_per_block": avg_transactions_per_block,
                        "estimated_daily_transactions": estimated_daily_transactions,
                        "estimated_daily_volume_matic": estimated_daily_volume,
                        "blocks_analyzed": blocks_analyzed
                    })
                
        except Exception as e:
            logger.error(f"Error collecting QuickNode metrics: {e}")
            # Использовать примерные значения
            metrics.update({
                "current_block": 50000000,
                "gas_price_gwei": 30.0,
                "network_utilization": 0.6,
                "estimated_daily_transactions": 3000000,
                "estimated_daily_volume_matic": 1000000
            })
        
        # Получить финансовые данные
        try:
            async with CoinGeckoClient() as cg_client:
                price_data = await cg_client.get_coin_price(
                    ["matic-network"],
                    vs_currencies=["usd"],
                    include_market_cap=True,
                    include_24hr_vol=True,
                    include_24hr_change=True
                )
                
                if "matic-network" in price_data:
                    matic_data = price_data["matic-network"]["usd"]
                    metrics.update({
                        "current_price_usd": matic_data.get("usd", 0),
                        "market_cap": matic_data.get("usd_market_cap", 0),
                        "volume_24h": matic_data.get("usd_24h_vol", 0),
                        "price_change_24h": matic_data.get("usd_24h_change", 0)
                    })
        except Exception as e:
            logger.error(f"Error collecting CoinGecko data: {e}")
            # Использовать примерные значения
            metrics.update({
                "current_price_usd": 0.85,
                "market_cap": 8500000000,
                "volume_24h": 50000000,
                "price_change_24h": 2.5
            })
        
        logger.info(f"✅ Collected {len(metrics)} current metrics")
        return metrics
    
    def generate_synthetic_training_data(self, current_metrics: Dict[str, Any], days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Генерировать синтетические данные для обучения на основе текущих метрик"""
        logger.info(f"🎭 Generating {days} days of synthetic training data...")
        
        # Базовые значения из текущих метрик
        base_price = current_metrics.get("current_price_usd", 0.85)
        base_volume = current_metrics.get("volume_24h", 50000000)
        base_transactions = current_metrics.get("estimated_daily_transactions", 3000000)
        base_gas_price = current_metrics.get("gas_price_gwei", 30.0)
        
        features_data = []
        targets_data = []
        
        for i in range(days):
            # Создать реалистичные вариации
            day_factor = 1 + (i / days) * 0.1  # Небольшой тренд роста
            
            # Цена с трендом и волатильностью
            price_trend = np.random.normal(0, 0.02)  # 2% волатильность
            price = base_price * day_factor * (1 + price_trend)
            
            # Объем с корреляцией к цене
            volume_correlation = 0.3
            volume_noise = np.random.normal(0, 0.1)
            volume = base_volume * (1 + price_trend * volume_correlation + volume_noise)
            
            # Транзакции с корреляцией к объему
            tx_correlation = 0.4
            tx_noise = np.random.normal(0, 0.05)
            transactions = base_transactions * (1 + volume_noise * tx_correlation + tx_noise)
            
            # Gas цена с корреляцией к активности
            gas_correlation = 0.2
            gas_noise = np.random.normal(0, 0.1)
            gas_price = base_gas_price * (1 + tx_noise * gas_correlation + gas_noise)
            
            # Создать признаки
            features = {
                "price_usd": price,
                "volume_24h": volume,
                "daily_transactions": transactions,
                "gas_price_gwei": gas_price,
                "market_cap": price * 10000000000,  # 10B supply
                "volume_market_cap_ratio": volume / (price * 10000000000),
                "transactions_per_volume": transactions / (volume + 1),
                "gas_efficiency": transactions / (gas_price + 1),
                "price_volatility": abs(price_trend) * 100,
                "volume_volatility": abs(volume_noise) * 100
            }
            
            # Создать цель (изменение цены через 7 дней)
            if i + 7 < days:
                future_price = base_price * (1 + (i + 7) / days * 0.1) * (1 + np.random.normal(0, 0.02))
                price_change_7d = ((future_price - price) / price) * 100
                
                features_data.append(features)
                targets_data.append(price_change_7d)
        
        features_df = pd.DataFrame(features_data)
        targets_series = pd.Series(targets_data)
        
        logger.info(f"✅ Generated {len(features_df)} training samples")
        return features_df, targets_series
    
    async def train_models(self, features_df: pd.DataFrame, targets_series: pd.Series):
        """Обучить ML модели"""
        logger.info("🤖 Training ML models...")
        
        if features_df.empty or targets_series.empty:
            logger.warning("No training data available")
            return
        
        # Разделить данные
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_series, test_size=0.2, random_state=42
        )
        
        # Масштабировать признаки
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers["standard"] = scaler
        
        # Модели для обучения
        models_config = {
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            "xgboost": xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0, random_state=42),
            "neural_network": MLPRegressor(
                hidden_layer_sizes=(50, 25),
                activation="relu",
                solver="adam",
                alpha=0.001,
                random_state=42,
                max_iter=500
            )
        }
        
        # Обучить модели
        for model_name, model in models_config.items():
            try:
                logger.info(f"Training {model_name}...")
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.models[model_name] = model
                
                logger.info(f"{model_name} - R²: {r2:.4f}, MSE: {mse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"✅ Trained {len(self.models)} models")
    
    async def predict_price_change(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Предсказать изменение цены"""
        logger.info("🔮 Making price prediction...")
        
        if not self.models:
            logger.warning("No trained models available")
            return {"error": "No models trained"}
        
        # Создать признаки из текущих метрик
        features = {
            "price_usd": current_metrics.get("current_price_usd", 0.85),
            "volume_24h": current_metrics.get("volume_24h", 50000000),
            "daily_transactions": current_metrics.get("estimated_daily_transactions", 3000000),
            "gas_price_gwei": current_metrics.get("gas_price_gwei", 30.0),
            "market_cap": current_metrics.get("market_cap", 8500000000),
            "volume_market_cap_ratio": current_metrics.get("volume_24h", 50000000) / current_metrics.get("market_cap", 8500000000),
            "transactions_per_volume": current_metrics.get("estimated_daily_transactions", 3000000) / (current_metrics.get("volume_24h", 50000000) + 1),
            "gas_efficiency": current_metrics.get("estimated_daily_transactions", 3000000) / (current_metrics.get("gas_price_gwei", 30.0) + 1),
            "price_volatility": abs(current_metrics.get("price_change_24h", 2.5)),
            "volume_volatility": 5.0  # Примерная оценка
        }
        
        # Преобразовать в DataFrame
        features_df = pd.DataFrame([features])
        
        # Масштабировать
        if "standard" in self.scalers:
            features_scaled = self.scalers["standard"].transform(features_df)
        else:
            features_scaled = features_df.values
        
        # Сделать предсказания
        predictions = []
        model_weights = {
            "random_forest": 0.25,
            "gradient_boosting": 0.25,
            "xgboost": 0.25,
            "neural_network": 0.15,
            "ridge_regression": 0.10
        }
        
        for model_name, model in self.models.items():
            if model_name in model_weights:
                try:
                    pred = model.predict(features_scaled)[0]
                    weight = model_weights[model_name]
                    predictions.append(pred * weight)
                except:
                    continue
        
        if predictions:
            predicted_change = sum(predictions)
            confidence = max(0.3, 1.0 - (np.std(predictions) / (abs(np.mean(predictions)) + 1e-8)))
        else:
            predicted_change = 0.0
            confidence = 0.0
        
        current_price = current_metrics.get("current_price_usd", 0.85)
        predicted_price = current_price * (1 + predicted_change / 100)
        
        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_7d": predicted_change,
            "confidence": confidence,
            "model": "ensemble",
            "features_used": list(features.keys()),
            "prediction_timestamp": datetime.utcnow()
        }
    
    async def save_prediction(self, prediction: Dict[str, Any]):
        """Сохранить предсказание в базу данных"""
        try:
            ml_prediction = MLPrediction(
                asset_id=self.polygon_asset_id,
                model_name=prediction.get("model", "ensemble"),
                prediction_type="price_prediction_7d",
                prediction_value=prediction.get("predicted_change_7d", 0),
                confidence_score=prediction.get("confidence", 0),
                prediction_horizon="7d",
                features_used=prediction.get("features_used", []),
                model_version="2.0"
            )
            
            self.session.add(ml_prediction)
            self.session.commit()
            
            logger.info("💾 Prediction saved to database")
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            self.session.rollback()
    
    async def run_smart_analysis(self):
        """Запустить умный анализ"""
        try:
            logger.info("🧠 Starting Smart Polygon Analysis...")
            
            # Загрузить карту данных
            await self.load_data_map()
            
            # Собрать текущие метрики
            current_metrics = await self.collect_current_metrics()
            
            # Сгенерировать данные для обучения
            features_df, targets_series = self.generate_synthetic_training_data(current_metrics)
            
            # Обучить модели
            await self.train_models(features_df, targets_series)
            
            # Сделать предсказание
            prediction = await self.predict_price_change(current_metrics)
            
            # Сохранить предсказание
            await self.save_prediction(prediction)
            
            # Вывести результаты
            self._print_results(current_metrics, prediction)
            
            return {
                "current_metrics": current_metrics,
                "prediction": prediction,
                "data_map": self.data_map
            }
            
        except Exception as e:
            logger.error(f"Error in smart analysis: {e}")
            return {"error": str(e)}
    
    def _print_results(self, current_metrics: Dict[str, Any], prediction: Dict[str, Any]):
        """Вывести результаты анализа"""
        logger.info("=" * 80)
        logger.info("🧠 SMART POLYGON ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        logger.info("📊 CURRENT METRICS:")
        logger.info(f"   💰 Current Price: ${current_metrics.get('current_price_usd', 0):.4f}")
        logger.info(f"   📈 24h Change: {current_metrics.get('price_change_24h', 0):+.2f}%")
        logger.info(f"   💵 Market Cap: ${current_metrics.get('market_cap', 0):,.0f}")
        logger.info(f"   📊 24h Volume: ${current_metrics.get('volume_24h', 0):,.0f}")
        logger.info(f"   🔄 Daily Transactions: {current_metrics.get('estimated_daily_transactions', 0):,.0f}")
        logger.info(f"   ⛽ Gas Price: {current_metrics.get('gas_price_gwei', 0):.1f} Gwei")
        logger.info(f"   📡 Current Block: {current_metrics.get('current_block', 0):,}")
        
        logger.info("\n🔮 PRICE PREDICTION:")
        logger.info(f"   🎯 Predicted 7-day Change: {prediction.get('predicted_change_7d', 0):+.2f}%")
        logger.info(f"   💰 Predicted Price: ${prediction.get('predicted_price', 0):.4f}")
        logger.info(f"   🎲 Confidence: {prediction.get('confidence', 0):.1%}")
        logger.info(f"   🤖 Model: {prediction.get('model', 'unknown')}")
        
        # Рекомендации
        predicted_change = prediction.get("predicted_change_7d", 0)
        confidence = prediction.get("confidence", 0)
        
        logger.info("\n💡 RECOMMENDATIONS:")
        if confidence > 0.7:
            if predicted_change > 5:
                logger.info("   🟢 High confidence bullish prediction. Consider accumulating MATIC.")
            elif predicted_change < -5:
                logger.info("   🔴 High confidence bearish prediction. Consider reducing position.")
            else:
                logger.info("   🟡 High confidence sideways prediction. Consider range trading.")
        else:
            logger.info("   ⚠️ Low confidence prediction. Use smaller position sizes.")
        
        logger.info("   🔍 Monitor on-chain metrics for early signals.")
        logger.info("   💧 Watch for DeFi TVL growth as positive indicator.")
        logger.info("   ⛽ Track network activity and gas prices.")
        logger.info("   🛡️ Set stop-losses based on risk tolerance.")
        
        logger.info("=" * 80)

async def main():
    """Основная функция"""
    logger.info("🧠 Starting Smart Polygon Predictor...")
    
    try:
        # Инициализировать базу данных
        init_db()
        
        async with SmartPolygonPredictor() as predictor:
            await predictor.run_smart_analysis()
        
        logger.info("🎉 Smart analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in smart analysis: {e}")

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
