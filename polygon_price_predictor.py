#!/usr/bin/env python3
"""
ML модель для предсказания цены MATIC на следующую неделю
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import joblib
import json

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics, MLPrediction
)
from config.settings import settings

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
from sklearn.svm import SVR

class PolygonPricePredictor:
    """ML модель для предсказания цены MATIC"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.polygon_asset_id = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = []
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def setup_polygon_asset(self):
        """Настроить актив Polygon"""
        try:
            matic_asset = self.session.query(CryptoAsset).filter(
                CryptoAsset.symbol == "MATIC"
            ).first()
            
            if not matic_asset:
                raise ValueError("MATIC asset not found in database")
            
            self.polygon_asset_id = matic_asset.id
            logger.info(f"Polygon asset setup complete. Asset ID: {self.polygon_asset_id}")
            
        except Exception as e:
            logger.error(f"Error setting up Polygon asset: {e}")
            raise
    
    async def prepare_training_data(self, days_back: int = 365) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовить данные для обучения модели"""
        try:
            # Получить диапазон дат
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Preparing training data from {start_date} to {end_date}")
            
            # Собрать все метрики за период
            features_data = []
            targets_data = []
            
            # Получить все метрики для MATIC
            onchain_metrics = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == self.polygon_asset_id,
                OnChainMetrics.timestamp >= start_date,
                OnChainMetrics.timestamp <= end_date
            ).order_by(OnChainMetrics.timestamp).all()
            
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            github_metrics = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == self.polygon_asset_id,
                GitHubMetrics.timestamp >= start_date,
                GitHubMetrics.timestamp <= end_date
            ).order_by(GitHubMetrics.timestamp).all()
            
            community_metrics = self.session.query(CommunityMetrics).filter(
                CommunityMetrics.asset_id == self.polygon_asset_id,
                CommunityMetrics.timestamp >= start_date,
                CommunityMetrics.timestamp <= end_date
            ).order_by(CommunityMetrics.timestamp).all()
            
            trending_metrics = self.session.query(TrendingMetrics).filter(
                TrendingMetrics.asset_id == self.polygon_asset_id,
                TrendingMetrics.timestamp >= start_date,
                TrendingMetrics.timestamp <= end_date
            ).order_by(TrendingMetrics.timestamp).all()
            
            network_metrics = self.session.query(NetworkMetrics).filter(
                NetworkMetrics.timestamp >= start_date,
                NetworkMetrics.timestamp <= end_date
            ).order_by(NetworkMetrics.timestamp).all()
            
            # Создать словари для быстрого поиска по времени
            onchain_dict = {m.timestamp: m for m in onchain_metrics}
            financial_dict = {m.timestamp: m for m in financial_metrics}
            github_dict = {m.timestamp: m for m in github_metrics}
            community_dict = {m.timestamp: m for m in community_metrics}
            trending_dict = {m.timestamp: m for m in trending_metrics}
            network_dict = {m.timestamp: m for m in network_metrics}
            
            # Собрать все временные метки
            all_timestamps = set()
            for metrics_dict in [onchain_dict, financial_dict, github_dict, 
                               community_dict, trending_dict, network_dict]:
                all_timestamps.update(metrics_dict.keys())
            
            # Создать признаки для каждого временного момента
            for timestamp in sorted(all_timestamps):
                feature_vector = {}
                
                # On-chain признаки
                onchain = onchain_dict.get(timestamp)
                if onchain:
                    feature_vector.update({
                        "tvl": float(onchain.tvl or 0),
                        "tvl_change_24h": float(onchain.tvl_change_24h or 0),
                        "tvl_change_7d": float(onchain.tvl_change_7d or 0),
                        "daily_transactions": float(onchain.daily_transactions or 0),
                        "active_addresses_24h": float(onchain.active_addresses_24h or 0),
                        "transaction_volume_24h": float(onchain.transaction_volume_24h or 0),
                        "gas_price_avg": float(onchain.gas_price_avg or 0),
                        "contract_interactions_24h": float(onchain.contract_interactions_24h or 0),
                        "liquidity_pools_count": float(onchain.liquidity_pools_count or 0),
                        "liquidity_pools_tvl": float(onchain.liquidity_pools_tvl or 0),
                    })
                
                # Финансовые признаки
                financial = financial_dict.get(timestamp)
                if financial:
                    feature_vector.update({
                        "price_usd": float(financial.price_usd or 0),
                        "market_cap": float(financial.market_cap or 0),
                        "volume_24h": float(financial.volume_24h or 0),
                        "volatility_24h": float(financial.volatility_24h or 0),
                        "price_change_24h": float(financial.price_change_24h or 0),
                        "price_change_7d": float(financial.price_change_7d or 0),
                        "price_change_30d": float(financial.price_change_30d or 0),
                        "volume_market_cap_ratio": float(financial.volume_market_cap_ratio or 0),
                        "liquidity_score": float(financial.liquidity_score or 0),
                    })
                    
                    # Использовать изменение цены через 7 дней как цель
                    future_timestamp = timestamp + timedelta(days=7)
                    future_financial = next((m for m in financial_metrics 
                                           if abs((m.timestamp - future_timestamp).total_seconds()) < 3600), None)
                    
                    if future_financial and financial.price_usd and future_financial.price_usd:
                        # Рассчитать процентное изменение цены за 7 дней
                        price_change_7d = ((future_financial.price_usd - financial.price_usd) / financial.price_usd) * 100
                        
                        features_data.append(feature_vector)
                        targets_data.append(price_change_7d)
                
                # GitHub признаки
                github = github_dict.get(timestamp)
                if github:
                    feature_vector.update({
                        "commits_24h": float(github.commits_24h or 0),
                        "commits_7d": float(github.commits_7d or 0),
                        "commits_30d": float(github.commits_30d or 0),
                        "active_contributors_30d": float(github.active_contributors_30d or 0),
                        "stars": float(github.stars or 0),
                        "forks": float(github.forks or 0),
                        "open_issues": float(github.open_issues or 0),
                        "open_prs": float(github.open_prs or 0),
                        "code_quality_score": float(github.code_quality_score or 0),
                    })
                
                # Сообщество признаки
                community = community_dict.get(timestamp)
                if community:
                    feature_vector.update({
                        "twitter_followers": float(community.twitter_followers or 0),
                        "telegram_members": float(community.telegram_members or 0),
                        "discord_members": float(community.discord_members or 0),
                        "social_engagement_rate": float(community.social_engagement_rate or 0),
                        "brand_awareness_score": float(community.brand_awareness_score or 0),
                    })
                
                # Трендовые признаки
                trending = trending_dict.get(timestamp)
                if trending:
                    feature_vector.update({
                        "momentum_score": float(trending.momentum_score or 0),
                        "trend_strength": float(trending.trend_strength or 0),
                        "fear_greed_index": float(trending.fear_greed_index or 0),
                        "social_sentiment": float(trending.social_sentiment or 0),
                        "anomaly_score": float(trending.anomaly_score or 0),
                    })
                
                # Сетевые признаки
                network = network_dict.get(timestamp)
                if network:
                    feature_vector.update({
                        "block_time_avg": float(network.block_time_avg or 0),
                        "network_utilization": float(network.network_utilization or 0),
                        "validator_count": float(network.validator_count or 0),
                        "gas_price_avg_network": float(network.gas_price_avg or 0),
                    })
            
            if not features_data:
                logger.warning("No training data found")
                return pd.DataFrame(), pd.Series()
            
            # Преобразовать в DataFrame
            features_df = pd.DataFrame(features_data)
            targets_series = pd.Series(targets_data)
            
            # Обработать пропущенные значения
            features_df = self._handle_missing_values(features_df)
            
            # Инженерия признаков
            features_df = self._engineer_features(features_df)
            
            # Удалить строки с NaN целями
            valid_indices = ~targets_series.isna()
            features_df = features_df[valid_indices]
            targets_series = targets_series[valid_indices]
            
            logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
            
            return features_df, targets_series
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработать пропущенные значения"""
        # Заполнить пропущенные значения медианой для числовых столбцов
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Заполнить оставшиеся пропущенные значения нулями
        df = df.fillna(0)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать дополнительные признаки"""
        # Взаимодействие признаков
        if "price_usd" in df.columns and "volume_24h" in df.columns:
            df["price_volume_interaction"] = df["price_usd"] * df["volume_24h"]
        
        if "market_cap" in df.columns and "tvl" in df.columns:
            df["tvl_market_cap_ratio"] = df["tvl"] / (df["market_cap"] + 1)
        
        if "daily_transactions" in df.columns and "active_addresses_24h" in df.columns:
            df["transactions_per_address"] = df["daily_transactions"] / (df["active_addresses_24h"] + 1)
        
        # Волатильность признаков
        for col in ["price_usd", "tvl", "volume_24h"]:
            if col in df.columns:
                df[f"{col}_volatility"] = df[col].rolling(window=7, min_periods=1).std()
        
        # Технические индикаторы
        if "price_usd" in df.columns:
            df["price_ma_7"] = df["price_usd"].rolling(window=7, min_periods=1).mean()
            df["price_ma_30"] = df["price_usd"].rolling(window=30, min_periods=1).mean()
            df["price_rsi"] = self._calculate_rsi(df["price_usd"])
        
        # Логарифмические преобразования для больших значений
        for col in ["market_cap", "volume_24h", "tvl", "transaction_volume_24h"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Рассчитать RSI индикатор"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Заполнить NaN средним значением
    
    async def train_models(self, features_df: pd.DataFrame, targets_series: pd.Series):
        """Обучить модели машинного обучения"""
        try:
            if features_df.empty or targets_series.empty:
                logger.warning("No data available for training")
                return
            
            logger.info("Starting model training...")
            
            # Разделить данные на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets_series, test_size=0.2, random_state=42
            )
            
            # Масштабировать признаки
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers["standard"] = scaler
            
            # Определить модели для обучения
            models_config = {
                "random_forest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                "xgboost": xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                "linear_regression": LinearRegression(),
                "ridge_regression": Ridge(alpha=1.0, random_state=42),
                "lasso_regression": Lasso(alpha=0.1, random_state=42),
                "neural_network": MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    random_state=42,
                    max_iter=1000
                ),
                "svr": SVR(kernel="rbf", C=1.0, gamma="scale")
            }
            
            # Обучить каждую модель
            for model_name, model in models_config.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Обучить модель
                    model.fit(X_train_scaled, y_train)
                    
                    # Сделать предсказания
                    y_pred = model.predict(X_test_scaled)
                    
                    # Рассчитать метрики
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Кросс-валидация
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=TimeSeriesSplit(n_splits=5),
                        scoring='neg_mean_squared_error'
                    )
                    
                    # Важность признаков
                    feature_importance = {}
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(features_df.columns, model.feature_importances_))
                        self.feature_importance[model_name] = feature_importance
                    
                    # Сохранить модель
                    self.models[model_name] = model
                    
                    logger.info(f"{model_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
                    logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            logger.info(f"Training completed. Trained {len(self.models)} models.")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    async def predict_price_change_7d(self, model_name: str = "ensemble") -> Dict[str, Any]:
        """Предсказать изменение цены MATIC на 7 дней"""
        try:
            # Получить последние признаки
            features = await self._get_latest_features()
            if not features:
                raise ValueError("No features available for prediction")
            
            # Преобразовать в DataFrame
            features_df = pd.DataFrame([features])
            features_df = self._handle_missing_values(features_df)
            features_df = self._engineer_features(features_df)
            
            # Масштабировать признаки
            if "standard" in self.scalers:
                features_scaled = self.scalers["standard"].transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Сделать предсказание
            if model_name == "ensemble":
                predictions = []
                model_weights = {
                    "random_forest": 0.25,
                    "gradient_boosting": 0.25,
                    "xgboost": 0.25,
                    "neural_network": 0.15,
                    "ridge_regression": 0.10
                }
                
                weighted_predictions = []
                for name, model in self.models.items():
                    if name in model_weights and hasattr(model, 'predict'):
                        pred = model.predict(features_scaled)[0]
                        weight = model_weights[name]
                        weighted_predictions.append(pred * weight)
                
                if weighted_predictions:
                    prediction_value = sum(weighted_predictions)
                    confidence_score = 1.0 - (np.std(weighted_predictions) / (abs(np.mean(weighted_predictions)) + 1e-8))
                else:
                    prediction_value = 0.0
                    confidence_score = 0.0
            else:
                if model_name in self.models:
                    model = self.models[model_name]
                    prediction_value = model.predict(features_scaled)[0]
                    confidence_score = 0.8  # Базовая уверенность
                else:
                    raise ValueError(f"Model {model_name} not found")
            
            # Получить текущую цену
            current_price = features.get("price_usd", 0)
            predicted_price = current_price * (1 + prediction_value / 100)
            
            # Создать результат
            result = {
                "model_name": model_name,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change_7d_percent": prediction_value,
                "confidence_score": confidence_score,
                "prediction_timestamp": datetime.utcnow(),
                "features_used": list(features.keys()),
                "feature_importance": self.feature_importance.get(model_name, {})
            }
            
            # Сохранить предсказание в базу данных
            await self._save_prediction(result)
            
            logger.info(f"Price prediction: {prediction_value:.2f}% change, confidence: {confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making price prediction: {e}")
            raise
    
    async def _get_latest_features(self) -> Optional[Dict[str, Any]]:
        """Получить последние признаки для предсказания"""
        try:
            features = {}
            
            # Получить последние метрики
            latest_onchain = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == self.polygon_asset_id
            ).order_by(OnChainMetrics.timestamp.desc()).first()
            
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_github = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == self.polygon_asset_id
            ).order_by(GitHubMetrics.timestamp.desc()).first()
            
            latest_community = self.session.query(CommunityMetrics).filter(
                CommunityMetrics.asset_id == self.polygon_asset_id
            ).order_by(CommunityMetrics.timestamp.desc()).first()
            
            latest_trending = self.session.query(TrendingMetrics).filter(
                TrendingMetrics.asset_id == self.polygon_asset_id
            ).order_by(TrendingMetrics.timestamp.desc()).first()
            
            latest_network = self.session.query(NetworkMetrics).order_by(
                NetworkMetrics.timestamp.desc()
            ).first()
            
            # Извлечь признаки
            if latest_onchain:
                features.update({
                    "tvl": float(latest_onchain.tvl or 0),
                    "tvl_change_24h": float(latest_onchain.tvl_change_24h or 0),
                    "tvl_change_7d": float(latest_onchain.tvl_change_7d or 0),
                    "daily_transactions": float(latest_onchain.daily_transactions or 0),
                    "active_addresses_24h": float(latest_onchain.active_addresses_24h or 0),
                    "transaction_volume_24h": float(latest_onchain.transaction_volume_24h or 0),
                    "gas_price_avg": float(latest_onchain.gas_price_avg or 0),
                    "contract_interactions_24h": float(latest_onchain.contract_interactions_24h or 0),
                    "liquidity_pools_count": float(latest_onchain.liquidity_pools_count or 0),
                    "liquidity_pools_tvl": float(latest_onchain.liquidity_pools_tvl or 0),
                })
            
            if latest_financial:
                features.update({
                    "price_usd": float(latest_financial.price_usd or 0),
                    "market_cap": float(latest_financial.market_cap or 0),
                    "volume_24h": float(latest_financial.volume_24h or 0),
                    "volatility_24h": float(latest_financial.volatility_24h or 0),
                    "price_change_24h": float(latest_financial.price_change_24h or 0),
                    "price_change_7d": float(latest_financial.price_change_7d or 0),
                    "price_change_30d": float(latest_financial.price_change_30d or 0),
                    "volume_market_cap_ratio": float(latest_financial.volume_market_cap_ratio or 0),
                    "liquidity_score": float(latest_financial.liquidity_score or 0),
                })
            
            if latest_github:
                features.update({
                    "commits_24h": float(latest_github.commits_24h or 0),
                    "commits_7d": float(latest_github.commits_7d or 0),
                    "commits_30d": float(latest_github.commits_30d or 0),
                    "active_contributors_30d": float(latest_github.active_contributors_30d or 0),
                    "stars": float(latest_github.stars or 0),
                    "forks": float(latest_github.forks or 0),
                    "open_issues": float(latest_github.open_issues or 0),
                    "open_prs": float(latest_github.open_prs or 0),
                    "code_quality_score": float(latest_github.code_quality_score or 0),
                })
            
            if latest_community:
                features.update({
                    "twitter_followers": float(latest_community.twitter_followers or 0),
                    "telegram_members": float(latest_community.telegram_members or 0),
                    "discord_members": float(latest_community.discord_members or 0),
                    "social_engagement_rate": float(latest_community.social_engagement_rate or 0),
                    "brand_awareness_score": float(latest_community.brand_awareness_score or 0),
                })
            
            if latest_trending:
                features.update({
                    "momentum_score": float(latest_trending.momentum_score or 0),
                    "trend_strength": float(latest_trending.trend_strength or 0),
                    "fear_greed_index": float(latest_trending.fear_greed_index or 0),
                    "social_sentiment": float(latest_trending.social_sentiment or 0),
                    "anomaly_score": float(latest_trending.anomaly_score or 0),
                })
            
            if latest_network:
                features.update({
                    "block_time_avg": float(latest_network.block_time_avg or 0),
                    "network_utilization": float(latest_network.network_utilization or 0),
                    "validator_count": float(latest_network.validator_count or 0),
                    "gas_price_avg_network": float(latest_network.gas_price_avg or 0),
                })
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error getting latest features: {e}")
            return None
    
    async def _save_prediction(self, prediction_result: Dict[str, Any]):
        """Сохранить предсказание в базу данных"""
        try:
            prediction = MLPrediction(
                asset_id=self.polygon_asset_id,
                model_name=prediction_result["model_name"],
                prediction_type="price_prediction_7d",
                prediction_value=prediction_result["price_change_7d_percent"],
                confidence_score=prediction_result["confidence_score"],
                prediction_horizon="7d",
                features_used=prediction_result["features_used"],
                model_version="1.0"
            )
            
            self.session.add(prediction)
            self.session.commit()
            
            logger.info("Prediction saved to database")
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            self.session.rollback()
    
    async def get_prediction_analysis(self) -> Dict[str, Any]:
        """Получить анализ предсказаний"""
        try:
            # Получить последние предсказания
            recent_predictions = self.session.query(MLPrediction).filter(
                MLPrediction.asset_id == self.polygon_asset_id,
                MLPrediction.prediction_type == "price_prediction_7d"
            ).order_by(MLPrediction.created_at.desc()).limit(10).all()
            
            if not recent_predictions:
                return {"message": "No predictions available"}
            
            # Анализ предсказаний
            predictions_data = []
            for pred in recent_predictions:
                predictions_data.append({
                    "timestamp": pred.created_at,
                    "model": pred.model_name,
                    "prediction": pred.prediction_value,
                    "confidence": pred.confidence_score
                })
            
            # Статистика
            avg_prediction = np.mean([p["prediction"] for p in predictions_data])
            avg_confidence = np.mean([p["confidence"] for p in predictions_data])
            
            # Тренд предсказаний
            recent_trend = "bullish" if avg_prediction > 2 else "bearish" if avg_prediction < -2 else "sideways"
            
            analysis = {
                "recent_predictions": predictions_data,
                "average_prediction_7d": avg_prediction,
                "average_confidence": avg_confidence,
                "prediction_trend": recent_trend,
                "total_predictions": len(recent_predictions),
                "model_performance": self._get_model_performance_summary()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting prediction analysis: {e}")
            return {"error": str(e)}
    
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Получить сводку производительности моделей"""
        return {
            "trained_models": list(self.models.keys()),
            "available_scalers": list(self.scalers.keys()),
            "feature_importance_available": list(self.feature_importance.keys())
        }
    
    async def save_models(self, filepath: str = "models/polygon"):
        """Сохранить обученные модели"""
        try:
            import os
            os.makedirs(filepath, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = f"{filepath}/{model_name}_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
            
            for scaler_name, scaler in self.scalers.items():
                scaler_path = f"{filepath}/{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler: {scaler_path}")
            
            # Сохранить важность признаков
            importance_path = f"{filepath}/feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self, filepath: str = "models/polygon"):
        """Загрузить обученные модели"""
        try:
            import os
            if not os.path.exists(filepath):
                logger.warning(f"Model directory does not exist: {filepath}")
                return
            
            # Загрузить модели
            for filename in os.listdir(filepath):
                if filename.endswith("_model.joblib"):
                    model_name = filename.replace("_model.joblib", "")
                    model_path = os.path.join(filepath, filename)
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                
                elif filename.endswith("_scaler.joblib"):
                    scaler_name = filename.replace("_scaler.joblib", "")
                    scaler_path = os.path.join(filepath, filename)
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler: {scaler_name}")
            
            # Загрузить важность признаков
            importance_path = os.path.join(filepath, "feature_importance.json")
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                logger.info("Loaded feature importance")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

async def main():
    """Основная функция"""
    logger.info("🚀 Starting Polygon price prediction...")
    
    try:
        # Инициализировать базу данных
        init_db()
        
        async with PolygonPricePredictor() as predictor:
            # Настроить актив Polygon
            await predictor.setup_polygon_asset()
            
            # Подготовить данные для обучения
            features_df, targets_series = await predictor.prepare_training_data(days_back=180)
            
            if not features_df.empty:
                # Обучить модели
                await predictor.train_models(features_df, targets_series)
                
                # Сделать предсказание
                prediction = await predictor.predict_price_change_7d()
                
                # Получить анализ
                analysis = await predictor.get_prediction_analysis()
                
                # Сохранить модели
                await predictor.save_models()
                
                # Вывести результаты
                logger.info("🎯 PREDICTION RESULTS:")
                logger.info(f"Current MATIC Price: ${prediction['current_price']:.4f}")
                logger.info(f"Predicted 7-day Change: {prediction['price_change_7d_percent']:.2f}%")
                logger.info(f"Predicted Price: ${prediction['predicted_price']:.4f}")
                logger.info(f"Confidence Score: {prediction['confidence_score']:.2f}")
                logger.info(f"Model Used: {prediction['model_name']}")
                
                logger.info("📊 ANALYSIS:")
                logger.info(f"Average Prediction: {analysis['average_prediction_7d']:.2f}%")
                logger.info(f"Average Confidence: {analysis['average_confidence']:.2f}")
                logger.info(f"Prediction Trend: {analysis['prediction_trend']}")
                
            else:
                logger.warning("No training data available. Please collect data first.")
        
        logger.info("🎉 Polygon price prediction completed!")
        
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
