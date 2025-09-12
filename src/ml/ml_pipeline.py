import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
import json
from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal
from database.models_v2 import (
    CryptoAsset, OnChainMetrics, FinancialMetrics, GitHubMetrics,
    TokenomicsMetrics, SecurityMetrics, CommunityMetrics, 
    PartnershipMetrics, NetworkMetrics, TrendingMetrics, MLPrediction
)
from config.settings import settings

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    name: str
    model_class: Any
    hyperparameters: Dict[str, Any]
    feature_importance: bool = True
    cross_validation: bool = True
    cv_folds: int = 5

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    name: str
    source_table: str
    source_column: str
    transformation: str = "none"  # none, log, sqrt, standardize, normalize
    lag_periods: List[int] = None  # For time series features
    rolling_windows: List[int] = None  # For rolling statistics
    interaction_features: List[str] = None  # For feature interactions

@dataclass
class MLPipelineResult:
    """Result of ML pipeline execution"""
    model_name: str
    prediction_type: str
    asset_id: int
    prediction_value: float
    confidence_score: float
    feature_importance: Dict[str, float]
    model_metrics: Dict[str, float]
    prediction_horizon: str
    created_at: datetime

class CryptoMLPipeline:
    """Advanced ML pipeline for crypto analytics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_configs = self._initialize_feature_configs()
        self.model_configs = self._initialize_model_configs()
        self.session = None
        
    def _initialize_feature_configs(self) -> List[FeatureConfig]:
        """Initialize feature engineering configurations"""
        return [
            # On-chain features
            FeatureConfig("tvl", "onchain_metrics", "tvl", "log"),
            FeatureConfig("tvl_change_24h", "onchain_metrics", "tvl_change_24h", "standardize"),
            FeatureConfig("tvl_change_7d", "onchain_metrics", "tvl_change_7d", "standardize"),
            FeatureConfig("daily_transactions", "onchain_metrics", "daily_transactions", "log"),
            FeatureConfig("active_addresses_24h", "onchain_metrics", "active_addresses_24h", "log"),
            FeatureConfig("transaction_volume_24h", "onchain_metrics", "transaction_volume_24h", "log"),
            FeatureConfig("gas_price_avg", "onchain_metrics", "gas_price_avg", "log"),
            FeatureConfig("contract_interactions_24h", "onchain_metrics", "contract_interactions_24h", "log"),
            
            # Financial features
            FeatureConfig("price_usd", "financial_metrics", "price_usd", "log"),
            FeatureConfig("market_cap", "financial_metrics", "market_cap", "log"),
            FeatureConfig("volume_24h", "financial_metrics", "volume_24h", "log"),
            FeatureConfig("volatility_24h", "financial_metrics", "volatility_24h", "standardize"),
            FeatureConfig("price_change_24h", "financial_metrics", "price_change_24h", "standardize"),
            FeatureConfig("price_change_7d", "financial_metrics", "price_change_7d", "standardize"),
            FeatureConfig("price_change_30d", "financial_metrics", "price_change_30d", "standardize"),
            
            # GitHub features
            FeatureConfig("commits_24h", "github_metrics", "commits_24h", "log"),
            FeatureConfig("commits_7d", "github_metrics", "commits_7d", "log"),
            FeatureConfig("commits_30d", "github_metrics", "commits_30d", "log"),
            FeatureConfig("active_contributors_30d", "github_metrics", "active_contributors_30d", "log"),
            FeatureConfig("stars", "github_metrics", "stars", "log"),
            FeatureConfig("forks", "github_metrics", "forks", "log"),
            FeatureConfig("open_issues", "github_metrics", "open_issues", "log"),
            FeatureConfig("open_prs", "github_metrics", "open_prs", "log"),
            FeatureConfig("code_quality_score", "github_metrics", "code_quality_score", "standardize"),
            
            # Tokenomics features
            FeatureConfig("circulating_supply", "tokenomics_metrics", "circulating_supply", "log"),
            FeatureConfig("total_supply", "tokenomics_metrics", "total_supply", "log"),
            FeatureConfig("inflation_rate", "tokenomics_metrics", "inflation_rate", "standardize"),
            FeatureConfig("burn_rate", "tokenomics_metrics", "burn_rate", "standardize"),
            
            # Security features
            FeatureConfig("audit_score", "security_metrics", "audit_score", "standardize"),
            FeatureConfig("vulnerability_score", "security_metrics", "vulnerability_score", "standardize"),
            FeatureConfig("contract_verified", "security_metrics", "contract_verified", "none"),
            
            # Community features
            FeatureConfig("twitter_followers", "community_metrics", "twitter_followers", "log"),
            FeatureConfig("telegram_members", "community_metrics", "telegram_members", "log"),
            FeatureConfig("discord_members", "community_metrics", "discord_members", "log"),
            FeatureConfig("social_engagement_rate", "community_metrics", "social_engagement_rate", "standardize"),
            
            # Network features
            FeatureConfig("block_time_avg", "network_metrics", "block_time_avg", "log"),
            FeatureConfig("network_utilization", "network_metrics", "network_utilization", "standardize"),
            FeatureConfig("validator_count", "network_metrics", "validator_count", "log"),
            
            # Trending features
            FeatureConfig("momentum_score", "trending_metrics", "momentum_score", "standardize"),
            FeatureConfig("fear_greed_index", "trending_metrics", "fear_greed_index", "standardize"),
            FeatureConfig("social_sentiment", "trending_metrics", "social_sentiment", "standardize"),
        ]
    
    def _initialize_model_configs(self) -> List[MLModelConfig]:
        """Initialize ML model configurations"""
        return [
            MLModelConfig(
                name="random_forest",
                model_class=RandomForestRegressor,
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                }
            ),
            MLModelConfig(
                name="gradient_boosting",
                model_class=GradientBoostingRegressor,
                hyperparameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                }
            ),
            MLModelConfig(
                name="linear_regression",
                model_class=LinearRegression,
                hyperparameters={}
            ),
            MLModelConfig(
                name="ridge_regression",
                model_class=Ridge,
                hyperparameters={
                    "alpha": 1.0,
                    "random_state": 42
                }
            ),
            MLModelConfig(
                name="lasso_regression",
                model_class=Lasso,
                hyperparameters={
                    "alpha": 0.1,
                    "random_state": 42
                }
            ),
            MLModelConfig(
                name="neural_network",
                model_class=MLPRegressor,
                hyperparameters={
                    "hidden_layer_sizes": (100, 50),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.001,
                    "random_state": 42,
                    "max_iter": 1000
                }
            )
        ]
    
    async def __aenter__(self):
        self.session = SessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def prepare_training_data(
        self, 
        asset_ids: List[int] = None,
        days_back: int = 365
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for ML models"""
        try:
            # Get date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Get all assets if not specified
            if asset_ids is None:
                assets = self.session.query(CryptoAsset).filter(CryptoAsset.is_active == True).all()
                asset_ids = [asset.id for asset in assets]
            
            # Collect features for all assets
            features_data = []
            targets_data = []
            
            for asset_id in asset_ids:
                asset_features = await self._extract_asset_features(asset_id, start_date, end_date)
                if asset_features:
                    features_data.extend(asset_features["features"])
                    targets_data.extend(asset_features["targets"])
            
            if not features_data:
                logger.warning("No training data found")
                return pd.DataFrame(), pd.Series()
            
            # Convert to DataFrames
            features_df = pd.DataFrame(features_data)
            targets_series = pd.Series(targets_data)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Feature engineering
            features_df = self._engineer_features(features_df)
            
            # Remove rows with NaN targets
            valid_indices = ~targets_series.isna()
            features_df = features_df[valid_indices]
            targets_series = targets_series[valid_indices]
            
            logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
            
            return features_df, targets_series
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    async def _extract_asset_features(
        self, 
        asset_id: int, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract features for a specific asset"""
        try:
            features = []
            targets = []
            
            # Get all metrics for the asset in the date range
            onchain_metrics = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == asset_id,
                OnChainMetrics.timestamp >= start_date,
                OnChainMetrics.timestamp <= end_date
            ).order_by(OnChainMetrics.timestamp).all()
            
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            github_metrics = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == asset_id,
                GitHubMetrics.timestamp >= start_date,
                GitHubMetrics.timestamp <= end_date
            ).order_by(GitHubMetrics.timestamp).all()
            
            tokenomics_metrics = self.session.query(TokenomicsMetrics).filter(
                TokenomicsMetrics.asset_id == asset_id,
                TokenomicsMetrics.timestamp >= start_date,
                TokenomicsMetrics.timestamp <= end_date
            ).order_by(TokenomicsMetrics.timestamp).all()
            
            security_metrics = self.session.query(SecurityMetrics).filter(
                SecurityMetrics.asset_id == asset_id,
                SecurityMetrics.timestamp >= start_date,
                SecurityMetrics.timestamp <= end_date
            ).order_by(SecurityMetrics.timestamp).all()
            
            community_metrics = self.session.query(CommunityMetrics).filter(
                CommunityMetrics.asset_id == asset_id,
                CommunityMetrics.timestamp >= start_date,
                CommunityMetrics.timestamp <= end_date
            ).order_by(CommunityMetrics.timestamp).all()
            
            network_metrics = self.session.query(NetworkMetrics).filter(
                NetworkMetrics.timestamp >= start_date,
                NetworkMetrics.timestamp <= end_date
            ).order_by(NetworkMetrics.timestamp).all()
            
            trending_metrics = self.session.query(TrendingMetrics).filter(
                TrendingMetrics.asset_id == asset_id,
                TrendingMetrics.timestamp >= start_date,
                TrendingMetrics.timestamp <= end_date
            ).order_by(TrendingMetrics.timestamp).all()
            
            # Combine metrics by timestamp
            all_timestamps = set()
            for metrics_list in [onchain_metrics, financial_metrics, github_metrics, 
                               tokenomics_metrics, security_metrics, community_metrics, 
                               network_metrics, trending_metrics]:
                for metric in metrics_list:
                    all_timestamps.add(metric.timestamp)
            
            # Create feature vectors for each timestamp
            for timestamp in sorted(all_timestamps):
                feature_vector = {}
                
                # Extract on-chain features
                onchain = next((m for m in onchain_metrics if m.timestamp == timestamp), None)
                if onchain:
                    feature_vector.update({
                        "tvl": onchain.tvl,
                        "tvl_change_24h": onchain.tvl_change_24h,
                        "tvl_change_7d": onchain.tvl_change_7d,
                        "daily_transactions": onchain.daily_transactions,
                        "active_addresses_24h": onchain.active_addresses_24h,
                        "transaction_volume_24h": onchain.transaction_volume_24h,
                        "gas_price_avg": onchain.gas_price_avg,
                        "contract_interactions_24h": onchain.contract_interactions_24h,
                    })
                
                # Extract financial features
                financial = next((m for m in financial_metrics if m.timestamp == timestamp), None)
                if financial:
                    feature_vector.update({
                        "price_usd": financial.price_usd,
                        "market_cap": financial.market_cap,
                        "volume_24h": financial.volume_24h,
                        "volatility_24h": financial.volatility_24h,
                        "price_change_24h": financial.price_change_24h,
                        "price_change_7d": financial.price_change_7d,
                        "price_change_30d": financial.price_change_30d,
                    })
                    
                    # Use future price change as target (7 days ahead)
                    future_financial = next((m for m in financial_metrics 
                                           if m.timestamp > timestamp and 
                                           (m.timestamp - timestamp).days >= 7), None)
                    if future_financial:
                        target = future_financial.price_change_7d
                        if target is not None:
                            features.append(feature_vector)
                            targets.append(target)
                
                # Extract other features
                github = next((m for m in github_metrics if m.timestamp == timestamp), None)
                if github:
                    feature_vector.update({
                        "commits_24h": github.commits_24h,
                        "commits_7d": github.commits_7d,
                        "commits_30d": github.commits_30d,
                        "active_contributors_30d": github.active_contributors_30d,
                        "stars": github.stars,
                        "forks": github.forks,
                        "open_issues": github.open_issues,
                        "open_prs": github.open_prs,
                        "code_quality_score": github.code_quality_score,
                    })
                
                tokenomics = next((m for m in tokenomics_metrics if m.timestamp == timestamp), None)
                if tokenomics:
                    feature_vector.update({
                        "circulating_supply": tokenomics.circulating_supply,
                        "total_supply": tokenomics.total_supply,
                        "inflation_rate": tokenomics.inflation_rate,
                        "burn_rate": tokenomics.burn_rate,
                    })
                
                security = next((m for m in security_metrics if m.timestamp == timestamp), None)
                if security:
                    feature_vector.update({
                        "audit_score": security.audit_score,
                        "vulnerability_score": security.vulnerability_score,
                        "contract_verified": 1 if security.contract_verified else 0,
                    })
                
                community = next((m for m in community_metrics if m.timestamp == timestamp), None)
                if community:
                    feature_vector.update({
                        "twitter_followers": community.twitter_followers,
                        "telegram_members": community.telegram_members,
                        "discord_members": community.discord_members,
                        "social_engagement_rate": community.social_engagement_rate,
                    })
                
                network = next((m for m in network_metrics if m.timestamp == timestamp), None)
                if network:
                    feature_vector.update({
                        "block_time_avg": network.block_time_avg,
                        "network_utilization": network.network_utilization,
                        "validator_count": network.validator_count,
                    })
                
                trending = next((m for m in trending_metrics if m.timestamp == timestamp), None)
                if trending:
                    feature_vector.update({
                        "momentum_score": trending.momentum_score,
                        "fear_greed_index": trending.fear_greed_index,
                        "social_sentiment": trending.social_sentiment,
                    })
            
            return {"features": features, "targets": targets}
            
        except Exception as e:
            logger.error(f"Error extracting features for asset {asset_id}: {e}")
            return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill remaining missing values with 0
        df = df.fillna(0)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        # Create interaction features
        if "price_usd" in df.columns and "volume_24h" in df.columns:
            df["price_volume_interaction"] = df["price_usd"] * df["volume_24h"]
        
        if "market_cap" in df.columns and "circulating_supply" in df.columns:
            df["market_cap_supply_ratio"] = df["market_cap"] / (df["circulating_supply"] + 1)
        
        # Create ratio features
        if "tvl" in df.columns and "market_cap" in df.columns:
            df["tvl_market_cap_ratio"] = df["tvl"] / (df["market_cap"] + 1)
        
        if "daily_transactions" in df.columns and "active_addresses_24h" in df.columns:
            df["transactions_per_address"] = df["daily_transactions"] / (df["active_addresses_24h"] + 1)
        
        # Create volatility features
        for col in ["price_usd", "tvl", "volume_24h"]:
            if col in df.columns:
                df[f"{col}_volatility"] = df[col].rolling(window=7, min_periods=1).std()
        
        return df
    
    async def train_models(
        self, 
        features_df: pd.DataFrame, 
        targets_series: pd.Series,
        model_names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Train ML models"""
        if model_names is None:
            model_names = [config.name for config in self.model_configs]
        
        results = {}
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_series, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers["standard"] = scaler
        
        for model_name in model_names:
            try:
                config = next((c for c in self.model_configs if c.name == model_name), None)
                if not config:
                    logger.warning(f"Model config not found: {model_name}")
                    continue
                
                logger.info(f"Training {model_name}...")
                
                # Create and train model
                model = config.model_class(**config.hyperparameters)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = []
                if config.cross_validation:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=TimeSeriesSplit(n_splits=config.cv_folds),
                        scoring='neg_mean_squared_error'
                    )
                
                # Feature importance
                feature_importance = {}
                if config.feature_importance and hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(features_df.columns, model.feature_importances_))
                
                # Store model
                self.models[model_name] = model
                
                results[model_name] = {
                    "model": model,
                    "metrics": {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "cv_scores": cv_scores.tolist() if len(cv_scores) > 0 else [],
                        "cv_mean": cv_scores.mean() if len(cv_scores) > 0 else 0,
                        "cv_std": cv_scores.std() if len(cv_scores) > 0 else 0
                    },
                    "feature_importance": feature_importance,
                    "predictions": y_pred.tolist(),
                    "actual": y_test.tolist()
                }
                
                logger.info(f"{model_name} trained - R²: {r2:.4f}, MSE: {mse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def predict_investment_score(
        self, 
        asset_id: int, 
        model_name: str = "ensemble"
    ) -> MLPipelineResult:
        """Predict investment score for an asset"""
        try:
            # Get latest features for the asset
            features = await self._get_latest_features(asset_id)
            if not features:
                raise ValueError(f"No features found for asset {asset_id}")
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            features_df = self._handle_missing_values(features_df)
            features_df = self._engineer_features(features_df)
            
            # Scale features
            if "standard" in self.scalers:
                features_scaled = self.scalers["standard"].transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            if model_name == "ensemble":
                predictions = []
                for name, model in self.models.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(features_scaled)[0]
                        predictions.append(pred)
                
                if predictions:
                    prediction_value = np.mean(predictions)
                    confidence_score = 1.0 - np.std(predictions) / (np.mean(predictions) + 1e-8)
                else:
                    prediction_value = 0.0
                    confidence_score = 0.0
            else:
                if model_name in self.models:
                    model = self.models[model_name]
                    prediction_value = model.predict(features_scaled)[0]
                    confidence_score = 0.8  # Default confidence
                else:
                    raise ValueError(f"Model {model_name} not found")
            
            # Get feature importance
            feature_importance = {}
            if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
                feature_importance = dict(zip(features_df.columns, self.models[model_name].feature_importances_))
            
            return MLPipelineResult(
                model_name=model_name,
                prediction_type="investment_score",
                asset_id=asset_id,
                prediction_value=float(prediction_value),
                confidence_score=float(confidence_score),
                feature_importance=feature_importance,
                model_metrics={},
                prediction_horizon="7d",
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error predicting investment score for asset {asset_id}: {e}")
            raise
    
    async def _get_latest_features(self, asset_id: int) -> Optional[Dict[str, Any]]:
        """Get latest features for an asset"""
        try:
            features = {}
            
            # Get latest metrics
            latest_onchain = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == asset_id
            ).order_by(OnChainMetrics.timestamp.desc()).first()
            
            latest_financial = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == asset_id
            ).order_by(FinancialMetrics.timestamp.desc()).first()
            
            latest_github = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == asset_id
            ).order_by(GitHubMetrics.timestamp.desc()).first()
            
            latest_tokenomics = self.session.query(TokenomicsMetrics).filter(
                TokenomicsMetrics.asset_id == asset_id
            ).order_by(TokenomicsMetrics.timestamp.desc()).first()
            
            latest_security = self.session.query(SecurityMetrics).filter(
                SecurityMetrics.asset_id == asset_id
            ).order_by(SecurityMetrics.timestamp.desc()).first()
            
            latest_community = self.session.query(CommunityMetrics).filter(
                CommunityMetrics.asset_id == asset_id
            ).order_by(CommunityMetrics.timestamp.desc()).first()
            
            latest_trending = self.session.query(TrendingMetrics).filter(
                TrendingMetrics.asset_id == asset_id
            ).order_by(TrendingMetrics.timestamp.desc()).first()
            
            # Extract features
            if latest_onchain:
                features.update({
                    "tvl": latest_onchain.tvl or 0,
                    "tvl_change_24h": latest_onchain.tvl_change_24h or 0,
                    "tvl_change_7d": latest_onchain.tvl_change_7d or 0,
                    "daily_transactions": latest_onchain.daily_transactions or 0,
                    "active_addresses_24h": latest_onchain.active_addresses_24h or 0,
                    "transaction_volume_24h": latest_onchain.transaction_volume_24h or 0,
                    "gas_price_avg": latest_onchain.gas_price_avg or 0,
                    "contract_interactions_24h": latest_onchain.contract_interactions_24h or 0,
                })
            
            if latest_financial:
                features.update({
                    "price_usd": latest_financial.price_usd or 0,
                    "market_cap": latest_financial.market_cap or 0,
                    "volume_24h": latest_financial.volume_24h or 0,
                    "volatility_24h": latest_financial.volatility_24h or 0,
                    "price_change_24h": latest_financial.price_change_24h or 0,
                    "price_change_7d": latest_financial.price_change_7d or 0,
                    "price_change_30d": latest_financial.price_change_30d or 0,
                })
            
            if latest_github:
                features.update({
                    "commits_24h": latest_github.commits_24h or 0,
                    "commits_7d": latest_github.commits_7d or 0,
                    "commits_30d": latest_github.commits_30d or 0,
                    "active_contributors_30d": latest_github.active_contributors_30d or 0,
                    "stars": latest_github.stars or 0,
                    "forks": latest_github.forks or 0,
                    "open_issues": latest_github.open_issues or 0,
                    "open_prs": latest_github.open_prs or 0,
                    "code_quality_score": latest_github.code_quality_score or 0,
                })
            
            if latest_tokenomics:
                features.update({
                    "circulating_supply": latest_tokenomics.circulating_supply or 0,
                    "total_supply": latest_tokenomics.total_supply or 0,
                    "inflation_rate": latest_tokenomics.inflation_rate or 0,
                    "burn_rate": latest_tokenomics.burn_rate or 0,
                })
            
            if latest_security:
                features.update({
                    "audit_score": latest_security.audit_score or 0,
                    "vulnerability_score": latest_security.vulnerability_score or 0,
                    "contract_verified": 1 if latest_security.contract_verified else 0,
                })
            
            if latest_community:
                features.update({
                    "twitter_followers": latest_community.twitter_followers or 0,
                    "telegram_members": latest_community.telegram_members or 0,
                    "discord_members": latest_community.discord_members or 0,
                    "social_engagement_rate": latest_community.social_engagement_rate or 0,
                })
            
            if latest_trending:
                features.update({
                    "momentum_score": latest_trending.momentum_score or 0,
                    "fear_greed_index": latest_trending.fear_greed_index or 0,
                    "social_sentiment": latest_trending.social_sentiment or 0,
                })
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error getting latest features for asset {asset_id}: {e}")
            return None
    
    async def save_models(self, filepath: str = None):
        """Save trained models to disk"""
        if filepath is None:
            filepath = settings.ML_MODEL_PATH
        
        try:
            for model_name, model in self.models.items():
                model_path = f"{filepath}/{model_name}_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = f"{filepath}/{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self, filepath: str = None):
        """Load trained models from disk"""
        if filepath is None:
            filepath = settings.ML_MODEL_PATH
        
        try:
            import os
            if not os.path.exists(filepath):
                logger.warning(f"Model directory does not exist: {filepath}")
                return
            
            # Load models
            for config in self.model_configs:
                model_path = f"{filepath}/{config.name}_model.joblib"
                if os.path.exists(model_path):
                    self.models[config.name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_path}")
            
            # Load scalers
            scaler_path = f"{filepath}/standard_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers["standard"] = joblib.load(scaler_path)
                logger.info(f"Loaded scaler: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def generate_predictions_for_all_assets(self) -> List[MLPipelineResult]:
        """Generate predictions for all active assets"""
        results = []
        
        try:
            assets = self.session.query(CryptoAsset).filter(CryptoAsset.is_active == True).all()
            
            for asset in assets:
                try:
                    result = await self.predict_investment_score(asset.id)
                    results.append(result)
                    
                    # Store prediction in database
                    prediction = MLPrediction(
                        asset_id=asset.id,
                        model_name=result.model_name,
                        prediction_type=result.prediction_type,
                        prediction_value=result.prediction_value,
                        confidence_score=result.confidence_score,
                        prediction_horizon=result.prediction_horizon,
                        features_used=list(result.feature_importance.keys()),
                        model_version="1.0"
                    )
                    self.session.add(prediction)
                    
                except Exception as e:
                    logger.error(f"Error generating prediction for asset {asset.symbol}: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"Generated {len(results)} predictions")
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            self.session.rollback()
        
        return results
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        summary = {
            "trained_models": list(self.models.keys()),
            "available_scalers": list(self.scalers.keys()),
            "model_configs": len(self.model_configs),
            "feature_configs": len(self.feature_configs)
        }
        
        return summary
