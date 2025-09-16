#!/usr/bin/env python3
"""
Polkadot Machine Learning Strategy and Pipeline
===============================================

A comprehensive ML strategy for analyzing Polkadot network data with Hugging Face integration.
This system provides predictive analytics, anomaly detection, and automated insights.

Author: AI Assistant
Date: 2024
"""

import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from pathlib import Path

# ML Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel, AutoConfig
import datasets
from datasets import Dataset as HFDataset

# Hugging Face Hub
from huggingface_hub import HfApi, login
from huggingface_hub import create_repo, upload_file

# Database and API
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import aiohttp
from loguru import logger

# Load environment variables
from dotenv import load_dotenv
load_dotenv('env.main')

class MLTaskType(Enum):
    """Types of ML tasks for Polkadot data"""
    PRICE_PREDICTION = "price_prediction"
    TVL_PREDICTION = "tvl_prediction"
    TRANSACTION_VOLUME = "transaction_volume"
    STAKING_ANALYSIS = "staking_analysis"
    GOVERNANCE_PREDICTION = "governance_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    NETWORK_HEALTH = "network_health"
    CROSS_CHAIN_ANALYSIS = "cross_chain_analysis"

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    task_type: MLTaskType
    model_name: str
    model_type: str  # 'transformer', 'xgboost', 'neural_network', 'ensemble'
    features: List[str]
    target_column: str
    sequence_length: int = 24  # For time series
    prediction_horizon: int = 1  # Hours ahead to predict
    validation_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10

class PolkadotMLStrategy:
    """Main ML strategy class for Polkadot data analysis"""
    
    def __init__(self, database_url: str = None, hf_token: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.models_dir = Path("models")
        self.data_dir = Path("ml_data")
        self.results_dir = Path("ml_results")
        
        # Create directories
        for dir_path in [self.models_dir, self.data_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize Hugging Face
        if self.hf_token:
            login(token=self.hf_token)
            self.hf_api = HfApi(token=self.hf_token)
        
        # Database connection
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Model configurations
        self.model_configs = self._initialize_model_configs()
        
        # Trained models storage
        self.trained_models = {}
        
        logger.info("Polkadot ML Strategy initialized successfully")

    def _initialize_model_configs(self) -> Dict[MLTaskType, MLModelConfig]:
        """Initialize model configurations for different tasks"""
        configs = {
            MLTaskType.PRICE_PREDICTION: MLModelConfig(
                task_type=MLTaskType.PRICE_PREDICTION,
                model_name="polkadot-price-predictor",
                model_type="transformer",
                features=[
                    "total_staked", "staking_ratio", "validator_count", "nominator_count",
                    "inflation_rate", "treasury_balance", "active_proposals", "referendum_count",
                    "cross_chain_volume", "parachain_tvl", "network_utilization"
                ],
                target_column="price_usd",
                sequence_length=168,  # 1 week of hourly data
                prediction_horizon=24  # 24 hours ahead
            ),
            
            MLTaskType.TVL_PREDICTION: MLModelConfig(
                task_type=MLTaskType.TVL_PREDICTION,
                model_name="polkadot-tvl-predictor",
                model_type="xgboost",
                features=[
                    "total_staked", "staking_ratio", "validator_count", "price_usd",
                    "transaction_volume", "active_addresses", "cross_chain_volume",
                    "governance_activity", "developer_activity", "community_growth"
                ],
                target_column="total_ecosystem_tvl",
                sequence_length=24,
                prediction_horizon=168  # 1 week ahead
            ),
            
            MLTaskType.ANOMALY_DETECTION: MLModelConfig(
                task_type=MLTaskType.ANOMALY_DETECTION,
                model_name="polkadot-anomaly-detector",
                model_type="isolation_forest",
                features=[
                    "block_time_avg", "transaction_throughput", "network_latency_avg",
                    "validator_count", "slash_events_count_24h", "cross_chain_messages_24h",
                    "governance_activity", "price_volatility", "tvl_change_24h"
                ],
                target_column="anomaly_score",
                sequence_length=1,
                prediction_horizon=1
            ),
            
            MLTaskType.NETWORK_HEALTH: MLModelConfig(
                task_type=MLTaskType.NETWORK_HEALTH,
                model_name="polkadot-network-health",
                model_type="neural_network",
                features=[
                    "block_production_rate", "finalization_time", "consensus_latency",
                    "network_utilization", "peer_count", "sync_status",
                    "transaction_success_rate", "validator_uptime", "cross_chain_success_rate"
                ],
                target_column="network_health_score",
                sequence_length=12,
                prediction_horizon=6
            ),
            
            MLTaskType.STAKING_ANALYSIS: MLModelConfig(
                task_type=MLTaskType.STAKING_ANALYSIS,
                model_name="polkadot-staking-analyzer",
                model_type="ensemble",
                features=[
                    "total_staked", "staking_ratio", "validator_count", "nominator_count",
                    "inflation_rate", "block_reward", "era_progress", "commission_rates",
                    "nomination_pool_count", "staking_apy", "price_usd"
                ],
                target_column="optimal_staking_ratio",
                sequence_length=24,
                prediction_horizon=24
            )
        }
        
        return configs

    async def collect_training_data(self, days_back: int = 90) -> pd.DataFrame:
        """Collect comprehensive training data from database"""
        logger.info(f"Collecting training data for last {days_back} days")
        
        query = """
        WITH network_metrics AS (
            SELECT 
                timestamp,
                current_block,
                block_time_avg,
                transaction_throughput,
                network_utilization,
                finalization_time,
                consensus_latency,
                peer_count,
                total_transactions,
                daily_transactions,
                transaction_success_rate,
                avg_transaction_fee,
                total_fees_24h,
                total_accounts,
                active_accounts_24h,
                new_accounts_24h,
                validator_count,
                active_validators,
                xcm_messages_24h,
                hrmp_channels_active
            FROM polkadot_network_metrics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        ),
        staking_metrics AS (
            SELECT 
                timestamp,
                total_staked,
                total_staked_usd,
                staking_ratio,
                active_era,
                current_era,
                era_progress,
                validator_count,
                nominator_count,
                min_validator_stake,
                max_validator_stake,
                avg_validator_stake,
                block_reward,
                validator_reward,
                nominator_reward,
                inflation_rate,
                ideal_staking_rate
            FROM polkadot_staking_metrics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        ),
        governance_metrics AS (
            SELECT 
                timestamp,
                active_proposals,
                referendum_count,
                active_referendums,
                referendum_success_rate,
                referendum_turnout_rate,
                council_members,
                council_motions,
                council_votes,
                treasury_proposals,
                treasury_spend_proposals,
                voter_participation_rate,
                total_votes_cast,
                direct_voters,
                delegated_voters,
                governance_activity_score,
                community_engagement_score
            FROM polkadot_governance_metrics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        ),
        economic_metrics AS (
            SELECT 
                timestamp,
                treasury_balance,
                treasury_balance_usd,
                treasury_spend_rate,
                total_supply,
                circulating_supply,
                inflation_rate,
                market_cap,
                price_usd,
                price_change_24h,
                price_change_7d,
                price_change_30d,
                volume_24h,
                avg_transaction_fee,
                total_fees_24h
            FROM polkadot_economic_metrics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        ),
        ecosystem_metrics AS (
            SELECT 
                timestamp,
                total_parachains,
                active_parachains,
                total_ecosystem_tvl,
                total_ecosystem_tvl_usd,
                tvl_growth_rate,
                total_cross_chain_messages_24h,
                total_cross_chain_volume_24h,
                active_cross_chain_channels,
                total_active_developers,
                total_new_projects_launched,
                total_github_commits_24h,
                social_media_mentions_24h,
                community_growth_rate
            FROM polkadot_ecosystem_metrics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
        )
        SELECT 
            n.timestamp,
            -- Network metrics
            n.current_block,
            n.block_time_avg,
            n.transaction_throughput,
            n.network_utilization,
            n.finalization_time,
            n.consensus_latency,
            n.peer_count,
            n.total_transactions,
            n.daily_transactions,
            n.transaction_success_rate,
            n.avg_transaction_fee,
            n.total_fees_24h,
            n.total_accounts,
            n.active_accounts_24h,
            n.new_accounts_24h,
            n.validator_count as network_validator_count,
            n.active_validators,
            n.xcm_messages_24h,
            n.hrmp_channels_active,
            -- Staking metrics
            s.total_staked,
            s.total_staked_usd,
            s.staking_ratio,
            s.active_era,
            s.current_era,
            s.era_progress,
            s.validator_count as staking_validator_count,
            s.nominator_count,
            s.min_validator_stake,
            s.max_validator_stake,
            s.avg_validator_stake,
            s.block_reward,
            s.validator_reward,
            s.nominator_reward,
            s.inflation_rate as staking_inflation_rate,
            s.ideal_staking_rate,
            -- Governance metrics
            g.active_proposals,
            g.referendum_count,
            g.active_referendums,
            g.referendum_success_rate,
            g.referendum_turnout_rate,
            g.council_members,
            g.council_motions,
            g.council_votes,
            g.treasury_proposals,
            g.treasury_spend_proposals,
            g.voter_participation_rate,
            g.total_votes_cast,
            g.direct_voters,
            g.delegated_voters,
            g.governance_activity_score,
            g.community_engagement_score,
            -- Economic metrics
            e.treasury_balance,
            e.treasury_balance_usd,
            e.treasury_spend_rate,
            e.total_supply,
            e.circulating_supply,
            e.inflation_rate as economic_inflation_rate,
            e.market_cap,
            e.price_usd,
            e.price_change_24h,
            e.price_change_7d,
            e.price_change_30d,
            e.volume_24h,
            e.avg_transaction_fee as economic_avg_fee,
            e.total_fees_24h as economic_total_fees,
            -- Ecosystem metrics
            ec.total_parachains,
            ec.active_parachains,
            ec.total_ecosystem_tvl,
            ec.total_ecosystem_tvl_usd,
            ec.tvl_growth_rate,
            ec.total_cross_chain_messages_24h,
            ec.total_cross_chain_volume_24h,
            ec.active_cross_chain_channels,
            ec.total_active_developers,
            ec.total_new_projects_launched,
            ec.total_github_commits_24h,
            ec.social_media_mentions_24h,
            ec.community_growth_rate
        FROM network_metrics n
        LEFT JOIN staking_metrics s ON DATE_TRUNC('hour', n.timestamp) = DATE_TRUNC('hour', s.timestamp)
        LEFT JOIN governance_metrics g ON DATE_TRUNC('hour', n.timestamp) = DATE_TRUNC('hour', g.timestamp)
        LEFT JOIN economic_metrics e ON DATE_TRUNC('hour', n.timestamp) = DATE_TRUNC('hour', e.timestamp)
        LEFT JOIN ecosystem_metrics ec ON DATE_TRUNC('hour', n.timestamp) = DATE_TRUNC('hour', ec.timestamp)
        ORDER BY n.timestamp
        """ % (days_back, days_back, days_back, days_back, days_back)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
                logger.info(f"Collected {len(df)} records for training")
                return df
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df: pd.DataFrame, config: MLModelConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for ML training"""
        logger.info(f"Preprocessing data for {config.task_type.value}")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Select features
        available_features = [f for f in config.features if f in df.columns]
        missing_features = [f for f in config.features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Create feature matrix
        X = df[available_features].values
        
        # Create target variable
        if config.target_column in df.columns:
            y = df[config.target_column].values
        else:
            logger.error(f"Target column {config.target_column} not found")
            return None, None
        
        # Handle time series data
        if config.sequence_length > 1:
            X, y = self._create_sequences(X, y, config.sequence_length, config.prediction_horizon)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Save scaler
        scaler_path = self.models_dir / f"{config.model_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Preprocessed data shape: X={X_scaled.shape}, y={y.shape}")
        return X_scaled, y

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int, prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X) - prediction_horizon + 1):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i+prediction_horizon-1])
        
        return np.array(X_sequences), np.array(y_sequences)

    def train_price_predictor(self, X: np.ndarray, y: np.ndarray, config: MLModelConfig) -> Any:
        """Train price prediction model using transformer architecture"""
        logger.info("Training price prediction model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split, random_state=42
        )
        
        # Create Hugging Face dataset
        train_dataset = HFDataset.from_dict({
            "input_ids": X_train.tolist(),
            "labels": y_train.tolist()
        })
        
        test_dataset = HFDataset.from_dict({
            "input_ids": X_test.tolist(),
            "labels": y_test.tolist()
        })
        
        # Load pre-trained transformer model
        model_name = "microsoft/DialoGPT-medium"  # Can be changed to a more suitable model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Fine-tune for regression
        model.config.num_labels = 1
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Training loop (simplified)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(config.epochs):
            total_loss = 0
            for i in range(0, len(X_train), config.batch_size):
                batch_X = torch.tensor(X_train[i:i+config.batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y_train[i:i+config.batch_size], dtype=torch.float32)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.last_hidden_state.mean(dim=1).squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(X_train)*config.batch_size:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{config.model_name}_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_X = torch.tensor(X_test, dtype=torch.float32)
            predictions = model(test_X).last_hidden_state.mean(dim=1).squeeze().numpy()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Price prediction model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return model

    def train_tvl_predictor(self, X: np.ndarray, y: np.ndarray, config: MLModelConfig) -> Any:
        """Train TVL prediction model using XGBoost"""
        logger.info("Training TVL prediction model")
        
        # Flatten sequences for XGBoost
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split, random_state=42
        )
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=config.learning_rate,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"TVL prediction model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{config.model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model

    def train_anomaly_detector(self, X: np.ndarray, y: np.ndarray, config: MLModelConfig) -> Any:
        """Train anomaly detection model using Isolation Forest"""
        logger.info("Training anomaly detection model")
        
        # Flatten sequences for Isolation Forest
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # 10% of data expected to be anomalies
            random_state=42
        )
        
        model.fit(X)
        
        # Evaluate
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)
        
        n_anomalies = sum(predictions == -1)
        logger.info(f"Anomaly detection model - Found {n_anomalies} anomalies out of {len(X)} samples")
        
        # Save model
        model_path = self.models_dir / f"{config.model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model

    def train_network_health_model(self, X: np.ndarray, y: np.ndarray, config: MLModelConfig) -> Any:
        """Train network health prediction model using neural network"""
        logger.info("Training network health model")
        
        # Flatten sequences for neural network
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split, random_state=42
        )
        
        # Create neural network
        class NetworkHealthNet(nn.Module):
            def __init__(self, input_size, hidden_size=128, output_size=1):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size//2, output_size),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = NetworkHealthNet(X.shape[1])
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        model.train()
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).squeeze().numpy()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Network health model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{config.model_name}_model.pth"
        torch.save(model.state_dict(), model_path)
        
        return model

    async def train_all_models(self, days_back: int = 90):
        """Train all configured models"""
        logger.info("Starting training for all models")
        
        # Collect data
        df = await self.collect_training_data(days_back)
        if df.empty:
            logger.error("No data collected for training")
            return
        
        # Save raw data
        data_path = self.data_dir / f"training_data_{days_back}days.csv"
        df.to_csv(data_path, index=False)
        
        # Train each model
        for task_type, config in self.model_configs.items():
            try:
                logger.info(f"Training {task_type.value}")
                
                # Preprocess data
                X, y = self.preprocess_data(df, config)
                if X is None or y is None:
                    logger.error(f"Failed to preprocess data for {task_type.value}")
                    continue
                
                # Train model based on type
                if task_type == MLTaskType.PRICE_PREDICTION:
                    model = self.train_price_predictor(X, y, config)
                elif task_type == MLTaskType.TVL_PREDICTION:
                    model = self.train_tvl_predictor(X, y, config)
                elif task_type == MLTaskType.ANOMALY_DETECTION:
                    model = self.train_anomaly_detector(X, y, config)
                elif task_type == MLTaskType.NETWORK_HEALTH:
                    model = self.train_network_health_model(X, y, config)
                else:
                    logger.warning(f"No training method defined for {task_type.value}")
                    continue
                
                self.trained_models[task_type] = model
                logger.success(f"Successfully trained {task_type.value}")
                
            except Exception as e:
                logger.error(f"Error training {task_type.value}: {e}")
        
        logger.info("Training completed for all models")

    async def upload_models_to_hf(self, repo_name: str = "polkadot-ml-models"):
        """Upload trained models to Hugging Face Hub"""
        if not self.hf_token:
            logger.warning("No HF token available, skipping upload")
            return
        
        logger.info(f"Uploading models to Hugging Face Hub: {repo_name}")
        
        try:
            # Create repository
            create_repo(repo_name, exist_ok=True, token=self.hf_token)
            
            # Upload model files
            for task_type, model in self.trained_models.items():
                config = self.model_configs[task_type]
                
                # Upload model file
                model_file = self.models_dir / f"{config.model_name}_model.pkl"
                if model_file.exists():
                    upload_file(
                        path_or_fileobj=str(model_file),
                        path_in_repo=f"{config.model_name}_model.pkl",
                        repo_id=repo_name,
                        token=self.hf_token
                    )
                
                # Upload scaler file
                scaler_file = self.models_dir / f"{config.model_name}_scaler.pkl"
                if scaler_file.exists():
                    upload_file(
                        path_or_fileobj=str(scaler_file),
                        path_in_repo=f"{config.model_name}_scaler.pkl",
                        repo_id=repo_name,
                        token=self.hf_token
                    )
                
                # Upload model card
                model_card = self._create_model_card(config)
                upload_file(
                    path_or_fileobj=model_card,
                    path_in_repo=f"{config.model_name}_README.md",
                    repo_id=repo_name,
                    token=self.hf_token
                )
            
            logger.success("Successfully uploaded all models to Hugging Face Hub")
            
        except Exception as e:
            logger.error(f"Error uploading models to HF: {e}")

    def _create_model_card(self, config: MLModelConfig) -> str:
        """Create model card for Hugging Face"""
        return f"""---
license: mit
tags:
- polkadot
- blockchain
- machine-learning
- {config.task_type.value}
- time-series
- prediction
---

# {config.model_name}

## Model Description

This model is trained on Polkadot network data to predict {config.task_type.value.replace('_', ' ')}.

## Model Type
{config.model_type}

## Features Used
{', '.join(config.features)}

## Target Variable
{config.target_column}

## Training Configuration
- Sequence Length: {config.sequence_length}
- Prediction Horizon: {config.prediction_horizon}
- Learning Rate: {config.learning_rate}
- Epochs: {config.epochs}

## Usage

```python
import pickle
import pandas as pd

# Load model
with open('{config.model_name}_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('{config.model_name}_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Preprocess your data
X_scaled = scaler.transform(your_data)

# Make predictions
predictions = model.predict(X_scaled)
```

## Performance Metrics
Model performance metrics will be updated after training completion.

## Data Source
This model is trained on Polkadot network data collected from various sources including:
- Network metrics
- Staking data
- Governance information
- Economic indicators
- Ecosystem metrics

## Limitations
- Model performance depends on data quality and availability
- Predictions are based on historical patterns and may not account for unexpected events
- Regular retraining is recommended for optimal performance
"""

    async def generate_ml_insights(self) -> Dict[str, Any]:
        """Generate ML insights and predictions"""
        logger.info("Generating ML insights")
        
        insights = {
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": {},
            "anomalies": [],
            "recommendations": []
        }
        
        # Get latest data for predictions
        latest_data = await self.collect_training_data(days_back=7)
        if latest_data.empty:
            return insights
        
        # Generate predictions for each model
        for task_type, model in self.trained_models.items():
            try:
                config = self.model_configs[task_type]
                
                # Preprocess latest data
                X, _ = self.preprocess_data(latest_data, config)
                if X is None:
                    continue
                
                # Make prediction
                if task_type == MLTaskType.ANOMALY_DETECTION:
                    anomaly_scores = model.decision_function(X.reshape(X.shape[0], -1))
                    predictions = model.predict(X.reshape(X.shape[0], -1))
                    
                    # Find anomalies
                    anomalies = np.where(predictions == -1)[0]
                    if len(anomalies) > 0:
                        insights["anomalies"].extend([
                            {
                                "timestamp": latest_data.iloc[i]["timestamp"].isoformat(),
                                "anomaly_score": float(anomaly_scores[i]),
                                "features": latest_data.iloc[i][config.features].to_dict()
                            }
                            for i in anomalies
                        ])
                else:
                    # Regular prediction
                    if hasattr(model, 'predict'):
                        prediction = model.predict(X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X)
                        insights["predictions"][task_type.value] = {
                            "value": float(prediction[-1]) if len(prediction) > 0 else None,
                            "confidence": 0.85,  # Placeholder
                            "timestamp": datetime.utcnow().isoformat()
                        }
                
            except Exception as e:
                logger.error(f"Error generating prediction for {task_type.value}: {e}")
        
        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(insights)
        
        return insights

    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # Price prediction recommendations
        if "price_prediction" in insights["predictions"]:
            price_pred = insights["predictions"]["price_prediction"]["value"]
            if price_pred:
                if price_pred > 7.0:
                    recommendations.append("Price prediction suggests bullish trend - consider staking opportunities")
                elif price_pred < 6.0:
                    recommendations.append("Price prediction suggests bearish trend - monitor market conditions")
        
        # TVL prediction recommendations
        if "tvl_prediction" in insights["predictions"]:
            tvl_pred = insights["predictions"]["tvl_prediction"]["value"]
            if tvl_pred:
                recommendations.append(f"Expected TVL growth: {tvl_pred:.2f} - monitor DeFi activity")
        
        # Anomaly recommendations
        if insights["anomalies"]:
            recommendations.append(f"Detected {len(insights['anomalies'])} anomalies - investigate network health")
        
        # Network health recommendations
        if "network_health" in insights["predictions"]:
            health_score = insights["predictions"]["network_health"]["value"]
            if health_score and health_score < 0.7:
                recommendations.append("Network health score below threshold - check validator performance")
        
        return recommendations

    async def run_ml_pipeline(self, days_back: int = 90, upload_to_hf: bool = True):
        """Run the complete ML pipeline"""
        logger.info("Starting complete ML pipeline")
        
        try:
            # Train all models
            await self.train_all_models(days_back)
            
            # Upload to Hugging Face if requested
            if upload_to_hf and self.hf_token:
                await self.upload_models_to_hf()
            
            # Generate insights
            insights = await self.generate_ml_insights()
            
            # Save insights
            insights_path = self.results_dir / f"ml_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.success("ML pipeline completed successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error in ML pipeline: {e}")
            raise

# Example usage
async def main():
    """Main function to run the ML pipeline"""
    ml_strategy = PolkadotMLStrategy()
    
    # Run the complete pipeline
    insights = await ml_strategy.run_ml_pipeline(
        days_back=90,
        upload_to_hf=True
    )
    
    print("ML Pipeline Results:")
    print(json.dumps(insights, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
