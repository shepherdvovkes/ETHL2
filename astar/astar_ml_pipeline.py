#!/usr/bin/env python3
"""
Astar ML Pipeline
================

Machine learning pipeline using aggregate Astar data for predictions and analysis.
Uses RTX 4090 acceleration for training advanced models.

Features:
- Network activity prediction
- Token price forecasting
- DeFi protocol analysis
- Smart contract interaction prediction
- Market trend analysis
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from loguru import logger
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
warnings.filterwarnings('ignore')

class AstarMLDataset(Dataset):
    """Custom dataset for Astar time series data"""
    
    def __init__(self, features, targets, sequence_length=12):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.features[idx:idx+self.sequence_length],
            self.targets[idx+self.sequence_length]
        )

class AstarPredictor(nn.Module):
    """Neural network for Astar predictions"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            current_size = hidden_size
            hidden_size = hidden_size // 2
        
        layers.append(nn.Linear(current_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AstarMLPipeline:
    """ML pipeline for Astar network analysis"""
    
    def __init__(self, database_path="astar_aggregate_data.db"):
        self.database_path = database_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Initializing Astar ML Pipeline on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_astar_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Astar data from database"""
        logger.info("Loading Astar data from database...")
        
        conn = sqlite3.connect(self.database_path)
        
        dataframes = {}
        
        try:
            # Load network stats
            network_df = pd.read_sql_query("SELECT * FROM astar_network_stats ORDER BY timestamp", conn)
            if not network_df.empty:
                network_df['timestamp'] = pd.to_datetime(network_df['timestamp'])
                dataframes['network'] = network_df
                logger.info(f"Loaded {len(network_df)} network stats records")
            
            # Load token stats
            token_df = pd.read_sql_query("SELECT * FROM astar_token_stats ORDER BY timestamp", conn)
            if not token_df.empty:
                token_df['timestamp'] = pd.to_datetime(token_df['timestamp'])
                dataframes['tokens'] = token_df
                logger.info(f"Loaded {len(token_df)} token stats records")
            
            # Load DeFi stats
            defi_df = pd.read_sql_query("SELECT * FROM astar_defi_stats ORDER BY timestamp", conn)
            if not defi_df.empty:
                defi_df['timestamp'] = pd.to_datetime(defi_df['timestamp'])
                dataframes['defi'] = defi_df
                logger.info(f"Loaded {len(defi_df)} DeFi stats records")
            
            # Load contract stats
            contract_df = pd.read_sql_query("SELECT * FROM astar_contract_stats ORDER BY timestamp", conn)
            if not contract_df.empty:
                contract_df['timestamp'] = pd.to_datetime(contract_df['timestamp'])
                dataframes['contracts'] = contract_df
                logger.info(f"Loaded {len(contract_df)} contract stats records")
            
            # Load market data
            market_df = pd.read_sql_query("SELECT * FROM astar_market_data ORDER BY timestamp", conn)
            if not market_df.empty:
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                dataframes['market'] = market_df
                logger.info(f"Loaded {len(market_df)} market data records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
        finally:
            conn.close()
        
        return dataframes
    
    def create_synthetic_time_series(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create synthetic time series data for ML training"""
        logger.info("Creating synthetic time series data...")
        
        if not dataframes:
            logger.error("No data available for time series creation")
            return pd.DataFrame()
        
        # Use network data as base
        base_data = dataframes.get('network', pd.DataFrame())
        if base_data.empty:
            logger.error("No network data available")
            return pd.DataFrame()
        
        # Create synthetic time series with realistic patterns
        synthetic_data = []
        
        # Generate 30 days of hourly data
        start_time = datetime.utcnow() - timedelta(days=30)
        
        for day in range(30):
            for hour in range(24):
                timestamp = start_time + timedelta(days=day, hours=hour)
                
                # Create realistic variations
                daily_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
                hourly_factor = 1 + 0.2 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
                noise = np.random.normal(1, 0.1)  # Random noise
                
                # Base values from actual data
                base_block = base_data['current_block'].iloc[0] if not base_data.empty else 10000000
                base_utilization = base_data['network_utilization'].iloc[0] if not base_data.empty else 0.3
                
                record = {
                    'timestamp': timestamp,
                    'block_number': base_block + (day * 24 + hour) * 6,  # ~6 second blocks
                    'network_utilization': min(1.0, base_utilization * daily_factor * hourly_factor * noise),
                    'gas_price_avg': 0.00002 * daily_factor * hourly_factor * noise,
                    'gas_used_avg': int(15000000 * daily_factor * hourly_factor * noise),
                    'gas_limit_avg': 30000000,
                    'active_validators': 1000,
                    'block_time_avg': 6.0,
                    'peer_count': 50 + int(10 * daily_factor * noise),
                    'total_transactions': int(1000 * daily_factor * hourly_factor * noise),
                    'contract_interactions': int(500 * daily_factor * hourly_factor * noise),
                    'defi_volume': 1000000 * daily_factor * hourly_factor * noise,
                    'token_price': 0.1 * daily_factor * hourly_factor * noise,
                    'market_cap': 1000000000 * daily_factor * hourly_factor * noise
                }
                
                # Calculate derived metrics
                record['gas_utilization'] = record['gas_used_avg'] / record['gas_limit_avg']
                record['network_activity'] = record['total_transactions'] * record['network_utilization']
                record['defi_activity'] = record['defi_volume'] * record['network_utilization']
                record['contract_activity'] = record['contract_interactions'] * record['gas_utilization']
                
                synthetic_data.append(record)
        
        df = pd.DataFrame(synthetic_data)
        logger.info(f"Created synthetic dataset with {len(df)} records")
        return df
    
    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create feature matrix and targets"""
        logger.info("Creating feature matrix...")
        
        if df.empty:
            logger.error("No data to create features from")
            return None, None
        
        # Select numeric features
        feature_cols = [
            'block_number', 'network_utilization', 'gas_price_avg', 'gas_used_avg', 'gas_limit_avg',
            'active_validators', 'block_time_avg', 'peer_count', 'total_transactions',
            'contract_interactions', 'defi_volume', 'token_price', 'market_cap',
            'gas_utilization', 'network_activity', 'defi_activity', 'contract_activity'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Create feature matrix
        X = df[available_cols].fillna(0).values
        
        # Create multiple targets for different prediction tasks
        targets = {
            'network_activity': df['network_activity'].shift(-1).fillna(0).values,
            'token_price': df['token_price'].shift(-1).fillna(0).values,
            'defi_volume': df['defi_volume'].shift(-1).fillna(0).values,
            'contract_activity': df['contract_activity'].shift(-1).fillna(0).values
        }
        
        # Remove last row (no target)
        X = X[:-1]
        for key in targets:
            targets[key] = targets[key][:-1]
        
        logger.info(f"Created feature matrix: {X.shape}")
        logger.info(f"Available targets: {list(targets.keys())}")
        
        return X, targets
    
    def train_neural_network(self, X, y, target_name: str):
        """Train neural network with GPU acceleration"""
        logger.info(f"Training Neural Network for {target_name} with RTX 4090...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Create model
        model = AstarPredictor(
            input_size=X_train.shape[1],
            hidden_size=256,
            num_layers=3
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).squeeze().cpu().numpy()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Neural Network ({target_name}) - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        model_name = f"astar_{target_name}_neural_network"
        torch.save(model.state_dict(), f'{model_name}.pth')
        joblib.dump(scaler, f'{model_name}_scaler.pkl')
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model
    
    def train_xgboost_model(self, X, y, target_name: str):
        """Train XGBoost model"""
        logger.info(f"Training XGBoost model for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"XGBoost ({target_name}) - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        model_name = f"astar_{target_name}_xgboost"
        joblib.dump(model, f'{model_name}.pkl')
        joblib.dump(scaler, f'{model_name}_scaler.pkl')
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model
    
    def run_astar_ml_pipeline(self):
        """Run the complete Astar ML pipeline"""
        logger.info("🚀 Starting Astar ML Pipeline")
        
        try:
            # Load data
            dataframes = self.load_astar_data()
            if not dataframes:
                logger.error("No data available for training")
                return
            
            # Create synthetic time series
            df = self.create_synthetic_time_series(dataframes)
            if df.empty:
                logger.error("No time series data created")
                return
            
            # Create features
            X, targets = self.create_features(df)
            if X is None or not targets:
                logger.error("No features created")
                return
            
            # Train models for each target
            logger.info("Training models...")
            
            for target_name, y in targets.items():
                logger.info(f"Training models for {target_name}...")
                
                # XGBoost
                self.train_xgboost_model(X, y, target_name)
                
                # Neural Network
                self.train_neural_network(X, y, target_name)
            
            # Generate insights
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "network": "Astar (ASTR)",
                "data_points": len(df),
                "features_created": X.shape[1],
                "models_trained": list(self.models.keys()),
                "targets_analyzed": list(targets.keys()),
                "recommendations": [
                    "Monitor network utilization trends for capacity planning",
                    "Track token price movements for investment decisions",
                    "Analyze DeFi volume for protocol performance",
                    "Watch contract activity for developer engagement",
                    "Use ML predictions for automated trading strategies",
                    "Implement real-time monitoring based on predictions"
                ]
            }
            
            # Save insights
            with open('astar_ml_insights.json', 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.success("🎉 Astar ML pipeline completed successfully!")
            return insights
            
        except Exception as e:
            logger.error(f"Error in ML pipeline: {e}")
            raise

def main():
    """Main function"""
    # Initialize ML pipeline
    ml_pipeline = AstarMLPipeline()
    
    # Run Astar ML pipeline
    insights = ml_pipeline.run_astar_ml_pipeline()
    
    if insights:
        print("\n📊 Astar ML Results:")
        print("=" * 50)
        print(f"Data points: {insights['data_points']}")
        print(f"Features created: {insights['features_created']}")
        print(f"Models trained: {len(insights['models_trained'])}")
        print(f"Targets analyzed: {insights['targets_analyzed']}")
        
        print("\n🎯 Key Insights:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")

if __name__ == "__main__":
    main()
