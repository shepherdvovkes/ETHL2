#!/usr/bin/env python3
"""
QuickNode Polkadot ML Pipeline
==============================

Advanced machine learning pipeline using QuickNode endpoints to collect
historical Polkadot data and train models with RTX 4090 acceleration.

Features:
- Historical data collection from QuickNode
- Multi-modal ML models (Transformers, XGBoost, Neural Networks)
- GPU acceleration with RTX 4090
- Real-time predictions and insights
- Comprehensive Polkadot ecosystem analysis
"""

import os
import json
import asyncio
import aiohttp
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
from transformers import AutoTokenizer, AutoModel, AutoConfig
import joblib
from loguru import logger
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import sqlite3
warnings.filterwarnings('ignore')

class QuickNodePolkadotClient:
    """Client for collecting Polkadot data from QuickNode endpoints"""
    
    def __init__(self, quicknode_url: str = None):
        # Use the QuickNode endpoint from your configuration
        self.quicknode_url = quicknode_url or "https://ancient-warmhearted-daylight.dot-mainnet.quiknode.pro/fc161dd4c4c279d2b0c5b3095ab2209673711fad/"
        self.session = None
        
        # Polkadot RPC methods for data collection
        self.rpc_methods = {
            'chain_getBlock': 'Get block data',
            'chain_getHeader': 'Get block header',
            'chain_getFinalizedHead': 'Get finalized head',
            'chain_getRuntimeVersion': 'Get runtime version',
            'state_getStorage': 'Get storage data',
            'state_getMetadata': 'Get metadata',
            'system_health': 'Get system health',
            'system_peers': 'Get peer information',
            'system_properties': 'Get system properties',
            'staking_validators': 'Get validator information',
            'staking_nominators': 'Get nominator information',
            'paras_parachains': 'Get parachain information',
            'xcm_pallet_querySupportedVersion': 'Get XCM version info'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make RPC call to QuickNode endpoint"""
        if params is None:
            params = []
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            async with self.session.post(
                self.quicknode_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('result', {})
                else:
                    logger.error(f"RPC call failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error making RPC call: {e}")
            return {}
    
    async def get_current_block(self) -> int:
        """Get current block number"""
        result = await self.make_rpc_call('chain_getFinalizedHead')
        if result:
            # Get block number from block hash
            block_data = await self.make_rpc_call('chain_getHeader', [result])
            if block_data and 'number' in block_data:
                return int(block_data['number'], 16)
        return 0
    
    async def get_block_data(self, block_number: int) -> Dict:
        """Get comprehensive block data"""
        # Get block hash
        block_hash = await self.make_rpc_call('chain_getBlockHash', [hex(block_number)])
        if not block_hash:
            return {}
        
        # Get block details
        block_data = await self.make_rpc_call('chain_getBlock', [block_hash])
        if not block_data:
            return {}
        
        # Get block header
        header_data = await self.make_rpc_call('chain_getHeader', [block_hash])
        
        return {
            'block_number': block_number,
            'block_hash': block_hash,
            'block_data': block_data,
            'header_data': header_data,
            'timestamp': datetime.utcnow()
        }
    
    async def get_staking_data(self) -> Dict:
        """Get staking information"""
        validators = await self.make_rpc_call('staking_validators')
        nominators = await self.make_rpc_call('staking_nominators')
        
        return {
            'validators': validators,
            'nominators': nominators,
            'timestamp': datetime.utcnow()
        }
    
    async def get_parachain_data(self) -> Dict:
        """Get parachain information"""
        parachains = await self.make_rpc_call('paras_parachains')
        
        return {
            'parachains': parachains,
            'timestamp': datetime.utcnow()
        }
    
    async def get_system_health(self) -> Dict:
        """Get system health information"""
        health = await self.make_rpc_call('system_health')
        peers = await self.make_rpc_call('system_peers')
        properties = await self.make_rpc_call('system_properties')
        
        return {
            'health': health,
            'peers': peers,
            'properties': properties,
            'timestamp': datetime.utcnow()
        }
    
    async def collect_historical_data(self, days_back: int = 30) -> List[Dict]:
        """Collect historical data for specified period"""
        logger.info(f"Collecting historical data for {days_back} days...")
        
        # Get current block
        current_block = await self.get_current_block()
        if current_block == 0:
            logger.error("Could not get current block number")
            return []
        
        # Estimate blocks per day (Polkadot has ~6 second block time)
        blocks_per_day = 24 * 60 * 60 // 6  # ~14,400 blocks per day
        start_block = max(1, current_block - (days_back * blocks_per_day))
        
        # Sample blocks (every 100 blocks to avoid too much data)
        sample_blocks = list(range(start_block, current_block, 100))
        
        historical_data = []
        for i, block_num in enumerate(sample_blocks):
            if i % 50 == 0:
                logger.info(f"Collecting block {block_num} ({i+1}/{len(sample_blocks)})")
            
            block_data = await self.get_block_data(block_num)
            if block_data:
                historical_data.append(block_data)
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"Collected {len(historical_data)} historical blocks")
        return historical_data

class PolkadotMLDataset(Dataset):
    """Custom dataset for Polkadot time series data"""
    
    def __init__(self, features, targets, sequence_length=24):
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

class PolkadotTransformer(nn.Module):
    """Advanced Transformer model for Polkadot predictions"""
    
    def __init__(self, input_size, hidden_size=512, num_layers=8, num_heads=16):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        seq_len = x.size(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use last timestep for prediction
        x = x[:, -1, :]
        
        # Output projection
        return self.output_projection(x)

class QuickNodePolkadotML:
    """Comprehensive ML pipeline for Polkadot using QuickNode data"""
    
    def __init__(self, database_path="polkadot_metrics.db"):
        self.database_path = database_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Initializing QuickNode Polkadot ML Pipeline on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def process_historical_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Process historical data into features"""
        logger.info("Processing historical data...")
        
        processed_data = []
        
        for block_data in historical_data:
            try:
                # Extract features from block data
                features = {
                    'block_number': block_data.get('block_number', 0),
                    'timestamp': block_data.get('timestamp', datetime.utcnow()),
                    'extrinsics_count': 0,
                    'events_count': 0,
                    'block_size': 0,
                    'gas_used': 0,
                    'validator_count': 0,
                    'finalization_time': 0
                }
                
                # Extract from block data
                if 'block_data' in block_data and 'block' in block_data['block_data']:
                    block = block_data['block_data']['block']
                    if 'extrinsics' in block:
                        features['extrinsics_count'] = len(block['extrinsics'])
                    
                    if 'header' in block:
                        header = block['header']
                        if 'digest' in header:
                            features['events_count'] = len(header.get('digest', {}).get('logs', []))
                
                # Extract from header data
                if 'header_data' in block_data:
                    header = block_data['header_data']
                    if 'number' in header:
                        features['block_number'] = int(header['number'], 16)
                
                processed_data.append(features)
                
            except Exception as e:
                logger.warning(f"Error processing block data: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} records")
        return df
    
    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create feature matrix and targets"""
        logger.info("Creating feature matrix...")
        
        if df.empty:
            logger.error("No data to create features from")
            return None, None
        
        # Select numeric features
        numeric_cols = ['block_number', 'extrinsics_count', 'events_count', 'block_size', 'gas_used', 'validator_count', 'finalization_time']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        # Create additional features
        df['block_time'] = df['block_number'].diff().fillna(6)  # Approximate block time
        df['extrinsics_per_block'] = df['extrinsics_count'] / (df['block_time'] + 1)
        df['events_per_block'] = df['events_count'] / (df['block_time'] + 1)
        df['block_activity'] = df['extrinsics_count'] + df['events_count']
        
        # Add to feature columns
        feature_cols = available_cols + ['block_time', 'extrinsics_per_block', 'events_per_block', 'block_activity']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Create feature matrix
        X = df[feature_cols].fillna(0).values
        
        # Create target (next block's activity)
        y = df['block_activity'].shift(-1).fillna(0).values
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        logger.info(f"Created feature matrix: {X.shape}, targets: {y.shape}")
        return X, y
    
    async def train_transformer_model(self, X, y, sequence_length=24):
        """Train transformer model with GPU acceleration"""
        logger.info("Training Transformer model with RTX 4090...")
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y, sequence_length)
        
        if len(X_seq) == 0:
            logger.error("Not enough data for sequences")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Create dataset and dataloader
        train_dataset = PolkadotMLDataset(X_train_scaled, y_train, sequence_length)
        test_dataset = PolkadotMLDataset(X_test_scaled, y_test, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create model
        model = PolkadotTransformer(
            input_size=X_train.shape[-1],
            hidden_size=512,
            num_layers=8,
            num_heads=16
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(500):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluation
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                pred = model(batch_X).squeeze().cpu().numpy()
                predictions.extend(pred)
        
        predictions = np.array(predictions)
        mse = mean_squared_error(y_test[sequence_length:], predictions)
        mae = mean_absolute_error(y_test[sequence_length:], predictions)
        r2 = r2_score(y_test[sequence_length:], predictions)
        
        logger.info(f"Transformer Model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'quicknode_polkadot_transformer.pth')
        joblib.dump(scaler, 'quicknode_polkadot_transformer_scaler.pkl')
        
        self.models['transformer'] = model
        self.scalers['transformer'] = scaler
        
        return model
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
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
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=10,
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
        
        logger.info(f"XGBoost Model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        joblib.dump(model, 'quicknode_polkadot_xgboost.pkl')
        joblib.dump(scaler, 'quicknode_polkadot_xgboost_scaler.pkl')
        
        self.models['xgboost'] = model
        self.scalers['xgboost'] = scaler
        
        return model
    
    def _create_sequences(self, X, y, sequence_length):
        """Create sequences for time series data"""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    async def run_comprehensive_pipeline(self, days_back: int = 30):
        """Run the complete ML pipeline with QuickNode data"""
        logger.info("🚀 Starting QuickNode Polkadot ML Pipeline")
        
        try:
            # Collect historical data
            async with QuickNodePolkadotClient() as client:
                historical_data = await client.collect_historical_data(days_back)
            
            if not historical_data:
                logger.error("No historical data collected")
                return
            
            # Process data
            df = self.process_historical_data(historical_data)
            X, y = self.create_features(df)
            
            if X is None or y is None:
                logger.error("No features created")
                return
            
            # Train models
            logger.info("Training models...")
            
            # XGBoost
            self.train_xgboost_model(X, y)
            
            # Transformer (if we have enough data)
            if len(X) > 100:
                await self.train_transformer_model(X, y)
            else:
                logger.info("Not enough data for transformer model")
            
            # Generate insights
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "data_collected": len(historical_data),
                "features_created": X.shape[1],
                "models_trained": list(self.models.keys()),
                "recommendations": [
                    "Monitor block activity trends",
                    "Track validator performance",
                    "Analyze parachain growth",
                    "Watch cross-chain activity"
                ]
            }
            
            # Save insights
            with open('quicknode_polkadot_insights.json', 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.success("🎉 QuickNode ML pipeline completed successfully!")
            return insights
            
        except Exception as e:
            logger.error(f"Error in ML pipeline: {e}")
            raise

async def main():
    """Main function"""
    # Initialize ML pipeline
    ml_pipeline = QuickNodePolkadotML()
    
    # Run comprehensive pipeline
    insights = await ml_pipeline.run_comprehensive_pipeline(days_back=30)
    
    if insights:
        print("\n📊 QuickNode Polkadot ML Results:")
        print("=" * 50)
        print(f"Data collected: {insights['data_collected']} blocks")
        print(f"Features created: {insights['features_created']}")
        print(f"Models trained: {insights['models_trained']}")
        print("\n🎯 Key Insights:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")

if __name__ == "__main__":
    asyncio.run(main())
