#!/usr/bin/env python3
"""
Parachain ML Pipeline
====================

Focused machine learning pipeline using smaller Polkadot parachains
for more manageable data collection and training with RTX 4090.

Target Parachains:
- Astar (ASTR) - Smart contracts
- Moonbeam (GLMR) - EVM compatibility  
- Acala (ACA) - DeFi
- Parallel (PARA) - DeFi
- HydraDX (HDX) - DEX
- Bifrost (BNC) - Liquid staking
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
import joblib
from loguru import logger
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
warnings.filterwarnings('ignore')

class ParachainClient:
    """Client for collecting data from specific parachains"""
    
    def __init__(self):
        # Smaller parachains with their RPC endpoints
        self.parachains = {
            "astar": {
                "id": 2006,
                "name": "Astar",
                "symbol": "ASTR",
                "rpc": "https://rpc.astar.network",
                "ws": "wss://rpc.astar.network",
                "category": "smart_contracts",
                "description": "Multi-VM smart contract platform"
            },
            "moonbeam": {
                "id": 2004,
                "name": "Moonbeam",
                "symbol": "GLMR", 
                "rpc": "https://rpc.api.moonbeam.network",
                "ws": "wss://wss.api.moonbeam.network",
                "category": "evm_compatibility",
                "description": "EVM-compatible smart contract platform"
            },
            "acala": {
                "id": 2000,
                "name": "Acala",
                "symbol": "ACA",
                "rpc": "https://eth-rpc-acala.aca-api.network",
                "ws": "wss://eth-rpc-acala.aca-api.network",
                "category": "defi",
                "description": "DeFi hub and stablecoin platform"
            },
            "parallel": {
                "id": 2012,
                "name": "Parallel",
                "symbol": "PARA",
                "rpc": "https://rpc.parallel.fi",
                "ws": "wss://rpc.parallel.fi",
                "category": "defi",
                "description": "DeFi and liquid staking platform"
            },
            "hydradx": {
                "id": 2034,
                "name": "HydraDX",
                "symbol": "HDX",
                "rpc": "https://rpc.hydradx.cloud",
                "ws": "wss://rpc.hydradx.cloud",
                "category": "dex",
                "description": "Decentralized exchange"
            },
            "bifrost": {
                "id": 2030,
                "name": "Bifrost",
                "symbol": "BNC",
                "rpc": "https://rpc.bifrost-rpc.liebi.com",
                "ws": "wss://rpc.bifrost-rpc.liebi.com",
                "category": "liquid_staking",
                "description": "Liquid staking protocol"
            }
        }
        
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_rpc_call(self, rpc_url: str, method: str, params: List = None) -> Dict:
        """Make RPC call to parachain endpoint"""
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
                rpc_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('result', {})
                else:
                    logger.warning(f"RPC call failed for {rpc_url}: {response.status}")
                    return {}
        except Exception as e:
            logger.warning(f"Error making RPC call to {rpc_url}: {e}")
            return {}
    
    async def get_parachain_metrics(self, parachain_name: str) -> Dict:
        """Get comprehensive metrics for a specific parachain"""
        if parachain_name not in self.parachains:
            logger.error(f"Unknown parachain: {parachain_name}")
            return {}
        
        parachain = self.parachains[parachain_name]
        rpc_url = parachain['rpc']
        
        logger.info(f"Collecting metrics for {parachain['name']} ({parachain['symbol']})")
        
        # Collect various metrics
        metrics = {
            'parachain_id': parachain['id'],
            'name': parachain['name'],
            'symbol': parachain['symbol'],
            'category': parachain['category'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Get block information
            block_number = await self.make_rpc_call(rpc_url, 'eth_blockNumber')
            if block_number:
                metrics['current_block'] = int(block_number, 16)
            
            # Get gas price
            gas_price = await self.make_rpc_call(rpc_url, 'eth_gasPrice')
            if gas_price:
                metrics['gas_price'] = int(gas_price, 16) / 1e18  # Convert to ETH
            
            # Get network info
            peer_count = await self.make_rpc_call(rpc_url, 'net_peerCount')
            if peer_count:
                metrics['peer_count'] = int(peer_count, 16)
            
            # Get latest block details
            if 'current_block' in metrics:
                block_data = await self.make_rpc_call(
                    rpc_url, 
                    'eth_getBlockByNumber', 
                    [hex(metrics['current_block']), True]
                )
                if block_data:
                    metrics['block_size'] = len(str(block_data))
                    metrics['transaction_count'] = len(block_data.get('transactions', []))
                    metrics['gas_used'] = int(block_data.get('gasUsed', '0x0'), 16)
                    metrics['gas_limit'] = int(block_data.get('gasLimit', '0x0'), 16)
                    
                    # Calculate utilization
                    if metrics['gas_limit'] > 0:
                        metrics['gas_utilization'] = metrics['gas_used'] / metrics['gas_limit']
                    else:
                        metrics['gas_utilization'] = 0
            
            # Get account count (approximate)
            # This is a simplified approach - in practice you'd use more sophisticated methods
            metrics['estimated_accounts'] = metrics.get('current_block', 0) * 10  # Rough estimate
            
            logger.info(f"Collected {len(metrics)} metrics for {parachain['name']}")
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {parachain_name}: {e}")
        
        return metrics
    
    async def collect_all_parachain_data(self) -> List[Dict]:
        """Collect data from all configured parachains"""
        logger.info("Collecting data from all parachains...")
        
        all_data = []
        
        for parachain_name in self.parachains.keys():
            try:
                metrics = await self.get_parachain_metrics(parachain_name)
                if metrics:
                    all_data.append(metrics)
                
                # Rate limiting between parachains
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data for {parachain_name}: {e}")
                continue
        
        logger.info(f"Collected data from {len(all_data)} parachains")
        return all_data

class ParachainMLDataset(Dataset):
    """Custom dataset for parachain time series data"""
    
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

class ParachainPredictor(nn.Module):
    """Neural network for parachain activity prediction"""
    
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class ParachainMLPipeline:
    """ML pipeline for parachain analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Initializing Parachain ML Pipeline on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_synthetic_data(self, parachain_data: List[Dict]) -> pd.DataFrame:
        """Create synthetic time series data based on parachain metrics"""
        logger.info("Creating synthetic time series data...")
        
        if not parachain_data:
            logger.error("No parachain data available")
            return pd.DataFrame()
        
        # Create time series data by simulating historical trends
        all_records = []
        
        for parachain in parachain_data:
            base_metrics = {
                'parachain_id': parachain['parachain_id'],
                'name': parachain['name'],
                'symbol': parachain['symbol'],
                'category': parachain['category']
            }
            
            # Generate 30 days of synthetic data
            for day in range(30):
                for hour in range(24):
                    timestamp = datetime.utcnow() - timedelta(days=30-day, hours=23-hour)
                    
                    # Create realistic variations
                    base_block = parachain.get('current_block', 1000000)
                    base_tx_count = parachain.get('transaction_count', 100)
                    base_gas_price = parachain.get('gas_price', 0.00002)
                    
                    # Add daily and hourly patterns
                    daily_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
                    hourly_factor = 1 + 0.2 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
                    noise = np.random.normal(1, 0.1)  # Random noise
                    
                    record = {
                        'timestamp': timestamp,
                        'parachain_id': base_metrics['parachain_id'],
                        'name': base_metrics['name'],
                        'symbol': base_metrics['symbol'],
                        'category': base_metrics['category'],
                        'block_number': base_block + (day * 24 + hour) * 6,  # ~6 second blocks
                        'transaction_count': int(base_tx_count * daily_factor * hourly_factor * noise),
                        'gas_price': base_gas_price * daily_factor * hourly_factor * noise,
                        'gas_used': int(parachain.get('gas_used', 1000000) * daily_factor * hourly_factor * noise),
                        'gas_limit': parachain.get('gas_limit', 30000000),
                        'peer_count': parachain.get('peer_count', 50),
                        'estimated_accounts': int(parachain.get('estimated_accounts', 100000) * (1 + day * 0.01)),
                        'gas_utilization': parachain.get('gas_utilization', 0.3) * daily_factor * hourly_factor * noise
                    }
                    
                    # Calculate derived metrics
                    record['gas_utilization'] = min(1.0, record['gas_used'] / record['gas_limit'])
                    record['tx_per_block'] = record['transaction_count'] / 6  # Approximate
                    record['network_activity'] = record['transaction_count'] * record['gas_utilization']
                    
                    all_records.append(record)
        
        df = pd.DataFrame(all_records)
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
            'block_number', 'transaction_count', 'gas_price', 'gas_used', 'gas_limit',
            'peer_count', 'estimated_accounts', 'gas_utilization', 'tx_per_block', 'network_activity'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Create feature matrix
        X = df[available_cols].fillna(0).values
        
        # Create target (next hour's network activity)
        df_sorted = df.sort_values('timestamp')
        y = df_sorted['network_activity'].shift(-1).fillna(0).values
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        logger.info(f"Created feature matrix: {X.shape}, targets: {y.shape}")
        return X, y
    
    def train_neural_network(self, X, y):
        """Train neural network with GPU acceleration"""
        logger.info("Training Neural Network with RTX 4090...")
        
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
        model = ParachainPredictor(
            input_size=X_train.shape[1],
            hidden_size=256
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
        
        logger.info(f"Neural Network - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'parachain_neural_network.pth')
        joblib.dump(scaler, 'parachain_neural_network_scaler.pkl')
        
        self.models['neural_network'] = model
        self.scalers['neural_network'] = scaler
        
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
        
        logger.info(f"XGBoost Model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save model
        joblib.dump(model, 'parachain_xgboost.pkl')
        joblib.dump(scaler, 'parachain_xgboost_scaler.pkl')
        
        self.models['xgboost'] = model
        self.scalers['xgboost'] = scaler
        
        return model
    
    async def run_parachain_pipeline(self):
        """Run the complete parachain ML pipeline"""
        logger.info("🚀 Starting Parachain ML Pipeline")
        
        try:
            # Collect parachain data
            async with ParachainClient() as client:
                parachain_data = await client.collect_all_parachain_data()
            
            if not parachain_data:
                logger.error("No parachain data collected")
                return
            
            # Create synthetic time series data
            df = self.create_synthetic_data(parachain_data)
            X, y = self.create_features(df)
            
            if X is None or y is None:
                logger.error("No features created")
                return
            
            # Train models
            logger.info("Training models...")
            
            # XGBoost
            self.train_xgboost_model(X, y)
            
            # Neural Network
            self.train_neural_network(X, y)
            
            # Generate insights
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "parachains_analyzed": len(parachain_data),
                "data_points": len(df),
                "features_created": X.shape[1],
                "models_trained": list(self.models.keys()),
                "parachain_summary": [
                    {
                        "name": p['name'],
                        "symbol": p['symbol'],
                        "category": p['category'],
                        "current_block": p.get('current_block', 0),
                        "transaction_count": p.get('transaction_count', 0)
                    }
                    for p in parachain_data
                ],
                "recommendations": [
                    "Monitor Astar smart contract activity",
                    "Track Moonbeam EVM compatibility metrics",
                    "Analyze Acala DeFi protocol growth",
                    "Watch Parallel liquid staking trends",
                    "Monitor HydraDX DEX volume",
                    "Track Bifrost staking derivatives"
                ]
            }
            
            # Save insights
            with open('parachain_ml_insights.json', 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.success("🎉 Parachain ML pipeline completed successfully!")
            return insights
            
        except Exception as e:
            logger.error(f"Error in ML pipeline: {e}")
            raise

async def main():
    """Main function"""
    # Initialize ML pipeline
    ml_pipeline = ParachainMLPipeline()
    
    # Run parachain pipeline
    insights = await ml_pipeline.run_parachain_pipeline()
    
    if insights:
        print("\n📊 Parachain ML Results:")
        print("=" * 50)
        print(f"Parachains analyzed: {insights['parachains_analyzed']}")
        print(f"Data points: {insights['data_points']}")
        print(f"Features created: {insights['features_created']}")
        print(f"Models trained: {insights['models_trained']}")
        
        print("\n🏗️ Parachain Summary:")
        for p in insights['parachain_summary']:
            print(f"  • {p['name']} ({p['symbol']}) - {p['category']} - Block: {p['current_block']}")
        
        print("\n🎯 Key Insights:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")

if __name__ == "__main__":
    asyncio.run(main())
