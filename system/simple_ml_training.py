#!/usr/bin/env python3
"""
Simple ML Training Script for L2 Networks Data
Uses existing JSON data to train ML models with RTX 4090 acceleration
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def load_l2_data():
    """Load L2 networks data from JSON file"""
    logger.info("Loading L2 networks data...")
    
    with open('l2_networks_analysis.json', 'r') as f:
        data = json.load(f)
    
    networks = data['networks']
    logger.info(f"Loaded {len(networks)} L2 networks")
    
    return networks

def extract_features(networks):
    """Extract features from L2 networks data"""
    logger.info("Extracting features from L2 networks data...")
    
    features = []
    targets = []
    
    for network in networks:
        try:
            # Extract basic features
            basic = network.get('basic_info', {})
            tech = network.get('technical_specs', {})
            perf = network.get('performance', {})
            econ = network.get('economics', {})
            sec = network.get('security', {})
            
            # Create feature vector
            feature_vector = [
                # Basic metrics
                basic.get('tvl_usd', 0) / 1e9,  # TVL in billions
                basic.get('tps', 0),
                
                # Performance metrics
                perf.get('transactions_per_second', 0),
                perf.get('gas_fee_reduction', 0),
                perf.get('throughput_improvement', 0),
                
                # Economic metrics
                econ.get('total_value_locked', 0) / 1e9,  # TVL in billions
                econ.get('daily_volume', 0) / 1e6,  # Daily volume in millions
                econ.get('active_users_24h', 0),
                econ.get('transaction_fees_24h', 0),
                econ.get('revenue_24h', 0),
                econ.get('market_cap', 0) / 1e9,  # Market cap in billions
                
                # Security metrics
                sec.get('validator_count', 0),
                1 if sec.get('slashing_mechanism', False) else 0,
                1 if sec.get('multisig_required', False) else 0,
                
                # Technical features
                1 if tech.get('evm_compatibility', False) else 0,
                1 if tech.get('fraud_proofs', False) else 0,
                1 if tech.get('zero_knowledge_proofs', False) else 0,
            ]
            
            # Target: TVL growth potential (simplified)
            tvl = basic.get('tvl_usd', 0)
            daily_volume = econ.get('daily_volume', 0)
            active_users = econ.get('active_users_24h', 0)
            
            # Calculate a simple growth score
            if tvl > 0 and daily_volume > 0:
                growth_score = (daily_volume / tvl) * (active_users / 1000) * 100
            else:
                growth_score = 0
            
            features.append(feature_vector)
            targets.append(growth_score)
            
        except Exception as e:
            logger.warning(f"Error processing network: {e}")
            continue
    
    return np.array(features), np.array(targets)

def create_neural_network(input_size, hidden_size=128):
    """Create a neural network for GPU training"""
    class L2NetworkPredictor(nn.Module):
        def __init__(self, input_size, hidden_size):
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
    
    return L2NetworkPredictor(input_size, hidden_size)

def train_models(X, y):
    """Train multiple ML models"""
    logger.info("Training ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    results['Random Forest'] = {'MSE': rf_mse, 'R²': rf_r2}
    
    # 2. XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    results['XGBoost'] = {'MSE': xgb_mse, 'R²': xgb_r2}
    
    # 3. Neural Network (GPU accelerated)
    logger.info("Training Neural Network on GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = create_neural_network(X_train_scaled.shape[1])
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        nn_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
    # Ensure predictions are in the right format
    if nn_pred.ndim == 0:
        nn_pred = np.array([nn_pred])
    
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred)
    results['Neural Network (GPU)'] = {'MSE': nn_mse, 'R²': nn_r2}
    
    return results, scaler, model

def main():
    """Main training function"""
    logger.info("🚀 Starting L2 Networks ML Training with RTX 4090")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("⚠️ GPU not available, using CPU")
    
    # Load and process data
    networks = load_l2_data()
    X, y = extract_features(networks)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    if len(X) == 0:
        logger.error("No data available for training")
        return
    
    # Train models
    results, scaler, model = train_models(X, y)
    
    # Print results
    logger.info("📊 Training Results:")
    logger.info("=" * 50)
    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  MSE: {metrics['MSE']:.4f}")
        logger.info(f"  R²: {metrics['R²']:.4f}")
    
    # Save models
    import joblib
    joblib.dump(scaler, 'l2_scaler.pkl')
    torch.save(model.state_dict(), 'l2_neural_network.pth')
    logger.info("✅ Models saved successfully")
    
    logger.success("🎉 ML training completed successfully!")

if __name__ == "__main__":
    main()
