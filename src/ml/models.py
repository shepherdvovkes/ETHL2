import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import json
from loguru import logger
from config.settings import settings
from huggingface_hub import hf_hub_download, login
import os

# Login to Hugging Face
if settings.HF_TOKEN:
    login(token=settings.HF_TOKEN)

class InvestmentScoreModel(nn.Module):
    """Neural network model for investment score prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32, 16]):
        super(InvestmentScoreModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DeFiAnalyticsML:
    """Machine Learning pipeline for DeFi analytics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = "investment_score"
        self.model_path = settings.ML_MODEL_PATH
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        # Select relevant features
        feature_columns = [
            # On-chain metrics
            'tvl', 'tvl_change_24h', 'tvl_change_7d',
            'daily_transactions', 'transaction_volume_24h',
            'active_addresses_24h', 'new_addresses_24h',
            'new_contracts_deployed', 'contract_interactions_24h',
            'gas_price_avg', 'network_utilization',
            
            # GitHub metrics
            'commits_30d', 'open_prs', 'merged_prs_7d',
            'active_contributors_30d', 'stars', 'forks',
            'open_issues', 'closed_issues_7d',
            
            # Financial metrics
            'market_cap', 'volume_24h', 'volatility_24h',
            'volatility_7d', 'price_change_24h', 'price_change_7d',
            'price_change_30d', 'circulating_supply', 'total_supply'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Create additional engineered features
        if 'market_cap' in data.columns and 'volume_24h' in data.columns:
            data['volume_market_cap_ratio'] = data['volume_24h'] / (data['market_cap'] + 1)
        
        if 'tvl' in data.columns and 'market_cap' in data.columns:
            data['tvl_market_cap_ratio'] = data['tvl'] / (data['market_cap'] + 1)
        
        if 'active_addresses_24h' in data.columns and 'daily_transactions' in data.columns:
            data['tx_per_address'] = data['daily_transactions'] / (data['active_addresses_24h'] + 1)
        
        if 'commits_30d' in data.columns and 'active_contributors_30d' in data.columns:
            data['commits_per_contributor'] = data['commits_30d'] / (data['active_contributors_30d'] + 1)
        
        if 'stars' in data.columns and 'forks' in data.columns:
            data['stars_forks_ratio'] = data['stars'] / (data['forks'] + 1)
        
        # Add new features to list
        additional_features = [
            'volume_market_cap_ratio', 'tvl_market_cap_ratio', 'tx_per_address',
            'commits_per_contributor', 'stars_forks_ratio'
        ]
        
        self.feature_columns = available_features + [
            col for col in additional_features if col in data.columns
        ]
        
        # Fill missing values
        data_processed = data[self.feature_columns].fillna(0)
        
        # Handle infinite values
        data_processed = data_processed.replace([np.inf, -np.inf], 0)
        
        return data_processed
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """Create target variable for investment score based on multiple factors"""
        score = 0.0
        
        # TVL growth (25%)
        if 'tvl_change_7d' in data.columns:
            tvl_score = np.clip(data['tvl_change_7d'] / 100, -1, 1) * 0.25
            score += tvl_score
        
        # Price performance (20%)
        if 'price_change_7d' in data.columns:
            price_score = np.clip(data['price_change_7d'] / 100, -1, 1) * 0.20
            score += price_score
        
        # Development activity (20%)
        if 'commits_30d' in data.columns:
            dev_score = np.clip(data['commits_30d'] / 100, 0, 1) * 0.20
            score += dev_score
        
        # Volume activity (15%)
        if 'volume_24h' in data.columns and 'market_cap' in data.columns:
            volume_ratio = data['volume_24h'] / (data['market_cap'] + 1)
            volume_score = np.clip(volume_ratio * 100, 0, 1) * 0.15
            score += volume_score
        
        # User activity (10%)
        if 'active_addresses_24h' in data.columns:
            user_score = np.clip(data['active_addresses_24h'] / 10000, 0, 1) * 0.10
            score += user_score
        
        # Community engagement (10%)
        if 'stars' in data.columns:
            community_score = np.clip(data['stars'] / 1000, 0, 1) * 0.10
            score += community_score
        
        # Normalize to 0-1 range
        return np.clip(score, 0, 1)
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train multiple ML models"""
        logger.info("Starting model training...")
        
        # Prepare features and target
        X = self.prepare_features(data)
        y = self.create_target_variable(data)
        
        logger.info(f"Training data shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        scores = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name in ['linear_regression', 'ridge_regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            scores[name] = {
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            # Save model
            self.models[name] = model
            
            logger.info(f"Model {name} trained - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}")
        
        return scores
    
    def train_neural_network(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train neural network model"""
        logger.info("Training neural network...")
        
        # Prepare features and target
        X = self.prepare_features(data)
        y = self.create_target_variable(data)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        # Create model
        model = InvestmentScoreModel(input_size=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
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
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            mse = criterion(y_pred, y_test).item()
            r2 = r2_score(y_test.numpy(), y_pred.numpy())
            mae = mean_absolute_error(y_test.numpy(), y_pred.numpy())
        
        self.models['neural_network'] = model
        
        logger.info(f"Neural network trained - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def predict_investment_score(self, data: Dict[str, float], model_name: str = 'random_forest') -> float:
        """Predict investment score for given data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        if model_name == 'neural_network':
            X_tensor = torch.FloatTensor(X.values)
            model.eval()
            with torch.no_grad():
                prediction = model(X_tensor).item()
        else:
            if model_name in ['linear_regression', 'ridge_regression'] and 'main' in self.scalers:
                X_scaled = self.scalers['main'].transform(X)
                prediction = model.predict(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
        
        return float(prediction)
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def save_models(self, path: str = None):
        """Save trained models"""
        if path is None:
            path = self.model_path
        
        os.makedirs(path, exist_ok=True)
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'neural_network':
                joblib.dump(model, f"{path}/{name}.joblib")
        
        # Save neural network
        if 'neural_network' in self.models:
            torch.save(self.models['neural_network'].state_dict(), f"{path}/neural_network.pth")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{name}_scaler.joblib")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{path}/feature_columns.joblib")
        
        # Save model metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "model_path": path,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = None):
        """Load trained models"""
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            logger.warning(f"Model path {path} does not exist")
            return
        
        # Load sklearn models
        for name in ['random_forest', 'gradient_boosting', 'linear_regression', 'ridge_regression']:
            model_path = f"{path}/{name}.joblib"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
        
        # Load neural network
        nn_path = f"{path}/neural_network.pth"
        if os.path.exists(nn_path):
            # Load feature columns first to get input size
            features_path = f"{path}/feature_columns.joblib"
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
                model = InvestmentScoreModel(input_size=len(self.feature_columns))
                model.load_state_dict(torch.load(nn_path))
                self.models['neural_network'] = model
                logger.info("Loaded neural network model")
        
        # Load scalers
        for name in ['main']:
            scaler_path = f"{path}/{name}_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
                logger.info(f"Loaded {name} scaler")
        
        # Load feature columns
        features_path = f"{path}/feature_columns.joblib"
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
            logger.info(f"Loaded feature columns: {len(self.feature_columns)} features")
        
        logger.info(f"Models loaded from {path}")
    
    def evaluate_model_performance(self, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, float]:
        """Evaluate model performance on new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Prepare features and target
        X = self.prepare_features(data)
        y = self.create_target_variable(data)
        
        model = self.models[model_name]
        
        # Make predictions
        if model_name == 'neural_network':
            X_tensor = torch.FloatTensor(X.values)
            model.eval()
            with torch.no_grad():
                y_pred = model(X_tensor).numpy().flatten()
        else:
            if model_name in ['linear_regression', 'ridge_regression'] and 'main' in self.scalers:
                X_scaled = self.scalers['main'].transform(X)
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            "available_models": list(self.models.keys()),
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
            "scalers": list(self.scalers.keys()),
            "model_path": self.model_path
        }
        
        # Add feature importance for tree-based models
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name in self.models:
                importance = self.get_feature_importance(model_name)
                summary[f"{model_name}_top_features"] = dict(list(importance.items())[:10])
        
        return summary
