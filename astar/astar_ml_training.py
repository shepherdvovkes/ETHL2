#!/usr/bin/env python3
"""
Astar ML Training Pipeline
Train XGBoost and Neural Network models for price prediction
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AstarPricePredictor:
    """Astar price prediction using multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load processed ML data"""
        print("📊 Loading ML training data...")
        
        # Load features
        self.features_df = pd.read_csv('astar_ml_features.csv')
        
        # Load feature names
        with open('astar_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Prepare feature matrix
        X = self.features_df[self.feature_names].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Target variables - predict current price based on network activity
        y_price = self.features_df['price_usd'].fillna(0)
        
        # Create price change targets based on current price patterns
        y_1h = self.features_df['price_change_1h'].fillna(0)
        y_24h = self.features_df['price_change_24h'].fillna(0)
        
        print(f"✅ Loaded {len(X):,} samples with {len(self.feature_names)} features")
        
        return X, y_price, y_1h, y_24h
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data for training"""
        # For price prediction, use all samples
        # For price change, filter out samples with zero change
        if 'price_change' in str(y.name) if hasattr(y, 'name') else False:
            valid_indices = y != 0
            X_valid = X[valid_indices]
            y_valid = y[valid_indices]
            print(f"📈 Valid samples for training: {len(X_valid):,}")
        else:
            X_valid = X
            y_valid = y
            print(f"📈 All samples for training: {len(X_valid):,}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, target_name):
        """Train XGBoost model"""
        print(f"🚀 Training XGBoost for {target_name}...")
        
        # XGBoost parameters
        params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        results = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"✅ XGBoost {target_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.6f}")
        
        return results
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, target_name):
        """Train Neural Network model"""
        print(f"🧠 Training Neural Network for {target_name}...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Neural Network architecture
        class PricePredictorNN(nn.Module):
            def __init__(self, input_size):
                super(PricePredictorNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Initialize model
        model = PricePredictorNN(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'astar_nn_{target_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'astar_nn_{target_name}_best.pth'))
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze().numpy()
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"✅ Neural Network {target_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.6f}")
        
        return results
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, target_name):
        """Train Random Forest model"""
        print(f"🌲 Training Random Forest for {target_name}...")
        
        # Random Forest parameters
        params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        results = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"✅ Random Forest {target_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.6f}")
        
        return results
    
    def train_all_models(self):
        """Train all models for both 1h and 24h predictions"""
        print("🚀 Starting Astar ML Training Pipeline")
        print("=" * 60)
        
        # Load data
        X, y_price, y_1h, y_24h = self.load_data()
        
        # Train models for price prediction
        print("\n💰 Training models for price prediction...")
        X_train_price, X_test_price, y_train_price, y_test_price, scaler_price = self.prepare_data(X, y_price)
        
        self.results['price'] = {}
        self.scalers['price'] = scaler_price
        
        # XGBoost Price
        self.results['price']['xgboost'] = self.train_xgboost(
            X_train_price, X_test_price, y_train_price, y_test_price, 'price'
        )
        
        # Neural Network Price
        self.results['price']['neural_network'] = self.train_neural_network(
            X_train_price, X_test_price, y_train_price, y_test_price, 'price'
        )
        
        # Random Forest Price
        self.results['price']['random_forest'] = self.train_random_forest(
            X_train_price, X_test_price, y_train_price, y_test_price, 'price'
        )
        
        # Train models for 1-hour prediction
        print("\n📈 Training models for 1-hour price prediction...")
        X_train_1h, X_test_1h, y_train_1h, y_test_1h, scaler_1h = self.prepare_data(X, y_1h)
        
        self.results['1h'] = {}
        self.scalers['1h'] = scaler_1h
        
        # XGBoost 1h
        self.results['1h']['xgboost'] = self.train_xgboost(
            X_train_1h, X_test_1h, y_train_1h, y_test_1h, '1h'
        )
        
        # Neural Network 1h
        self.results['1h']['neural_network'] = self.train_neural_network(
            X_train_1h, X_test_1h, y_train_1h, y_test_1h, '1h'
        )
        
        # Random Forest 1h
        self.results['1h']['random_forest'] = self.train_random_forest(
            X_train_1h, X_test_1h, y_train_1h, y_test_1h, '1h'
        )
        
        # Train models for 24-hour prediction
        print("\n📈 Training models for 24-hour price prediction...")
        X_train_24h, X_test_24h, y_train_24h, y_test_24h, scaler_24h = self.prepare_data(X, y_24h)
        
        self.results['24h'] = {}
        self.scalers['24h'] = scaler_24h
        
        # XGBoost 24h
        self.results['24h']['xgboost'] = self.train_xgboost(
            X_train_24h, X_test_24h, y_train_24h, y_test_24h, '24h'
        )
        
        # Neural Network 24h
        self.results['24h']['neural_network'] = self.train_neural_network(
            X_train_24h, X_test_24h, y_train_24h, y_test_24h, '24h'
        )
        
        # Random Forest 24h
        self.results['24h']['random_forest'] = self.train_random_forest(
            X_train_24h, X_test_24h, y_train_24h, y_test_24h, '24h'
        )
        
        # Save models and results
        self.save_models()
        self.generate_report()
        
        return self.results
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving trained models...")
        
        # Save XGBoost models
        self.results['price']['xgboost']['model'].save_model('astar_xgboost_price.json')
        self.results['1h']['xgboost']['model'].save_model('astar_xgboost_1h.json')
        self.results['24h']['xgboost']['model'].save_model('astar_xgboost_24h.json')
        
        # Save scalers
        with open('astar_scaler_price.pkl', 'wb') as f:
            pickle.dump(self.scalers['price'], f)
        with open('astar_scaler_1h.pkl', 'wb') as f:
            pickle.dump(self.scalers['1h'], f)
        with open('astar_scaler_24h.pkl', 'wb') as f:
            pickle.dump(self.scalers['24h'], f)
        
        # Save feature names
        with open('astar_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        print("✅ Models saved successfully!")
    
    def generate_report(self):
        """Generate training report"""
        print("\n📊 Generating ML Training Report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "total_samples": len(self.features_df),
                "features": len(self.feature_names),
                "feature_names": self.feature_names
            },
            "model_performance": {}
        }
        
        # Performance summary
        for timeframe in ['price', '1h', '24h']:
            report["model_performance"][timeframe] = {}
            for model_name in ['xgboost', 'neural_network', 'random_forest']:
                if model_name in self.results[timeframe]:
                    results = self.results[timeframe][model_name]
                    report["model_performance"][timeframe][model_name] = {
                        "r2_score": float(results['r2']),
                        "rmse": float(results['rmse']),
                        "mae": float(results['mae']),
                        "mse": float(results['mse'])
                    }
        
        # Best models
        best_models = {}
        for timeframe in ['price', '1h', '24h']:
            best_r2 = -1
            best_model = None
            for model_name, results in self.results[timeframe].items():
                if results['r2'] > best_r2:
                    best_r2 = results['r2']
                    best_model = model_name
            best_models[timeframe] = {
                "model": best_model,
                "r2_score": float(best_r2)
            }
        
        report["best_models"] = best_models
        
        # Save report
        with open('astar_ml_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n🎯 Model Performance Summary:")
        print("=" * 50)
        
        for timeframe in ['price', '1h', '24h']:
            print(f"\n📈 {timeframe.upper()} Price Prediction:")
            for model_name, results in self.results[timeframe].items():
                print(f"  {model_name.replace('_', ' ').title()}: R² = {results['r2']:.4f}, RMSE = {results['rmse']:.6f}")
        
        print(f"\n🏆 Best Models:")
        for timeframe, info in best_models.items():
            print(f"  {timeframe.upper()}: {info['model'].replace('_', ' ').title()} (R² = {info['r2_score']:.4f})")
        
        print(f"\n💾 Report saved: astar_ml_training_report.json")

def main():
    """Main function"""
    predictor = AstarPricePredictor()
    results = predictor.train_all_models()
    
    print(f"\n🎉 ML Training Complete!")
    print(f"📊 Ready for price prediction with trained models")

if __name__ == "__main__":
    main()
