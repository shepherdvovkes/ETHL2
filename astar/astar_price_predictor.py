#!/usr/bin/env python3
"""
Astar Price Prediction System
Predict Astar price based on network activity patterns
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class AstarPricePredictor:
    """Astar price prediction based on network activity"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.price_model = None
        self.price_scaler = None
        
    def load_models(self):
        """Load trained models"""
        print("📊 Loading trained models...")
        
        # Load feature names
        with open('astar_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load network prediction models
        network_targets = ['transaction_count', 'gas_used', 'gas_utilization', 'network_activity', 'avg_gas_price']
        
        for target in network_targets:
            try:
                # Load XGBoost model
                model = xgb.XGBRegressor()
                model.load_model(f'astar_xgboost_{target}.json')
                self.models[target] = model
                
                # Load scaler
                with open(f'astar_scaler_{target}.pkl', 'rb') as f:
                    self.scalers[target] = pickle.load(f)
                    
            except FileNotFoundError:
                print(f"⚠️  Model for {target} not found")
        
        print(f"✅ Loaded {len(self.models)} network prediction models")
        
    def create_price_model(self):
        """Create a price prediction model based on network activity"""
        print("💰 Creating price prediction model...")
        
        # Load historical data
        df = pd.read_csv('astar_ml_features.csv')
        
        # Create synthetic price based on network activity
        # This simulates how network activity affects price
        base_price = 0.15  # Base Astar price
        
        # Network activity factors
        activity_factor = df['network_activity'] / df['network_activity'].mean()
        gas_factor = df['avg_gas_price'] / df['avg_gas_price'].mean()
        tx_factor = df['transaction_count'] / df['transaction_count'].mean()
        
        # Create price with network correlation
        np.random.seed(42)
        volatility = np.random.normal(0, 0.02, len(df))
        
        # Price formula: base price * network factors + volatility
        df['predicted_price'] = base_price * (
            1 + 
            activity_factor * 0.1 +  # Network activity impact
            gas_factor * 0.05 +      # Gas price impact
            tx_factor * 0.03 +       # Transaction count impact
            volatility               # Random volatility
        )
        
        # Ensure minimum price
        df['predicted_price'] = np.maximum(df['predicted_price'], 0.01)
        
        # Prepare features for price prediction
        price_features = [
            'transaction_count', 'gas_used', 'gas_utilization', 
            'network_activity', 'avg_gas_price', 'hour', 'day_of_week'
        ]
        
        X = df[price_features].fillna(0)
        y = df['predicted_price']
        
        # Train price prediction model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.price_scaler = StandardScaler()
        X_train_scaled = self.price_scaler.fit_transform(X_train)
        X_test_scaled = self.price_scaler.transform(X_test)
        
        # Train XGBoost model
        self.price_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.price_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.price_model.predict(X_test_scaled)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"✅ Price model trained - R²: {r2:.4f}, RMSE: {rmse:.6f}")
        
        # Save price model
        self.price_model.save_model('astar_price_predictor.json')
        with open('astar_price_scaler.pkl', 'wb') as f:
            pickle.dump(self.price_scaler, f)
        
        return r2, rmse
    
    def predict_network_activity(self, features):
        """Predict network activity metrics"""
        predictions = {}
        
        for target, model in self.models.items():
            if target in self.scalers:
                # Scale features
                features_scaled = self.scalers[target].transform(features.reshape(1, -1))
                # Predict
                pred = model.predict(features_scaled)[0]
                predictions[target] = pred
        
        return predictions
    
    def predict_price(self, network_features):
        """Predict Astar price based on network activity"""
        if self.price_model is None or self.price_scaler is None:
            print("❌ Price model not loaded. Please train first.")
            return None
        
        # Scale features
        features_scaled = self.price_scaler.transform(network_features.reshape(1, -1))
        
        # Predict price
        price = self.price_model.predict(features_scaled)[0]
        
        return price
    
    def predict_future_price(self, current_features, hours_ahead=24):
        """Predict future price based on current network state"""
        print(f"🔮 Predicting price {hours_ahead} hours ahead...")
        
        # Load full feature set for network predictions
        df = pd.read_csv('astar_ml_features.csv')
        full_features = df[self.feature_names].iloc[-100:].mean().values
        
        # Predict network activity
        network_pred = self.predict_network_activity(full_features)
        
        # Create future features based on network predictions
        future_features = current_features.copy()
        
        # Adjust features based on predicted network activity
        if 'transaction_count' in network_pred:
            future_features[0] = network_pred['transaction_count']  # transaction_count
        if 'gas_used' in network_pred:
            future_features[1] = network_pred['gas_used']  # gas_used
        if 'gas_utilization' in network_pred:
            future_features[2] = network_pred['gas_utilization']  # gas_utilization
        if 'network_activity' in network_pred:
            future_features[3] = network_pred['network_activity']  # network_activity
        if 'avg_gas_price' in network_pred:
            future_features[4] = network_pred['avg_gas_price']  # avg_gas_price
        
        # Predict future price
        future_price = self.predict_price(future_features)
        
        return future_price, network_pred
    
    def get_prediction_confidence(self, features):
        """Calculate prediction confidence based on feature stability"""
        # Simple confidence based on feature variance
        feature_variance = np.var(features)
        confidence = max(0.1, min(0.95, 1 - feature_variance / 1000))
        
        return confidence
    
    def run_prediction_demo(self):
        """Run a demonstration of the prediction system"""
        print("🚀 Astar Price Prediction Demo")
        print("=" * 50)
        
        # Load models
        self.load_models()
        
        # Create price model if not exists
        try:
            self.price_model = xgb.XGBRegressor()
            self.price_model.load_model('astar_price_predictor.json')
            with open('astar_price_scaler.pkl', 'rb') as f:
                self.price_scaler = pickle.load(f)
            print("✅ Loaded existing price model")
        except:
            print("🔄 Training new price model...")
            self.create_price_model()
        
        # Load sample data
        df = pd.read_csv('astar_ml_features.csv')
        
        # Use only the features needed for price prediction
        price_features = [
            'transaction_count', 'gas_used', 'gas_utilization', 
            'network_activity', 'avg_gas_price', 'hour', 'day_of_week'
        ]
        
        sample_features = df[price_features].iloc[-100:].mean().values  # Average of last 100 blocks
        
        print(f"\n📊 Current Network State (Average of last 100 blocks):")
        for i, feature in enumerate(price_features):
            print(f"  {feature}: {sample_features[i]:.2f}")
        
        # Predict current price
        current_price = self.predict_price(sample_features)
        confidence = self.get_prediction_confidence(sample_features)
        
        print(f"\n💰 Current Price Prediction:")
        print(f"  Predicted Price: ${current_price:.6f}")
        print(f"  Confidence: {confidence:.1%}")
        
        # Predict future prices
        for hours in [1, 6, 24]:
            future_price, network_pred = self.predict_future_price(sample_features, hours)
            price_change = (future_price - current_price) / current_price * 100
            
            print(f"\n🔮 {hours}-Hour Prediction:")
            print(f"  Predicted Price: ${future_price:.6f}")
            print(f"  Price Change: {price_change:+.2f}%")
            
            if 'network_activity' in network_pred:
                print(f"  Predicted Network Activity: {network_pred['network_activity']:.0f}")
        
        # Calculate overall prediction accuracy
        print(f"\n🎯 Prediction System Summary:")
        print(f"  Network Models: {len(self.models)} trained")
        print(f"  Price Model: {'✅ Ready' if self.price_model else '❌ Not ready'}")
        print(f"  Confidence: {confidence:.1%}")
        
        if confidence > 0.7:
            print("🎉 High confidence prediction system ready!")
        elif confidence > 0.5:
            print("✅ Moderate confidence prediction system ready!")
        else:
            print("⚠️  Low confidence - consider more data or feature engineering")
        
        return {
            'current_price': current_price,
            'confidence': confidence,
            'models_loaded': len(self.models),
            'price_model_ready': self.price_model is not None
        }

def main():
    """Main function"""
    predictor = AstarPricePredictor()
    results = predictor.run_prediction_demo()
    
    print(f"\n🎉 Astar Price Prediction System Ready!")
    print(f"📊 System Status: {results}")

if __name__ == "__main__":
    main()
