#!/usr/bin/env python3
"""
Astar Network Activity Prediction
Predict network metrics based on historical data
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class AstarNetworkPredictor:
    """Astar network activity prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load processed ML data"""
        print("📊 Loading network data for prediction...")
        
        # Load features
        self.features_df = pd.read_csv('astar_ml_features.csv')
        
        # Load feature names
        with open('astar_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Prepare feature matrix
        X = self.features_df[self.feature_names].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Target variables - predict network activity metrics
        targets = {
            'transaction_count': self.features_df['transaction_count'],
            'gas_used': self.features_df['gas_used'],
            'gas_utilization': self.features_df['gas_utilization'],
            'network_activity': self.features_df['network_activity'],
            'avg_gas_price': self.features_df['avg_gas_price']
        }
        
        print(f"✅ Loaded {len(X):,} samples with {len(self.feature_names)} features")
        
        return X, targets
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data for training"""
        # Use all samples for network prediction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
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
        
        print(f"✅ XGBoost {target_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.2f}")
        
        return results
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, target_name):
        """Train Random Forest model"""
        print(f"🌲 Training Random Forest for {target_name}...")
        
        # Random Forest parameters
        params = {
            'n_estimators': 300,
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
        
        print(f"✅ Random Forest {target_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.2f}")
        
        return results
    
    def train_all_models(self):
        """Train all models for network prediction"""
        print("🚀 Starting Astar Network Prediction Training")
        print("=" * 60)
        
        # Load data
        X, targets = self.load_data()
        
        # Train models for each target
        for target_name, y in targets.items():
            print(f"\n📈 Training models for {target_name}...")
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(X, y)
            
            self.results[target_name] = {}
            self.scalers[target_name] = scaler
            
            # XGBoost
            self.results[target_name]['xgboost'] = self.train_xgboost(
                X_train, X_test, y_train, y_test, target_name
            )
            
            # Random Forest
            self.results[target_name]['random_forest'] = self.train_random_forest(
                X_train, X_test, y_train, y_test, target_name
            )
        
        # Save models and results
        self.save_models()
        self.generate_report()
        
        return self.results
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving trained models...")
        
        # Save XGBoost models
        for target_name in self.results.keys():
            self.results[target_name]['xgboost']['model'].save_model(f'astar_xgboost_{target_name}.json')
        
        # Save scalers
        for target_name, scaler in self.scalers.items():
            with open(f'astar_scaler_{target_name}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save feature names
        with open('astar_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        print("✅ Models saved successfully!")
    
    def generate_report(self):
        """Generate training report"""
        print("\n📊 Generating Network Prediction Report...")
        
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
        for target_name in self.results.keys():
            report["model_performance"][target_name] = {}
            for model_name in ['xgboost', 'random_forest']:
                if model_name in self.results[target_name]:
                    results = self.results[target_name][model_name]
                    report["model_performance"][target_name][model_name] = {
                        "r2_score": float(results['r2']),
                        "rmse": float(results['rmse']),
                        "mae": float(results['mae']),
                        "mse": float(results['mse'])
                    }
        
        # Best models
        best_models = {}
        for target_name in self.results.keys():
            best_r2 = -1
            best_model = None
            for model_name, results in self.results[target_name].items():
                if results['r2'] > best_r2:
                    best_r2 = results['r2']
                    best_model = model_name
            best_models[target_name] = {
                "model": best_model,
                "r2_score": float(best_r2)
            }
        
        report["best_models"] = best_models
        
        # Save report
        with open('astar_network_prediction_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n🎯 Network Prediction Performance Summary:")
        print("=" * 60)
        
        for target_name in self.results.keys():
            print(f"\n📈 {target_name.replace('_', ' ').title()}:")
            for model_name, results in self.results[target_name].items():
                print(f"  {model_name.replace('_', ' ').title()}: R² = {results['r2']:.4f}, RMSE = {results['rmse']:.2f}")
        
        print(f"\n🏆 Best Models:")
        for target_name, info in best_models.items():
            print(f"  {target_name.replace('_', ' ').title()}: {info['model'].replace('_', ' ').title()} (R² = {info['r2_score']:.4f})")
        
        print(f"\n💾 Report saved: astar_network_prediction_report.json")
        
        # Calculate overall accuracy
        avg_r2 = np.mean([info['r2_score'] for info in best_models.values()])
        print(f"\n🎯 Overall Average R² Score: {avg_r2:.4f}")
        
        if avg_r2 > 0.8:
            print("🎉 Excellent prediction accuracy! Models are ready for production.")
        elif avg_r2 > 0.6:
            print("✅ Good prediction accuracy! Models show strong performance.")
        elif avg_r2 > 0.4:
            print("⚠️  Moderate prediction accuracy. Consider more data or feature engineering.")
        else:
            print("❌ Low prediction accuracy. Need more data or different approach.")

def main():
    """Main function"""
    predictor = AstarNetworkPredictor()
    results = predictor.train_all_models()
    
    print(f"\n🎉 Network Prediction Training Complete!")
    print(f"📊 Ready for network activity prediction with trained models")

if __name__ == "__main__":
    main()
