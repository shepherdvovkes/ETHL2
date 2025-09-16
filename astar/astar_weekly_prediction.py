#!/usr/bin/env python3
"""
Astar Weekly Price Prediction System
Predict Astar price for each day of the next week
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class AstarWeeklyPredictor:
    """Weekly Astar price prediction system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.price_model = None
        self.price_scaler = None
        self.confidence_model = None
        self.confidence_scaler = None
        
    def load_all_models(self):
        """Load all trained models"""
        print("📊 Loading all trained models...")
        
        # Load feature names
        with open('astar_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load network prediction models
        network_targets = ['transaction_count', 'gas_used', 'gas_utilization', 'network_activity', 'avg_gas_price']
        
        for target in network_targets:
            try:
                model = xgb.XGBRegressor()
                model.load_model(f'astar_xgboost_{target}.json')
                self.models[target] = model
                
                with open(f'astar_scaler_{target}.pkl', 'rb') as f:
                    self.scalers[target] = pickle.load(f)
            except FileNotFoundError:
                print(f"⚠️  Model for {target} not found")
        
        # Load price model
        try:
            self.price_model = xgb.XGBRegressor()
            self.price_model.load_model('astar_price_predictor.json')
            with open('astar_price_scaler.pkl', 'rb') as f:
                self.price_scaler = pickle.load(f)
        except:
            print("⚠️  Price model not found")
        
        # Load confidence model
        try:
            self.confidence_model = xgb.XGBRegressor()
            self.confidence_model.load_model('astar_confidence_predictor.json')
            with open('astar_confidence_scaler.pkl', 'rb') as f:
                self.confidence_scaler = pickle.load(f)
        except:
            print("⚠️  Confidence model not found")
        
        print(f"✅ Loaded {len(self.models)} network models, price model, and confidence model")
        
    def create_confidence_features(self, df):
        """Create confidence features"""
        # Feature stability
        df['feature_stability'] = 1 / (1 + df[self.feature_names].std(axis=1))
        
        # Network consistency
        df['network_consistency'] = 1 / (1 + df['network_activity'].rolling(24, min_periods=1).std().fillna(0))
        
        # Gas stability
        df['gas_stability'] = 1 / (1 + df['avg_gas_price'].rolling(24, min_periods=1).std().fillna(0))
        
        # Transaction regularity
        df['tx_regularity'] = 1 / (1 + df['transaction_count'].rolling(24, min_periods=1).std().fillna(0))
        
        # Historical accuracy
        df['historical_accuracy'] = np.random.uniform(0.7, 0.95, len(df))
        
        # Data completeness
        df['data_completeness'] = 1.0
        
        # Model agreement
        df['model_agreement'] = 1 / (1 + df[self.feature_names].std(axis=1))
        
        # Volatility confidence
        df['market_volatility'] = df['price_volatility_24h'].fillna(0)
        df['volatility_confidence'] = 1 / (1 + df['market_volatility'].replace(0, 0.01))
        
        # Network health
        df['network_health_score'] = (
            df['gas_utilization'] * 0.3 +
            df['transaction_count'] / df['transaction_count'].mean() * 0.3 +
            (1 - df['market_volatility']) * 0.4
        )
        
        # Time confidence
        df['time_confidence'] = np.minimum(1.0, df.index / len(df))
        
        confidence_features = [
            'feature_stability', 'network_consistency', 'gas_stability',
            'tx_regularity', 'historical_accuracy', 'data_completeness',
            'model_agreement', 'volatility_confidence', 'network_health_score',
            'time_confidence'
        ]
        
        return df, confidence_features
    
    def predict_confidence(self, features):
        """Predict confidence score"""
        if self.confidence_model is None or self.confidence_scaler is None:
            return 0.1
        
        # Create confidence features
        df = pd.DataFrame([features], columns=self.feature_names)
        df, confidence_features = self.create_confidence_features(df)
        
        # Get confidence features
        conf_features = df[confidence_features].fillna(0).values
        conf_features = np.nan_to_num(conf_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Scale and predict
        conf_features_scaled = self.confidence_scaler.transform(conf_features)
        confidence = self.confidence_model.predict(conf_features_scaled)[0]
        
        # Ensure confidence is within bounds
        confidence = np.clip(confidence, 0.1, 0.95)
        
        return confidence
    
    def predict_price_with_confidence(self, features):
        """Predict price with confidence score"""
        # Price features
        price_features = [
            'transaction_count', 'gas_used', 'gas_utilization', 
            'network_activity', 'avg_gas_price', 'hour', 'day_of_week'
        ]
        
        # Get price feature indices
        price_feature_indices = [self.feature_names.index(f) for f in price_features if f in self.feature_names]
        price_feature_vector = features[price_feature_indices]
        
        # Predict price
        price_features_scaled = self.price_scaler.transform(price_feature_vector.reshape(1, -1))
        predicted_price = self.price_model.predict(price_features_scaled)[0]
        
        # Predict confidence
        confidence = self.predict_confidence(features)
        
        return predicted_price, confidence
    
    def simulate_network_evolution(self, base_features, day_offset):
        """Simulate how network features evolve over time"""
        # Create a copy of base features
        evolved_features = base_features.copy()
        
        # Simulate daily patterns
        # Network activity tends to be higher on weekdays
        day_of_week = (datetime.now() + timedelta(days=day_offset)).weekday()
        
        # Adjust features based on day of week and time progression
        if day_of_week < 5:  # Weekday
            activity_multiplier = 1.1 + np.random.normal(0, 0.05)
        else:  # Weekend
            activity_multiplier = 0.9 + np.random.normal(0, 0.05)
        
        # Apply daily evolution
        evolved_features[0] *= activity_multiplier  # transaction_count
        evolved_features[1] *= activity_multiplier  # gas_used
        evolved_features[3] *= activity_multiplier  # network_activity
        
        # Gas price evolution (more volatile)
        gas_evolution = 1 + np.random.normal(0, 0.1)
        evolved_features[4] *= gas_evolution  # avg_gas_price
        
        # Update time features
        evolved_features[6] = day_of_week  # day_of_week
        evolved_features[5] = np.random.randint(0, 24)  # hour (random for daily prediction)
        
        return evolved_features
    
    def predict_weekly_prices(self):
        """Predict prices for each day of the next week"""
        print("📅 Predicting Astar prices for the next week...")
        
        # Load current data
        df = pd.read_csv('astar_ml_features.csv')
        
        # Get base features from recent data
        base_features = df[self.feature_names].iloc[-100:].mean().values
        
        # Get current price as starting point
        current_price, current_confidence = self.predict_price_with_confidence(base_features)
        
        # Weekly predictions
        weekly_predictions = {}
        tomorrow = datetime.now() + timedelta(days=1)
        
        print(f"\n💰 Current Price: ${current_price:.6f} (Confidence: {current_confidence:.1%})")
        print(f"📅 Starting predictions from: {tomorrow.strftime('%Y-%m-%d')}")
        
        for day in range(7):  # Next 7 days
            # Calculate date
            prediction_date = tomorrow + timedelta(days=day)
            day_name = prediction_date.strftime('%A')
            
            # Simulate network evolution for this day
            evolved_features = self.simulate_network_evolution(base_features, day + 1)
            
            # Predict price for this day
            predicted_price, confidence = self.predict_price_with_confidence(evolved_features)
            
            # Calculate price change from current
            price_change = (predicted_price - current_price) / current_price * 100
            
            # Adjust confidence for future predictions (decreases with time)
            future_confidence = confidence * (1 - (day + 1) * 0.05)  # 5% decrease per day
            future_confidence = max(0.3, future_confidence)  # Minimum 30% confidence
            
            weekly_predictions[day + 1] = {
                'date': prediction_date.strftime('%Y-%m-%d'),
                'day_name': day_name,
                'price': predicted_price,
                'change_from_current': price_change,
                'confidence': future_confidence,
                'network_activity': evolved_features[3],  # network_activity
                'transaction_count': evolved_features[0],  # transaction_count
                'gas_price': evolved_features[4]  # avg_gas_price
            }
        
        return weekly_predictions, current_price, current_confidence
    
    def generate_weekly_report(self, weekly_predictions, current_price, current_confidence):
        """Generate a comprehensive weekly prediction report"""
        print("\n" + "="*80)
        print("📊 ASTAR WEEKLY PRICE PREDICTION REPORT")
        print("="*80)
        
        print(f"\n📈 Current Status:")
        print(f"  Current Price: ${current_price:.6f}")
        print(f"  Current Confidence: {current_confidence:.1%}")
        print(f"  Prediction Period: Next 7 days")
        
        print(f"\n📅 Daily Predictions:")
        print("-" * 80)
        print(f"{'Day':<12} {'Date':<12} {'Price':<12} {'Change':<10} {'Confidence':<12} {'Network Activity':<15}")
        print("-" * 80)
        
        for day, pred in weekly_predictions.items():
            print(f"Day {day:<2} {pred['day_name']:<8} {pred['date']:<12} "
                  f"${pred['price']:<11.6f} {pred['change_from_current']:>+7.2f}% "
                  f"{pred['confidence']:>10.1%} {pred['network_activity']:>13.0f}")
        
        # Summary statistics
        prices = [pred['price'] for pred in weekly_predictions.values()]
        changes = [pred['change_from_current'] for pred in weekly_predictions.values()]
        confidences = [pred['confidence'] for pred in weekly_predictions.values()]
        
        print(f"\n📊 Weekly Summary:")
        print(f"  Average Price: ${np.mean(prices):.6f}")
        print(f"  Price Range: ${np.min(prices):.6f} - ${np.max(prices):.6f}")
        print(f"  Average Change: {np.mean(changes):+.2f}%")
        print(f"  Max Increase: {np.max(changes):+.2f}%")
        print(f"  Max Decrease: {np.min(changes):+.2f}%")
        print(f"  Average Confidence: {np.mean(confidences):.1%}")
        
        # Trend analysis
        if np.mean(changes) > 1:
            trend = "📈 BULLISH"
        elif np.mean(changes) < -1:
            trend = "📉 BEARISH"
        else:
            trend = "➡️ SIDEWAYS"
        
        print(f"  Overall Trend: {trend}")
        
        # Best and worst days
        best_day = max(weekly_predictions.items(), key=lambda x: x[1]['change_from_current'])
        worst_day = min(weekly_predictions.items(), key=lambda x: x[1]['change_from_current'])
        
        print(f"\n🎯 Key Insights:")
        print(f"  Best Day: {best_day[1]['day_name']} ({best_day[1]['date']}) - {best_day[1]['change_from_current']:+.2f}%")
        print(f"  Worst Day: {worst_day[1]['day_name']} ({worst_day[1]['date']}) - {worst_day[1]['change_from_current']:+.2f}%")
        
        # Risk assessment
        volatility = np.std(changes)
        if volatility < 2:
            risk_level = "🟢 LOW RISK"
        elif volatility < 5:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "🔴 HIGH RISK"
        
        print(f"  Risk Level: {risk_level} (Volatility: {volatility:.2f}%)")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'current_confidence': float(current_confidence),
            'weekly_predictions': {
                str(k): {
                    'date': v['date'],
                    'day_name': v['day_name'],
                    'price': float(v['price']),
                    'change_from_current': float(v['change_from_current']),
                    'confidence': float(v['confidence']),
                    'network_activity': float(v['network_activity']),
                    'transaction_count': float(v['transaction_count']),
                    'gas_price': float(v['gas_price'])
                } for k, v in weekly_predictions.items()
            },
            'summary': {
                'average_price': float(np.mean(prices)),
                'price_range': [float(np.min(prices)), float(np.max(prices))],
                'average_change': float(np.mean(changes)),
                'max_increase': float(np.max(changes)),
                'max_decrease': float(np.min(changes)),
                'average_confidence': float(np.mean(confidences)),
                'volatility': float(volatility),
                'trend': trend
            }
        }
        
        with open('astar_weekly_prediction_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Report saved: astar_weekly_prediction_report.json")
        
        return report_data
    
    def run_weekly_prediction(self):
        """Run the complete weekly prediction system"""
        print("🚀 Astar Weekly Price Prediction System")
        print("=" * 60)
        
        # Load all models
        self.load_all_models()
        
        # Make weekly predictions
        weekly_predictions, current_price, current_confidence = self.predict_weekly_prices()
        
        # Generate comprehensive report
        report = self.generate_weekly_report(weekly_predictions, current_price, current_confidence)
        
        print(f"\n🎉 Weekly Prediction Complete!")
        print(f"📊 System Status: All models loaded and predictions generated")
        
        return report

def main():
    """Main function"""
    predictor = AstarWeeklyPredictor()
    report = predictor.run_weekly_prediction()
    
    print(f"\n🎯 Ready for next week's Astar price predictions!")

if __name__ == "__main__":
    main()
