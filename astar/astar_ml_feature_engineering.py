#!/usr/bin/env python3
"""
Astar ML Feature Engineering Pipeline
Extract and engineer features from collected Astar data for price prediction
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AstarFeatureEngineer:
    """Feature engineering for Astar price prediction"""
    
    def __init__(self, db_path="astar_multithreaded_data.db"):
        self.db_path = db_path
        self.features_df = None
        self.scaler = StandardScaler()
        
    def load_raw_data(self):
        """Load raw data from database"""
        print("📊 Loading raw data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Load blocks data
            blocks_query = """
            SELECT 
                block_number,
                timestamp,
                gas_limit,
                gas_used,
                transaction_count,
                block_size,
                difficulty
            FROM astar_blocks
            ORDER BY block_number
            """
            
            blocks_df = pd.read_sql_query(blocks_query, conn)
            blocks_df['timestamp'] = pd.to_datetime(blocks_df['timestamp'])
            
            # Load transactions data
            tx_query = """
            SELECT 
                block_number,
                gas,
                gas_price,
                value,
                from_address,
                to_address
            FROM astar_transactions
            ORDER BY block_number
            """
            
            tx_df = pd.read_sql_query(tx_query, conn)
            
            # Load metrics data if available
            try:
                metrics_query = """
                SELECT 
                    timestamp,
                    block_number,
                    price_usd,
                    market_cap_usd,
                    volume_24h_usd,
                    gas_price_avg,
                    network_utilization,
                    active_addresses_24h
                FROM astar_metrics
                ORDER BY timestamp
                """
                metrics_df = pd.read_sql_query(metrics_query, conn)
                metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            except:
                print("⚠️  No metrics data found, will use synthetic price data")
                metrics_df = None
            
            print(f"✅ Loaded {len(blocks_df):,} blocks and {len(tx_df):,} transactions")
            
            return blocks_df, tx_df, metrics_df
            
        finally:
            conn.close()
    
    def create_synthetic_price_data(self, blocks_df):
        """Create synthetic price data based on network activity"""
        print("💰 Creating synthetic price data based on network activity...")
        
        # Calculate network activity metrics
        blocks_df['network_activity'] = blocks_df['transaction_count'] * blocks_df['gas_used']
        blocks_df['gas_utilization'] = blocks_df['gas_used'] / blocks_df['gas_limit']
        
        # Create price based on network activity (synthetic)
        base_price = 0.15  # Base Astar price
        activity_factor = blocks_df['network_activity'] / blocks_df['network_activity'].mean()
        gas_factor = blocks_df['gas_utilization'] / blocks_df['gas_utilization'].mean()
        
        # Synthetic price with some volatility
        np.random.seed(42)  # For reproducibility
        volatility = np.random.normal(0, 0.05, len(blocks_df))
        
        blocks_df['price_usd'] = base_price * (1 + activity_factor * 0.1 + gas_factor * 0.05 + volatility)
        blocks_df['price_usd'] = np.maximum(blocks_df['price_usd'], 0.01)  # Minimum price
        
        # Create market cap and volume
        total_supply = 7_000_000_000  # Astar total supply
        blocks_df['market_cap_usd'] = blocks_df['price_usd'] * total_supply
        blocks_df['volume_24h_usd'] = blocks_df['market_cap_usd'] * np.random.uniform(0.01, 0.1, len(blocks_df))
        
        return blocks_df
    
    def engineer_features(self, blocks_df, tx_df, metrics_df=None):
        """Engineer features for ML models"""
        print("🔧 Engineering ML features...")
        
        # Start with blocks data
        features_df = blocks_df.copy()
        
        # 1. Basic Network Features
        features_df['gas_utilization'] = features_df['gas_used'] / features_df['gas_limit']
        features_df['block_efficiency'] = features_df['transaction_count'] / features_df['block_size']
        features_df['network_activity'] = features_df['transaction_count'] * features_df['gas_used']
        
        # 2. Time-based Features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # 3. Rolling Window Features (7-day, 24-hour windows)
        features_df = features_df.sort_values('block_number')
        
        # 24-hour rolling features
        features_df['tx_count_24h'] = features_df['transaction_count'].rolling(window=24*60, min_periods=1).mean()
        features_df['gas_used_24h'] = features_df['gas_used'].rolling(window=24*60, min_periods=1).mean()
        features_df['gas_util_24h'] = features_df['gas_utilization'].rolling(window=24*60, min_periods=1).mean()
        
        # 7-day rolling features
        features_df['tx_count_7d'] = features_df['transaction_count'].rolling(window=7*24*60, min_periods=1).mean()
        features_df['gas_used_7d'] = features_df['gas_used'].rolling(window=7*24*60, min_periods=1).mean()
        features_df['network_activity_7d'] = features_df['network_activity'].rolling(window=7*24*60, min_periods=1).mean()
        
        # 4. Transaction Aggregations
        # Convert hex strings to numeric values
        tx_df['gas_numeric'] = pd.to_numeric(tx_df['gas'], errors='coerce')
        tx_df['gas_price_numeric'] = tx_df['gas_price'].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else pd.to_numeric(x, errors='coerce'))
        tx_df['value_numeric'] = tx_df['value'].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else pd.to_numeric(x, errors='coerce'))
        
        tx_agg = tx_df.groupby('block_number').agg({
            'gas_numeric': ['mean', 'std', 'sum'],
            'gas_price_numeric': ['mean', 'std'],
            'value_numeric': ['count', 'sum']
        }).reset_index()
        
        tx_agg.columns = ['block_number', 'avg_tx_gas', 'std_tx_gas', 'total_tx_gas',
                         'avg_gas_price', 'std_gas_price', 'tx_count_verified', 'total_value']
        
        # Merge transaction features
        features_df = features_df.merge(tx_agg, on='block_number', how='left')
        
        # 5. Price and Market Features
        if metrics_df is not None:
            # Use real price data if available
            price_features = metrics_df[['block_number', 'price_usd', 'market_cap_usd', 'volume_24h_usd']]
            features_df = features_df.merge(price_features, on='block_number', how='left')
        else:
            # Use synthetic price data
            features_df = self.create_synthetic_price_data(features_df)
        
        # 6. Price-based Features
        features_df['price_change_1h'] = features_df['price_usd'].pct_change(periods=60)
        features_df['price_change_24h'] = features_df['price_usd'].pct_change(periods=24*60)
        features_df['price_volatility_24h'] = features_df['price_change_1h'].rolling(window=24*60).std()
        
        # 7. Network Health Features
        features_df['network_health'] = (
            features_df['gas_utilization'] * 0.3 +
            features_df['transaction_count'] / features_df['transaction_count'].mean() * 0.3 +
            (1 - features_df['price_volatility_24h'].fillna(0)) * 0.4
        )
        
        # 8. DeFi Activity Features (simplified)
        features_df['defi_activity'] = features_df['transaction_count'] * features_df['avg_gas_price'].fillna(0)
        features_df['contract_activity'] = features_df['tx_count_verified'].fillna(0) / features_df['transaction_count']
        
        # 9. Momentum Features
        features_df['tx_momentum'] = features_df['transaction_count'] - features_df['tx_count_24h']
        features_df['gas_momentum'] = features_df['gas_used'] - features_df['gas_used_24h']
        features_df['activity_momentum'] = features_df['network_activity'] - features_df['network_activity_7d']
        
        # 10. Target Variables (Future Price Predictions)
        features_df['price_1h_future'] = features_df['price_usd'].shift(-60)
        features_df['price_24h_future'] = features_df['price_usd'].shift(-24*60)
        features_df['price_change_1h_future'] = features_df['price_1h_future'] / features_df['price_usd'] - 1
        features_df['price_change_24h_future'] = features_df['price_24h_future'] / features_df['price_usd'] - 1
        
        # Fill NaN values instead of dropping
        features_df = features_df.fillna(0)
        
        # Convert all columns to numeric where possible
        for col in features_df.columns:
            if col not in ['timestamp', 'block_number']:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        print(f"✅ Engineered {len(features_df):,} feature rows")
        print(f"📊 Features: {len(features_df.columns)} columns")
        
        return features_df
    
    def select_features(self, features_df):
        """Select and prepare features for ML models"""
        print("🎯 Selecting features for ML models...")
        
        # Define feature columns
        feature_cols = [
            # Network Activity
            'transaction_count', 'gas_used', 'gas_limit', 'gas_utilization',
            'network_activity', 'block_efficiency',
            
            # Time Features
            'hour', 'day_of_week', 'is_weekend',
            
            # Rolling Features
            'tx_count_24h', 'gas_used_24h', 'gas_util_24h',
            'tx_count_7d', 'gas_used_7d', 'network_activity_7d',
            
            # Transaction Features
            'avg_tx_gas', 'std_tx_gas', 'total_tx_gas',
            'avg_gas_price', 'std_gas_price', 'total_value',
            
            # Price Features
            'price_usd', 'market_cap_usd', 'volume_24h_usd',
            'price_change_1h', 'price_change_24h', 'price_volatility_24h',
            
            # Network Health
            'network_health', 'defi_activity', 'contract_activity',
            
            # Momentum
            'tx_momentum', 'gas_momentum', 'activity_momentum'
        ]
        
        # Select available features
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        # Prepare feature matrix
        X = features_df[available_features].fillna(0)
        
        # Convert to numeric and handle infinite values
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Target variables
        y_1h = pd.to_numeric(features_df['price_change_1h_future'], errors='coerce').fillna(0)
        y_24h = pd.to_numeric(features_df['price_change_24h_future'], errors='coerce').fillna(0)
        
        print(f"✅ Selected {len(available_features)} features")
        print(f"📈 Features: {available_features}")
        
        return X, y_1h, y_24h, available_features
    
    def prepare_ml_data(self):
        """Complete ML data preparation pipeline"""
        print("🚀 Starting Astar ML Feature Engineering Pipeline")
        print("=" * 60)
        
        # Load raw data
        blocks_df, tx_df, metrics_df = self.load_raw_data()
        
        # Engineer features
        features_df = self.engineer_features(blocks_df, tx_df, metrics_df)
        
        # Select features
        X, y_1h, y_24h, feature_names = self.select_features(features_df)
        
        # Save processed data
        features_df.to_csv('astar_ml_features.csv', index=False)
        
        # Save feature names
        with open('astar_feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(X),
            "total_features": len(feature_names),
            "feature_names": feature_names,
            "target_1h_samples": len(y_1h[y_1h != 0]),
            "target_24h_samples": len(y_24h[y_24h != 0]),
            "data_quality": {
                "missing_values": int(X.isnull().sum().sum()),
                "infinite_values": int(np.isinf(X.select_dtypes(include=[np.number])).sum().sum()),
                "feature_ranges": X.describe().to_dict()
            }
        }
        
        with open('astar_ml_data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ ML Data Preparation Complete!")
        print(f"📊 Samples: {len(X):,}")
        print(f"🎯 Features: {len(feature_names)}")
        print(f"💾 Saved: astar_ml_features.csv")
        print(f"📋 Summary: astar_ml_data_summary.json")
        
        return X, y_1h, y_24h, feature_names

def main():
    """Main function"""
    engineer = AstarFeatureEngineer()
    X, y_1h, y_24h, feature_names = engineer.prepare_ml_data()
    
    print(f"\n🎉 Ready for ML Training!")
    print(f"📈 Next: Train XGBoost and Neural Network models")

if __name__ == "__main__":
    main()
