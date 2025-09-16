#!/usr/bin/env python3
"""
Polkadot Parachain Fraud Detection System
=========================================

Specialized latent space for detecting scam/fraudulent activity in Polkadot parachains.
This system identifies suspicious patterns that indicate potential fraud:

Fraud Indicators:
- Unusual transaction patterns
- Suspicious token movements
- Abnormal contract interactions
- Pump and dump schemes
- Rug pull indicators
- Wash trading patterns
- Flash loan attacks
- MEV exploitation
- Cross-chain bridge exploits
- Governance manipulation

Features:
- Real-time fraud detection
- Historical pattern analysis
- Risk scoring system
- Alert generation
- Multi-parachain monitoring
"""

import asyncio
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports for fraud detection
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments
)

@dataclass
class FraudDetectionConfig:
    """Configuration for fraud detection system"""
    # Model configuration
    base_model: str = "microsoft/DialoGPT-medium"
    latent_dim: int = 64
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    
    # Fraud detection thresholds
    fraud_threshold: float = 0.7
    suspicious_threshold: float = 0.5
    risk_levels: List[str] = None
    
    # Feature engineering
    lookback_window: int = 24  # hours
    min_transactions: int = 10
    max_velocity_threshold: float = 1000.0  # transactions per hour
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        if self.risk_levels is None:
            self.risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

class FraudFeatureExtractor:
    """Extract fraud-specific features from Polkadot data"""
    
    def __init__(self, config: FraudDetectionConfig):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def extract_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features that indicate potential fraud"""
        print("Extracting fraud detection features...")
        
        # Basic fraud indicators
        fraud_features = pd.DataFrame(index=df.index)
        
        # 1. Transaction velocity patterns (pump and dump detection)
        fraud_features['tx_velocity'] = df['extrinsics_count'].rolling(window=6).mean()
        fraud_features['tx_velocity_std'] = df['extrinsics_count'].rolling(window=6).std()
        fraud_features['tx_velocity_spike'] = (df['extrinsics_count'] > fraud_features['tx_velocity'] + 3 * fraud_features['tx_velocity_std']).astype(int)
        
        # 2. Block size anomalies (unusual activity)
        fraud_features['block_size_anomaly'] = (df['block_size'] > df['block_size'].quantile(0.95)).astype(int)
        fraud_features['block_size_ratio'] = df['block_size'] / df['block_size'].rolling(window=24).mean()
        
        # 3. Event pattern analysis (suspicious contract interactions)
        fraud_features['event_velocity'] = df['events_count'].rolling(window=6).mean()
        fraud_features['event_spike'] = (df['events_count'] > fraud_features['event_velocity'] + 2 * df['events_count'].rolling(window=6).std()).astype(int)
        
        # 4. Cross-chain activity patterns (bridge exploits)
        fraud_features['cross_chain_anomaly'] = (df['cross_chain_messages'] > df['cross_chain_messages'].quantile(0.9)).astype(int)
        fraud_features['cross_chain_ratio'] = df['cross_chain_messages'] / (df['cross_chain_messages'].rolling(window=24).mean() + 1)
        
        # 5. Parachain activity patterns (rug pull indicators)
        fraud_features['parachain_activity_spike'] = (df['parachain_blocks'] > df['parachain_blocks'].rolling(window=12).mean() * 2).astype(int)
        
        # 6. Temporal patterns (wash trading, MEV)
        fraud_features['hour'] = df['timestamp'].dt.hour
        fraud_features['day_of_week'] = df['timestamp'].dt.dayofweek
        fraud_features['is_weekend'] = (fraud_features['day_of_week'] >= 5).astype(int)
        fraud_features['is_night'] = ((fraud_features['hour'] >= 22) | (fraud_features['hour'] <= 6)).astype(int)
        
        # 7. Block interval anomalies (network manipulation)
        fraud_features['block_interval'] = df['block_number'].diff()
        fraud_features['block_interval_anomaly'] = (fraud_features['block_interval'] > fraud_features['block_interval'].quantile(0.95)).astype(int)
        
        # 8. Efficiency patterns (flash loan attacks)
        fraud_features['efficiency'] = df['extrinsics_count'] / (df['block_size'] / 1000 + 1)
        fraud_features['efficiency_anomaly'] = (fraud_features['efficiency'] > fraud_features['efficiency'].quantile(0.9)).astype(int)
        
        # 9. Network health indicators (governance manipulation)
        fraud_features['validator_anomaly'] = (df['validator_count'] < df['validator_count'].quantile(0.1)).astype(int)
        fraud_features['finalization_delay'] = df['finalization_time']
        fraud_features['finalization_anomaly'] = (fraud_features['finalization_delay'] > fraud_features['finalization_delay'].quantile(0.95)).astype(int)
        
        # 10. Composite fraud scores
        fraud_features['activity_score'] = (
            fraud_features['tx_velocity_spike'] * 0.3 +
            fraud_features['event_spike'] * 0.2 +
            fraud_features['cross_chain_anomaly'] * 0.2 +
            fraud_features['parachain_activity_spike'] * 0.15 +
            fraud_features['efficiency_anomaly'] * 0.15
        )
        
        fraud_features['temporal_score'] = (
            fraud_features['is_weekend'] * 0.4 +
            fraud_features['is_night'] * 0.3 +
            fraud_features['block_interval_anomaly'] * 0.3
        )
        
        fraud_features['network_score'] = (
            fraud_features['validator_anomaly'] * 0.5 +
            fraud_features['finalization_anomaly'] * 0.5
        )
        
        # Overall fraud risk score
        fraud_features['fraud_risk_score'] = (
            fraud_features['activity_score'] * 0.5 +
            fraud_features['temporal_score'] * 0.3 +
            fraud_features['network_score'] * 0.2
        )
        
        # Fill NaN values
        fraud_features = fraud_features.fillna(0)
        
        print(f"Extracted {len(fraud_features.columns)} fraud detection features")
        return fraud_features
    
    def create_fraud_labels(self, fraud_features: pd.DataFrame) -> np.ndarray:
        """Create fraud labels based on risk scores"""
        # Define fraud thresholds
        high_risk = fraud_features['fraud_risk_score'] > 0.7
        medium_risk = (fraud_features['fraud_risk_score'] > 0.4) & (fraud_features['fraud_risk_score'] <= 0.7)
        low_risk = fraud_features['fraud_risk_score'] <= 0.4
        
        # Create labels: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk
        labels = np.zeros(len(fraud_features))
        labels[medium_risk] = 1
        labels[high_risk] = 2
        
        return labels.astype(int)

class FraudDetectionModel(nn.Module):
    """Neural network for fraud detection"""
    
    def __init__(self, input_dim: int, config: FraudDetectionConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fraud classification head
        self.fraud_classifier = nn.Sequential(
            nn.Linear(config.hidden_dims[2], 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # 3 classes: Low, Medium, High risk
        )
        
        # Risk score regression head
        self.risk_regressor = nn.Sequential(
            nn.Linear(config.hidden_dims[2], 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        fraud_logits = self.fraud_classifier(features)
        risk_score = self.risk_regressor(features)
        return fraud_logits, risk_score

class FraudDataset(Dataset):
    """Dataset for fraud detection"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class PolkadotFraudDetector:
    """Main fraud detection system"""
    
    def __init__(self, config: FraudDetectionConfig):
        self.config = config
        self.feature_extractor = FraudFeatureExtractor(config)
        self.model = None
        self.scaler = RobustScaler()
        self.fraud_thresholds = {
            'LOW': 0.0,
            'MEDIUM': 0.4,
            'HIGH': 0.7,
            'CRITICAL': 0.9
        }
        
    def load_polkadot_data(self) -> pd.DataFrame:
        """Load Polkadot data from database"""
        conn = sqlite3.connect("polkadot_archive_data.db")
        
        query = """
        SELECT 
            block_number,
            timestamp,
            extrinsics_count,
            events_count,
            block_size,
            validator_count,
            finalization_time,
            parachain_blocks,
            cross_chain_messages
        FROM block_metrics 
        ORDER BY block_number
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        return df
    
    def prepare_fraud_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for fraud detection"""
        print("Preparing fraud detection data...")
        
        # Load data
        df = self.load_polkadot_data()
        print(f"Loaded {len(df)} block records")
        
        # Extract fraud features
        fraud_features = self.feature_extractor.extract_fraud_features(df)
        
        # Create fraud labels
        fraud_labels = self.feature_extractor.create_fraud_labels(fraud_features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(fraud_features.values)
        
        print(f"Created fraud detection dataset: {features_scaled.shape}")
        print(f"Fraud distribution: {np.bincount(fraud_labels)}")
        
        return features_scaled, fraud_labels
    
    def train_fraud_detector(self, features: np.ndarray, labels: np.ndarray):
        """Train the fraud detection model"""
        print("Training fraud detection model...")
        
        # Create dataset
        dataset = FraudDataset(features, labels)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        self.model = FraudDetectionModel(features.shape[1], self.config).to(self.config.device)
        
        # Optimizer and loss functions
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        classification_loss = nn.CrossEntropyLoss()
        regression_loss = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)
                
                optimizer.zero_grad()
                
                fraud_logits, risk_scores = self.model(batch_features)
                
                # Combined loss
                cls_loss = classification_loss(fraud_logits, batch_labels)
                reg_loss = regression_loss(risk_scores.squeeze(), batch_labels.float() / 2.0)  # Normalize to 0-1
                loss = cls_loss + 0.5 * reg_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.config.device)
                    batch_labels = batch_labels.to(self.config.device)
                    
                    fraud_logits, risk_scores = self.model(batch_features)
                    
                    cls_loss = classification_loss(fraud_logits, batch_labels)
                    reg_loss = regression_loss(risk_scores.squeeze(), batch_labels.float() / 2.0)
                    loss = cls_loss + 0.5 * reg_loss
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Fraud detection model training completed!")
        return train_losses, val_losses
    
    def detect_fraud(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect fraud in given features"""
        if self.model is None:
            raise ValueError("Model not trained. Run train_fraud_detector first.")
        
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.config.device)
            fraud_logits, risk_scores = self.model(features_tensor)
            
            # Get predictions
            fraud_predictions = torch.argmax(fraud_logits, dim=1).cpu().numpy()
            risk_scores = risk_scores.squeeze().cpu().numpy()
            
            # Determine risk levels
            risk_levels = []
            for score in risk_scores:
                if score >= self.fraud_thresholds['CRITICAL']:
                    risk_levels.append('CRITICAL')
                elif score >= self.fraud_thresholds['HIGH']:
                    risk_levels.append('HIGH')
                elif score >= self.fraud_thresholds['MEDIUM']:
                    risk_levels.append('MEDIUM')
                else:
                    risk_levels.append('LOW')
            
            return {
                'fraud_predictions': fraud_predictions,
                'risk_scores': risk_scores,
                'risk_levels': risk_levels,
                'fraud_count': np.sum(fraud_predictions >= 1),
                'high_risk_count': np.sum(np.array(risk_levels) == 'HIGH') + np.sum(np.array(risk_levels) == 'CRITICAL')
            }
    
    def generate_fraud_report(self, features: np.ndarray, timestamps: List[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive fraud detection report"""
        print("Generating fraud detection report...")
        
        fraud_results = self.detect_fraud(features)
        
        # Create detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(features),
            'fraud_detection_summary': {
                'total_fraud_cases': int(fraud_results['fraud_count']),
                'high_risk_cases': int(fraud_results['high_risk_count']),
                'fraud_percentage': float(fraud_results['fraud_count'] / len(features) * 100),
                'high_risk_percentage': float(fraud_results['high_risk_count'] / len(features) * 100)
            },
            'risk_distribution': {
                'LOW': int(np.sum(np.array(fraud_results['risk_levels']) == 'LOW')),
                'MEDIUM': int(np.sum(np.array(fraud_results['risk_levels']) == 'MEDIUM')),
                'HIGH': int(np.sum(np.array(fraud_results['risk_levels']) == 'HIGH')),
                'CRITICAL': int(np.sum(np.array(fraud_results['risk_levels']) == 'CRITICAL'))
            },
            'average_risk_score': float(np.mean(fraud_results['risk_scores'])),
            'max_risk_score': float(np.max(fraud_results['risk_scores'])),
            'fraud_patterns': self._analyze_fraud_patterns(features, fraud_results)
        }
        
        # Save report
        with open('polkadot_fraud_detection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Fraud detection report saved to polkadot_fraud_detection_report.json")
        return report
    
    def _analyze_fraud_patterns(self, features: np.ndarray, fraud_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific fraud patterns"""
        high_risk_indices = np.where(np.array(fraud_results['risk_levels']) == 'HIGH')[0]
        critical_indices = np.where(np.array(fraud_results['risk_levels']) == 'CRITICAL')[0]
        
        if len(high_risk_indices) == 0 and len(critical_indices) == 0:
            return {'no_high_risk_patterns': True}
        
        # Analyze patterns in high-risk samples
        suspicious_samples = np.concatenate([high_risk_indices, critical_indices])
        suspicious_features = features[suspicious_samples]
        
        # Calculate feature importance for fraud detection
        feature_importance = np.mean(suspicious_features, axis=0) - np.mean(features, axis=0)
        
        return {
            'suspicious_samples_count': len(suspicious_samples),
            'top_fraud_indicators': feature_importance.argsort()[-5:].tolist(),
            'feature_importance_scores': feature_importance.tolist()
        }
    
    def visualize_fraud_detection(self, features: np.ndarray, fraud_results: Dict[str, Any]):
        """Create fraud detection visualizations"""
        print("Creating fraud detection visualizations...")
        
        # Create risk score distribution
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Distribution', 'Risk Level Distribution', 
                          'Fraud Detection Timeline', 'Feature Importance'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Risk score histogram
        fig.add_trace(
            go.Histogram(x=fraud_results['risk_scores'], nbinsx=50, name='Risk Scores'),
            row=1, col=1
        )
        
        # Risk level pie chart
        risk_counts = {level: fraud_results['risk_levels'].count(level) for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
        fig.add_trace(
            go.Pie(labels=list(risk_counts.keys()), values=list(risk_counts.values()), name='Risk Levels'),
            row=1, col=2
        )
        
        # Timeline scatter plot
        fig.add_trace(
            go.Scatter(
                y=fraud_results['risk_scores'],
                mode='markers',
                marker=dict(
                    color=fraud_results['risk_scores'],
                    colorscale='RdYlBu_r',
                    size=4
                ),
                name='Risk Timeline'
            ),
            row=2, col=1
        )
        
        # Feature importance (if available)
        if 'fraud_patterns' in fraud_results and 'feature_importance_scores' in fraud_results['fraud_patterns']:
            importance = fraud_results['fraud_patterns']['feature_importance_scores']
            top_features = np.argsort(importance)[-10:]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Feature_{i}' for i in top_features],
                    y=[importance[i] for i in top_features],
                    name='Feature Importance'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Polkadot Parachain Fraud Detection Analysis",
            height=800,
            showlegend=False
        )
        
        fig.write_html("polkadot_fraud_detection_analysis.html")
        print("Fraud detection visualization saved to polkadot_fraud_detection_analysis.html")

async def main():
    """Main function for fraud detection"""
    print("🚨 Starting Polkadot Parachain Fraud Detection System")
    
    # Configuration optimized for fraud detection
    config = FraudDetectionConfig(
        base_model="microsoft/DialoGPT-medium",
        latent_dim=64,
        hidden_dims=[128, 64, 32],
        learning_rate=1e-4,
        batch_size=64,
        epochs=100,
        fraud_threshold=0.7,
        suspicious_threshold=0.5
    )
    
    print(f"Configuration:")
    print(f"- Latent dimensions: {config.latent_dim}")
    print(f"- Hidden layers: {config.hidden_dims}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Fraud threshold: {config.fraud_threshold}")
    print(f"- Device: {config.device}")
    
    # Initialize fraud detector
    detector = PolkadotFraudDetector(config)
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Prepare fraud detection data
        features, labels = detector.prepare_fraud_data()
        
        # Train fraud detection model
        train_losses, val_losses = detector.train_fraud_detector(features, labels)
        
        # Generate fraud detection report
        fraud_report = detector.generate_fraud_report(features)
        
        # Create visualizations
        fraud_results = detector.detect_fraud(features)
        detector.visualize_fraud_detection(features, fraud_results)
        
        print("🎉 Fraud detection system completed!")
        print(f"Total samples analyzed: {fraud_report['total_samples']}")
        print(f"Fraud cases detected: {fraud_report['fraud_detection_summary']['total_fraud_cases']}")
        print(f"High-risk cases: {fraud_report['fraud_detection_summary']['high_risk_cases']}")
        print(f"Fraud percentage: {fraud_report['fraud_detection_summary']['fraud_percentage']:.2f}%")
        print(f"Average risk score: {fraud_report['average_risk_score']:.3f}")
        
        # Print risk distribution
        print("\n📊 Risk Distribution:")
        for level, count in fraud_report['risk_distribution'].items():
            print(f"{level}: {count} cases")
        
    except Exception as e:
        print(f"❌ Error in fraud detection: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
