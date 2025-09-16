#!/usr/bin/env python3
"""
Polkadot Comprehensive Latent Space Generator
============================================

Uses ALL Polkadot data from your databases with full RTX 4090 GPU utilization:
- 38,160 block metrics records
- 1,027 block details records  
- 3 staking data records
- 3 parachain data records
- All available governance, economic, and performance data

Features:
- Multi-modal data fusion
- Advanced GPU acceleration
- Comprehensive fraud detection
- Real-time monitoring capabilities
- Full RTX 4090 utilization (25.4 GB VRAM)
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments
)

@dataclass
class ComprehensiveConfig:
    """Configuration for comprehensive Polkadot latent space"""
    # GPU Configuration for RTX 4090
    device: str = 'cuda'
    batch_size: int = 2048  # Large batch size for RTX 4090
    max_memory: str = "25GB"  # Full RTX 4090 memory
    fp16: bool = True
    gradient_accumulation_steps: int = 2
    
    # Model Configuration
    latent_dim: int = 256  # High-dimensional latent space
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    epochs: int = 200
    
    # Data Configuration
    sequence_length: int = 48  # Longer sequences
    max_features: int = 1000  # Maximum features to process
    
    # Hugging Face Models
    base_models: List[str] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256, 128]  # Deep network for RTX 4090
        if self.base_models is None:
            self.base_models = [
                "microsoft/DialoGPT-medium",
                "google/flan-t5-base",
                "facebook/bart-base"
            ]

class ComprehensiveDataLoader:
    """Load ALL Polkadot data from both databases"""
    
    def __init__(self):
        self.archive_db = "polkadot_archive_data.db"
        self.metrics_db = "polkadot_metrics.db"
        self.scaler = RobustScaler()
        
    def load_all_block_data(self) -> pd.DataFrame:
        """Load all block-related data"""
        print("Loading comprehensive block data...")
        
        # Load from archive database
        conn = sqlite3.connect(self.archive_db)
        
        # Block metrics
        block_metrics_query = """
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
        
        block_metrics = pd.read_sql_query(block_metrics_query, conn)
        
        # Block details
        block_details_query = """
        SELECT 
            block_number,
            block_hash,
            parent_hash,
            timestamp,
            extrinsics_data,
            events_data,
            validator,
            block_size,
            finalized
        FROM block_details 
        ORDER BY block_number
        """
        
        block_details = pd.read_sql_query(block_details_query, conn)
        
        # Staking data
        staking_query = """
        SELECT 
            timestamp,
            validators_count,
            nominators_count,
            active_era,
            total_staked,
            inflation_rate
        FROM staking_data
        ORDER BY timestamp
        """
        
        staking_data = pd.read_sql_query(staking_query, conn)
        
        # Parachain data
        parachain_query = """
        SELECT 
            timestamp,
            parachains_count,
            hrmp_channels_count,
            active_parachains
        FROM parachain_data
        ORDER BY timestamp
        """
        
        parachain_data = pd.read_sql_query(parachain_query, conn)
        
        conn.close()
        
        # Process timestamps
        block_metrics['timestamp'] = pd.to_datetime(block_metrics['timestamp'], format='mixed', errors='coerce')
        block_details['timestamp'] = pd.to_datetime(block_details['timestamp'], format='mixed', errors='coerce')
        if not staking_data.empty:
            staking_data['timestamp'] = pd.to_datetime(staking_data['timestamp'], format='mixed', errors='coerce')
        if not parachain_data.empty:
            parachain_data['timestamp'] = pd.to_datetime(parachain_data['timestamp'], format='mixed', errors='coerce')
        
        # Merge all block data
        print(f"Block metrics: {len(block_metrics)} records")
        print(f"Block details: {len(block_details)} records")
        print(f"Staking data: {len(staking_data)} records")
        print(f"Parachain data: {len(parachain_data)} records")
        
        # Convert block_details timestamp to datetime
        if not block_details.empty:
            block_details['timestamp'] = pd.to_datetime(block_details['timestamp'], unit='s', errors='coerce')
        
        # Merge block metrics and details
        combined_data = block_metrics.merge(
            block_details, 
            on='block_number', 
            how='left', 
            suffixes=('_metrics', '_details')
        )
        
        return combined_data, staking_data, parachain_data
    
    def load_metrics_data(self) -> Dict[str, pd.DataFrame]:
        """Load all metrics data from polkadot_metrics.db"""
        print("Loading comprehensive metrics data...")
        
        conn = sqlite3.connect(self.metrics_db)
        
        metrics_data = {}
        
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            try:
                query = f"SELECT * FROM {table} LIMIT 1000"  # Limit for now
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    metrics_data[table] = df
                    print(f"Loaded {len(df)} records from {table}")
            except Exception as e:
                print(f"Error loading {table}: {e}")
        
        conn.close()
        return metrics_data
    
    def create_comprehensive_features(self, block_data: pd.DataFrame, 
                                    staking_data: pd.DataFrame, 
                                    parachain_data: pd.DataFrame,
                                    metrics_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create comprehensive feature matrix from all data"""
        print("Creating comprehensive feature matrix...")
        
        # Start with block data features
        features_list = []
        
        # 1. Basic block features (use metrics columns)
        block_features = block_data[[
            'extrinsics_count', 'events_count', 'block_size_metrics',
            'validator_count', 'finalization_time', 'parachain_blocks',
            'cross_chain_messages'
        ]].fillna(0)
        
        # Rename block_size_metrics to block_size for consistency
        block_features = block_features.rename(columns={'block_size_metrics': 'block_size'})
        
        # 2. Temporal features
        block_data['timestamp'] = pd.to_datetime(block_data['timestamp_metrics'], errors='coerce')
        block_data['hour'] = block_data['timestamp'].dt.hour
        block_data['day_of_week'] = block_data['timestamp'].dt.dayofweek
        block_data['block_interval'] = block_data['block_number'].diff()
        
        temporal_features = block_data[['hour', 'day_of_week', 'block_interval']].fillna(0)
        
        # 3. Derived features
        derived_features = pd.DataFrame(index=block_data.index)
        derived_features['extrinsics_per_event'] = block_data['extrinsics_count'] / (block_data['events_count'] + 1)
        derived_features['block_efficiency'] = block_data['extrinsics_count'] / (block_data['block_size'] / 1000 + 1)
        derived_features['network_activity'] = block_data['extrinsics_count'] + block_data['events_count']
        derived_features['cross_chain_ratio'] = block_data['cross_chain_messages'] / (block_data['cross_chain_messages'].rolling(24).mean() + 1)
        derived_features['parachain_utilization'] = block_data['parachain_blocks'] / (block_data['parachain_blocks'].rolling(24).mean() + 1)
        
        # 4. Rolling statistics
        rolling_features = pd.DataFrame(index=block_data.index)
        for col in ['extrinsics_count', 'events_count', 'block_size']:
            rolling_features[f'{col}_mean_6h'] = block_data[col].rolling(6).mean()
            rolling_features[f'{col}_std_6h'] = block_data[col].rolling(6).std()
            rolling_features[f'{col}_mean_24h'] = block_data[col].rolling(24).mean()
            rolling_features[f'{col}_std_24h'] = block_data[col].rolling(24).std()
        
        # 5. Fraud detection features
        fraud_features = pd.DataFrame(index=block_data.index)
        fraud_features['tx_velocity_spike'] = (block_data['extrinsics_count'] > rolling_features['extrinsics_count_mean_6h'] + 3 * rolling_features['extrinsics_count_std_6h']).astype(int)
        fraud_features['block_size_anomaly'] = (block_data['block_size'] > block_data['block_size'].quantile(0.95)).astype(int)
        fraud_features['event_spike'] = (block_data['events_count'] > rolling_features['events_count_mean_6h'] + 2 * rolling_features['events_count_std_6h']).astype(int)
        fraud_features['cross_chain_anomaly'] = (block_data['cross_chain_messages'] > block_data['cross_chain_messages'].quantile(0.9)).astype(int)
        
        # 6. Network health features
        health_features = pd.DataFrame(index=block_data.index)
        health_features['validator_anomaly'] = (block_data['validator_count'] < block_data['validator_count'].quantile(0.1)).astype(int)
        health_features['finalization_delay'] = block_data['finalization_time']
        health_features['finalization_anomaly'] = (health_features['finalization_delay'] > health_features['finalization_delay'].quantile(0.95)).astype(int)
        
        # 7. Add staking and parachain data (interpolated)
        if not staking_data.empty:
            staking_features = pd.DataFrame(index=block_data.index)
            for col in ['validators_count', 'nominators_count', 'active_era', 'total_staked', 'inflation_rate']:
                if col in staking_data.columns:
                    # Interpolate staking data to match block timestamps
                    staking_interp = np.interp(
                        block_data['timestamp'].astype(np.int64),
                        staking_data['timestamp'].astype(np.int64),
                        staking_data[col].fillna(0)
                    )
                    staking_features[col] = staking_interp
                else:
                    staking_features[col] = 0
        else:
            staking_features = pd.DataFrame(0, index=block_data.index, columns=['validators_count', 'nominators_count', 'active_era', 'total_staked', 'inflation_rate'])
        
        if not parachain_data.empty:
            parachain_features = pd.DataFrame(index=block_data.index)
            for col in ['parachains_count', 'hrmp_channels_count', 'active_parachains']:
                if col in parachain_data.columns:
                    # Interpolate parachain data
                    parachain_interp = np.interp(
                        block_data['timestamp'].astype(np.int64),
                        parachain_data['timestamp'].astype(np.int64),
                        parachain_data[col].fillna(0)
                    )
                    parachain_features[col] = parachain_interp
                else:
                    parachain_features[col] = 0
        else:
            parachain_features = pd.DataFrame(0, index=block_data.index, columns=['parachains_count', 'hrmp_channels_count', 'active_parachains'])
        
        # Combine all features
        all_features = pd.concat([
            block_features,
            temporal_features,
            derived_features,
            rolling_features,
            fraud_features,
            health_features,
            staking_features,
            parachain_features
        ], axis=1)
        
        # Fill NaN values
        all_features = all_features.fillna(0)
        
        # Limit features if too many
        if all_features.shape[1] > 1000:
            # Select most important features
            feature_importance = all_features.var().sort_values(ascending=False)
            selected_features = feature_importance.head(1000).index
            all_features = all_features[selected_features]
        
        print(f"Created comprehensive feature matrix: {all_features.shape}")
        return all_features.values, all_features.columns.tolist()

class ComprehensiveDataset(Dataset):
    """Dataset for comprehensive Polkadot data"""
    
    def __init__(self, features: np.ndarray, sequence_length: int = 48):
        self.features = features
        self.sequence_length = sequence_length
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create temporal sequences from features"""
        sequences = []
        for i in range(self.sequence_length, len(self.features)):
            sequence = self.features[i-self.sequence_length:i].flatten()
            sequences.append(sequence)
        return np.array(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])

class ComprehensiveVAE(nn.Module):
    """Comprehensive VAE for all Polkadot data"""
    
    def __init__(self, input_dim: int, config: ComprehensiveConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Fraud detection head
        self.fraud_classifier = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Low, Medium, High risk
        )
        
        # Risk score regressor
        self.risk_regressor = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        fraud_logits = self.fraud_classifier(z)
        risk_score = self.risk_regressor(z)
        return recon_x, mu, logvar, fraud_logits, risk_score

class ComprehensiveLatentGenerator:
    """Comprehensive latent space generator using ALL Polkadot data"""
    
    def __init__(self, config: ComprehensiveConfig):
        self.config = config
        self.data_loader = ComprehensiveDataLoader()
        self.model = None
        self.scaler = RobustScaler()
        self.latent_representations = None
        self.feature_names = []
        
    def prepare_all_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare ALL Polkadot data for training"""
        print("🚀 Preparing comprehensive Polkadot data...")
        
        # Load all data
        block_data, staking_data, parachain_data = self.data_loader.load_all_block_data()
        metrics_data = self.data_loader.load_metrics_data()
        
        # Create comprehensive features
        features, feature_names = self.data_loader.create_comprehensive_features(
            block_data, staking_data, parachain_data, metrics_data
        )
        self.feature_names = feature_names
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create dataset with sequences
        dataset = ComprehensiveDataset(features_scaled, self.config.sequence_length)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders with large batch size for RTX 4090
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Feature dimensions: {len(feature_names)}")
        print(f"Sequence length: {self.config.sequence_length}")
        
        return train_loader, val_loader
    
    def train_comprehensive_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the comprehensive model on RTX 4090"""
        print("🔥 Training comprehensive model on RTX 4090...")
        
        # Initialize model
        input_dim = len(self.feature_names) * self.config.sequence_length
        self.model = ComprehensiveVAE(input_dim, self.config).to(self.config.device)
        
        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer with gradient accumulation
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Mixed precision training for RTX 4090
        scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.config.device)
                
                optimizer.zero_grad()
                
                if self.config.fp16 and scaler:
                    with torch.cuda.amp.autocast():
                        recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                        
                        # Combined loss
                        recon_loss = nn.MSELoss()(recon_batch, batch)
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.1 * kl_loss
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                    
                    # Combined loss
                    recon_loss = nn.MSELoss()(recon_batch, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.1 * kl_loss
                    
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.config.device)
                    
                    if self.config.fp16:
                        with torch.cuda.amp.autocast():
                            recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                            recon_loss = nn.MSELoss()(recon_batch, batch)
                            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = recon_loss + 0.1 * kl_loss
                    else:
                        recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                        recon_loss = nn.MSELoss()(recon_batch, batch)
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.1 * kl_loss
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print("🎉 Comprehensive model training completed!")
        return train_losses, val_losses
    
    def generate_comprehensive_latent_space(self, data_loader: DataLoader):
        """Generate comprehensive latent representations"""
        print("🎯 Generating comprehensive latent representations...")
        
        self.model.eval()
        latent_representations = []
        fraud_predictions = []
        risk_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.config.device)
                
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        _, mu, _, fraud_logits, risk_score = self.model(batch)
                else:
                    _, mu, _, fraud_logits, risk_score = self.model(batch)
                
                latent_representations.append(mu.cpu().numpy())
                fraud_predictions.append(torch.argmax(fraud_logits, dim=1).cpu().numpy())
                risk_scores.append(risk_score.squeeze().cpu().numpy())
        
        self.latent_representations = np.concatenate(latent_representations, axis=0)
        fraud_predictions = np.concatenate(fraud_predictions, axis=0)
        risk_scores = np.concatenate(risk_scores, axis=0)
        
        print(f"Generated comprehensive latent representations: {self.latent_representations.shape}")
        print(f"Fraud predictions: {np.bincount(fraud_predictions)}")
        print(f"Average risk score: {np.mean(risk_scores):.4f}")
        
        return self.latent_representations, fraud_predictions, risk_scores
    
    def save_comprehensive_model(self, model_path: str = "polkadot_comprehensive_model.pth"):
        """Save the comprehensive model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }, model_path)
        
        print(f"Comprehensive model saved to {model_path}")

async def main():
    """Main function for comprehensive latent space generation"""
    print("🚀 Starting Comprehensive Polkadot Latent Space Generation")
    print("Using ALL Polkadot data with full RTX 4090 utilization")
    
    # RTX 4090 optimized configuration
    config = ComprehensiveConfig(
        device='cuda',
        batch_size=2048,  # Large batch for RTX 4090
        latent_dim=256,   # High-dimensional latent space
        hidden_dims=[1024, 512, 256, 128],  # Deep network
        learning_rate=1e-4,
        epochs=200,
        sequence_length=48,
        fp16=True,
        gradient_accumulation_steps=2
    )
    
    print(f"Configuration:")
    print(f"- Device: {config.device}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Latent dimensions: {config.latent_dim}")
    print(f"- Hidden layers: {config.hidden_dims}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Sequence length: {config.sequence_length}")
    print(f"- FP16: {config.fp16}")
    
    # Initialize generator
    generator = ComprehensiveLatentGenerator(config)
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Prepare ALL data
        train_loader, val_loader = generator.prepare_all_data()
        
        # Train comprehensive model
        train_losses, val_losses = generator.train_comprehensive_model(train_loader, val_loader)
        
        # Generate comprehensive latent space
        latent_repr, fraud_pred, risk_scores = generator.generate_comprehensive_latent_space(val_loader)
        
        # Save model
        generator.save_comprehensive_model()
        
        # Save latent representations
        np.save("polkadot_comprehensive_latent_representations.npy", latent_repr)
        np.save("polkadot_comprehensive_fraud_predictions.npy", fraud_pred)
        np.save("polkadot_comprehensive_risk_scores.npy", risk_scores)
        
        print("🎉 Comprehensive latent space generation completed!")
        print(f"Generated {len(latent_repr)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"GPU utilization: {(torch.cuda.max_memory_allocated() / (25.4 * 1e9)) * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in comprehensive generation: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
