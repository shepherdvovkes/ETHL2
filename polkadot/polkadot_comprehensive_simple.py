#!/usr/bin/env python3
"""
Polkadot Comprehensive Latent Space - Simplified Version
=======================================================

Uses ALL Polkadot data with full RTX 4090 GPU utilization.
Simplified approach to avoid column conflicts.
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
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SimpleConfig:
    """Simplified configuration for comprehensive Polkadot latent space"""
    # GPU Configuration for RTX 4090
    device: str = 'cuda'
    batch_size: int = 2048  # Large batch size for RTX 4090
    fp16: bool = True
    
    # Model Configuration
    latent_dim: int = 256  # High-dimensional latent space
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    epochs: int = 100
    
    # Data Configuration
    sequence_length: int = 24  # Shorter sequences for stability
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]  # Simpler network

class SimpleDataLoader:
    """Simplified data loader for all Polkadot data"""
    
    def __init__(self):
        self.archive_db = "polkadot_archive_data.db"
        self.metrics_db = "polkadot_metrics.db"
        self.scaler = RobustScaler()
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all available Polkadot data"""
        print("Loading ALL Polkadot data...")
        
        # Load from archive database
        conn = sqlite3.connect(self.archive_db)
        
        # Block metrics (main data)
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
        print(f"Loaded {len(block_metrics)} block metrics records")
        
        # Block details (additional data)
        block_details_query = """
        SELECT 
            block_number,
            block_hash,
            parent_hash,
            timestamp as timestamp_details,
            extrinsics_data,
            events_data,
            validator,
            block_size as block_size_details,
            finalized
        FROM block_details 
        ORDER BY block_number
        """
        
        block_details = pd.read_sql_query(block_details_query, conn)
        print(f"Loaded {len(block_details)} block details records")
        
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
        print(f"Loaded {len(staking_data)} staking records")
        
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
        print(f"Loaded {len(parachain_data)} parachain records")
        
        conn.close()
        
        # Process timestamps
        block_metrics['timestamp'] = pd.to_datetime(block_metrics['timestamp'], format='mixed', errors='coerce')
        if not block_details.empty:
            block_details['timestamp_details'] = pd.to_datetime(block_details['timestamp_details'], unit='s', errors='coerce')
        if not staking_data.empty:
            staking_data['timestamp'] = pd.to_datetime(staking_data['timestamp'], format='mixed', errors='coerce')
        if not parachain_data.empty:
            parachain_data['timestamp'] = pd.to_datetime(parachain_data['timestamp'], format='mixed', errors='coerce')
        
        return block_metrics, block_details, staking_data, parachain_data
    
    def create_comprehensive_features(self, block_metrics: pd.DataFrame, 
                                    block_details: pd.DataFrame,
                                    staking_data: pd.DataFrame, 
                                    parachain_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create comprehensive features from all data"""
        print("Creating comprehensive features from ALL data...")
        
        # Start with block metrics as base
        features_df = block_metrics.copy()
        
        # Add temporal features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['block_interval'] = features_df['block_number'].diff()
        
        # Add derived features
        features_df['extrinsics_per_event'] = features_df['extrinsics_count'] / (features_df['events_count'] + 1)
        features_df['block_efficiency'] = features_df['extrinsics_count'] / (features_df['block_size'] / 1000 + 1)
        features_df['network_activity'] = features_df['extrinsics_count'] + features_df['events_count']
        features_df['cross_chain_ratio'] = features_df['cross_chain_messages'] / (features_df['cross_chain_messages'].rolling(24).mean() + 1)
        features_df['parachain_utilization'] = features_df['parachain_blocks'] / (features_df['parachain_blocks'].rolling(24).mean() + 1)
        
        # Add rolling statistics
        for col in ['extrinsics_count', 'events_count', 'block_size']:
            features_df[f'{col}_mean_6h'] = features_df[col].rolling(6).mean()
            features_df[f'{col}_std_6h'] = features_df[col].rolling(6).std()
            features_df[f'{col}_mean_24h'] = features_df[col].rolling(24).mean()
            features_df[f'{col}_std_24h'] = features_df[col].rolling(24).std()
        
        # Add fraud detection features
        features_df['tx_velocity_spike'] = (features_df['extrinsics_count'] > features_df['extrinsics_count_mean_6h'] + 3 * features_df['extrinsics_count_std_6h']).astype(int)
        features_df['block_size_anomaly'] = (features_df['block_size'] > features_df['block_size'].quantile(0.95)).astype(int)
        features_df['event_spike'] = (features_df['events_count'] > features_df['events_count_mean_6h'] + 2 * features_df['events_count_std_6h']).astype(int)
        features_df['cross_chain_anomaly'] = (features_df['cross_chain_messages'] > features_df['cross_chain_messages'].quantile(0.9)).astype(int)
        
        # Add network health features
        features_df['validator_anomaly'] = (features_df['validator_count'] < features_df['validator_count'].quantile(0.1)).astype(int)
        features_df['finalization_anomaly'] = (features_df['finalization_time'] > features_df['finalization_time'].quantile(0.95)).astype(int)
        
        # Merge with block details if available
        if not block_details.empty:
            # Merge on block_number
            features_df = features_df.merge(
                block_details[['block_number', 'block_hash', 'validator', 'finalized']], 
                on='block_number', 
                how='left'
            )
            
            # Add block details features
            features_df['has_block_hash'] = features_df['block_hash'].notna().astype(int)
            features_df['has_validator'] = features_df['validator'].notna().astype(int)
            features_df['is_finalized'] = features_df['finalized'].fillna(True).astype(int)
        
        # Add staking data (interpolated)
        if not staking_data.empty:
            for col in ['validators_count', 'nominators_count', 'active_era', 'total_staked', 'inflation_rate']:
                if col in staking_data.columns:
                    # Simple interpolation
                    staking_interp = np.interp(
                        features_df['timestamp'].astype(np.int64),
                        staking_data['timestamp'].astype(np.int64),
                        staking_data[col].fillna(0)
                    )
                    features_df[f'staking_{col}'] = staking_interp
                else:
                    features_df[f'staking_{col}'] = 0
        else:
            for col in ['validators_count', 'nominators_count', 'active_era', 'total_staked', 'inflation_rate']:
                features_df[f'staking_{col}'] = 0
        
        # Add parachain data (interpolated)
        if not parachain_data.empty:
            for col in ['parachains_count', 'hrmp_channels_count', 'active_parachains']:
                if col in parachain_data.columns:
                    parachain_interp = np.interp(
                        features_df['timestamp'].astype(np.int64),
                        parachain_data['timestamp'].astype(np.int64),
                        parachain_data[col].fillna(0)
                    )
                    features_df[f'parachain_{col}'] = parachain_interp
                else:
                    features_df[f'parachain_{col}'] = 0
        else:
            for col in ['parachains_count', 'hrmp_channels_count', 'active_parachains']:
                features_df[f'parachain_{col}'] = 0
        
        # Select numerical columns only
        numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp and block_number
        numerical_cols = [col for col in numerical_cols if col not in ['block_number', 'timestamp']]
        
        # Create feature matrix
        features_matrix = features_df[numerical_cols].fillna(0).values
        
        print(f"Created comprehensive feature matrix: {features_matrix.shape}")
        print(f"Features: {len(numerical_cols)}")
        
        return features_matrix, numerical_cols

class SimpleDataset(Dataset):
    """Simplified dataset for comprehensive data"""
    
    def __init__(self, features: np.ndarray, sequence_length: int = 24):
        self.features = features
        self.sequence_length = sequence_length
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create temporal sequences"""
        sequences = []
        for i in range(self.sequence_length, len(self.features)):
            sequence = self.features[i-self.sequence_length:i].flatten()
            sequences.append(sequence)
        return np.array(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])

class SimpleVAE(nn.Module):
    """Simplified VAE for comprehensive data"""
    
    def __init__(self, input_dim: int, config: SimpleConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
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
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Fraud detection head
        self.fraud_classifier = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # Low, Medium, High risk
        )
        
        # Risk score regressor
        self.risk_regressor = nn.Sequential(
            nn.Linear(config.latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
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

class SimpleComprehensiveGenerator:
    """Simplified comprehensive generator"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.data_loader = SimpleDataLoader()
        self.model = None
        self.scaler = RobustScaler()
        self.latent_representations = None
        self.feature_names = []
        
    def prepare_all_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare ALL data"""
        print("🚀 Preparing ALL Polkadot data...")
        
        # Load all data
        block_metrics, block_details, staking_data, parachain_data = self.data_loader.load_all_data()
        
        # Create comprehensive features
        features, feature_names = self.data_loader.create_comprehensive_features(
            block_metrics, block_details, staking_data, parachain_data
        )
        self.feature_names = feature_names
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create dataset
        dataset = SimpleDataset(features_scaled, self.config.sequence_length)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
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
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model on RTX 4090"""
        print("🔥 Training comprehensive model on RTX 4090...")
        
        # Initialize model
        input_dim = len(self.feature_names) * self.config.sequence_length
        self.model = SimpleVAE(input_dim, self.config).to(self.config.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.config.device)
                
                optimizer.zero_grad()
                
                if self.config.fp16 and scaler:
                    with torch.cuda.amp.autocast():
                        recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                        recon_loss = nn.MSELoss()(recon_batch, batch)
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.1 * kl_loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon_batch, mu, logvar, fraud_logits, risk_scores = self.model(batch)
                    recon_loss = nn.MSELoss()(recon_batch, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.1 * kl_loss
                    
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
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
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print("🎉 Training completed!")
    
    def generate_latent_space(self, data_loader: DataLoader):
        """Generate latent representations"""
        print("🎯 Generating comprehensive latent space...")
        
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
        
        print(f"Generated latent representations: {self.latent_representations.shape}")
        print(f"Fraud predictions: {np.bincount(fraud_predictions)}")
        print(f"Average risk score: {np.mean(risk_scores):.4f}")
        
        return self.latent_representations, fraud_predictions, risk_scores
    
    def save_model(self, model_path: str = "polkadot_simple_comprehensive_model.pth"):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'scaler_params': {
                'center': self.scaler.center_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }, model_path)
        
        print(f"Model saved to {model_path}")

async def main():
    """Main function"""
    print("🚀 Starting Simplified Comprehensive Polkadot Latent Space")
    print("Using ALL Polkadot data with full RTX 4090 utilization")
    
    # RTX 4090 optimized configuration
    config = SimpleConfig(
        device='cuda',
        batch_size=2048,
        latent_dim=256,
        hidden_dims=[512, 256, 128],
        learning_rate=1e-4,
        epochs=100,
        sequence_length=24,
        fp16=True
    )
    
    print(f"Configuration:")
    print(f"- Device: {config.device}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Latent dimensions: {config.latent_dim}")
    print(f"- Hidden layers: {config.hidden_dims}")
    print(f"- Epochs: {config.epochs}")
    print(f"- FP16: {config.fp16}")
    
    # Initialize generator
    generator = SimpleComprehensiveGenerator(config)
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Prepare ALL data
        train_loader, val_loader = generator.prepare_all_data()
        
        # Train model
        generator.train_model(train_loader, val_loader)
        
        # Generate latent space
        latent_repr, fraud_pred, risk_scores = generator.generate_latent_space(val_loader)
        
        # Save model
        generator.save_model()
        
        # Save results
        np.save("polkadot_simple_latent_representations.npy", latent_repr)
        np.save("polkadot_simple_fraud_predictions.npy", fraud_pred)
        np.save("polkadot_simple_risk_scores.npy", risk_scores)
        
        print("🎉 Comprehensive latent space generation completed!")
        print(f"Generated {len(latent_repr)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"GPU utilization: {(torch.cuda.max_memory_allocated() / (25.4 * 1e9)) * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
