#!/usr/bin/env python3
"""
Polkadot Latent Space Generator
===============================

Creates a latent space representation of Polkadot mainnet data using:
- Variational Autoencoder (VAE) for continuous latent space
- Time series features for temporal patterns
- Multi-modal data fusion (blocks, staking, governance)
- Dimensionality reduction and visualization

Features:
- Block-level patterns (extrinsics, events, size)
- Network health indicators
- Temporal dynamics
- Cross-chain activity patterns
- Governance and staking patterns
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
class LatentSpaceConfig:
    """Configuration for latent space generation"""
    latent_dim: int = 32
    hidden_dims: List[int] = None
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    beta: float = 1.0  # VAE beta parameter
    sequence_length: int = 24  # Hours of data for temporal features
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

class PolkadotDataLoader:
    """Load and preprocess Polkadot data for latent space generation"""
    
    def __init__(self, database_path: str = "polkadot_archive_data.db"):
        self.database_path = database_path
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_block_metrics(self) -> pd.DataFrame:
        """Load and preprocess block metrics data"""
        conn = sqlite3.connect(self.database_path)
        
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
        
        # Convert timestamp to datetime with mixed format handling
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Create temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['block_interval'] = df['block_number'].diff()
        
        # Create derived features
        df['extrinsics_per_event'] = df['extrinsics_count'] / (df['events_count'] + 1)
        df['block_efficiency'] = df['extrinsics_count'] / (df['block_size'] / 1000 + 1)
        df['network_activity'] = df['extrinsics_count'] + df['events_count']
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def load_network_metrics(self) -> pd.DataFrame:
        """Load staking and governance data"""
        conn = sqlite3.connect(self.database_path)
        
        # Load staking data
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
        
        staking_df = pd.read_sql_query(staking_query, conn)
        
        # Load parachain data
        parachain_query = """
        SELECT 
            timestamp,
            parachains_count,
            hrmp_channels_count,
            active_parachains
        FROM parachain_data
        ORDER BY timestamp
        """
        
        parachain_df = pd.read_sql_query(parachain_query, conn)
        
        conn.close()
        
        # Convert timestamps with mixed format handling
        if not staking_df.empty:
            staking_df['timestamp'] = pd.to_datetime(staking_df['timestamp'], format='mixed', errors='coerce')
        if not parachain_df.empty:
            parachain_df['timestamp'] = pd.to_datetime(parachain_df['timestamp'], format='mixed', errors='coerce')
        
        return staking_df, parachain_df
    
    def create_temporal_features(self, df: pd.DataFrame, sequence_length: int = 24) -> np.ndarray:
        """Create temporal sequence features"""
        features = []
        
        # Select numerical features for sequences
        numerical_cols = [
            'extrinsics_count', 'events_count', 'block_size', 
            'validator_count', 'finalization_time', 'parachain_blocks',
            'cross_chain_messages', 'extrinsics_per_event', 
            'block_efficiency', 'network_activity'
        ]
        
        # Ensure all columns exist
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        for i in range(sequence_length, len(df)):
            sequence = df[available_cols].iloc[i-sequence_length:i].values
            features.append(sequence.flatten())
        
        return np.array(features)
    
    def prepare_features(self, sequence_length: int = 24) -> Tuple[np.ndarray, List[str]]:
        """Prepare all features for latent space generation"""
        # Load data
        block_df = self.load_block_metrics()
        staking_df, parachain_df = self.load_network_metrics()
        
        print(f"Loaded {len(block_df)} block records")
        print(f"Loaded {len(staking_df)} staking records")
        print(f"Loaded {len(parachain_df)} parachain records")
        
        # Create temporal features
        temporal_features = self.create_temporal_features(block_df, sequence_length)
        
        # Create static features (network-level)
        static_features = []
        for i in range(len(temporal_features)):
            # Get corresponding block data
            block_idx = i + sequence_length
            if block_idx < len(block_df):
                block_data = block_df.iloc[block_idx]
                
                # Static features
                static_feat = [
                    block_data['hour'],
                    block_data['day_of_week'],
                    block_data['block_interval'] if not pd.isna(block_data['block_interval']) else 0
                ]
                
                # Add network metrics if available
                if not staking_df.empty:
                    # Find closest staking data
                    block_time = block_data['timestamp']
                    time_diffs = abs((staking_df['timestamp'] - block_time).dt.total_seconds())
                    if len(time_diffs) > 0:
                        closest_idx = time_diffs.idxmin()
                        staking_data = staking_df.iloc[closest_idx]
                        static_feat.extend([
                            staking_data['validators_count'],
                            staking_data['nominators_count'],
                            staking_data['active_era'],
                            staking_data['total_staked'],
                            staking_data['inflation_rate']
                        ])
                    else:
                        static_feat.extend([0, 0, 0, 0, 0])
                else:
                    static_feat.extend([0, 0, 0, 0, 0])
                
                # Add parachain metrics if available
                if not parachain_df.empty:
                    block_time = block_data['timestamp']
                    time_diffs = abs((parachain_df['timestamp'] - block_time).dt.total_seconds())
                    if len(time_diffs) > 0:
                        closest_idx = time_diffs.idxmin()
                        parachain_data = parachain_df.iloc[closest_idx]
                        static_feat.extend([
                            parachain_data['parachains_count'],
                            parachain_data['hrmp_channels_count'],
                            parachain_data['active_parachains']
                        ])
                    else:
                        static_feat.extend([0, 0, 0])
                else:
                    static_feat.extend([0, 0, 0])
                
                static_features.append(static_feat)
        
        static_features = np.array(static_features)
        
        # Combine temporal and static features
        combined_features = np.concatenate([temporal_features, static_features], axis=1)
        
        # Create feature names
        temporal_names = []
        numerical_cols = [
            'extrinsics_count', 'events_count', 'block_size', 
            'validator_count', 'finalization_time', 'parachain_blocks',
            'cross_chain_messages', 'extrinsics_per_event', 
            'block_efficiency', 'network_activity'
        ]
        
        for t in range(sequence_length):
            for col in numerical_cols:
                temporal_names.append(f"{col}_t{t}")
        
        static_names = [
            'hour', 'day_of_week', 'block_interval',
            'validators_count', 'nominators_count', 'active_era',
            'total_staked', 'inflation_rate',
            'parachains_count', 'hrmp_channels_count', 'active_parachains'
        ]
        
        self.feature_names = temporal_names + static_names
        
        print(f"Created feature matrix: {combined_features.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        return combined_features, self.feature_names

class PolkadotDataset(Dataset):
    """PyTorch dataset for Polkadot data"""
    
    def __init__(self, features: np.ndarray):
        self.features = torch.FloatTensor(features)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for Polkadot latent space"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
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
        return recon_x, mu, logvar

class PolkadotLatentSpaceGenerator:
    """Main class for generating Polkadot latent space"""
    
    def __init__(self, config: LatentSpaceConfig):
        self.config = config
        self.data_loader = PolkadotDataLoader()
        self.model = None
        self.scaler = StandardScaler()
        self.latent_representations = None
        self.feature_names = []
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training"""
        print("Loading and preprocessing Polkadot data...")
        
        # Load features
        features, feature_names = self.data_loader.prepare_features(self.config.sequence_length)
        self.feature_names = feature_names
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create dataset
        dataset = PolkadotDataset(features_scaled)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the VAE model"""
        print("Initializing VAE model...")
        
        # Initialize model
        input_dim = len(self.feature_names)
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims
        ).to(self.config.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"Training VAE for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.config.device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                
                # VAE loss
                recon_loss = nn.MSELoss()(recon_batch, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.config.beta * kl_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.config.device)
                    recon_batch, mu, logvar = self.model(batch)
                    
                    recon_loss = nn.MSELoss()(recon_batch, batch)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.config.beta * kl_loss
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config.__dict__
        }
        
        with open('polkadot_vae_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("Training completed!")
        return train_losses, val_losses
    
    def generate_latent_representations(self, data_loader: DataLoader):
        """Generate latent representations for all data"""
        print("Generating latent representations...")
        
        self.model.eval()
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.config.device)
                mu, logvar = self.model.encode(batch)
                latent_representations.append(mu.cpu().numpy())
        
        self.latent_representations = np.concatenate(latent_representations, axis=0)
        print(f"Generated latent representations: {self.latent_representations.shape}")
        
        return self.latent_representations
    
    def visualize_latent_space(self, save_path: str = "polkadot_latent_space.html"):
        """Create interactive visualization of latent space"""
        if self.latent_representations is None:
            raise ValueError("No latent representations available. Run generate_latent_representations first.")
        
        print("Creating latent space visualization...")
        
        # Reduce to 2D for visualization
        if self.latent_representations.shape[1] > 2:
            print("Reducing dimensionality to 2D using t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = tsne.fit_transform(self.latent_representations)
        else:
            latent_2d = self.latent_representations
        
        # Create interactive plot
        fig = px.scatter(
            x=latent_2d[:, 0], 
            y=latent_2d[:, 1],
            title="Polkadot Network Latent Space",
            labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'},
            hover_name=[f"Sample {i}" for i in range(len(latent_2d))]
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=False
        )
        
        fig.write_html(save_path)
        print(f"Latent space visualization saved to {save_path}")
        
        return fig
    
    def analyze_latent_clusters(self, n_clusters: int = 5):
        """Analyze clusters in latent space"""
        if self.latent_representations is None:
            raise ValueError("No latent representations available.")
        
        print(f"Analyzing {n_clusters} clusters in latent space...")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.latent_representations)
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            cluster_analysis[cluster_id] = {
                'size': int(cluster_size),
                'percentage': float((cluster_size / len(cluster_labels)) * 100),
                'center': kmeans.cluster_centers_[cluster_id].tolist()
            }
        
        # Save analysis
        with open('polkadot_latent_clusters.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        print("Cluster analysis completed!")
        return cluster_analysis, cluster_labels
    
    def save_model(self, model_path: str = "polkadot_vae_model.pth"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "polkadot_vae_model.pth"):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        # Restore config
        self.config = LatentSpaceConfig(**checkpoint['config'])
        
        # Restore feature names
        self.feature_names = checkpoint['feature_names']
        
        # Restore scaler
        scaler_params = checkpoint['scaler_params']
        self.scaler.mean_ = np.array(scaler_params['mean'])
        self.scaler.scale_ = np.array(scaler_params['scale'])
        
        # Initialize and load model
        input_dim = len(self.feature_names)
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims
        ).to(self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")

async def main():
    """Main function to run latent space generation"""
    print("🚀 Starting Polkadot Latent Space Generation")
    
    # Configuration
    config = LatentSpaceConfig(
        latent_dim=32,
        hidden_dims=[128, 64],
        learning_rate=1e-3,
        batch_size=256,
        epochs=50,
        sequence_length=24
    )
    
    # Initialize generator
    generator = PolkadotLatentSpaceGenerator(config)
    
    try:
        # Prepare data
        train_loader, val_loader = generator.prepare_data()
        
        # Train model
        train_losses, val_losses = generator.train_model(train_loader, val_loader)
        
        # Generate latent representations
        generator.generate_latent_representations(val_loader)
        
        # Create visualizations
        generator.visualize_latent_space()
        
        # Analyze clusters
        cluster_analysis, cluster_labels = generator.analyze_latent_clusters()
        
        # Save model
        generator.save_model()
        
        print("🎉 Polkadot latent space generation completed!")
        print(f"Generated {len(generator.latent_representations)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        
    except Exception as e:
        print(f"Error in latent space generation: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
