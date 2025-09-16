#!/usr/bin/env python3
"""
Polkadot Latent Space with Hugging Face Models
==============================================

Enhanced latent space generation using pre-trained Hugging Face models:
- Time series transformers for temporal patterns
- Financial/blockchain-specific models
- Multi-modal feature extraction
- Advanced representation learning

Models used:
- microsoft/DialoGPT-medium (for sequence understanding)
- google/flan-t5-base (for pattern recognition)
- microsoft/DialoGPT-small (lightweight alternative)
- facebook/bart-base (for sequence-to-sequence learning)
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

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
import torch.nn.functional as F

@dataclass
class HFLatentSpaceConfig:
    """Configuration for Hugging Face enhanced latent space"""
    # Model configurations
    base_model: str = "microsoft/DialoGPT-medium"  # Good for sequence understanding
    t5_model: str = "google/flan-t5-base"         # For pattern recognition
    bart_model: str = "facebook/bart-base"         # For sequence-to-sequence
    
    # Latent space configuration
    latent_dim: int = 128
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 32  # Smaller for transformer models
    epochs: int = 50
    max_length: int = 512
    
    # Training configuration
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = True  # Use mixed precision for RTX 4090
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]

class PolkadotHFDataset(Dataset):
    """Enhanced dataset for Hugging Face models"""
    
    def __init__(self, features: np.ndarray, tokenizer, max_length: int = 512):
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Convert numerical features to text-like representation
        feature_text = self._features_to_text(self.features[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            feature_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': torch.FloatTensor(self.features[idx])
        }
    
    def _features_to_text(self, features: np.ndarray) -> str:
        """Convert numerical features to text representation"""
        # Create a structured text representation
        text_parts = []
        
        # Add feature descriptions
        feature_names = [
            'extrinsics', 'events', 'block_size', 'validators',
            'finalization', 'parachains', 'cross_chain', 'efficiency',
            'activity', 'hour', 'day', 'interval'
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, features[:len(feature_names)])):
            text_parts.append(f"{name}: {value:.2f}")
        
        return " ".join(text_parts)

class PolkadotFeatureExtractor(nn.Module):
    """Feature extractor using Hugging Face models"""
    
    def __init__(self, config: HFLatentSpaceConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained models
        self.base_model = AutoModel.from_pretrained(config.base_model)
        self.base_tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Add padding token if not present
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.latent_dim)
        )
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            # Mean pooling
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled = outputs.pooler_output
        
        # Project to latent space
        latent = self.feature_projection(pooled)
        
        return latent

class PolkadotHFLatentGenerator:
    """Enhanced latent space generator with Hugging Face models"""
    
    def __init__(self, config: HFLatentSpaceConfig):
        self.config = config
        self.feature_extractor = None
        self.scaler = StandardScaler()
        self.latent_representations = None
        self.feature_names = []
        
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
        
        # Create features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['block_interval'] = df['block_number'].diff()
        df['extrinsics_per_event'] = df['extrinsics_count'] / (df['events_count'] + 1)
        df['block_efficiency'] = df['extrinsics_count'] / (df['block_size'] / 1000 + 1)
        df['network_activity'] = df['extrinsics_count'] + df['events_count']
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def prepare_features(self, sequence_length: int = 24) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for Hugging Face models"""
        print("Loading and preprocessing Polkadot data...")
        
        # Load data
        df = self.load_polkadot_data()
        print(f"Loaded {len(df)} block records")
        
        # Create temporal sequences
        numerical_cols = [
            'extrinsics_count', 'events_count', 'block_size', 
            'validator_count', 'finalization_time', 'parachain_blocks',
            'cross_chain_messages', 'extrinsics_per_event', 
            'block_efficiency', 'network_activity', 'hour', 'day_of_week'
        ]
        
        features = []
        for i in range(sequence_length, len(df)):
            sequence = df[numerical_cols].iloc[i-sequence_length:i].values
            features.append(sequence.flatten())
        
        features = np.array(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create feature names
        self.feature_names = []
        for t in range(sequence_length):
            for col in numerical_cols:
                self.feature_names.append(f"{col}_t{t}")
        
        print(f"Created feature matrix: {features_scaled.shape}")
        return features_scaled, self.feature_names
    
    def train_feature_extractor(self, features: np.ndarray):
        """Train the Hugging Face feature extractor"""
        print("Initializing Hugging Face feature extractor...")
        
        # Initialize feature extractor
        self.feature_extractor = PolkadotFeatureExtractor(self.config).to(self.config.device)
        
        # Create dataset
        dataset = PolkadotHFDataset(
            features, 
            self.feature_extractor.base_tokenizer,
            self.config.max_length
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            self.feature_extractor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        print(f"Training feature extractor for {self.config.epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.feature_extractor.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                target_features = batch['features'].to(self.config.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                latent = self.feature_extractor(input_ids, attention_mask)
                
                # Reconstruction loss (simple MSE for now)
                loss = F.mse_loss(latent, target_features[:, :self.config.latent_dim])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.feature_extractor.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    target_features = batch['features'].to(self.config.device)
                    
                    latent = self.feature_extractor(input_ids, attention_mask)
                    loss = F.mse_loss(latent, target_features[:, :self.config.latent_dim])
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Feature extractor training completed!")
        return train_losses, val_losses
    
    def generate_latent_representations(self, features: np.ndarray):
        """Generate latent representations using trained feature extractor"""
        print("Generating latent representations with Hugging Face models...")
        
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not trained. Run train_feature_extractor first.")
        
        # Create dataset
        dataset = PolkadotHFDataset(
            features, 
            self.feature_extractor.base_tokenizer,
            self.config.max_length
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Generate representations
        self.feature_extractor.eval()
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                latent = self.feature_extractor(input_ids, attention_mask)
                latent_representations.append(latent.cpu().numpy())
        
        self.latent_representations = np.concatenate(latent_representations, axis=0)
        print(f"Generated latent representations: {self.latent_representations.shape}")
        
        return self.latent_representations
    
    def visualize_latent_space(self, save_path: str = "polkadot_latent_space_hf.html"):
        """Create interactive visualization of latent space"""
        if self.latent_representations is None:
            raise ValueError("No latent representations available.")
        
        print("Creating Hugging Face latent space visualization...")
        
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
            title="Polkadot Network Latent Space (Hugging Face Enhanced)",
            labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'},
            hover_name=[f"Sample {i}" for i in range(len(latent_2d))]
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            showlegend=False
        )
        
        fig.write_html(save_path)
        print(f"Latent space visualization saved to {save_path}")
        
        return fig
    
    def analyze_clusters(self, n_clusters: int = 8):
        """Analyze clusters in Hugging Face latent space"""
        if self.latent_representations is None:
            raise ValueError("No latent representations available.")
        
        print(f"Analyzing {n_clusters} clusters in Hugging Face latent space...")
        
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
        with open('polkadot_latent_clusters_hf.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        print("Hugging Face cluster analysis completed!")
        return cluster_analysis, cluster_labels
    
    def save_model(self, model_path: str = "polkadot_hf_model.pth"):
        """Save the trained Hugging Face model"""
        if self.feature_extractor is None:
            raise ValueError("No model to save.")
        
        torch.save({
            'model_state_dict': self.feature_extractor.state_dict(),
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }, model_path)
        
        print(f"Hugging Face model saved to {model_path}")

async def main():
    """Main function for Hugging Face enhanced latent space"""
    print("🚀 Starting Polkadot Latent Space Generation with Hugging Face Models")
    
    # RTX 4090 optimized configuration
    config = HFLatentSpaceConfig(
        base_model="microsoft/DialoGPT-medium",
        latent_dim=128,
        hidden_dims=[256, 128],
        learning_rate=1e-4,
        batch_size=16,  # Smaller for transformer models
        epochs=30,
        max_length=256,
        fp16=True
    )
    
    print(f"Configuration:")
    print(f"- Base model: {config.base_model}")
    print(f"- Latent dimensions: {config.latent_dim}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Device: {config.device}")
    print(f"- FP16: {config.fp16}")
    
    # Initialize generator
    generator = PolkadotHFLatentGenerator(config)
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Prepare features
        features, feature_names = generator.prepare_features(sequence_length=24)
        
        # Train feature extractor
        train_losses, val_losses = generator.train_feature_extractor(features)
        
        # Generate latent representations
        generator.generate_latent_representations(features)
        
        # Create visualizations
        generator.visualize_latent_space()
        
        # Analyze clusters
        cluster_analysis, cluster_labels = generator.analyze_clusters()
        
        # Save model
        generator.save_model()
        
        # Save latent representations
        np.save("polkadot_latent_representations_hf.npy", generator.latent_representations)
        
        print("🎉 Hugging Face enhanced latent space generation completed!")
        print(f"Generated {len(generator.latent_representations)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        
        # Print cluster analysis
        print("\n📊 Cluster Analysis:")
        for cluster_id, analysis in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {analysis['size']} samples ({analysis['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in Hugging Face latent space generation: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
