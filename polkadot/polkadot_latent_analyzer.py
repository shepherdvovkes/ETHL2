#!/usr/bin/env python3
"""
Polkadot Latent Space Analyzer
==============================

Advanced analysis tools for the generated Polkadot latent space:
- Latent space exploration
- Pattern discovery
- Anomaly detection
- Network state classification
- Temporal dynamics analysis
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

class PolkadotLatentAnalyzer:
    """Advanced analyzer for Polkadot latent space"""
    
    def __init__(self, model_path: str = "polkadot_vae_model.pth", 
                 database_path: str = "polkadot_archive_data.db"):
        self.model_path = model_path
        self.database_path = database_path
        self.latent_representations = None
        self.feature_names = []
        self.block_data = None
        self.cluster_labels = None
        
    def load_latent_data(self, latent_file: str = "polkadot_latent_representations.npy"):
        """Load pre-computed latent representations"""
        try:
            self.latent_representations = np.load(latent_file)
            print(f"Loaded latent representations: {self.latent_representations.shape}")
        except FileNotFoundError:
            print(f"Latent file {latent_file} not found. Please generate latent space first.")
            return False
        return True
    
    def load_block_data(self):
        """Load corresponding block data for analysis"""
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
        
        self.block_data = pd.read_sql_query(query, conn)
        self.block_data['timestamp'] = pd.to_datetime(self.block_data['timestamp'])
        conn.close()
        
        print(f"Loaded {len(self.block_data)} block records")
        return self.block_data
    
    def analyze_latent_dynamics(self, window_size: int = 100):
        """Analyze temporal dynamics in latent space"""
        if self.latent_representations is None:
            print("No latent representations loaded")
            return
        
        print("Analyzing latent space dynamics...")
        
        # Calculate latent space velocity (change over time)
        latent_velocity = np.diff(self.latent_representations, axis=0)
        latent_speed = np.linalg.norm(latent_velocity, axis=1)
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(latent_speed).rolling(window=window_size).mean()
        rolling_std = pd.Series(latent_speed).rolling(window=window_size).std()
        
        # Detect regime changes (high velocity periods)
        threshold = rolling_mean.mean() + 2 * rolling_std.std()
        regime_changes = latent_speed > threshold
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Latent Space Velocity', 'Regime Changes'),
            vertical_spacing=0.1
        )
        
        # Velocity plot
        fig.add_trace(
            go.Scatter(
                y=latent_speed,
                mode='lines',
                name='Velocity',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=rolling_mean,
                mode='lines',
                name='Rolling Mean',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Regime changes
        fig.add_trace(
            go.Scatter(
                y=regime_changes.astype(int),
                mode='lines',
                name='Regime Changes',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Polkadot Latent Space Dynamics",
            height=600,
            showlegend=True
        )
        
        fig.write_html("polkadot_latent_dynamics.html")
        print("Latent dynamics analysis saved to polkadot_latent_dynamics.html")
        
        return {
            'velocity': latent_speed,
            'rolling_mean': rolling_mean,
            'regime_changes': regime_changes,
            'threshold': threshold
        }
    
    def detect_anomalies(self, contamination: float = 0.1):
        """Detect anomalies in latent space using isolation forest"""
        if self.latent_representations is None:
            print("No latent representations loaded")
            return
        
        print(f"Detecting anomalies with contamination rate: {contamination}")
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.latent_representations)
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(self.latent_representations)
        
        # Identify anomalies
        anomalies = anomaly_labels == -1
        n_anomalies = np.sum(anomalies)
        
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(anomalies)*100:.1f}%)")
        
        # Create visualization
        if self.latent_representations.shape[1] > 2:
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(self.latent_representations)
        else:
            latent_2d = self.latent_representations
        
        fig = px.scatter(
            x=latent_2d[:, 0],
            y=latent_2d[:, 1],
            color=anomaly_labels,
            title=f"Anomaly Detection in Polkadot Latent Space ({n_anomalies} anomalies)",
            labels={'x': 'PC1', 'y': 'PC2'},
            color_discrete_map={1: 'blue', -1: 'red'}
        )
        
        fig.write_html("polkadot_latent_anomalies.html")
        print("Anomaly detection results saved to polkadot_latent_anomalies.html")
        
        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'anomaly_indices': np.where(anomalies)[0]
        }
    
    def analyze_network_states(self, n_clusters: int = 5):
        """Analyze different network states using clustering"""
        if self.latent_representations is None:
            print("No latent representations loaded")
            return
        
        print(f"Analyzing network states with {n_clusters} clusters...")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.latent_representations)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.latent_representations, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Calculate cluster statistics
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_std = np.std(self.latent_representations[cluster_mask], axis=0)
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'percentage': (cluster_size / len(self.cluster_labels)) * 100,
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'indices': np.where(cluster_mask)[0].tolist()
            }
        
        # Create visualization
        if self.latent_representations.shape[1] > 2:
            # Use UMAP for better visualization
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            latent_2d = reducer.fit_transform(self.latent_representations)
        else:
            latent_2d = self.latent_representations
        
        fig = px.scatter(
            x=latent_2d[:, 0],
            y=latent_2d[:, 1],
            color=self.cluster_labels,
            title=f"Polkadot Network States ({n_clusters} clusters)",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
            hover_name=[f"Sample {i}" for i in range(len(latent_2d))]
        )
        
        fig.write_html("polkadot_network_states.html")
        print("Network states analysis saved to polkadot_network_states.html")
        
        # Save detailed analysis
        with open('polkadot_network_states_analysis.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        return cluster_analysis
    
    def correlate_with_block_metrics(self):
        """Correlate latent space with actual block metrics"""
        if self.latent_representations is None or self.block_data is None:
            print("Missing latent representations or block data")
            return
        
        print("Correlating latent space with block metrics...")
        
        # Align data (latent representations might be fewer due to sequence length)
        n_latent = len(self.latent_representations)
        n_blocks = len(self.block_data)
        
        if n_latent < n_blocks:
            # Use the last n_latent blocks
            block_data_aligned = self.block_data.tail(n_latent).reset_index(drop=True)
        else:
            block_data_aligned = self.block_data.copy()
        
        # Calculate correlations
        correlations = {}
        
        # Select numerical columns for correlation
        numerical_cols = [
            'extrinsics_count', 'events_count', 'block_size',
            'validator_count', 'finalization_time', 'parachain_blocks',
            'cross_chain_messages'
        ]
        
        for i, col in enumerate(numerical_cols):
            if col in block_data_aligned.columns:
                # Calculate correlation with each latent dimension
                latent_correlations = []
                for j in range(min(10, self.latent_representations.shape[1])):  # Top 10 latent dims
                    corr = np.corrcoef(
                        self.latent_representations[:len(block_data_aligned), j],
                        block_data_aligned[col]
                    )[0, 1]
                    latent_correlations.append(corr)
                
                correlations[col] = latent_correlations
        
        # Create correlation heatmap
        corr_df = pd.DataFrame(correlations).T
        corr_df.columns = [f'Latent_{i}' for i in range(corr_df.shape[1])]
        
        fig = px.imshow(
            corr_df,
            title="Correlation between Latent Space and Block Metrics",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig.write_html("polkadot_latent_correlations.html")
        print("Correlation analysis saved to polkadot_latent_correlations.html")
        
        return correlations
    
    def generate_network_insights(self):
        """Generate comprehensive network insights"""
        print("Generating comprehensive network insights...")
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.latent_representations) if self.latent_representations is not None else 0,
            'latent_dimensions': self.latent_representations.shape[1] if self.latent_representations is not None else 0
        }
        
        # Analyze latent space properties
        if self.latent_representations is not None:
            # Calculate latent space statistics
            latent_mean = np.mean(self.latent_representations, axis=0)
            latent_std = np.std(self.latent_representations, axis=0)
            
            insights['latent_space'] = {
                'mean': latent_mean.tolist(),
                'std': latent_std.tolist(),
                'variance_explained': np.var(self.latent_representations, axis=0).tolist()
            }
            
            # Identify most variable dimensions
            variance = np.var(self.latent_representations, axis=0)
            top_variable_dims = np.argsort(variance)[-5:][::-1]
            insights['most_variable_dimensions'] = top_variable_dims.tolist()
        
        # Network state analysis
        if self.cluster_labels is not None:
            unique_clusters = np.unique(self.cluster_labels)
            insights['network_states'] = {
                'n_states': len(unique_clusters),
                'state_distribution': {
                    str(cluster): int(np.sum(self.cluster_labels == cluster))
                    for cluster in unique_clusters
                }
            }
        
        # Save insights
        with open('polkadot_network_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print("Network insights saved to polkadot_network_insights.json")
        return insights
    
    def create_dashboard(self):
        """Create comprehensive dashboard"""
        print("Creating comprehensive analysis dashboard...")
        
        # Load all data
        if not self.load_latent_data():
            return
        
        self.load_block_data()
        
        # Run all analyses
        dynamics = self.analyze_latent_dynamics()
        anomalies = self.detect_anomalies()
        network_states = self.analyze_network_states()
        correlations = self.correlate_with_block_metrics()
        insights = self.generate_network_insights()
        
        print("🎉 Comprehensive analysis completed!")
        print("Generated files:")
        print("- polkadot_latent_dynamics.html")
        print("- polkadot_latent_anomalies.html") 
        print("- polkadot_network_states.html")
        print("- polkadot_latent_correlations.html")
        print("- polkadot_network_states_analysis.json")
        print("- polkadot_network_insights.json")

def main():
    """Main function"""
    analyzer = PolkadotLatentAnalyzer()
    analyzer.create_dashboard()

if __name__ == "__main__":
    main()
