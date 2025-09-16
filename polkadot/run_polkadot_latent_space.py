#!/usr/bin/env python3
"""
Polkadot Latent Space Runner
============================

Quick runner script for generating Polkadot latent space with different configurations.
"""

import asyncio
import argparse
from polkadot_latent_space import PolkadotLatentSpaceGenerator, LatentSpaceConfig

async def run_latent_space_generation(config_name: str = "default"):
    """Run latent space generation with different configurations"""
    
    configs = {
        "default": LatentSpaceConfig(
            latent_dim=32,
            hidden_dims=[128, 64],
            learning_rate=1e-3,
            batch_size=256,
            epochs=50,
            sequence_length=24
        ),
        "high_dim": LatentSpaceConfig(
            latent_dim=64,
            hidden_dims=[256, 128, 64],
            learning_rate=1e-3,
            batch_size=128,
            epochs=100,
            sequence_length=48
        ),
        "fast": LatentSpaceConfig(
            latent_dim=16,
            hidden_dims=[64, 32],
            learning_rate=2e-3,
            batch_size=512,
            epochs=25,
            sequence_length=12
        ),
        "deep": LatentSpaceConfig(
            latent_dim=48,
            hidden_dims=[256, 128, 64, 32],
            learning_rate=5e-4,
            batch_size=128,
            epochs=150,
            sequence_length=36
        )
    }
    
    if config_name not in configs:
        print(f"Available configurations: {list(configs.keys())}")
        return
    
    config = configs[config_name]
    print(f"🚀 Running Polkadot Latent Space Generation with '{config_name}' configuration")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Hidden layers: {config.hidden_dims}")
    print(f"Epochs: {config.epochs}")
    
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
        generator.visualize_latent_space(f"polkadot_latent_space_{config_name}.html")
        
        # Analyze clusters
        cluster_analysis, cluster_labels = generator.analyze_latent_clusters()
        
        # Save model
        generator.save_model(f"polkadot_vae_model_{config_name}.pth")
        
        print("🎉 Polkadot latent space generation completed!")
        print(f"Generated {len(generator.latent_representations)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        
        # Print cluster analysis
        print("\n📊 Cluster Analysis:")
        for cluster_id, analysis in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {analysis['size']} samples ({analysis['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"Error in latent space generation: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Polkadot latent space")
    parser.add_argument(
        "--config", 
        choices=["default", "high_dim", "fast", "deep"],
        default="default",
        help="Configuration to use for latent space generation"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_latent_space_generation(args.config))

if __name__ == "__main__":
    main()
