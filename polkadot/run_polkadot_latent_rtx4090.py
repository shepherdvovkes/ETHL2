#!/usr/bin/env python3
"""
Polkadot Latent Space Training for RTX 4090
===========================================

Optimized configuration for RTX 4090 with 25.4 GB VRAM
"""

import asyncio
import torch
from polkadot_latent_space import PolkadotLatentSpaceGenerator, LatentSpaceConfig

async def train_on_rtx4090():
    """Train Polkadot latent space model optimized for RTX 4090"""
    
    print("🚀 Training Polkadot Latent Space on RTX 4090")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # RTX 4090 optimized configuration
    config = LatentSpaceConfig(
        latent_dim=64,                    # Higher dimensional latent space
        hidden_dims=[512, 256, 128, 64],  # Deeper network to utilize GPU
        learning_rate=1e-3,               # Standard learning rate
        batch_size=1024,                  # Large batch size for RTX 4090
        epochs=200,                       # More epochs for better convergence
        beta=1.0,                         # VAE beta parameter
        sequence_length=48,               # Longer sequences for better temporal modeling
        device='cuda'                     # Force CUDA usage
    )
    
    print(f"Configuration:")
    print(f"- Latent dimensions: {config.latent_dim}")
    print(f"- Hidden layers: {config.hidden_dims}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Epochs: {config.epochs}")
    print(f"- Sequence length: {config.sequence_length}")
    print(f"- Device: {config.device}")
    
    # Initialize generator
    generator = PolkadotLatentSpaceGenerator(config)
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Prepare data
        print("\n📊 Preparing data...")
        train_loader, val_loader = generator.prepare_data()
        
        # Check GPU memory usage
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Train model
        print("\n🔥 Starting training on RTX 4090...")
        train_losses, val_losses = generator.train_model(train_loader, val_loader)
        
        # Check GPU memory usage
        print(f"GPU Memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Generate latent representations
        print("\n🎯 Generating latent representations...")
        generator.generate_latent_representations(val_loader)
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        generator.visualize_latent_space("polkadot_latent_space_rtx4090.html")
        
        # Analyze clusters
        print("\n🔍 Analyzing clusters...")
        cluster_analysis, cluster_labels = generator.analyze_latent_clusters(n_clusters=8)
        
        # Save model
        print("\n💾 Saving model...")
        generator.save_model("polkadot_vae_model_rtx4090.pth")
        
        # Save latent representations
        import numpy as np
        np.save("polkadot_latent_representations_rtx4090.npy", generator.latent_representations)
        
        print("\n🎉 Training completed successfully!")
        print(f"Generated {len(generator.latent_representations)} latent representations")
        print(f"Latent space dimension: {config.latent_dim}")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        # Print cluster analysis
        print("\n📊 Cluster Analysis:")
        for cluster_id, analysis in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {analysis['size']} samples ({analysis['percentage']:.1f}%)")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"- Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"- GPU utilization: {(torch.cuda.max_memory_allocated() / (25.4 * 1e9)) * 100:.1f}%")
        
        return generator
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        # Clear GPU memory on error
        torch.cuda.empty_cache()
        raise
    finally:
        # Clear GPU memory
        torch.cuda.empty_cache()

async def main():
    """Main function"""
    await train_on_rtx4090()

if __name__ == "__main__":
    asyncio.run(main())
