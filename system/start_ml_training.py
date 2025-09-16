#!/usr/bin/env python3
"""
Quick Start Script for Polkadot ML Training
Starts the ML training pipeline with RTX 4090 optimization
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_environment():
    """Check if the ML environment is ready"""
    print("🔍 Checking ML Environment...")
    
    # Check if virtual environment exists
    if not Path("ml_env").exists():
        print("❌ ML environment not found. Please run setup_gpu_ml_environment.sh first.")
        return False
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check Hugging Face
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Hugging Face authenticated as: {user['name']}")
    except Exception as e:
        print(f"⚠️  Hugging Face authentication issue: {e}")
    
    return True

def start_training():
    """Start the ML training pipeline"""
    print("\n🚀 Starting Polkadot ML Training Pipeline...")
    
    # Check if training script exists
    if not Path("polkadot_ml_training_pipeline.py").exists():
        print("❌ Training pipeline not found. Please ensure polkadot_ml_training_pipeline.py exists.")
        return False
    
    # Set environment variables for GPU optimization
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8'
    })
    
    # Start training
    try:
        cmd = [sys.executable, "polkadot_ml_training_pipeline.py"]
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed with exit code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Polkadot ML Training Quick Start")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Ask user if they want to proceed
    print("\n" + "=" * 50)
    response = input("🚀 Ready to start training? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = start_training()
        if success:
            print("\n🎉 Training completed! Check the results in the output above.")
        else:
            print("\n❌ Training failed. Check the error messages above.")
    else:
        print("👋 Training cancelled. Run this script again when ready!")

if __name__ == "__main__":
    main()
