#!/bin/bash
# RTX 4090 ML Environment Setup Script
# ====================================
# This script sets up the complete environment for training ML models on RTX 4090

set -e

echo "🚀 Setting up RTX 4090 ML Environment for Polkadot Data Analysis"
echo "================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
print_status "Detected Ubuntu version: $UBUNTU_VERSION"

# Check if RTX 4090 is detected
print_header "Checking RTX 4090 GPU..."
if lspci | grep -i nvidia | grep -i "4090\|RTX"; then
    print_status "RTX 4090 detected!"
else
    print_warning "RTX 4090 not detected. Continuing with setup anyway..."
fi

# Update system packages
print_header "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
print_header "Installing essential packages..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-tk \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libffi-dev \
    libssl-dev

# Install NVIDIA drivers (latest stable)
print_header "Installing NVIDIA drivers..."
sudo apt install -y nvidia-driver-535
print_warning "NVIDIA drivers installed. A reboot may be required after setup."

# Install CUDA Toolkit 12.1 (compatible with RTX 4090)
print_header "Installing CUDA Toolkit 12.1..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Install cuDNN
print_header "Installing cuDNN..."
sudo apt install -y libcudnn8 libcudnn8-dev

# Set up CUDA environment variables
print_header "Setting up CUDA environment variables..."
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc

# Source the bashrc
source ~/.bashrc

# Create Python virtual environment
print_header "Creating Python virtual environment..."
cd /home/vovkes/ETHL2
python3 -m venv ml_env
source ml_env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
print_header "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML and data science packages
print_header "Installing ML and data science packages..."
pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    jupyter==1.0.0 \
    jupyterlab==4.0.7 \
    ipykernel==6.25.0 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    catboost==1.2.2 \
    optuna==3.3.0 \
    mlflow==2.5.0 \
    wandb==0.15.8 \
    tensorboard==2.13.0

# Install Hugging Face packages
print_header "Installing Hugging Face packages..."
pip install \
    transformers==4.33.2 \
    datasets==2.14.5 \
    accelerate==0.21.0 \
    huggingface-hub==0.16.4 \
    tokenizers==0.13.3 \
    safetensors==0.3.3

# Install additional ML packages
print_header "Installing additional ML packages..."
pip install \
    torch-audio \
    torchvision \
    torchaudio \
    timm==0.9.2 \
    efficientnet-pytorch==0.7.1 \
    albumentations==1.3.1 \
    opencv-python==4.8.0.76 \
    Pillow==10.0.0 \
    scipy==1.11.1 \
    statsmodels==0.14.0 \
    ta-lib==0.4.25 \
    yfinance==0.2.18 \
    ccxt==4.0.77

# Install database and API packages
print_header "Installing database and API packages..."
pip install \
    sqlalchemy==2.0.19 \
    psycopg2-binary==2.9.7 \
    asyncpg==0.28.0 \
    aiohttp==3.8.5 \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    loguru==0.7.0 \
    fastapi==0.101.1 \
    uvicorn==0.23.2 \
    pydantic==2.1.1

# Install monitoring and logging
print_header "Installing monitoring and logging packages..."
pip install \
    prometheus-client==0.17.1 \
    grafana-api==1.0.3 \
    redis==4.6.0 \
    celery==5.3.1 \
    flower==2.0.1

# Create requirements file
print_header "Creating requirements file..."
pip freeze > requirements_ml.txt

# Create GPU test script
print_header "Creating GPU test script..."
cat > test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Test Script for RTX 4090
============================
Tests CUDA availability and GPU performance
"""

import torch
import numpy as np
import time
from datetime import datetime

def test_cuda_availability():
    """Test CUDA availability"""
    print("🔍 Testing CUDA Availability")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("❌ CUDA not available!")
        return False
    
    return True

def test_gpu_performance():
    """Test GPU performance with matrix operations"""
    print("\n🚀 Testing GPU Performance")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping performance test")
        return
    
    device = torch.device('cuda:0')
    
    # Test matrix multiplication
    sizes = [1000, 2000, 4000, 8000]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrix multiplication...")
        
        # Create random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        for _ in range(3):
            _ = torch.mm(a, b)
        
        torch.cuda.synchronize()
        
        # Time the operation
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"  Average time: {avg_time:.4f} seconds")
        print(f"  Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Clear memory
        del a, b, c
        torch.cuda.empty_cache()

def test_neural_network_training():
    """Test neural network training on GPU"""
    print("\n🧠 Testing Neural Network Training")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping neural network test")
        return
    
    device = torch.device('cuda:0')
    
    # Create a simple neural network
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(1000, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, 1000)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create model and data
    model = SimpleNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create synthetic data
    batch_size = 64
    x = torch.randn(batch_size, 1000, device=device)
    y = torch.randn(batch_size, 1000, device=device)
    
    # Training loop
    print("Training neural network...")
    start_time = time.time()
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Final loss: {loss.item():.6f}")

def main():
    """Main test function"""
    print("🎯 RTX 4090 GPU Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Test CUDA availability
    if not test_cuda_availability():
        print("\n❌ CUDA tests failed. Please check your installation.")
        return
    
    # Test GPU performance
    test_gpu_performance()
    
    # Test neural network training
    test_neural_network_training()
    
    print("\n✅ All GPU tests completed successfully!")
    print(f"Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
EOF

# Make test script executable
chmod +x test_gpu.py

# Create ML training configuration
print_header "Creating ML training configuration..."
cat > ml_config.py << 'EOF'
"""
ML Training Configuration for RTX 4090
======================================
Optimized settings for training on RTX 4090
"""

import torch
import os

class MLConfig:
    """Configuration for ML training on RTX 4090"""
    
    # GPU Settings
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory
    MIXED_PRECISION = True  # Use mixed precision training
    
    # Training Settings
    BATCH_SIZE = 64  # Optimized for RTX 4090
    LEARNING_RATE = 0.001
    EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 50
    
    # Model Settings
    HIDDEN_SIZE = 2048
    NUM_LAYERS = 6
    DROPOUT = 0.1
    
    # Data Settings
    SEQUENCE_LENGTH = 168  # 1 week of hourly data
    PREDICTION_HORIZON = 24  # 24 hours ahead
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Optimization Settings
    GRADIENT_CLIPPING = 1.0
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 10
    
    # Logging and Monitoring
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 50
    
    # Paths
    MODELS_DIR = "models"
    DATA_DIR = "ml_data"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    
    @classmethod
    def setup_gpu_memory(cls):
        """Setup GPU memory allocation"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)
            torch.cuda.empty_cache()
            print(f"GPU memory setup complete. Using {cls.GPU_MEMORY_FRACTION*100}% of available memory.")
    
    @classmethod
    def get_optimizer_config(cls, model):
        """Get optimizer configuration"""
        return {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    
    @classmethod
    def get_scheduler_config(cls, optimizer):
        """Get scheduler configuration"""
        return {
            'mode': 'min',
            'factor': 0.5,
            'patience': cls.SCHEDULER_PATIENCE,
            'verbose': True
        }
EOF

# Create optimized training script
print_header "Creating optimized training script..."
cat > polkadot_ml_training_optimized.py << 'EOF'
#!/usr/bin/env python3
"""
Optimized Polkadot ML Training for RTX 4090
===========================================
High-performance training script optimized for RTX 4090
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import time
from loguru import logger
from ml_config import MLConfig
from polkadot_ml_strategy import PolkadotMLStrategy, MLTaskType

class OptimizedPolkadotTrainer:
    """Optimized trainer for Polkadot ML models on RTX 4090"""
    
    def __init__(self):
        self.config = MLConfig()
        self.config.setup_gpu_memory()
        self.device = self.config.DEVICE
        self.ml_strategy = PolkadotMLStrategy()
        
        # Create directories
        for dir_path in [self.config.MODELS_DIR, self.config.DATA_DIR, 
                        self.config.LOGS_DIR, self.config.RESULTS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def create_optimized_model(self, input_size: int, output_size: int = 1) -> nn.Module:
        """Create optimized neural network model for RTX 4090"""
        
        class OptimizedPolkadotNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super().__init__()
                
                layers = []
                current_size = input_size
                
                for i in range(num_layers):
                    layers.extend([
                        nn.Linear(current_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    current_size = hidden_size
                
                layers.append(nn.Linear(current_size, output_size))
                
                self.network = nn.Sequential(*layers)
                
                # Initialize weights
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                return self.network(x)
        
        return OptimizedPolkadotNet(
            input_size=input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            output_size=output_size,
            dropout=self.config.DROPOUT
        ).to(self.device)
    
    def train_model_optimized(self, X: np.ndarray, y: np.ndarray, task_type: MLTaskType):
        """Train model with RTX 4090 optimizations"""
        logger.info(f"Training optimized model for {task_type.value}")
        
        # Prepare data
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = self.create_optimized_model(X.shape[1])
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), **self.config.get_optimizer_config(model))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.config.get_scheduler_config(optimizer)
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.MIXED_PRECISION else None
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                
                if self.config.MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output.squeeze(), target)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIPPING)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output.squeeze(), target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIPPING)
                    optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Logging
            if epoch % self.config.LOG_INTERVAL == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                if epoch % self.config.SAVE_INTERVAL == 0:
                    self.save_model(model, task_type, epoch, avg_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best loss: {best_loss:.6f}")
        
        return model
    
    def save_model(self, model: nn.Module, task_type: MLTaskType, epoch: int, loss: float):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_type.value}_model_epoch_{epoch}_loss_{loss:.6f}_{timestamp}.pth"
        filepath = Path(self.config.MODELS_DIR) / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'config': self.config.__dict__,
            'task_type': task_type.value,
            'timestamp': timestamp
        }, filepath)
        
        logger.info(f"Model saved: {filepath}")
    
    async def run_optimized_training(self, days_back: int = 90):
        """Run optimized training pipeline"""
        logger.info("Starting optimized training pipeline")
        
        # Collect data
        df = await self.ml_strategy.collect_training_data(days_back)
        if df.empty:
            logger.error("No data collected for training")
            return
        
        # Train models for each task
        for task_type, config in self.ml_strategy.model_configs.items():
            try:
                logger.info(f"Training {task_type.value}")
                
                # Preprocess data
                X, y = self.ml_strategy.preprocess_data(df, config)
                if X is None or y is None:
                    logger.error(f"Failed to preprocess data for {task_type.value}")
                    continue
                
                # Train model
                model = self.train_model_optimized(X, y, task_type)
                
                logger.success(f"Successfully trained {task_type.value}")
                
            except Exception as e:
                logger.error(f"Error training {task_type.value}: {e}")
        
        logger.info("Optimized training pipeline completed")

# Example usage
async def main():
    """Main function"""
    trainer = OptimizedPolkadotTrainer()
    await trainer.run_optimized_training(days_back=90)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
EOF

# Make training script executable
chmod +x polkadot_ml_training_optimized.py

# Create systemd service for automatic training
print_header "Creating systemd service for automatic training..."
sudo tee /etc/systemd/system/polkadot-ml-training.service > /dev/null << 'EOF'
[Unit]
Description=Polkadot ML Training Service
After=network.target postgresql.service

[Service]
Type=simple
User=vovkes
WorkingDirectory=/home/vovkes/ETHL2
Environment=PATH=/home/vovkes/ETHL2/ml_env/bin
ExecStart=/home/vovkes/ETHL2/ml_env/bin/python /home/vovkes/ETHL2/polkadot_ml_training_optimized.py
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring script
print_header "Creating GPU monitoring script..."
cat > monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Monitoring Script for RTX 4090
==================================
Monitors GPU usage, temperature, and memory during training
"""

import subprocess
import time
import json
from datetime import datetime
import psutil
import GPUtil

def get_gpu_info():
    """Get GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # RTX 4090
            return {
                'name': gpu.name,
                'temperature': gpu.temperature,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'gpu_utilization': gpu.load * 100,
                'power_draw': gpu.powerDraw if hasattr(gpu, 'powerDraw') else 'N/A'
            }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def get_system_info():
    """Get system information"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'timestamp': datetime.now().isoformat()
    }

def monitor_gpu(interval=5):
    """Monitor GPU continuously"""
    print("🎯 Starting GPU monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            gpu_info = get_gpu_info()
            system_info = get_system_info()
            
            if gpu_info:
                print(f"\n📊 GPU Status - {system_info['timestamp']}")
                print(f"  GPU: {gpu_info['name']}")
                print(f"  Temperature: {gpu_info['temperature']}°C")
                print(f"  Memory: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_info['memory_percent']:.1f}%)")
                print(f"  Utilization: {gpu_info['gpu_utilization']:.1f}%")
                print(f"  Power: {gpu_info['power_draw']}W")
                print(f"  CPU: {system_info['cpu_percent']:.1f}%")
                print(f"  RAM: {system_info['memory_percent']:.1f}%")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_gpu()
EOF

# Make monitoring script executable
chmod +x monitor_gpu.py

# Install additional monitoring tools
print_header "Installing monitoring tools..."
pip install GPUtil psutil

# Create startup script
print_header "Creating startup script..."
cat > start_ml_training.sh << 'EOF'
#!/bin/bash
# Start ML Training Environment
# ============================

echo "🚀 Starting Polkadot ML Training Environment"
echo "============================================="

# Activate virtual environment
source /home/vovkes/ETHL2/ml_env/bin/activate

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Test GPU
echo "🔍 Testing GPU..."
python test_gpu.py

# Start training
echo "🧠 Starting ML training..."
python polkadot_ml_training_optimized.py

echo "✅ ML training environment started"
EOF

chmod +x start_ml_training.sh

# Create quick start script
print_header "Creating quick start script..."
cat > quick_start_ml.sh << 'EOF'
#!/bin/bash
# Quick Start ML Training
# ======================

echo "⚡ Quick Start ML Training on RTX 4090"
echo "======================================"

# Activate environment
source /home/vovkes/ETHL2/ml_env/bin/activate

# Set CUDA
export CUDA_VISIBLE_DEVICES=0

# Run training
python polkadot_ml_training_optimized.py
EOF

chmod +x quick_start_ml.sh

# Final setup
print_header "Final setup..."
sudo systemctl daemon-reload
sudo systemctl enable polkadot-ml-training.service

# Create logs directory
mkdir -p logs

print_status "Setup completed successfully!"
print_warning "IMPORTANT: Please reboot your system to ensure all drivers are properly loaded."
print_status "After reboot, you can start training with: ./start_ml_training.sh"
print_status "Or use the quick start: ./quick_start_ml.sh"
print_status "Monitor GPU with: python monitor_gpu.py"

echo ""
echo "🎯 RTX 4090 ML Environment Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Reboot your system"
echo "2. Run: ./start_ml_training.sh"
echo "3. Monitor with: python monitor_gpu.py"
echo ""
echo "Happy training! 🚀"
