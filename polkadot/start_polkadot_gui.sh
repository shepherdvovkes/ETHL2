#!/bin/bash
"""
Polkadot GUI Dashboard Startup Script
====================================

Starts the Streamlit GUI dashboard for monitoring parachain activity and fraud detection.
"""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export HF_TOKEN=your_huggingface_token_here

# Activate virtual environment
source /home/vovkes/ETHL2/polkadot_latent_env/bin/activate

# Set working directory
cd /home/vovkes/ETHL2

# Start the Streamlit dashboard
echo "🚀 Starting Polkadot GUI Dashboard..."
echo "📊 Real-time parachain activity monitoring"
echo "🔍 Fraud detection with ML predictions"
echo "🎯 Risk assessment and anomaly detection"
echo "🔗 Access at: http://localhost:8501"
echo ""

streamlit run polkadot_gui_dashboard.py --server.port 8501 --server.address 0.0.0.0
