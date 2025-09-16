#!/bin/bash

# RTX 4090 Performance Benchmark Script
# This script helps you test your GPU performance with various tools

echo "🚀 RTX 4090 Performance Benchmark Suite 🚀"
echo "=========================================="
echo ""

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    echo "Run: sudo apt install nvidia-driver-535 (or latest version)"
    exit 1
fi

echo "🔍 GPU Information:"
echo "==================="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits
echo ""

echo "📊 Performance Tests Available:"
echo "==============================="
echo "1. GPU Stress Test (nvidia-smi monitoring)"
echo "2. SuperTuxKart Performance Test"
echo "3. FlightGear Performance Test"
echo "4. Minetest with Shaders Test"
echo "5. System Information"
echo ""

read -p "Enter test number (1-5) or 'q' to quit: " choice

case $choice in
    1)
        echo "🔥 Starting GPU Stress Test..."
        echo "Launching SuperTuxKart in background for stress testing..."
        echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
        echo "Press Ctrl+C to stop monitoring"
        supertuxkart &
        sleep 2
        watch -n 1 nvidia-smi
        ;;
    2)
        echo "🏎️  SuperTuxKart Performance Test..."
        echo "Launching with maximum settings..."
        supertuxkart
        ;;
    3)
        echo "✈️  FlightGear Performance Test..."
        echo "Launching with high detail settings..."
        flightgear
        ;;
    4)
        echo "🧱 Minetest Performance Test..."
        echo "Launching with shader support..."
        minetest
        ;;
    5)
        echo "💻 System Information:"
        echo "======================"
        echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
        echo "RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
        echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
        echo "CUDA: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits)"
        ;;
    q|Q)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac
