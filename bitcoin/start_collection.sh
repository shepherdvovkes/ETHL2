#!/bin/bash

# Bitcoin Chain Collector Startup Script
# Starts the Bitcoin blockchain collection with 10 workers

echo "🚀 Starting Bitcoin Chain Collector"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "bitcoin_chain_collector.py" ]; then
    echo "❌ Error: Please run this script from the bitcoin directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed"
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing requirements..."
pip install -r collector_requirements.txt

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🗄️  Database will be created at: bitcoin_chain.db"
echo "📊 Logs will be written to: bitcoin_collector.log"
echo "👥 Using 10 workers for parallel collection"
echo "🌐 QuickNode endpoint configured"
echo ""

# Ask for confirmation
read -p "Do you want to start the collection? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Collection cancelled"
    exit 1
fi

echo "🚀 Starting collection..."
echo "Press Ctrl+C to stop the collection"
echo ""

# Start the collector
python3 run_collector.py

echo ""
echo "✅ Collection process completed"
