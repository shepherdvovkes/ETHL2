#!/bin/bash

# Setup PM2 Environment for Comprehensive Polkadot Metrics System
# This script sets up the PM2 environment and installs dependencies

echo "🔧 Setting up PM2 Environment for Comprehensive Polkadot Metrics"
echo "==============================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -"
    echo "   sudo apt-get install -y nodejs"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Install PM2 globally if not already installed
if ! command -v pm2 &> /dev/null; then
    echo "📦 Installing PM2 globally..."
    npm install -g pm2
    if [ $? -eq 0 ]; then
        echo "✅ PM2 installed successfully"
    else
        echo "❌ Failed to install PM2"
        exit 1
    fi
else
    echo "✅ PM2 is already installed"
fi

# Check PM2 version
echo "📋 PM2 Version:"
pm2 --version

# Setup PM2 startup script
echo "🚀 Setting up PM2 startup script..."
pm2 startup
echo "💡 Please run the command shown above to enable PM2 startup on boot"

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs

# Set proper permissions
echo "🔐 Setting proper permissions..."
chmod +x start_comprehensive_services.sh
chmod +x stop_comprehensive_services.sh
chmod +x restart_comprehensive_services.sh

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Please ensure Python dependencies are installed."
fi

# Test PM2 functionality
echo "🧪 Testing PM2 functionality..."
pm2 list

echo ""
echo "✅ PM2 Environment Setup Complete!"
echo "==============================================================="
echo ""
echo "🚀 Next Steps:"
echo "   1. Run setup: python3 setup_comprehensive_polkadot_metrics.py"
echo "   2. Start services: ./start_comprehensive_services.sh"
echo "   3. Monitor services: pm2 monit"
echo ""
echo "📋 Available Commands:"
echo "   • ./start_comprehensive_services.sh   - Start all services"
echo "   • ./stop_comprehensive_services.sh    - Stop all services"
echo "   • ./restart_comprehensive_services.sh - Restart all services"
echo "   • pm2 status                          - Show service status"
echo "   • pm2 logs                            - Show all logs"
echo "   • pm2 monit                           - Monitor services"
echo ""
echo "🌐 Service Ports:"
echo "   • Comprehensive Polkadot Metrics: 8008"
echo "   • Legacy Polkadot Metrics:        8007"
echo "   • Avalanche Metrics:              8006"
echo "   • L2 Monitoring Dashboard:        8005"
echo "==============================================================="
