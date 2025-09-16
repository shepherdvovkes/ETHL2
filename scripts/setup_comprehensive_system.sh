#!/bin/bash

# Comprehensive Polkadot Metrics System - Complete Setup Script
# This script sets up the entire system with PM2

echo "🚀 Comprehensive Polkadot Metrics System - Complete Setup"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Consider running as a regular user."
fi

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python3
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi
print_status "Python3 is installed: $(python3 --version)"

# Check pip3
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3 first."
    exit 1
fi
print_status "pip3 is installed: $(pip3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js is not installed. Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    if [ $? -eq 0 ]; then
        print_status "Node.js installed: $(node --version)"
    else
        print_error "Failed to install Node.js"
        exit 1
    fi
else
    print_status "Node.js is installed: $(node --version)"
fi

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi
print_status "npm is installed: $(npm --version)"

# Install PM2 if not already installed
if ! command -v pm2 &> /dev/null; then
    print_info "Installing PM2..."
    npm install -g pm2
    if [ $? -eq 0 ]; then
        print_status "PM2 installed: $(pm2 --version)"
    else
        print_error "Failed to install PM2"
        exit 1
    fi
else
    print_status "PM2 is already installed: $(pm2 --version)"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p src/database
mkdir -p src/api
mkdir -p src/config
print_status "Directories created"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_status "Python dependencies installed"
    else
        print_warning "Some Python dependencies may have failed to install"
    fi
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip3 install fastapi uvicorn sqlalchemy psycopg2-binary aiohttp loguru
fi

# Setup PM2 startup
echo ""
echo "🚀 Setting up PM2 startup..."
pm2 startup
print_info "Please run the command shown above to enable PM2 startup on boot"

# Setup database
echo ""
echo "🗄️  Setting up database..."
if [ -f "setup_comprehensive_polkadot_metrics.py" ]; then
    python3 setup_comprehensive_polkadot_metrics.py
    if [ $? -eq 0 ]; then
        print_status "Database setup completed"
    else
        print_warning "Database setup may have issues. Check logs for details."
    fi
else
    print_warning "Database setup script not found. Please run setup manually."
fi

# Test the system
echo ""
echo "🧪 Testing the system..."
if [ -f "test_comprehensive_metrics.py" ]; then
    python3 test_comprehensive_metrics.py
    if [ $? -eq 0 ]; then
        print_status "System tests passed"
    else
        print_warning "Some tests failed. Check logs for details."
    fi
else
    print_warning "Test script not found. Skipping tests."
fi

# Start services with PM2
echo ""
echo "🚀 Starting services with PM2..."
./start_comprehensive_services.sh

# Show final status
echo ""
echo "📊 Final Status:"
echo "================"
pm2 status

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🌐 Available Services:"
echo "   • Comprehensive Polkadot Metrics: http://localhost:8008"
echo "   • Legacy Polkadot Metrics:        http://localhost:8007"
echo "   • Avalanche Metrics:              http://localhost:8006"
echo "   • L2 Monitoring Dashboard:        http://localhost:8005"
echo ""
echo "📋 Management Commands:"
echo "   • ./start_comprehensive_services.sh   - Start all services"
echo "   • ./stop_comprehensive_services.sh    - Stop all services"
echo "   • ./restart_comprehensive_services.sh - Restart all services"
echo "   • ./pm2_monitor.sh                    - Monitor services"
echo "   • pm2 status                          - Show service status"
echo "   • pm2 logs                            - Show all logs"
echo "   • pm2 monit                           - Open monitoring interface"
echo ""
echo "📁 Log Files:"
echo "   • logs/polkadot-comprehensive-metrics.log"
echo "   • logs/polkadot-data-collector.log"
echo "   • logs/avalanche-metrics.log"
echo "   • logs/l2-monitoring-dashboard.log"
echo ""
echo "🔧 Troubleshooting:"
echo "   • Check logs: pm2 logs"
echo "   • Restart services: pm2 restart all"
echo "   • Monitor resources: pm2 monit"
echo "   • Check system status: ./pm2_monitor.sh status"
echo ""
echo "========================================================="
echo "✅ Comprehensive Polkadot Metrics System is ready!"
echo "========================================================="
