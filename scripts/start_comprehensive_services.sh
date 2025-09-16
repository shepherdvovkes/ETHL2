#!/bin/bash

# Comprehensive Polkadot Metrics System - PM2 Startup Script
# This script starts all services using PM2

echo "🚀 Starting Comprehensive Polkadot Metrics System with PM2"
echo "=========================================================="

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed. Please install PM2 first:"
    echo "   npm install -g pm2"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Stop any existing PM2 processes
echo "🛑 Stopping existing PM2 processes..."
pm2 stop all
pm2 delete all

# Start services using PM2
echo "🚀 Starting services with PM2..."

# Start comprehensive Polkadot metrics server
echo "   📊 Starting Comprehensive Polkadot Metrics Server (Port 8008)..."
pm2 start ecosystem.comprehensive.config.js --only polkadot-comprehensive-metrics

# Start data collector
echo "   📈 Starting Data Collector..."
pm2 start ecosystem.comprehensive.config.js --only polkadot-data-collector

# Start legacy Polkadot metrics server
echo "   📊 Starting Legacy Polkadot Metrics Server (Port 8007)..."
pm2 start ecosystem.comprehensive.config.js --only polkadot-legacy-metrics

# Start Avalanche metrics server
echo "   🏔️  Starting Avalanche Metrics Server (Port 8006)..."
pm2 start ecosystem.comprehensive.config.js --only avalanche-metrics

# Start L2 monitoring dashboard
echo "   🔗 Starting L2 Monitoring Dashboard (Port 8005)..."
pm2 start ecosystem.comprehensive.config.js --only l2-monitoring-dashboard

# Save PM2 configuration
echo "💾 Saving PM2 configuration..."
pm2 save

# Show status
echo ""
echo "✅ All services started successfully!"
echo "=========================================================="
echo "📊 Service Status:"
pm2 status

echo ""
echo "🌐 Available Dashboards:"
echo "   • Comprehensive Polkadot Metrics: http://localhost:8008"
echo "   • Legacy Polkadot Metrics:        http://localhost:8007"
echo "   • Avalanche Metrics:              http://localhost:8006"
echo "   • L2 Monitoring Dashboard:        http://localhost:8005"

echo ""
echo "📋 Useful PM2 Commands:"
echo "   • pm2 status                    - Show service status"
echo "   • pm2 logs                      - Show all logs"
echo "   • pm2 logs polkadot-comprehensive-metrics - Show specific service logs"
echo "   • pm2 restart all               - Restart all services"
echo "   • pm2 stop all                  - Stop all services"
echo "   • pm2 monit                     - Monitor services"

echo ""
echo "🔧 Management Commands:"
echo "   • ./stop_comprehensive_services.sh  - Stop all services"
echo "   • ./restart_comprehensive_services.sh - Restart all services"
echo "   • ./setup_comprehensive_polkadot_metrics.py - Setup database"

echo ""
echo "=========================================================="
echo "🎉 Comprehensive Polkadot Metrics System is now running!"
echo "=========================================================="
