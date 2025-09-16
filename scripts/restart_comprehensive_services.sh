#!/bin/bash

# Comprehensive Polkadot Metrics System - PM2 Restart Script
# This script restarts all services using PM2

echo "🔄 Restarting Comprehensive Polkadot Metrics System"
echo "==================================================="

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed. Please install PM2 first:"
    echo "   npm install -g pm2"
    exit 1
fi

# Restart all PM2 processes
echo "🔄 Restarting all PM2 processes..."
pm2 restart all

# Show status
echo ""
echo "📊 Service Status:"
pm2 status

echo ""
echo "✅ All services restarted successfully!"
echo "==================================================="
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
echo "   • pm2 monit                     - Monitor services"
