#!/bin/bash

# Comprehensive Polkadot Metrics System - PM2 Stop Script
# This script stops all services using PM2

echo "🛑 Stopping Comprehensive Polkadot Metrics System"
echo "=================================================="

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed."
    exit 1
fi

# Stop all PM2 processes
echo "🛑 Stopping all PM2 processes..."
pm2 stop all

# Show status
echo ""
echo "📊 Service Status:"
pm2 status

echo ""
echo "✅ All services stopped successfully!"
echo "=================================================="
echo ""
echo "💡 To start services again, run:"
echo "   ./start_comprehensive_services.sh"
echo ""
echo "💡 To restart services, run:"
echo "   ./restart_comprehensive_services.sh"
