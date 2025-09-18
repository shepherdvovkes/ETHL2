#!/bin/bash

# Bitcoin Ultrafast Archive Node Stop Script
# Safely stops all Bitcoin services

set -e

echo "🛑 Stopping Bitcoin Ultrafast Archive Node..."

# Configuration
BITCOIN_DIR="/home/vovkes/ETHL2/bitcoin"

# Check if Docker Compose file exists
if [ ! -f "$BITCOIN_DIR/docker-compose.yml" ]; then
    echo "❌ Docker Compose file not found at $BITCOIN_DIR/docker-compose.yml"
    exit 1
fi

# Stop all services
echo "🛑 Stopping all Bitcoin services..."
cd "$BITCOIN_DIR"
docker-compose down

# Wait for graceful shutdown
echo "⏳ Waiting for graceful shutdown..."
sleep 5

# Check if containers are stopped
echo "🔍 Checking container status..."
if docker-compose ps | grep -q "Up"; then
    echo "⚠️  Some containers are still running, forcing stop..."
    docker-compose kill
    docker-compose down --remove-orphans
fi

# Clean up any orphaned containers
echo "🧹 Cleaning up orphaned containers..."
docker-compose down --remove-orphans

# Display final status
echo ""
echo "✅ Bitcoin Ultrafast Archive Node stopped successfully!"
echo ""
echo "📊 Final Status:"
docker-compose ps

echo ""
echo "💡 To start the node again, run:"
echo "   $BITCOIN_DIR/scripts/start_bitcoin_node.sh"
echo ""
echo "💾 Your blockchain data is preserved in:"
echo "   $BITCOIN_DIR/data"
echo ""

