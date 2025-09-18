#!/bin/bash

# Bitcoin Ultrafast Archive Node Startup Script
# Optimized for maximum performance and reliability

set -e

echo "🚀 Starting Bitcoin Ultrafast Archive Node..."

# Configuration
BITCOIN_DIR="/home/vovkes/ETHL2/bitcoin"
DATA_DIR="$BITCOIN_DIR/data"
LOGS_DIR="$BITCOIN_DIR/logs"
INDEXER_DIR="$BITCOIN_DIR/indexer_data"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$DATA_DIR" "$LOGS_DIR" "$INDEXER_DIR"

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 "$DATA_DIR" "$LOGS_DIR" "$INDEXER_DIR"

# Check system requirements
echo "🔍 Checking system requirements..."

# Check available disk space (need at least 500GB for full archive)
AVAILABLE_SPACE=$(df "$DATA_DIR" | awk 'NR==2 {print $4}')
REQUIRED_SPACE=$((500 * 1024 * 1024)) # 500GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "⚠️  Warning: Available disk space ($(($AVAILABLE_SPACE / 1024 / 1024))GB) is less than recommended 500GB"
    echo "   The node may run out of space during initial sync"
fi

# Check memory (need at least 8GB)
TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
if [ "$TOTAL_MEM" -lt 8192 ]; then
    echo "⚠️  Warning: System has less than 8GB RAM ($TOTAL_MEM MB)"
    echo "   Performance may be suboptimal"
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Pull latest Bitcoin Core image
echo "📥 Pulling latest Bitcoin Core image..."
docker pull bitcoin/bitcoin:latest

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f "$BITCOIN_DIR/docker-compose.yml" down

# Start the Bitcoin node
echo "🚀 Starting Bitcoin archive node..."
cd "$BITCOIN_DIR"
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if Bitcoin Core is running
echo "🔍 Checking Bitcoin Core status..."
if docker-compose ps | grep -q "bitcoin-archive-node.*Up"; then
    echo "✅ Bitcoin Core is running"
else
    echo "❌ Bitcoin Core failed to start"
    docker-compose logs bitcoin-core
    exit 1
fi

# Check if indexer is running
echo "🔍 Checking Bitcoin indexer status..."
if docker-compose ps | grep -q "bitcoin-indexer.*Up"; then
    echo "✅ Bitcoin indexer is running"
else
    echo "⚠️  Bitcoin indexer is not running (this is optional)"
fi

# Check if monitor is running
echo "🔍 Checking Bitcoin monitor status..."
if docker-compose ps | grep -q "bitcoin-monitor.*Up"; then
    echo "✅ Bitcoin monitor is running"
else
    echo "⚠️  Bitcoin monitor is not running"
fi

# Display connection information
echo ""
echo "🎉 Bitcoin Ultrafast Archive Node is running!"
echo ""
echo "📊 Connection Information:"
echo "   RPC URL: http://localhost:8332"
echo "   RPC User: bitcoin"
echo "   RPC Password: ultrafast_archive_node_2024"
echo "   P2P Port: 8333"
echo ""
echo "📈 Monitoring:"
echo "   Logs: $LOGS_DIR"
echo "   Metrics: $LOGS_DIR/bitcoin_metrics.json"
echo ""
echo "🔧 Management Commands:"
echo "   View logs: docker-compose -f $BITCOIN_DIR/docker-compose.yml logs -f"
echo "   Stop node: docker-compose -f $BITCOIN_DIR/docker-compose.yml down"
echo "   Restart: docker-compose -f $BITCOIN_DIR/docker-compose.yml restart"
echo ""
echo "📋 Next Steps:"
echo "   1. Wait for initial blockchain sync (this may take several days)"
echo "   2. Monitor progress with: docker-compose logs -f bitcoin-core"
echo "   3. Check sync status with: docker exec bitcoin-archive-node bitcoin-cli getblockchaininfo"
echo ""

# Show initial sync status
echo "🔍 Initial sync status:"
docker exec bitcoin-archive-node bitcoin-cli getblockchaininfo | jq '.blocks, .headers, .verificationprogress' 2>/dev/null || echo "   RPC not ready yet, please wait..."

echo ""
echo "✨ Setup complete! Your Bitcoin archive node is now running."

