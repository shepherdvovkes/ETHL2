#!/bin/bash

# Ethereum Archive Node Startup Script
# This script starts Geth and Lighthouse in archive mode

set -e

echo "🚀 Starting Ethereum Archive Node..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if the data drive is mounted
if [ ! -d "/mnt/ethereum-data/ethereum" ]; then
    echo "❌ Ethereum data drive not mounted. Mounting now..."
    sudo mount /dev/sda /mnt/ethereum-data
fi

# Check if JWT secret exists
if [ ! -f "/mnt/ethereum-data/ethereum/geth-data/geth/jwtsecret" ]; then
    echo "🔑 Creating JWT secret..."
    openssl rand -hex 32 > /mnt/ethereum-data/ethereum/geth-data/geth/jwtsecret
    chmod 600 /mnt/ethereum-data/ethereum/geth-data/geth/jwtsecret
fi

# Set proper permissions
echo "🔧 Setting permissions..."
sudo chown -R $USER:$USER /mnt/ethereum-data/ethereum/
sudo chmod -R 755 /mnt/ethereum-data/ethereum/

# Create log directories
mkdir -p /home/vovkes/ETHL2/ethereum/fullnode/logs

# Start the services
echo "🐳 Starting Docker containers..."

# Start Geth first
echo "⛏️  Starting Geth archive node..."
docker-compose up -d geth

# Wait for Geth to be ready
echo "⏳ Waiting for Geth to initialize..."
sleep 30

# Check Geth health
echo "🔍 Checking Geth status..."
docker-compose logs geth | tail -20

# Start Lighthouse
echo "🏮 Starting Lighthouse beacon node..."
docker-compose up -d lighthouse

# Wait for Lighthouse to be ready
echo "⏳ Waiting for Lighthouse to initialize..."
sleep 60

# Check Lighthouse health
echo "🔍 Checking Lighthouse status..."
docker-compose logs lighthouse | tail -20

# Show status
echo "📊 Ethereum Archive Node Status:"
echo "=================================="
docker-compose ps

echo ""
echo "🌐 Access URLs:"
echo "  Geth HTTP RPC: http://localhost:8545"
echo "  Geth WebSocket: ws://localhost:8546"
echo "  Lighthouse API: http://localhost:5052"
echo "  Prometheus: http://localhost:9090 (if monitoring enabled)"
echo "  Grafana: http://localhost:3000 (if monitoring enabled)"
echo ""

echo "📝 Useful commands:"
echo "  View logs: docker-compose logs -f [service]"
echo "  Stop services: docker-compose down"
echo "  Restart: docker-compose restart [service]"
echo "  Monitor sync: docker-compose exec geth geth attach --exec eth.syncing"
echo ""

echo "✅ Ethereum Archive Node started successfully!"
echo "🔗 The node will continue syncing in the background."
echo "📈 Monitor the sync progress with: docker-compose logs -f geth"
