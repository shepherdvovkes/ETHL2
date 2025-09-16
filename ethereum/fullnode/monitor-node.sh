#!/bin/bash

# Ethereum Archive Node Monitoring Script

echo "🔍 Ethereum Archive Node Status Monitor"
echo "========================================"

# Check if containers are running
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "⛏️  Geth Status:"
echo "---------------"

# Check Geth sync status
if docker-compose exec -T geth geth attach --exec "eth.syncing" 2>/dev/null | grep -q "false"; then
    echo "✅ Geth is fully synced"
    CURRENT_BLOCK=$(docker-compose exec -T geth geth attach --exec "eth.blockNumber" 2>/dev/null | tr -d '"')
    echo "📦 Current block: $CURRENT_BLOCK"
else
    echo "🔄 Geth is syncing..."
    SYNC_STATUS=$(docker-compose exec -T geth geth attach --exec "eth.syncing" 2>/dev/null)
    echo "📈 Sync status: $SYNC_STATUS"
fi

# Check Geth peer count
PEER_COUNT=$(docker-compose exec -T geth geth attach --exec "net.peerCount" 2>/dev/null | tr -d '"')
echo "👥 Connected peers: $PEER_COUNT"

echo ""
echo "🏮 Lighthouse Status:"
echo "--------------------"

# Check Lighthouse sync status
LIGHTHOUSE_SYNC=$(docker-compose exec -T lighthouse lighthouse beacon sync_status 2>/dev/null | head -5)
echo "📈 Lighthouse sync:"
echo "$LIGHTHOUSE_SYNC"

echo ""
echo "💾 Disk Usage:"
echo "-------------"
df -h /mnt/ethereum-data | tail -1

echo ""
echo "📊 Resource Usage:"
echo "-----------------"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

echo ""
echo "🔗 Network Connections:"
echo "----------------------"
netstat -tuln | grep -E "(8545|8546|9000|5052|5053)" | head -10

echo ""
echo "📝 Recent Logs (last 10 lines):"
echo "------------------------------"
echo "Geth:"
docker-compose logs --tail=5 geth

echo ""
echo "Lighthouse:"
docker-compose logs --tail=5 lighthouse
