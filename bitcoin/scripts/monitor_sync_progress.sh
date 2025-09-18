#!/bin/bash

# Bitcoin Sync Progress Monitor
# Real-time monitoring of blockchain synchronization progress

set -e

echo "📊 Bitcoin Sync Progress Monitor"
echo "================================"

# Configuration
BITCOIN_DIR="/home/vovkes/ETHL2/bitcoin"
CONTAINER_NAME="bitcoin-archive-node"

# Function to get blockchain info
get_blockchain_info() {
    docker exec "$CONTAINER_NAME" bitcoin-cli getblockchaininfo 2>/dev/null || echo "{}"
}

# Function to get network info
get_network_info() {
    docker exec "$CONTAINER_NAME" bitcoin-cli getnetworkinfo 2>/dev/null || echo "{}"
}

# Function to get mempool info
get_mempool_info() {
    docker exec "$CONTAINER_NAME" bitcoin-cli getmempoolinfo 2>/dev/null || echo "{}"
}

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [ "$bytes" -gt 1073741824 ]; then
        echo "$(($bytes / 1073741824))GB"
    elif [ "$bytes" -gt 1048576 ]; then
        echo "$(($bytes / 1048576))MB"
    elif [ "$bytes" -gt 1024 ]; then
        echo "$(($bytes / 1024))KB"
    else
        echo "${bytes}B"
    fi
}

# Function to calculate sync percentage
calculate_sync_percentage() {
    local blocks=$1
    local headers=$2
    if [ "$headers" -gt 0 ]; then
        echo "scale=2; $blocks * 100 / $headers" | bc -l
    else
        echo "0"
    fi
}

# Function to estimate time remaining
estimate_time_remaining() {
    local blocks=$1
    local headers=$2
    local start_time=$3
    
    if [ "$blocks" -gt 0 ] && [ "$headers" -gt 0 ]; then
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining_blocks=$((headers - blocks))
        
        if [ "$blocks" -gt 0 ]; then
            local blocks_per_second=$(echo "scale=2; $blocks / $elapsed" | bc -l)
            local seconds_remaining=$(echo "scale=0; $remaining_blocks / $blocks_per_second" | bc -l)
            
            # Convert to human readable format
            local days=$((seconds_remaining / 86400))
            local hours=$(((seconds_remaining % 86400) / 3600))
            local minutes=$(((seconds_remaining % 3600) / 60))
            
            if [ "$days" -gt 0 ]; then
                echo "${days}d ${hours}h ${minutes}m"
            elif [ "$hours" -gt 0 ]; then
                echo "${hours}h ${minutes}m"
            else
                echo "${minutes}m"
            fi
        fi
    else
        echo "Calculating..."
    fi
}

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Bitcoin container is not running!"
    echo "   Start it with: $BITCOIN_DIR/scripts/start_bitcoin_node.sh"
    exit 1
fi

echo "🔄 Starting real-time sync monitoring..."
echo "   Press Ctrl+C to stop"
echo ""

# Initialize tracking variables
start_time=$(date +%s)
last_blocks=0
last_time=$start_time

# Main monitoring loop
while true; do
    # Get blockchain information
    blockchain_info=$(get_blockchain_info)
    network_info=$(get_network_info)
    mempool_info=$(get_mempool_info)
    
    # Extract key metrics
    blocks=$(echo "$blockchain_info" | jq -r '.blocks // 0')
    headers=$(echo "$blockchain_info" | jq -r '.headers // 0')
    verification_progress=$(echo "$blockchain_info" | jq -r '.verificationprogress // 0')
    chain=$(echo "$blockchain_info" | jq -r '.chain // "unknown"')
    initial_block_download=$(echo "$blockchain_info" | jq -r '.initialblockdownload // true')
    
    connections=$(echo "$network_info" | jq -r '.connections // 0')
    version=$(echo "$network_info" | jq -r '.version // 0')
    
    mempool_size=$(echo "$mempool_info" | jq -r '.size // 0')
    mempool_bytes=$(echo "$mempool_info" | jq -r '.bytes // 0')
    
    # Calculate sync percentage
    sync_percentage=$(calculate_sync_percentage "$blocks" "$headers")
    
    # Calculate sync speed
    current_time=$(date +%s)
    time_diff=$((current_time - last_time))
    blocks_diff=$((blocks - last_blocks))
    
    if [ "$time_diff" -gt 0 ] && [ "$blocks_diff" -gt 0 ]; then
        sync_speed=$(echo "scale=2; $blocks_diff / $time_diff" | bc -l)
    else
        sync_speed="0"
    fi
    
    # Estimate time remaining
    time_remaining=$(estimate_time_remaining "$blocks" "$headers" "$start_time")
    
    # Clear screen and display status
    clear
    echo "📊 Bitcoin Sync Progress Monitor - $(date)"
    echo "=============================================="
    echo ""
    echo "🔗 Network: $chain"
    echo "📦 Blocks: $blocks / $headers ($sync_percentage%)"
    echo "✅ Verification Progress: $(echo "scale=2; $verification_progress * 100" | bc -l)%"
    echo "🔄 Initial Block Download: $initial_block_download"
    echo ""
    echo "🌐 Network Status:"
    echo "   Connections: $connections"
    echo "   Version: $version"
    echo ""
    echo "⚡ Sync Performance:"
    echo "   Speed: ${sync_speed} blocks/sec"
    echo "   Estimated Time Remaining: $time_remaining"
    echo ""
    echo "💾 Mempool:"
    echo "   Transactions: $mempool_size"
    echo "   Size: $(format_bytes $mempool_bytes)"
    echo ""
    
    # Check if sync is complete
    if [ "$initial_block_download" = "false" ]; then
        echo "🎉 SYNC COMPLETE! Your Bitcoin node is fully synchronized."
        echo "   You can now use it for transactions and queries."
        break
    fi
    
    # Update tracking variables
    last_blocks=$blocks
    last_time=$current_time
    
    # Wait before next update
    sleep 10
done

echo ""
echo "✨ Monitoring complete!"

