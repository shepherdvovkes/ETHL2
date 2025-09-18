#!/bin/bash

# Bitcoin Node Health Check Script
# Comprehensive health monitoring for Bitcoin archive node

set -e

echo "🏥 Bitcoin Node Health Check"
echo "============================"

# Configuration
BITCOIN_DIR="/home/vovkes/ETHL2/bitcoin"
CONTAINER_NAME="bitcoin-archive-node"
RPC_URL="http://bitcoin:ultrafast_archive_node_2024@localhost:8332"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Function to check if container is running
check_container_status() {
    echo "🔍 Checking container status..."
    
    if docker ps | grep -q "$CONTAINER_NAME"; then
        print_status "OK" "Bitcoin container is running"
        
        # Get container stats
        container_stats=$(docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" "$CONTAINER_NAME" | tail -n 1)
        print_status "INFO" "Container stats: $container_stats"
    else
        print_status "ERROR" "Bitcoin container is not running"
        return 1
    fi
}

# Function to check RPC connectivity
check_rpc_connectivity() {
    echo ""
    echo "🔍 Checking RPC connectivity..."
    
    # Test basic RPC call
    if docker exec "$CONTAINER_NAME" bitcoin-cli -conf=/home/bitcoin/.bitcoin/bitcoin.conf getblockchaininfo >/dev/null 2>&1; then
        print_status "OK" "RPC is responding"
    else
        print_status "ERROR" "RPC is not responding"
        return 1
    fi
}

# Function to check blockchain status
check_blockchain_status() {
    echo ""
    echo "🔍 Checking blockchain status..."
    
    blockchain_info=$(docker exec "$CONTAINER_NAME" bitcoin-cli -conf=/home/bitcoin/.bitcoin/bitcoin.conf getblockchaininfo 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        blocks=$(echo "$blockchain_info" | jq -r '.blocks // 0')
        headers=$(echo "$blockchain_info" | jq -r '.headers // 0')
        verification_progress=$(echo "$blockchain_info" | jq -r '.verificationprogress // 0')
        initial_block_download=$(echo "$blockchain_info" | jq -r '.initialblockdownload // true')
        chain=$(echo "$blockchain_info" | jq -r '.chain // "unknown"')
        
        print_status "INFO" "Chain: $chain"
        print_status "INFO" "Blocks: $blocks / $headers"
        print_status "INFO" "Verification Progress: $(echo "scale=2; $verification_progress * 100" | bc -l)%"
        
        if [ "$initial_block_download" = "false" ]; then
            print_status "OK" "Node is fully synchronized"
        else
            print_status "WARNING" "Node is still syncing (Initial Block Download: $initial_block_download)"
        fi
    else
        print_status "ERROR" "Failed to get blockchain info"
    fi
}

# Function to check network connectivity
check_network_status() {
    echo ""
    echo "🔍 Checking network status..."
    
    network_info=$(docker exec "$CONTAINER_NAME" bitcoin-cli -conf=/home/bitcoin/.bitcoin/bitcoin.conf getnetworkinfo 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        connections=$(echo "$network_info" | jq -r '.connections // 0')
        version=$(echo "$network_info" | jq -r '.version // 0')
        subversion=$(echo "$network_info" | jq -r '.subversion // "unknown"')
        
        print_status "INFO" "Version: $subversion"
        print_status "INFO" "Connections: $connections"
        
        if [ "$connections" -gt 0 ]; then
            print_status "OK" "Node has active connections"
        else
            print_status "WARNING" "Node has no connections"
        fi
    else
        print_status "ERROR" "Failed to get network info"
    fi
}

# Function to check mempool status
check_mempool_status() {
    echo ""
    echo "🔍 Checking mempool status..."
    
    mempool_info=$(docker exec "$CONTAINER_NAME" bitcoin-cli -conf=/home/bitcoin/.bitcoin/bitcoin.conf getmempoolinfo 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        mempool_size=$(echo "$mempool_info" | jq -r '.size // 0')
        mempool_bytes=$(echo "$mempool_info" | jq -r '.bytes // 0')
        
        print_status "INFO" "Mempool transactions: $mempool_size"
        print_status "INFO" "Mempool size: $(($mempool_bytes / 1024 / 1024))MB"
        
        if [ "$mempool_size" -gt 0 ]; then
            print_status "OK" "Mempool is active"
        else
            print_status "INFO" "Mempool is empty"
        fi
    else
        print_status "ERROR" "Failed to get mempool info"
    fi
}

# Function to check disk usage
check_disk_usage() {
    echo ""
    echo "🔍 Checking disk usage..."
    
    data_dir="$BITCOIN_DIR/data"
    if [ -d "$data_dir" ]; then
        disk_usage=$(df -h "$data_dir" | awk 'NR==2 {print $5}' | sed 's/%//')
        available_space=$(df -h "$data_dir" | awk 'NR==2 {print $4}')
        
        print_status "INFO" "Data directory: $data_dir"
        print_status "INFO" "Available space: $available_space"
        
        if [ "$disk_usage" -lt 80 ]; then
            print_status "OK" "Disk usage: ${disk_usage}%"
        elif [ "$disk_usage" -lt 90 ]; then
            print_status "WARNING" "Disk usage: ${disk_usage}% (getting full)"
        else
            print_status "ERROR" "Disk usage: ${disk_usage}% (critical)"
        fi
    else
        print_status "ERROR" "Data directory not found: $data_dir"
    fi
}

# Function to check log files
check_log_files() {
    echo ""
    echo "🔍 Checking log files..."
    
    logs_dir="$BITCOIN_DIR/logs"
    if [ -d "$logs_dir" ]; then
        log_count=$(find "$logs_dir" -name "*.log" | wc -l)
        print_status "INFO" "Log files found: $log_count"
        
        # Check for recent errors
        if [ -f "$logs_dir/bitcoin_monitor.log" ]; then
            error_count=$(grep -c "ERROR" "$logs_dir/bitcoin_monitor.log" 2>/dev/null || echo "0")
            if [ "$error_count" -gt 0 ]; then
                print_status "WARNING" "Found $error_count errors in monitor log"
            else
                print_status "OK" "No errors in monitor log"
            fi
        fi
    else
        print_status "WARNING" "Logs directory not found: $logs_dir"
    fi
}

# Function to check system resources
check_system_resources() {
    echo ""
    echo "🔍 Checking system resources..."
    
    # Check memory usage
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    print_status "INFO" "Memory usage: ${memory_usage}%"
    
    # Check CPU load
    cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    print_status "INFO" "CPU load: $cpu_load"
    
    # Check if system is under stress
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        print_status "WARNING" "High memory usage detected"
    fi
    
    if (( $(echo "$cpu_load > 4" | bc -l) )); then
        print_status "WARNING" "High CPU load detected"
    fi
}

# Main health check function
main() {
    echo "Starting comprehensive health check..."
    echo ""
    
    # Run all checks
    check_container_status
    check_rpc_connectivity
    check_blockchain_status
    check_network_status
    check_mempool_status
    check_disk_usage
    check_log_files
    check_system_resources
    
    echo ""
    echo "🏥 Health check complete!"
    echo ""
    echo "💡 For detailed monitoring, run:"
    echo "   $BITCOIN_DIR/scripts/monitor_sync_progress.sh"
    echo ""
    echo "📊 For real-time metrics, check:"
    echo "   $BITCOIN_DIR/logs/bitcoin_metrics.json"
    echo ""
}

# Run main function
main

