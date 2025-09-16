#!/bin/bash

# Solana System Startup Script
# Starts the complete Solana data collection and metrics system

echo "🚀 Starting Solana System..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import aiohttp, sqlite3, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Installing..."
    pip3 install aiohttp
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to start a service in background
start_service() {
    local service_name=$1
    local script_name=$2
    local log_file="logs/${service_name}.log"
    
    echo "🔄 Starting $service_name..."
    nohup python3 $script_name > $log_file 2>&1 &
    local pid=$!
    echo $pid > "logs/${service_name}.pid"
    echo "✅ $service_name started with PID $pid"
    sleep 2
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ $service_name is running (PID: $pid)"
            return 0
        else
            echo "❌ $service_name is not running"
            return 1
        fi
    else
        echo "❌ $service_name PID file not found"
        return 1
    fi
}

# Start services
echo "🔄 Starting Solana services..."

# Start the comprehensive metrics server
start_service "solana_server" "solana_comprehensive_server.py"

# Start the archive collector (this will run in background and collect historical data)
start_service "solana_archive" "solana_archive_collector.py"

# Wait a moment for services to start
sleep 5

# Check if services are running
echo "🔍 Checking service status..."
check_service "solana_server"
check_service "solana_archive"

# Display service information
echo ""
echo "📊 Solana System Status:"
echo "=========================="
echo "🌐 Main API Server: http://localhost:8001"
echo "📈 Metrics Endpoint: http://localhost:9091/metrics"
echo "🔌 WebSocket: ws://localhost:8001/ws"
echo "📚 Archive API: http://localhost:8001/api/archive"
echo "📈 Stats API: http://localhost:8001/api/stats"
echo ""

# Display log files
echo "📝 Log Files:"
echo "============="
echo "Server Log: logs/solana_server.log"
echo "Archive Log: logs/solana_archive.log"
echo ""

# Run system test
echo "🧪 Running system tests..."
python3 test_solana_system.py

echo ""
echo "🎉 Solana system startup completed!"
echo ""
echo "To stop the system, run: ./stop_solana_system.sh"
echo "To view logs, run: tail -f logs/solana_server.log"
echo "To check status, run: ./check_solana_status.sh"


