#!/bin/bash

# Solana System Status Check Script
# Checks the status of all Solana services

echo "🔍 Solana System Status Check"
echo "=============================="

# Function to check service status
check_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            echo "✅ $service_name: RUNNING (PID: $pid)"
            
            # Get memory usage
            local memory=$(ps -p $pid -o rss= | awk '{print $1/1024 " MB"}')
            echo "   Memory Usage: $memory"
            
            # Get CPU usage (if available)
            if command -v top &> /dev/null; then
                local cpu=$(top -p $pid -n 1 -b | grep $pid | awk '{print $9}')
                if [ ! -z "$cpu" ]; then
                    echo "   CPU Usage: $cpu%"
                fi
            fi
            
            return 0
        else
            echo "❌ $service_name: NOT RUNNING (stale PID file)"
            return 1
        fi
    else
        echo "❌ $service_name: NOT RUNNING (no PID file)"
        return 1
    fi
}

# Check services
echo ""
echo "📊 Service Status:"
echo "------------------"
check_service "solana_server"
check_service "solana_archive"

# Check database files
echo ""
echo "🗄️ Database Status:"
echo "-------------------"
if [ -f "solana_data.db" ]; then
    local size=$(du -h solana_data.db | cut -f1)
    echo "✅ Main Database: solana_data.db ($size)"
else
    echo "❌ Main Database: solana_data.db (not found)"
fi

if [ -f "solana_archive_data.db" ]; then
    local size=$(du -h solana_archive_data.db | cut -f1)
    echo "✅ Archive Database: solana_archive_data.db ($size)"
else
    echo "❌ Archive Database: solana_archive_data.db (not found)"
fi

# Check log files
echo ""
echo "📝 Log Files:"
echo "-------------"
if [ -f "logs/solana_server.log" ]; then
    local size=$(du -h logs/solana_server.log | cut -f1)
    echo "✅ Server Log: logs/solana_server.log ($size)"
else
    echo "❌ Server Log: logs/solana_server.log (not found)"
fi

if [ -f "logs/solana_archive.log" ]; then
    local size=$(du -h logs/solana_archive.log | cut -f1)
    echo "✅ Archive Log: logs/solana_archive.log ($size)"
else
    echo "❌ Archive Log: logs/solana_archive.log (not found)"
fi

# Test API endpoints
echo ""
echo "🌐 API Endpoint Tests:"
echo "----------------------"

# Test main API
if curl -s http://localhost:8001/ > /dev/null 2>&1; then
    echo "✅ Main API: http://localhost:8001 (accessible)"
else
    echo "❌ Main API: http://localhost:8001 (not accessible)"
fi

# Test metrics endpoint
if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
    echo "✅ Metrics API: http://localhost:9091/metrics (accessible)"
else
    echo "❌ Metrics API: http://localhost:9091/metrics (not accessible)"
fi

# Show recent log entries
echo ""
echo "📋 Recent Log Entries:"
echo "----------------------"
if [ -f "logs/solana_server.log" ]; then
    echo "Server Log (last 5 lines):"
    tail -5 logs/solana_server.log
    echo ""
fi

if [ -f "logs/solana_archive.log" ]; then
    echo "Archive Log (last 5 lines):"
    tail -5 logs/solana_archive.log
fi

echo ""
echo "🎯 Quick Commands:"
echo "------------------"
echo "Start system: ./start_solana_system.sh"
echo "Stop system: ./stop_solana_system.sh"
echo "View server logs: tail -f logs/solana_server.log"
echo "View archive logs: tail -f logs/solana_archive.log"
echo "Test system: python3 test_solana_system.py"


