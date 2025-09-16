#!/bin/bash

# Solana System Stop Script
# Stops all Solana services

echo "🛑 Stopping Solana System..."

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            echo "🔄 Stopping $service_name (PID: $pid)..."
            kill $pid
            sleep 2
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "⚠️ Force killing $service_name..."
                kill -9 $pid
            fi
            
            rm -f $pid_file
            echo "✅ $service_name stopped"
        else
            echo "⚠️ $service_name was not running"
            rm -f $pid_file
        fi
    else
        echo "⚠️ $service_name PID file not found"
    fi
}

# Stop services
stop_service "solana_server"
stop_service "solana_archive"

# Kill any remaining Python processes related to Solana
echo "🔄 Cleaning up remaining processes..."
pkill -f "solana_comprehensive_server.py"
pkill -f "solana_archive_collector.py"

echo "✅ Solana system stopped"


