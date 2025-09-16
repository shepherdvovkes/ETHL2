#!/bin/bash

# PM2 Monitor Script for Comprehensive Polkadot Metrics System
# This script provides monitoring and management capabilities

echo "📊 PM2 Monitor for Comprehensive Polkadot Metrics System"
echo "======================================================="

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed. Please install PM2 first:"
    echo "   npm install -g pm2"
    exit 1
fi

# Function to show service status
show_status() {
    echo "📊 Service Status:"
    echo "=================="
    pm2 status
    echo ""
}

# Function to show logs
show_logs() {
    local service_name=$1
    if [ -z "$service_name" ]; then
        echo "📋 All Service Logs:"
        echo "==================="
        pm2 logs --lines 50
    else
        echo "📋 Logs for $service_name:"
        echo "========================="
        pm2 logs $service_name --lines 50
    fi
    echo ""
}

# Function to show service info
show_info() {
    local service_name=$1
    if [ -z "$service_name" ]; then
        echo "📋 All Service Information:"
        echo "=========================="
        pm2 list
    else
        echo "📋 Information for $service_name:"
        echo "==============================="
        pm2 show $service_name
    fi
    echo ""
}

# Function to restart service
restart_service() {
    local service_name=$1
    if [ -z "$service_name" ]; then
        echo "🔄 Restarting all services..."
        pm2 restart all
    else
        echo "🔄 Restarting $service_name..."
        pm2 restart $service_name
    fi
    echo "✅ Restart completed"
    echo ""
}

# Function to stop service
stop_service() {
    local service_name=$1
    if [ -z "$service_name" ]; then
        echo "🛑 Stopping all services..."
        pm2 stop all
    else
        echo "🛑 Stopping $service_name..."
        pm2 stop $service_name
    fi
    echo "✅ Stop completed"
    echo ""
}

# Function to start service
start_service() {
    local service_name=$1
    if [ -z "$service_name" ]; then
        echo "🚀 Starting all services..."
        pm2 start all
    else
        echo "🚀 Starting $service_name..."
        pm2 start $service_name
    fi
    echo "✅ Start completed"
    echo ""
}

# Function to show help
show_help() {
    echo "📋 Available Commands:"
    echo "====================="
    echo "  status [service]     - Show service status"
    echo "  logs [service]       - Show service logs"
    echo "  info [service]       - Show service information"
    echo "  restart [service]    - Restart service(s)"
    echo "  stop [service]       - Stop service(s)"
    echo "  start [service]      - Start service(s)"
    echo "  monit                - Open PM2 monitoring interface"
    echo "  help                 - Show this help"
    echo ""
    echo "📋 Available Services:"
    echo "====================="
    echo "  polkadot-comprehensive-metrics"
    echo "  polkadot-data-collector"
    echo "  polkadot-legacy-metrics"
    echo "  avalanche-metrics"
    echo "  l2-monitoring-dashboard"
    echo ""
    echo "📋 Examples:"
    echo "==========="
    echo "  $0 status"
    echo "  $0 logs polkadot-comprehensive-metrics"
    echo "  $0 restart polkadot-data-collector"
    echo "  $0 monit"
    echo ""
}

# Main script logic
case "$1" in
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "info")
        show_info "$2"
        ;;
    "restart")
        restart_service "$2"
        show_status
        ;;
    "stop")
        stop_service "$2"
        show_status
        ;;
    "start")
        start_service "$2"
        show_status
        ;;
    "monit")
        echo "🚀 Opening PM2 monitoring interface..."
        pm2 monit
        ;;
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
