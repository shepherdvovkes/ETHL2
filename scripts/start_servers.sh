#!/bin/bash

# ETHL2 Server Management Script
# This script manages all servers using PM2

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/vovkes/ETHL2"
ECOSYSTEM_CONFIG="$PROJECT_DIR/ecosystem.config.js"
LOGS_DIR="$PROJECT_DIR/logs"

# Ensure we're in the project directory
cd "$PROJECT_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PM2 is installed
check_pm2() {
    if ! command -v pm2 &> /dev/null; then
        print_error "PM2 is not installed. Please install it first:"
        echo "npm install -g pm2"
        exit 1
    fi
    print_success "PM2 is installed"
}

# Function to create logs directory
create_logs_dir() {
    if [ ! -d "$LOGS_DIR" ]; then
        mkdir -p "$LOGS_DIR"
        print_success "Created logs directory: $LOGS_DIR"
    else
        print_status "Logs directory exists: $LOGS_DIR"
    fi
}

# Function to check if ecosystem config exists
check_ecosystem_config() {
    if [ ! -f "$ECOSYSTEM_CONFIG" ]; then
        print_error "Ecosystem config file not found: $ECOSYSTEM_CONFIG"
        exit 1
    fi
    print_success "Ecosystem config found: $ECOSYSTEM_CONFIG"
}

# Function to stop all existing processes
stop_all_servers() {
    print_status "Stopping all existing ETHL2 servers..."
    pm2 stop ecosystem.config.js --silent || true
    pm2 delete ecosystem.config.js --silent || true
    print_success "All servers stopped"
}

# Function to start all servers
start_all_servers() {
    print_status "Starting all ETHL2 servers..."
    
    # Start all servers defined in ecosystem config
    pm2 start ecosystem.config.js
    
    print_success "All servers started successfully!"
}

# Function to show server status
show_status() {
    print_status "Server Status:"
    echo "=================================="
    pm2 list
    echo ""
    print_status "Server URLs:"
    echo "=================================="
    echo "• DEFIMON Main API:     http://localhost:8000"
    echo "• DEFIMON Backend API:  http://localhost:8002"
    echo "• Avalanche Server:     http://localhost:8003"
    echo "• Avalanche Simple:     http://localhost:8001"
    echo "• Avalanche Limited:    http://localhost:8004"
    echo "• Investment Monitor:   http://localhost:8005"
    echo "• L2 Dashboard:         http://localhost:8006"
    echo ""
    print_status "API Documentation:"
    echo "• Main API Docs:        http://localhost:8000/docs"
    echo "• Backend API Docs:     http://localhost:8002/docs"
    echo "• Avalanche API Docs:   http://localhost:8003/docs"
}

# Function to show logs
show_logs() {
    local service_name=${1:-""}
    if [ -z "$service_name" ]; then
        print_status "Showing logs for all services (press Ctrl+C to exit):"
        pm2 logs
    else
        print_status "Showing logs for $service_name (press Ctrl+C to exit):"
        pm2 logs "$service_name"
    fi
}

# Function to restart all servers
restart_all_servers() {
    print_status "Restarting all ETHL2 servers..."
    pm2 restart ecosystem.config.js
    print_success "All servers restarted"
}

# Function to save PM2 configuration
save_pm2_config() {
    print_status "Saving PM2 configuration..."
    pm2 save
    print_success "PM2 configuration saved"
}

# Function to setup PM2 startup
setup_startup() {
    print_status "Setting up PM2 startup script..."
    pm2 startup
    print_warning "Please run the command shown above as root/sudo to enable PM2 startup"
}

# Function to show help
show_help() {
    echo "ETHL2 Server Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all servers"
    echo "  stop      - Stop all servers"
    echo "  restart   - Restart all servers"
    echo "  status    - Show server status"
    echo "  logs      - Show logs for all servers"
    echo "  logs [service] - Show logs for specific service"
    echo "  save      - Save PM2 configuration"
    echo "  startup   - Setup PM2 startup script"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs defimon-main"
    echo "  $0 restart"
}

# Main script logic
main() {
    local command=${1:-"start"}
    
    case "$command" in
        "start")
            check_pm2
            create_logs_dir
            check_ecosystem_config
            stop_all_servers
            start_all_servers
            show_status
            save_pm2_config
            ;;
        "stop")
            check_pm2
            stop_all_servers
            ;;
        "restart")
            check_pm2
            restart_all_servers
            show_status
            ;;
        "status")
            check_pm2
            show_status
            ;;
        "logs")
            check_pm2
            show_logs "$2"
            ;;
        "save")
            check_pm2
            save_pm2_config
            ;;
        "startup")
            check_pm2
            setup_startup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
