#!/bin/bash

# Polkadot Metrics Server Startup Script
# This script sets up and starts the Polkadot metrics collection server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/vovkes/ETHL2"
POLKADOT_SERVER_PORT=8007
LOG_DIR="$PROJECT_DIR/logs"

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check PM2
    if ! command -v pm2 &> /dev/null; then
        print_error "PM2 is not installed. Please install it first:"
        echo "npm install -g pm2"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_DIR/polkadot_metrics_server.py" ]; then
        print_error "Polkadot server files not found in $PROJECT_DIR"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Create logs directory
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        print_success "Created logs directory: $LOG_DIR"
    else
        print_status "Logs directory exists: $LOG_DIR"
    fi
    
    # Create src directory if it doesn't exist
    if [ ! -d "$PROJECT_DIR/src" ]; then
        mkdir -p "$PROJECT_DIR/src"
        print_warning "Created src directory (this might indicate incomplete setup)"
    fi
}

# Function to setup database
setup_database() {
    print_status "Setting up Polkadot database tables..."
    
    cd "$PROJECT_DIR"
    
    if [ -f "setup_polkadot_database.py" ]; then
        python3 setup_polkadot_database.py
        print_success "Database setup completed"
    else
        print_error "Database setup script not found"
        exit 1
    fi
}

# Function to check if server is already running
check_existing_server() {
    print_status "Checking for existing Polkadot server..."
    
    if pm2 list | grep -q "polkadot-metrics"; then
        print_warning "Polkadot server is already running"
        echo "Current status:"
        pm2 list | grep "polkadot-metrics"
        
        read -p "Do you want to restart it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Restarting Polkadot server..."
            pm2 restart polkadot-metrics
            print_success "Polkadot server restarted"
        else
            print_status "Keeping existing server running"
        fi
        return 0
    else
        print_status "No existing Polkadot server found"
        return 1
    fi
}

# Function to start the server
start_server() {
    print_status "Starting Polkadot metrics server..."
    
    cd "$PROJECT_DIR"
    
    # Start the server using PM2
    pm2 start ecosystem.config.js --only polkadot-metrics
    
    # Wait a moment for the server to start
    sleep 3
    
    # Check if the server started successfully
    if pm2 list | grep -q "polkadot-metrics.*online"; then
        print_success "Polkadot server started successfully!"
    else
        print_error "Failed to start Polkadot server"
        print_status "Checking logs for errors..."
        pm2 logs polkadot-metrics --lines 20
        exit 1
    fi
}

# Function to test the server
test_server() {
    print_status "Testing server connectivity..."
    
    # Wait a bit more for the server to fully initialize
    sleep 5
    
    # Test health endpoint
    if curl -s -f "http://localhost:$POLKADOT_SERVER_PORT/health" > /dev/null; then
        print_success "Server health check passed"
    else
        print_warning "Server health check failed - server might still be starting"
    fi
    
    # Test root endpoint
    if curl -s -f "http://localhost:$POLKADOT_SERVER_PORT/" > /dev/null; then
        print_success "Server is responding to requests"
    else
        print_warning "Server not responding yet - check logs if issues persist"
    fi
}

# Function to show server information
show_server_info() {
    print_status "Server Information:"
    echo "=================================="
    echo "• Server Name:    polkadot-metrics"
    echo "• Server URL:     http://localhost:$POLKADOT_SERVER_PORT"
    echo "• API Docs:       http://localhost:$POLKADOT_SERVER_PORT/docs"
    echo "• Health Check:   http://localhost:$POLKADOT_SERVER_PORT/health"
    echo "• Logs:          pm2 logs polkadot-metrics"
    echo "• Status:        pm2 status"
    echo ""
    print_status "Key API Endpoints:"
    echo "=================================="
    echo "• Network Info:   GET /network/info"
    echo "• Network Metrics: GET /network/metrics"
    echo "• Parachains:     GET /parachains"
    echo "• Staking:        GET /staking/metrics"
    echo "• Governance:     GET /governance/metrics"
    echo "• Economic:       GET /economic/metrics"
    echo "• Cross-chain:    GET /cross-chain/metrics"
    echo "• Historical:     GET /historical/{days}"
    echo "• Manual Collect: POST /collect"
    echo ""
}

# Function to show help
show_help() {
    echo "Polkadot Metrics Server Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-only     Only setup database, don't start server"
    echo "  --force-restart  Force restart even if server is running"
    echo "  --test-only      Only test existing server, don't start"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full setup and start"
    echo "  $0 --setup-only       # Only setup database"
    echo "  $0 --force-restart    # Force restart server"
    echo "  $0 --test-only        # Test existing server"
}

# Main execution
main() {
    local setup_only=false
    local force_restart=false
    local test_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                setup_only=true
                shift
                ;;
            --force-restart)
                force_restart=true
                shift
                ;;
            --test-only)
                test_only=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_status "Starting Polkadot Metrics Server Setup..."
    echo "Project Directory: $PROJECT_DIR"
    echo "Server Port: $POLKADOT_SERVER_PORT"
    echo ""
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Create directories
    create_directories
    
    # Setup database
    setup_database
    
    if [ "$setup_only" = true ]; then
        print_success "Database setup completed. Server not started."
        exit 0
    fi
    
    if [ "$test_only" = true ]; then
        test_server
        show_server_info
        exit 0
    fi
    
    # Check for existing server
    if [ "$force_restart" = false ] && check_existing_server; then
        show_server_info
        exit 0
    fi
    
    # Start the server
    start_server
    
    # Test the server
    test_server
    
    # Show server information
    show_server_info
    
    print_success "Polkadot Metrics Server is ready!"
    print_status "Use 'pm2 logs polkadot-metrics' to view logs"
    print_status "Use 'pm2 status' to check server status"
}

# Run main function
main "$@"
