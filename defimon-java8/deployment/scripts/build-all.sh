#!/bin/bash

# DEFIMON Java 8 Microservices Build Script
# Builds all services in the correct order with dependencies

set -e

echo "🚀 Building DEFIMON Java 8 Microservices Platform..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Use local Maven installation
MAVEN_HOME="/home/vovkes/ETHL2/apache-maven-3.9.6"
MVN_CMD="$MAVEN_HOME/bin/mvn"

# Check if local Maven is available
if [ ! -f "$MVN_CMD" ]; then
    print_error "Local Maven not found at $MVN_CMD. Please check the path."
    exit 1
fi

print_status "Using Maven at: $MVN_CMD"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_status "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Build shared libraries first
print_status "📦 Building shared libraries..."
cd shared-libraries

# Build defimon-common
print_status "Building defimon-common..."
cd defimon-common
$MVN_CMD clean package -DskipTests
if [ $? -eq 0 ]; then
    print_success "defimon-common built successfully"
else
    print_error "Failed to build defimon-common"
    exit 1
fi
cd ..

# Build other shared libraries
for lib in defimon-security defimon-monitoring defimon-blockchain; do
    if [ -d "$lib" ]; then
        print_status "Building $lib..."
        cd "$lib"
        $MVN_CMD clean package -DskipTests
        if [ $? -eq 0 ]; then
            print_success "$lib built successfully"
        else
            print_error "Failed to build $lib"
            exit 1
        fi
        cd ..
    fi
done

cd "$PROJECT_ROOT"

# Build platform services
print_status "🏗️ Building platform services..."
cd platform-services

services=(
    "eureka-server"
    "config-server"
    "admin-server"
    "api-gateway"
)

for service in "${services[@]}"; do
    if [ -d "$service" ]; then
        print_status "Building $service..."
        cd "$service"
        $MVN_CMD clean package -DskipTests
        if [ $? -eq 0 ]; then
            print_success "$service built successfully"
            # Build Docker image
            docker build -t "defimon/$service:latest" .
            if [ $? -eq 0 ]; then
                print_success "Docker image for $service created successfully"
            else
                print_warning "Failed to create Docker image for $service"
            fi
        else
            print_error "Failed to build $service"
            exit 1
        fi
        cd ..
    fi
done

cd "$PROJECT_ROOT"

# Build core services
print_status "💼 Building core services..."
cd core-services

services=(
    "asset-management-service"
    "blockchain-integration-service"
    "analytics-engine-service"
    "ml-inference-service"
)

for service in "${services[@]}"; do
    if [ -d "$service" ]; then
        print_status "Building $service..."
        cd "$service"
        $MVN_CMD clean package -DskipTests
        if [ $? -eq 0 ]; then
            print_success "$service built successfully"
            # Build Docker image
            docker build -t "defimon/$service:latest" .
            if [ $? -eq 0 ]; then
                print_success "Docker image for $service created successfully"
            else
                print_warning "Failed to create Docker image for $service"
            fi
        else
            print_error "Failed to build $service"
            exit 1
        fi
        cd ..
    fi
done

cd "$PROJECT_ROOT"

# Build data services
print_status "📊 Building data services..."
cd data-services

services=(
    "data-collector-service"
    "stream-processing-service"
    "batch-processing-service"
    "cache-management-service"
)

for service in "${services[@]}"; do
    if [ -d "$service" ]; then
        print_status "Building $service..."
        cd "$service"
        $MVN_CMD clean package -DskipTests
        if [ $? -eq 0 ]; then
            print_success "$service built successfully"
            # Build Docker image
            docker build -t "defimon/$service:latest" .
            if [ $? -eq 0 ]; then
                print_success "Docker image for $service created successfully"
            else
                print_warning "Failed to create Docker image for $service"
            fi
        else
            print_error "Failed to build $service"
            exit 1
        fi
        cd ..
    fi
done

cd "$PROJECT_ROOT"

# Build blockchain services
print_status "⛓️ Building blockchain services..."
cd blockchain-services

services=(
    "bitcoin-service"
    "ethereum-service"
    "polygon-service"
    "multichain-service"
)

for service in "${services[@]}"; do
    if [ -d "$service" ]; then
        print_status "Building $service..."
        cd "$service"
        $MVN_CMD clean package -DskipTests
        if [ $? -eq 0 ]; then
            print_success "$service built successfully"
            # Build Docker image
            docker build -t "defimon/$service:latest" .
            if [ $? -eq 0 ]; then
                print_success "Docker image for $service created successfully"
            else
                print_warning "Failed to create Docker image for $service"
            fi
        else
            print_error "Failed to build $service"
            exit 1
        fi
        cd ..
    fi
done

cd "$PROJECT_ROOT"

# Build Python services (if any)
if [ -d "python-services" ]; then
    print_status "🐍 Building Python services..."
    cd python-services
    
    for service in */; do
        if [ -d "$service" ]; then
            service_name=$(basename "$service")
            print_status "Building $service_name..."
            cd "$service"
            
            # Build Docker image for Python service
            docker build -t "defimon/$service_name:latest" .
            if [ $? -eq 0 ]; then
                print_success "Docker image for $service_name created successfully"
            else
                print_warning "Failed to create Docker image for $service_name"
            fi
            cd ..
        fi
    done
    
    cd "$PROJECT_ROOT"
fi

# List all built Docker images
print_status "📋 Built Docker images:"
docker images | grep defimon || print_warning "No DEFIMON images found"

print_success "🎉 All DEFIMON services built successfully!"
print_status "Next steps:"
echo "  1. Run 'docker-compose up -d' to start the platform"
echo "  2. Access services:"
echo "     - API Gateway: http://localhost:8080"
echo "     - Eureka Dashboard: http://localhost:8761"
echo "     - Grafana: http://localhost:3000 (admin/admin)"
echo "     - Prometheus: http://localhost:9090"
echo "     - Jaeger: http://localhost:16686"
echo ""
print_status "For production deployment, use: ./deployment/scripts/deploy-prod.sh"
