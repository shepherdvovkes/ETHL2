#!/bin/bash

# DEFIMON Java 8 Microservices Development Deployment Script
# Deploys the platform for development environment

set -e

echo "🚀 Deploying DEFIMON to Development Environment..."
echo "================================================="

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

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_status "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build all services first
print_status "Building all services..."
./deployment/scripts/build-all.sh

if [ $? -ne 0 ]; then
    print_error "Build failed. Aborting deployment."
    exit 1
fi

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans

# Start infrastructure services first
print_status "🏗️ Starting infrastructure services..."
docker-compose up -d zookeeper kafka redis postgresql influxdb mongodb

# Wait for infrastructure to be ready
print_status "⏳ Waiting for infrastructure services to be ready..."
sleep 30

# Check if services are ready
print_status "Checking infrastructure services health..."

# Check PostgreSQL
print_status "Checking PostgreSQL..."
until docker-compose exec postgresql pg_isready -U defimon -d defimon_db; do
    print_status "Waiting for PostgreSQL..."
    sleep 5
done
print_success "PostgreSQL is ready"

# Check Redis
print_status "Checking Redis..."
until docker-compose exec redis redis-cli ping; do
    print_status "Waiting for Redis..."
    sleep 5
done
print_success "Redis is ready"

# Check Kafka
print_status "Checking Kafka..."
until docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list; do
    print_status "Waiting for Kafka..."
    sleep 5
done
print_success "Kafka is ready"

# Start service discovery
print_status "🔍 Starting service discovery..."
docker-compose up -d eureka-server config-server

# Wait for service discovery
print_status "⏳ Waiting for service discovery..."
sleep 20

# Check Eureka
print_status "Checking Eureka Server..."
until curl -f http://localhost:8761/actuator/health > /dev/null 2>&1; do
    print_status "Waiting for Eureka Server..."
    sleep 5
done
print_success "Eureka Server is ready"

# Start API Gateway
print_status "🌐 Starting API Gateway..."
docker-compose up -d api-gateway

# Wait for API Gateway
sleep 10

# Start core Java services
print_status "☕ Starting Java services..."
docker-compose up -d data-collector bitcoin-service ethereum-service polygon-service

# Start blockchain services
print_status "⛓️ Starting blockchain services..."
# Services already started above

# Start monitoring
print_status "📊 Starting monitoring services..."
docker-compose up -d prometheus grafana jaeger

# Wait for all services to be ready
print_status "⏳ Waiting for all services to be ready..."
sleep 30

# Check service health
print_status "🔍 Checking service health..."

services=(
    "eureka-server:8761"
    "config-server:8888"
    "api-gateway:8080"
    "data-collector:8100"
    "bitcoin-service:8200"
    "ethereum-service:8201"
    "polygon-service:8202"
    "prometheus:9090"
    "grafana:3000"
    "jaeger:16686"
)

for service in "${services[@]}"; do
    service_name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    print_status "Checking $service_name..."
    if curl -f "http://localhost:$port/actuator/health" > /dev/null 2>&1 || 
       curl -f "http://localhost:$port/health" > /dev/null 2>&1 ||
       curl -f "http://localhost:$port/" > /dev/null 2>&1; then
        print_success "$service_name is healthy"
    else
        print_warning "$service_name health check failed (may still be starting)"
    fi
done

# Show running containers
print_status "📋 Running containers:"
docker-compose ps

print_success "✅ DEFIMON deployed successfully to development environment!"
echo ""
print_status "🌐 Services available at:"
echo "  - API Gateway: http://localhost:8080"
echo "  - Eureka Dashboard: http://localhost:8761"
echo "  - Config Server: http://localhost:8888"
echo "  - Data Collector: http://localhost:8100"
echo "  - Bitcoin Service: http://localhost:8200"
echo "  - Ethereum Service: http://localhost:8201"
echo "  - Polygon Service: http://localhost:8202"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jaeger: http://localhost:16686"
echo ""
print_status "📊 Monitoring:"
echo "  - View logs: docker-compose logs -f [service-name]"
echo "  - Scale service: docker-compose up -d --scale [service-name]=3"
echo "  - Stop platform: docker-compose down"
echo ""
print_status "🔧 Development commands:"
echo "  - Rebuild service: docker-compose build [service-name] && docker-compose up -d [service-name]"
echo "  - View service logs: docker-compose logs -f [service-name]"
echo "  - Access service shell: docker-compose exec [service-name] sh"
