#!/bin/bash

# DEFIMON Java 8 Test Runner Script
# Runs comprehensive tests for all services

set -e

echo "🧪 Running DEFIMON Java 8 Tests..."
echo "=================================="

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

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    print_error "Maven is not installed. Please install Maven first."
    exit 1
fi

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run tests for a module
run_module_tests() {
    local module_name=$1
    local module_path=$2
    
    if [ ! -d "$module_path" ]; then
        print_warning "Module $module_name not found at $module_path, skipping..."
        return 0
    fi
    
    print_status "Testing $module_name..."
    cd "$module_path"
    
    # Run tests
    if mvn test -q; then
        print_success "$module_name tests passed"
        ((PASSED_TESTS++))
    else
        print_error "$module_name tests failed"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
    cd "$PROJECT_ROOT"
}

# 1. Test Shared Libraries
print_status "📚 Testing shared libraries..."
run_module_tests "defimon-common" "shared-libraries/defimon-common"

# 2. Test Platform Services
print_status "🏗️ Testing platform services..."
run_module_tests "eureka-server" "platform-services/eureka-server"
run_module_tests "config-server" "platform-services/config-server"
run_module_tests "api-gateway" "platform-services/api-gateway"

# 3. Test Data Services
print_status "📊 Testing data services..."
run_module_tests "data-collector-service" "data-services/data-collector-service"

# 4. Test Blockchain Services
print_status "⛓️ Testing blockchain services..."
run_module_tests "bitcoin-service" "blockchain-services/bitcoin-service"

# 5. Run Integration Tests
print_status "🔗 Running integration tests..."

# Create a simple integration test
cat > integration-test.sh << 'EOF'
#!/bin/bash
echo "Running integration tests..."

# Test 1: Check if all services can be built
echo "Test 1: Building all services..."
if ./deployment/scripts/build-all.sh > /dev/null 2>&1; then
    echo "✅ All services built successfully"
else
    echo "❌ Service build failed"
    exit 1
fi

# Test 2: Check Docker Compose syntax
echo "Test 2: Validating Docker Compose..."
if docker-compose config > /dev/null 2>&1; then
    echo "✅ Docker Compose configuration is valid"
else
    echo "❌ Docker Compose configuration is invalid"
    exit 1
fi

# Test 3: Check if all required files exist
echo "Test 3: Checking required files..."
required_files=(
    "parent-pom.xml"
    "docker-compose.yml"
    "README.md"
    "shared-libraries/defimon-common/pom.xml"
    "platform-services/eureka-server/pom.xml"
    "platform-services/config-server/pom.xml"
    "platform-services/api-gateway/pom.xml"
    "data-services/data-collector-service/pom.xml"
    "blockchain-services/bitcoin-service/pom.xml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        exit 1
    fi
done

echo "✅ All integration tests passed"
EOF

chmod +x integration-test.sh
if ./integration-test.sh; then
    print_success "Integration tests passed"
    ((PASSED_TESTS++))
else
    print_error "Integration tests failed"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Clean up
rm -f integration-test.sh

# 6. Run Code Quality Checks
print_status "🔍 Running code quality checks..."

# Check for compilation errors
print_status "Checking compilation..."
if mvn compile -q -pl shared-libraries/defimon-common; then
    print_success "defimon-common compiles successfully"
else
    print_error "defimon-common compilation failed"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Check for missing dependencies
print_status "Checking dependencies..."
if mvn dependency:resolve -q -pl shared-libraries/defimon-common; then
    print_success "Dependencies resolved successfully"
else
    print_error "Dependency resolution failed"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# 7. Test Results Summary
echo ""
echo "📊 Test Results Summary"
echo "======================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    print_success "🎉 All tests passed! Migration is 100% successful!"
    echo ""
    print_status "Next steps:"
    echo "  1. Deploy the platform: ./deployment/scripts/deploy-dev.sh"
    echo "  2. Test the services: curl http://localhost:8080/actuator/health"
    echo "  3. Monitor with Grafana: http://localhost:3000"
    exit 0
else
    print_error "❌ Some tests failed. Please fix the issues before proceeding."
    echo ""
    print_status "Common fixes:"
    echo "  1. Check for missing imports"
    echo "  2. Verify Maven dependencies"
    echo "  3. Fix compilation errors"
    echo "  4. Update test configurations"
    exit 1
fi
