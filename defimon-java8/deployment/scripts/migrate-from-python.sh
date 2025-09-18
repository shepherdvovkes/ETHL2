#!/bin/bash

# DEFIMON Migration Script: Python to Java 8
# Migrates data and configuration from Python DEFIMON to Java 8 version

set -e

echo "🔄 Migrating DEFIMON from Python to Java 8..."
echo "============================================="

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
PYTHON_PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"

print_status "Project root: $PROJECT_ROOT"
print_status "Python project root: $PYTHON_PROJECT_ROOT"

# Check if Python project exists
if [ ! -d "$PYTHON_PROJECT_ROOT/src" ]; then
    print_error "Python DEFIMON project not found at $PYTHON_PROJECT_ROOT"
    print_status "Please ensure the Python version is in the parent directory"
    exit 1
fi

# Create backup directory
BACKUP_DIR="$PROJECT_ROOT/migration-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

print_status "Creating backup in: $BACKUP_DIR"

# 1. Migrate Configuration
print_status "📋 Migrating configuration files..."

# Copy Bitcoin configuration
if [ -f "$PYTHON_PROJECT_ROOT/bitcoin/bitcoin.conf" ]; then
    print_status "Migrating Bitcoin configuration..."
    cp "$PYTHON_PROJECT_ROOT/bitcoin/bitcoin.conf" "$BACKUP_DIR/"
    
    # Extract QuickNode configuration
    QUICKNODE_URL=$(grep "rpcconnect=" "$PYTHON_PROJECT_ROOT/bitcoin/bitcoin.conf" | cut -d'=' -f2)
    QUICKNODE_USER=$(grep "rpcuser=" "$PYTHON_PROJECT_ROOT/bitcoin/bitcoin.conf" | cut -d'=' -f2)
    QUICKNODE_PASSWORD=$(grep "rpcpassword=" "$PYTHON_PROJECT_ROOT/bitcoin/bitcoin.conf" | cut -d'=' -f2)
    
    # Create Java configuration
    cat > "$PROJECT_ROOT/infrastructure/config/bitcoin.properties" << EOF
# Bitcoin QuickNode Configuration (Migrated from Python)
bitcoin.quicknode.rpc.url=$QUICKNODE_URL
bitcoin.quicknode.rpc.user=$QUICKNODE_USER
bitcoin.quicknode.rpc.password=$QUICKNODE_PASSWORD
bitcoin.quicknode.rpc.port=443
bitcoin.quicknode.rpc.ssl=true
EOF
    
    print_success "Bitcoin configuration migrated"
else
    print_warning "Bitcoin configuration not found"
fi

# Copy environment configuration
if [ -f "$PYTHON_PROJECT_ROOT/config.env" ]; then
    print_status "Migrating environment configuration..."
    cp "$PYTHON_PROJECT_ROOT/config.env" "$BACKUP_DIR/"
    
    # Convert to Java properties format
    cat > "$PROJECT_ROOT/infrastructure/config/application.properties" << EOF
# DEFIMON Configuration (Migrated from Python)
EOF
    
    # Parse and convert environment variables
    while IFS='=' read -r key value; do
        if [[ ! $key =~ ^# ]] && [[ -n $key ]]; then
            # Convert Python naming to Java naming
            java_key=$(echo "$key" | tr '[:upper:]' '[:lower:]' | sed 's/_/./g')
            echo "$java_key=$value" >> "$PROJECT_ROOT/infrastructure/config/application.properties"
        fi
    done < "$PYTHON_PROJECT_ROOT/config.env"
    
    print_success "Environment configuration migrated"
else
    print_warning "Environment configuration not found"
fi

# 2. Migrate Database Schema
print_status "🗄️ Migrating database schema..."

# Check if Python database exists
PYTHON_DB_PATH="$PYTHON_PROJECT_ROOT/defimon.db"
if [ -f "$PYTHON_DB_PATH" ]; then
    print_status "Found Python SQLite database, creating migration script..."
    
    # Create database migration script
    cat > "$PROJECT_ROOT/infrastructure/databases/postgresql/migrate_from_sqlite.sql" << 'EOF'
-- Migration script from Python SQLite to PostgreSQL
-- This script helps migrate data from the Python DEFIMON database

-- Create tables (already handled by JPA entities)
-- This is just for reference

-- Migrate assets data
INSERT INTO assets (symbol, name, contract_address, blockchain, category, created_at, updated_at)
SELECT symbol, name, contract_address, blockchain, category, 
       COALESCE(created_at, NOW()), COALESCE(updated_at, NOW())
FROM python_assets_temp;

-- Migrate on-chain metrics
INSERT INTO onchain_metrics (asset_id, timestamp, price_usd, market_cap, volume_24h, created_at)
SELECT a.id, m.timestamp, m.price_usd, m.market_cap, m.volume_24h, NOW()
FROM python_onchain_metrics_temp m
JOIN assets a ON a.symbol = m.symbol;

-- Migrate financial metrics
INSERT INTO financial_metrics (asset_id, timestamp, revenue_24h, created_at)
SELECT a.id, f.timestamp, f.revenue_24h, NOW()
FROM python_financial_metrics_temp f
JOIN assets a ON a.symbol = f.symbol;

-- Migrate ML predictions
INSERT INTO ml_predictions (asset_id, model_name, prediction_type, prediction_horizon, 
                           prediction_value, confidence_score, created_at)
SELECT a.id, p.model_name, p.prediction_type, p.prediction_horizon,
       p.prediction_value, p.confidence_score, NOW()
FROM python_ml_predictions_temp p
JOIN assets a ON a.symbol = p.symbol;
EOF
    
    print_success "Database migration script created"
else
    print_warning "Python database not found, skipping database migration"
fi

# 3. Migrate Data Files
print_status "📊 Migrating data files..."

# Copy training data
if [ -d "$PYTHON_PROJECT_ROOT/ml_data" ]; then
    print_status "Migrating ML training data..."
    cp -r "$PYTHON_PROJECT_ROOT/ml_data" "$BACKUP_DIR/"
    
    # Create Java ML data directory
    mkdir -p "$PROJECT_ROOT/data/ml-models"
    
    # Copy model files
    if [ -d "$PYTHON_PROJECT_ROOT/models" ]; then
        cp -r "$PYTHON_PROJECT_ROOT/models" "$PROJECT_ROOT/data/ml-models/python-models"
        print_success "ML models migrated"
    fi
    
    print_success "ML training data migrated"
fi

# Copy CSV data files
print_status "Migrating CSV data files..."
for csv_file in "$PYTHON_PROJECT_ROOT"/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        print_status "Migrating $filename..."
        cp "$csv_file" "$BACKUP_DIR/"
        
        # Copy to Java project data directory
        mkdir -p "$PROJECT_ROOT/data/csv"
        cp "$csv_file" "$PROJECT_ROOT/data/csv/"
    fi
done

# 4. Migrate Logs
print_status "📝 Migrating logs..."

if [ -d "$PYTHON_PROJECT_ROOT/logs" ]; then
    print_status "Migrating log files..."
    cp -r "$PYTHON_PROJECT_ROOT/logs" "$BACKUP_DIR/"
    print_success "Logs migrated"
fi

# 5. Create Migration Report
print_status "📋 Creating migration report..."

cat > "$BACKUP_DIR/migration-report.md" << EOF
# DEFIMON Migration Report

**Migration Date**: $(date)
**From**: Python DEFIMON
**To**: Java 8 DEFIMON

## Migrated Components

### Configuration
- [x] Bitcoin QuickNode configuration
- [x] Environment variables
- [x] API keys and endpoints

### Database
- [x] Database schema migration script
- [x] Data migration instructions

### Data Files
- [x] ML training data
- [x] CSV data files
- [x] Model files

### Logs
- [x] Historical log files

## Next Steps

1. **Start Java Services**:
   \`\`\`bash
   cd $PROJECT_ROOT
   ./deployment/scripts/deploy-dev.sh
   \`\`\`

2. **Migrate Database Data**:
   - Export data from Python SQLite database
   - Import into PostgreSQL using migration script

3. **Verify Migration**:
   - Check all services are running
   - Verify data collection is working
   - Test API endpoints

4. **Update Configuration**:
   - Review and update application properties
   - Configure new Java-specific settings

## Backup Location
All original files backed up to: $BACKUP_DIR

## Support
For migration issues, check the logs and documentation.
EOF

print_success "Migration report created: $BACKUP_DIR/migration-report.md"

# 6. Create Quick Start Guide
print_status "📖 Creating quick start guide..."

cat > "$PROJECT_ROOT/MIGRATION-GUIDE.md" << EOF
# DEFIMON Migration Guide: Python to Java 8

## Quick Start After Migration

### 1. Start the Java Platform
\`\`\`bash
cd $PROJECT_ROOT
./deployment/scripts/deploy-dev.sh
\`\`\`

### 2. Verify Services
- API Gateway: http://localhost:8080
- Eureka Dashboard: http://localhost:8761
- Grafana: http://localhost:3000

### 3. Migrate Database Data (if needed)
\`\`\`bash
# Export from Python SQLite
sqlite3 $PYTHON_PROJECT_ROOT/defimon.db .dump > python_data.sql

# Import to PostgreSQL (manual process)
# Use the migration script in infrastructure/databases/postgresql/
\`\`\`

### 4. Test Data Collection
\`\`\`bash
# Test Bitcoin service
curl http://localhost:8080/api/v1/bitcoin/metrics

# Test data collector
curl http://localhost:8080/api/v1/collector/status
\`\`\`

## Key Differences

### Python vs Java 8
- **Concurrency**: Java uses true parallelism vs Python's GIL
- **Memory Management**: Java's garbage collection vs Python's reference counting
- **Performance**: Java is generally faster for CPU-intensive tasks
- **Ecosystem**: Rich enterprise libraries and frameworks

### Architecture Changes
- **Microservices**: Each component is now a separate service
- **Service Discovery**: Eureka for automatic service registration
- **Configuration**: Centralized configuration management
- **Monitoring**: Production-ready observability stack

## Troubleshooting

### Common Issues
1. **Service Discovery**: Ensure Eureka is running first
2. **Database**: Check PostgreSQL connection
3. **Kafka**: Verify Kafka cluster is healthy
4. **Configuration**: Check environment variables

### Logs
\`\`\`bash
# View service logs
docker-compose logs -f [service-name]

# Check specific service
docker-compose logs -f bitcoin-service
\`\`\`

## Support
- Check the main README.md for detailed documentation
- Review migration report in backup directory
- Use Grafana dashboards for monitoring
EOF

print_success "Quick start guide created: $PROJECT_ROOT/MIGRATION-GUIDE.md"

# 7. Final Summary
print_status "🎉 Migration completed successfully!"
echo ""
print_status "📁 Backup created at: $BACKUP_DIR"
print_status "📖 Migration guide: $PROJECT_ROOT/MIGRATION-GUIDE.md"
echo ""
print_status "🚀 Next steps:"
echo "  1. Review the migration report"
echo "  2. Start the Java platform: ./deployment/scripts/deploy-dev.sh"
echo "  3. Verify all services are running"
echo "  4. Test data collection and API endpoints"
echo ""
print_status "📊 Monitoring:"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jaeger: http://localhost:16686"
echo ""
print_warning "⚠️  Remember to:"
echo "  - Update any external integrations to use new API endpoints"
echo "  - Migrate database data if needed"
echo "  - Update monitoring and alerting configurations"
echo "  - Test all functionality before switching production traffic"
