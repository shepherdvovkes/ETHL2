#!/usr/bin/env python3
"""
Solana System Setup Script
Sets up the complete Solana data collection and serving system
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'aiohttp',
        'asyncio',
        'sqlite3',
        'json',
        'logging',
        'datetime',
        'concurrent.futures',
        'threading',
        'signal',
        'pathlib'
    ]
    
    logger.info("🔍 Checking Python packages...")
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.warning(f"⚠️  {package} is not available (this might be normal for built-in modules)")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'solana_backups',
        'solana_logs',
        'solana_models'
    ]
    
    logger.info("📁 Creating directories...")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return True

def setup_database():
    """Setup Solana database schema"""
    logger.info("🗄️  Setting up Solana database...")
    
    try:
        # Import and run database schema creation
        from solana_database_schema import create_solana_database_schema, create_archive_database_schema
        
        # Create main database
        if create_solana_database_schema():
            logger.info("✅ Main Solana database created")
        else:
            logger.error("❌ Failed to create main Solana database")
            return False
        
        # Create archive database
        if create_archive_database_schema():
            logger.info("✅ Archive Solana database created")
        else:
            logger.error("❌ Failed to create archive Solana database")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service files"""
    logger.info("🔧 Creating systemd service files...")
    
    # Solana data collector service
    collector_service = """[Unit]
Description=Solana Data Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/vovkes/ETHL2
ExecStart=/usr/bin/python3 /home/vovkes/ETHL2/solana_data_collector.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/home/vovkes/ETHL2

[Install]
WantedBy=multi-user.target
"""
    
    # Solana metrics server service
    server_service = """[Unit]
Description=Solana Metrics Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/vovkes/ETHL2
ExecStart=/usr/bin/python3 /home/vovkes/ETHL2/solana_metrics_server.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/home/vovkes/ETHL2

[Install]
WantedBy=multi-user.target
"""
    
    # Solana archive collector service
    archive_service = """[Unit]
Description=Solana Archive Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/vovkes/ETHL2
ExecStart=/usr/bin/python3 /home/vovkes/ETHL2/solana_archive_collector.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/home/vovkes/ETHL2

[Install]
WantedBy=multi-user.target
"""
    
    try:
        # Write service files
        with open('/etc/systemd/system/solana-data-collector.service', 'w') as f:
            f.write(collector_service)
        
        with open('/etc/systemd/system/solana-metrics-server.service', 'w') as f:
            f.write(server_service)
        
        with open('/etc/systemd/system/solana-archive-collector.service', 'w') as f:
            f.write(archive_service)
        
        logger.info("✅ Systemd service files created")
        
        # Reload systemd
        run_command("systemctl daemon-reload", "Reloading systemd")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create systemd services: {e}")
        return False

def create_startup_scripts():
    """Create startup and management scripts"""
    logger.info("📝 Creating startup scripts...")
    
    # Start all services script
    start_script = """#!/bin/bash
# Start Solana System Services

echo "🚀 Starting Solana system services..."

# Start data collector
echo "Starting Solana data collector..."
systemctl start solana-data-collector
systemctl enable solana-data-collector

# Start metrics server
echo "Starting Solana metrics server..."
systemctl start solana-metrics-server
systemctl enable solana-metrics-server

# Start archive collector (optional)
echo "Starting Solana archive collector..."
systemctl start solana-archive-collector
systemctl enable solana-archive-collector

echo "✅ All Solana services started!"
echo "📡 Metrics server: http://localhost:8001"
echo "📊 Prometheus metrics: http://localhost:9091/metrics"
echo "🔌 WebSocket: ws://localhost:8001/ws"
"""
    
    # Stop all services script
    stop_script = """#!/bin/bash
# Stop Solana System Services

echo "🛑 Stopping Solana system services..."

# Stop all services
systemctl stop solana-data-collector
systemctl stop solana-metrics-server
systemctl stop solana-archive-collector

echo "✅ All Solana services stopped!"
"""
    
    # Status check script
    status_script = """#!/bin/bash
# Check Solana System Status

echo "📊 Solana System Status:"
echo "========================="

echo "Data Collector:"
systemctl is-active solana-data-collector
systemctl is-enabled solana-data-collector

echo "Metrics Server:"
systemctl is-active solana-metrics-server
systemctl is-enabled solana-metrics-server

echo "Archive Collector:"
systemctl is-active solana-archive-collector
systemctl is-enabled solana-archive-collector

echo ""
echo "📡 API Endpoints:"
echo "Main API: http://localhost:8001"
echo "Metrics: http://localhost:9091/metrics"
echo "WebSocket: ws://localhost:8001/ws"
"""
    
    try:
        # Write scripts
        with open('start_solana_system.sh', 'w') as f:
            f.write(start_script)
        
        with open('stop_solana_system.sh', 'w') as f:
            f.write(stop_script)
        
        with open('check_solana_status.sh', 'w') as f:
            f.write(status_script)
        
        # Make scripts executable
        os.chmod('start_solana_system.sh', 0o755)
        os.chmod('stop_solana_system.sh', 0o755)
        os.chmod('check_solana_status.sh', 0o755)
        
        logger.info("✅ Startup scripts created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create startup scripts: {e}")
        return False

def create_monitoring_script():
    """Create monitoring and health check script"""
    logger.info("📊 Creating monitoring script...")
    
    monitoring_script = """#!/usr/bin/env python3
'''
Solana System Monitor
Monitors the health and performance of Solana data collection system
'''

import requests
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_api_health():
    """Check API health"""
    try:
        response = requests.get('http://localhost:8001/api/stats', timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ API Health: OK")
            logger.info(f"📊 Stats: {data}")
            return True
        else:
            logger.error(f"❌ API Health: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API Health: {e}")
        return False

def check_metrics_endpoint():
    """Check metrics endpoint"""
    try:
        response = requests.get('http://localhost:9091/metrics', timeout=10)
        if response.status_code == 200:
            logger.info("✅ Metrics endpoint: OK")
            return True
        else:
            logger.error(f"❌ Metrics endpoint: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Metrics endpoint: {e}")
        return False

def check_database_health():
    """Check database health"""
    try:
        import sqlite3
        
        # Check main database
        conn = sqlite3.connect('solana_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM solana_blocks")
        block_count = cursor.fetchone()[0]
        conn.close()
        
        # Check archive database
        conn = sqlite3.connect('solana_archive_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM solana_archive_blocks")
        archive_block_count = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"✅ Database Health: OK")
        logger.info(f"📊 Main DB blocks: {block_count}")
        logger.info(f"📊 Archive DB blocks: {archive_block_count}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database Health: {e}")
        return False

def main():
    """Main monitoring function"""
    logger.info("🔍 Starting Solana system health check...")
    
    api_health = check_api_health()
    metrics_health = check_metrics_endpoint()
    db_health = check_database_health()
    
    if api_health and metrics_health and db_health:
        logger.info("🎉 All systems healthy!")
        return True
    else:
        logger.error("⚠️  Some systems are unhealthy!")
        return False

if __name__ == "__main__":
    main()
"""
    
    try:
        with open('monitor_solana_system.py', 'w') as f:
            f.write(monitoring_script)
        
        os.chmod('monitor_solana_system.py', 0o755)
        logger.info("✅ Monitoring script created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create monitoring script: {e}")
        return False

def create_documentation():
    """Create documentation files"""
    logger.info("📚 Creating documentation...")
    
    readme_content = """# Solana Data Collection System

A comprehensive system for collecting, storing, and serving Solana blockchain data with real-time monitoring and archive capabilities.

## 🚀 Features

- **Real-time Data Collection**: 10 concurrent workers collecting blocks, transactions, accounts, tokens, programs, and validators
- **Archive Collection**: Complete historical data collection from genesis to present
- **REST API**: Comprehensive API for accessing collected data
- **WebSocket Support**: Real-time data streaming
- **Prometheus Metrics**: Monitoring and alerting support
- **High Performance**: Optimized database schema with proper indexing
- **Fault Tolerance**: Automatic retry mechanisms and error handling

## 📁 System Components

### Core Components
- `solana_data_collector.py` - Real-time data collection with 10 workers
- `solana_archive_collector.py` - Historical data collection
- `solana_metrics_server.py` - REST API and WebSocket server
- `solana_database_schema.py` - Database schema creation

### Configuration
- `solana_config.env` - Main configuration file
- `config.env` - Updated with Solana endpoints

### Management Scripts
- `start_solana_system.sh` - Start all services
- `stop_solana_system.sh` - Stop all services
- `check_solana_status.sh` - Check service status
- `monitor_solana_system.py` - Health monitoring

## 🛠️ Installation

1. **Setup Database Schema**:
   ```bash
   python3 solana_database_schema.py
   ```

2. **Install System Services**:
   ```bash
   sudo python3 setup_solana_system.py
   ```

3. **Start Services**:
   ```bash
   ./start_solana_system.sh
   ```

## 📡 API Endpoints

### REST API (Port 8001)
- `GET /` - Service information
- `GET /api/blocks` - Get blocks (supports limit, slot, range)
- `GET /api/transactions` - Get transactions (supports limit, signature)
- `GET /api/network_metrics` - Get network metrics
- `GET /api/validators` - Get validator information
- `GET /api/programs` - Get program information
- `GET /api/archive` - Get archive data
- `GET /api/stats` - Get collection statistics

### WebSocket (Port 8001)
- `ws://localhost:8001/ws` - Real-time data streaming
  - `subscribe_blocks` - Subscribe to latest blocks
  - `subscribe_metrics` - Subscribe to network metrics
  - `get_stats` - Get current statistics

### Metrics (Port 9091)
- `GET /metrics` - Prometheus-style metrics

## 📊 Database Schema

### Main Database (`solana_data.db`)
- `solana_blocks` - Block data
- `solana_transactions` - Transaction data
- `solana_accounts` - Account information
- `solana_token_accounts` - Token account data
- `solana_tokens` - Token information
- `solana_programs` - Program data
- `solana_validators` - Validator information
- `solana_staking_accounts` - Staking data
- `solana_epoch_info` - Epoch information
- `solana_network_metrics` - Network metrics
- `solana_price_data` - Price data
- `solana_defi_protocols` - DeFi protocol data
- `solana_nft_collections` - NFT collection data

### Archive Database (`solana_archive_data.db`)
- `solana_archive_blocks` - Historical block data
- `solana_archive_transactions` - Historical transaction data
- `solana_archive_network_metrics` - Historical network metrics
- `solana_archive_progress` - Collection progress tracking

## 🔧 Configuration

Edit `solana_config.env` to customize:

- **RPC Endpoints**: Solana RPC and WebSocket URLs
- **Collection Intervals**: Data collection frequencies
- **Rate Limits**: API rate limiting
- **Database Settings**: Database paths and optimization
- **Monitoring**: Health checks and alerting

## 📈 Monitoring

### Health Checks
```bash
python3 monitor_solana_system.py
```

### Service Status
```bash
./check_solana_status.sh
```

### Logs
- Data Collector: `solana_collector.log`
- Archive Collector: `solana_archive_collector.log`
- Metrics Server: `solana_metrics_server.log`

## 🚨 Troubleshooting

### Common Issues

1. **RPC Rate Limiting**: Adjust rate limits in config
2. **Database Locked**: Check for concurrent access
3. **Memory Issues**: Optimize batch sizes
4. **Network Issues**: Check RPC endpoint connectivity

### Performance Tuning

1. **Database Optimization**:
   - Enable WAL mode
   - Increase cache size
   - Use memory temp storage

2. **Collection Optimization**:
   - Adjust worker count
   - Optimize batch sizes
   - Tune collection intervals

## 📝 License

This project is part of the ETHL2 monitoring system.

## 🤝 Contributing

Contributions are welcome! Please ensure all tests pass and documentation is updated.
"""
    
    try:
        with open('SOLANA_SYSTEM_README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info("✅ Documentation created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create documentation: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Solana system setup...")
    
    # Check if running as root for systemd services
    if os.geteuid() != 0:
        logger.warning("⚠️  Not running as root. Systemd services will not be created.")
        create_systemd = False
    else:
        create_systemd = True
    
    # Setup steps
    steps = [
        ("Checking Python packages", check_python_packages),
        ("Creating directories", create_directories),
        ("Setting up database", setup_database),
        ("Creating startup scripts", create_startup_scripts),
        ("Creating monitoring script", create_monitoring_script),
        ("Creating documentation", create_documentation),
    ]
    
    if create_systemd:
        steps.append(("Creating systemd services", create_systemd_service))
    
    # Execute setup steps
    success = True
    for step_name, step_func in steps:
        logger.info(f"🔄 {step_name}...")
        if not step_func():
            logger.error(f"❌ {step_name} failed!")
            success = False
        else:
            logger.info(f"✅ {step_name} completed!")
    
    if success:
        logger.info("🎉 Solana system setup completed successfully!")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("1. Review configuration in solana_config.env")
        logger.info("2. Start services: ./start_solana_system.sh")
        logger.info("3. Check status: ./check_solana_status.sh")
        logger.info("4. Monitor health: python3 monitor_solana_system.py")
        logger.info("")
        logger.info("📡 API will be available at:")
        logger.info("   - Main API: http://localhost:8001")
        logger.info("   - Metrics: http://localhost:9091/metrics")
        logger.info("   - WebSocket: ws://localhost:8001/ws")
    else:
        logger.error("❌ Solana system setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
