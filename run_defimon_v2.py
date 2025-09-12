#!/usr/bin/env python3
"""
DEFIMON v2 - Запуск системы аналитики крипто-активов
Поддерживает 50+ блокчейнов и все метрики
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# Add src to path
sys.path.append('src')

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Checking dependencies...")
    
    required_files = [
        "src/database/models_v2.py",
        "src/database/blockchain_init.py",
        "migrate_database.py",
        "setup_database_v2.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found!")
    return True

def check_database():
    """Проверка базы данных"""
    print("🗄️  Checking database connection...")
    
    try:
        from src.database.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def setup_database():
    """Настройка базы данных"""
    print("⚙️  Setting up database...")
    
    try:
        # Check if migration is needed
        if os.path.exists("data/defimon.db"):
            print("📦 Found existing SQLite database. Running migration...")
            result = subprocess.run([sys.executable, "migrate_database.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Migration completed successfully!")
            else:
                print(f"❌ Migration failed: {result.stderr}")
                return False
        else:
            print("🆕 Setting up new database...")
            result = subprocess.run([sys.executable, "setup_database_v2.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Database setup completed successfully!")
            else:
                print(f"❌ Database setup failed: {result.stderr}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def start_data_collector():
    """Запуск сборщика данных"""
    print("📊 Starting data collector...")
    
    try:
        from src.worker.data_collector import DataCollectorWorker
        collector = DataCollectorWorker()
        
        # Start in background
        import threading
        collector_thread = threading.Thread(target=collector.start_collection, daemon=True)
        collector_thread.start()
        
        print("✅ Data collector started!")
        return True
    except Exception as e:
        print(f"❌ Failed to start data collector: {e}")
        return False

def start_api_server():
    """Запуск API сервера"""
    print("🚀 Starting API server...")
    
    try:
        from src.main import app
        import uvicorn
        
        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def show_dashboard_info():
    """Показать информацию о дашборде"""
    print("\n" + "="*60)
    print("🎉 DEFIMON v2 - Analytics System Started!")
    print("="*60)
    print("🌐 Supported Blockchains: 50+")
    print("📊 Metrics Categories: 10")
    print("🤖 ML Models: 4")
    print("📈 Real-time Analytics: Enabled")
    print("="*60)
    print("\n📱 Dashboard Access:")
    print("   • Web Interface: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("\n🔗 API Endpoints:")
    print("   • GET /api/assets - List all assets")
    print("   • GET /api/analytics/{asset_id} - Asset analytics")
    print("   • GET /api/predictions - ML predictions")
    print("   • GET /api/competitors - Competitor analysis")
    print("   • GET /api/blockchains - Supported blockchains")
    print("\n📊 Metrics Available:")
    print("   • On-chain metrics (TVL, transactions, addresses)")
    print("   • Financial metrics (price, volume, volatility)")
    print("   • Tokenomics (supply, distribution, vesting)")
    print("   • GitHub metrics (commits, PRs, contributors)")
    print("   • Security metrics (audits, vulnerabilities)")
    print("   • Community metrics (social, engagement)")
    print("   • Partnership metrics (integrations, listings)")
    print("   • Network metrics (performance, security)")
    print("   • Trending metrics (momentum, sentiment)")
    print("   • Cross-chain metrics (bridges, liquidity)")
    print("\n🤖 ML Predictions:")
    print("   • Investment Score (0-1)")
    print("   • Price Predictions (1d, 7d, 30d)")
    print("   • Risk Assessment")
    print("   • Growth Potential")
    print("\n🛠️  Management:")
    print("   • POST /api/retrain - Retrain ML models")
    print("   • GET /api/stats - Collection statistics")
    print("   • GET /api/health - System health")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60)

def main():
    """Main function"""
    print("🚀 DEFIMON v2 - Crypto Analytics System")
    print("=" * 50)
    print("🌐 Supporting 50+ blockchains")
    print("📊 Comprehensive metrics collection")
    print("🤖 Advanced ML predictions")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        return 1
    
    # Setup database
    if not setup_database():
        print("❌ Database setup failed!")
        return 1
    
    # Check database connection
    if not check_database():
        print("❌ Database connection failed!")
        return 1
    
    # Start data collector
    if not start_data_collector():
        print("❌ Failed to start data collector!")
        return 1
    
    # Show dashboard info
    show_dashboard_info()
    
    # Start API server
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DEFIMON v2...")
        print("✅ System stopped successfully!")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
