#!/usr/bin/env python3
"""
DEFIMON Analytics System Runner
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import pandas
        import numpy
        import torch
        import sklearn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models",
        "data",
        "src/web/dist"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_config():
    """Check if configuration file exists"""
    config_file = "config.env"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found")
        print("Please create config.env with your API keys")
        return False
    
    print("✅ Configuration file found")
    return True

def start_database():
    """Start PostgreSQL and Redis (if needed)"""
    print("📊 Database setup:")
    print("   - Make sure PostgreSQL is running on localhost:5432")
    print("   - Make sure Redis is running on localhost:6379")
    print("   - Create database 'defimon_db' if it doesn't exist")

def main():
    """Main function to run DEFIMON system"""
    print("🚀 Starting DEFIMON Analytics System...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check configuration
    if not check_config():
        sys.exit(1)
    
    # Database setup
    start_database()
    
    print("\n🎯 Starting DEFIMON API Server...")
    print("   - API will be available at: http://localhost:8000")
    print("   - Web interface at: http://localhost:8000/static/index.html")
    print("   - API docs at: http://localhost:8000/docs")
    print("\n📈 Features:")
    print("   - Real-time data collection from QuickNode (Polygon)")
    print("   - Etherscan integration for contract analysis")
    print("   - ML-powered investment scoring")
    print("   - Web dashboard for analytics")
    print("\n" + "=" * 50)
    
    # Change to src directory and run main.py
    os.chdir("src")
    
    try:
        # Run the FastAPI application
        subprocess.run([
            sys.executable, "main.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 DEFIMON system stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running DEFIMON: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
