#!/usr/bin/env python3
"""
TRON Metrics Server Setup Script
Sets up the TRON monitoring system with database initialization and configuration
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.tron_models import Base

# Load environment variables
load_dotenv("tron_config.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.info("✅ Logs directory created/verified")

def setup_database():
    """Setup TRON metrics database"""
    try:
        database_url = os.getenv("TRON_DATABASE_URL", "postgresql://defimon:password@localhost:5432/tron_metrics_db")
        engine = create_engine(database_url, echo=False)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection test successful")
            
        return True
    except Exception as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        return False

def install_dependencies():
    """Install required Python dependencies"""
    try:
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "sqlalchemy>=2.0.0",
            "aiohttp>=3.9.0",
            "python-dotenv>=1.0.0",
            "psycopg2-binary>=2.9.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0"
        ]
        
        for requirement in requirements:
            logger.info(f"Installing {requirement}...")
            subprocess.run([sys.executable, "-m", "pip", "install", requirement], 
                         check=True, capture_output=True)
        
        logger.info("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {str(e)}")
        return False

def verify_environment():
    """Verify environment configuration"""
    required_vars = [
        "QUICKNODE_TRON_HTTP_ENDPOINT",
        "TRON_DATABASE_URL",
        "API_PORT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ Environment configuration verified")
    return True

def create_startup_script():
    """Create startup script for the TRON server"""
    startup_script = """#!/bin/bash
# TRON Metrics Server Startup Script

echo "🚀 Starting TRON Metrics Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start the server
python tron_metrics_server.py
"""
    
    with open("start_tron_server.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod("start_tron_server.sh", 0o755)
    logger.info("✅ Startup script created: start_tron_server.sh")

def create_pm2_config():
    """Create PM2 configuration for the TRON server"""
    pm2_config = {
        "apps": [{
            "name": "tron-metrics-server",
            "script": "tron_metrics_server.py",
            "cwd": "/home/vovkes/ETHL2",
            "interpreter": "python3",
            "env": {
                "PYTHONPATH": "/home/vovkes/ETHL2/src"
            },
            "instances": 1,
            "exec_mode": "fork",
            "watch": False,
            "max_memory_restart": "1G",
            "error_file": "./logs/tron-server-error.log",
            "out_file": "./logs/tron-server-out.log",
            "log_file": "./logs/tron-server.log",
            "time": True,
            "autorestart": True,
            "max_restarts": 10,
            "min_uptime": "10s"
        }]
    }
    
    import json
    with open("ecosystem.tron.config.js", "w") as f:
        f.write("module.exports = ")
        f.write(json.dumps(pm2_config, indent=2))
    
    logger.info("✅ PM2 configuration created: ecosystem.tron.config.js")

def main():
    """Main setup function"""
    logger.info("🚀 Starting TRON Metrics Server Setup...")
    
    # Create logs directory
    create_logs_directory()
    
    # Verify environment
    if not verify_environment():
        logger.error("❌ Environment verification failed. Please check your tron_config.env file.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Dependency installation failed.")
        return False
    
    # Setup database
    if not setup_database():
        logger.error("❌ Database setup failed.")
        return False
    
    # Create startup script
    create_startup_script()
    
    # Create PM2 config
    create_pm2_config()
    
    logger.info("🎉 TRON Metrics Server setup completed successfully!")
    logger.info("📋 Next steps:")
    logger.info("   1. Update tron_config.env with your actual API keys and endpoints")
    logger.info("   2. Start the server: ./start_tron_server.sh")
    logger.info("   3. Or use PM2: pm2 start ecosystem.tron.config.js")
    logger.info("   4. Access dashboard: http://localhost:8008/dashboard")
    logger.info("   5. API docs: http://localhost:8008/docs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
