#!/usr/bin/env python3
"""
Avalanche Network Real-Time Metrics Server Setup Script
Initializes database, creates tables, and sets up the monitoring system
"""

import os
import sys
import asyncio
import subprocess
from datetime import datetime
from loguru import logger

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.database import get_db_session, engine
from database.models_v2 import Base, Blockchain
from database.blockchain_init import init_blockchains
from config.settings import settings

def check_requirements():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking requirements...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "aiohttp",
        "sqlalchemy",
        "psycopg2-binary",
        "redis",
        "loguru",
        "pydantic",
        "python-dotenv",
        "requests",
        "schedule"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All requirements satisfied")
    return True

def check_environment():
    """Check environment configuration"""
    logger.info("🔧 Checking environment configuration...")
    
    # Check if config.env exists
    if not os.path.exists("config.env"):
        if os.path.exists("avalanche_config.env"):
            logger.info("📋 Copying avalanche_config.env to config.env...")
            import shutil
            shutil.copy("avalanche_config.env", "config.env")
            logger.warning("⚠️  Please update config.env with your actual API keys and configuration")
        else:
            logger.error("❌ config.env not found. Please create it from avalanche_config.env template")
            return False
    
    # Check database connection
    try:
        with get_db_session() as db:
            db.execute("SELECT 1")
        logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.info("Please ensure PostgreSQL is running and DATABASE_URL is correct")
        return False
    
    # Check Redis connection (optional)
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        logger.info("Redis is optional but recommended for caching")
    
    logger.info("✅ Environment configuration check complete")
    return True

def setup_database():
    """Setup database tables and initial data"""
    logger.info("🗄️  Setting up database...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created")
        
        # Initialize blockchains
        init_blockchains()
        logger.info("✅ Blockchains initialized")
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        logger.info("✅ Logs directory created")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        logger.info("✅ Models directory created")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service file for production deployment"""
    logger.info("🔧 Creating systemd service file...")
    
    current_dir = os.path.abspath(os.path.dirname(__file__))
    python_path = sys.executable
    
    service_content = f"""[Unit]
Description=Avalanche Network Real-Time Metrics Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory={current_dir}
Environment=PATH={os.path.dirname(python_path)}
Environment=PYTHONPATH={current_dir}
ExecStart={python_path} {current_dir}/run_avalanche_server.py --mode full
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "/etc/systemd/system/avalanche-metrics.service"
    
    try:
        with open("avalanche-metrics.service", "w") as f:
            f.write(service_content)
        
        logger.info("✅ Systemd service file created: avalanche-metrics.service")
        logger.info("To install the service:")
        logger.info("  sudo cp avalanche-metrics.service /etc/systemd/system/")
        logger.info("  sudo systemctl daemon-reload")
        logger.info("  sudo systemctl enable avalanche-metrics")
        logger.info("  sudo systemctl start avalanche-metrics")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create systemd service: {e}")
        return False

def create_docker_compose():
    """Create Docker Compose configuration"""
    logger.info("🐳 Creating Docker Compose configuration...")
    
    docker_compose_content = """version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: defimon_db
      POSTGRES_USER: defimon
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U defimon"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  avalanche-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://defimon:password@postgres:5432/defimon_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./config.env:/app/config.env
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
"""
    
    try:
        with open("docker-compose.avalanche.yml", "w") as f:
            f.write(docker_compose_content)
        
        logger.info("✅ Docker Compose file created: docker-compose.avalanche.yml")
        logger.info("To run with Docker Compose:")
        logger.info("  docker-compose -f docker-compose.avalanche.yml up -d")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create Docker Compose file: {e}")
        return False

def create_dockerfile():
    """Create Dockerfile for containerized deployment"""
    logger.info("🐳 Creating Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run_avalanche_server.py", "--mode", "full"]
"""
    
    try:
        with open("Dockerfile.avalanche", "w") as f:
            f.write(dockerfile_content)
        
        logger.info("✅ Dockerfile created: Dockerfile.avalanche")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create Dockerfile: {e}")
        return False

def create_requirements():
    """Create requirements.txt file"""
    logger.info("📦 Creating requirements.txt...")
    
    requirements = """# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
loguru==0.7.2
pydantic==2.5.0
python-dotenv==1.0.0
requests==2.31.0
schedule==1.2.0

# Database
alembic==1.13.1

# Monitoring and alerting
prometheus-client==0.19.0

# Data processing
pandas==2.1.4
numpy==1.25.2

# Async utilities
asyncio-mqtt==0.16.1

# Development dependencies (optional)
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
"""
    
    try:
        with open("requirements.avalanche.txt", "w") as f:
            f.write(requirements)
        
        logger.info("✅ Requirements file created: requirements.avalanche.txt")
        logger.info("To install dependencies:")
        logger.info("  pip install -r requirements.avalanche.txt")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to create requirements file: {e}")
        return False

def print_startup_instructions():
    """Print startup instructions"""
    print("\n" + "="*80)
    print("🚀 AVALANCHE NETWORK REAL-TIME METRICS SERVER")
    print("="*80)
    print("Setup completed successfully!")
    print("\n📋 NEXT STEPS:")
    print("1. Update config.env with your API keys and configuration")
    print("2. Ensure PostgreSQL and Redis are running")
    print("3. Start the server with one of the following commands:")
    print("\n🔧 DEVELOPMENT:")
    print("  python run_avalanche_server.py --mode full")
    print("\n🐳 DOCKER:")
    print("  docker-compose -f docker-compose.avalanche.yml up -d")
    print("\n🔧 PRODUCTION (systemd):")
    print("  sudo systemctl start avalanche-metrics")
    print("\n📊 ACCESS THE SERVER:")
    print("  • API Documentation: http://localhost:8000/docs")
    print("  • Health Check: http://localhost:8000/health")
    print("  • Metrics Summary: http://localhost:8000/metrics/summary")
    print("\n🔗 KEY ENDPOINTS:")
    print("  • /metrics/network-performance - Network metrics")
    print("  • /metrics/economic - Economic data")
    print("  • /metrics/defi - DeFi ecosystem metrics")
    print("  • /metrics/subnets - Subnet data")
    print("  • /metrics/security - Security metrics")
    print("  • /metrics/all - All metrics")
    print("  • /historical/24 - Historical data (24 hours)")
    print("\n🚨 MONITORING:")
    print("  • Real-time data collection every 30 seconds to 1 hour")
    print("  • Automatic alerting via email, Slack, Telegram, webhooks")
    print("  • Comprehensive monitoring of 12 metric categories")
    print("\n📈 MONITORED METRICS:")
    print("  • Network Performance (TPS, gas prices, finality)")
    print("  • Economic Data (price, volume, market cap)")
    print("  • DeFi Ecosystem (TVL, protocols, yields)")
    print("  • Subnet Analysis (count, activity, validators)")
    print("  • Security Metrics (validators, staking, audits)")
    print("  • Development Activity (GitHub, contracts)")
    print("  • User Behavior (whales, retail vs institutional)")
    print("  • Competitive Position (market share, performance)")
    print("  • Technical Health (RPC, uptime, infrastructure)")
    print("  • Risk Assessment (centralization, technical, market)")
    print("  • Macro Environment (market conditions, regulations)")
    print("  • Ecosystem Health (community, partnerships)")
    print("="*80)

def main():
    """Main setup function"""
    logger.info("🚀 Starting Avalanche Network Real-Time Metrics Server Setup")
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed")
        return False
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Setup database
    if not setup_database():
        logger.error("❌ Database setup failed")
        return False
    
    # Create deployment files
    create_systemd_service()
    create_docker_compose()
    create_dockerfile()
    create_requirements()
    
    # Print instructions
    print_startup_instructions()
    
    logger.info("✅ Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
