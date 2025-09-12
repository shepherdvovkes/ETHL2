#!/usr/bin/env python3
"""
DEFIMON Analytics Backend v2.0
Advanced crypto analytics backend with machine learning
"""

import asyncio
import uvicorn
import sys
import os
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings

def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        "logs/backend.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )

def check_environment():
    """Check environment configuration"""
    logger.info("Checking environment configuration...")
    
    # Check required environment variables
    required_vars = [
        "QUICKNODE_API_KEY",
        "QUICKNODE_HTTP_ENDPOINT",
        "DATABASE_URL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(settings, var, None):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some features may not work properly")
    
    # Check optional variables
    optional_vars = [
        "ETHERSCAN_API_KEY",
        "COINGECKO_API_KEY",
        "GITHUB_CLIENT_ID",
        "GITHUB_CLIENT_SECRET",
        "HF_TOKEN"
    ]
    
    available_optional = []
    for var in optional_vars:
        if getattr(settings, var, None):
            available_optional.append(var)
    
    logger.info(f"Available optional services: {available_optional}")
    
    return len(missing_vars) == 0

def print_startup_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 DEFIMON Analytics Backend v2.0                        ║
    ║                                                              ║
    ║    Advanced Crypto Analytics with Machine Learning          ║
    ║                                                              ║
    ║    Features:                                                 ║
    ║    ✅ 50+ Blockchain Support                                ║
    ║    ✅ 10 Categories of Metrics                              ║
    ║    ✅ Machine Learning Predictions                          ║
    ║    ✅ Real-time Data Collection                             ║
    ║    ✅ GitHub Integration                                    ║
    ║    ✅ Advanced Analytics                                    ║
    ║                                                              ║
    ║    API Documentation: http://localhost:8000/docs            ║
    ║    GitHub Auth: http://localhost:8000/auth/github           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def initialize_backend():
    """Initialize backend components"""
    logger.info("Initializing DEFIMON Analytics Backend...")
    
    try:
        # Import and initialize database
        from database.database import init_db
        init_db()
        logger.info("✅ Database initialized")
        
        # Initialize blockchain clients
        from api.blockchain_client import BlockchainClient
        supported_blockchains = BlockchainClient.get_supported_blockchains()
        logger.info(f"✅ {len(supported_blockchains)} blockchains supported")
        
        # Initialize metrics mapper
        from api.metrics_mapper import MetricsMapper
        metrics_mapper = MetricsMapper()
        metrics_summary = metrics_mapper.get_metrics_summary()
        logger.info(f"✅ {metrics_summary['total_metrics']} metrics configured")
        
        # Initialize ML pipeline
        from ml.ml_pipeline import CryptoMLPipeline
        ml_pipeline = CryptoMLPipeline()
        await ml_pipeline.load_models()
        logger.info("✅ ML pipeline initialized")
        
        logger.info("🎉 Backend initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Backend initialization failed: {e}")
        raise

def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Print banner
    print_startup_banner()
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        logger.warning("Environment check failed, but continuing...")
    
    # Initialize backend
    try:
        asyncio.run(initialize_backend())
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        sys.exit(1)
    
    # Start the server
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    
    try:
        uvicorn.run(
            "src.api.backend_api:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=False,  # Set to True for development
            workers=settings.API_WORKERS,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
