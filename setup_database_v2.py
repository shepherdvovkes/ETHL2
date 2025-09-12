#!/usr/bin/env python3
"""
Database setup script for Ethereum L2 Analytics System
"""

import os
import sys
import asyncio
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with placeholder values
QUICKNODE_API_KEY=your_quicknode_api_key_here
QUICKNODE_HTTP_ENDPOINT=your_quicknode_http_endpoint_here
QUICKNODE_WSS_ENDPOINT=your_quicknode_wss_endpoint_here
ETHERSCAN_API_KEY=753BZTQQDZ1B6TYNDUPQAZHPDWSMWXUXGQ
INFURA_API_KEY=your_infura_api_key_here
HF_TOKEN=your_huggingface_token_here
GITHUB_TOKEN=your_github_token_here
COINGECKO_API_KEY=your_coingecko_api_key_here

# Database Configuration
DATABASE_URL=postgresql://defimon:password@localhost:5432/defimon_db
REDIS_URL=redis://localhost:6379

async def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='defimon',
            password='password',
            database='postgres'  # Connect to default postgres database
        )
        
        # Check if database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", 'defimon_db'
        )
        
        if not result:
            # Create database
            await conn.execute('CREATE DATABASE defimon_db')
            logger.info("Database 'defimon_db' created successfully")
        else:
            logger.info("Database 'defimon_db' already exists")
            
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

async def setup_tables():
    """Set up database tables"""
    try:
        engine = create_async_engine(DATABASE_URL)
        
        # Create tables using SQLAlchemy models
        from src.database.models_v2 import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error setting up tables: {e}")
        raise

async def main():
    """Main setup function"""
    try:
        logger.info("Starting database setup...")
        
        # Create database
        await create_database()
        
        # Setup tables
        await setup_tables()
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
