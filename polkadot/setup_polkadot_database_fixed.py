#!/usr/bin/env python3
"""
Fixed Polkadot Database Setup Script
Creates the correct database schema for Polkadot metrics
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.database import engine, SessionLocal
from database.polkadot_comprehensive_models import Base
from sqlalchemy import text
from loguru import logger

def setup_database():
    """Setup the database with correct schema"""
    try:
        logger.info("Setting up Polkadot database...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Test database connection
        db = SessionLocal()
        try:
            # Test query
            result = db.execute(text("SELECT 1")).fetchone()
            logger.info(f"Database connection test: {result}")
        finally:
            db.close()
        
        logger.success("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_database()
