#!/usr/bin/env python3
"""
Complete migration script to create all Polkadot metrics tables.
This script creates all comprehensive metric models to reach our target of 350-400 metrics.
"""

import sys
import os
import asyncio
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.polkadot_models import Base

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def migrate_all_metrics():
    """Create all Polkadot metrics tables"""
    try:
        # Database connection
        DATABASE_URL = "sqlite:///polkadot_metrics.db"
        engine = create_engine(DATABASE_URL, echo=False)
        
        logger.info("🚀 Starting complete Polkadot metrics migration...")
        
        # Create all tables
        logger.info("📊 Creating all metric tables...")
        Base.metadata.create_all(engine)
        
        logger.info("✅ All metric tables created successfully!")
        
        # Verify tables were created
        logger.info("🔍 Verifying table creation...")
        
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"📊 Total tables created: {len(tables)}")
            
            # Count total columns across all tables
            total_columns = 0
            for table in tables:
                result = conn.execute(text(f"PRAGMA table_info({table})"))
                columns = result.fetchall()
                total_columns += len(columns)
                logger.info(f"📋 Table {table}: {len(columns)} columns")
        
        logger.info(f"🎯 Total database columns: {total_columns}")
        
        if total_columns >= 350:
            logger.info("🎉 SUCCESS: Target of 350+ metrics achieved!")
        elif total_columns >= 300:
            logger.info("📈 EXCELLENT: 300+ metrics achieved!")
        else:
            logger.info(f"📈 Progress: {total_columns}/350 metrics ({(total_columns/350)*100:.1f}%)")
        
        logger.info("✅ Complete Polkadot metrics migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_all_metrics())
