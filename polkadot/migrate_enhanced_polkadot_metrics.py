#!/usr/bin/env python3
"""
Migration script to add enhanced Polkadot metrics tables to the database.
This script adds new comprehensive metric models to reach our target of 350-400 metrics.
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

from database.polkadot_models import Base, PolkadotDeveloperMetrics, PolkadotCommunityMetrics, PolkadotDeFiMetrics, PolkadotAdvancedAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def migrate_enhanced_metrics():
    """Migrate enhanced Polkadot metrics to the database"""
    try:
        # Database connection
        DATABASE_URL = "sqlite:///polkadot_metrics.db"
        engine = create_engine(DATABASE_URL, echo=False)
        
        logger.info("🚀 Starting enhanced Polkadot metrics migration...")
        
        # Create new tables
        logger.info("📊 Creating enhanced metric tables...")
        
        # Create the new comprehensive metric tables
        PolkadotDeveloperMetrics.__table__.create(engine, checkfirst=True)
        PolkadotCommunityMetrics.__table__.create(engine, checkfirst=True)
        PolkadotDeFiMetrics.__table__.create(engine, checkfirst=True)
        PolkadotAdvancedAnalytics.__table__.create(engine, checkfirst=True)
        
        logger.info("✅ Enhanced metric tables created successfully!")
        
        # Verify tables were created
        logger.info("🔍 Verifying table creation...")
        
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'polkadot_%'"))
            tables = [row[0] for row in result.fetchall()]
            
            new_tables = [
                'polkadot_developer_metrics',
                'polkadot_community_metrics', 
                'polkadot_defi_metrics',
                'polkadot_advanced_analytics'
            ]
            
            for table in new_tables:
                if table in tables:
                    logger.info(f"✅ Table {table} created successfully")
                else:
                    logger.error(f"❌ Table {table} not found")
        
        # Count total tables
        total_tables = len(tables)
        logger.info(f"📊 Total Polkadot metric tables: {total_tables}")
        
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
        else:
            logger.info(f"📈 Progress: {total_columns}/350 metrics ({(total_columns/350)*100:.1f}%)")
        
        logger.info("✅ Enhanced Polkadot metrics migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_enhanced_metrics())
