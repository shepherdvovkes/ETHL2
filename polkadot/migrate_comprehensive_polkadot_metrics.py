#!/usr/bin/env python3
"""
Database Migration Script for Comprehensive Polkadot Metrics
Adds all new comprehensive metrics tables to the database
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.database import engine, SessionLocal
from database.polkadot_models import (
    Base, PolkadotSecurityMetrics, PolkadotValidatorMetrics,
    PolkadotParachainSlotMetrics, PolkadotCrossChainAdvancedMetrics,
    PolkadotGovernanceAdvancedMetrics, PolkadotEconomicAdvancedMetrics,
    PolkadotInfrastructureMetrics
)
from loguru import logger

def migrate_database():
    """Create all new comprehensive metrics tables"""
    try:
        logger.info("Starting comprehensive Polkadot metrics database migration...")
        
        # Create all new tables
        Base.metadata.create_all(bind=engine, tables=[
            PolkadotSecurityMetrics.__table__,
            PolkadotValidatorMetrics.__table__,
            PolkadotParachainSlotMetrics.__table__,
            PolkadotCrossChainAdvancedMetrics.__table__,
            PolkadotGovernanceAdvancedMetrics.__table__,
            PolkadotEconomicAdvancedMetrics.__table__,
            PolkadotInfrastructureMetrics.__table__
        ])
        
        logger.info("✅ Successfully created all comprehensive metrics tables:")
        logger.info("  - polkadot_security_metrics")
        logger.info("  - polkadot_validator_metrics")
        logger.info("  - polkadot_parachain_slot_metrics")
        logger.info("  - polkadot_cross_chain_advanced_metrics")
        logger.info("  - polkadot_governance_advanced_metrics")
        logger.info("  - polkadot_economic_advanced_metrics")
        logger.info("  - polkadot_infrastructure_metrics")
        
        # Verify tables were created
        db = SessionLocal()
        try:
            # Test query to verify tables exist
            result = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'polkadot_%'")
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"📊 Total Polkadot tables in database: {len(tables)}")
            for table in sorted(tables):
                logger.info(f"  ✓ {table}")
                
        except Exception as e:
            logger.error(f"Error verifying tables: {e}")
        finally:
            db.close()
        
        logger.info("🎉 Database migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
