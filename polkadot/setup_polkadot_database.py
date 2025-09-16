#!/usr/bin/env python3
"""
Setup Polkadot Database Tables
Creates all necessary tables for Polkadot and parachain metrics
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger

from database.database import Base, engine
from database.polkadot_models import (
    PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
    PolkadotStakingMetrics, PolkadotGovernanceMetrics,
    PolkadotEconomicMetrics, ParachainMetrics,
    ParachainCrossChainMetrics, PolkadotEcosystemMetrics,
    PolkadotPerformanceMetrics
)

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def create_polkadot_tables():
    """Create all Polkadot-related tables"""
    try:
        logger.info("Creating Polkadot database tables...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.success("Polkadot database tables created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

def initialize_polkadot_network():
    """Initialize Polkadot network in database"""
    try:
        from database.database import SessionLocal
        
        db = SessionLocal()
        
        # Check if Polkadot network already exists
        existing_network = db.query(PolkadotNetwork).filter(
            PolkadotNetwork.name == "Polkadot"
        ).first()
        
        if not existing_network:
            # Create Polkadot network
            polkadot_network = PolkadotNetwork(
                name="Polkadot",
                chain_id="polkadot",
                rpc_endpoint="https://rpc.polkadot.io",
                ws_endpoint="wss://rpc.polkadot.io",
                is_mainnet=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(polkadot_network)
            db.commit()
            db.refresh(polkadot_network)
            
            logger.success(f"Created Polkadot network with ID: {polkadot_network.id}")
        else:
            logger.info(f"Polkadot network already exists with ID: {existing_network.id}")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Error initializing Polkadot network: {e}")
        raise

def initialize_parachains():
    """Initialize parachains in database"""
    try:
        from database.database import SessionLocal
        
        db = SessionLocal()
        
        # Get Polkadot network
        polkadot_network = db.query(PolkadotNetwork).filter(
            PolkadotNetwork.name == "Polkadot"
        ).first()
        
        if not polkadot_network:
            logger.error("Polkadot network not found. Please run initialize_polkadot_network() first.")
            return
        
        # Top 20 most active parachains
        parachains_data = [
            {"id": 2004, "name": "Moonbeam", "symbol": "GLMR", "status": "active"},
            {"id": 2026, "name": "Nodle", "symbol": "NODL", "status": "active"},
            {"id": 2035, "name": "Phala Network", "symbol": "PHA", "status": "active"},
            {"id": 2091, "name": "Frequency", "symbol": "FRQCY", "status": "active"},
            {"id": 2046, "name": "NeuroWeb", "symbol": "NEURO", "status": "active"},
            {"id": 2034, "name": "HydraDX", "symbol": "HDX", "status": "active"},
            {"id": 2030, "name": "Bifrost", "symbol": "BNC", "status": "active"},
            {"id": 1000, "name": "AssetHub", "symbol": "DOT", "status": "active"},
            {"id": 2006, "name": "Astar", "symbol": "ASTR", "status": "active"},
            {"id": 2104, "name": "Manta", "symbol": "MANTA", "status": "active"},
            {"id": 2000, "name": "Acala", "symbol": "ACA", "status": "active"},
            {"id": 2012, "name": "Parallel", "symbol": "PARA", "status": "active"},
            {"id": 2002, "name": "Clover", "symbol": "CLV", "status": "active"},
            {"id": 2013, "name": "Litentry", "symbol": "LIT", "status": "active"},
            {"id": 2011, "name": "Equilibrium", "symbol": "EQ", "status": "active"},
            {"id": 2018, "name": "SubDAO", "symbol": "GOV", "status": "active"},
            {"id": 2092, "name": "Zeitgeist", "symbol": "ZTG", "status": "active"},
            {"id": 2121, "name": "Efinity", "symbol": "EFI", "status": "active"},
            {"id": 2019, "name": "Composable", "symbol": "LAYR", "status": "active"},
            {"id": 2085, "name": "KILT Protocol", "symbol": "KILT", "status": "active"}
        ]
        
        created_count = 0
        updated_count = 0
        
        for parachain_data in parachains_data:
            # Check if parachain already exists
            existing_parachain = db.query(Parachain).filter(
                Parachain.parachain_id == parachain_data["id"]
            ).first()
            
            if not existing_parachain:
                # Create new parachain
                parachain = Parachain(
                    parachain_id=parachain_data["id"],
                    name=parachain_data["name"],
                    symbol=parachain_data["symbol"],
                    network_id=polkadot_network.id,
                    status=parachain_data["status"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                db.add(parachain)
                created_count += 1
                logger.info(f"Created parachain: {parachain_data['name']} (ID: {parachain_data['id']})")
            else:
                # Update existing parachain
                existing_parachain.name = parachain_data["name"]
                existing_parachain.symbol = parachain_data["symbol"]
                existing_parachain.status = parachain_data["status"]
                existing_parachain.updated_at = datetime.utcnow()
                updated_count += 1
                logger.info(f"Updated parachain: {parachain_data['name']} (ID: {parachain_data['id']})")
        
        db.commit()
        db.close()
        
        logger.success(f"Parachains initialization completed: {created_count} created, {updated_count} updated")
        
    except Exception as e:
        logger.error(f"Error initializing parachains: {e}")
        raise

def verify_database_setup():
    """Verify that all tables and data are properly set up"""
    try:
        from database.database import SessionLocal
        
        db = SessionLocal()
        
        # Check tables
        tables_to_check = [
            PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
            PolkadotStakingMetrics, PolkadotGovernanceMetrics,
            PolkadotEconomicMetrics, ParachainMetrics,
            ParachainCrossChainMetrics, PolkadotEcosystemMetrics,
            PolkadotPerformanceMetrics
        ]
        
        logger.info("Verifying database setup...")
        
        for table in tables_to_check:
            count = db.query(table).count()
            logger.info(f"Table {table.__tablename__}: {count} records")
        
        # Check specific data
        polkadot_network = db.query(PolkadotNetwork).filter(
            PolkadotNetwork.name == "Polkadot"
        ).first()
        
        if polkadot_network:
            logger.success(f"Polkadot network found: ID {polkadot_network.id}")
        else:
            logger.warning("Polkadot network not found")
        
        parachain_count = db.query(Parachain).count()
        logger.success(f"Parachains in database: {parachain_count}")
        
        db.close()
        
        logger.success("Database setup verification completed!")
        
    except Exception as e:
        logger.error(f"Error verifying database setup: {e}")
        raise

def main():
    """Main setup function"""
    try:
        logger.info("Starting Polkadot database setup...")
        
        # Create tables
        create_polkadot_tables()
        
        # Initialize network
        initialize_polkadot_network()
        
        # Initialize parachains
        initialize_parachains()
        
        # Verify setup
        verify_database_setup()
        
        logger.success("Polkadot database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
