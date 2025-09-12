#!/usr/bin/env python3
"""
Скрипт для добавления блокчейна Ethereum в базу данных
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models_v2 import Blockchain

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'defimon_db',
    'user': 'defimon',
    'password': 'password'
}

def add_ethereum_blockchain():
    """Добавить блокчейн Ethereum в базу данных"""
    logger.info("Adding Ethereum blockchain to database...")
    
    try:
        # Create engine and session
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if Ethereum already exists
        existing_ethereum = session.query(Blockchain).filter_by(name='Ethereum').first()
        if existing_ethereum:
            logger.info(f"Ethereum blockchain already exists with ID: {existing_ethereum.id}")
            return existing_ethereum.id
        
        # Create Ethereum blockchain
        ethereum = Blockchain(
            name='Ethereum',
            symbol='ETH',
            chain_id=1,
            blockchain_type='mainnet',
            rpc_url='https://mainnet.infura.io/v3/',
            explorer_url='https://etherscan.io',
            native_token='ETH',
            is_active=True,
            launch_date=datetime(2015, 7, 30),  # Ethereum launch date
            description='The world computer - decentralized platform for smart contracts',
            created_at=datetime.utcnow()
        )
        
        session.add(ethereum)
        session.commit()
        
        logger.info(f"✅ Ethereum blockchain added with ID: {ethereum.id}")
        return ethereum.id
        
    except Exception as e:
        logger.error(f"❌ Error adding Ethereum blockchain: {e}")
        session.rollback()
        return None
    finally:
        session.close()

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Добавить Ethereum
    add_ethereum_blockchain()
