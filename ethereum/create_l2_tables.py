#!/usr/bin/env python3
"""
Скрипт для создания таблиц Layer 2 в базе данных
"""

import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine
from src.database.models_v2 import Base
from src.database.l2_models import *  # Import all L2 models

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'defimon_db',
    'user': 'defimon',
    'password': 'password'
}

def create_l2_tables():
    """Создать таблицы L2 в базе данных"""
    logger.info("Creating L2 tables in database...")
    
    try:
        # Create engine
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        logger.info("✅ L2 tables created successfully!")
        
        # List created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        l2_tables = [table for table in tables if table.startswith('l2_')]
        logger.info(f"Created L2 tables: {l2_tables}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating L2 tables: {e}")
        return False

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Создать таблицы
    create_l2_tables()
