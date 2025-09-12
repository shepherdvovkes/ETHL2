#!/usr/bin/env python3
"""
Создание правильной схемы базы данных для DEFIMON v2
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import init_db, engine
from database.models_v2 import Base

def create_database_schema():
    """Создать схему базы данных"""
    print("🗄️  Creating database schema...")
    
    try:
        # Создать все таблицы
        Base.metadata.create_all(bind=engine)
        print("✅ Database schema created successfully!")
        
        # Проверить созданные таблицы
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"📊 Created {len(tables)} tables:")
        for table in sorted(tables):
            print(f"   - {table}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating database schema: {e}")
        return False

if __name__ == "__main__":
    success = create_database_schema()
    if success:
        print("🎉 Database setup completed!")
    else:
        print("❌ Database setup failed!")
        sys.exit(1)
