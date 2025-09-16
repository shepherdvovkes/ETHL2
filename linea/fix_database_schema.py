#!/usr/bin/env python3
"""
Fix LINEA Database Schema
Fix integer overflow issues for large blockchain values
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema():
    """Fix the database schema to handle large integers properly"""
    db_path = "linea_archive_data.db"
    
    logger.info("🔧 Fixing LINEA database schema...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create backup of existing data
        logger.info("📋 Creating backup of existing data...")
        cursor.execute("CREATE TABLE IF NOT EXISTS linea_archive_transactions_backup AS SELECT * FROM linea_archive_transactions")
        
        # Drop the problematic index
        logger.info("🗑️ Dropping problematic indexes...")
        try:
            cursor.execute("DROP INDEX IF EXISTS idx_archive_transactions_value")
        except:
            pass
        
        # Recreate transactions table with proper schema
        logger.info("🔄 Recreating transactions table...")
        cursor.execute("DROP TABLE IF EXISTS linea_archive_transactions_new")
        
        cursor.execute("""
            CREATE TABLE linea_archive_transactions_new (
                transaction_hash TEXT PRIMARY KEY,
                block_number INTEGER NOT NULL,
                block_hash TEXT,
                transaction_index INTEGER,
                from_address TEXT,
                to_address TEXT,
                value TEXT,  -- Changed to TEXT for large integers
                gas INTEGER,
                gas_price TEXT,  -- Changed to TEXT for large integers
                max_fee_per_gas TEXT,  -- Changed to TEXT for large integers
                max_priority_fee_per_gas TEXT,  -- Changed to TEXT for large integers
                nonce INTEGER,
                input_data TEXT,
                v TEXT,
                r TEXT,
                s TEXT,
                type INTEGER DEFAULT 2,
                access_list TEXT,
                chain_id INTEGER DEFAULT 59144,
                blob_versioned_hashes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Copy data from backup, converting large integers to text
        logger.info("📊 Migrating data...")
        cursor.execute("""
            INSERT INTO linea_archive_transactions_new 
            SELECT 
                transaction_hash,
                block_number,
                block_hash,
                transaction_index,
                from_address,
                to_address,
                CAST(value AS TEXT),
                gas,
                CAST(gas_price AS TEXT),
                CAST(max_fee_per_gas AS TEXT),
                CAST(max_priority_fee_per_gas AS TEXT),
                nonce,
                input_data,
                v,
                r,
                s,
                type,
                access_list,
                chain_id,
                blob_versioned_hashes,
                created_at,
                updated_at
            FROM linea_archive_transactions_backup
        """)
        
        # Drop old table and rename new one
        logger.info("🔄 Replacing old table...")
        cursor.execute("DROP TABLE linea_archive_transactions")
        cursor.execute("ALTER TABLE linea_archive_transactions_new RENAME TO linea_archive_transactions")
        
        # Recreate indexes
        logger.info("📇 Recreating indexes...")
        cursor.execute("CREATE INDEX idx_archive_transactions_block_number ON linea_archive_transactions(block_number)")
        cursor.execute("CREATE INDEX idx_archive_transactions_from ON linea_archive_transactions(from_address)")
        cursor.execute("CREATE INDEX idx_archive_transactions_to ON linea_archive_transactions(to_address)")
        cursor.execute("CREATE INDEX idx_archive_transactions_type ON linea_archive_transactions(type)")
        
        # Clean up backup
        cursor.execute("DROP TABLE linea_archive_transactions_backup")
        
        conn.commit()
        logger.info("✅ Database schema fixed successfully!")
        
        # Verify the fix
        cursor.execute("SELECT COUNT(*) FROM linea_archive_transactions")
        count = cursor.fetchone()[0]
        logger.info(f"📊 Total transactions: {count:,}")
        
        # Test a sample query
        cursor.execute("SELECT transaction_hash, value, gas_price FROM linea_archive_transactions LIMIT 3")
        samples = cursor.fetchall()
        logger.info("📋 Sample data:")
        for sample in samples:
            logger.info(f"   TX: {sample[0][:10]}..., Value: {sample[1]}, Gas Price: {sample[2]}")
        
    except Exception as e:
        logger.error(f"❌ Error fixing schema: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database_schema()
