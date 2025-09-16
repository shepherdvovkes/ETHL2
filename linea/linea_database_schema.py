#!/usr/bin/env python3
"""
LINEA Database Schema Setup
Creates comprehensive database schema for LINEA blockchain data collection
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LineaDatabaseSchema:
    """Database schema setup for LINEA data collection"""
    
    def __init__(self, db_path: str = "linea_archive_data.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=20000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        logger.info(f"Connected to database: {self.db_path}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_tables(self):
        """Create all database tables"""
        logger.info("Creating LINEA database tables...")
        
        # Archive blocks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_blocks (
                block_number INTEGER PRIMARY KEY,
                block_hash TEXT UNIQUE,
                parent_hash TEXT,
                timestamp DATETIME,
                gas_limit INTEGER DEFAULT 0,
                gas_used INTEGER DEFAULT 0,
                base_fee_per_gas INTEGER DEFAULT 0,
                difficulty INTEGER DEFAULT 0,
                total_difficulty TEXT,
                size INTEGER DEFAULT 0,
                transaction_count INTEGER DEFAULT 0,
                extra_data TEXT,
                mix_hash TEXT,
                nonce TEXT,
                receipts_root TEXT,
                sha3_uncles TEXT,
                state_root TEXT,
                transactions_root TEXT,
                withdrawals_root TEXT,
                withdrawals TEXT,
                blob_gas_used INTEGER DEFAULT 0,
                excess_blob_gas INTEGER DEFAULT 0,
                parent_beacon_block_root TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive transactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_transactions (
                transaction_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                block_hash TEXT,
                transaction_index INTEGER,
                from_address TEXT,
                to_address TEXT,
                value TEXT,
                gas INTEGER DEFAULT 0,
                gas_price INTEGER DEFAULT 0,
                max_fee_per_gas INTEGER DEFAULT 0,
                max_priority_fee_per_gas INTEGER DEFAULT 0,
                nonce INTEGER DEFAULT 0,
                input_data TEXT,
                v TEXT,
                r TEXT,
                s TEXT,
                type INTEGER DEFAULT 2,
                access_list TEXT,
                chain_id INTEGER DEFAULT 59144,
                blob_versioned_hashes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (block_number) REFERENCES linea_archive_blocks(block_number)
            )
        """)
        
        # Archive transaction receipts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_transaction_receipts (
                transaction_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                block_hash TEXT,
                transaction_index INTEGER,
                from_address TEXT,
                to_address TEXT,
                cumulative_gas_used INTEGER DEFAULT 0,
                effective_gas_price INTEGER DEFAULT 0,
                gas_used INTEGER DEFAULT 0,
                contract_address TEXT,
                logs TEXT,
                logs_bloom TEXT,
                status INTEGER DEFAULT 1,
                type INTEGER DEFAULT 2,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transaction_hash) REFERENCES linea_archive_transactions(transaction_hash)
            )
        """)
        
        # Archive network metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_network_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                block_number INTEGER,
                tps REAL DEFAULT 0,
                block_time_avg REAL DEFAULT 2.0,
                block_time_std REAL DEFAULT 0,
                gas_utilization REAL DEFAULT 0,
                gas_price_avg REAL DEFAULT 0,
                gas_price_median REAL DEFAULT 0,
                gas_price_std REAL DEFAULT 0,
                transaction_count INTEGER DEFAULT 0,
                unique_addresses_count INTEGER DEFAULT 0,
                contract_creation_count INTEGER DEFAULT 0,
                total_gas_used INTEGER DEFAULT 0,
                total_gas_limit INTEGER DEFAULT 0,
                total_fees INTEGER DEFAULT 0,
                network_hashrate REAL DEFAULT 0,
                network_difficulty REAL DEFAULT 0,
                pending_transactions_count INTEGER DEFAULT 0,
                mempool_size INTEGER DEFAULT 0,
                defi_tvl REAL DEFAULT 0,
                defi_protocols_count INTEGER DEFAULT 0,
                dex_volume_24h REAL DEFAULT 0,
                bridge_volume_24h REAL DEFAULT 0,
                nft_transactions_24h INTEGER DEFAULT 0,
                nft_volume_24h REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive accounts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_accounts (
                address TEXT PRIMARY KEY,
                balance TEXT DEFAULT '0',
                nonce INTEGER DEFAULT 0,
                code TEXT,
                storage_root TEXT,
                is_contract BOOLEAN DEFAULT FALSE,
                contract_name TEXT,
                contract_symbol TEXT,
                contract_decimals INTEGER,
                contract_total_supply TEXT,
                first_seen_block INTEGER,
                last_seen_block INTEGER,
                transaction_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive contracts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_contracts (
                address TEXT PRIMARY KEY,
                creator_address TEXT,
                creation_block INTEGER,
                creation_transaction TEXT,
                bytecode TEXT,
                source_code TEXT,
                abi TEXT,
                contract_name TEXT,
                contract_type TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN DEFAULT FALSE,
                runs INTEGER DEFAULT 200,
                license TEXT,
                is_verified BOOLEAN DEFAULT FALSE,
                verification_date DATETIME,
                proxy_implementation TEXT,
                proxy_type TEXT,
                is_proxy BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive tokens table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_tokens (
                address TEXT PRIMARY KEY,
                name TEXT,
                symbol TEXT,
                decimals INTEGER DEFAULT 18,
                total_supply TEXT DEFAULT '0',
                circulating_supply TEXT DEFAULT '0',
                token_type TEXT DEFAULT 'ERC20',
                is_native BOOLEAN DEFAULT FALSE,
                first_seen_block INTEGER,
                last_seen_block INTEGER,
                holder_count INTEGER DEFAULT 0,
                transfer_count INTEGER DEFAULT 0,
                volume_24h TEXT DEFAULT '0',
                volume_7d TEXT DEFAULT '0',
                volume_30d TEXT DEFAULT '0',
                price_usd REAL DEFAULT 0,
                market_cap_usd REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive DeFi protocols table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_defi_protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_name TEXT,
                protocol_address TEXT,
                protocol_type TEXT,
                tvl_usd REAL DEFAULT 0,
                tvl_tokens TEXT,
                total_volume_24h REAL DEFAULT 0,
                total_volume_7d REAL DEFAULT 0,
                total_volume_30d REAL DEFAULT 0,
                active_users_24h INTEGER DEFAULT 0,
                active_users_7d INTEGER DEFAULT 0,
                transaction_count_24h INTEGER DEFAULT 0,
                transaction_count_7d INTEGER DEFAULT 0,
                fees_24h REAL DEFAULT 0,
                fees_7d REAL DEFAULT 0,
                revenue_24h REAL DEFAULT 0,
                revenue_7d REAL DEFAULT 0,
                block_number INTEGER,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive bridge transactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_bridge_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_hash TEXT,
                bridge_type TEXT,
                from_chain TEXT,
                to_chain TEXT,
                from_address TEXT,
                to_address TEXT,
                token_address TEXT,
                token_symbol TEXT,
                amount TEXT,
                amount_usd REAL DEFAULT 0,
                fee_amount TEXT,
                fee_usd REAL DEFAULT 0,
                block_number INTEGER,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Archive progress tracking table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_archive_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_type TEXT,
                start_block INTEGER,
                end_block INTEGER,
                current_block INTEGER,
                total_blocks INTEGER,
                completed_blocks INTEGER DEFAULT 0,
                failed_blocks INTEGER DEFAULT 0,
                status TEXT,
                progress_percentage REAL DEFAULT 0,
                started_at DATETIME,
                last_updated_at DATETIME,
                completed_at DATETIME,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time blocks table (for current data)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_blocks (
                block_number INTEGER PRIMARY KEY,
                block_hash TEXT UNIQUE,
                parent_hash TEXT,
                timestamp DATETIME,
                gas_limit INTEGER DEFAULT 0,
                gas_used INTEGER DEFAULT 0,
                base_fee_per_gas INTEGER DEFAULT 0,
                difficulty INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                transaction_count INTEGER DEFAULT 0,
                extra_data TEXT,
                mix_hash TEXT,
                nonce TEXT,
                receipts_root TEXT,
                sha3_uncles TEXT,
                state_root TEXT,
                transactions_root TEXT,
                withdrawals_root TEXT,
                withdrawals TEXT,
                blob_gas_used INTEGER DEFAULT 0,
                excess_blob_gas INTEGER DEFAULT 0,
                parent_beacon_block_root TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time transactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_transactions (
                transaction_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                block_hash TEXT,
                transaction_index INTEGER,
                from_address TEXT,
                to_address TEXT,
                value TEXT,
                gas INTEGER DEFAULT 0,
                gas_price INTEGER DEFAULT 0,
                max_fee_per_gas INTEGER DEFAULT 0,
                max_priority_fee_per_gas INTEGER DEFAULT 0,
                nonce INTEGER DEFAULT 0,
                input_data TEXT,
                v TEXT,
                r TEXT,
                s TEXT,
                type INTEGER DEFAULT 2,
                access_list TEXT,
                chain_id INTEGER DEFAULT 59144,
                blob_versioned_hashes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (block_number) REFERENCES linea_blocks(block_number)
            )
        """)
        
        # Real-time transaction receipts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_transaction_receipts (
                transaction_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                block_hash TEXT,
                transaction_index INTEGER,
                from_address TEXT,
                to_address TEXT,
                cumulative_gas_used INTEGER DEFAULT 0,
                effective_gas_price INTEGER DEFAULT 0,
                gas_used INTEGER DEFAULT 0,
                contract_address TEXT,
                logs TEXT,
                logs_bloom TEXT,
                status INTEGER DEFAULT 1,
                type INTEGER DEFAULT 2,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transaction_hash) REFERENCES linea_transactions(transaction_hash)
            )
        """)
        
        # Real-time network metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_network_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                block_number INTEGER,
                tps REAL DEFAULT 0,
                block_time_avg REAL DEFAULT 2.0,
                gas_utilization REAL DEFAULT 0,
                gas_price_avg REAL DEFAULT 0,
                gas_price_median REAL DEFAULT 0,
                transaction_count INTEGER DEFAULT 0,
                unique_addresses_count INTEGER DEFAULT 0,
                contract_creation_count INTEGER DEFAULT 0,
                total_gas_used INTEGER DEFAULT 0,
                total_gas_limit INTEGER DEFAULT 0,
                total_fees INTEGER DEFAULT 0,
                pending_transactions_count INTEGER DEFAULT 0,
                mempool_size INTEGER DEFAULT 0,
                defi_tvl REAL DEFAULT 0,
                defi_protocols_count INTEGER DEFAULT 0,
                dex_volume_24h REAL DEFAULT 0,
                bridge_volume_24h REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time accounts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_accounts (
                address TEXT PRIMARY KEY,
                balance TEXT DEFAULT '0',
                nonce INTEGER DEFAULT 0,
                code TEXT,
                storage_root TEXT,
                is_contract BOOLEAN DEFAULT FALSE,
                contract_name TEXT,
                contract_symbol TEXT,
                contract_decimals INTEGER,
                contract_total_supply TEXT,
                first_seen_block INTEGER,
                last_seen_block INTEGER,
                transaction_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time contracts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_contracts (
                address TEXT PRIMARY KEY,
                creator_address TEXT,
                creation_block INTEGER,
                creation_transaction TEXT,
                bytecode TEXT,
                source_code TEXT,
                abi TEXT,
                contract_name TEXT,
                contract_type TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN DEFAULT FALSE,
                runs INTEGER DEFAULT 200,
                license TEXT,
                is_verified BOOLEAN DEFAULT FALSE,
                verification_date DATETIME,
                proxy_implementation TEXT,
                proxy_type TEXT,
                is_proxy BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time tokens table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_tokens (
                address TEXT PRIMARY KEY,
                name TEXT,
                symbol TEXT,
                decimals INTEGER DEFAULT 18,
                total_supply TEXT DEFAULT '0',
                circulating_supply TEXT DEFAULT '0',
                token_type TEXT DEFAULT 'ERC20',
                is_native BOOLEAN DEFAULT FALSE,
                first_seen_block INTEGER,
                last_seen_block INTEGER,
                holder_count INTEGER DEFAULT 0,
                transfer_count INTEGER DEFAULT 0,
                volume_24h TEXT DEFAULT '0',
                volume_7d TEXT DEFAULT '0',
                volume_30d TEXT DEFAULT '0',
                price_usd REAL DEFAULT 0,
                market_cap_usd REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time DeFi protocols table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_defi_protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_name TEXT,
                protocol_address TEXT,
                protocol_type TEXT,
                tvl_usd REAL DEFAULT 0,
                tvl_tokens TEXT,
                total_volume_24h REAL DEFAULT 0,
                total_volume_7d REAL DEFAULT 0,
                total_volume_30d REAL DEFAULT 0,
                active_users_24h INTEGER DEFAULT 0,
                active_users_7d INTEGER DEFAULT 0,
                transaction_count_24h INTEGER DEFAULT 0,
                transaction_count_7d INTEGER DEFAULT 0,
                fees_24h REAL DEFAULT 0,
                fees_7d REAL DEFAULT 0,
                revenue_24h REAL DEFAULT 0,
                revenue_7d REAL DEFAULT 0,
                block_number INTEGER,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Real-time bridge transactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS linea_bridge_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_hash TEXT,
                bridge_type TEXT,
                from_chain TEXT,
                to_chain TEXT,
                from_address TEXT,
                to_address TEXT,
                token_address TEXT,
                token_symbol TEXT,
                amount TEXT,
                amount_usd REAL DEFAULT 0,
                fee_amount TEXT,
                fee_usd REAL DEFAULT 0,
                block_number INTEGER,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        self.create_indexes()
        
        self.conn.commit()
        logger.info("✅ All LINEA database tables created successfully")
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        logger.info("Creating database indexes...")
        
        indexes = [
            # Archive table indexes
            "CREATE INDEX IF NOT EXISTS idx_archive_blocks_timestamp ON linea_archive_blocks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_archive_blocks_hash ON linea_archive_blocks(block_hash)",
            "CREATE INDEX IF NOT EXISTS idx_archive_blocks_parent_hash ON linea_archive_blocks(parent_hash)",
            "CREATE INDEX IF NOT EXISTS idx_archive_transactions_block_number ON linea_archive_transactions(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_archive_transactions_from ON linea_archive_transactions(from_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_transactions_to ON linea_archive_transactions(to_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_transactions_value ON linea_archive_transactions(value)",
            "CREATE INDEX IF NOT EXISTS idx_archive_transactions_type ON linea_archive_transactions(type)",
            "CREATE INDEX IF NOT EXISTS idx_archive_receipts_block_number ON linea_archive_transaction_receipts(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_archive_receipts_status ON linea_archive_transaction_receipts(status)",
            "CREATE INDEX IF NOT EXISTS idx_archive_receipts_contract ON linea_archive_transaction_receipts(contract_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_network_metrics_block ON linea_archive_network_metrics(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_archive_network_metrics_timestamp ON linea_archive_network_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_archive_accounts_contract ON linea_archive_accounts(is_contract)",
            "CREATE INDEX IF NOT EXISTS idx_archive_accounts_balance ON linea_archive_accounts(balance)",
            "CREATE INDEX IF NOT EXISTS idx_archive_contracts_creator ON linea_archive_contracts(creator_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_contracts_type ON linea_archive_contracts(contract_type)",
            "CREATE INDEX IF NOT EXISTS idx_archive_contracts_verified ON linea_archive_contracts(is_verified)",
            "CREATE INDEX IF NOT EXISTS idx_archive_tokens_type ON linea_archive_tokens(token_type)",
            "CREATE INDEX IF NOT EXISTS idx_archive_tokens_native ON linea_archive_tokens(is_native)",
            "CREATE INDEX IF NOT EXISTS idx_archive_defi_protocols_type ON linea_archive_defi_protocols(protocol_type)",
            "CREATE INDEX IF NOT EXISTS idx_archive_defi_protocols_address ON linea_archive_defi_protocols(protocol_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_bridge_from_chain ON linea_archive_bridge_transactions(from_chain)",
            "CREATE INDEX IF NOT EXISTS idx_archive_bridge_to_chain ON linea_archive_bridge_transactions(to_chain)",
            "CREATE INDEX IF NOT EXISTS idx_archive_bridge_token ON linea_archive_bridge_transactions(token_address)",
            "CREATE INDEX IF NOT EXISTS idx_archive_progress_type ON linea_archive_progress(collection_type)",
            "CREATE INDEX IF NOT EXISTS idx_archive_progress_status ON linea_archive_progress(status)",
            
            # Real-time table indexes
            "CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON linea_blocks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_blocks_hash ON linea_blocks(block_hash)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_block_number ON linea_transactions(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_from ON linea_transactions(from_address)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_to ON linea_transactions(to_address)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_block_number ON linea_transaction_receipts(block_number)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_status ON linea_transaction_receipts(status)",
            "CREATE INDEX IF NOT EXISTS idx_network_metrics_timestamp ON linea_network_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_accounts_contract ON linea_accounts(is_contract)",
            "CREATE INDEX IF NOT EXISTS idx_contracts_type ON linea_contracts(contract_type)",
            "CREATE INDEX IF NOT EXISTS idx_tokens_type ON linea_tokens(token_type)",
            "CREATE INDEX IF NOT EXISTS idx_defi_protocols_type ON linea_defi_protocols(protocol_type)",
            "CREATE INDEX IF NOT EXISTS idx_bridge_from_chain ON linea_bridge_transactions(from_chain)",
            "CREATE INDEX IF NOT EXISTS idx_bridge_to_chain ON linea_bridge_transactions(to_chain)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
        
        logger.info("✅ All database indexes created successfully")
    
    def initialize_sample_data(self):
        """Initialize sample data for testing"""
        logger.info("Initializing sample data...")
        
        # Insert LINEA native token
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO linea_tokens 
            (address, name, symbol, decimals, token_type, is_native)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("0x0000000000000000000000000000000000000000", "Ethereum", "ETH", 18, "native", True))
        
        # Insert known LINEA bridge contracts
        bridge_contracts = [
            ("0xA0b86a33E6441E0a4bFc0B4d5F3F3E5A4F3F3F3F", "LINEA Bridge", "bridge"),
            ("0xd19bae9c65bde34f26c2ee8f2f3f3e5a4f3f3f3f", "LINEA Message Service", "bridge")
        ]
        
        for address, name, contract_type in bridge_contracts:
            cursor.execute("""
                INSERT OR REPLACE INTO linea_contracts 
                (address, contract_name, contract_type, is_verified)
                VALUES (?, ?, ?, ?)
            """, (address, name, contract_type, True))
        
        self.conn.commit()
        logger.info("✅ Sample data initialized successfully")
    
    def get_table_info(self):
        """Get information about all tables"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        logger.info("📊 Database Tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            logger.info(f"  - {table_name}: {count} rows")
    
    def setup_database(self):
        """Complete database setup"""
        logger.info("🚀 Setting up LINEA database...")
        
        try:
            self.connect()
            self.create_tables()
            self.initialize_sample_data()
            self.get_table_info()
            logger.info("✅ LINEA database setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
        finally:
            self.close()

def main():
    """Main function"""
    # Create database for archive data
    archive_schema = LineaDatabaseSchema("linea_archive_data.db")
    archive_schema.setup_database()
    
    # Create database for real-time data
    realtime_schema = LineaDatabaseSchema("linea_data.db")
    realtime_schema.setup_database()
    
    logger.info("🎉 All LINEA databases created successfully!")

if __name__ == "__main__":
    main()
