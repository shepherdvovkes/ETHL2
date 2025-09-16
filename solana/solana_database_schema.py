#!/usr/bin/env python3
"""
Solana Database Schema Creation
Creates comprehensive database schema for Solana blockchain data collection
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

def create_solana_database_schema():
    """Create comprehensive Solana database schema"""
    
    # Create main database
    db_path = Path("solana_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🗄️  Creating Solana database schema...")
    
    # 1. Blockchain info table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_blockchain_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            rpc_url TEXT NOT NULL,
            wss_url TEXT NOT NULL,
            explorer_url TEXT,
            native_token TEXT NOT NULL,
            genesis_slot INTEGER,
            genesis_timestamp TIMESTAMP,
            current_epoch INTEGER,
            current_slot INTEGER,
            slots_per_epoch INTEGER,
            target_slot_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. Blocks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot INTEGER UNIQUE NOT NULL,
            blockhash TEXT NOT NULL,
            parent_slot INTEGER,
            parent_blockhash TEXT,
            timestamp TIMESTAMP NOT NULL,
            block_time REAL,
            block_height INTEGER,
            transaction_count INTEGER,
            total_fee NUMERIC(20,9),
            reward NUMERIC(20,9),
            leader TEXT,
            leader_reward NUMERIC(20,9),
            vote_accounts_count INTEGER,
            vote_accounts_stake NUMERIC(20,9),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT UNIQUE NOT NULL,
            slot INTEGER NOT NULL,
            block_time TIMESTAMP,
            fee NUMERIC(20,9) NOT NULL,
            success BOOLEAN NOT NULL,
            error TEXT,
            compute_units_consumed INTEGER,
            compute_units_price NUMERIC(20,9),
            accounts_count INTEGER,
            instructions_count INTEGER,
            log_messages TEXT,
            inner_instructions TEXT,
            pre_balances TEXT,
            post_balances TEXT,
            pre_token_balances TEXT,
            post_token_balances TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (slot) REFERENCES solana_blocks(slot)
        )
    """)
    
    # 4. Accounts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE NOT NULL,
            owner TEXT,
            lamports NUMERIC(20,9),
            data_length INTEGER,
            executable BOOLEAN DEFAULT FALSE,
            rent_epoch INTEGER,
            account_type TEXT,
            is_native BOOLEAN DEFAULT FALSE,
            is_token_account BOOLEAN DEFAULT FALSE,
            is_program BOOLEAN DEFAULT FALSE,
            is_system_account BOOLEAN DEFAULT FALSE,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 5. Token accounts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_token_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE NOT NULL,
            mint TEXT NOT NULL,
            owner TEXT NOT NULL,
            amount NUMERIC(20,9) NOT NULL,
            delegate TEXT,
            delegated_amount NUMERIC(20,9),
            is_initialized BOOLEAN DEFAULT FALSE,
            is_frozen BOOLEAN DEFAULT FALSE,
            is_native BOOLEAN DEFAULT FALSE,
            rent_exempt_reserve NUMERIC(20,9),
            close_authority TEXT,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mint) REFERENCES solana_tokens(mint)
        )
    """)
    
    # 6. Tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mint TEXT UNIQUE NOT NULL,
            name TEXT,
            symbol TEXT,
            decimals INTEGER,
            supply NUMERIC(20,9),
            freeze_authority TEXT,
            mint_authority TEXT,
            is_initialized BOOLEAN DEFAULT FALSE,
            token_program TEXT,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 7. Programs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_programs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            program_id TEXT UNIQUE NOT NULL,
            name TEXT,
            description TEXT,
            program_type TEXT,
            is_upgradeable BOOLEAN,
            program_data_address TEXT,
            program_data_length INTEGER,
            authority TEXT,
            slot INTEGER,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 8. Validators table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_validators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vote_account TEXT UNIQUE NOT NULL,
            node_pubkey TEXT,
            commission INTEGER,
            last_vote INTEGER,
            root_slot INTEGER,
            credits INTEGER,
            epoch_credits TEXT,
            activated_stake NUMERIC(20,9),
            version TEXT,
            delinquent BOOLEAN DEFAULT FALSE,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 9. Staking accounts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_staking_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stake_account TEXT UNIQUE NOT NULL,
            voter_pubkey TEXT NOT NULL,
            stake_lamports NUMERIC(20,9),
            activation_epoch INTEGER,
            deactivation_epoch INTEGER,
            warmup_cooldown_rate REAL,
            stake_type TEXT,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (voter_pubkey) REFERENCES solana_validators(vote_account)
        )
    """)
    
    # 10. Epoch info table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_epoch_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER UNIQUE NOT NULL,
            slot_index INTEGER,
            slots_in_epoch INTEGER,
            absolute_slot INTEGER,
            block_height INTEGER,
            transaction_count INTEGER,
            total_fees NUMERIC(20,9),
            total_rewards NUMERIC(20,9),
            inflation_rate REAL,
            total_stake NUMERIC(20,9),
            active_stake NUMERIC(20,9),
            delinquent_stake NUMERIC(20,9),
            validator_count INTEGER,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 11. Network metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_network_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            slot INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            
            -- Performance metrics
            tps REAL,
            block_time_avg REAL,
            block_time_std REAL,
            slot_time_avg REAL,
            slot_time_std REAL,
            
            -- Network health
            skipped_slots INTEGER,
            skipped_slots_percentage REAL,
            vote_accounts_count INTEGER,
            total_stake NUMERIC(20,9),
            active_stake NUMERIC(20,9),
            delinquent_stake NUMERIC(20,9),
            
            -- Economic metrics
            total_supply NUMERIC(20,9),
            circulating_supply NUMERIC(20,9),
            inflation_rate REAL,
            staking_ratio REAL,
            
            -- Fee metrics
            avg_fee_per_transaction NUMERIC(20,9),
            total_fees_24h NUMERIC(20,9),
            priority_fee_avg NUMERIC(20,9),
            
            -- Validator metrics
            validator_count INTEGER,
            active_validator_count INTEGER,
            delinquent_validator_count INTEGER,
            avg_commission REAL,
            
            -- DeFi metrics
            defi_tvl NUMERIC(20,9),
            defi_protocols_count INTEGER,
            dex_volume_24h NUMERIC(20,9),
            
            -- NFT metrics
            nft_transactions_24h INTEGER,
            nft_volume_24h NUMERIC(20,9),
            nft_marketplaces_count INTEGER
        )
    """)
    
    # 12. Price data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            price_usd NUMERIC(20,8),
            market_cap NUMERIC(30,8),
            volume_24h NUMERIC(30,8),
            price_change_1h REAL,
            price_change_24h REAL,
            price_change_7d REAL,
            price_change_30d REAL,
            market_cap_rank INTEGER,
            circulating_supply NUMERIC(30,8),
            total_supply NUMERIC(30,8),
            max_supply NUMERIC(30,8),
            ath NUMERIC(20,8),
            atl NUMERIC(20,8),
            ath_date TIMESTAMP,
            atl_date TIMESTAMP
        )
    """)
    
    # 13. DeFi protocols table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_defi_protocols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_name TEXT UNIQUE NOT NULL,
            protocol_type TEXT,
            program_id TEXT,
            tvl NUMERIC(20,9),
            volume_24h NUMERIC(20,9),
            fees_24h NUMERIC(20,9),
            users_24h INTEGER,
            transactions_24h INTEGER,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 14. NFT collections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_nft_collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_name TEXT,
            collection_symbol TEXT,
            mint TEXT UNIQUE NOT NULL,
            creator TEXT,
            royalty_percentage REAL,
            total_supply INTEGER,
            floor_price NUMERIC(20,9),
            volume_24h NUMERIC(20,9),
            volume_7d NUMERIC(20,9),
            volume_30d NUMERIC(20,9),
            sales_24h INTEGER,
            sales_7d INTEGER,
            sales_30d INTEGER,
            holders_count INTEGER,
            first_seen_slot INTEGER,
            last_updated_slot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 15. Archive collection status table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_archive_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_type TEXT NOT NULL,
            start_slot INTEGER,
            end_slot INTEGER,
            current_slot INTEGER,
            total_records INTEGER,
            status TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for performance
    print("📊 Creating indexes...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_blocks_slot ON solana_blocks(slot)",
        "CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON solana_blocks(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_blocks_leader ON solana_blocks(leader)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_signature ON solana_transactions(signature)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_slot ON solana_transactions(slot)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_block_time ON solana_transactions(block_time)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_success ON solana_transactions(success)",
        "CREATE INDEX IF NOT EXISTS idx_accounts_address ON solana_accounts(address)",
        "CREATE INDEX IF NOT EXISTS idx_accounts_owner ON solana_accounts(owner)",
        "CREATE INDEX IF NOT EXISTS idx_accounts_type ON solana_accounts(account_type)",
        "CREATE INDEX IF NOT EXISTS idx_token_accounts_address ON solana_token_accounts(address)",
        "CREATE INDEX IF NOT EXISTS idx_token_accounts_mint ON solana_token_accounts(mint)",
        "CREATE INDEX IF NOT EXISTS idx_token_accounts_owner ON solana_token_accounts(owner)",
        "CREATE INDEX IF NOT EXISTS idx_tokens_mint ON solana_tokens(mint)",
        "CREATE INDEX IF NOT EXISTS idx_tokens_symbol ON solana_tokens(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_programs_program_id ON solana_programs(program_id)",
        "CREATE INDEX IF NOT EXISTS idx_validators_vote_account ON solana_validators(vote_account)",
        "CREATE INDEX IF NOT EXISTS idx_validators_node_pubkey ON solana_validators(node_pubkey)",
        "CREATE INDEX IF NOT EXISTS idx_staking_accounts_stake_account ON solana_staking_accounts(stake_account)",
        "CREATE INDEX IF NOT EXISTS idx_staking_accounts_voter ON solana_staking_accounts(voter_pubkey)",
        "CREATE INDEX IF NOT EXISTS idx_epoch_info_epoch ON solana_epoch_info(epoch)",
        "CREATE INDEX IF NOT EXISTS idx_network_metrics_timestamp ON solana_network_metrics(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_network_metrics_slot ON solana_network_metrics(slot)",
        "CREATE INDEX IF NOT EXISTS idx_network_metrics_epoch ON solana_network_metrics(epoch)",
        "CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON solana_price_data(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON solana_price_data(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_defi_protocols_name ON solana_defi_protocols(protocol_name)",
        "CREATE INDEX IF NOT EXISTS idx_nft_collections_mint ON solana_nft_collections(mint)",
        "CREATE INDEX IF NOT EXISTS idx_archive_status_type ON solana_archive_status(collection_type)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    # Insert initial blockchain info
    cursor.execute("""
        INSERT OR REPLACE INTO solana_blockchain_info 
        (chain_id, name, symbol, rpc_url, wss_url, explorer_url, native_token, 
         genesis_slot, genesis_timestamp, slots_per_epoch, target_slot_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "solana-mainnet",
        "Solana",
        "SOL",
        "https://delicate-fluent-slug.solana-mainnet.quiknode.pro/e9182c45c76c38f1bb92dfa46a30eeef12be0025/",
        "wss://delicate-fluent-slug.solana-mainnet.quiknode.pro/e9182c45c76c38f1bb92dfa46a30eeef12be0025/",
        "https://explorer.solana.com",
        "SOL",
        0,
        "2020-03-16 00:00:00",
        432000,
        400
    ))
    
    conn.commit()
    conn.close()
    
    print("✅ Solana database schema created successfully!")
    print(f"📁 Database file: {db_path.absolute()}")
    
    return True

def create_archive_database_schema():
    """Create archive database schema for historical data"""
    
    db_path = Path("solana_archive_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🗄️  Creating Solana archive database schema...")
    
    # Archive blocks table (partitioned by date)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_archive_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot INTEGER UNIQUE NOT NULL,
            blockhash TEXT NOT NULL,
            parent_slot INTEGER,
            parent_blockhash TEXT,
            timestamp TIMESTAMP NOT NULL,
            block_time REAL,
            block_height INTEGER,
            transaction_count INTEGER,
            total_fee NUMERIC(20,9),
            reward NUMERIC(20,9),
            leader TEXT,
            leader_reward NUMERIC(20,9),
            vote_accounts_count INTEGER,
            vote_accounts_stake NUMERIC(20,9),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Archive transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_archive_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT UNIQUE NOT NULL,
            slot INTEGER NOT NULL,
            block_time TIMESTAMP,
            fee NUMERIC(20,9) NOT NULL,
            success BOOLEAN NOT NULL,
            error TEXT,
            compute_units_consumed INTEGER,
            compute_units_price NUMERIC(20,9),
            accounts_count INTEGER,
            instructions_count INTEGER,
            log_messages TEXT,
            inner_instructions TEXT,
            pre_balances TEXT,
            post_balances TEXT,
            pre_token_balances TEXT,
            post_token_balances TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Archive network metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_archive_network_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            slot INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            tps REAL,
            block_time_avg REAL,
            block_time_std REAL,
            slot_time_avg REAL,
            slot_time_std REAL,
            skipped_slots INTEGER,
            skipped_slots_percentage REAL,
            vote_accounts_count INTEGER,
            total_stake NUMERIC(20,9),
            active_stake NUMERIC(20,9),
            delinquent_stake NUMERIC(20,9),
            total_supply NUMERIC(20,9),
            circulating_supply NUMERIC(20,9),
            inflation_rate REAL,
            staking_ratio REAL,
            avg_fee_per_transaction NUMERIC(20,9),
            total_fees_24h NUMERIC(20,9),
            priority_fee_avg NUMERIC(20,9),
            validator_count INTEGER,
            active_validator_count INTEGER,
            delinquent_validator_count INTEGER,
            avg_commission REAL,
            defi_tvl NUMERIC(20,9),
            defi_protocols_count INTEGER,
            dex_volume_24h NUMERIC(20,9),
            nft_transactions_24h INTEGER,
            nft_volume_24h NUMERIC(20,9),
            nft_marketplaces_count INTEGER
        )
    """)
    
    # Archive collection progress table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solana_archive_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_type TEXT NOT NULL,
            start_slot INTEGER,
            end_slot INTEGER,
            current_slot INTEGER,
            total_slots INTEGER,
            completed_slots INTEGER,
            failed_slots INTEGER,
            status TEXT,
            progress_percentage REAL,
            started_at TIMESTAMP,
            last_updated_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create archive indexes
    archive_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_archive_blocks_slot ON solana_archive_blocks(slot)",
        "CREATE INDEX IF NOT EXISTS idx_archive_blocks_timestamp ON solana_archive_blocks(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_archive_blocks_leader ON solana_archive_blocks(leader)",
        "CREATE INDEX IF NOT EXISTS idx_archive_transactions_signature ON solana_archive_transactions(signature)",
        "CREATE INDEX IF NOT EXISTS idx_archive_transactions_slot ON solana_archive_transactions(slot)",
        "CREATE INDEX IF NOT EXISTS idx_archive_transactions_block_time ON solana_archive_transactions(block_time)",
        "CREATE INDEX IF NOT EXISTS idx_archive_network_metrics_timestamp ON solana_archive_network_metrics(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_archive_network_metrics_slot ON solana_archive_network_metrics(slot)",
        "CREATE INDEX IF NOT EXISTS idx_archive_network_metrics_epoch ON solana_archive_network_metrics(epoch)",
        "CREATE INDEX IF NOT EXISTS idx_archive_progress_type ON solana_archive_progress(collection_type)"
    ]
    
    for index_sql in archive_indexes:
        cursor.execute(index_sql)
    
    conn.commit()
    conn.close()
    
    print("✅ Solana archive database schema created successfully!")
    print(f"📁 Archive database file: {db_path.absolute()}")
    
    return True

if __name__ == "__main__":
    print("🚀 Setting up Solana database schemas...")
    
    try:
        # Create main database
        create_solana_database_schema()
        
        # Create archive database
        create_archive_database_schema()
        
        print("🎉 Solana database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error setting up Solana databases: {e}")
        exit(1)
