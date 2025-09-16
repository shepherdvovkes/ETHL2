#!/usr/bin/env python3
"""
Миграция базы данных DEFIMON для поддержки всех метрик и 50+ блокчейнов
"""

import os
import sys
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv("config.env")

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://defimon:password@localhost:5432/defimon_db"
)

def migrate_from_sqlite_to_postgres():
    """Миграция данных из SQLite в PostgreSQL"""
    
    print("🔄 Starting migration from SQLite to PostgreSQL...")
    
    # SQLite database path
    sqlite_path = Path("data/defimon.db")
    if not sqlite_path.exists():
        print("❌ SQLite database not found. Please run setup_database.py first.")
        return False
    
    # Create PostgreSQL engine
    postgres_engine = create_engine(DATABASE_URL, echo=False)
    postgres_session = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)()
    
    # SQLite connection
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        # Create new tables in PostgreSQL
        print("📋 Creating new tables in PostgreSQL...")
        from src.database.models_v2 import Base
        Base.metadata.create_all(bind=postgres_engine)
        
        # Migrate crypto_assets
        print("🪙 Migrating crypto assets...")
        sqlite_cursor.execute("SELECT * FROM crypto_assets")
        assets = sqlite_cursor.fetchall()
        
        for asset in assets:
            # Insert into blockchains table first
            blockchain_query = text("""
                INSERT INTO blockchains (name, symbol, chain_id, blockchain_type, native_token, is_active)
                VALUES (:name, :symbol, :chain_id, :type, :native_token, true)
                ON CONFLICT (name) DO NOTHING
            """)
            
            blockchain_name = asset[5] if len(asset) > 5 else "polygon"  # blockchain field
            postgres_session.execute(blockchain_query, {
                "name": blockchain_name.title(),
                "symbol": blockchain_name.upper(),
                "chain_id": 137 if blockchain_name == "polygon" else 1,
                "type": "mainnet",
                "native_token": blockchain_name.upper()
            })
        
        postgres_session.commit()
        
        # Get blockchain IDs
        blockchain_map = {}
        blockchain_result = postgres_session.execute(text("SELECT id, name FROM blockchains"))
        for row in blockchain_result:
            blockchain_map[row[1].lower()] = row[0]
        
        # Insert crypto assets
        for asset in assets:
            asset_query = text("""
                INSERT INTO crypto_assets (symbol, name, contract_address, blockchain_id, category, github_repo, website, description, is_active)
                VALUES (:symbol, :name, :contract_address, :blockchain_id, :category, :github_repo, :website, :description, true)
                ON CONFLICT (symbol, blockchain_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    contract_address = EXCLUDED.contract_address,
                    category = EXCLUDED.category,
                    github_repo = EXCLUDED.github_repo,
                    website = EXCLUDED.website,
                    description = EXCLUDED.description
            """)
            
            blockchain_name = asset[5] if len(asset) > 5 else "polygon"
            blockchain_id = blockchain_map.get(blockchain_name.lower(), 1)
            
            postgres_session.execute(asset_query, {
                "symbol": asset[1],
                "name": asset[2],
                "contract_address": asset[3],
                "blockchain_id": blockchain_id,
                "category": asset[6] if len(asset) > 6 else "DeFi",
                "github_repo": asset[7] if len(asset) > 7 else None,
                "website": asset[8] if len(asset) > 8 else None,
                "description": asset[9] if len(asset) > 9 else None
            })
        
        postgres_session.commit()
        
        # Get asset IDs mapping
        asset_map = {}
        asset_result = postgres_session.execute(text("SELECT id, symbol, blockchain_id FROM crypto_assets"))
        for row in asset_result:
            key = f"{row[1]}_{row[2]}"
            asset_map[key] = row[0]
        
        # Migrate onchain_metrics
        print("⛓️  Migrating on-chain metrics...")
        sqlite_cursor.execute("SELECT * FROM onchain_metrics")
        onchain_metrics = sqlite_cursor.fetchall()
        
        for metric in onchain_metrics:
            # Find corresponding asset in new structure
            sqlite_cursor.execute("SELECT symbol, blockchain FROM crypto_assets WHERE id = ?", (metric[1],))
            asset_info = sqlite_cursor.fetchone()
            if asset_info:
                blockchain_name = asset_info[1] if asset_info[1] else "polygon"
                blockchain_id = blockchain_map.get(blockchain_name.lower(), 1)
                key = f"{asset_info[0]}_{blockchain_id}"
                new_asset_id = asset_map.get(key)
                
                if new_asset_id:
                    onchain_query = text("""
                        INSERT INTO onchain_metrics (
                            asset_id, timestamp, tvl, tvl_change_24h, tvl_change_7d,
                            daily_transactions, transaction_volume_24h, avg_transaction_fee,
                            active_addresses_24h, new_addresses_24h, unique_users_7d,
                            new_contracts_deployed, contract_interactions_24h,
                            block_time_avg, gas_price_avg, network_utilization
                        ) VALUES (
                            :asset_id, :timestamp, :tvl, :tvl_change_24h, :tvl_change_7d,
                            :daily_transactions, :transaction_volume_24h, :avg_transaction_fee,
                            :active_addresses_24h, :new_addresses_24h, :unique_users_7d,
                            :new_contracts_deployed, :contract_interactions_24h,
                            :block_time_avg, :gas_price_avg, :network_utilization
                        )
                    """)
                    
                    postgres_session.execute(onchain_query, {
                        "asset_id": new_asset_id,
                        "timestamp": metric[2],
                        "tvl": metric[3],
                        "tvl_change_24h": metric[4],
                        "tvl_change_7d": metric[5],
                        "daily_transactions": metric[6],
                        "transaction_volume_24h": metric[7],
                        "avg_transaction_fee": metric[8],
                        "active_addresses_24h": metric[9],
                        "new_addresses_24h": metric[10],
                        "unique_users_7d": metric[11],
                        "new_contracts_deployed": metric[12],
                        "contract_interactions_24h": metric[13],
                        "block_time_avg": metric[14],
                        "gas_price_avg": metric[15],
                        "network_utilization": metric[16]
                    })
        
        postgres_session.commit()
        
        # Migrate financial_metrics
        print("💰 Migrating financial metrics...")
        sqlite_cursor.execute("SELECT * FROM financial_metrics")
        financial_metrics = sqlite_cursor.fetchall()
        
        for metric in financial_metrics:
            # Find corresponding asset
            sqlite_cursor.execute("SELECT symbol, blockchain FROM crypto_assets WHERE id = ?", (metric[1],))
            asset_info = sqlite_cursor.fetchone()
            if asset_info:
                blockchain_name = asset_info[1] if asset_info[1] else "polygon"
                blockchain_id = blockchain_map.get(blockchain_name.lower(), 1)
                key = f"{asset_info[0]}_{blockchain_id}"
                new_asset_id = asset_map.get(key)
                
                if new_asset_id:
                    financial_query = text("""
                        INSERT INTO financial_metrics (
                            asset_id, timestamp, price_usd, market_cap, fully_diluted_valuation,
                            volume_24h, volume_7d, volume_change_24h, volatility_24h, volatility_7d, volatility_30d,
                            bid_ask_spread, order_book_depth, circulating_supply, total_supply, max_supply,
                            price_change_1h, price_change_24h, price_change_7d, price_change_30d
                        ) VALUES (
                            :asset_id, :timestamp, :price_usd, :market_cap, :fully_diluted_valuation,
                            :volume_24h, :volume_7d, :volume_change_24h, :volatility_24h, :volatility_7d, :volatility_30d,
                            :bid_ask_spread, :order_book_depth, :circulating_supply, :total_supply, :max_supply,
                            :price_change_1h, :price_change_24h, :price_change_7d, :price_change_30d
                        )
                    """)
                    
                    postgres_session.execute(financial_query, {
                        "asset_id": new_asset_id,
                        "timestamp": metric[2],
                        "price_usd": metric[3],
                        "market_cap": metric[4],
                        "fully_diluted_valuation": metric[5],
                        "volume_24h": metric[6],
                        "volume_7d": metric[7],
                        "volume_change_24h": metric[8],
                        "volatility_24h": metric[9],
                        "volatility_7d": metric[10],
                        "volatility_30d": metric[11],
                        "bid_ask_spread": metric[12],
                        "order_book_depth": metric[13],
                        "circulating_supply": metric[14],
                        "total_supply": metric[15],
                        "max_supply": metric[16],
                        "price_change_1h": metric[17],
                        "price_change_24h": metric[18],
                        "price_change_7d": metric[19],
                        "price_change_30d": metric[20]
                    })
        
        postgres_session.commit()
        
        # Migrate github_metrics
        print("👨‍💻 Migrating GitHub metrics...")
        sqlite_cursor.execute("SELECT * FROM github_metrics")
        github_metrics = sqlite_cursor.fetchall()
        
        for metric in github_metrics:
            # Find corresponding asset
            sqlite_cursor.execute("SELECT symbol, blockchain FROM crypto_assets WHERE id = ?", (metric[1],))
            asset_info = sqlite_cursor.fetchone()
            if asset_info:
                blockchain_name = asset_info[1] if asset_info[1] else "polygon"
                blockchain_id = blockchain_map.get(blockchain_name.lower(), 1)
                key = f"{asset_info[0]}_{blockchain_id}"
                new_asset_id = asset_map.get(key)
                
                if new_asset_id:
                    github_query = text("""
                        INSERT INTO github_metrics (
                            asset_id, timestamp, commits_24h, commits_7d, commits_30d,
                            open_prs, merged_prs_7d, closed_prs_7d, open_issues, closed_issues_7d,
                            active_contributors_30d, total_contributors, stars, forks, stars_change_7d,
                            primary_language, languages_distribution
                        ) VALUES (
                            :asset_id, :timestamp, :commits_24h, :commits_7d, :commits_30d,
                            :open_prs, :merged_prs_7d, :closed_prs_7d, :open_issues, :closed_issues_7d,
                            :active_contributors_30d, :total_contributors, :stars, :forks, :stars_change_7d,
                            :primary_language, :languages_distribution
                        )
                    """)
                    
                    postgres_session.execute(github_query, {
                        "asset_id": new_asset_id,
                        "timestamp": metric[2],
                        "commits_24h": metric[3],
                        "commits_7d": metric[4],
                        "commits_30d": metric[5],
                        "open_prs": metric[6],
                        "merged_prs_7d": metric[7],
                        "closed_prs_7d": metric[8],
                        "open_issues": metric[9],
                        "closed_issues_7d": metric[10],
                        "active_contributors_30d": metric[11],
                        "total_contributors": metric[12],
                        "stars": metric[13],
                        "forks": metric[14],
                        "stars_change_7d": metric[15],
                        "primary_language": metric[16],
                        "languages_distribution": metric[17]
                    })
        
        postgres_session.commit()
        
        # Migrate ML predictions
        print("🤖 Migrating ML predictions...")
        sqlite_cursor.execute("SELECT * FROM ml_predictions")
        ml_predictions = sqlite_cursor.fetchall()
        
        for prediction in ml_predictions:
            # Find corresponding asset
            sqlite_cursor.execute("SELECT symbol, blockchain FROM crypto_assets WHERE id = ?", (prediction[1],))
            asset_info = sqlite_cursor.fetchone()
            if asset_info:
                blockchain_name = asset_info[1] if asset_info[1] else "polygon"
                blockchain_id = blockchain_map.get(blockchain_name.lower(), 1)
                key = f"{asset_info[0]}_{blockchain_id}"
                new_asset_id = asset_map.get(key)
                
                if new_asset_id:
                    ml_query = text("""
                        INSERT INTO ml_predictions (
                            asset_id, model_name, prediction_type, prediction_value,
                            confidence_score, prediction_horizon, features_used, model_version, created_at
                        ) VALUES (
                            :asset_id, :model_name, :prediction_type, :prediction_value,
                            :confidence_score, :prediction_horizon, :features_used, :model_version, :created_at
                        )
                    """)
                    
                    postgres_session.execute(ml_query, {
                        "asset_id": new_asset_id,
                        "model_name": prediction[2],
                        "prediction_type": prediction[3],
                        "prediction_value": prediction[4],
                        "confidence_score": prediction[5],
                        "prediction_horizon": prediction[6],
                        "features_used": prediction[7],
                        "model_version": prediction[8],
                        "created_at": prediction[9]
                    })
        
        postgres_session.commit()
        
        print("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        postgres_session.rollback()
        print(f"❌ Migration failed: {e}")
        return False
    finally:
        sqlite_conn.close()
        postgres_session.close()

def create_indexes():
    """Создание индексов для оптимизации производительности"""
    
    print("📊 Creating database indexes...")
    
    engine = create_engine(DATABASE_URL, echo=False)
    
    indexes = [
        # Blockchain indexes
        "CREATE INDEX IF NOT EXISTS idx_blockchains_name ON blockchains(name)",
        "CREATE INDEX IF NOT EXISTS idx_blockchains_symbol ON blockchains(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_blockchains_chain_id ON blockchains(chain_id)",
        
        # Asset indexes
        "CREATE INDEX IF NOT EXISTS idx_crypto_assets_symbol ON crypto_assets(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_crypto_assets_blockchain_id ON crypto_assets(blockchain_id)",
        "CREATE INDEX IF NOT EXISTS idx_crypto_assets_category ON crypto_assets(category)",
        "CREATE INDEX IF NOT EXISTS idx_crypto_assets_contract_address ON crypto_assets(contract_address)",
        
        # Metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_onchain_metrics_asset_timestamp ON onchain_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_financial_metrics_asset_timestamp ON financial_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_github_metrics_asset_timestamp ON github_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_tokenomics_metrics_asset_timestamp ON tokenomics_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_security_metrics_asset_timestamp ON security_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_community_metrics_asset_timestamp ON community_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_partnership_metrics_asset_timestamp ON partnership_metrics(asset_id, timestamp)",
        
        # Network metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_network_metrics_blockchain_timestamp ON network_metrics(blockchain_id, timestamp)",
        
        # ML predictions indexes
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_asset_model ON ml_predictions(asset_id, model_name)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_type_horizon ON ml_predictions(prediction_type, prediction_horizon)",
        
        # Smart contracts indexes
        "CREATE INDEX IF NOT EXISTS idx_smart_contracts_address ON smart_contracts(contract_address)",
        "CREATE INDEX IF NOT EXISTS idx_smart_contracts_blockchain ON smart_contracts(blockchain_id)",
        "CREATE INDEX IF NOT EXISTS idx_smart_contracts_asset ON smart_contracts(asset_id)",
        
        # Trending metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_trending_metrics_asset_timestamp ON trending_metrics(asset_id, timestamp)",
        
        # Cross-chain metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_cross_chain_metrics_asset_timestamp ON cross_chain_metrics(asset_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_cross_chain_metrics_source_target ON cross_chain_metrics(source_blockchain_id, target_blockchain_id)",
    ]
    
    with engine.connect() as conn:
        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
                print(f"✅ Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"⚠️  Index creation warning: {e}")
        
        conn.commit()
    
    print("✅ All indexes created successfully!")

def main():
    """Main migration function"""
    print("🚀 DEFIMON - Database Migration")
    print("=" * 50)
    
    # Check if we're migrating from SQLite
    if os.path.exists("data/defimon.db"):
        print("📦 Found existing SQLite database. Starting migration...")
        success = migrate_from_sqlite_to_postgres()
        if not success:
            print("❌ Migration failed. Please check the logs.")
            return
    else:
        print("📦 No existing SQLite database found. Creating new PostgreSQL structure...")
        from src.database.models_v2 import Base
        engine = create_engine(DATABASE_URL, echo=False)
        Base.metadata.create_all(bind=engine)
        print("✅ New PostgreSQL structure created!")
    
    # Initialize blockchains
    print("\n🌐 Initializing blockchains...")
    from src.database.blockchain_init import init_blockchains
    init_blockchains()
    
    # Create indexes
    print("\n📊 Creating indexes...")
    create_indexes()
    
    print("\n🎉 Migration completed successfully!")
    print("   - All tables created")
    print("   - 50+ blockchains initialized")
    print("   - Indexes created for performance")
    print("   - Ready for data collection")

if __name__ == "__main__":
    main()
