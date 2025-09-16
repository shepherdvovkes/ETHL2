#!/usr/bin/env python3
"""
Database Migration: Add security_score column to security_metrics table

This script:
1. Adds the security_score column to the security_metrics table
2. Calculates and populates security scores for existing records
3. Updates the database schema
"""

import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.security_calculator import SecurityScoreCalculator
from src.config.settings import settings


async def migrate_security_score():
    """Run the security score migration"""
    
    print("🔧 Starting Security Score Migration...")
    
    conn = None
    try:
        # Connect to database using DATABASE_URL
        conn = psycopg2.connect(settings.DATABASE_URL)
        conn.autocommit = False
        cursor = conn.cursor()
        
        print("✅ Connected to database")
        
        # Step 1: Check if security_score column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'security_metrics' 
            AND column_name = 'security_score'
        """)
        
        if cursor.fetchone():
            print("⚠️  security_score column already exists, skipping column creation")
        else:
            # Step 2: Add security_score column
            print("📝 Adding security_score column to security_metrics table...")
            cursor.execute("""
                ALTER TABLE security_metrics 
                ADD COLUMN security_score FLOAT
            """)
            print("✅ Added security_score column")
        
        # Step 3: Get all existing security metrics records
        print("📊 Fetching existing security metrics records...")
        cursor.execute("""
            SELECT id, asset_id, timestamp, audit_status, audit_firm, audit_score,
                   contract_verified, source_code_available, vulnerability_score,
                   multisig_wallets, timelock_mechanisms, upgrade_mechanisms,
                   emergency_pause, governance_decentralization, validator_distribution,
                   node_distribution, treasury_control, reentrancy_protection,
                   overflow_protection, access_control, pause_functionality
            FROM security_metrics
            WHERE security_score IS NULL
            ORDER BY id
        """)
        
        records = cursor.fetchall()
        print(f"📈 Found {len(records)} records to update")
        
        if records:
            # Step 4: Calculate and update security scores
            print("🧮 Calculating security scores for existing records...")
            
            updated_count = 0
            for record in records:
                record_id = record[0]
                
                # Convert record to dictionary for security calculator
                security_metrics = {
                    'audit_status': record[3],
                    'audit_firm': record[4],
                    'audit_score': record[5],
                    'contract_verified': record[6],
                    'source_code_available': record[7],
                    'vulnerability_score': record[8],
                    'multisig_wallets': record[9],
                    'timelock_mechanisms': record[10],
                    'upgrade_mechanisms': record[11],
                    'emergency_pause': record[12],
                    'governance_decentralization': record[13],
                    'validator_distribution': record[14],
                    'node_distribution': record[15],
                    'treasury_control': record[16],
                    'reentrancy_protection': record[17],
                    'overflow_protection': record[18],
                    'access_control': record[19],
                    'pause_functionality': record[20]
                }
                
                # Calculate security score
                security_score = SecurityScoreCalculator.calculate_asset_security_score(security_metrics)
                
                # Update the record
                cursor.execute("""
                    UPDATE security_metrics 
                    SET security_score = %s 
                    WHERE id = %s
                """, (security_score, record_id))
                
                updated_count += 1
                
                if updated_count % 100 == 0:
                    print(f"📊 Updated {updated_count}/{len(records)} records...")
                    conn.commit()  # Commit in batches
            
            # Final commit
            conn.commit()
            print(f"✅ Successfully updated {updated_count} security score records")
        
        # Step 5: Update L2 security metrics as well (if table exists)
        print("🔄 Checking for L2 security metrics table...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'l2_security_metrics'
            )
        """)
        
        l2_table_exists = cursor.fetchone()[0]
        
        if l2_table_exists:
            print("🔄 Updating L2 security metrics...")
            cursor.execute("""
                SELECT id, l2_network_id, timestamp, audit_count, audit_firms, bug_bounty_program,
                       validator_count, slashing_mechanism, multisig_required, upgrade_mechanism,
                       time_to_finality, contract_verified, source_code_available,
                       reentrancy_protection, overflow_protection, access_control, emergency_pause,
                       sequencer_decentralization, governance_decentralization, treasury_control
                FROM l2_security_metrics
                WHERE security_score IS NULL
                ORDER BY id
            """)
        else:
            print("⚠️  L2 security metrics table does not exist, skipping L2 updates")
            cursor.execute("SELECT 1 WHERE FALSE")  # Empty result set
        
        l2_records = cursor.fetchall()
        print(f"📈 Found {len(l2_records)} L2 records to update")
        
        if l2_records:
            l2_updated_count = 0
            for record in l2_records:
                record_id = record[0]
                
                # Convert record to dictionary for L2 security calculator
                l2_security_metrics = {
                    'audit_count': record[3],
                    'audit_firms': record[4],
                    'bug_bounty_program': record[5],
                    'validator_count': record[6],
                    'slashing_mechanism': record[7],
                    'multisig_required': record[8],
                    'upgrade_mechanism': record[9],
                    'time_to_finality': record[10],
                    'contract_verified': record[11],
                    'source_code_available': record[12],
                    'reentrancy_protection': record[13],
                    'overflow_protection': record[14],
                    'access_control': record[15],
                    'emergency_pause': record[16],
                    'sequencer_decentralization': record[17],
                    'governance_decentralization': record[18],
                    'treasury_control': record[19]
                }
                
                # Calculate L2 security score
                l2_security_score = SecurityScoreCalculator.calculate_l2_security_score(l2_security_metrics)
                
                # Update the record
                cursor.execute("""
                    UPDATE l2_security_metrics 
                    SET security_score = %s 
                    WHERE id = %s
                """, (l2_security_score, record_id))
                
                l2_updated_count += 1
                
                if l2_updated_count % 50 == 0:
                    print(f"📊 Updated {l2_updated_count}/{len(l2_records)} L2 records...")
                    conn.commit()  # Commit in batches
            
            # Final commit
            conn.commit()
            print(f"✅ Successfully updated {l2_updated_count} L2 security score records")
        
        # Step 6: Verify the migration
        print("🔍 Verifying migration results...")
        
        # Check security_metrics
        cursor.execute("""
            SELECT COUNT(*) as total, 
                   COUNT(security_score) as with_score,
                   AVG(security_score) as avg_score,
                   MIN(security_score) as min_score,
                   MAX(security_score) as max_score
            FROM security_metrics
        """)
        
        result = cursor.fetchone()
        print(f"📊 Security Metrics Summary:")
        print(f"   Total records: {result[0]}")
        print(f"   Records with score: {result[1]}")
        avg_score = f"{result[2]:.2f}" if result[2] is not None else "N/A"
        min_score = f"{result[3]:.2f}" if result[3] is not None else "N/A"
        max_score = f"{result[4]:.2f}" if result[4] is not None else "N/A"
        print(f"   Average score: {avg_score}")
        print(f"   Min score: {min_score}")
        print(f"   Max score: {max_score}")
        
        # Check l2_security_metrics (if table exists)
        if l2_table_exists:
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       COUNT(security_score) as with_score,
                       AVG(security_score) as avg_score,
                       MIN(security_score) as min_score,
                       MAX(security_score) as max_score
                FROM l2_security_metrics
            """)
            
            l2_result = cursor.fetchone()
            print(f"📊 L2 Security Metrics Summary:")
            print(f"   Total records: {l2_result[0]}")
            print(f"   Records with score: {l2_result[1]}")
            l2_avg_score = f"{l2_result[2]:.2f}" if l2_result[2] is not None else "N/A"
            l2_min_score = f"{l2_result[3]:.2f}" if l2_result[3] is not None else "N/A"
            l2_max_score = f"{l2_result[4]:.2f}" if l2_result[4] is not None else "N/A"
            print(f"   Average score: {l2_avg_score}")
            print(f"   Min score: {l2_min_score}")
            print(f"   Max score: {l2_max_score}")
        else:
            print("📊 L2 Security Metrics: Table does not exist")
        
        print("🎉 Security Score Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            print("🔌 Database connection closed")


if __name__ == "__main__":
    asyncio.run(migrate_security_score())
