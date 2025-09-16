#!/usr/bin/env python3
"""
Monitor Polkadot Archive Collection
==================================

Monitor the progress of the archive data collection.
"""

import sqlite3
import time
import os
from datetime import datetime

def monitor_collection():
    """Monitor the collection progress"""
    db_path = "polkadot_archive_data.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found. Collection may not have started yet.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get block metrics count
        cursor.execute("SELECT COUNT(*) FROM block_metrics")
        block_count = cursor.fetchone()[0]
        
        # Get latest block
        cursor.execute("SELECT MAX(block_number) FROM block_metrics")
        latest_block = cursor.fetchone()[0] or 0
        
        # Get staking data count
        cursor.execute("SELECT COUNT(*) FROM staking_data")
        staking_count = cursor.fetchone()[0]
        
        # Get parachain data count
        cursor.execute("SELECT COUNT(*) FROM parachain_data")
        parachain_count = cursor.fetchone()[0]
        
        # Get governance data count
        cursor.execute("SELECT COUNT(*) FROM governance_data")
        governance_count = cursor.fetchone()[0]
        
        # Get database size
        db_size = os.path.getsize(db_path) / 1024  # KB
        
        print("📊 Polkadot Archive Collection Status")
        print("=" * 40)
        print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📦 Database Size: {db_size:.1f} KB")
        print(f"🔢 Block Records: {block_count:,}")
        print(f"📈 Latest Block: {latest_block:,}")
        print(f"🏛️  Staking Records: {staking_count}")
        print(f"🔗 Parachain Records: {parachain_count}")
        print(f"🗳️  Governance Records: {governance_count}")
        
        # Check if collection is still running
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'run_polkadot_archive_collector.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🟢 Collection Status: RUNNING")
        else:
            print("🔴 Collection Status: STOPPED")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error monitoring collection: {e}")

if __name__ == "__main__":
    monitor_collection()
