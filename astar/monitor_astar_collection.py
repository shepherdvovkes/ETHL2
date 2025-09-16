#!/usr/bin/env python3
"""
Astar Collection Monitor
=======================

Monitor the progress of Astar data collection in real-time.
"""

import sqlite3
import time
import os
from datetime import datetime
from loguru import logger

def monitor_collection():
    """Monitor the collection progress"""
    db_path = "astar_multithreaded_data.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found. Collection may not have started yet.")
        return
    
    print("🔍 Monitoring Astar Data Collection Progress")
    print("=" * 50)
    
    start_time = time.time()
    last_blocks = 0
    last_transactions = 0
    
    try:
        while True:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get current counts
            cursor.execute("SELECT COUNT(*) FROM astar_blocks")
            blocks_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM astar_transactions")
            transactions_count = cursor.fetchone()[0]
            
            # Get block range
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
            block_range = cursor.fetchone()
            
            # Get recent activity
            cursor.execute("SELECT AVG(transaction_count) FROM astar_blocks ORDER BY block_number DESC LIMIT 100")
            avg_tx_per_block = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # Calculate rates
            elapsed_time = time.time() - start_time
            blocks_rate = (blocks_count - last_blocks) / 5 if elapsed_time > 0 else 0  # Per 5 seconds
            tx_rate = (transactions_count - last_transactions) / 5 if elapsed_time > 0 else 0
            
            # Clear screen and show progress
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🔍 Astar Data Collection Monitor")
            print("=" * 50)
            print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"📊 Blocks Collected: {blocks_count:,}")
            print(f"💸 Transactions Collected: {transactions_count:,}")
            print(f"📈 Block Range: {block_range[0]:,} - {block_range[1]:,}")
            print(f"📊 Avg TX per Block: {avg_tx_per_block:.2f}")
            print(f"🚀 Blocks/sec: {blocks_rate:.1f}")
            print(f"💸 TX/sec: {tx_rate:.1f}")
            print(f"⏱️  Elapsed: {elapsed_time/60:.1f} minutes")
            print("=" * 50)
            print("Press Ctrl+C to stop monitoring")
            
            last_blocks = blocks_count
            last_transactions = transactions_count
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
    except Exception as e:
        print(f"❌ Error monitoring: {e}")

if __name__ == "__main__":
    monitor_collection()