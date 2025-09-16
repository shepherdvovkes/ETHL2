#!/usr/bin/env python3
"""
Quick Astar Collection Progress Check
===================================
"""

import sqlite3
import os
from datetime import datetime

def check_progress():
    db_path = "astar_multithreaded_data.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get current counts
        cursor.execute("SELECT COUNT(*) FROM astar_blocks")
        blocks_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM astar_transactions")
        tx_count = cursor.fetchone()[0]
        
        # Get block range
        cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
        block_range = cursor.fetchone()
        
        # Get recent activity
        cursor.execute("SELECT AVG(transaction_count) FROM astar_blocks ORDER BY block_number DESC LIMIT 100")
        avg_tx_per_block = cursor.fetchone()[0] or 0
        
        print("🔍 Astar Collection Progress")
        print("=" * 40)
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Blocks Collected: {blocks_count:,}")
        print(f"💸 Transactions Collected: {tx_count:,}")
        print(f"📈 Block Range: {block_range[0]:,} - {block_range[1]:,}")
        print(f"📊 Avg TX per Block: {avg_tx_per_block:.2f}")
        print(f"📏 Total Range: {block_range[1] - block_range[0] + 1:,} blocks")
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_progress()
