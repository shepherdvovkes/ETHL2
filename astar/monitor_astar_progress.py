#!/usr/bin/env python3
"""
Monitor Astar Collection Progress
================================

Real-time monitoring of Astar data collection progress.
"""

import sqlite3
import time
import os
from datetime import datetime

def monitor_progress():
    """Monitor collection progress in real-time"""
    print("🔍 Astar Collection Progress Monitor")
    print("=" * 50)
    
    # Check which database to monitor
    db_files = [
        'astar_market_enhanced_data.db',
        'astar_multithreaded_data.db',
        'astar_enhanced_market_data.db'
    ]
    
    db_path = None
    for db_file in db_files:
        if os.path.exists(db_file):
            db_path = db_file
            break
    
    if not db_path:
        print("❌ No Astar database found")
        return
    
    print(f"📊 Monitoring: {db_path}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    start_time = time.time()
    last_blocks = 0
    last_transactions = 0
    
    try:
        while True:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            try:
                # Try market enhanced tables first
                try:
                    cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_blocks")
                    blocks_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_transactions")
                    transactions_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_market_enhanced_blocks")
                    block_range = cursor.fetchone()
                    
                    cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_combined_data")
                    combined_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM astar_market_enhanced_market_data")
                    market_count = cursor.fetchone()[0]
                    
                    table_type = "Market Enhanced"
                    
                except sqlite3.OperationalError:
                    # Fallback to regular tables
                    cursor.execute("SELECT COUNT(*) FROM astar_blocks")
                    blocks_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM astar_transactions")
                    transactions_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM astar_blocks")
                    block_range = cursor.fetchone()
                    
                    combined_count = 0
                    market_count = 0
                    table_type = "Standard"
                
                # Calculate rates
                elapsed_time = time.time() - start_time
                blocks_rate = (blocks_count - last_blocks) / 5 if elapsed_time > 0 else 0
                tx_rate = (transactions_count - last_transactions) / 5 if elapsed_time > 0 else 0
                
                # Clear screen and show progress
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🔍 Astar Collection Progress Monitor")
                print("=" * 50)
                print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
                print(f"📊 Database: {db_path}")
                print(f"📊 Table Type: {table_type}")
                print(f"📦 Blocks Collected: {blocks_count:,}")
                print(f"💸 Transactions Collected: {transactions_count:,}")
                
                if block_range[0] and block_range[1]:
                    print(f"📈 Block Range: {block_range[0]:,} - {block_range[1]:,}")
                    blocks_missing = 10227287 - block_range[1] if block_range[1] else 10227287
                    print(f"📊 Blocks Remaining: {blocks_missing:,}")
                    
                    progress = (block_range[1] / 10227287) * 100 if block_range[1] else 0
                    print(f"📊 Progress: {progress:.2f}%")
                
                if combined_count > 0:
                    print(f"🔄 Combined Data Points: {combined_count:,}")
                
                if market_count > 0:
                    print(f"💰 Market Data Points: {market_count:,}")
                
                print(f"🚀 Blocks/sec: {blocks_rate:.1f}")
                print(f"💸 TX/sec: {tx_rate:.1f}")
                print(f"⏱️  Elapsed: {elapsed_time/60:.1f} minutes")
                
                # Estimate completion
                if blocks_rate > 0 and block_range[1]:
                    remaining_blocks = 10227287 - block_range[1]
                    eta_seconds = remaining_blocks / blocks_rate
                    eta_hours = eta_seconds / 3600
                    print(f"⏰ ETA: {eta_hours:.1f} hours")
                
                print("=" * 50)
                print("Press Ctrl+C to stop monitoring")
                
                last_blocks = blocks_count
                last_transactions = transactions_count
                
            except Exception as e:
                print(f"❌ Error reading database: {e}")
            
            finally:
                conn.close()
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
    except Exception as e:
        print(f"❌ Error monitoring: {e}")

if __name__ == "__main__":
    monitor_progress()
