#!/usr/bin/env python3
"""
Monitor Comprehensive Polkadot Archive Collection
================================================

Real-time monitoring of the comprehensive archive data collection.
"""

import sqlite3
import time
import os
import subprocess
from datetime import datetime, timedelta

def get_collection_stats():
    """Get current collection statistics"""
    db_path = "polkadot_archive_data.db"
    
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get block metrics count
        cursor.execute("SELECT COUNT(*) FROM block_metrics")
        block_count = cursor.fetchone()[0]
        
        # Get latest block
        cursor.execute("SELECT MAX(block_number) FROM block_metrics")
        latest_block = cursor.fetchone()[0] or 0
        
        # Get oldest block
        cursor.execute("SELECT MIN(block_number) FROM block_metrics")
        oldest_block = cursor.fetchone()[0] or 0
        
        # Get database size
        db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        
        # Get collection rate (blocks per minute)
        cursor.execute("SELECT COUNT(*) FROM block_metrics WHERE timestamp > datetime('now', '-1 minute')")
        recent_blocks = cursor.fetchone()[0]
        
        # Get total time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM block_metrics")
        time_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'block_count': block_count,
            'latest_block': latest_block,
            'oldest_block': oldest_block,
            'db_size_mb': db_size,
            'recent_blocks': recent_blocks,
            'time_range': time_range
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return None

def is_collection_running():
    """Check if collection process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'run_polkadot_archive_collector.py'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def estimate_completion():
    """Estimate completion time based on current progress"""
    stats = get_collection_stats()
    if not stats or stats['block_count'] < 100:
        return "Calculating..."
    
    # Comprehensive config: 365 days, every 5th block
    # Polkadot produces ~6 blocks per minute = 8,640 blocks per day
    # 365 days * 8,640 blocks/day / 5 (sample rate) = ~630,720 blocks total
    total_expected_blocks = 630720
    
    current_progress = stats['block_count'] / total_expected_blocks
    if current_progress >= 1.0:
        return "Completed!"
    
    # Estimate based on recent collection rate
    if stats['recent_blocks'] > 0:
        blocks_per_minute = stats['recent_blocks']
        remaining_blocks = total_expected_blocks - stats['block_count']
        remaining_minutes = remaining_blocks / blocks_per_minute
        eta = datetime.now() + timedelta(minutes=remaining_minutes)
        return eta.strftime("%Y-%m-%d %H:%M:%S")
    
    return "Calculating..."

def monitor_collection():
    """Main monitoring loop"""
    print("🚀 Comprehensive Polkadot Archive Collection Monitor")
    print("=" * 60)
    print("📊 Configuration: 365 days, every 5th block, 25 workers")
    print("🎯 Expected Total: ~630,720 blocks")
    print("=" * 60)
    
    start_time = datetime.now()
    last_count = 0
    
    while True:
        try:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🚀 Comprehensive Polkadot Archive Collection Monitor")
            print("=" * 60)
            print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Running for: {datetime.now() - start_time}")
            print()
            
            # Check if process is running
            if not is_collection_running():
                print("🔴 Collection Status: STOPPED")
                print("✅ Collection may have completed or encountered an error.")
                break
            
            print("🟢 Collection Status: RUNNING")
            print()
            
            # Get current stats
            stats = get_collection_stats()
            if not stats:
                print("❌ No data available yet...")
                time.sleep(5)
                continue
            
            # Calculate progress
            total_expected = 630720
            progress_percent = (stats['block_count'] / total_expected) * 100
            
            # Calculate collection rate
            current_count = stats['block_count']
            if last_count > 0:
                rate = current_count - last_count
                print(f"📈 Collection Rate: {rate} blocks in last 5 seconds")
            last_count = current_count
            
            print(f"📊 Progress: {progress_percent:.2f}% ({stats['block_count']:,} / {total_expected:,} blocks)")
            print(f"📦 Database Size: {stats['db_size_mb']:.1f} MB")
            print(f"🔢 Block Range: {stats['oldest_block']:,} to {stats['latest_block']:,}")
            print(f"📈 Recent Activity: {stats['recent_blocks']} blocks in last minute")
            
            if stats['time_range'] and stats['time_range'][0]:
                print(f"📅 Time Range: {stats['time_range'][0]} to {stats['time_range'][1]}")
            
            print()
            print(f"⏰ Estimated Completion: {estimate_completion()}")
            print()
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_collection()
