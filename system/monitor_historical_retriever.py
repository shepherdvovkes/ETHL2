#!/usr/bin/env python3
"""
Monitor for Polkadot Historical Data Retriever
==============================================

This script monitors the progress of the historical data retriever
and provides real-time statistics.
"""

import sqlite3
import time
import requests
from datetime import datetime, timezone
import json

def get_current_mainnet_block():
    """Get current block from Polkadot mainnet"""
    try:
        response = requests.post(
            "https://rpc.polkadot.io",
            json={"id": 1, "jsonrpc": "2.0", "method": "chain_getBlock", "params": []},
            timeout=10
        )
        result = response.json()
        if "result" in result:
            return int(result["result"]["block"]["header"]["number"], 16)
    except Exception as e:
        print(f"Error getting mainnet block: {e}")
    return None

def get_database_stats():
    """Get statistics from the database"""
    try:
        conn = sqlite3.connect("polkadot_archive_data.db")
        cursor = conn.cursor()
        
        # Get block metrics stats
        cursor.execute("SELECT COUNT(*) as total_blocks, MAX(block_number) as latest_block FROM block_metrics")
        block_stats = cursor.fetchone()
        
        # Get detailed block stats
        cursor.execute("SELECT COUNT(*) as total_details FROM block_details")
        detail_stats = cursor.fetchone()
        
        # Get recent collection rate
        cursor.execute("""
            SELECT COUNT(*) as recent_blocks 
            FROM block_metrics 
            WHERE created_at > datetime('now', '-1 hour')
        """)
        recent_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_blocks": block_stats[0],
            "latest_block": block_stats[1],
            "total_details": detail_stats[0],
            "recent_blocks": recent_stats[0]
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return None

def main():
    """Main monitoring loop"""
    print("🔍 Polkadot Historical Data Retriever Monitor")
    print("=" * 50)
    
    while True:
        try:
            # Get current mainnet block
            mainnet_block = get_current_mainnet_block()
            
            # Get database stats
            db_stats = get_database_stats()
            
            if db_stats and mainnet_block:
                gap = mainnet_block - db_stats["latest_block"]
                progress = (db_stats["latest_block"] / mainnet_block) * 100
                
                print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   Mainnet Block:     {mainnet_block:,}")
                print(f"   Database Block:    {db_stats['latest_block']:,}")
                print(f"   Gap:               {gap:,} blocks")
                print(f"   Progress:          {progress:.2f}%")
                print(f"   Total Blocks:      {db_stats['total_blocks']:,}")
                print(f"   Detailed Records:  {db_stats['total_details']:,}")
                print(f"   Recent (1h):       {db_stats['recent_blocks']:,} blocks")
                
                if gap <= 0:
                    print("   ✅ Database is up-to-date!")
                elif gap < 1000:
                    print("   🟡 Almost caught up!")
                else:
                    print("   🔄 Still catching up...")
                    
            else:
                print(f"❌ Error getting data at {datetime.now().strftime('%H:%M:%S')}")
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    main()


