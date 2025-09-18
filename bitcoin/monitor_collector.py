#!/usr/bin/env python3
"""
Bitcoin Chain Collector Monitor
Monitor the progress of blockchain data collection
"""

import sqlite3
import time
import json
from datetime import datetime, timedelta
from bitcoin_chain_collector import BitcoinDatabase

class CollectorMonitor:
    """Monitor for Bitcoin chain collection progress"""
    
    def __init__(self, db_path: str = "bitcoin_chain.db"):
        self.db_path = db_path
        self.db = BitcoinDatabase(db_path)
    
    def get_progress_stats(self) -> dict:
        """Get detailed progress statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Overall progress
            cursor.execute("SELECT COUNT(*) FROM blocks")
            total_blocks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            total_transactions = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(height) FROM blocks")
            latest_height = cursor.fetchone()[0] or 0
            
            # Worker progress
            cursor.execute("""
                SELECT worker_id, status, blocks_collected, transactions_collected,
                       started_at, completed_at
                FROM collection_progress
                ORDER BY worker_id
            """)
            worker_progress = cursor.fetchall()
            
            # Collection rate (last hour)
            cursor.execute("""
                SELECT COUNT(*) FROM collection_progress 
                WHERE status = 'completed' AND completed_at > datetime('now', '-1 hour')
            """)
            recent_blocks = cursor.fetchone()[0]
            
            # Time estimates
            cursor.execute("""
                SELECT AVG(blocks_collected) FROM collection_progress 
                WHERE status = 'completed' AND completed_at > datetime('now', '-1 hour')
            """)
            avg_blocks_per_hour = cursor.fetchone()[0] or 0
            
            return {
                'total_blocks': total_blocks,
                'total_transactions': total_transactions,
                'latest_height': latest_height,
                'recent_blocks_per_hour': recent_blocks,
                'avg_blocks_per_hour': avg_blocks_per_hour,
                'worker_progress': [
                    {
                        'worker_id': row[0],
                        'status': row[1],
                        'blocks_collected': row[2],
                        'transactions_collected': row[3],
                        'started_at': row[4],
                        'completed_at': row[5]
                    }
                    for row in worker_progress
                ]
            }
    
    def get_blockchain_info(self) -> dict:
        """Get blockchain information from collected data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get block height range
            cursor.execute("SELECT MIN(height), MAX(height) FROM blocks")
            min_height, max_height = cursor.fetchone()
            
            # Get transaction statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_txs,
                    AVG(size) as avg_tx_size,
                    AVG(fee) as avg_fee,
                    SUM(fee) as total_fees
                FROM transactions
            """)
            tx_stats = cursor.fetchone()
            
            # Get block statistics
            cursor.execute("""
                SELECT 
                    AVG(size) as avg_block_size,
                    AVG(tx_count) as avg_tx_per_block,
                    AVG(difficulty) as avg_difficulty
                FROM blocks
            """)
            block_stats = cursor.fetchone()
            
            return {
                'height_range': {'min': min_height, 'max': max_height},
                'transaction_stats': {
                    'total': tx_stats[0],
                    'avg_size': tx_stats[1],
                    'avg_fee': tx_stats[2],
                    'total_fees': tx_stats[3]
                },
                'block_stats': {
                    'avg_size': block_stats[0],
                    'avg_tx_per_block': block_stats[1],
                    'avg_difficulty': block_stats[2]
                }
            }
    
    def print_progress_report(self):
        """Print a detailed progress report"""
        stats = self.get_progress_stats()
        blockchain_info = self.get_blockchain_info()
        
        print("=" * 60)
        print("📊 BITCOIN CHAIN COLLECTION PROGRESS REPORT")
        print("=" * 60)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Overall Progress
        print("📈 OVERALL PROGRESS:")
        print(f"   • Blocks Collected: {stats['total_blocks']:,}")
        print(f"   • Transactions Collected: {stats['total_transactions']:,}")
        print(f"   • Latest Block Height: {stats['latest_height']:,}")
        print(f"   • Recent Rate: {stats['recent_blocks_per_hour']} blocks/hour")
        print(f"   • Average Rate: {stats['avg_blocks_per_hour']:.1f} blocks/hour")
        print()
        
        # Blockchain Info
        if blockchain_info['height_range']['min'] is not None:
            print("🔗 BLOCKCHAIN DATA:")
            print(f"   • Height Range: {blockchain_info['height_range']['min']:,} - {blockchain_info['height_range']['max']:,}")
            print(f"   • Total Transactions: {blockchain_info['transaction_stats']['total']:,}")
            print(f"   • Average TX Size: {blockchain_info['transaction_stats']['avg_size']:.1f} bytes")
            print(f"   • Average TX Fee: {blockchain_info['transaction_stats']['avg_fee']:.8f} BTC")
            print(f"   • Average Block Size: {blockchain_info['block_stats']['avg_size']:.1f} bytes")
            print(f"   • Average TX per Block: {blockchain_info['block_stats']['avg_tx_per_block']:.1f}")
            print()
        
        # Worker Progress
        print("👥 WORKER PROGRESS:")
        for worker in stats['worker_progress']:
            status_emoji = "✅" if worker['status'] == 'completed' else "🔄" if worker['status'] == 'started' else "❌"
            print(f"   {status_emoji} Worker {worker['worker_id']}: {worker['status']}")
            print(f"      • Blocks: {worker['blocks_collected']:,}")
            print(f"      • Transactions: {worker['transactions_collected']:,}")
            if worker['started_at']:
                print(f"      • Started: {worker['started_at']}")
            if worker['completed_at']:
                print(f"      • Completed: {worker['completed_at']}")
            print()
        
        # Time Estimates
        if stats['avg_blocks_per_hour'] > 0:
            remaining_blocks = 800000 - stats['latest_height']  # Approximate total Bitcoin blocks
            if remaining_blocks > 0:
                eta_hours = remaining_blocks / stats['avg_blocks_per_hour']
                eta_days = eta_hours / 24
                print("⏱️  TIME ESTIMATES:")
                print(f"   • Remaining Blocks: {remaining_blocks:,}")
                print(f"   • Estimated Time: {eta_days:.1f} days ({eta_hours:.1f} hours)")
                print()
        
        print("=" * 60)
    
    def save_progress_json(self, filename: str = "collection_progress.json"):
        """Save progress data to JSON file"""
        stats = self.get_progress_stats()
        blockchain_info = self.get_blockchain_info()
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'progress': stats,
            'blockchain_info': blockchain_info
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Progress data saved to {filename}")
    
    def monitor_continuous(self, interval: int = 60):
        """Continuously monitor progress"""
        print(f"🔄 Starting continuous monitoring (updating every {interval} seconds)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.print_progress_report()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Bitcoin chain collection progress")
    parser.add_argument("--db", default="bitcoin_chain.db", help="Database file path")
    parser.add_argument("--json", help="Save progress to JSON file")
    parser.add_argument("--continuous", type=int, metavar="SECONDS", 
                       help="Continuous monitoring with specified interval")
    
    args = parser.parse_args()
    
    monitor = CollectorMonitor(args.db)
    
    if args.continuous:
        monitor.monitor_continuous(args.continuous)
    else:
        monitor.print_progress_report()
        
        if args.json:
            monitor.save_progress_json(args.json)

if __name__ == "__main__":
    main()
