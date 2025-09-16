#!/usr/bin/env python3
"""
Data Collection Scheduler for Polkadot Metrics
==============================================

This script schedules and manages the collection of real data from external sources.
It runs the data collection pipeline at regular intervals and manages the data flow.
"""

import asyncio
import schedule
import time
from datetime import datetime
from loguru import logger
import subprocess
import os
import signal
import sys

# Configure logging
logger.add("logs/data_scheduler.log", rotation="1 day", retention="30 days")

class DataCollectionScheduler:
    """Manages scheduled data collection"""
    
    def __init__(self):
        self.running = True
        self.collection_interval = 5  # minutes
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def run_data_collection(self):
        """Run the data collection process"""
        try:
            logger.info("Starting scheduled data collection...")
            
            # Run the real data collector
            process = await asyncio.create_subprocess_exec(
                sys.executable, "real_data_collector.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Data collection completed successfully")
                if stdout:
                    logger.info(f"Output: {stdout.decode()}")
            else:
                logger.error(f"Data collection failed with return code {process.returncode}")
                if stderr:
                    logger.error(f"Error: {stderr.decode()}")
                    
        except Exception as e:
            logger.error(f"Error running data collection: {e}")
    
    def schedule_collections(self):
        """Setup collection schedule"""
        logger.info(f"Scheduling data collection every {self.collection_interval} minutes")
        
        # Schedule data collection every 5 minutes
        schedule.every(self.collection_interval).minutes.do(
            lambda: asyncio.run(self.run_data_collection())
        )
        
        # Schedule daily cleanup at midnight
        schedule.every().day.at("00:00").do(self.daily_cleanup)
        
        # Schedule weekly report on Sundays at 6 AM
        schedule.every().sunday.at("06:00").do(self.weekly_report)
    
    def daily_cleanup(self):
        """Daily cleanup tasks"""
        logger.info("Running daily cleanup...")
        
        try:
            # Clean up old log files
            os.system("find logs/ -name '*.log' -mtime +30 -delete")
            
            # Optimize database
            os.system("sqlite3 polkadot_metrics.db 'VACUUM;'")
            
            logger.info("Daily cleanup completed")
        except Exception as e:
            logger.error(f"Error during daily cleanup: {e}")
    
    def weekly_report(self):
        """Generate weekly data collection report"""
        logger.info("Generating weekly report...")
        
        try:
            # Get database statistics
            import sqlite3
            conn = sqlite3.connect("polkadot_metrics.db")
            cursor = conn.cursor()
            
            # Count records in each table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            report = "Weekly Data Collection Report\n"
            report += "=" * 40 + "\n"
            report += f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            total_records = 0
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                total_records += count
                report += f"{table_name}: {count} records\n"
            
            report += f"\nTotal Records: {total_records}\n"
            report += f"Data Collection Interval: {self.collection_interval} minutes\n"
            
            # Save report
            with open(f"reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.txt", "w") as f:
                f.write(report)
            
            logger.info(f"Weekly report generated: {total_records} total records")
            conn.close()
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
    
    async def run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Starting data collection scheduler...")
        
        self.schedule_collections()
        
        while self.running:
            try:
                schedule.run_pending()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Data collection scheduler stopped")

async def main():
    """Main function"""
    logger.info("Starting Polkadot Data Collection Scheduler...")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    scheduler = DataCollectionScheduler()
    
    try:
        await scheduler.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
    finally:
        logger.info("Scheduler shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
