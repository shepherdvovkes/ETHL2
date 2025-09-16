#!/usr/bin/env python3
"""
Astar Collection Completion Notifier
Monitors the Astar data collection process and sends notifications when complete
"""

import asyncio
import sqlite3
import time
import os
import subprocess
import json
from datetime import datetime
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger

class AstarCompletionNotifier:
    """Monitors Astar collection and sends completion notifications"""
    
    def __init__(self, 
                 db_path: str = "astar_multithreaded_data.db",
                 target_block: int = 10227287,
                 check_interval: int = 30,
                 email_config: Optional[dict] = None):
        self.db_path = db_path
        self.target_block = target_block
        self.check_interval = check_interval
        self.email_config = email_config
        self.start_time = time.time()
        self.initial_blocks = self._get_current_block_count()
        self.notification_sent = False
        
    def _get_current_block_count(self) -> int:
        """Get current block count from database"""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM astar_blocks")
                count = cursor.fetchone()[0]
                conn.close()
                return count
            return 0
        except Exception as e:
            logger.warning(f"Error getting block count: {e}")
            return 0
    
    def _get_latest_block_number(self) -> int:
        """Get latest block number from database"""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(block_number) FROM astar_blocks")
                result = cursor.fetchone()[0]
                conn.close()
                return result or 0
            return 0
        except Exception as e:
            logger.warning(f"Error getting latest block: {e}")
            return 0
    
    def _get_process_status(self) -> dict:
        """Check if collection process is still running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "start_full_astar_collection.py"],
                capture_output=True,
                text=True
            )
            is_running = bool(result.stdout.strip())
            
            if is_running:
                pid = result.stdout.strip()
                return {"running": True, "pid": pid}
            else:
                return {"running": False, "pid": None}
        except Exception as e:
            logger.warning(f"Error checking process status: {e}")
            return {"running": False, "pid": None}
    
    def _send_desktop_notification(self, title: str, message: str):
        """Send desktop notification"""
        try:
            # Try notify-send (Linux)
            subprocess.run([
                "notify-send", 
                "-i", "applications-development",
                "-u", "critical",
                title,
                message
            ], check=False)
            logger.info("Desktop notification sent")
        except Exception as e:
            logger.warning(f"Failed to send desktop notification: {e}")
    
    def _send_email_notification(self, subject: str, body: str):
        """Send email notification"""
        if not self.email_config:
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
            server.quit()
            
            logger.info("Email notification sent")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _create_completion_report(self) -> dict:
        """Create completion report with statistics"""
        current_blocks = self._get_current_block_count()
        latest_block = self._get_latest_block_number()
        elapsed_time = time.time() - self.start_time
        
        # Get transaction count
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM astar_transactions")
            tx_count = cursor.fetchone()[0]
            
            # Get market data count
            cursor.execute("SELECT COUNT(*) FROM astar_market_data")
            market_count = cursor.fetchone()[0]
            
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting counts: {e}")
            tx_count = 0
            market_count = 0
        
        blocks_collected = current_blocks - self.initial_blocks
        
        return {
            "completion_time": datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed_time / 60, 2),
            "blocks_collected": blocks_collected,
            "total_blocks": current_blocks,
            "latest_block": latest_block,
            "target_block": self.target_block,
            "transactions": tx_count,
            "market_data_points": market_count,
            "blocks_per_minute": round(blocks_collected / (elapsed_time / 60), 2) if elapsed_time > 0 else 0,
            "completion_percentage": round((latest_block - (latest_block - blocks_collected)) / self.target_block * 100, 2)
        }
    
    def _send_completion_notifications(self, report: dict):
        """Send all completion notifications"""
        title = "🎉 Astar Collection Complete!"
        
        # Desktop notification
        desktop_msg = f"Collected {report['blocks_collected']:,} blocks in {report['elapsed_minutes']:.1f} minutes"
        self._send_desktop_notification(title, desktop_msg)
        
        # Email notification
        if self.email_config:
            subject = "Astar Blockchain Collection Completed Successfully"
            body = f"""
            <h2>🎉 Astar Collection Complete!</h2>
            <p><strong>Collection completed at:</strong> {report['completion_time']}</p>
            
            <h3>📊 Collection Statistics:</h3>
            <ul>
                <li><strong>Blocks Collected:</strong> {report['blocks_collected']:,}</li>
                <li><strong>Total Blocks:</strong> {report['total_blocks']:,}</li>
                <li><strong>Latest Block:</strong> {report['latest_block']:,}</li>
                <li><strong>Target Block:</strong> {report['target_block']:,}</li>
                <li><strong>Transactions:</strong> {report['transactions']:,}</li>
                <li><strong>Market Data Points:</strong> {report['market_data_points']:,}</li>
                <li><strong>Duration:</strong> {report['elapsed_minutes']:.1f} minutes</li>
                <li><strong>Rate:</strong> {report['blocks_per_minute']:.1f} blocks/minute</li>
                <li><strong>Completion:</strong> {report['completion_percentage']:.1f}%</li>
            </ul>
            
            <h3>🎯 Mission Accomplished!</h3>
            <p>The Astar blockchain data collection with integrated market data has been completed successfully.</p>
            
            <p><em>Generated by Astar Completion Notifier</em></p>
            """
            self._send_email_notification(subject, body)
        
        # Log completion
        logger.info("🎉 Astar Collection Complete!")
        logger.info(f"📊 Blocks: {report['blocks_collected']:,}")
        logger.info(f"⏱️  Time: {report['elapsed_minutes']:.1f} minutes")
        logger.info(f"🚀 Rate: {report['blocks_per_minute']:.1f} blocks/min")
    
    def _send_error_notification(self, error_msg: str):
        """Send error notification"""
        title = "❌ Astar Collection Error"
        message = f"Collection process stopped unexpectedly: {error_msg}"
        
        self._send_desktop_notification(title, message)
        
        if self.email_config:
            subject = "Astar Collection Error"
            body = f"""
            <h2>❌ Astar Collection Error</h2>
            <p><strong>Error occurred at:</strong> {datetime.now().isoformat()}</p>
            <p><strong>Error message:</strong> {error_msg}</p>
            <p>Please check the collection logs for more details.</p>
            """
            self._send_email_notification(subject, body)
        
        logger.error(f"❌ Collection error: {error_msg}")
    
    async def monitor_completion(self):
        """Main monitoring loop"""
        logger.info("🔔 Starting Astar completion monitoring...")
        logger.info(f"🎯 Target block: {self.target_block:,}")
        logger.info(f"⏰ Check interval: {self.check_interval} seconds")
        
        while not self.notification_sent:
            try:
                # Check process status
                process_status = self._get_process_status()
                latest_block = self._get_latest_block_number()
                
                # Log progress
                logger.info(f"📊 Latest block: {latest_block:,} | Process running: {process_status['running']}")
                
                # Check if target reached
                if latest_block >= self.target_block:
                    logger.info("🎯 Target block reached!")
                    report = self._create_completion_report()
                    self._send_completion_notifications(report)
                    self.notification_sent = True
                    break
                
                # Check if process stopped unexpectedly
                if not process_status['running'] and latest_block < self.target_block:
                    error_msg = f"Process stopped at block {latest_block:,}, target was {self.target_block:,}"
                    self._send_error_notification(error_msg)
                    self.notification_sent = True
                    break
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("👋 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
        
        if self.notification_sent:
            logger.info("✅ Notification monitoring complete")

async def main():
    """Main function"""
    # Email configuration (optional)
    email_config = None
    # Uncomment and configure if you want email notifications:
    # email_config = {
    #     'from_email': 'your-email@gmail.com',
    #     'to_email': 'recipient@gmail.com',
    #     'smtp_server': 'smtp.gmail.com',
    #     'smtp_port': 587,
    #     'password': 'your-app-password'
    # }
    
    notifier = AstarCompletionNotifier(
        db_path="astar_multithreaded_data.db",
        target_block=10227287,  # Current Astar block
        check_interval=30,  # Check every 30 seconds
        email_config=email_config
    )
    
    await notifier.monitor_completion()

if __name__ == "__main__":
    asyncio.run(main())
