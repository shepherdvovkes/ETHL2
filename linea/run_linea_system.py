#!/usr/bin/env python3
"""
LINEA System Runner
Orchestrates the LINEA blockchain data collection system
"""

import asyncio
import subprocess
import sys
import time
import logging
import os
import signal
from pathlib import Path
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LineaSystemRunner:
    """Main system runner for LINEA data collection"""
    
    def __init__(self):
        self.processes = {}
        self.is_running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_database(self):
        """Setup database schema"""
        logger.info("🗄️  Setting up LINEA database schema...")
        try:
            result = subprocess.run([
                sys.executable, "linea_database_schema.py"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info("✅ Database schema setup completed")
                return True
            else:
                logger.error(f"❌ Database schema setup failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Database schema setup error: {e}")
            return False
    
    def start_data_collector(self):
        """Start the LINEA data collector"""
        logger.info("🚀 Starting LINEA data collector...")
        try:
            process = subprocess.Popen([
                sys.executable, "linea_data_collector.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
               cwd=os.getcwd())
            
            self.processes['data_collector'] = process
            logger.info(f"✅ Data collector started with PID {process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start data collector: {e}")
            return False
    
    def start_archive_collector(self):
        """Start the LINEA archive collector"""
        logger.info("📚 Starting LINEA archive collector...")
        try:
            process = subprocess.Popen([
                sys.executable, "linea_archive_collector.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               cwd=os.getcwd())
            
            self.processes['archive_collector'] = process
            logger.info(f"✅ Archive collector started with PID {process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start archive collector: {e}")
            return False
    
    def start_metrics_server(self):
        """Start the LINEA metrics server"""
        logger.info("🌐 Starting LINEA metrics server...")
        try:
            process = subprocess.Popen([
                sys.executable, "linea_metrics_server.py", "--host", "0.0.0.0", "--port", "8008"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               cwd=os.getcwd())
            
            self.processes['metrics_server'] = process
            logger.info(f"✅ Metrics server started with PID {process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.is_running:
            try:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"❌ Process {name} (PID {process.pid}) has stopped")
                        del self.processes[name]
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                time.sleep(5)
    
    def start_system(self):
        """Start the complete LINEA system"""
        logger.info("🚀 Starting LINEA blockchain data collection system...")
        self.is_running = True
        
        # Setup database
        if not self.setup_database():
            logger.error("❌ Failed to setup database, exiting")
            return False
        
        # Start data collector
        if not self.start_data_collector():
            logger.error("❌ Failed to start data collector")
            self.stop_system()
            return False
        
        # Start archive collector
        if not self.start_archive_collector():
            logger.error("❌ Failed to start archive collector")
            self.stop_system()
            return False
        
        # Start metrics server
        if not self.start_metrics_server():
            logger.error("❌ Failed to start metrics server")
            self.stop_system()
            return False
        
        logger.info("✅ All LINEA system components started successfully!")
        logger.info("📊 System Status:")
        logger.info(f"   - Data Collector: PID {self.processes.get('data_collector', {}).pid}")
        logger.info(f"   - Archive Collector: PID {self.processes.get('archive_collector', {}).pid}")
        logger.info(f"   - Metrics Server: PID {self.processes.get('metrics_server', {}).pid}")
        logger.info("🌐 Metrics Server available at: http://localhost:8008")
        logger.info("📚 API Documentation: http://localhost:8008/docs")
        
        return True
    
    def stop_system(self):
        """Stop all system components"""
        logger.info("🛑 Stopping LINEA system...")
        self.is_running = False
        
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name} (PID {process.pid})...")
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  {name} didn't stop gracefully, killing...")
                process.kill()
                process.wait()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("✅ LINEA system stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)
    
    def run(self):
        """Main run function"""
        try:
            if self.start_system():
                # Start monitoring in a separate thread
                monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
                monitor_thread.start()
                
                logger.info("🔄 System is running... Press Ctrl+C to stop")
                
                # Keep main thread alive
                while self.is_running:
                    time.sleep(1)
            else:
                logger.error("❌ Failed to start system")
                sys.exit(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop_system()

def main():
    """Main function"""
    runner = LineaSystemRunner()
    runner.run()

if __name__ == "__main__":
    main()
