#!/usr/bin/env python3
"""
Restart Avalanche Server with Rate Limiting
Stops the current server and starts a new one with reduced API calls
"""

import subprocess
import time
import signal
import os
import sys
from loguru import logger

def stop_existing_servers():
    """Stop any existing Avalanche servers"""
    logger.info("🛑 Stopping existing Avalanche servers...")
    
    try:
        # Kill any existing Python processes running Avalanche servers
        subprocess.run(['pkill', '-f', 'avalanche_api_server'], check=False)
        subprocess.run(['pkill', '-f', 'avalanche_realtime_server'], check=False)
        subprocess.run(['pkill', '-f', 'run_avalanche_server'], check=False)
        
        # Wait a moment for processes to stop
        time.sleep(2)
        
        logger.info("✅ Existing servers stopped")
        
    except Exception as e:
        logger.warning(f"Error stopping servers: {e}")

def start_rate_limited_server():
    """Start the rate-limited server"""
    logger.info("🚀 Starting rate-limited Avalanche server...")
    
    try:
        # Start the API server with rate limiting
        cmd = [
            sys.executable, 'avalanche_api_server.py'
        ]
        
        # Start in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        logger.info(f"✅ Rate-limited server started with PID: {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return None

def main():
    """Main function"""
    logger.info("🔄 Restarting Avalanche Server with Rate Limiting")
    
    # Stop existing servers
    stop_existing_servers()
    
    # Start new rate-limited server
    process = start_rate_limited_server()
    
    if process:
        logger.info("✅ Server restart completed successfully")
        logger.info("📊 Dashboard available at: http://localhost:8000")
        logger.info("🔍 Health check: http://localhost:8000/health")
        logger.info("📈 Metrics: http://localhost:8000/metrics/summary")
        logger.info("")
        logger.info("Rate limiting changes:")
        logger.info("• CoinGecko API: 5-minute intervals (was 1 minute)")
        logger.info("• GitHub API: 10-minute intervals (was 1 minute)")
        logger.info("• Development metrics: Using fallback data")
        logger.info("• Economic metrics: Using fallback data with periodic updates")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            # Keep the script running
            process.wait()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping server...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
            logger.info("✅ Server stopped")
    else:
        logger.error("❌ Failed to start server")

if __name__ == "__main__":
    main()
