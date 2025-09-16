#!/usr/bin/env python3
"""
Avalanche Network Real-Time Metrics Server
Main startup script that orchestrates all components
"""

import asyncio
import signal
import sys
import os
import time
from datetime import datetime
from loguru import logger
import argparse
import json
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from avalanche_realtime_server import RealTimeDataCollector
from avalanche_monitoring_system import AvalancheMonitoringSystem
from avalanche_api_server import app as api_app
import uvicorn
from config.settings import settings

class AvalancheServerOrchestrator:
    """Orchestrates all Avalanche server components"""
    
    def __init__(self):
        self.data_collector = None
        self.monitoring_system = None
        self.api_server_task = None
        self.running = False
        self.start_time = None
        
        # Component status
        self.component_status = {
            "data_collector": False,
            "monitoring_system": False,
            "api_server": False
        }
    
    async def start_data_collector(self):
        """Start the real-time data collector"""
        logger.info("🚀 Starting Avalanche Data Collector...")
        
        self.data_collector = RealTimeDataCollector()
        await self.data_collector.__aenter__()
        
        # Start collection tasks
        await self.data_collector.start_all_collection_tasks()
        self.component_status["data_collector"] = True
        
        logger.info("✅ Data Collector started successfully")
    
    async def start_monitoring_system(self):
        """Start the monitoring and alerting system"""
        logger.info("🔍 Starting Avalanche Monitoring System...")
        
        self.monitoring_system = AvalancheMonitoringSystem()
        
        # Start monitoring with data collector
        monitoring_task = asyncio.create_task(
            self.monitoring_system.start_monitoring(self.data_collector)
        )
        
        self.component_status["monitoring_system"] = True
        logger.info("✅ Monitoring System started successfully")
        
        return monitoring_task
    
    async def start_api_server(self):
        """Start the FastAPI server"""
        logger.info("🌐 Starting Avalanche API Server...")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=api_app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        # Start server in background
        self.api_server_task = asyncio.create_task(server.serve())
        self.component_status["api_server"] = True
        
        logger.info(f"✅ API Server started on {settings.API_HOST}:{settings.API_PORT}")
        return self.api_server_task
    
    async def start_all_components(self):
        """Start all server components"""
        logger.info("🚀 Starting Avalanche Network Real-Time Metrics Server")
        self.start_time = datetime.utcnow()
        self.running = True
        
        try:
            # Start data collector first
            await self.start_data_collector()
            
            # Start monitoring system
            monitoring_task = await self.start_monitoring_system()
            
            # Start API server
            api_task = await self.start_api_server()
            
            # Print startup summary
            self.print_startup_summary()
            
            # Wait for all tasks
            await asyncio.gather(monitoring_task, api_task)
        
        except Exception as e:
            logger.error(f"Error starting components: {e}")
            await self.stop_all_components()
            raise
    
    async def stop_all_components(self):
        """Stop all server components"""
        logger.info("🛑 Stopping Avalanche Server components...")
        self.running = False
        
        # Stop monitoring system
        if self.monitoring_system:
            self.monitoring_system.stop_monitoring()
            self.component_status["monitoring_system"] = False
            logger.info("✅ Monitoring System stopped")
        
        # Stop data collector
        if self.data_collector:
            await self.data_collector.stop()
            await self.data_collector.__aexit__(None, None, None)
            self.component_status["data_collector"] = False
            logger.info("✅ Data Collector stopped")
        
        # Stop API server
        if self.api_server_task and not self.api_server_task.done():
            self.api_server_task.cancel()
            try:
                await self.api_server_task
            except asyncio.CancelledError:
                pass
            self.component_status["api_server"] = False
            logger.info("✅ API Server stopped")
        
        logger.info("✅ All components stopped successfully")
    
    def print_startup_summary(self):
        """Print startup summary"""
        print("\n" + "="*80)
        print("🚀 AVALANCHE NETWORK REAL-TIME METRICS SERVER")
        print("="*80)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🌐 API Server: http://{settings.API_HOST}:{settings.API_PORT}")
        print(f"📚 API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        print(f"🔍 Health Check: http://{settings.API_HOST}:{settings.API_PORT}/health")
        print("\n📊 COMPONENT STATUS:")
        for component, status in self.component_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        print("\n🔗 KEY ENDPOINTS:")
        print(f"  • Metrics Summary: /metrics/summary")
        print(f"  • Network Performance: /metrics/network-performance")
        print(f"  • Economic Metrics: /metrics/economic")
        print(f"  • DeFi Metrics: /metrics/defi")
        print(f"  • Subnet Metrics: /metrics/subnets")
        print(f"  • Security Metrics: /metrics/security")
        print(f"  • All Metrics: /metrics/all")
        print(f"  • Historical Data: /historical/24")
        print(f"  • Trigger Collection: /collect")
        
        print("\n⚙️ COLLECTION INTERVALS:")
        intervals = {
            "Network Performance": "30 seconds",
            "Economic Data": "1 minute",
            "DeFi Metrics": "2 minutes",
            "Subnet Data": "5 minutes",
            "Security Status": "10 minutes",
            "Development Activity": "30 minutes",
            "User Behavior": "5 minutes",
            "Competitive Position": "1 hour",
            "Technical Health": "1 minute",
            "Risk Indicators": "30 minutes",
            "Macro Environment": "30 minutes",
            "Ecosystem Health": "1 hour"
        }
        
        for metric, interval in intervals.items():
            print(f"  • {metric}: {interval}")
        
        print("\n🚨 ALERTING CHANNELS:")
        print("  • Email notifications")
        print("  • Webhook alerts")
        print("  • Slack integration")
        print("  • Telegram bot")
        
        print("\n📈 MONITORED METRICS:")
        print("  • Transaction throughput and finality")
        print("  • Gas prices and network utilization")
        print("  • Market cap and trading volume")
        print("  • DeFi TVL and protocol activity")
        print("  • Subnet count and validator status")
        print("  • Security metrics and risk indicators")
        print("  • Development activity and ecosystem health")
        
        print("="*80)
        print("🎯 Server is running! Press Ctrl+C to stop.")
        print("="*80 + "\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": uptime,
            "components": self.component_status.copy(),
            "data_collector_status": self.data_collector.get_status() if self.data_collector else {},
            "monitoring_status": self.monitoring_system.get_status() if self.monitoring_system else {}
        }

# Global orchestrator instance
orchestrator = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    if orchestrator:
        asyncio.create_task(orchestrator.stop_all_components())

async def main():
    """Main function"""
    global orchestrator
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Avalanche Network Real-Time Metrics Server")
    parser.add_argument("--mode", choices=["full", "collector", "api", "monitoring"], 
                       default="full", help="Server mode to run")
    parser.add_argument("--port", type=int, default=settings.API_PORT, 
                       help="API server port")
    parser.add_argument("--host", default=settings.API_HOST, 
                       help="API server host")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/avalanche_server_{time}.log",
        rotation="1 day",
        retention="30 days",
        level=args.log_level
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize orchestrator
    orchestrator = AvalancheServerOrchestrator()
    
    try:
        if args.mode == "full":
            # Run all components
            await orchestrator.start_all_components()
        
        elif args.mode == "collector":
            # Run only data collector
            await orchestrator.start_data_collector()
            while orchestrator.running:
                await asyncio.sleep(1)
        
        elif args.mode == "api":
            # Run only API server
            await orchestrator.start_api_server()
            while orchestrator.running:
                await asyncio.sleep(1)
        
        elif args.mode == "monitoring":
            # Run only monitoring system
            await orchestrator.start_data_collector()
            await orchestrator.start_monitoring_system()
            while orchestrator.running:
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await orchestrator.stop_all_components()
        logger.info("🏁 Avalanche Server shutdown complete")

if __name__ == "__main__":
    # Check if running in production
    if os.getenv("ENVIRONMENT") == "production":
        # Production configuration
        logger.info("Running in production mode")
    
    # Run the server
    asyncio.run(main())
