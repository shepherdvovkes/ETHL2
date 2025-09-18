#!/usr/bin/env python3
"""
Bitcoin Archive Node Monitor
Ultrafast monitoring system for Bitcoin Core archive node
"""

import json
import time
import logging
import requests
import psutil
import docker
from datetime import datetime
from pathlib import Path
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/bitcoin_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BitcoinNodeMonitor:
    def __init__(self):
        self.rpc_url = "http://bitcoin:ultrafast_archive_node_2024@bitcoin-core:8332"
        self.docker_client = docker.from_env()
        self.monitoring_data = {}
        
    def get_rpc_data(self, method, params=None):
        """Get data from Bitcoin RPC"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or []
            }
            response = requests.post(self.rpc_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('result')
        except Exception as e:
            logger.error(f"RPC call failed for {method}: {e}")
            return None
    
    def get_blockchain_info(self):
        """Get blockchain information"""
        return self.get_rpc_data("getblockchaininfo")
    
    def get_network_info(self):
        """Get network information"""
        return self.get_rpc_data("getnetworkinfo")
    
    def get_mempool_info(self):
        """Get mempool information"""
        return self.get_rpc_data("getmempoolinfo")
    
    def get_peer_info(self):
        """Get peer information"""
        return self.get_rpc_data("getpeerinfo")
    
    def get_system_stats(self):
        """Get system resource statistics"""
        try:
            # Get container stats
            container = self.docker_client.containers.get('bitcoin-archive-node')
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(memory_usage / 1024 / 1024, 2),
                'memory_limit_mb': round(memory_limit / 1024 / 1024, 2),
                'memory_percent': round(memory_percent, 2),
                'network_rx_bytes': stats['networks']['bitcoin-network']['rx_bytes'],
                'network_tx_bytes': stats['networks']['bitcoin-network']['tx_bytes']
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return None
    
    def get_disk_usage(self):
        """Get disk usage for Bitcoin data directory"""
        try:
            data_path = Path('/home/vovkes/ETHL2/bitcoin/data')
            if data_path.exists():
                total, used, free = psutil.disk_usage(str(data_path))
                return {
                    'total_gb': round(total / 1024 / 1024 / 1024, 2),
                    'used_gb': round(used / 1024 / 1024 / 1024, 2),
                    'free_gb': round(free / 1024 / 1024 / 1024, 2),
                    'usage_percent': round((used / total) * 100, 2)
                }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
        return None
    
    def check_sync_status(self):
        """Check if node is fully synced"""
        try:
            blockchain_info = self.get_blockchain_info()
            if blockchain_info:
                return {
                    'synced': not blockchain_info.get('initialblockdownload', True),
                    'blocks': blockchain_info.get('blocks', 0),
                    'headers': blockchain_info.get('headers', 0),
                    'verification_progress': blockchain_info.get('verificationprogress', 0),
                    'chain': blockchain_info.get('chain', 'unknown')
                }
        except Exception as e:
            logger.error(f"Failed to check sync status: {e}")
        return None
    
    def collect_metrics(self):
        """Collect all monitoring metrics"""
        logger.info("Collecting Bitcoin node metrics...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'blockchain': self.get_blockchain_info(),
            'network': self.get_network_info(),
            'mempool': self.get_mempool_info(),
            'peers': self.get_peer_info(),
            'system': self.get_system_stats(),
            'disk': self.get_disk_usage(),
            'sync_status': self.check_sync_status()
        }
        
        # Save metrics to file
        metrics_file = Path('/app/logs/bitcoin_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log key metrics
        if metrics['blockchain']:
            logger.info(f"Block height: {metrics['blockchain'].get('blocks', 'N/A')}")
            logger.info(f"Sync progress: {metrics['sync_status'].get('verification_progress', 0):.2%}")
        
        if metrics['system']:
            logger.info(f"CPU: {metrics['system']['cpu_percent']}%")
            logger.info(f"Memory: {metrics['system']['memory_percent']}%")
        
        if metrics['network']:
            logger.info(f"Connections: {metrics['network'].get('connections', 'N/A')}")
        
        return metrics
    
    def run_monitoring_loop(self):
        """Run continuous monitoring"""
        logger.info("Starting Bitcoin node monitoring...")
        
        while True:
            try:
                self.collect_metrics()
                time.sleep(60)  # Collect metrics every minute
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    monitor = BitcoinNodeMonitor()
    monitor.run_monitoring_loop()

