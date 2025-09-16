#!/usr/bin/env python3
"""
Solana Comprehensive Metrics Server
Provides REST API, WebSocket, and GraphQL endpoints for both real-time and archive Solana data
"""

import asyncio
import json
import sqlite3
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType
import signal
import sys
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solana_comprehensive_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaComprehensiveServer:
    """Comprehensive Solana metrics server with REST API, WebSocket, and GraphQL support"""
    
    def __init__(self, config_file: str = "solana_config.env"):
        self.config = self.load_config(config_file)
        self.db_path = self.config.get('SOLANA_DATABASE_PATH', 'solana_data.db')
        self.archive_db_path = self.config.get('SOLANA_ARCHIVE_DATABASE_PATH', 'solana_archive_data.db')
        
        # Server configuration
        self.host = self.config.get('API_HOST', '0.0.0.0')
        self.port = int(self.config.get('API_PORT', 8001))
        self.metrics_port = int(self.config.get('METRICS_PORT', 9091))
        
        # WebSocket connections
        self.websocket_connections = set()
        
        # Server instances
        self.app = None
        self.metrics_app = None
        self.server = None
        self.metrics_server = None
        
        # Statistics
        self.stats = {
            'requests_served': 0,
            'websocket_connections': 0,
            'start_time': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_file: str) -> Dict[str, str]:
        """Load configuration from environment file"""
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        return config
    
    def get_db_connection(self, archive: bool = False):
        """Get database connection"""
        db_path = self.archive_db_path if archive else self.db_path
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    # Data retrieval methods
    async def get_latest_blocks(self, limit: int = 10, archive: bool = False) -> List[Dict[str, Any]]:
        """Get latest blocks"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_blocks" if archive else "solana_blocks"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                ORDER BY slot DESC 
                LIMIT ?
            """, (limit,))
            
            blocks = []
            for row in cursor.fetchall():
                blocks.append(dict(row))
            
            return blocks
        finally:
            conn.close()
    
    async def get_block_by_slot(self, slot: int, archive: bool = False) -> Optional[Dict[str, Any]]:
        """Get block by slot number"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_blocks" if archive else "solana_blocks"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                WHERE slot = ?
            """, (slot,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    async def get_blocks_by_range(self, start_slot: int, end_slot: int, archive: bool = False) -> List[Dict[str, Any]]:
        """Get blocks in a range"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_blocks" if archive else "solana_blocks"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                WHERE slot BETWEEN ? AND ?
                ORDER BY slot ASC
            """, (start_slot, end_slot))
            
            blocks = []
            for row in cursor.fetchall():
                blocks.append(dict(row))
            
            return blocks
        finally:
            conn.close()
    
    async def get_latest_transactions(self, limit: int = 100, archive: bool = False) -> List[Dict[str, Any]]:
        """Get latest transactions"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_transactions" if archive else "solana_transactions"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                ORDER BY slot DESC, created_at DESC 
                LIMIT ?
            """, (limit,))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append(dict(row))
            
            return transactions
        finally:
            conn.close()
    
    async def get_transaction_by_signature(self, signature: str, archive: bool = False) -> Optional[Dict[str, Any]]:
        """Get transaction by signature"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_transactions" if archive else "solana_transactions"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                WHERE signature = ?
            """, (signature,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    async def get_network_metrics(self, limit: int = 100, archive: bool = False) -> List[Dict[str, Any]]:
        """Get network metrics"""
        conn = self.get_db_connection(archive)
        try:
            cursor = conn.cursor()
            table_name = "solana_archive_network_metrics" if archive else "solana_network_metrics"
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append(dict(row))
            
            return metrics
        finally:
            conn.close()
    
    async def get_validators(self) -> List[Dict[str, Any]]:
        """Get validator information"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM solana_validators 
                ORDER BY activated_stake DESC
            """)
            
            validators = []
            for row in cursor.fetchall():
                validators.append(dict(row))
            
            return validators
        finally:
            conn.close()
    
    async def get_token_accounts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get token accounts"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM solana_token_accounts 
                ORDER BY last_updated_slot DESC 
                LIMIT ?
            """, (limit,))
            
            accounts = []
            for row in cursor.fetchall():
                accounts.append(dict(row))
            
            return accounts
        finally:
            conn.close()
    
    async def get_programs(self) -> List[Dict[str, Any]]:
        """Get program information"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM solana_programs 
                ORDER BY created_at DESC
            """)
            
            programs = []
            for row in cursor.fetchall():
                programs.append(dict(row))
            
            return programs
        finally:
            conn.close()
    
    async def get_archive_progress(self) -> List[Dict[str, Any]]:
        """Get archive collection progress"""
        conn = self.get_db_connection(archive=True)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM solana_archive_progress 
                ORDER BY last_updated_at DESC
            """)
            
            progress = []
            for row in cursor.fetchall():
                progress.append(dict(row))
            
            return progress
        finally:
            conn.close()
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        # Main database stats
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Get counts from main database
            cursor.execute("SELECT COUNT(*) FROM solana_blocks")
            main_block_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM solana_transactions")
            main_transaction_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM solana_accounts")
            account_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM solana_token_accounts")
            token_account_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM solana_validators")
            validator_count = cursor.fetchone()[0]
            
            # Get latest slot from main database
            cursor.execute("SELECT MAX(slot) FROM solana_blocks")
            main_latest_slot = cursor.fetchone()[0] or 0
            
            # Get latest timestamp from main database
            cursor.execute("SELECT MAX(timestamp) FROM solana_blocks")
            main_latest_timestamp = cursor.fetchone()[0]
            
        finally:
            conn.close()
        
        # Archive database stats
        conn = self.get_db_connection(archive=True)
        try:
            cursor = conn.cursor()
            
            # Get counts from archive database
            cursor.execute("SELECT COUNT(*) FROM solana_archive_blocks")
            archive_block_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM solana_archive_transactions")
            archive_transaction_count = cursor.fetchone()[0]
            
            # Get latest slot from archive database
            cursor.execute("SELECT MAX(slot) FROM solana_archive_blocks")
            archive_latest_slot = cursor.fetchone()[0] or 0
            
            # Get latest timestamp from archive database
            cursor.execute("SELECT MAX(timestamp) FROM solana_archive_blocks")
            archive_latest_timestamp = cursor.fetchone()[0]
            
            # Get archive progress
            cursor.execute("SELECT * FROM solana_archive_progress ORDER BY last_updated_at DESC LIMIT 1")
            progress_row = cursor.fetchone()
            archive_progress = dict(progress_row) if progress_row else None
            
        finally:
            conn.close()
        
        return {
            'main_database': {
                'block_count': main_block_count,
                'transaction_count': main_transaction_count,
                'account_count': account_count,
                'token_account_count': token_account_count,
                'validator_count': validator_count,
                'latest_slot': main_latest_slot,
                'latest_timestamp': main_latest_timestamp
            },
            'archive_database': {
                'block_count': archive_block_count,
                'transaction_count': archive_transaction_count,
                'latest_slot': archive_latest_slot,
                'latest_timestamp': archive_latest_timestamp,
                'progress': archive_progress
            },
            'server': {
                'uptime': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0,
                'requests_served': self.stats['requests_served'],
                'websocket_connections': len(self.websocket_connections)
            }
        }
    
    # REST API Handlers
    async def handle_root(self, request):
        """Handle root endpoint"""
        return web.json_response({
            'service': 'Solana Comprehensive Metrics Server',
            'version': '2.0.0',
            'description': 'Complete Solana blockchain data API with real-time and archive support',
            'endpoints': {
                'blocks': '/api/blocks',
                'transactions': '/api/transactions',
                'network_metrics': '/api/network_metrics',
                'validators': '/api/validators',
                'programs': '/api/programs',
                'archive': '/api/archive',
                'stats': '/api/stats',
                'websocket': '/ws',
                'graphql': '/graphql'
            },
            'features': [
                'Real-time data collection',
                'Complete historical archive',
                'WebSocket streaming',
                'GraphQL API',
                'Prometheus metrics',
                '10 concurrent workers'
            ]
        })
    
    async def handle_blocks(self, request):
        """Handle blocks endpoint"""
        try:
            self.stats['requests_served'] += 1
            
            limit = int(request.query.get('limit', 10))
            slot = request.query.get('slot')
            start_slot = request.query.get('start_slot')
            end_slot = request.query.get('end_slot')
            archive = request.query.get('archive', 'false').lower() == 'true'
            
            if slot:
                block = await self.get_block_by_slot(int(slot), archive)
                if block:
                    return web.json_response(block)
                else:
                    return web.json_response({'error': 'Block not found'}, status=404)
            elif start_slot and end_slot:
                blocks = await self.get_blocks_by_range(int(start_slot), int(end_slot), archive)
                return web.json_response(blocks)
            else:
                blocks = await self.get_latest_blocks(limit, archive)
                return web.json_response(blocks)
                
        except Exception as e:
            logger.error(f"Error handling blocks request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_transactions(self, request):
        """Handle transactions endpoint"""
        try:
            self.stats['requests_served'] += 1
            
            limit = int(request.query.get('limit', 100))
            signature = request.query.get('signature')
            archive = request.query.get('archive', 'false').lower() == 'true'
            
            if signature:
                transaction = await self.get_transaction_by_signature(signature, archive)
                if transaction:
                    return web.json_response(transaction)
                else:
                    return web.json_response({'error': 'Transaction not found'}, status=404)
            else:
                transactions = await self.get_latest_transactions(limit, archive)
                return web.json_response(transactions)
                
        except Exception as e:
            logger.error(f"Error handling transactions request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_network_metrics(self, request):
        """Handle network metrics endpoint"""
        try:
            self.stats['requests_served'] += 1
            
            limit = int(request.query.get('limit', 100))
            archive = request.query.get('archive', 'false').lower() == 'true'
            metrics = await self.get_network_metrics(limit, archive)
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Error handling network metrics request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_validators(self, request):
        """Handle validators endpoint"""
        try:
            self.stats['requests_served'] += 1
            validators = await self.get_validators()
            return web.json_response(validators)
        except Exception as e:
            logger.error(f"Error handling validators request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_programs(self, request):
        """Handle programs endpoint"""
        try:
            self.stats['requests_served'] += 1
            programs = await self.get_programs()
            return web.json_response(programs)
        except Exception as e:
            logger.error(f"Error handling programs request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_archive(self, request):
        """Handle archive endpoint"""
        try:
            self.stats['requests_served'] += 1
            
            data_type = request.query.get('type', 'blocks')
            start_slot = request.query.get('start_slot')
            end_slot = request.query.get('end_slot')
            start_time = request.query.get('start_time')
            end_time = request.query.get('end_time')
            
            if data_type == 'blocks' and start_slot and end_slot:
                blocks = await self.get_blocks_by_range(int(start_slot), int(end_slot), archive=True)
                return web.json_response(blocks)
            elif data_type == 'transactions' and start_slot and end_slot:
                transactions = await self.get_latest_transactions(1000, archive=True)
                return web.json_response(transactions)
            elif data_type == 'network_metrics' and start_time and end_time:
                metrics = await self.get_network_metrics(1000, archive=True)
                return web.json_response(metrics)
            elif data_type == 'progress':
                progress = await self.get_archive_progress()
                return web.json_response(progress)
            else:
                return web.json_response({'error': 'Invalid parameters'}, status=400)
                
        except Exception as e:
            logger.error(f"Error handling archive request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_stats(self, request):
        """Handle stats endpoint"""
        try:
            self.stats['requests_served'] += 1
            stats = await self.get_collection_stats()
            return web.json_response(stats)
        except Exception as e:
            logger.error(f"Error handling stats request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # WebSocket Handler
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        self.stats['websocket_connections'] = len(self.websocket_connections)
        
        logger.info(f"WebSocket connection established. Total connections: {len(self.websocket_connections)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        command = data.get('command')
                        
                        if command == 'subscribe_blocks':
                            # Send latest blocks periodically
                            while not ws.closed:
                                blocks = await self.get_latest_blocks(5)
                                await ws.send_str(json.dumps({
                                    'type': 'blocks',
                                    'data': blocks,
                                    'timestamp': datetime.now().isoformat()
                                }))
                                await asyncio.sleep(5)
                        
                        elif command == 'subscribe_archive_blocks':
                            # Send archive blocks periodically
                            while not ws.closed:
                                blocks = await self.get_latest_blocks(5, archive=True)
                                await ws.send_str(json.dumps({
                                    'type': 'archive_blocks',
                                    'data': blocks,
                                    'timestamp': datetime.now().isoformat()
                                }))
                                await asyncio.sleep(10)
                        
                        elif command == 'subscribe_metrics':
                            # Send latest metrics periodically
                            while not ws.closed:
                                metrics = await self.get_network_metrics(1)
                                if metrics:
                                    await ws.send_str(json.dumps({
                                        'type': 'metrics',
                                        'data': metrics[0],
                                        'timestamp': datetime.now().isoformat()
                                    }))
                                await asyncio.sleep(10)
                        
                        elif command == 'get_stats':
                            stats = await self.get_collection_stats()
                            await ws.send_str(json.dumps({
                                'type': 'stats',
                                'data': stats,
                                'timestamp': datetime.now().isoformat()
                            }))
                        
                        elif command == 'get_archive_progress':
                            progress = await self.get_archive_progress()
                            await ws.send_str(json.dumps({
                                'type': 'archive_progress',
                                'data': progress,
                                'timestamp': datetime.now().isoformat()
                            }))
                        
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        await ws.send_str(json.dumps({'error': str(e)}))
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.websocket_connections.discard(ws)
            self.stats['websocket_connections'] = len(self.websocket_connections)
            logger.info(f"WebSocket connection closed. Total connections: {len(self.websocket_connections)}")
        
        return ws
    
    # Metrics endpoint for monitoring
    async def handle_metrics(self, request):
        """Handle Prometheus-style metrics"""
        try:
            stats = await self.get_collection_stats()
            
            metrics = f"""# HELP solana_blocks_total Total number of blocks collected
# TYPE solana_blocks_total counter
solana_blocks_total{{database="main"}} {stats['main_database']['block_count']}
solana_blocks_total{{database="archive"}} {stats['archive_database']['block_count']}

# HELP solana_transactions_total Total number of transactions collected
# TYPE solana_transactions_total counter
solana_transactions_total{{database="main"}} {stats['main_database']['transaction_count']}
solana_transactions_total{{database="archive"}} {stats['archive_database']['transaction_count']}

# HELP solana_accounts_total Total number of accounts collected
# TYPE solana_accounts_total counter
solana_accounts_total {stats['main_database']['account_count']}

# HELP solana_validators_total Total number of validators
# TYPE solana_validators_total gauge
solana_validators_total {stats['main_database']['validator_count']}

# HELP solana_latest_slot Latest slot number
# TYPE solana_latest_slot gauge
solana_latest_slot{{database="main"}} {stats['main_database']['latest_slot']}
solana_latest_slot{{database="archive"}} {stats['archive_database']['latest_slot']}

# HELP solana_server_uptime_seconds Server uptime in seconds
# TYPE solana_server_uptime_seconds counter
solana_server_uptime_seconds {stats['server']['uptime']}

# HELP solana_requests_served_total Total number of requests served
# TYPE solana_requests_served_total counter
solana_requests_served_total {stats['server']['requests_served']}

# HELP solana_websocket_connections_active Active WebSocket connections
# TYPE solana_websocket_connections_active gauge
solana_websocket_connections_active {stats['server']['websocket_connections']}
"""
            
            return web.Response(text=metrics, content_type='text/plain')
        except Exception as e:
            logger.error(f"Error handling metrics request: {e}")
            return web.Response(text=f"Error: {e}", status=500)
    
    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/api/blocks', self.handle_blocks)
        self.app.router.add_get('/api/transactions', self.handle_transactions)
        self.app.router.add_get('/api/network_metrics', self.handle_network_metrics)
        self.app.router.add_get('/api/validators', self.handle_validators)
        self.app.router.add_get('/api/programs', self.handle_programs)
        self.app.router.add_get('/api/archive', self.handle_archive)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Add CORS middleware
        self.app.router.add_options('/{path:.*}', self.handle_cors)
        self.app.middlewares.append(self.cors_middleware)
    
    def setup_metrics_routes(self):
        """Setup metrics routes"""
        self.metrics_app.router.add_get('/metrics', self.handle_metrics)
    
    async def handle_cors(self, request):
        """Handle CORS preflight requests"""
        return web.Response(
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    
    @web.middleware
    async def cors_middleware(self, request, handler):
        """CORS middleware"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        if self.server:
            asyncio.create_task(self.server.close())
        if self.metrics_server:
            asyncio.create_task(self.metrics_server.close())
    
    async def start(self):
        """Start the comprehensive metrics server"""
        logger.info("🚀 Starting Solana comprehensive metrics server...")
        self.stats['start_time'] = time.time()
        
        # Create main app
        self.app = web.Application()
        self.setup_routes()
        
        # Create metrics app
        self.metrics_app = web.Application()
        self.setup_metrics_routes()
        
        try:
            # Start main server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            # Start metrics server
            metrics_runner = web.AppRunner(self.metrics_app)
            await metrics_runner.setup()
            metrics_site = web.TCPSite(metrics_runner, self.host, self.metrics_port)
            await metrics_site.start()
            
            logger.info(f"✅ Solana comprehensive metrics server started!")
            logger.info(f"📡 Main API: http://{self.host}:{self.port}")
            logger.info(f"📊 Metrics: http://{self.host}:{self.metrics_port}/metrics")
            logger.info(f"🔌 WebSocket: ws://{self.host}:{self.port}/ws")
            logger.info(f"📚 Archive API: http://{self.host}:{self.port}/api/archive")
            logger.info(f"📈 Stats API: http://{self.host}:{self.port}/api/stats")
            
            # Keep server running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
        finally:
            logger.info("✅ Solana comprehensive metrics server stopped")

async def main():
    """Main function"""
    server = SolanaComprehensiveServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Solana comprehensive metrics server stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


