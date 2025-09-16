#!/usr/bin/env python3
"""
LINEA Metrics Server
Serves LINEA blockchain data and metrics via REST API
"""

import asyncio
import json
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import aiohttp
from web3 import Web3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class BlockData(BaseModel):
    block_number: int
    block_hash: str
    parent_hash: str
    timestamp: datetime
    gas_limit: int
    gas_used: int
    base_fee_per_gas: int
    transaction_count: int
    size: int

class TransactionData(BaseModel):
    transaction_hash: str
    block_number: int
    from_address: str
    to_address: str
    value: str
    gas: int
    gas_price: int
    nonce: int
    type: int
    status: int

class NetworkMetrics(BaseModel):
    timestamp: datetime
    block_number: int
    tps: float
    block_time_avg: float
    gas_utilization: float
    gas_price_avg: float
    transaction_count: int
    unique_addresses_count: int
    total_gas_used: int
    total_fees: int

class AccountData(BaseModel):
    address: str
    balance: str
    nonce: int
    is_contract: bool
    transaction_count: int

class TokenData(BaseModel):
    address: str
    name: str
    symbol: str
    decimals: int
    total_supply: str
    token_type: str

class DeFiProtocolData(BaseModel):
    protocol_name: str
    protocol_address: str
    protocol_type: str
    tvl_usd: float
    block_number: int
    timestamp: datetime

class LineaMetricsServer:
    """LINEA metrics server with FastAPI"""
    
    def __init__(self, config_file: str = "linea_config.env"):
        self.config = self.load_config(config_file)
        self.db_path = self.config.get('LINEA_DATABASE_PATH', 'linea_data.db')
        self.archive_db_path = self.config.get('LINEA_ARCHIVE_DATABASE_PATH', 'linea_archive_data.db')
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="LINEA Blockchain Metrics API",
            description="Comprehensive LINEA blockchain data and metrics API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Web3 connection
        self.rpc_url = self.config.get('LINEA_RPC_URL')
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Setup routes
        self.setup_routes()
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
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
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Root endpoint with API information"""
            return """
            <html>
                <head>
                    <title>LINEA Blockchain Metrics API</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .header { color: #2c3e50; }
                        .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                        .method { color: #28a745; font-weight: bold; }
                    </style>
                </head>
                <body>
                    <h1 class="header">🚀 LINEA Blockchain Metrics API</h1>
                    <p>Comprehensive LINEA blockchain data and metrics API</p>
                    
                    <h2>Available Endpoints:</h2>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/health - API health check
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/network/status - Network status
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/blocks/latest - Latest block information
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/blocks/{block_number} - Get specific block
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/blocks - List blocks with pagination
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/transactions/{tx_hash} - Get transaction details
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/transactions - List transactions with filters
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/accounts/{address} - Get account information
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/contracts - List contracts
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/tokens - List tokens
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/defi/protocols - List DeFi protocols
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/metrics/network - Network metrics
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/metrics/summary - Metrics summary
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /api/stats - Collection statistics
                    </div>
                    
                    <h2>Interactive Documentation:</h2>
                    <p><a href="/docs">Swagger UI Documentation</a></p>
                    <p><a href="/redoc">ReDoc Documentation</a></p>
                </body>
            </html>
            """
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check database connection
                conn = self.get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM linea_blocks")
                block_count = cursor.fetchone()[0]
                conn.close()
                
                # Check Web3 connection
                latest_block = self.w3.eth.block_number
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "database": "connected",
                    "web3": "connected",
                    "latest_block": latest_block,
                    "total_blocks": block_count
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
        
        @self.app.get("/api/network/status")
        async def network_status():
            """Get network status"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Get latest block info
                cursor.execute("""
                    SELECT block_number, block_hash, timestamp, transaction_count, gas_used, gas_limit
                    FROM linea_blocks 
                    ORDER BY block_number DESC 
                    LIMIT 1
                """)
                latest_block = cursor.fetchone()
                
                # Get network metrics
                cursor.execute("""
                    SELECT tps, gas_utilization, gas_price_avg, unique_addresses_count
                    FROM linea_network_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                metrics = cursor.fetchone()
                
                conn.close()
                
                return {
                    "network": "LINEA",
                    "chain_id": 59144,
                    "latest_block": {
                        "number": latest_block[0] if latest_block else 0,
                        "hash": latest_block[1] if latest_block else "",
                        "timestamp": latest_block[2].isoformat() if latest_block and latest_block[2] else None,
                        "transaction_count": latest_block[3] if latest_block else 0,
                        "gas_used": latest_block[4] if latest_block else 0,
                        "gas_limit": latest_block[5] if latest_block else 0
                    },
                    "metrics": {
                        "tps": metrics[0] if metrics else 0,
                        "gas_utilization": metrics[1] if metrics else 0,
                        "gas_price_avg": metrics[2] if metrics else 0,
                        "unique_addresses": metrics[3] if metrics else 0
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Network status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/blocks/latest")
        async def get_latest_block():
            """Get latest block information"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT block_number, block_hash, parent_hash, timestamp, gas_limit, 
                           gas_used, base_fee_per_gas, transaction_count, size
                    FROM linea_blocks 
                    ORDER BY block_number DESC 
                    LIMIT 1
                """)
                block = cursor.fetchone()
                conn.close()
                
                if not block:
                    raise HTTPException(status_code=404, detail="No blocks found")
                
                return BlockData(
                    block_number=block[0],
                    block_hash=block[1],
                    parent_hash=block[2],
                    timestamp=block[3],
                    gas_limit=block[4],
                    gas_used=block[5],
                    base_fee_per_gas=block[6],
                    transaction_count=block[7],
                    size=block[8]
                )
            except Exception as e:
                logger.error(f"Latest block error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/blocks/{block_number}")
        async def get_block(block_number: int):
            """Get specific block by number"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT block_number, block_hash, parent_hash, timestamp, gas_limit, 
                           gas_used, base_fee_per_gas, transaction_count, size
                    FROM linea_blocks 
                    WHERE block_number = ?
                """, (block_number,))
                block = cursor.fetchone()
                conn.close()
                
                if not block:
                    raise HTTPException(status_code=404, detail="Block not found")
                
                return BlockData(
                    block_number=block[0],
                    block_hash=block[1],
                    parent_hash=block[2],
                    timestamp=block[3],
                    gas_limit=block[4],
                    gas_used=block[5],
                    base_fee_per_gas=block[6],
                    transaction_count=block[7],
                    size=block[8]
                )
            except Exception as e:
                logger.error(f"Get block error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/blocks")
        async def list_blocks(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0),
            order: str = Query("desc", regex="^(asc|desc)$")
        ):
            """List blocks with pagination"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                order_clause = "DESC" if order == "desc" else "ASC"
                
                cursor.execute(f"""
                    SELECT block_number, block_hash, parent_hash, timestamp, gas_limit, 
                           gas_used, base_fee_per_gas, transaction_count, size
                    FROM linea_blocks 
                    ORDER BY block_number {order_clause}
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                blocks = cursor.fetchall()
                conn.close()
                
                return [
                    BlockData(
                        block_number=block[0],
                        block_hash=block[1],
                        parent_hash=block[2],
                        timestamp=block[3],
                        gas_limit=block[4],
                        gas_used=block[5],
                        base_fee_per_gas=block[6],
                        transaction_count=block[7],
                        size=block[8]
                    ) for block in blocks
                ]
            except Exception as e:
                logger.error(f"List blocks error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/transactions/{tx_hash}")
        async def get_transaction(tx_hash: str):
            """Get transaction by hash"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT t.transaction_hash, t.block_number, t.from_address, t.to_address, 
                           t.value, t.gas, t.gas_price, t.nonce, t.type,
                           r.status
                    FROM linea_transactions t
                    LEFT JOIN linea_transaction_receipts r ON t.transaction_hash = r.transaction_hash
                    WHERE t.transaction_hash = ?
                """, (tx_hash,))
                tx = cursor.fetchone()
                conn.close()
                
                if not tx:
                    raise HTTPException(status_code=404, detail="Transaction not found")
                
                return TransactionData(
                    transaction_hash=tx[0],
                    block_number=tx[1],
                    from_address=tx[2],
                    to_address=tx[3],
                    value=tx[4],
                    gas=tx[5],
                    gas_price=tx[6],
                    nonce=tx[7],
                    type=tx[8],
                    status=tx[9] if tx[9] is not None else 1
                )
            except Exception as e:
                logger.error(f"Get transaction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/transactions")
        async def list_transactions(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0),
            block_number: Optional[int] = Query(None),
            address: Optional[str] = Query(None)
        ):
            """List transactions with filters"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                where_clauses = []
                params = []
                
                if block_number:
                    where_clauses.append("t.block_number = ?")
                    params.append(block_number)
                
                if address:
                    where_clauses.append("(t.from_address = ? OR t.to_address = ?)")
                    params.extend([address, address])
                
                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                params.extend([limit, offset])
                
                cursor.execute(f"""
                    SELECT t.transaction_hash, t.block_number, t.from_address, t.to_address, 
                           t.value, t.gas, t.gas_price, t.nonce, t.type,
                           r.status
                    FROM linea_transactions t
                    LEFT JOIN linea_transaction_receipts r ON t.transaction_hash = r.transaction_hash
                    {where_clause}
                    ORDER BY t.block_number DESC, t.transaction_index ASC
                    LIMIT ? OFFSET ?
                """, params)
                transactions = cursor.fetchall()
                conn.close()
                
                return [
                    TransactionData(
                        transaction_hash=tx[0],
                        block_number=tx[1],
                        from_address=tx[2],
                        to_address=tx[3],
                        value=tx[4],
                        gas=tx[5],
                        gas_price=tx[6],
                        nonce=tx[7],
                        type=tx[8],
                        status=tx[9] if tx[9] is not None else 1
                    ) for tx in transactions
                ]
            except Exception as e:
                logger.error(f"List transactions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/accounts/{address}")
        async def get_account(address: str):
            """Get account information"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT address, balance, nonce, is_contract, transaction_count
                    FROM linea_accounts 
                    WHERE address = ?
                """, (address,))
                account = cursor.fetchone()
                conn.close()
                
                if not account:
                    raise HTTPException(status_code=404, detail="Account not found")
                
                return AccountData(
                    address=account[0],
                    balance=account[1],
                    nonce=account[2],
                    is_contract=bool(account[3]),
                    transaction_count=account[4]
                )
            except Exception as e:
                logger.error(f"Get account error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/contracts")
        async def list_contracts(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0),
            contract_type: Optional[str] = Query(None)
        ):
            """List contracts"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                where_clause = ""
                params = []
                
                if contract_type:
                    where_clause = "WHERE contract_type = ?"
                    params.append(contract_type)
                
                params.extend([limit, offset])
                
                cursor.execute(f"""
                    SELECT address, contract_name, contract_type, is_verified, creation_block
                    FROM linea_contracts 
                    {where_clause}
                    ORDER BY creation_block DESC
                    LIMIT ? OFFSET ?
                """, params)
                contracts = cursor.fetchall()
                conn.close()
                
                return [
                    {
                        "address": contract[0],
                        "name": contract[1],
                        "type": contract[2],
                        "verified": bool(contract[3]),
                        "creation_block": contract[4]
                    } for contract in contracts
                ]
            except Exception as e:
                logger.error(f"List contracts error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/tokens")
        async def list_tokens(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0),
            token_type: Optional[str] = Query(None)
        ):
            """List tokens"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                where_clause = ""
                params = []
                
                if token_type:
                    where_clause = "WHERE token_type = ?"
                    params.append(token_type)
                
                params.extend([limit, offset])
                
                cursor.execute(f"""
                    SELECT address, name, symbol, decimals, total_supply, token_type, is_native
                    FROM linea_tokens 
                    {where_clause}
                    ORDER BY first_seen_block DESC
                    LIMIT ? OFFSET ?
                """, params)
                tokens = cursor.fetchall()
                conn.close()
                
                return [
                    TokenData(
                        address=token[0],
                        name=token[1],
                        symbol=token[2],
                        decimals=token[3],
                        total_supply=token[4],
                        token_type=token[5]
                    ) for token in tokens
                ]
            except Exception as e:
                logger.error(f"List tokens error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/defi/protocols")
        async def list_defi_protocols(
            limit: int = Query(10, ge=1, le=100),
            offset: int = Query(0, ge=0)
        ):
            """List DeFi protocols"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT protocol_name, protocol_address, protocol_type, tvl_usd, 
                           block_number, timestamp
                    FROM linea_defi_protocols 
                    ORDER BY tvl_usd DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                protocols = cursor.fetchall()
                conn.close()
                
                return [
                    DeFiProtocolData(
                        protocol_name=protocol[0],
                        protocol_address=protocol[1],
                        protocol_type=protocol[2],
                        tvl_usd=protocol[3],
                        block_number=protocol[4],
                        timestamp=protocol[5]
                    ) for protocol in protocols
                ]
            except Exception as e:
                logger.error(f"List DeFi protocols error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/network")
        async def get_network_metrics(
            hours: int = Query(24, ge=1, le=168)
        ):
            """Get network metrics for specified time period"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                since = datetime.now() - timedelta(hours=hours)
                
                cursor.execute("""
                    SELECT timestamp, block_number, tps, block_time_avg, gas_utilization, 
                           gas_price_avg, transaction_count, unique_addresses_count, 
                           total_gas_used, total_fees
                    FROM linea_network_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """, (since,))
                metrics = cursor.fetchall()
                conn.close()
                
                return [
                    NetworkMetrics(
                        timestamp=metric[0],
                        block_number=metric[1],
                        tps=metric[2],
                        block_time_avg=metric[3],
                        gas_utilization=metric[4],
                        gas_price_avg=metric[5],
                        transaction_count=metric[6],
                        unique_addresses_count=metric[7],
                        total_gas_used=metric[8],
                        total_fees=metric[9]
                    ) for metric in metrics
                ]
            except Exception as e:
                logger.error(f"Network metrics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/summary")
        async def get_metrics_summary():
            """Get metrics summary"""
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Get counts
                cursor.execute("SELECT COUNT(*) FROM linea_blocks")
                total_blocks = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_transactions")
                total_transactions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_accounts")
                total_accounts = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_contracts")
                total_contracts = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_tokens")
                total_tokens = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_defi_protocols")
                total_defi_protocols = cursor.fetchone()[0]
                
                # Get latest metrics
                cursor.execute("""
                    SELECT tps, gas_utilization, gas_price_avg, unique_addresses_count
                    FROM linea_network_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                latest_metrics = cursor.fetchone()
                
                # Get 24h stats
                yesterday = datetime.now() - timedelta(days=1)
                cursor.execute("""
                    SELECT COUNT(*), SUM(transaction_count), AVG(tps), AVG(gas_utilization)
                    FROM linea_network_metrics 
                    WHERE timestamp >= ?
                """, (yesterday,))
                stats_24h = cursor.fetchone()
                
                conn.close()
                
                return {
                    "summary": {
                        "total_blocks": total_blocks,
                        "total_transactions": total_transactions,
                        "total_accounts": total_accounts,
                        "total_contracts": total_contracts,
                        "total_tokens": total_tokens,
                        "total_defi_protocols": total_defi_protocols
                    },
                    "current_metrics": {
                        "tps": latest_metrics[0] if latest_metrics else 0,
                        "gas_utilization": latest_metrics[1] if latest_metrics else 0,
                        "gas_price_avg": latest_metrics[2] if latest_metrics else 0,
                        "unique_addresses": latest_metrics[3] if latest_metrics else 0
                    },
                    "stats_24h": {
                        "data_points": stats_24h[0] if stats_24h else 0,
                        "total_transactions": stats_24h[1] if stats_24h else 0,
                        "avg_tps": stats_24h[2] if stats_24h else 0,
                        "avg_gas_utilization": stats_24h[3] if stats_24h else 0
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Metrics summary error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/stats")
        async def get_collection_stats():
            """Get data collection statistics"""
            try:
                # This would typically come from the collector
                # For now, return basic database stats
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM linea_blocks")
                blocks = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_transactions")
                transactions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM linea_accounts")
                accounts = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    "collection_stats": {
                        "blocks_collected": blocks,
                        "transactions_collected": transactions,
                        "accounts_collected": accounts,
                        "last_updated": datetime.now().isoformat()
                    },
                    "database_size": {
                        "realtime_db": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
                        "archive_db": os.path.getsize(self.archive_db_path) if os.path.exists(self.archive_db_path) else 0
                    }
                }
            except Exception as e:
                logger.error(f"Collection stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def create_app():
    """Create FastAPI app instance"""
    server = LineaMetricsServer()
    return server.app

app = create_app()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LINEA Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting LINEA Metrics Server on {args.host}:{args.port}")
    
    uvicorn.run(
        "linea_metrics_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
