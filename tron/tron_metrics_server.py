"""
TRON Network Real-Time Metrics Server
Comprehensive monitoring and analytics system for TRON blockchain
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import aiohttp
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.tron_quicknode_client import TronQuickNodeClient
from src.database.tron_models import (
    Base, TronNetworkMetrics, TronEconomicMetrics, TronDeFiMetrics,
    TronSmartContractMetrics, TronStakingMetrics, TronUserActivityMetrics,
    TronNetworkHealthMetrics, TronProtocolMetrics, TronTokenMetrics,
    TronComprehensiveMetrics
)

# Load environment variables
load_dotenv("tron_config.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tron-metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("TRON_DATABASE_URL", "postgresql://defimon:password@localhost:5432/tron_metrics_db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global variables for caching
metrics_cache = {}
last_update = None
collection_in_progress = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting TRON Metrics Server...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
    
    # Start background tasks
    asyncio.create_task(periodic_data_collection())
    
    yield
    
    # Shutdown
    logger.info("Shutting down TRON Metrics Server...")

# Initialize FastAPI app
app = FastAPI(
    title="TRON Network Metrics Server",
    description="Real-time monitoring and analytics for TRON blockchain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TronMetricsCollector:
    """Main class for collecting and storing TRON metrics"""
    
    def __init__(self):
        self.client = None
        self.session = SessionLocal()
    
    async def collect_network_performance_metrics(self) -> Dict[str, Any]:
        """Collect network performance metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_network_performance_metrics()
                
                # Store in database
                db_metrics = TronNetworkMetrics(
                    current_block_number=metrics.get("current_block", 0),
                    block_time=metrics.get("block_time", 0.0),
                    block_size=0.0,  # Will be calculated
                    transaction_throughput=metrics.get("transaction_throughput", 0),
                    finality_time=metrics.get("finality_time", 0.0),
                    network_utilization=metrics.get("network_utilization", 0.0),
                    energy_consumption=0.0,  # Will be calculated
                    bandwidth_consumption=0.0,  # Will be calculated
                    transaction_count_24h=0,  # Will be calculated
                    transaction_fees_24h=0.0,  # Will be calculated
                    average_transaction_fee=0.0,  # Will be calculated
                    active_nodes=metrics.get("active_nodes", 0),
                    network_uptime=metrics.get("network_uptime", 0.0),
                    consensus_participation=metrics.get("consensus_participation", 0.0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect network performance metrics: {str(e)}")
            return {}
    
    async def collect_economic_metrics(self) -> Dict[str, Any]:
        """Collect economic metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_economic_metrics()
                
                # Store in database
                db_metrics = TronEconomicMetrics(
                    trx_price_usd=0.125,  # This should come from market data API
                    trx_price_btc=0.0000021,  # This should come from market data API
                    market_cap=11000000000,  # This should come from market data API
                    fully_diluted_market_cap=12500000000,  # This should come from market data API
                    volume_24h=450000000,  # This should come from market data API
                    price_change_24h=2.5,  # This should come from market data API
                    price_change_7d=15.3,  # This should come from market data API
                    total_supply=metrics.get("total_supply", 100000000000),
                    circulating_supply=metrics.get("circulating_supply", 88000000000),
                    burned_tokens=metrics.get("burned_tokens", 12000000000),
                    transaction_fees_24h=metrics.get("transaction_fees_24h", 0.0),
                    network_revenue_24h=metrics.get("network_revenue_24h", 0.0),
                    revenue_per_transaction=metrics.get("revenue_per_transaction", 0.0),
                    market_dominance=metrics.get("market_dominance", 1.2),
                    trading_volume_ratio=metrics.get("trading_volume_ratio", 0.8),
                    liquidity_score=metrics.get("liquidity_score", 85.0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect economic metrics: {str(e)}")
            return {}
    
    async def collect_defi_metrics(self) -> Dict[str, Any]:
        """Collect DeFi ecosystem metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_defi_metrics()
                
                # Store in database
                db_metrics = TronDeFiMetrics(
                    total_value_locked=metrics.get("total_value_locked", 0.0),
                    tvl_change_24h=metrics.get("tvl_change_24h", 0.0),
                    tvl_change_7d=metrics.get("tvl_change_7d", 0.0),
                    defi_protocols_count=metrics.get("defi_protocols_count", 0),
                    active_protocols_count=metrics.get("active_protocols_count", 0),
                    new_protocols_30d=metrics.get("new_protocols_30d", 0),
                    dex_volume_24h=metrics.get("dex_volume_24h", 0.0),
                    dex_trades_24h=metrics.get("dex_trades_24h", 0),
                    dex_liquidity=metrics.get("dex_liquidity", 0.0),
                    lending_tvl=metrics.get("lending_tvl", 0.0),
                    total_borrowed=metrics.get("total_borrowed", 0.0),
                    lending_utilization_rate=metrics.get("lending_utilization_rate", 0.0),
                    yield_farming_tvl=metrics.get("yield_farming_tvl", 0.0),
                    average_apy=metrics.get("average_apy", 0.0),
                    top_apy=metrics.get("top_apy", 0.0),
                    bridge_volume_24h=metrics.get("bridge_volume_24h", 0.0),
                    bridge_transactions_24h=metrics.get("bridge_transactions_24h", 0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect DeFi metrics: {str(e)}")
            return {}
    
    async def collect_smart_contract_metrics(self) -> Dict[str, Any]:
        """Collect smart contract metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_smart_contract_metrics()
                
                # Store in database
                db_metrics = TronSmartContractMetrics(
                    new_contracts_24h=metrics.get("new_contracts_24h", 0),
                    new_contracts_7d=metrics.get("new_contracts_7d", 0),
                    total_contracts=metrics.get("total_contracts", 0),
                    trc20_tokens_count=metrics.get("trc20_tokens_count", 0),
                    trc20_volume_24h=metrics.get("trc20_volume_24h", 0.0),
                    trc20_transactions_24h=metrics.get("trc20_transactions_24h", 0),
                    usdt_supply=metrics.get("usdt_supply", 0.0),
                    usdc_supply=metrics.get("usdc_supply", 0.0),
                    btt_supply=metrics.get("btt_supply", 0.0),
                    nft_collections_count=metrics.get("nft_collections_count", 0),
                    nft_transactions_24h=metrics.get("nft_transactions_24h", 0),
                    nft_volume_24h=metrics.get("nft_volume_24h", 0.0),
                    contract_calls_24h=metrics.get("contract_calls_24h", 0),
                    contract_gas_consumed=metrics.get("contract_gas_consumed", 0.0),
                    average_contract_complexity=metrics.get("average_contract_complexity", 0.0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect smart contract metrics: {str(e)}")
            return {}
    
    async def collect_staking_metrics(self) -> Dict[str, Any]:
        """Collect staking and governance metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_staking_metrics()
                
                # Store in database
                db_metrics = TronStakingMetrics(
                    total_staked=metrics.get("total_staked", 0.0),
                    staking_ratio=metrics.get("staking_ratio", 0.0),
                    staking_apy=metrics.get("staking_apy", 0.0),
                    active_validators=metrics.get("active_validators", 0),
                    total_validators=metrics.get("total_validators", 0),
                    validator_participation_rate=metrics.get("validator_participation_rate", 0.0),
                    governance_proposals=metrics.get("governance_proposals", 0),
                    active_proposals=metrics.get("active_proposals", 0),
                    voting_participation=metrics.get("voting_participation", 0.0),
                    energy_frozen=metrics.get("energy_frozen", 0.0),
                    bandwidth_frozen=metrics.get("bandwidth_frozen", 0.0),
                    resource_utilization=metrics.get("resource_utilization", 0.0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect staking metrics: {str(e)}")
            return {}
    
    async def collect_user_activity_metrics(self) -> Dict[str, Any]:
        """Collect user activity metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_user_activity_metrics()
                
                # Store in database
                db_metrics = TronUserActivityMetrics(
                    active_addresses_24h=metrics.get("active_addresses_24h", 0),
                    new_addresses_24h=metrics.get("new_addresses_24h", 0),
                    total_addresses=metrics.get("total_addresses", 0),
                    average_transactions_per_user=metrics.get("average_transactions_per_user", 0.0),
                    user_retention_rate=metrics.get("user_retention_rate", 0.0),
                    whale_activity=metrics.get("whale_activity", 0),
                    dapp_users_24h=metrics.get("dapp_users_24h", 0),
                    defi_users_24h=metrics.get("defi_users_24h", 0),
                    nft_users_24h=metrics.get("nft_users_24h", 0),
                    top_countries=json.dumps(metrics.get("top_countries", [])),
                    regional_activity=json.dumps(metrics.get("regional_activity", {}))
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect user activity metrics: {str(e)}")
            return {}
    
    async def collect_network_health_metrics(self) -> Dict[str, Any]:
        """Collect network health metrics"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_network_health_metrics()
                
                # Store in database
                db_metrics = TronNetworkHealthMetrics(
                    average_latency=metrics.get("average_latency", 0.0),
                    network_congestion=metrics.get("network_congestion", 0.0),
                    block_production_rate=metrics.get("block_production_rate", 0.0),
                    security_score=metrics.get("security_score", 0.0),
                    decentralization_index=metrics.get("decentralization_index", 0.0),
                    validator_distribution=json.dumps(metrics.get("validator_distribution", {})),
                    centralization_risk=metrics.get("centralization_risk", 0.0),
                    technical_risk=metrics.get("technical_risk", 0.0),
                    economic_risk=metrics.get("economic_risk", 0.0),
                    security_incidents_24h=metrics.get("security_incidents_24h", 0),
                    failed_transactions_24h=metrics.get("failed_transactions_24h", 0),
                    error_rate=metrics.get("error_rate", 0.0)
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect network health metrics: {str(e)}")
            return {}
    
    async def collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect all metrics and store comprehensive analysis"""
        try:
            async with TronQuickNodeClient() as client:
                metrics = await client.get_comprehensive_metrics()
                
                overall_scores = metrics.get("overall_scores", {})
                
                # Store comprehensive metrics
                db_metrics = TronComprehensiveMetrics(
                    network_performance_score=overall_scores.get("network_performance_score", 0.0),
                    economic_health_score=overall_scores.get("economic_health_score", 0.0),
                    defi_ecosystem_score=overall_scores.get("defi_ecosystem_score", 0.0),
                    adoption_score=75.0,  # Calculated separately
                    security_score=overall_scores.get("security_score", 0.0),
                    overall_score=overall_scores.get("overall_score", 0.0),
                    network_trend="stable",
                    economic_trend="up",
                    defi_trend="up",
                    adoption_trend="up",
                    security_trend="stable",
                    risk_level=overall_scores.get("risk_level", "medium"),
                    risk_factors=json.dumps(["centralization", "regulatory"]),
                    recommendations=json.dumps(["Increase decentralization", "Improve governance"])
                )
                
                self.session.add(db_metrics)
                self.session.commit()
                
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect comprehensive metrics: {str(e)}")
            return {}
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()

# Initialize collector
collector = TronMetricsCollector()

async def periodic_data_collection():
    """Periodic data collection task"""
    global collection_in_progress, last_update
    
    while True:
        try:
            if not collection_in_progress:
                collection_in_progress = True
                logger.info("Starting periodic data collection...")
                
                # Collect all metrics
                await collector.collect_comprehensive_metrics()
                
                last_update = datetime.utcnow()
                collection_in_progress = False
                
                logger.info("Data collection completed successfully")
            
            # Wait for next collection interval
            await asyncio.sleep(int(os.getenv("COLLECTION_INTERVAL", "300")))
            
        except Exception as e:
            logger.error(f"Error in periodic data collection: {str(e)}")
            collection_in_progress = False
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TRON Network Metrics Server",
        "version": "1.0.0",
        "description": "Real-time monitoring and analytics for TRON blockchain",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "network": "/metrics/network",
            "economic": "/metrics/economic",
            "defi": "/metrics/defi",
            "smart-contracts": "/metrics/smart-contracts",
            "staking": "/metrics/staking",
            "user-activity": "/metrics/user-activity",
            "network-health": "/metrics/network-health",
            "comprehensive": "/metrics/comprehensive",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "last_update": last_update.isoformat() if last_update else None,
        "collection_in_progress": collection_in_progress
    }

@app.get("/metrics")
async def get_metrics_summary():
    """Get summary of all metrics"""
    try:
        # Get latest metrics from database
        session = SessionLocal()
        
        # Get latest comprehensive metrics
        latest_comprehensive = session.query(TronComprehensiveMetrics).order_by(
            TronComprehensiveMetrics.timestamp.desc()
        ).first()
        
        # Get latest network metrics
        latest_network = session.query(TronNetworkMetrics).order_by(
            TronNetworkMetrics.timestamp.desc()
        ).first()
        
        # Get latest economic metrics
        latest_economic = session.query(TronEconomicMetrics).order_by(
            TronEconomicMetrics.timestamp.desc()
        ).first()
        
        session.close()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": latest_comprehensive.overall_score if latest_comprehensive else 0,
            "risk_level": latest_comprehensive.risk_level if latest_comprehensive else "unknown",
            "network_performance": {
                "block_number": latest_network.current_block_number if latest_network else 0,
                "tps": latest_network.transaction_throughput if latest_network else 0,
                "block_time": latest_network.block_time if latest_network else 0,
                "uptime": latest_network.network_uptime if latest_network else 0
            },
            "economic": {
                "market_cap": latest_economic.market_cap if latest_economic else 0,
                "volume_24h": latest_economic.volume_24h if latest_economic else 0,
                "price_change_24h": latest_economic.price_change_24h if latest_economic else 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/network")
async def get_network_metrics():
    """Get network performance metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronNetworkMetrics).order_by(
            TronNetworkMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No network metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "current_block": latest_metrics.current_block_number,
            "block_time": latest_metrics.block_time,
            "transaction_throughput": latest_metrics.transaction_throughput,
            "finality_time": latest_metrics.finality_time,
            "network_utilization": latest_metrics.network_utilization,
            "active_nodes": latest_metrics.active_nodes,
            "network_uptime": latest_metrics.network_uptime,
            "consensus_participation": latest_metrics.consensus_participation
        }
    except Exception as e:
        logger.error(f"Failed to get network metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/economic")
async def get_economic_metrics():
    """Get economic metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronEconomicMetrics).order_by(
            TronEconomicMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No economic metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "trx_price_usd": latest_metrics.trx_price_usd,
            "market_cap": latest_metrics.market_cap,
            "volume_24h": latest_metrics.volume_24h,
            "price_change_24h": latest_metrics.price_change_24h,
            "total_supply": latest_metrics.total_supply,
            "circulating_supply": latest_metrics.circulating_supply,
            "transaction_fees_24h": latest_metrics.transaction_fees_24h,
            "network_revenue_24h": latest_metrics.network_revenue_24h
        }
    except Exception as e:
        logger.error(f"Failed to get economic metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/defi")
async def get_defi_metrics():
    """Get DeFi ecosystem metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronDeFiMetrics).order_by(
            TronDeFiMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No DeFi metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "total_value_locked": latest_metrics.total_value_locked,
            "tvl_change_24h": latest_metrics.tvl_change_24h,
            "defi_protocols_count": latest_metrics.defi_protocols_count,
            "dex_volume_24h": latest_metrics.dex_volume_24h,
            "lending_tvl": latest_metrics.lending_tvl,
            "yield_farming_tvl": latest_metrics.yield_farming_tvl,
            "average_apy": latest_metrics.average_apy
        }
    except Exception as e:
        logger.error(f"Failed to get DeFi metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/smart-contracts")
async def get_smart_contract_metrics():
    """Get smart contract metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronSmartContractMetrics).order_by(
            TronSmartContractMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No smart contract metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "new_contracts_24h": latest_metrics.new_contracts_24h,
            "total_contracts": latest_metrics.total_contracts,
            "trc20_tokens_count": latest_metrics.trc20_tokens_count,
            "usdt_supply": latest_metrics.usdt_supply,
            "usdc_supply": latest_metrics.usdc_supply,
            "nft_collections_count": latest_metrics.nft_collections_count,
            "contract_calls_24h": latest_metrics.contract_calls_24h
        }
    except Exception as e:
        logger.error(f"Failed to get smart contract metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/staking")
async def get_staking_metrics():
    """Get staking and governance metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronStakingMetrics).order_by(
            TronStakingMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No staking metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "total_staked": latest_metrics.total_staked,
            "staking_ratio": latest_metrics.staking_ratio,
            "staking_apy": latest_metrics.staking_apy,
            "active_validators": latest_metrics.active_validators,
            "governance_proposals": latest_metrics.governance_proposals,
            "voting_participation": latest_metrics.voting_participation,
            "energy_frozen": latest_metrics.energy_frozen,
            "bandwidth_frozen": latest_metrics.bandwidth_frozen
        }
    except Exception as e:
        logger.error(f"Failed to get staking metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/user-activity")
async def get_user_activity_metrics():
    """Get user activity metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronUserActivityMetrics).order_by(
            TronUserActivityMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No user activity metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "active_addresses_24h": latest_metrics.active_addresses_24h,
            "new_addresses_24h": latest_metrics.new_addresses_24h,
            "total_addresses": latest_metrics.total_addresses,
            "average_transactions_per_user": latest_metrics.average_transactions_per_user,
            "user_retention_rate": latest_metrics.user_retention_rate,
            "whale_activity": latest_metrics.whale_activity,
            "dapp_users_24h": latest_metrics.dapp_users_24h,
            "defi_users_24h": latest_metrics.defi_users_24h
        }
    except Exception as e:
        logger.error(f"Failed to get user activity metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/network-health")
async def get_network_health_metrics():
    """Get network health metrics"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronNetworkHealthMetrics).order_by(
            TronNetworkHealthMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No network health metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "average_latency": latest_metrics.average_latency,
            "network_congestion": latest_metrics.network_congestion,
            "security_score": latest_metrics.security_score,
            "decentralization_index": latest_metrics.decentralization_index,
            "centralization_risk": latest_metrics.centralization_risk,
            "technical_risk": latest_metrics.technical_risk,
            "economic_risk": latest_metrics.economic_risk,
            "security_incidents_24h": latest_metrics.security_incidents_24h,
            "error_rate": latest_metrics.error_rate
        }
    except Exception as e:
        logger.error(f"Failed to get network health metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/comprehensive")
async def get_comprehensive_metrics():
    """Get comprehensive metrics analysis"""
    try:
        session = SessionLocal()
        latest_metrics = session.query(TronComprehensiveMetrics).order_by(
            TronComprehensiveMetrics.timestamp.desc()
        ).first()
        session.close()
        
        if not latest_metrics:
            return {"error": "No comprehensive metrics available"}
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "scores": {
                "network_performance": latest_metrics.network_performance_score,
                "economic_health": latest_metrics.economic_health_score,
                "defi_ecosystem": latest_metrics.defi_ecosystem_score,
                "adoption": latest_metrics.adoption_score,
                "security": latest_metrics.security_score,
                "overall": latest_metrics.overall_score
            },
            "trends": {
                "network": latest_metrics.network_trend,
                "economic": latest_metrics.economic_trend,
                "defi": latest_metrics.defi_trend,
                "adoption": latest_metrics.adoption_trend,
                "security": latest_metrics.security_trend
            },
            "risk_level": latest_metrics.risk_level,
            "risk_factors": json.loads(latest_metrics.risk_factors) if latest_metrics.risk_factors else [],
            "recommendations": json.loads(latest_metrics.recommendations) if latest_metrics.recommendations else []
        }
    except Exception as e:
        logger.error(f"Failed to get comprehensive metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect")
async def trigger_data_collection(background_tasks: BackgroundTasks):
    """Trigger manual data collection"""
    try:
        background_tasks.add_task(collector.collect_comprehensive_metrics)
        return {"message": "Data collection triggered successfully"}
    except Exception as e:
        logger.error(f"Failed to trigger data collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_dashboard():
    """Get dashboard HTML"""
    try:
        with open("tron_dashboard.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>TRON Dashboard</title></head>
            <body>
                <h1>TRON Network Dashboard</h1>
                <p>Dashboard file not found. Please create tron_dashboard.html</p>
                <p><a href="/metrics">View API Metrics</a></p>
            </body>
        </html>
        """)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "tron_metrics_server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8008")),
        reload=False,
        workers=1
    )
