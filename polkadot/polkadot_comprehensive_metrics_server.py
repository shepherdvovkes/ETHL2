#!/usr/bin/env python3
"""
Comprehensive Polkadot Metrics API Server
Provides comprehensive metrics for Polkadot network and all parachains
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.database import SessionLocal, engine
from database.polkadot_comprehensive_models import (
    PolkadotNetwork, Parachain, PolkadotNetworkMetrics,
    PolkadotStakingMetrics, PolkadotGovernanceMetrics,
    PolkadotEconomicMetrics, ParachainMetrics,
    ParachainCrossChainMetrics, PolkadotEcosystemMetrics,
    PolkadotPerformanceMetrics, PolkadotSecurityMetrics,
    PolkadotDeveloperMetrics, ParachainDeFiMetrics,
    ParachainPerformanceMetrics, ParachainSecurityMetrics,
    ParachainDeveloperMetrics, TokenMarketData, ValidatorInfo
)
from api.polkadot_comprehensive_client import PolkadotComprehensiveClient
from config.settings import settings

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/polkadot-comprehensive-metrics.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Global variables
polkadot_client = None
data_collection_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global polkadot_client, data_collection_task
    
    logger.info("Starting Comprehensive Polkadot Metrics Server...")
    
    # Initialize database
    try:
        from database.polkadot_comprehensive_models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize Polkadot client
    try:
        polkadot_client = PolkadotComprehensiveClient()
        async with polkadot_client:
            health = await polkadot_client.health_check()
            logger.info(f"Polkadot client health: {health}")
    except Exception as e:
        logger.error(f"Polkadot client initialization failed: {e}")
        raise
    
    # Note: Data collection is handled by separate PM2 process
    logger.info("Data collection will be handled by separate PM2 process")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Comprehensive Polkadot Metrics Server...")

# Create FastAPI app
app = FastAPI(
    title="Comprehensive Polkadot Metrics API",
    description="Comprehensive API for collecting and serving Polkadot network and parachain metrics",
    version="2.0.0",
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

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Note: Background data collection is handled by separate PM2 process
# This function is kept for manual triggering via API

async def collect_and_store_metrics():
    """Collect and store all metrics"""
    global polkadot_client
    
    if not polkadot_client:
        logger.error("Polkadot client not initialized")
        return
    
    try:
        # Import the comprehensive data collector
        from collect_comprehensive_polkadot_data import ComprehensivePolkadotDataCollector
        
        collector = ComprehensivePolkadotDataCollector()
        await collector.initialize()
        await collector.collect_all_metrics()
        await collector.cleanup()
        
        logger.info("All metrics collected and stored successfully")
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")

# API Endpoints

@app.get("/")
async def root():
    """Serve the comprehensive Polkadot dashboard at root URL"""
    return FileResponse("polkadot_comprehensive_dashboard.html")

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Comprehensive Polkadot Metrics API Server",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "running",
        "dashboard": "http://localhost:8008/",
        "docs": "http://localhost:8008/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global polkadot_client
    
    try:
        if polkadot_client:
            async with polkadot_client:
                health = await polkadot_client.health_check()
                return {
                    "status": "healthy",
                    "polkadot_client": health,
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            return {
                "status": "unhealthy",
                "error": "Polkadot client not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Network Metrics Endpoints

@app.get("/api/network/overview")
async def get_network_overview():
    """Get comprehensive network overview"""
    try:
        db = SessionLocal()
        
        # Get latest network metrics
        latest_network_metrics = db.query(PolkadotNetworkMetrics).order_by(
            desc(PolkadotNetworkMetrics.timestamp)
        ).first()
        
        # Get latest staking metrics
        latest_staking_metrics = db.query(PolkadotStakingMetrics).order_by(
            desc(PolkadotStakingMetrics.timestamp)
        ).first()
        
        # Get latest economic metrics
        latest_economic_metrics = db.query(PolkadotEconomicMetrics).order_by(
            desc(PolkadotEconomicMetrics.timestamp)
        ).first()
        
        # Get latest ecosystem metrics
        latest_ecosystem_metrics = db.query(PolkadotEcosystemMetrics).order_by(
            desc(PolkadotEcosystemMetrics.timestamp)
        ).first()
        
        return {
            "network_metrics": {
                "current_block": latest_network_metrics.current_block if latest_network_metrics else None,
                "validator_count": latest_network_metrics.validator_count if latest_network_metrics else None,
                "peer_count": latest_network_metrics.peer_count if latest_network_metrics else None,
                "timestamp": latest_network_metrics.timestamp.isoformat() if latest_network_metrics else None
            },
            "staking_metrics": {
                "total_staked": float(latest_staking_metrics.total_staked) if latest_staking_metrics and latest_staking_metrics.total_staked else None,
                "validator_count": latest_staking_metrics.validator_count if latest_staking_metrics else None,
                "inflation_rate": latest_staking_metrics.inflation_rate if latest_staking_metrics else None,
                "timestamp": latest_staking_metrics.timestamp.isoformat() if latest_staking_metrics else None
            },
            "economic_metrics": {
                "treasury_balance": float(latest_economic_metrics.treasury_balance) if latest_economic_metrics and latest_economic_metrics.treasury_balance else None,
                "inflation_rate": latest_economic_metrics.inflation_rate if latest_economic_metrics else None,
                "timestamp": latest_economic_metrics.timestamp.isoformat() if latest_economic_metrics else None
            },
            "ecosystem_metrics": {
                "total_parachains": latest_ecosystem_metrics.total_parachains if latest_ecosystem_metrics else None,
                "active_parachains": latest_ecosystem_metrics.active_parachains if latest_ecosystem_metrics else None,
                "timestamp": latest_ecosystem_metrics.timestamp.isoformat() if latest_ecosystem_metrics else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting network overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/network/metrics")
async def get_network_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get network metrics history"""
    try:
        db = SessionLocal()
        
        metrics = db.query(PolkadotNetworkMetrics).order_by(
            desc(PolkadotNetworkMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "current_block": m.current_block,
                    "validator_count": m.validator_count,
                    "peer_count": m.peer_count,
                    "runtime_version": m.runtime_version,
                    "spec_version": m.spec_version,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting network metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/staking/metrics")
async def get_staking_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get staking metrics history"""
    try:
        db = SessionLocal()
        
        metrics = db.query(PolkadotStakingMetrics).order_by(
            desc(PolkadotStakingMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "total_staked": float(m.total_staked) if m.total_staked else None,
                    "validator_count": m.validator_count,
                    "waiting_validators": m.waiting_validators,
                    "nomination_pools_count": m.nomination_pools_count,
                    "inflation_rate": m.inflation_rate,
                    "active_era": m.active_era,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting staking metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/governance/metrics")
async def get_governance_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get governance metrics history"""
    try:
        db = SessionLocal()
        
        metrics = db.query(PolkadotGovernanceMetrics).order_by(
            desc(PolkadotGovernanceMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "active_proposals": m.active_proposals,
                    "referendum_count": m.referendum_count,
                    "council_members": m.council_members,
                    "treasury_proposals": m.treasury_proposals,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting governance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/economic/metrics")
async def get_economic_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get economic metrics history"""
    try:
        db = SessionLocal()
        
        metrics = db.query(PolkadotEconomicMetrics).order_by(
            desc(PolkadotEconomicMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "treasury_balance": float(m.treasury_balance) if m.treasury_balance else None,
                    "inflation_rate": m.inflation_rate,
                    "block_reward": float(m.block_reward) if m.block_reward else None,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting economic metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Parachain Endpoints

@app.get("/api/parachains")
async def get_parachains():
    """Get all parachains"""
    try:
        db = SessionLocal()
        
        parachains = db.query(Parachain).all()
        
        return {
            "parachains": [
                {
                    "id": p.id,
                    "parachain_id": p.parachain_id,
                    "name": p.name,
                    "symbol": p.symbol,
                    "category": p.category,
                    "status": p.status,
                    "created_at": p.created_at.isoformat()
                }
                for p in parachains
            ],
            "count": len(parachains),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting parachains: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/parachains/{parachain_id}/metrics")
async def get_parachain_metrics(parachain_id: int, limit: int = Query(100, ge=1, le=1000)):
    """Get metrics for a specific parachain"""
    try:
        db = SessionLocal()
        
        # Get parachain
        parachain = db.query(Parachain).filter(Parachain.parachain_id == parachain_id).first()
        if not parachain:
            raise HTTPException(status_code=404, detail=f"Parachain {parachain_id} not found")
        
        # Get parachain metrics
        metrics = db.query(ParachainMetrics).filter(
            ParachainMetrics.parachain_id == parachain.id
        ).order_by(desc(ParachainMetrics.timestamp)).limit(limit).all()
        
        # Get DeFi metrics if applicable
        defi_metrics = None
        if parachain.category == "defi":
            defi_metrics = db.query(ParachainDeFiMetrics).filter(
                ParachainDeFiMetrics.parachain_id == parachain.id
            ).order_by(desc(ParachainDeFiMetrics.timestamp)).limit(limit).all()
        
        return {
            "parachain": {
                "id": parachain.id,
                "parachain_id": parachain.parachain_id,
                "name": parachain.name,
                "symbol": parachain.symbol,
                "category": parachain.category,
                "status": parachain.status
            },
            "metrics": [
                {
                    "id": m.id,
                    "current_block": m.current_block,
                    "active_addresses_24h": m.active_addresses_24h,
                    "daily_transactions": m.daily_transactions,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "defi_metrics": [
                {
                    "id": m.id,
                    "total_tvl": float(m.total_tvl) if m.total_tvl else None,
                    "dex_volume_24h": float(m.dex_volume_24h) if m.dex_volume_24h else None,
                    "lending_tvl": float(m.lending_tvl) if m.lending_tvl else None,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in defi_metrics
            ] if defi_metrics else None,
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parachain metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/parachains/categories")
async def get_parachain_categories():
    """Get parachains grouped by category"""
    try:
        db = SessionLocal()
        
        # Get parachains grouped by category
        categories = db.query(
            Parachain.category,
            func.count(Parachain.id).label('count')
        ).group_by(Parachain.category).all()
        
        # Get detailed parachain info by category
        parachains_by_category = {}
        for category, count in categories:
            parachains = db.query(Parachain).filter(Parachain.category == category).all()
            parachains_by_category[category] = [
                {
                    "id": p.id,
                    "parachain_id": p.parachain_id,
                    "name": p.name,
                    "symbol": p.symbol,
                    "status": p.status
                }
                for p in parachains
            ]
        
        return {
            "categories": {category: count for category, count in categories},
            "parachains_by_category": parachains_by_category,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting parachain categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Cross-Chain Endpoints

@app.get("/api/cross-chain/metrics")
async def get_cross_chain_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get cross-chain metrics"""
    try:
        db = SessionLocal()
        
        metrics = db.query(ParachainCrossChainMetrics).order_by(
            desc(ParachainCrossChainMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "parachain_id": m.parachain_id,
                    "hrmp_channels_count": m.hrmp_channels_count,
                    "xcmp_channels_count": m.xcmp_channels_count,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cross-chain metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Token Market Data Endpoints

@app.get("/api/tokens/market-data")
async def get_token_market_data(limit: int = Query(100, ge=1, le=1000)):
    """Get token market data"""
    try:
        db = SessionLocal()
        
        market_data = db.query(TokenMarketData).order_by(
            desc(TokenMarketData.timestamp)
        ).limit(limit).all()
        
        return {
            "market_data": [
                {
                    "id": m.id,
                    "token_symbol": m.token_symbol,
                    "token_name": m.token_name,
                    "price_usd": float(m.price_usd) if m.price_usd else None,
                    "price_change_24h": m.price_change_24h,
                    "market_cap": float(m.market_cap) if m.market_cap else None,
                    "volume_24h": float(m.volume_24h) if m.volume_24h else None,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in market_data
            ],
            "count": len(market_data),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting token market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Validator Endpoints

@app.get("/api/validators")
async def get_validators():
    """Get validator information"""
    try:
        db = SessionLocal()
        
        validators = db.query(ValidatorInfo).order_by(
            desc(ValidatorInfo.timestamp)
        ).all()
        
        return {
            "validators": [
                {
                    "id": v.id,
                    "validator_address": v.validator_address,
                    "name": v.name,
                    "commission": v.commission,
                    "total_stake": float(v.total_stake) if v.total_stake else None,
                    "is_active": v.is_active,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in validators
            ],
            "count": len(validators),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting validators: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Ecosystem Endpoints

@app.get("/api/ecosystem/metrics")
async def get_ecosystem_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get ecosystem metrics"""
    try:
        db = SessionLocal()
        
        metrics = db.query(PolkadotEcosystemMetrics).order_by(
            desc(PolkadotEcosystemMetrics.timestamp)
        ).limit(limit).all()
        
        return {
            "metrics": [
                {
                    "id": m.id,
                    "total_parachains": m.total_parachains,
                    "active_parachains": m.active_parachains,
                    "active_cross_chain_channels": m.active_cross_chain_channels,
                    "total_active_developers": m.total_active_developers,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ecosystem metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Data Collection Endpoints

@app.post("/api/collect")
async def trigger_data_collection(background_tasks: BackgroundTasks):
    """Trigger manual data collection"""
    try:
        background_tasks.add_task(collect_and_store_metrics)
        return {
            "message": "Data collection triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Endpoint

@app.get("/dashboard")
async def dashboard():
    """Serve the comprehensive Polkadot dashboard"""
    return FileResponse("polkadot_comprehensive_dashboard.html")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8008))
    
    logger.info(f"Starting Comprehensive Polkadot Metrics Server on {host}:{port}")
    
    uvicorn.run(
        "polkadot_comprehensive_metrics_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
