#!/usr/bin/env python3
"""
Avalanche Network Metrics API Server
FastAPI server for accessing collected Avalanche network metrics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from avalanche_metrics_server import AvalancheMetricsCollector, AvalancheNetworkMetrics
from database.database import get_db_session
from database.models_v2 import Blockchain, NetworkMetrics, EconomicMetrics
from config.settings import settings
from api.github_client import GitHubClient

# Pydantic models for API responses
class NetworkPerformanceResponse(BaseModel):
    block_time: float
    transaction_throughput: int
    finality_time: float
    network_utilization: float
    gas_price_avg: float
    gas_price_median: float
    block_size_avg: float

class EconomicMetricsResponse(BaseModel):
    total_value_locked: float
    daily_volume: float
    active_addresses_24h: int
    new_addresses_24h: int
    transaction_fees_24h: float
    revenue_24h: float
    market_cap: float
    circulating_supply: float
    total_supply: float

class DeFiMetricsResponse(BaseModel):
    defi_protocols_count: int
    defi_tvl: float
    dex_volume_24h: float
    lending_volume_24h: float
    yield_farming_apy: float
    bridge_volume_24h: float

class SubnetMetricsResponse(BaseModel):
    subnet_count: int
    subnet_tvl: float
    subnet_activity: int
    custom_vm_usage: int

class SecurityMetricsResponse(BaseModel):
    validator_count: int
    staking_ratio: float
    validator_distribution: Dict[str, Any]
    slashing_events: int
    audit_count: int
    security_score: float
    audit_status: str
    bug_bounty_programs: int
    penetration_tests: int

class DevelopmentMetricsResponse(BaseModel):
    github_commits: int
    github_stars: int
    github_forks: int
    developer_count: int
    smart_contract_deployments: int
    subnet_launches: int

class UserBehaviorResponse(BaseModel):
    whale_activity: int
    retail_vs_institutional: Dict[str, float]
    holding_patterns: Dict[str, Any]
    transaction_sizes: Dict[str, float]
    address_concentration: float

class CompetitiveAnalysisResponse(BaseModel):
    market_share: float
    performance_vs_competitors: Dict[str, Any]
    ecosystem_growth: Dict[str, Any]
    developer_adoption: Dict[str, Any]

class TechnicalInfrastructureResponse(BaseModel):
    rpc_performance: Dict[str, float]
    node_distribution: Dict[str, Any]
    network_uptime: float
    upgrade_history: List[Dict[str, Any]]
    interoperability_score: float

class RiskAssessmentResponse(BaseModel):
    centralization_risks: Dict[str, Any]
    technical_risks: Dict[str, Any]
    regulatory_risks: Dict[str, Any]
    market_risks: Dict[str, Any]
    competition_risks: Dict[str, Any]

class MacroFactorsResponse(BaseModel):
    market_conditions: Dict[str, Any]
    institutional_adoption: Dict[str, Any]
    regulatory_environment: Dict[str, Any]
    economic_indicators: Dict[str, Any]

class EcosystemHealthResponse(BaseModel):
    community_growth: Dict[str, Any]
    media_coverage: Dict[str, Any]
    partnership_quality: Dict[str, Any]
    developer_experience: Dict[str, Any]

class AvalancheMetricsResponse(BaseModel):
    timestamp: datetime
    network_performance: NetworkPerformanceResponse
    economic_metrics: EconomicMetricsResponse
    defi_metrics: DeFiMetricsResponse
    subnet_metrics: SubnetMetricsResponse
    security_metrics: SecurityMetricsResponse
    development_metrics: DevelopmentMetricsResponse
    user_behavior: UserBehaviorResponse
    competitive_analysis: CompetitiveAnalysisResponse
    technical_infrastructure: TechnicalInfrastructureResponse
    risk_assessment: RiskAssessmentResponse
    macro_factors: MacroFactorsResponse
    ecosystem_health: EcosystemHealthResponse

class MetricsSummaryResponse(BaseModel):
    timestamp: datetime
    key_metrics: Dict[str, Any]
    status: str
    collection_time: float

# Global variables for caching
latest_metrics: Optional[AvalancheNetworkMetrics] = None
metrics_collection_time: Optional[datetime] = None
collection_in_progress = False

async def load_latest_metrics_from_database():
    """Load the latest metrics from database on startup"""
    global latest_metrics, metrics_collection_time
    
    try:
        logger.info("Loading latest metrics from database...")
        
        from sqlalchemy import desc
        from database.models_v2 import NetworkMetrics, EconomicMetrics, EcosystemMetrics, Blockchain
        
        # Use synchronous database session for now
        from database.database import SessionLocal
        db = SessionLocal()
        
        try:
            # Find Avalanche blockchain ID
            avalanche_blockchain = db.query(Blockchain).filter(Blockchain.name.ilike('%avalanche%')).first()
            
            if not avalanche_blockchain:
                logger.warning("Avalanche blockchain not found in database")
                return
            
            avalanche_id = avalanche_blockchain.id
            
            # Get latest network metrics
            latest_network = db.query(NetworkMetrics)\
                .filter(NetworkMetrics.blockchain_id == avalanche_id)\
                .order_by(desc(NetworkMetrics.timestamp))\
                .first()
            
            # Get latest economic metrics
            latest_economic = db.query(EconomicMetrics)\
                .filter(EconomicMetrics.blockchain_id == avalanche_id)\
                .order_by(desc(EconomicMetrics.timestamp))\
                .first()
            
            # Get latest ecosystem metrics
            latest_ecosystem = db.query(EcosystemMetrics)\
                .filter(EcosystemMetrics.blockchain_id == avalanche_id)\
                .order_by(desc(EcosystemMetrics.timestamp))\
                .first()
            
            if latest_network or latest_economic or latest_ecosystem:
                # Create AvalancheNetworkMetrics object from database data
                latest_metrics = AvalancheNetworkMetrics(
                    # Network Performance
                    block_time=latest_network.block_time_avg if latest_network else 2.0,
                    transaction_throughput=latest_network.transaction_throughput if latest_network else 0,
                    finality_time=2.0,  # Default for Avalanche
                    network_utilization=latest_network.network_utilization if latest_network and latest_network.network_utilization is not None else 0.0,
                    gas_price_avg=latest_network.gas_price_avg if latest_network and latest_network.gas_price_avg is not None else 0.0,
                    gas_price_median=latest_network.gas_price_avg if latest_network and latest_network.gas_price_avg is not None else 0.0,
                    block_size_avg=latest_network.block_size_avg if latest_network and latest_network.block_size_avg is not None else 0.0,
                    
                    # Economic Metrics
                    total_value_locked=float(latest_economic.total_value_locked) if latest_economic and latest_economic.total_value_locked else 0.0,
                    daily_volume=float(latest_economic.daily_volume) if latest_economic and latest_economic.daily_volume else 0.0,
                    active_addresses_24h=latest_economic.active_users_24h if latest_economic else 0,
                    new_addresses_24h=0,  # Not in current schema
                    transaction_fees_24h=float(latest_economic.transaction_fees_24h) if latest_economic and latest_economic.transaction_fees_24h else 0.0,
                    revenue_24h=0.0,  # Not in current schema
                    market_cap=float(latest_economic.market_cap) if latest_economic and latest_economic.market_cap else 0.0,
                    circulating_supply=0.0,  # Not in current schema
                    total_supply=0.0,  # Not in current schema
                    
                    # DeFi Metrics
                    defi_protocols_count=latest_ecosystem.defi_protocols_count if latest_ecosystem else 0,
                    defi_tvl=0.0,  # Not in current schema
                    dex_volume_24h=0.0,  # Not in current schema
                    lending_volume_24h=0.0,  # Not in current schema
                    yield_farming_apy=0.0,  # Not in current schema
                    bridge_volume_24h=0.0,  # Not in current schema
                    
                    # Subnet Metrics
                    subnet_count=0,  # Not in current schema
                    subnet_tvl=0.0,  # Not in current schema
                    subnet_activity=0.0,  # Not in current schema
                    custom_vm_usage=0,  # Not in current schema
                    
                    # Security Metrics
                    validator_count=latest_network.validator_count if latest_network else 0,
                    staking_ratio=latest_network.staking_ratio if latest_network else 0.0,
                    validator_distribution={},  # Not in current schema
                    slashing_events=0,  # Not in current schema
                    audit_count=0,  # Not in current schema
                    security_score=85.0,  # Default security score
                    
                    # Development Metrics (using fallback data)
                    github_commits=45,
                    github_stars=8500,
                    github_forks=1200,
                    developer_count=25,
                    smart_contract_deployments=12,
                    subnet_launches=2,
                    
                    # Other metrics (using defaults)
                    whale_activity=0,
                    retail_vs_institutional={},
                    holding_patterns={},
                    transaction_sizes={},
                    address_concentration=0.0,
                    market_share=0.0,
                    performance_vs_competitors={},
                    ecosystem_growth={},
                    developer_adoption={},
                    rpc_performance={},
                    node_distribution={},
                    network_uptime=0.0,
                    upgrade_history=[],
                    interoperability_score=0.0,
                    centralization_risks={},
                    technical_risks={},
                    regulatory_risks={},
                    market_risks={},
                    competition_risks={},
                    market_conditions={},
                    institutional_adoption={},
                    regulatory_environment={},
                    economic_indicators={},
                    community_growth={},
                    media_coverage={},
                    partnership_quality={},
                    developer_experience={},
                    
                    timestamp=datetime.utcnow()
                )
                
                metrics_collection_time = datetime.utcnow()
                logger.info("✅ Successfully loaded latest metrics from database")
            else:
                logger.warning("No metrics found in database")
                
        finally:
            db.close()
                
    except Exception as e:
        logger.error(f"Error loading metrics from database: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Avalanche Metrics API Server")
    
    # Load latest metrics from database
    await load_latest_metrics_from_database()
    
    # Start background metrics collection
    asyncio.create_task(periodic_metrics_collection())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Avalanche Metrics API Server")

# Initialize FastAPI app
app = FastAPI(
    title="Avalanche Network Metrics API",
    description="Comprehensive API for Avalanche network metrics and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """Serve the Avalanche dashboard"""
    return FileResponse("src/web/index.html")

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Avalanche Network Metrics API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/auth/github")
async def github_auth_start():
    """Start GitHub OAuth authentication"""
    try:
        github_client = GitHubClient()
        auth_url = github_client.get_authorization_url()
        
        return {
            "message": "Please visit the following URL to authorize GitHub access:",
            "auth_url": auth_url,
            "instructions": "After authorization, you'll be redirected back to this application"
        }
    except Exception as e:
        logger.error(f"Failed to start GitHub authorization: {e}")
        raise HTTPException(status_code=500, detail="Failed to start GitHub authorization")

@app.get("/auth/github/callback")
async def github_auth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """Handle GitHub OAuth callback"""
    try:
        if error:
            return {
                "error": f"GitHub authorization failed: {error}",
                "message": "Please try again"
            }
        
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code not provided")
        
        github_client = GitHubClient()
        success = await github_client.complete_authorization(code)
        
        if success:
            user_info = await github_client.get_user_info()
            return {
                "message": "GitHub authentication successful!",
                "user": user_info.get("login", "Unknown"),
                "status": "authenticated"
            }
        else:
            return {
                "error": "Failed to complete GitHub authorization",
                "message": "Please try again"
            }
            
    except Exception as e:
        logger.error(f"GitHub authorization callback failed: {e}")
        return {
            "error": f"Authorization failed: {str(e)}",
            "message": "Please try again"
        }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "latest_metrics_time": metrics_collection_time.isoformat() if metrics_collection_time else None,
        "collection_in_progress": collection_in_progress
    }

@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary():
    """Get summary of key Avalanche metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    key_metrics = {
        "transaction_throughput": latest_metrics.transaction_throughput,
        "market_cap": latest_metrics.market_cap,
        "daily_volume": latest_metrics.daily_volume,
        "validator_count": latest_metrics.validator_count,
        "defi_protocols_count": latest_metrics.defi_protocols_count,
        "subnet_count": latest_metrics.subnet_count,
        "active_addresses_24h": latest_metrics.active_addresses_24h,
        "gas_price_avg": latest_metrics.gas_price_avg,
        "block_time": latest_metrics.block_time,
        "finality_time": latest_metrics.finality_time,
        "staking_ratio": latest_metrics.staking_ratio,
        "defi_tvl": latest_metrics.defi_tvl
    }
    
    return MetricsSummaryResponse(
        timestamp=latest_metrics.timestamp,
        key_metrics=key_metrics,
        status="operational",
        collection_time=0.0  # Would be calculated from actual collection time
    )

@app.get("/metrics/network-performance", response_model=NetworkPerformanceResponse)
async def get_network_performance():
    """Get network performance metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return NetworkPerformanceResponse(
        block_time=latest_metrics.block_time,
        transaction_throughput=latest_metrics.transaction_throughput,
        finality_time=latest_metrics.finality_time,
        network_utilization=latest_metrics.network_utilization,
        gas_price_avg=latest_metrics.gas_price_avg,
        gas_price_median=latest_metrics.gas_price_median,
        block_size_avg=latest_metrics.block_size_avg
    )

@app.get("/metrics/economic", response_model=EconomicMetricsResponse)
async def get_economic_metrics():
    """Get economic metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return EconomicMetricsResponse(
        total_value_locked=latest_metrics.total_value_locked,
        daily_volume=latest_metrics.daily_volume,
        active_addresses_24h=latest_metrics.active_addresses_24h,
        new_addresses_24h=latest_metrics.new_addresses_24h,
        transaction_fees_24h=latest_metrics.transaction_fees_24h,
        revenue_24h=latest_metrics.revenue_24h,
        market_cap=latest_metrics.market_cap,
        circulating_supply=latest_metrics.circulating_supply,
        total_supply=latest_metrics.total_supply
    )

@app.get("/metrics/defi", response_model=DeFiMetricsResponse)
async def get_defi_metrics():
    """Get DeFi ecosystem metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return DeFiMetricsResponse(
        defi_protocols_count=latest_metrics.defi_protocols_count,
        defi_tvl=latest_metrics.defi_tvl,
        dex_volume_24h=latest_metrics.dex_volume_24h,
        lending_volume_24h=latest_metrics.lending_volume_24h,
        yield_farming_apy=latest_metrics.yield_farming_apy,
        bridge_volume_24h=latest_metrics.bridge_volume_24h
    )

@app.get("/metrics/subnets", response_model=SubnetMetricsResponse)
async def get_subnet_metrics():
    """Get subnet metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return SubnetMetricsResponse(
        subnet_count=latest_metrics.subnet_count,
        subnet_tvl=latest_metrics.subnet_tvl,
        subnet_activity=latest_metrics.subnet_activity,
        custom_vm_usage=latest_metrics.custom_vm_usage
    )

@app.get("/metrics/security", response_model=SecurityMetricsResponse)
async def get_security_metrics():
    """Get security metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return SecurityMetricsResponse(
        validator_count=latest_metrics.validator_count,
        staking_ratio=latest_metrics.staking_ratio,
        validator_distribution=latest_metrics.validator_distribution,
        slashing_events=latest_metrics.slashing_events,
        audit_count=latest_metrics.audit_count,
        security_score=latest_metrics.security_score,
        audit_status="audited" if latest_metrics.audit_count > 0 else "pending",
        bug_bounty_programs=1 if latest_metrics.audit_count > 0 else 0,
        penetration_tests=latest_metrics.audit_count
    )

@app.get("/metrics/development", response_model=DevelopmentMetricsResponse)
async def get_development_metrics():
    """Get development activity metrics"""
    if not latest_metrics:
        # Return fallback data if no metrics available
        return DevelopmentMetricsResponse(
            github_commits=45,
            github_stars=8500,
            github_forks=1200,
            developer_count=25,
            smart_contract_deployments=12,
            subnet_launches=2
        )
    
    return DevelopmentMetricsResponse(
        github_commits=latest_metrics.github_commits,
        github_stars=latest_metrics.github_stars,
        github_forks=latest_metrics.github_forks,
        developer_count=latest_metrics.developer_count,
        smart_contract_deployments=latest_metrics.smart_contract_deployments,
        subnet_launches=latest_metrics.subnet_launches
    )

@app.get("/metrics/user-behavior", response_model=UserBehaviorResponse)
async def get_user_behavior_metrics():
    """Get user behavior metrics"""
    if not latest_metrics:
        # Return fallback data if no metrics available
        return UserBehaviorResponse(
            whale_activity=0,
            retail_vs_institutional={"retail": 65.0, "institutional": 35.0},
            holding_patterns={"short_term": 40.0, "medium_term": 35.0, "long_term": 25.0},
            transaction_sizes={"average": 2.5, "median": 1.2, "total_analyzed": 0},
            address_concentration=0.15
        )
    
    # Ensure all fields have default values if they're None or undefined
    return UserBehaviorResponse(
        whale_activity=latest_metrics.whale_activity if hasattr(latest_metrics, 'whale_activity') and latest_metrics.whale_activity is not None else 0,
        retail_vs_institutional=latest_metrics.retail_vs_institutional if hasattr(latest_metrics, 'retail_vs_institutional') and latest_metrics.retail_vs_institutional else {"retail": 65.0, "institutional": 35.0},
        holding_patterns=latest_metrics.holding_patterns if hasattr(latest_metrics, 'holding_patterns') and latest_metrics.holding_patterns else {"short_term": 40.0, "medium_term": 35.0, "long_term": 25.0},
        transaction_sizes=latest_metrics.transaction_sizes if hasattr(latest_metrics, 'transaction_sizes') and latest_metrics.transaction_sizes else {"average": 2.5, "median": 1.2, "total_analyzed": 0},
        address_concentration=latest_metrics.address_concentration if hasattr(latest_metrics, 'address_concentration') and latest_metrics.address_concentration is not None else 0.15
    )

@app.get("/metrics/competitive", response_model=CompetitiveAnalysisResponse)
async def get_competitive_analysis():
    """Get competitive analysis metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    # Use real-time data if available, otherwise use fallback data
    market_share = latest_metrics.market_share if latest_metrics.market_share > 0 else 2.5
    
    performance_vs_competitors = latest_metrics.performance_vs_competitors if latest_metrics.performance_vs_competitors else {
        "ethereum": {"tps": 15, "fees": 20.0, "finality": 12.0},
        "solana": {"tps": 3000, "fees": 0.00025, "finality": 0.4},
        "polygon": {"tps": 7000, "fees": 0.01, "finality": 2.0},
        "avalanche": {"tps": 4500, "fees": 0.1, "finality": 1.0}
    }
    
    ecosystem_growth = latest_metrics.ecosystem_growth if latest_metrics.ecosystem_growth else {
        "dapp_count": 200,
        "developer_count": 500,
        "tvl_growth": 15.5
    }
    
    developer_adoption = latest_metrics.developer_adoption if latest_metrics.developer_adoption else {
        "active_developers": 500,
        "new_projects": 25,
        "github_activity": 85.0
    }
    
    return CompetitiveAnalysisResponse(
        market_share=market_share,
        performance_vs_competitors=performance_vs_competitors,
        ecosystem_growth=ecosystem_growth,
        developer_adoption=developer_adoption
    )

@app.get("/metrics/technical", response_model=TechnicalInfrastructureResponse)
async def get_technical_infrastructure():
    """Get technical infrastructure metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    # Use real-time data if available, otherwise use fallback data
    rpc_performance = latest_metrics.rpc_performance if latest_metrics.rpc_performance else {
        "response_time_ms": 150.0,
        "uptime": 99.9,
        "reliability": 99.8
    }
    
    node_distribution = latest_metrics.node_distribution if latest_metrics.node_distribution else {
        "geographic": {},
        "total_nodes": 1000
    }
    
    network_uptime = latest_metrics.network_uptime if latest_metrics.network_uptime > 0 else 99.9
    
    upgrade_history = latest_metrics.upgrade_history if latest_metrics.upgrade_history else [
        {"version": "v1.9.0", "date": "2023-01-01", "success": True},
        {"version": "v1.8.0", "date": "2022-10-01", "success": True}
    ]
    
    interoperability_score = latest_metrics.interoperability_score if latest_metrics.interoperability_score > 0 else 8.5
    
    return TechnicalInfrastructureResponse(
        rpc_performance=rpc_performance,
        node_distribution=node_distribution,
        network_uptime=network_uptime,
        upgrade_history=upgrade_history,
        interoperability_score=interoperability_score
    )

@app.get("/metrics/risk", response_model=RiskAssessmentResponse)
async def get_risk_assessment():
    """Get risk assessment metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return RiskAssessmentResponse(
        centralization_risks=latest_metrics.centralization_risks,
        technical_risks=latest_metrics.technical_risks,
        regulatory_risks=latest_metrics.regulatory_risks,
        market_risks=latest_metrics.market_risks,
        competition_risks=latest_metrics.competition_risks
    )

@app.get("/metrics/macro", response_model=MacroFactorsResponse)
async def get_macro_factors():
    """Get macro-economic factors"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return MacroFactorsResponse(
        market_conditions=latest_metrics.market_conditions,
        institutional_adoption=latest_metrics.institutional_adoption,
        regulatory_environment=latest_metrics.regulatory_environment,
        economic_indicators=latest_metrics.economic_indicators
    )

@app.get("/metrics/ecosystem", response_model=EcosystemHealthResponse)
async def get_ecosystem_health():
    """Get ecosystem health metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return EcosystemHealthResponse(
        community_growth=latest_metrics.community_growth,
        media_coverage=latest_metrics.media_coverage,
        partnership_quality=latest_metrics.partnership_quality,
        developer_experience=latest_metrics.developer_experience
    )

@app.get("/metrics/all", response_model=AvalancheMetricsResponse)
async def get_all_metrics():
    """Get all Avalanche metrics"""
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics data available")
    
    return AvalancheMetricsResponse(
        timestamp=latest_metrics.timestamp,
        network_performance=NetworkPerformanceResponse(
            block_time=latest_metrics.block_time,
            transaction_throughput=latest_metrics.transaction_throughput,
            finality_time=latest_metrics.finality_time,
            network_utilization=latest_metrics.network_utilization,
            gas_price_avg=latest_metrics.gas_price_avg,
            gas_price_median=latest_metrics.gas_price_median,
            block_size_avg=latest_metrics.block_size_avg
        ),
        economic_metrics=EconomicMetricsResponse(
            total_value_locked=latest_metrics.total_value_locked,
            daily_volume=latest_metrics.daily_volume,
            active_addresses_24h=latest_metrics.active_addresses_24h,
            new_addresses_24h=latest_metrics.new_addresses_24h,
            transaction_fees_24h=latest_metrics.transaction_fees_24h,
            revenue_24h=latest_metrics.revenue_24h,
            market_cap=latest_metrics.market_cap,
            circulating_supply=latest_metrics.circulating_supply,
            total_supply=latest_metrics.total_supply
        ),
        defi_metrics=DeFiMetricsResponse(
            defi_protocols_count=latest_metrics.defi_protocols_count,
            defi_tvl=latest_metrics.defi_tvl,
            dex_volume_24h=latest_metrics.dex_volume_24h,
            lending_volume_24h=latest_metrics.lending_volume_24h,
            yield_farming_apy=latest_metrics.yield_farming_apy,
            bridge_volume_24h=latest_metrics.bridge_volume_24h
        ),
        subnet_metrics=SubnetMetricsResponse(
            subnet_count=latest_metrics.subnet_count,
            subnet_tvl=latest_metrics.subnet_tvl,
            subnet_activity=latest_metrics.subnet_activity,
            custom_vm_usage=latest_metrics.custom_vm_usage
        ),
        security_metrics=SecurityMetricsResponse(
            validator_count=latest_metrics.validator_count,
            staking_ratio=latest_metrics.staking_ratio,
            validator_distribution=latest_metrics.validator_distribution,
            slashing_events=latest_metrics.slashing_events,
            audit_count=latest_metrics.audit_count,
            security_score=latest_metrics.security_score,
            audit_status="audited" if latest_metrics.audit_count > 0 else "pending",
            bug_bounty_programs=1 if latest_metrics.audit_count > 0 else 0,
            penetration_tests=latest_metrics.audit_count
        ),
        development_metrics=DevelopmentMetricsResponse(
            github_commits=latest_metrics.github_commits,
            github_stars=latest_metrics.github_stars,
            github_forks=latest_metrics.github_forks,
            developer_count=latest_metrics.developer_count,
            smart_contract_deployments=latest_metrics.smart_contract_deployments,
            subnet_launches=latest_metrics.subnet_launches
        ),
        user_behavior=UserBehaviorResponse(
            whale_activity=latest_metrics.whale_activity,
            retail_vs_institutional=latest_metrics.retail_vs_institutional,
            holding_patterns=latest_metrics.holding_patterns,
            transaction_sizes=latest_metrics.transaction_sizes,
            address_concentration=latest_metrics.address_concentration
        ),
        competitive_analysis=CompetitiveAnalysisResponse(
            market_share=latest_metrics.market_share if latest_metrics.market_share > 0 else 2.5,
            performance_vs_competitors=latest_metrics.performance_vs_competitors if latest_metrics.performance_vs_competitors else {
                "ethereum": {"tps": 15, "fees": 20.0, "finality": 12.0},
                "solana": {"tps": 3000, "fees": 0.00025, "finality": 0.4},
                "polygon": {"tps": 7000, "fees": 0.01, "finality": 2.0},
                "avalanche": {"tps": 4500, "fees": 0.1, "finality": 1.0}
            },
            ecosystem_growth=latest_metrics.ecosystem_growth if latest_metrics.ecosystem_growth else {
                "dapp_count": 200,
                "developer_count": 500,
                "tvl_growth": 15.5
            },
            developer_adoption=latest_metrics.developer_adoption if latest_metrics.developer_adoption else {
                "active_developers": 500,
                "new_projects": 25,
                "github_activity": 85.0
            }
        ),
        technical_infrastructure=TechnicalInfrastructureResponse(
            rpc_performance=latest_metrics.rpc_performance,
            node_distribution=latest_metrics.node_distribution,
            network_uptime=latest_metrics.network_uptime,
            upgrade_history=latest_metrics.upgrade_history,
            interoperability_score=latest_metrics.interoperability_score
        ),
        risk_assessment=RiskAssessmentResponse(
            centralization_risks=latest_metrics.centralization_risks,
            technical_risks=latest_metrics.technical_risks,
            regulatory_risks=latest_metrics.regulatory_risks,
            market_risks=latest_metrics.market_risks,
            competition_risks=latest_metrics.competition_risks
        ),
        macro_factors=MacroFactorsResponse(
            market_conditions=latest_metrics.market_conditions,
            institutional_adoption=latest_metrics.institutional_adoption,
            regulatory_environment=latest_metrics.regulatory_environment,
            economic_indicators=latest_metrics.economic_indicators
        ),
        ecosystem_health=EcosystemHealthResponse(
            community_growth=latest_metrics.community_growth,
            media_coverage=latest_metrics.media_coverage,
            partnership_quality=latest_metrics.partnership_quality,
            developer_experience=latest_metrics.developer_experience
        )
    )

@app.post("/collect")
async def trigger_metrics_collection(background_tasks: BackgroundTasks):
    """Trigger manual metrics collection"""
    global collection_in_progress
    
    if collection_in_progress:
        raise HTTPException(status_code=409, detail="Metrics collection already in progress")
    
    background_tasks.add_task(collect_metrics_background)
    
    return {"message": "Metrics collection started", "status": "collecting"}

@app.get("/historical/{hours}")
async def get_historical_metrics(hours: int = 24):
    """Get historical metrics from database"""
    if hours > 168:  # Max 1 week
        raise HTTPException(status_code=400, detail="Maximum 168 hours (1 week) allowed")
    
    try:
        with get_db_session() as db:
            # Get Avalanche blockchain
            avalanche = db.query(Blockchain).filter(Blockchain.chain_id == 43114).first()
            if not avalanche:
                raise HTTPException(status_code=404, detail="Avalanche blockchain not found")
            
            # Get historical network metrics
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            network_metrics = db.query(NetworkMetrics).filter(
                NetworkMetrics.blockchain_id == avalanche.id,
                NetworkMetrics.timestamp >= cutoff_time
            ).order_by(NetworkMetrics.timestamp.desc()).all()
            
            # Get historical economic metrics
            economic_metrics = db.query(EconomicMetrics).filter(
                EconomicMetrics.blockchain_id == avalanche.id,
                EconomicMetrics.timestamp >= cutoff_time
            ).order_by(EconomicMetrics.timestamp.desc()).all()
            
            return {
                "period_hours": hours,
                "network_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "block_time_avg": m.block_time_avg,
                        "transaction_throughput": m.transaction_throughput,
                        "network_utilization": m.network_utilization,
                        "gas_price_avg": m.gas_price_avg,
                        "validator_count": m.validator_count,
                        "staking_ratio": m.staking_ratio
                    } for m in network_metrics
                ],
                "economic_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "total_value_locked": float(m.total_value_locked) if m.total_value_locked else 0,
                        "daily_volume": float(m.daily_volume) if m.daily_volume else 0,
                        "active_users_24h": m.active_users_24h,
                        "transaction_fees_24h": float(m.transaction_fees_24h) if m.transaction_fees_24h else 0,
                        "market_cap": float(m.market_cap) if m.market_cap else 0
                    } for m in economic_metrics
                ]
            }
    
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving historical data")

async def collect_metrics_background():
    """Background task for collecting metrics"""
    global latest_metrics, metrics_collection_time, collection_in_progress
    
    collection_in_progress = True
    try:
        logger.info("Starting background metrics collection...")
        
        async with AvalancheMetricsCollector() as collector:
            metrics = await collector.collect_all_metrics()
            await collector.save_metrics_to_database(metrics)
            
            latest_metrics = metrics
            metrics_collection_time = datetime.utcnow()
            
            logger.info("Background metrics collection completed")
    
    except Exception as e:
        logger.error(f"Error in background metrics collection: {e}")
    
    finally:
        collection_in_progress = False

async def periodic_metrics_collection():
    """Periodic metrics collection task"""
    while True:
        try:
            await collect_metrics_background()
            # Wait for 1 hour before next collection
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in periodic metrics collection: {e}")
            # Wait 5 minutes before retrying
            await asyncio.sleep(300)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Avalanche Metrics API Server...")
    uvicorn.run(
        "avalanche_api_server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
