from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from loguru import logger
import os

from config.settings import settings
from database.database import init_db, get_db
from worker.data_collector import DataCollectorWorker

# Configure logging
logger.add("logs/defimon.log", rotation="1 day", retention="30 days")

# Global worker instance
worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global worker
    
    # Startup
    logger.info("Starting DEFIMON Analytics System...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Start background workers
    worker = DataCollectorWorker()
    asyncio.create_task(worker.start())
    logger.info("Background workers started")
    
    yield
    
    # Shutdown
    if worker:
        await worker.stop()
    logger.info("Shutting down DEFIMON Analytics System...")

# Create FastAPI app
app = FastAPI(
    title="DEFIMON Analytics API",
    description="DeFi Analytics System for crypto asset investment analysis",
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

# Mount static files for web interface
if os.path.exists("src/web/dist"):
    app.mount("/static", StaticFiles(directory="src/web/dist"), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DEFIMON Analytics API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "assets": "/api/assets",
            "analytics": "/api/analytics",
            "predictions": "/api/predictions",
            "competitors": "/api/competitors",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global worker
    
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "worker_running": worker.is_running if worker else False,
        "database": "connected",
        "apis": {
            "quicknode": "connected",
            "etherscan": "connected"
        }
    }
    
    return health_status

@app.get("/api/assets")
async def get_assets():
    """Get all crypto assets"""
    from database.models import CryptoAsset
    from sqlalchemy.orm import Session
    
    db = next(get_db())
    try:
        assets = db.query(CryptoAsset).all()
        return [
            {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "contract_address": asset.contract_address,
                "blockchain": asset.blockchain,
                "category": asset.category,
                "github_repo": asset.github_repo,
                "website": asset.website,
                "created_at": asset.created_at.isoformat(),
                "updated_at": asset.updated_at.isoformat()
            }
            for asset in assets
        ]
    finally:
        db.close()

@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: int):
    """Get specific asset with latest metrics"""
    from database.models import CryptoAsset, OnChainMetrics, FinancialMetrics, MLPrediction
    from sqlalchemy.orm import Session
    
    db = next(get_db())
    try:
        asset = db.query(CryptoAsset).filter(CryptoAsset.id == asset_id).first()
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        # Get latest metrics
        latest_onchain = db.query(OnChainMetrics).filter(
            OnChainMetrics.asset_id == asset_id
        ).order_by(OnChainMetrics.timestamp.desc()).first()
        
        latest_financial = db.query(FinancialMetrics).filter(
            FinancialMetrics.asset_id == asset_id
        ).order_by(FinancialMetrics.timestamp.desc()).first()
        
        latest_prediction = db.query(MLPrediction).filter(
            MLPrediction.asset_id == asset_id
        ).order_by(MLPrediction.created_at.desc()).first()
        
        return {
            "asset": {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "contract_address": asset.contract_address,
                "blockchain": asset.blockchain,
                "category": asset.category,
                "github_repo": asset.github_repo,
                "website": asset.website
            },
            "onchain_metrics": {
                "tvl": latest_onchain.tvl if latest_onchain else None,
                "daily_transactions": latest_onchain.daily_transactions if latest_onchain else None,
                "active_addresses_24h": latest_onchain.active_addresses_24h if latest_onchain else None,
                "transaction_volume_24h": latest_onchain.transaction_volume_24h if latest_onchain else None,
                "gas_price_avg": latest_onchain.gas_price_avg if latest_onchain else None,
                "timestamp": latest_onchain.timestamp.isoformat() if latest_onchain else None
            },
            "financial_metrics": {
                "total_supply": latest_financial.total_supply if latest_financial else None,
                "circulating_supply": latest_financial.circulating_supply if latest_financial else None,
                "volume_24h": latest_financial.volume_24h if latest_financial else None,
                "security_score": latest_financial.security_score if latest_financial else None,
                "is_verified": latest_financial.is_verified if latest_financial else None,
                "timestamp": latest_financial.timestamp.isoformat() if latest_financial else None
            },
            "prediction": {
                "investment_score": latest_prediction.prediction_value if latest_prediction else None,
                "confidence_score": latest_prediction.confidence_score if latest_prediction else None,
                "model_name": latest_prediction.model_name if latest_prediction else None,
                "prediction_horizon": latest_prediction.prediction_horizon if latest_prediction else None,
                "created_at": latest_prediction.created_at.isoformat() if latest_prediction else None
            }
        }
    finally:
        db.close()

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics overview"""
    from database.models import CryptoAsset, OnChainMetrics, FinancialMetrics, MLPrediction
    from sqlalchemy.orm import Session
    from sqlalchemy import func
    
    db = next(get_db())
    try:
        # Get basic stats
        total_assets = db.query(CryptoAsset).count()
        total_predictions = db.query(MLPrediction).count()
        
        # Get average investment scores
        avg_score = db.query(func.avg(MLPrediction.prediction_value)).scalar() or 0
        
        # Get top performing assets
        top_assets = db.query(
            CryptoAsset.symbol,
            MLPrediction.prediction_value
        ).join(MLPrediction).order_by(
            MLPrediction.prediction_value.desc()
        ).limit(10).all()
        
        return {
            "total_assets": total_assets,
            "total_predictions": total_predictions,
            "average_investment_score": round(avg_score, 4),
            "top_performing_assets": [
                {"symbol": symbol, "score": round(score, 4)}
                for symbol, score in top_assets
            ],
            "last_updated": "2024-01-01T00:00:00Z"
        }
    finally:
        db.close()

@app.get("/api/predictions")
async def get_predictions():
    """Get all predictions"""
    from database.models import CryptoAsset, MLPrediction
    from sqlalchemy.orm import Session
    
    db = next(get_db())
    try:
        predictions = db.query(
            CryptoAsset.symbol,
            CryptoAsset.name,
            MLPrediction.prediction_value,
            MLPrediction.confidence_score,
            MLPrediction.model_name,
            MLPrediction.created_at
        ).join(MLPrediction).order_by(
            MLPrediction.prediction_value.desc()
        ).all()
        
        return [
            {
                "symbol": symbol,
                "name": name,
                "investment_score": round(score, 4),
                "confidence_score": round(confidence, 4),
                "model_name": model_name,
                "created_at": created_at.isoformat()
            }
            for symbol, name, score, confidence, model_name, created_at in predictions
        ]
    finally:
        db.close()

@app.get("/api/competitors")
async def get_competitors():
    """Get competitor analysis"""
    return {
        "competitors": settings.COMPETITOR_PLATFORMS,
        "analysis": {
            "total_competitors": len(settings.COMPETITOR_PLATFORMS),
            "feature_gaps": [
                "investment_score_prediction",
                "team_activity_analysis",
                "smart_contract_deployment_tracking",
                "multi_chain_analysis",
                "real_time_risk_assessment"
            ],
            "our_advantages": [
                "AI-powered investment scoring",
                "Real-time on-chain data analysis",
                "Multi-source data aggregation",
                "Advanced ML models",
                "Polygon-focused analytics"
            ]
        }
    }

@app.post("/api/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    global worker
    
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not available")
    
    background_tasks.add_task(worker.retrain_models)
    
    return {"message": "Model retraining started", "status": "accepted"}

@app.get("/api/stats")
async def get_collection_stats():
    """Get data collection statistics"""
    global worker
    
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not available")
    
    return worker.get_collection_stats()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=1,  # Use 1 worker for development
        reload=True
    )
