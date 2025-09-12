from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import asyncio
from loguru import logger

from database.database import SessionLocal, init_db
from database.models_v2 import CryptoAsset, Blockchain, MLPrediction
from api.auth_handler import router as auth_router
from api.data_loader import DataLoader, DataCollectionRequest
from api.metrics_mapper import MetricsMapper
from api.blockchain_client import BlockchainClient, BlockchainType
from ml.ml_pipeline import CryptoMLPipeline
from config.settings import settings

# Pydantic models for API
class AssetRequest(BaseModel):
    symbol: str
    name: str
    contract_address: Optional[str] = None
    blockchain_id: int
    coingecko_id: Optional[str] = None
    github_repo: Optional[str] = None
    website: Optional[str] = None
    category: str = "DeFi"

class DataCollectionRequestModel(BaseModel):
    asset_id: Optional[int] = None
    blockchain_id: Optional[int] = None
    symbol: Optional[str] = None
    time_periods: List[str] = ["1w", "2w", "4w"]
    metrics: List[str] = []
    force_refresh: bool = False

class MLPredictionRequest(BaseModel):
    asset_id: int
    model_name: str = "ensemble"
    prediction_type: str = "investment_score"

class BlockchainRequest(BaseModel):
    name: str
    symbol: str
    chain_id: int
    blockchain_type: str = "mainnet"
    rpc_url: Optional[str] = None
    explorer_url: Optional[str] = None
    native_token: str

# Initialize FastAPI app
app = FastAPI(
    title="DEFIMON Analytics Backend",
    description="Advanced crypto analytics backend with ML predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global instances
metrics_mapper = MetricsMapper()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting DEFIMON Analytics Backend v2.0")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Initialize ML pipeline
    global ml_pipeline
    ml_pipeline = CryptoMLPipeline()
    await ml_pipeline.load_models()
    logger.info("ML pipeline initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DEFIMON Analytics Backend")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DEFIMON Analytics Backend</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; }
            .btn { display: inline-block; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 DEFIMON Analytics Backend v2.0</h1>
            <p>Advanced Crypto Analytics with Machine Learning</p>
        </div>
        
        <div class="section">
            <h2>📊 Available Endpoints</h2>
            <div class="endpoint">GET /api/assets - List all crypto assets</div>
            <div class="endpoint">POST /api/assets - Add new asset</div>
            <div class="endpoint">GET /api/blockchains - List supported blockchains</div>
            <div class="endpoint">POST /api/blockchains - Add new blockchain</div>
            <div class="endpoint">POST /api/collect-data - Collect data for assets</div>
            <div class="endpoint">GET /api/metrics - Get available metrics</div>
            <div class="endpoint">POST /api/predict - Generate ML predictions</div>
            <div class="endpoint">GET /api/predictions - Get ML predictions</div>
            <div class="endpoint">GET /api/health - Health check</div>
        </div>
        
        <div class="section">
            <h2>🔐 Authentication</h2>
            <a href="/auth/github" class="btn">GitHub Authorization</a>
            <a href="/auth/github/status" class="btn">Check Auth Status</a>
        </div>
        
        <div class="section">
            <h2>📚 Documentation</h2>
            <a href="/docs" class="btn">Swagger UI</a>
            <a href="/redoc" class="btn">ReDoc</a>
        </div>
        
        <div class="section">
            <h2>🎯 Features</h2>
            <ul>
                <li>✅ 50+ Blockchain Support</li>
                <li>✅ 10 Categories of Metrics</li>
                <li>✅ Machine Learning Predictions</li>
                <li>✅ Real-time Data Collection</li>
                <li>✅ GitHub Integration</li>
                <li>✅ Advanced Analytics</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "database": "connected",
            "ml_pipeline": "ready",
            "metrics_mapper": "ready"
        }
    }

# Asset Management Endpoints
@app.get("/api/assets")
async def get_assets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    blockchain_id: Optional[int] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of crypto assets"""
    query = db.query(CryptoAsset)
    
    if blockchain_id:
        query = query.filter(CryptoAsset.blockchain_id == blockchain_id)
    if category:
        query = query.filter(CryptoAsset.category == category)
    
    assets = query.offset(skip).limit(limit).all()
    
    return {
        "assets": [
            {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "contract_address": asset.contract_address,
                "blockchain_id": asset.blockchain_id,
                "category": asset.category,
                "coingecko_id": asset.coingecko_id,
                "github_repo": asset.github_repo,
                "website": asset.website,
                "is_active": asset.is_active,
                "is_verified": asset.is_verified,
                "created_at": asset.created_at.isoformat() if asset.created_at else None
            }
            for asset in assets
        ],
        "total": len(assets),
        "skip": skip,
        "limit": limit
    }

@app.post("/api/assets")
async def create_asset(asset_request: AssetRequest, db: Session = Depends(get_db)):
    """Create a new crypto asset"""
    try:
        # Check if asset already exists
        existing = db.query(CryptoAsset).filter(
            CryptoAsset.symbol == asset_request.symbol,
            CryptoAsset.blockchain_id == asset_request.blockchain_id
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Asset already exists")
        
        # Create new asset
        asset = CryptoAsset(
            symbol=asset_request.symbol,
            name=asset_request.name,
            contract_address=asset_request.contract_address,
            blockchain_id=asset_request.blockchain_id,
            category=asset_request.category,
            coingecko_id=asset_request.coingecko_id,
            github_repo=asset_request.github_repo,
            website=asset_request.website
        )
        
        db.add(asset)
        db.commit()
        db.refresh(asset)
        
        return {
            "message": "Asset created successfully",
            "asset": {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "contract_address": asset.contract_address,
                "blockchain_id": asset.blockchain_id,
                "category": asset.category
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchains")
async def get_blockchains(db: Session = Depends(get_db)):
    """Get list of supported blockchains"""
    blockchains = db.query(Blockchain).filter(Blockchain.is_active == True).all()
    
    return {
        "blockchains": [
            {
                "id": blockchain.id,
                "name": blockchain.name,
                "symbol": blockchain.symbol,
                "chain_id": blockchain.chain_id,
                "blockchain_type": blockchain.blockchain_type,
                "native_token": blockchain.native_token,
                "explorer_url": blockchain.explorer_url,
                "is_active": blockchain.is_active
            }
            for blockchain in blockchains
        ],
        "total": len(blockchains)
    }

@app.post("/api/blockchains")
async def create_blockchain(blockchain_request: BlockchainRequest, db: Session = Depends(get_db)):
    """Create a new blockchain"""
    try:
        # Check if blockchain already exists
        existing = db.query(Blockchain).filter(
            Blockchain.chain_id == blockchain_request.chain_id
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Blockchain already exists")
        
        # Create new blockchain
        blockchain = Blockchain(
            name=blockchain_request.name,
            symbol=blockchain_request.symbol,
            chain_id=blockchain_request.chain_id,
            blockchain_type=blockchain_request.blockchain_type,
            rpc_url=blockchain_request.rpc_url,
            explorer_url=blockchain_request.explorer_url,
            native_token=blockchain_request.native_token
        )
        
        db.add(blockchain)
        db.commit()
        db.refresh(blockchain)
        
        return {
            "message": "Blockchain created successfully",
            "blockchain": {
                "id": blockchain.id,
                "name": blockchain.name,
                "symbol": blockchain.symbol,
                "chain_id": blockchain.chain_id,
                "blockchain_type": blockchain.blockchain_type
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Collection Endpoints
@app.post("/api/collect-data")
async def collect_data(
    request: DataCollectionRequestModel,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Collect data for assets"""
    try:
        # Convert to DataCollectionRequest
        data_request = DataCollectionRequest(
            asset_id=request.asset_id,
            blockchain_id=request.blockchain_id,
            symbol=request.symbol,
            time_periods=request.time_periods,
            metrics=request.metrics,
            force_refresh=request.force_refresh
        )
        
        # Start data collection in background
        background_tasks.add_task(collect_data_background, data_request)
        
        return {
            "message": "Data collection started",
            "request": {
                "asset_id": request.asset_id,
                "blockchain_id": request.blockchain_id,
                "symbol": request.symbol,
                "time_periods": request.time_periods,
                "metrics": request.metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def collect_data_background(request: DataCollectionRequest):
    """Background task for data collection"""
    try:
        async with DataLoader() as data_loader:
            results = await data_loader.collect_data_for_asset(request)
            
            # Log results
            for result in results:
                if result.success:
                    logger.info(f"Data collection successful for asset {result.asset_id} - {result.time_period}")
                else:
                    logger.error(f"Data collection failed for asset {result.asset_id} - {result.time_period}: {result.errors}")
                    
    except Exception as e:
        logger.error(f"Background data collection error: {e}")

@app.get("/api/metrics")
async def get_metrics():
    """Get available metrics and their definitions"""
    try:
        summary = metrics_mapper.get_metrics_summary()
        
        # Get detailed metric definitions
        all_metrics = {}
        for metric_name in metrics_mapper.metrics_definitions.keys():
            metric_def = metrics_mapper.get_metric_definition(metric_name)
            if metric_def:
                all_metrics[metric_name] = {
                    "name": metric_def.name,
                    "category": metric_def.category.value,
                    "description": metric_def.description,
                    "data_sources": [source.value for source in metric_def.data_sources],
                    "required_fields": metric_def.required_fields,
                    "update_frequency": metric_def.update_frequency,
                    "priority": metric_def.priority
                }
        
        return {
            "summary": summary,
            "metrics": all_metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{asset_id}")
async def get_asset_metrics(asset_id: int, db: Session = Depends(get_db)):
    """Get available metrics for a specific asset"""
    try:
        asset = db.query(CryptoAsset).filter(CryptoAsset.id == asset_id).first()
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        # Prepare asset data
        asset_data = {
            "id": asset.id,
            "symbol": asset.symbol,
            "name": asset.name,
            "contract_address": asset.contract_address,
            "blockchain_id": asset.blockchain_id,
            "coingecko_id": asset.coingecko_id,
            "github_repo": asset.github_repo,
            "website": asset.website,
            "category": asset.category
        }
        
        # Get available metrics
        available_metrics = metrics_mapper.get_metrics_for_asset(asset_data)
        collection_plan = metrics_mapper.get_data_collection_plan(asset_data)
        validation_result = metrics_mapper.validate_asset_data(asset_data)
        
        return {
            "asset": asset_data,
            "available_metrics": available_metrics,
            "collection_plan": collection_plan,
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"Error getting asset metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML Prediction Endpoints
@app.post("/api/predict")
async def generate_prediction(
    request: MLPredictionRequest,
    background_tasks: BackgroundTasks
):
    """Generate ML prediction for an asset"""
    try:
        # Start prediction in background
        background_tasks.add_task(generate_prediction_background, request)
        
        return {
            "message": "Prediction generation started",
            "request": {
                "asset_id": request.asset_id,
                "model_name": request.model_name,
                "prediction_type": request.prediction_type
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_prediction_background(request: MLPredictionRequest):
    """Background task for prediction generation"""
    try:
        async with CryptoMLPipeline() as ml_pipeline:
            result = await ml_pipeline.predict_investment_score(
                request.asset_id, request.model_name
            )
            
            logger.info(f"Prediction generated for asset {request.asset_id}: {result.prediction_value}")
            
    except Exception as e:
        logger.error(f"Background prediction error: {e}")

@app.get("/api/predictions")
async def get_predictions(
    asset_id: Optional[int] = None,
    model_name: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get ML predictions"""
    try:
        query = db.query(MLPrediction)
        
        if asset_id:
            query = query.filter(MLPrediction.asset_id == asset_id)
        if model_name:
            query = query.filter(MLPrediction.model_name == model_name)
        
        predictions = query.order_by(MLPrediction.created_at.desc()).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "id": pred.id,
                    "asset_id": pred.asset_id,
                    "model_name": pred.model_name,
                    "prediction_type": pred.prediction_type,
                    "prediction_value": pred.prediction_value,
                    "confidence_score": pred.confidence_score,
                    "prediction_horizon": pred.prediction_horizon,
                    "features_used": pred.features_used,
                    "model_version": pred.model_version,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None
                }
                for pred in predictions
            ],
            "total": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-models")
async def train_models(background_tasks: BackgroundTasks):
    """Train ML models with latest data"""
    try:
        background_tasks.add_task(train_models_background)
        
        return {
            "message": "Model training started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_models_background():
    """Background task for model training"""
    try:
        async with CryptoMLPipeline() as ml_pipeline:
            # Prepare training data
            features_df, targets_series = await ml_pipeline.prepare_training_data()
            
            if len(features_df) > 0:
                # Train models
                results = await ml_pipeline.train_models(features_df, targets_series)
                
                # Save models
                await ml_pipeline.save_models()
                
                logger.info(f"Model training completed: {len(results)} models trained")
            else:
                logger.warning("No training data available")
                
    except Exception as e:
        logger.error(f"Background model training error: {e}")

# Blockchain-specific endpoints
@app.get("/api/blockchain/{blockchain_name}/stats")
async def get_blockchain_stats(blockchain_name: str):
    """Get blockchain network statistics"""
    try:
        blockchain_client = BlockchainClient.create_client(blockchain_name.lower())
        
        async with blockchain_client as client:
            stats = await client.get_network_stats()
            
        return {
            "blockchain": blockchain_name,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting blockchain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain/{blockchain_name}/historical")
async def get_blockchain_historical(
    blockchain_name: str,
    days: int = Query(7, ge=1, le=365)
):
    """Get historical blockchain data"""
    try:
        blockchain_client = BlockchainClient.create_client(blockchain_name.lower())
        
        async with blockchain_client as client:
            historical_data = await client.get_historical_data(days)
            
        return {
            "blockchain": blockchain_name,
            "historical_data": historical_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics/summary")
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Get analytics summary"""
    try:
        # Get counts
        total_assets = db.query(CryptoAsset).count()
        active_assets = db.query(CryptoAsset).filter(CryptoAsset.is_active == True).count()
        total_blockchains = db.query(Blockchain).filter(Blockchain.is_active == True).count()
        total_predictions = db.query(MLPrediction).count()
        
        # Get recent predictions
        recent_predictions = db.query(MLPrediction).order_by(
            MLPrediction.created_at.desc()
        ).limit(10).all()
        
        return {
            "summary": {
                "total_assets": total_assets,
                "active_assets": active_assets,
                "total_blockchains": total_blockchains,
                "total_predictions": total_predictions
            },
            "recent_predictions": [
                {
                    "asset_id": pred.asset_id,
                    "model_name": pred.model_name,
                    "prediction_value": pred.prediction_value,
                    "confidence_score": pred.confidence_score,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None
                }
                for pred in recent_predictions
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        workers=1
    )
