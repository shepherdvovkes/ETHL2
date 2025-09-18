import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from sqlalchemy.orm import Session

from database.database import SessionLocal, init_db
from database.models import (
    CryptoAsset, OnChainMetrics, GitHubMetrics, 
    FinancialMetrics, MLPrediction, SmartContract
)
from api.quicknode_client import QuickNodeClient
from api.etherscan_client import EtherscanClient
from ml.models import DeFiAnalyticsML
from config.settings import settings

class DataCollectorWorker:
    """Background worker for data collection"""
    
    def __init__(self):
        self.quicknode_client = None
        self.etherscan_client = None
        self.ml_pipeline = DeFiAnalyticsML()
        self.is_running = False
    
    async def start(self):
        """Start the data collection worker"""
        self.is_running = True
        logger.info("Data collection worker started")
        
        # Initialize database
        init_db()
        
        # Load ML models
        self.ml_pipeline.load_models()
        
        # Initialize default assets if database is empty
        await self.initialize_default_assets()
        
        while self.is_running:
            try:
                await self.collect_all_data()
                await asyncio.sleep(settings.COLLECTION_INTERVAL)
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def stop(self):
        """Stop the data collection worker"""
        self.is_running = False
        logger.info("Data collection worker stopped")
    
    async def initialize_default_assets(self):
        """Initialize default assets in database"""
        db = SessionLocal()
        try:
            # Check if assets already exist
            existing_count = db.query(CryptoAsset).count()
            if existing_count > 0:
                logger.info(f"Database already has {existing_count} assets")
                return
            
            logger.info("Initializing default assets...")
            
            for asset_data in settings.DEFAULT_ASSETS:
                # Check if asset already exists
                existing = db.query(CryptoAsset).filter(
                    CryptoAsset.symbol == asset_data["symbol"]
                ).first()
                
                if not existing:
                    asset = CryptoAsset(**asset_data)
                    db.add(asset)
                    
                    # Also add smart contract if address exists
                    if asset_data.get("contract"):
                        contract = SmartContract(
                            contract_address=asset_data["contract"],
                            blockchain=asset_data.get("blockchain", "polygon"),
                            contract_name=asset_data["name"],
                            contract_type="ERC20"
                        )
                        db.add(contract)
            
            db.commit()
            logger.info("Default assets initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing default assets: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def collect_all_data(self):
        """Collect data for all assets"""
        db = SessionLocal()
        try:
            # Get all assets
            assets = db.query(CryptoAsset).all()
            logger.info(f"Collecting data for {len(assets)} assets")
            
            # Initialize API clients
            async with QuickNodeClient() as qn_client, \
                     EtherscanClient() as es_client:
                
                self.quicknode_client = qn_client
                self.etherscan_client = es_client
                
                # Collect data for each asset
                for asset in assets:
                    try:
                        await self.collect_asset_data(asset, db)
                    except Exception as e:
                        logger.error(f"Error collecting data for {asset.symbol}: {e}")
                        continue
                
                # Generate ML predictions
                await self.generate_predictions(db)
                
        finally:
            db.close()
    
    async def collect_asset_data(self, asset: CryptoAsset, db: Session):
        """Collect data for a specific asset"""
        logger.info(f"Collecting data for {asset.symbol}")
        
        # Collect on-chain metrics
        onchain_data = await self.collect_onchain_metrics(asset)
        if onchain_data:
            onchain_metrics = OnChainMetrics(asset_id=asset.id, **onchain_data)
            db.add(onchain_metrics)
        
        # Collect financial metrics
        financial_data = await self.collect_financial_metrics(asset)
        if financial_data:
            financial_metrics = FinancialMetrics(asset_id=asset.id, **financial_data)
            db.add(financial_metrics)
        
        # Collect smart contract data
        if asset.contract_address:
            await self.collect_contract_data(asset, db)
        
        db.commit()
    
    async def collect_onchain_metrics(self, asset: CryptoAsset) -> Optional[Dict]:
        """Collect on-chain metrics using QuickNode and Etherscan"""
        try:
            if not asset.contract_address:
                return None
            
            # Get current block number
            current_block = await self.quicknode_client.get_block_number()
            if current_block == 0:
                return None
            
            # Calculate block range for 24h (Polygon has ~2s block time)
            blocks_per_day = 24 * 60 * 60 // 2  # Rough estimate for Polygon
            from_block = max(0, current_block - blocks_per_day)
            
            # Get transaction count
            tx_count = await self.quicknode_client.get_transaction_count(asset.contract_address)
            
            # Get active addresses
            active_addresses = await self.quicknode_client.get_active_addresses(from_block, current_block)
            
            # Get transaction volume
            tx_volume = await self.quicknode_client.get_transaction_volume(from_block, current_block)
            
            # Get gas price
            gas_price = await self.quicknode_client.get_gas_price()
            
            # Get network stats
            network_stats = await self.quicknode_client.get_network_stats()
            
            # Get token transfers for the day
            token_transfers = await self.quicknode_client.get_token_transfers(
                asset.contract_address, from_block, current_block
            )
            
            # Get contract interactions
            contract_interactions = await self.quicknode_client.get_contract_interactions(
                asset.contract_address, from_block, current_block
            )
            
            return {
                "daily_transactions": tx_count,
                "active_addresses_24h": active_addresses,
                "transaction_volume_24h": tx_volume,
                "contract_interactions_24h": len(contract_interactions),
                "gas_price_avg": gas_price,
                "network_utilization": network_stats.get("network_utilization", 0.0),
                "block_time_avg": 2.0,  # Polygon average
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error collecting on-chain metrics for {asset.symbol}: {e}")
            return None
    
    async def collect_financial_metrics(self, asset: CryptoAsset) -> Optional[Dict]:
        """Collect financial metrics using Etherscan"""
        try:
            if not asset.contract_address:
                return None
            
            # Get token info
            token_info = await self.etherscan_client.get_token_info(asset.contract_address)
            
            # Get token supply
            total_supply = await self.etherscan_client.get_token_supply(asset.contract_address)
            
            # Get recent token transfers for volume calculation
            current_block = await self.quicknode_client.get_block_number()
            blocks_per_day = 24 * 60 * 60 // 2
            from_block = max(0, current_block - blocks_per_day)
            
            token_transfers = await self.etherscan_client.get_token_transfers(
                asset.contract_address, 
                start_block=from_block,
                end_block=current_block
            )
            
            # Calculate volume from transfers
            volume_24h = 0.0
            for transfer in token_transfers:
                # This is a simplified calculation
                # In reality, you'd need to get the price at the time of transfer
                volume_24h += float(transfer.get("value", 0)) / 10**18
            
            # Get contract security analysis
            security_analysis = await self.etherscan_client.analyze_contract_security(asset.contract_address)
            
            return {
                "total_supply": total_supply,
                "circulating_supply": total_supply * 0.8,  # Rough estimate
                "volume_24h": volume_24h,
                "security_score": security_analysis.get("security_score", 0),
                "is_verified": security_analysis.get("is_verified", False),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error collecting financial metrics for {asset.symbol}: {e}")
            return None
    
    async def collect_contract_data(self, asset: CryptoAsset, db: Session):
        """Collect smart contract data"""
        try:
            if not asset.contract_address:
                return
            
            # Check if contract already exists
            existing_contract = db.query(SmartContract).filter(
                SmartContract.contract_address == asset.contract_address
            ).first()
            
            if existing_contract:
                return
            
            # Get contract source code and info
            contract_info = await self.etherscan_client.get_contract_source_code(asset.contract_address)
            creation_info = await self.etherscan_client.get_contract_creation(asset.contract_address)
            security_analysis = await self.etherscan_client.analyze_contract_security(asset.contract_address)
            
            # Create smart contract record
            contract = SmartContract(
                asset_id=asset.id,
                contract_address=asset.contract_address,
                blockchain=asset.blockchain,
                contract_name=contract_info.get("contract_name", asset.name),
                contract_type="ERC20",  # Default for tokens
                bytecode="",  # Would need to fetch separately
                abi=contract_info.get("abi", ""),
                verified=contract_info.get("source_code", "") != "",
                deployer_address=creation_info.get("contract_creator", ""),
                deployment_tx=creation_info.get("tx_hash", ""),
                audit_status="unaudited",  # Would need to check audit databases
                created_at=datetime.utcnow()
            )
            
            db.add(contract)
            
        except Exception as e:
            logger.error(f"Error collecting contract data for {asset.symbol}: {e}")
    
    async def generate_predictions(self, db: Session):
        """Generate ML predictions for all assets"""
        try:
            # Get latest metrics for all assets
            assets = db.query(CryptoAsset).all()
            
            for asset in assets:
                # Get latest metrics
                latest_onchain = db.query(OnChainMetrics).filter(
                    OnChainMetrics.asset_id == asset.id
                ).order_by(OnChainMetrics.timestamp.desc()).first()
                
                latest_financial = db.query(FinancialMetrics).filter(
                    FinancialMetrics.asset_id == asset.id
                ).order_by(FinancialMetrics.timestamp.desc()).first()
                
                # Combine metrics
                combined_data = {}
                
                if latest_onchain:
                    combined_data.update({
                        "tvl": latest_onchain.tvl or 0,
                        "daily_transactions": latest_onchain.daily_transactions or 0,
                        "active_addresses_24h": latest_onchain.active_addresses_24h or 0,
                        "transaction_volume_24h": latest_onchain.transaction_volume_24h or 0,
                        "contract_interactions_24h": latest_onchain.contract_interactions_24h or 0,
                        "gas_price_avg": latest_onchain.gas_price_avg or 0,
                        "network_utilization": latest_onchain.network_utilization or 0,
                    })
                
                if latest_financial:
                    combined_data.update({
                        "market_cap": latest_financial.market_cap or 0,
                        "volume_24h": latest_financial.volume_24h or 0,
                        "volatility_24h": latest_financial.volatility_24h or 0,
                        "price_change_24h": latest_financial.price_change_24h or 0,
                        "total_supply": latest_financial.total_supply or 0,
                        "circulating_supply": latest_financial.circulating_supply or 0,
                    })
                
                # Generate prediction
                if combined_data and len(combined_data) >= 5:  # Minimum features required
                    try:
                        # Use different models for ensemble prediction
                        predictions = {}
                        for model_name in ['random_forest', 'gradient_boosting', 'linear_regression']:
                            if model_name in self.ml_pipeline.models:
                                pred = self.ml_pipeline.predict_investment_score(combined_data, model_name)
                                predictions[model_name] = pred
                        
                        # Calculate ensemble prediction (average)
                        if predictions:
                            ensemble_prediction = sum(predictions.values()) / len(predictions)
                            
                            # Save prediction
                            prediction = MLPrediction(
                                asset_id=asset.id,
                                model_name="ensemble",
                                prediction_type="investment_score",
                                prediction_value=ensemble_prediction,
                                confidence_score=0.8,  # Placeholder
                                prediction_horizon="7d",
                                features_used=list(combined_data.keys()),
                                model_version="1.0"
                            )
                            db.add(prediction)
                            
                            logger.info(f"Generated prediction for {asset.symbol}: {ensemble_prediction:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error generating prediction for {asset.symbol}: {e}")
            
            db.commit()
            logger.info("ML predictions generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    async def collect_competitor_data(self):
        """Collect competitor analysis data"""
        # This would be implemented to analyze competitor platforms
        # For now, we'll create a placeholder
        logger.info("Competitor data collection not implemented yet")
    
    async def retrain_models(self):
        """Retrain ML models with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Get all historical data
            db = SessionLocal()
            
            # This would collect all historical metrics and retrain models
            # For now, we'll just reload existing models
            self.ml_pipeline.load_models()
            
            db.close()
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        db = SessionLocal()
        try:
            stats = {
                "total_assets": db.query(CryptoAsset).count(),
                "total_onchain_metrics": db.query(OnChainMetrics).count(),
                "total_financial_metrics": db.query(FinancialMetrics).count(),
                "total_predictions": db.query(MLPrediction).count(),
                "total_contracts": db.query(SmartContract).count(),
                "last_collection": datetime.utcnow().isoformat()
            }
            
            return stats
            
        finally:
            db.close()
