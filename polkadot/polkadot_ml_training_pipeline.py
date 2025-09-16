#!/usr/bin/env python3
"""
Polkadot ML Training Pipeline
============================

Automated training pipeline for Polkadot machine learning models.
This script handles data collection, preprocessing, training, and deployment.

Usage:
    python polkadot_ml_training_pipeline.py --task price_prediction --days 90
    python polkadot_ml_training_pipeline.py --task all --upload-hf
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from polkadot_ml_strategy import PolkadotMLStrategy, MLTaskType

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/ml_training.log", rotation="1 day", retention="30 days", level="DEBUG")

async def main():
    """Main training pipeline function"""
    parser = argparse.ArgumentParser(description="Polkadot ML Training Pipeline")
    parser.add_argument("--task", type=str, default="all", 
                       choices=["all", "price_prediction", "tvl_prediction", "anomaly_detection", "network_health", "staking_analysis"],
                       help="ML task to train")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data to use")
    parser.add_argument("--upload-hf", action="store_true", help="Upload models to Hugging Face")
    parser.add_argument("--repo-name", type=str, default="polkadot-ml-models", help="HF repository name")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    
    args = parser.parse_args()
    
    logger.info(f"Starting ML training pipeline for task: {args.task}")
    
    try:
        # Initialize ML strategy
        ml_strategy = PolkadotMLStrategy()
        
        if args.task == "all":
            # Train all models
            await ml_strategy.run_ml_pipeline(
                days_back=args.days,
                upload_to_hf=args.upload_hf
            )
        else:
            # Train specific model
            task_type = MLTaskType(args.task)
            config = ml_strategy.model_configs[task_type]
            
            # Collect and preprocess data
            df = await ml_strategy.collect_training_data(args.days)
            X, y = ml_strategy.preprocess_data(df, config)
            
            # Train specific model
            if task_type == MLTaskType.PRICE_PREDICTION:
                model = ml_strategy.train_price_predictor(X, y, config)
            elif task_type == MLTaskType.TVL_PREDICTION:
                model = ml_strategy.train_tvl_predictor(X, y, config)
            elif task_type == MLTaskType.ANOMALY_DETECTION:
                model = ml_strategy.train_anomaly_detector(X, y, config)
            elif task_type == MLTaskType.NETWORK_HEALTH:
                model = ml_strategy.train_network_health_model(X, y, config)
            
            ml_strategy.trained_models[task_type] = model
            
            # Upload to HF if requested
            if args.upload_hf:
                await ml_strategy.upload_models_to_hf(args.repo_name)
        
        logger.success("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
