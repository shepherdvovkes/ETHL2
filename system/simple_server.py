#!/usr/bin/env python3
"""
Simple server with basic data to test the dashboard
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import asyncio
import sys
import os
sys.path.append('src')

from api.avalanche_quicknode_client import AvalancheQuickNodeClient

app = FastAPI(title="Avalanche Network Metrics API", version="1.0.0")

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
    return FileResponse("index.html")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "latest_metrics_time": datetime.utcnow().isoformat(),
        "collection_in_progress": False
    }

async def get_basic_network_data():
    """Get basic network data from QuickNode"""
    try:
        async with AvalancheQuickNodeClient() as qn:
            block_number = await qn.get_c_chain_block_number()
            gas_price = await qn.get_c_chain_gas_price()
            
            return {
                "current_block": block_number,
                "gas_price_avg": gas_price,
                "gas_price_median": gas_price,
                "block_time": 2.0,
                "transaction_throughput": 4500,
                "network_utilization": 75.0,
                "finality_time": 1.0
            }
    except Exception as e:
        print(f"QuickNode error: {e}")
        return {
            "current_block": 0, "gas_price_avg": 0, "gas_price_median": 0,
            "block_time": 0, "transaction_throughput": 0,
            "network_utilization": 0, "finality_time": 0
        }

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of key Avalanche metrics"""
    network_data = await get_basic_network_data()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "key_metrics": {
            "transaction_throughput": network_data["transaction_throughput"],
            "market_cap": 0,
            "daily_volume": 0,
            "validator_count": 0,
            "defi_protocols_count": 0,
            "subnet_count": 0,
            "active_addresses_24h": 0,
            "gas_price_avg": network_data["gas_price_avg"],
            "block_time": network_data["block_time"],
            "finality_time": network_data["finality_time"],
            "staking_ratio": 0,
            "defi_tvl": 0
        },
        "status": "operational"
    }

@app.get("/metrics/network-performance")
async def get_network_performance():
    """Get network performance metrics"""
    network_data = await get_basic_network_data()
    
    return {
        "block_time": network_data["block_time"],
        "transaction_throughput": network_data["transaction_throughput"],
        "finality_time": network_data["finality_time"],
        "network_utilization": network_data["network_utilization"],
        "gas_price_avg": network_data["gas_price_avg"],
        "gas_price_median": network_data["gas_price_median"],
        "block_size_avg": 0,
        "current_block": network_data["current_block"]
    }

@app.get("/metrics/economic")
async def get_economic_metrics():
    """Get economic metrics"""
    return {
        "total_value_locked": 0,
        "daily_volume": 0,
        "active_addresses_24h": 0,
        "new_addresses_24h": 0,
        "transaction_fees_24h": 0,
        "revenue_24h": 0,
        "market_cap": 0,
        "circulating_supply": 0,
        "total_supply": 0,
        "price": 0
    }

@app.get("/metrics/defi")
async def get_defi_metrics():
    """Get DeFi ecosystem metrics"""
    return {
        "defi_protocols_count": 0,
        "defi_tvl": 0,
        "dex_volume_24h": 0,
        "lending_volume_24h": 0,
        "yield_farming_apy": 0,
        "bridge_volume_24h": 0
    }

@app.get("/metrics/subnet")
async def get_subnet_metrics():
    """Get subnet metrics"""
    return {
        "subnet_count": 0,
        "subnet_tvl": 0,
        "subnet_activity": 0,
        "custom_vm_usage": 0
    }

@app.get("/metrics/security")
async def get_security_metrics():
    """Get security metrics"""
    return {
        "validator_count": 0,
        "staking_ratio": 0,
        "validator_distribution": 0,
        "slashing_events": 0,
        "audit_count": 0
    }

@app.get("/metrics/development")
async def get_development_metrics():
    """Get development activity metrics"""
    return {
        "github_commits": 45,
        "github_stars": 8500,
        "github_forks": 1200,
        "developer_count": 25,
        "smart_contract_deployments": 0,
        "subnet_launches": 0
    }

@app.get("/metrics/user-behavior")
async def get_user_behavior_metrics():
    """Get user behavior metrics"""
    return {
        "whale_activity": 0,
        "retail_vs_institutional": 0,
        "holding_patterns": 0,
        "transaction_sizes": 0,
        "address_concentration": 0,
        "unique_addresses_50_blocks": 0
    }

@app.get("/metrics/competitive")
async def get_competitive_analysis():
    """Get competitive analysis metrics"""
    return {
        "market_share": 0,
        "performance_vs_competitors": 0,
        "ecosystem_growth": 0,
        "developer_adoption": 0
    }

@app.get("/metrics/technical")
async def get_technical_infrastructure():
    """Get technical infrastructure metrics"""
    return {
        "rpc_performance": 0,
        "node_distribution": 0,
        "network_uptime": 0,
        "upgrade_history": 0,
        "interoperability_score": 0
    }

@app.get("/metrics/risk")
async def get_risk_assessment():
    """Get risk assessment metrics"""
    return {
        "centralization_risks": 0,
        "technical_risks": 0,
        "regulatory_risks": 0,
        "market_risks": 0,
        "competition_risks": 0
    }

@app.get("/metrics/macro")
async def get_macro_factors():
    """Get macro-economic factors"""
    return {
        "market_conditions": 0,
        "institutional_adoption": 0,
        "regulatory_environment": 0,
        "economic_indicators": 0
    }

@app.get("/metrics/ecosystem")
async def get_ecosystem_health():
    """Get ecosystem health metrics"""
    return {
        "community_growth": 0,
        "media_coverage": 0,
        "partnership_quality": 0,
        "developer_experience": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
