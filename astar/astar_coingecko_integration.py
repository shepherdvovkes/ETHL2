#!/usr/bin/env python3
"""
Astar CoinGecko API Integration
==============================

Integrates CoinGecko API to fetch real-time market data for Astar (ASTR) token.
Provides price, volume, market cap, and other market metrics.

Features:
- Real-time price data
- 24h volume and price changes
- Market capitalization
- Historical price data
- Price volatility metrics
- Rate limiting and error handling
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import time
import warnings
warnings.filterwarnings('ignore')

class AstarCoinGeckoIntegration:
    """CoinGecko API integration for Astar market data"""
    
    def __init__(self, api_key: str = None):
        # CoinGecko API configuration
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key  # Optional API key for higher rate limits
        self.session = None
        
        # Astar token information
        self.token_info = {
            "id": "astar",  # CoinGecko token ID
            "symbol": "astr",
            "name": "Astar",
            "contract_address": "0x9a4bD57cb5ceD3D829C40D3D3dB9695CdCd7bD4e",  # Astar EVM address
            "chain_id": 592
        }
        
        # Rate limiting settings
        self.rate_limit_delay = 1.0  # 1 second between requests (free tier)
        self.last_request_time = 0
        
        # Cache settings
        self.cache_duration = 300  # 5 minutes cache
        self.price_cache = {}
        
    async def __aenter__(self):
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def _make_api_call(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API call to CoinGecko with rate limiting"""
        await self._rate_limit()
        
        if params is None:
            params = {}
        
        # Add API key if available
        if self.api_key:
            params['x_cg_demo_api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limit exceeded, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self._make_api_call(endpoint, params)
                else:
                    logger.error(f"API call failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error making API call: {e}")
            return {}
    
    async def get_current_price_data(self) -> Dict:
        """Get current price data for Astar"""
        logger.info("Fetching current Astar price data from CoinGecko...")
        
        # Check cache first
        cache_key = "current_price"
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                logger.info("Using cached price data")
                return cached_data
        
        params = {
            "ids": self.token_info["id"],
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
            "include_market_cap": "true",
            "include_last_updated_at": "true"
        }
        
        data = await self._make_api_call("simple/price", params)
        
        if data and self.token_info["id"] in data:
            token_data = data[self.token_info["id"]]
            
            price_data = {
                "timestamp": datetime.utcnow(),
                "price_usd": token_data.get("usd", 0.0),
                "market_cap_usd": token_data.get("usd_market_cap", 0.0),
                "volume_24h_usd": token_data.get("usd_24h_vol", 0.0),
                "price_change_24h": token_data.get("usd_24h_change", 0.0),
                "last_updated": datetime.fromtimestamp(token_data.get("last_updated_at", 0))
            }
            
            # Cache the data
            self.price_cache[cache_key] = (price_data, time.time())
            
            logger.success(f"Price data fetched: ${price_data['price_usd']:.4f}")
            return price_data
        else:
            logger.warning("No price data received from CoinGecko")
            return {}
    
    async def get_detailed_market_data(self) -> Dict:
        """Get detailed market data including additional metrics"""
        logger.info("Fetching detailed Astar market data...")
        
        # Get current price data
        price_data = await self.get_current_price_data()
        if not price_data:
            return {}
        
        # Get additional market data
        params = {
            "ids": self.token_info["id"],
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_7d_change": "true",
            "include_14d_change": "true",
            "include_30d_change": "true",
            "include_last_updated_at": "true"
        }
        
        data = await self._make_api_call("simple/price", params)
        
        if data and self.token_info["id"] in data:
            token_data = data[self.token_info["id"]]
            
            detailed_data = {
                **price_data,  # Include basic price data
                "price_change_7d": token_data.get("usd_7d_change", 0.0),
                "price_change_14d": token_data.get("usd_14d_change", 0.0),
                "price_change_30d": token_data.get("usd_30d_change", 0.0),
                "price_volatility_24h": abs(token_data.get("usd_24h_change", 0.0)),
                "market_cap_rank": await self._get_market_cap_rank()
            }
            
            logger.success("Detailed market data fetched successfully")
            return detailed_data
        else:
            return price_data
    
    async def _get_market_cap_rank(self) -> int:
        """Get market cap rank for Astar"""
        params = {
            "ids": self.token_info["id"],
            "vs_currencies": "usd",
            "include_market_cap": "true"
        }
        
        data = await self._make_api_call("simple/price", params)
        
        if data and self.token_info["id"] in data:
            # For market cap rank, we need to use a different endpoint
            # This is a simplified approach - in practice, you might need to
            # call the coins/markets endpoint and find the rank
            return 0  # Placeholder - would need additional API call
        
        return 0
    
    async def get_historical_price_data(self, days: int = 7) -> List[Dict]:
        """Get historical price data for specified days"""
        logger.info(f"Fetching {days} days of historical Astar price data...")
        
        params = {
            "id": self.token_info["id"],
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily" if days > 1 else "hourly"
        }
        
        data = await self._make_api_call(f"coins/{self.token_info['id']}/market_chart", params)
        
        if data and "prices" in data:
            historical_data = []
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            market_caps = data.get("market_caps", [])
            
            for i, price_point in enumerate(prices):
                timestamp = datetime.fromtimestamp(price_point[0] / 1000)
                price = price_point[1]
                volume = volumes[i][1] if i < len(volumes) else 0
                market_cap = market_caps[i][1] if i < len(market_caps) else 0
                
                historical_data.append({
                    "timestamp": timestamp,
                    "price_usd": price,
                    "volume_24h_usd": volume,
                    "market_cap_usd": market_cap
                })
            
            logger.success(f"Fetched {len(historical_data)} historical price points")
            return historical_data
        else:
            logger.warning("No historical data received")
            return []
    
    async def get_token_info(self) -> Dict:
        """Get comprehensive token information"""
        logger.info("Fetching Astar token information...")
        
        data = await self._make_api_call(f"coins/{self.token_info['id']}")
        
        if data:
            token_info = {
                "id": data.get("id", ""),
                "symbol": data.get("symbol", ""),
                "name": data.get("name", ""),
                "description": data.get("description", {}).get("en", ""),
                "market_cap_rank": data.get("market_cap_rank", 0),
                "total_supply": data.get("market_data", {}).get("total_supply", 0),
                "circulating_supply": data.get("market_data", {}).get("circulating_supply", 0),
                "max_supply": data.get("market_data", {}).get("max_supply", 0),
                "last_updated": data.get("last_updated", "")
            }
            
            logger.success("Token information fetched successfully")
            return token_info
        else:
            logger.warning("No token information received")
            return {}
    
    def calculate_price_metrics(self, price_data: Dict, historical_data: List[Dict] = None) -> Dict:
        """Calculate additional price metrics"""
        if not price_data:
            return {}
        
        metrics = {
            "price_usd": price_data.get("price_usd", 0.0),
            "market_cap_usd": price_data.get("market_cap_usd", 0.0),
            "volume_24h_usd": price_data.get("volume_24h_usd", 0.0),
            "price_change_24h": price_data.get("price_change_24h", 0.0),
            "price_change_7d": price_data.get("price_change_7d", 0.0),
            "price_volatility_24h": price_data.get("price_volatility_24h", 0.0)
        }
        
        # Calculate additional metrics if historical data is available
        if historical_data and len(historical_data) > 1:
            prices = [point["price_usd"] for point in historical_data]
            
            # Calculate price momentum (rate of change)
            if len(prices) >= 2:
                price_momentum = (prices[-1] - prices[0]) / prices[0] * 100
                metrics["price_momentum"] = price_momentum
            
            # Calculate volatility (standard deviation of price changes)
            if len(prices) > 1:
                price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                volatility = np.std(price_changes) * 100 if price_changes else 0
                metrics["price_volatility"] = volatility
            
            # Calculate volume trend
            volumes = [point["volume_24h_usd"] for point in historical_data if point["volume_24h_usd"]]
            if len(volumes) >= 2:
                volume_trend = (volumes[-1] - volumes[0]) / volumes[0] * 100 if volumes[0] > 0 else 0
                metrics["volume_trend"] = volume_trend
        
        return metrics
    
    async def get_comprehensive_market_data(self) -> Dict:
        """Get comprehensive market data combining current and historical data"""
        logger.info("Fetching comprehensive Astar market data...")
        
        try:
            # Get current detailed market data
            current_data = await self.get_detailed_market_data()
            if not current_data:
                logger.error("Failed to fetch current market data")
                return {}
            
            # Get historical data for the last 7 days
            historical_data = await self.get_historical_price_data(days=7)
            
            # Calculate additional metrics
            comprehensive_data = self.calculate_price_metrics(current_data, historical_data)
            
            # Add metadata
            comprehensive_data.update({
                "timestamp": datetime.utcnow(),
                "data_source": "coingecko",
                "token_id": self.token_info["id"],
                "historical_points": len(historical_data),
                "last_updated": current_data.get("last_updated", datetime.utcnow())
            })
            
            logger.success("Comprehensive market data fetched successfully")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive market data: {e}")
            return {}
    
    def save_market_data(self, market_data: Dict, db_path: str = "astar_market_data.db"):
        """Save market data to SQLite database"""
        if not market_data:
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS astar_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    price_usd REAL,
                    market_cap_usd REAL,
                    volume_24h_usd REAL,
                    price_change_24h REAL,
                    price_change_7d REAL,
                    price_change_14d REAL,
                    price_change_30d REAL,
                    price_volatility_24h REAL,
                    price_volatility REAL,
                    price_momentum REAL,
                    volume_trend REAL,
                    market_cap_rank INTEGER,
                    data_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert market data
            cursor.execute('''
                INSERT INTO astar_market_data 
                (timestamp, price_usd, market_cap_usd, volume_24h_usd, price_change_24h,
                 price_change_7d, price_change_14d, price_change_30d, price_volatility_24h,
                 price_volatility, price_momentum, volume_trend, market_cap_rank, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.get("timestamp"),
                market_data.get("price_usd", 0.0),
                market_data.get("market_cap_usd", 0.0),
                market_data.get("volume_24h_usd", 0.0),
                market_data.get("price_change_24h", 0.0),
                market_data.get("price_change_7d", 0.0),
                market_data.get("price_change_14d", 0.0),
                market_data.get("price_change_30d", 0.0),
                market_data.get("price_volatility_24h", 0.0),
                market_data.get("price_volatility", 0.0),
                market_data.get("price_momentum", 0.0),
                market_data.get("volume_trend", 0.0),
                market_data.get("market_cap_rank", 0),
                market_data.get("data_source", "coingecko")
            ))
            
            conn.commit()
            logger.success("Market data saved to database")
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            conn.rollback()
        finally:
            conn.close()

async def main():
    """Main function for testing CoinGecko integration"""
    async with AstarCoinGeckoIntegration() as integration:
        # Get comprehensive market data
        market_data = await integration.get_comprehensive_market_data()
        
        if market_data:
            print("\n🎯 Astar Market Data from CoinGecko:")
            print("=" * 50)
            print(f"💰 Price (USD): ${market_data.get('price_usd', 0):.4f}")
            print(f"📊 Market Cap: ${market_data.get('market_cap_usd', 0):,.0f}")
            print(f"📈 24h Volume: ${market_data.get('volume_24h_usd', 0):,.0f}")
            print(f"📉 24h Change: {market_data.get('price_change_24h', 0):.2f}%")
            print(f"📉 7d Change: {market_data.get('price_change_7d', 0):.2f}%")
            print(f"📊 Volatility: {market_data.get('price_volatility_24h', 0):.2f}%")
            print(f"⏰ Last Updated: {market_data.get('timestamp')}")
            
            # Save to database
            integration.save_market_data(market_data)
        else:
            print("❌ Failed to fetch market data")

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run the integration
    asyncio.run(main())
