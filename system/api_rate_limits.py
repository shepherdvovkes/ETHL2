#!/usr/bin/env python3
"""
API Rate Limiting Configuration
Limits the frequency of external API calls to prevent rate limiting
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import asyncio
from loguru import logger

class APIRateLimiter:
    """Rate limiter for external API calls"""
    
    def __init__(self):
        # Track last call times for each API
        self.last_calls = {
            'coingecko': None,
            'github': None,
            'quicknode': None
        }
        
        # Minimum intervals between calls (in seconds)
        self.min_intervals = {
            'coingecko': 300,  # 5 minutes
            'github': 600,     # 10 minutes  
            'quicknode': 30    # 30 seconds
        }
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = {
            'coingecko': 300,  # 5 minutes
            'github': 600,     # 10 minutes
            'quicknode': 60    # 1 minute
        }
    
    def can_make_call(self, api_name: str) -> bool:
        """Check if enough time has passed since last call"""
        if api_name not in self.last_calls:
            return True
            
        last_call = self.last_calls[api_name]
        if last_call is None:
            return True
            
        min_interval = self.min_intervals.get(api_name, 60)
        time_since_last = (datetime.utcnow() - last_call).total_seconds()
        
        return time_since_last >= min_interval
    
    def record_call(self, api_name: str):
        """Record that an API call was made"""
        self.last_calls[api_name] = datetime.utcnow()
    
    def get_cached_data(self, api_name: str, key: str) -> Any:
        """Get cached data if available and not expired"""
        cache_key = f"{api_name}_{key}"
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            cache_duration = self.cache_duration.get(api_name, 300)
            
            if (datetime.utcnow() - timestamp).total_seconds() < cache_duration:
                logger.info(f"Using cached data for {api_name}:{key}")
                return data
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    def cache_data(self, api_name: str, key: str, data: Any):
        """Cache API response data"""
        cache_key = f"{api_name}_{key}"
        self.cache[cache_key] = (data, datetime.utcnow())
        logger.info(f"Cached data for {api_name}:{key}")
    
    async def make_rate_limited_call(self, api_name: str, key: str, call_func, *args, **kwargs):
        """Make an API call with rate limiting and caching"""
        # Check cache first
        cached_data = self.get_cached_data(api_name, key)
        if cached_data is not None:
            return cached_data
        
        # Check if we can make a new call
        if not self.can_make_call(api_name):
            logger.warning(f"Rate limit reached for {api_name}, using fallback data")
            return self.get_fallback_data(api_name, key)
        
        try:
            # Make the API call
            result = await call_func(*args, **kwargs)
            if result:
                self.record_call(api_name)
                self.cache_data(api_name, key, result)
                return result
            else:
                logger.warning(f"Empty response from {api_name}, using fallback")
                return self.get_fallback_data(api_name, key)
                
        except Exception as e:
            logger.error(f"API call failed for {api_name}: {e}")
            return self.get_fallback_data(api_name, key)
    
    def get_fallback_data(self, api_name: str, key: str) -> Dict[str, Any]:
        """Get fallback data when API calls fail or are rate limited"""
        fallback_data = {
            'coingecko': {
                'avax_market_data': {
                    'market_cap': {'usd': 13000000000},
                    'total_volume': {'usd': 1350000000},
                    'current_price': {'usd': 30.8},
                    'circulating_supply': 422276596,
                    'total_supply': 720000000,
                    'price_change_percentage_24h': 2.5,
                    'price_change_percentage_7d': -5.2,
                    'price_change_percentage_30d': 15.8
                }
            },
            'github': {
                'avalanche_repo': {
                    'stargazers_count': 8500,
                    'forks_count': 1200,
                    'commits_count': 45,
                    'contributors_count': 25
                }
            },
            'quicknode': {
                'network_data': {
                    'block_time': 2.0,
                    'gas_price': 0.83,
                    'current_block': 68668393,
                    'transaction_throughput': 4500
                }
            }
        }
        
        return fallback_data.get(api_name, {}).get(key, {})

# Global rate limiter instance
rate_limiter = APIRateLimiter()

# Collection intervals (in seconds) - much longer to reduce API calls
COLLECTION_INTERVALS = {
    'network_performance': 300,    # 5 minutes (was 30s)
    'economic_data': 1800,         # 30 minutes (was 60s)
    'defi_metrics': 3600,          # 1 hour (was 120s)
    'subnet_data': 3600,           # 1 hour (was 300s)
    'security_status': 3600,       # 1 hour (was 600s)
    'development_activity': 7200,  # 2 hours (was 1800s)
    'user_behavior': 1800,         # 30 minutes (was 300s)
    'competitive_position': 14400, # 4 hours (was 3600s)
    'technical_health': 1800,      # 30 minutes (was 60s)
    'risk_indicators': 7200,       # 2 hours (was 1800s)
    'macro_environment': 7200,     # 2 hours (was 1800s)
    'ecosystem_health': 14400      # 4 hours (was 3600s)
}

# API call limits per hour
API_CALL_LIMITS = {
    'coingecko': 10,    # Max 10 calls per hour
    'github': 5,        # Max 5 calls per hour
    'quicknode': 120    # Max 120 calls per hour (2 per minute)
}
