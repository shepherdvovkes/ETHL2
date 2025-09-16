import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from config.settings import settings

class CoinMarketCapClient:
    """Client for CoinMarketCap API integration"""
    
    def __init__(self):
        self.api_key = settings.COINMARKETCAP_API_KEY
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.session = None
        self.rate_limit_delay = 0.5  # Delay between requests
    
    async def __aenter__(self):
        headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json"
        }
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to CoinMarketCap API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit exceeded
                    logger.warning("CoinMarketCap API rate limit exceeded, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    logger.error(f"CoinMarketCap API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"CoinMarketCap API request failed: {e}")
            return {}
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global cryptocurrency metrics"""
        return await self._make_request("global-metrics/quotes/latest")
    
    async def get_crypto_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get cryptocurrency quotes for specific symbols"""
        params = {
            "symbol": ",".join(symbols),
            "convert": "USD"
        }
        return await self._make_request("cryptocurrency/quotes/latest", params)
    
    async def get_avalanche_metrics(self) -> Dict[str, Any]:
        """Get Avalanche-specific metrics"""
        try:
            # Get Avalanche (AVAX) data
            avax_data = await self.get_crypto_quotes(["AVAX"])
            
            if avax_data and "data" in avax_data and "AVAX" in avax_data["data"]:
                avax = avax_data["data"]["AVAX"]
                return {
                    "price": avax["quote"]["USD"]["price"],
                    "market_cap": avax["quote"]["USD"]["market_cap"],
                    "volume_24h": avax["quote"]["USD"]["volume_24h"],
                    "percent_change_24h": avax["quote"]["USD"]["percent_change_24h"],
                    "percent_change_7d": avax["quote"]["USD"]["percent_change_7d"],
                    "market_cap_rank": avax["cmc_rank"],
                    "circulating_supply": avax["circulating_supply"],
                    "total_supply": avax["total_supply"]
                }
            else:
                logger.warning("No Avalanche data found in CoinMarketCap response")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting Avalanche metrics from CoinMarketCap: {e}")
            return {}
    
    async def get_competitor_analysis(self) -> Dict[str, Any]:
        """Get competitor analysis data"""
        try:
            # Get data for major competitors
            competitors = ["BTC", "ETH", "SOL", "MATIC", "ARB", "OP"]
            quotes_data = await self.get_crypto_quotes(competitors)
            
            if quotes_data and "data" in quotes_data:
                competitors_data = {}
                for symbol, data in quotes_data["data"].items():
                    competitors_data[symbol] = {
                        "price": data["quote"]["USD"]["price"],
                        "market_cap": data["quote"]["USD"]["market_cap"],
                        "market_cap_rank": data["cmc_rank"],
                        "percent_change_24h": data["quote"]["USD"]["percent_change_24h"]
                    }
                
                return competitors_data
            else:
                logger.warning("No competitor data found in CoinMarketCap response")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting competitor analysis from CoinMarketCap: {e}")
            return {}
