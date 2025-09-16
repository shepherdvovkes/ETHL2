import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from config.settings import settings

class CoinGeckoClient:
    """Client for CoinGecko API integration"""
    
    def __init__(self):
        self.api_key = settings.COINGECKO_API_KEY
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://pro-api.coingecko.com/api/v3"  # Pro API
        self.session = None
        self.rate_limit_delay = 5.0  # Increased delay between requests
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self, 
        endpoint: str,
        params: Optional[Dict] = None,
        use_pro: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to CoinGecko API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.pro_url if use_pro else self.base_url}/{endpoint}"
        
        headers = {}
        if self.api_key and use_pro:
            headers["x-cg-pro-api-key"] = self.api_key
        
        if params is None:
            params = {}
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit exceeded
                    logger.warning("Rate limit exceeded, waiting 300 seconds...")
                    await asyncio.sleep(300)  # Wait 5 minutes instead of 1 minute
                    return await self._make_request(endpoint, params, use_pro)
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"CoinGecko API request failed: {e}")
            return {}
    
    async def get_coin_list(self, include_platform: bool = True) -> List[Dict]:
        """Get list of all coins"""
        params = {"include_platform": str(include_platform).lower()}
        return await self._make_request("coins/list", params)
    
    async def get_coin_markets(
        self, 
        vs_currency: str = "usd",
        category: Optional[str] = None,
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: str = "24h,7d,30d"
    ) -> List[Dict]:
        """Get coin market data"""
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": str(sparkline).lower(),
            "price_change_percentage": price_change_percentage
        }
        
        if category:
            params["category"] = category
        
        return await self._make_request("coins/markets", params)
    
    async def get_coin_data(
        self, 
        coin_id: str,
        localization: bool = False,
        tickers: bool = True,
        market_data: bool = True,
        community_data: bool = True,
        developer_data: bool = True,
        sparkline: bool = False
    ) -> Dict[str, Any]:
        """Get detailed coin data"""
        params = {
            "localization": str(localization).lower(),
            "tickers": str(tickers).lower(),
            "market_data": str(market_data).lower(),
            "community_data": str(community_data).lower(),
            "developer_data": str(developer_data).lower(),
            "sparkline": str(sparkline).lower()
        }
        
        return await self._make_request(f"coins/{coin_id}", params)
    
    async def get_coin_market_chart(
        self, 
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 7,
        interval: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get coin market chart data"""
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        if interval:
            params["interval"] = interval
        
        return await self._make_request(f"coins/{coin_id}/market_chart", params)
    
    async def get_coin_market_chart_range(
        self, 
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int,
        vs_currency: str = "usd"
    ) -> Dict[str, Any]:
        """Get coin market chart data for specific date range"""
        params = {
            "vs_currency": vs_currency,
            "from": from_timestamp,
            "to": to_timestamp
        }
        
        return await self._make_request(f"coins/{coin_id}/market_chart/range", params)
    
    async def get_coin_ohlc(
        self, 
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 7
    ) -> List[List[float]]:
        """Get coin OHLC data"""
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        return await self._make_request(f"coins/{coin_id}/ohlc", params)
    
    async def get_coin_price(
        self, 
        coin_ids: List[str],
        vs_currencies: List[str] = ["usd"],
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
        include_24hr_change: bool = True,
        include_last_updated_at: bool = True
    ) -> Dict[str, Any]:
        """Get current price data for multiple coins"""
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
            "include_market_cap": str(include_market_cap).lower(),
            "include_24hr_vol": str(include_24hr_vol).lower(),
            "include_24hr_change": str(include_24hr_change).lower(),
            "include_last_updated_at": str(include_last_updated_at).lower()
        }
        
        return await self._make_request("simple/price", params)
    
    async def get_coin_contract_data(
        self, 
        platform_id: str,
        contract_address: str
    ) -> Dict[str, Any]:
        """Get coin data by contract address"""
        return await self._make_request(f"coins/{platform_id}/contract/{contract_address}")
    
    async def get_coin_contract_market_chart(
        self, 
        platform_id: str,
        contract_address: str,
        vs_currency: str = "usd",
        days: int = 7
    ) -> Dict[str, Any]:
        """Get market chart data for contract address"""
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        return await self._make_request(
            f"coins/{platform_id}/contract/{contract_address}/market_chart", 
            params
        )
    
    async def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency data"""
        return await self._make_request("global")
    
    async def get_trending_coins(self) -> Dict[str, Any]:
        """Get trending coins"""
        return await self._make_request("search/trending")
    
    async def get_categories(self, order: str = "market_cap_desc") -> List[Dict]:
        """Get coin categories"""
        params = {"order": order}
        return await self._make_request("coins/categories", params)
    
    async def get_exchanges(
        self, 
        per_page: int = 100,
        page: int = 1
    ) -> List[Dict]:
        """Get exchange data"""
        params = {
            "per_page": per_page,
            "page": page
        }
        
        return await self._make_request("exchanges", params)
    
    async def get_exchange_rates(self) -> Dict[str, Any]:
        """Get exchange rates"""
        return await self._make_request("exchange_rates")
    
    async def get_derivatives(self) -> List[Dict]:
        """Get derivatives data"""
        return await self._make_request("derivatives")
    
    async def get_derivatives_exchanges(self) -> List[Dict]:
        """Get derivatives exchanges"""
        return await self._make_request("derivatives/exchanges")
    
    async def get_nfts(self, order: str = "market_cap_usd_desc", per_page: int = 100) -> List[Dict]:
        """Get NFT data"""
        params = {
            "order": order,
            "per_page": per_page
        }
        
        return await self._make_request("nfts/list", params)
    
    async def get_asset_platforms(self) -> List[Dict]:
        """Get asset platforms (blockchains)"""
        return await self._make_request("asset_platforms")
    
    async def get_coin_by_id(self, coin_id: str) -> Dict[str, Any]:
        """Get coin by ID with all data"""
        return await self.get_coin_data(
            coin_id,
            localization=False,
            tickers=True,
            market_data=True,
            community_data=True,
            developer_data=True,
            sparkline=False
        )
    
    async def get_historical_data(
        self, 
        coin_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> Dict[str, Any]:
        """Get historical price data"""
        return await self.get_coin_market_chart(coin_id, vs_currency, days)
    
    async def get_multiple_coin_data(
        self, 
        coin_ids: List[str],
        vs_currency: str = "usd"
    ) -> Dict[str, Any]:
        """Get data for multiple coins efficiently"""
        # Get price data
        price_data = await self.get_coin_price(
            coin_ids, 
            [vs_currency],
            include_market_cap=True,
            include_24hr_vol=True,
            include_24hr_change=True
        )
        
        # Get market data for each coin
        market_data = {}
        for coin_id in coin_ids:
            try:
                coin_data = await self.get_coin_data(
                    coin_id,
                    market_data=True,
                    community_data=True,
                    developer_data=True
                )
                market_data[coin_id] = coin_data
            except Exception as e:
                logger.error(f"Error getting data for {coin_id}: {e}")
                continue
        
        return {
            "prices": price_data,
            "market_data": market_data
        }
    
    async def search_coins(self, query: str) -> Dict[str, Any]:
        """Search for coins"""
        params = {"query": query}
        return await self._make_request("search", params)
    
    async def get_coin_status_updates(
        self, 
        coin_id: str,
        per_page: int = 100,
        page: int = 1
    ) -> Dict[str, Any]:
        """Get status updates for a coin"""
        params = {
            "per_page": per_page,
            "page": page
        }
        
        return await self._make_request(f"coins/{coin_id}/status_updates", params)
    
    async def get_coin_community_data(self, coin_id: str) -> Dict[str, Any]:
        """Get community data for a coin"""
        return await self._make_request(f"coins/{coin_id}/community_data")
    
    async def get_coin_developer_data(self, coin_id: str) -> Dict[str, Any]:
        """Get developer data for a coin"""
        return await self._make_request(f"coins/{coin_id}/developer_data")
    
    async def get_coin_tickers(
        self, 
        coin_id: str,
        exchange_ids: Optional[str] = None,
        include_exchange_logo: bool = False,
        page: int = 1,
        order: str = "trust_score_desc",
        depth: bool = True
    ) -> Dict[str, Any]:
        """Get tickers for a coin"""
        params = {
            "include_exchange_logo": str(include_exchange_logo).lower(),
            "page": page,
            "order": order,
            "depth": str(depth).lower()
        }
        
        if exchange_ids:
            params["exchange_ids"] = exchange_ids
        
        return await self._make_request(f"coins/{coin_id}/tickers", params)
