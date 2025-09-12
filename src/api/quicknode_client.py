import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Any
from loguru import logger
from config.settings import settings

class QuickNodeClient:
    """Client for QuickNode API integration (Polygon)"""
    
    def __init__(self):
        self.api_key = settings.QUICKNODE_API_KEY
        self.http_endpoint = settings.QUICKNODE_HTTP_ENDPOINT
        self.wss_endpoint = settings.QUICKNODE_WSS_ENDPOINT
        self.session = None
        self.ws = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.ws:
            await self.ws.close()
    
    async def _make_request(
        self, 
        method: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to QuickNode API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        default_headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            async with self.session.request(
                method, self.http_endpoint, json=data, headers=default_headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"QuickNode API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"QuickNode API request failed: {e}")
            return {}
    
    async def get_block_number(self) -> int:
        """Get latest block number on Polygon"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        })
        
        if result and "result" in result:
            return int(result["result"], 16)
        return 0
    
    async def get_block_by_number(self, block_number: int) -> Dict[str, Any]:
        """Get block by number"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(block_number), True],
            "id": 1
        })
        
        if result and "result" in result:
            return result["result"]
        return {}
    
    async def get_transaction_count(
        self, 
        address: str
    ) -> int:
        """Get transaction count for address"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_getTransactionCount",
            "params": [address, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            return int(result["result"], 16)
        return 0
    
    async def get_balance(
        self, 
        address: str
    ) -> float:
        """Get balance for address in MATIC"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            wei_balance = int(result["result"], 16)
            return wei_balance / 10**18  # Convert wei to MATIC
        return 0.0
    
    async def get_contract_code(
        self, 
        address: str
    ) -> str:
        """Get contract code for address"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_getCode",
            "params": [address, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            return result["result"]
        return ""
    
    async def get_logs(
        self, 
        from_block: int, 
        to_block: int, 
        address: Optional[str] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get event logs"""
        params = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block)
        }
        
        if address:
            params["address"] = address
        if topics:
            params["topics"] = topics
        
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [params],
            "id": 1
        })
        
        if result and "result" in result:
            return result["result"]
        return []
    
    async def get_token_balance(
        self, 
        contract_address: str, 
        wallet_address: str,
        decimals: int = 18
    ) -> float:
        """Get ERC-20 token balance"""
        # ERC-20 balanceOf function call
        data = "0x70a08231" + wallet_address[2:].zfill(64)
        
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": contract_address,
                "data": data
            }, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            balance_hex = result["result"]
            if balance_hex == "0x":
                return 0.0
            return int(balance_hex, 16) / 10**decimals
        return 0.0
    
    async def get_token_info(
        self, 
        contract_address: str
    ) -> Dict[str, Any]:
        """Get ERC-20 token information (name, symbol, decimals)"""
        token_info = {}
        
        # Get token name
        name_data = "0x06fdde03"  # name() function
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": contract_address,
                "data": name_data
            }, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            # Decode string from hex
            name_hex = result["result"][2:]  # Remove 0x
            if len(name_hex) > 64:  # Has string data
                # Skip first 64 chars (offset) and decode the rest
                string_hex = name_hex[64:]
                try:
                    token_info["name"] = bytes.fromhex(string_hex).decode('utf-8').rstrip('\x00')
                except:
                    token_info["name"] = "Unknown"
        
        # Get token symbol
        symbol_data = "0x95d89b41"  # symbol() function
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": contract_address,
                "data": symbol_data
            }, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            symbol_hex = result["result"][2:]
            if len(symbol_hex) > 64:
                string_hex = symbol_hex[64:]
                try:
                    token_info["symbol"] = bytes.fromhex(string_hex).decode('utf-8').rstrip('\x00')
                except:
                    token_info["symbol"] = "Unknown"
        
        # Get token decimals
        decimals_data = "0x313ce567"  # decimals() function
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": contract_address,
                "data": decimals_data
            }, "latest"],
            "id": 1
        })
        
        if result and "result" in result:
            decimals_hex = result["result"]
            if decimals_hex != "0x":
                token_info["decimals"] = int(decimals_hex, 16)
            else:
                token_info["decimals"] = 18  # Default
        
        return token_info
    
    async def get_active_addresses(
        self, 
        from_block: int, 
        to_block: int
    ) -> int:
        """Get number of unique active addresses in block range"""
        logs = await self.get_logs(from_block, to_block)
        unique_addresses = set()
        
        for log in logs:
            if "address" in log:
                unique_addresses.add(log["address"])
            if "topics" in log and len(log["topics"]) > 1:
                # Extract addresses from topics (first 20 bytes)
                for topic in log["topics"][1:]:
                    if len(topic) >= 42:  # 0x + 40 hex chars
                        addr = "0x" + topic[-40:]
                        unique_addresses.add(addr)
        
        return len(unique_addresses)
    
    async def get_transaction_volume(
        self, 
        from_block: int, 
        to_block: int
    ) -> float:
        """Get total transaction volume in MATIC"""
        logs = await self.get_logs(from_block, to_block)
        total_volume = 0.0
        
        for log in logs:
            if "data" in log and len(log["data"]) > 2:
                try:
                    # Parse value from data (simplified)
                    value_hex = log["data"][2:]
                    if len(value_hex) >= 64:
                        value = int(value_hex[:64], 16)
                        total_volume += value / 10**18
                except:
                    continue
        
        return total_volume
    
    async def get_gas_price(self) -> float:
        """Get current gas price in Gwei"""
        result = await self._make_request("POST", {
            "jsonrpc": "2.0",
            "method": "eth_gasPrice",
            "params": [],
            "id": 1
        })
        
        if result and "result" in result:
            gas_price_wei = int(result["result"], 16)
            return gas_price_wei / 10**9  # Convert wei to Gwei
        return 0.0
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        current_block = await self.get_block_number()
        gas_price = await self.get_gas_price()
        
        # Get block info
        block_info = await self.get_block_by_number(current_block)
        
        stats = {
            "current_block": current_block,
            "gas_price_gwei": gas_price,
            "block_timestamp": block_info.get("timestamp", "0x0"),
            "block_size": len(block_info.get("transactions", [])),
            "network_utilization": 0.0  # Would need more data to calculate
        }
        
        return stats
    
    async def get_contract_interactions(
        self, 
        contract_address: str, 
        from_block: int, 
        to_block: int
    ) -> List[Dict]:
        """Get all interactions with a specific contract"""
        logs = await self.get_logs(from_block, to_block, address=contract_address)
        
        interactions = []
        for log in logs:
            interaction = {
                "transaction_hash": log.get("transactionHash"),
                "block_number": int(log.get("blockNumber", "0x0"), 16),
                "log_index": int(log.get("logIndex", "0x0"), 16),
                "topics": log.get("topics", []),
                "data": log.get("data", ""),
                "address": log.get("address")
            }
            interactions.append(interaction)
        
        return interactions
    
    async def get_token_transfers(
        self, 
        token_contract: str, 
        from_block: int, 
        to_block: int
    ) -> List[Dict]:
        """Get ERC-20 token transfers"""
        # ERC-20 Transfer event signature: Transfer(address,address,uint256)
        transfer_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        
        logs = await self.get_logs(
            from_block, 
            to_block, 
            address=token_contract,
            topics=[transfer_topic]
        )
        
        transfers = []
        for log in logs:
            if len(log.get("topics", [])) >= 3:
                from_addr = "0x" + log["topics"][1][-40:]
                to_addr = "0x" + log["topics"][2][-40:]
                
                # Parse amount from data
                amount_hex = log.get("data", "0x")
                if amount_hex != "0x":
                    amount = int(amount_hex, 16)
                else:
                    amount = 0
                
                transfer = {
                    "transaction_hash": log.get("transactionHash"),
                    "block_number": int(log.get("blockNumber", "0x0"), 16),
                    "from": from_addr,
                    "to": to_addr,
                    "amount": amount,
                    "token_contract": token_contract
                }
                transfers.append(transfer)
        
        return transfers
