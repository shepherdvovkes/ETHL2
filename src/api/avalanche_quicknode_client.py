import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Any
from loguru import logger
from config.settings import settings

class AvalancheQuickNodeClient:
    """Client for QuickNode API integration with Avalanche network"""
    
    def __init__(self):
        self.api_key = settings.QUICKNODE_API_KEY
        self.c_chain_endpoint = settings.QUICKNODE_AVALANCHE_C_CHAIN_ENDPOINT
        self.c_chain_wss_endpoint = settings.QUICKNODE_AVALANCHE_C_CHAIN_WSS_ENDPOINT
        self.p_chain_endpoint = settings.QUICKNODE_AVALANCHE_P_CHAIN_ENDPOINT
        self.x_chain_endpoint = settings.QUICKNODE_AVALANCHE_X_CHAIN_ENDPOINT
        self.session = None
        self.ws = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.ws:
            await self.ws.close()
    
    async def _make_rpc_request(
        self, 
        endpoint: str,
        method: str, 
        params: List = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make RPC request to Avalanche via QuickNode"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        default_headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        if headers:
            default_headers.update(headers)
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": 1
        }
        
        try:
            async with self.session.post(endpoint, json=payload, headers=default_headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"QuickNode Avalanche API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"QuickNode Avalanche API request failed: {e}")
            return {}
    
    # C-Chain (EVM) Methods
    async def get_c_chain_block_number(self) -> int:
        """Get latest block number on Avalanche C-Chain"""
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_blockNumber"
        )
        
        if result and "result" in result:
            return int(result["result"], 16)
        return 0
    
    async def get_c_chain_block_by_number(self, block_number: int) -> Dict[str, Any]:
        """Get block by number on C-Chain"""
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_getBlockByNumber",
            [hex(block_number), True]
        )
        
        if result and "result" in result:
            return result["result"]
        return {}
    
    async def get_c_chain_gas_price(self) -> float:
        """Get current gas price on C-Chain in Gwei"""
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_gasPrice"
        )
        
        if result and "result" in result:
            gas_price_wei = int(result["result"], 16)
            return gas_price_wei / 10**9  # Convert wei to Gwei
        return 0.0
    
    async def get_c_chain_balance(self, address: str) -> float:
        """Get AVAX balance for address on C-Chain"""
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_getBalance",
            [address, "latest"]
        )
        
        if result and "result" in result:
            wei_balance = int(result["result"], 16)
            return wei_balance / 10**18  # Convert wei to AVAX
        return 0.0
    
    async def get_c_chain_transaction_count(self, address: str) -> int:
        """Get transaction count for address on C-Chain"""
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_getTransactionCount",
            [address, "latest"]
        )
        
        if result and "result" in result:
            return int(result["result"], 16)
        return 0
    
    async def get_c_chain_logs(
        self, 
        from_block: int, 
        to_block: int, 
        address: Optional[str] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get event logs from C-Chain"""
        params = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block)
        }
        
        if address:
            params["address"] = address
        if topics:
            params["topics"] = topics
        
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_getLogs",
            [params]
        )
        
        if result and "result" in result:
            return result["result"]
        return []
    
    # P-Chain Methods
    async def get_p_chain_validators(self) -> List[Dict]:
        """Get P-Chain validators"""
        result = await self._make_rpc_request(
            self.p_chain_endpoint,
            "platform.getCurrentValidators",
            [{"subnetID": "11111111111111111111111111111111LpoYY"}]
        )
        
        if result and "result" in result:
            return result["result"].get("validators", [])
        return []
    
    async def get_p_chain_subnets(self) -> List[Dict]:
        """Get P-Chain subnets"""
        result = await self._make_rpc_request(
            self.p_chain_endpoint,
            "platform.getSubnets"
        )
        
        if result and "result" in result:
            return result["result"].get("subnets", [])
        return []
    
    async def get_p_chain_staking_info(self) -> Dict[str, Any]:
        """Get P-Chain staking information"""
        validators = await self.get_p_chain_validators()
        
        total_stake = 0
        active_validators = 0
        
        for validator in validators:
            end_time = validator.get("endTime", 0)
            if isinstance(end_time, str):
                end_time = int(end_time, 16) if end_time.startswith("0x") else int(end_time)
            if end_time > int(asyncio.get_event_loop().time()):
                active_validators += 1
                stake_amount = validator.get("stakeAmount", 0)
                if isinstance(stake_amount, str):
                    stake_amount = int(stake_amount, 16) if stake_amount.startswith("0x") else int(stake_amount)
                total_stake += stake_amount
        
        return {
            "total_stake": total_stake,
            "active_validators": active_validators,
            "total_validators": len(validators)
        }
    
    # X-Chain Methods
    async def get_x_chain_balance(self, address: str) -> Dict[str, float]:
        """Get X-Chain balances for address"""
        result = await self._make_rpc_request(
            self.x_chain_endpoint,
            "avm.getBalance",
            [address, "AVAX"]
        )
        
        if result and "result" in result:
            balance = int(result["result"].get("balance", "0"))
            return {"AVAX": balance / 10**9}  # Convert nAVAX to AVAX
        return {"AVAX": 0.0}
    
    async def get_x_chain_asset_info(self, asset_id: str) -> Dict[str, Any]:
        """Get X-Chain asset information"""
        result = await self._make_rpc_request(
            self.x_chain_endpoint,
            "avm.getAssetDescription",
            [asset_id]
        )
        
        if result and "result" in result:
            return result["result"]
        return {}
    
    # Network Statistics
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        c_chain_block = await self.get_c_chain_block_number()
        c_chain_gas = await self.get_c_chain_gas_price()
        p_chain_staking = await self.get_p_chain_staking_info()
        
        # Get recent block info
        block_info = await self.get_c_chain_block_by_number(c_chain_block)
        
        stats = {
            "c_chain": {
                "current_block": c_chain_block,
                "gas_price_gwei": c_chain_gas,
                "block_timestamp": block_info.get("timestamp", "0x0"),
                "block_size": len(block_info.get("transactions", [])),
            },
            "p_chain": {
                "total_stake": p_chain_staking["total_stake"],
                "active_validators": p_chain_staking["active_validators"],
                "total_validators": p_chain_staking["total_validators"]
            },
            "network_utilization": 0.0  # Would need more data to calculate
        }
        
        return stats
    
    # DeFi and Token Methods
    async def get_token_balance(
        self, 
        contract_address: str, 
        wallet_address: str,
        decimals: int = 18
    ) -> float:
        """Get ERC-20 token balance on C-Chain"""
        # ERC-20 balanceOf function call
        data = "0x70a08231" + wallet_address[2:].zfill(64)
        
        result = await self._make_rpc_request(
            self.c_chain_endpoint,
            "eth_call",
            [{
                "to": contract_address,
                "data": data
            }, "latest"]
        )
        
        if result and "result" in result:
            balance_hex = result["result"]
            if balance_hex == "0x":
                return 0.0
            return int(balance_hex, 16) / 10**decimals
        return 0.0
    
    async def get_token_transfers(
        self, 
        token_contract: str, 
        from_block: int, 
        to_block: int
    ) -> List[Dict]:
        """Get ERC-20 token transfers on C-Chain"""
        # ERC-20 Transfer event signature: Transfer(address,address,uint256)
        transfer_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        
        logs = await self.get_c_chain_logs(
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
    
    async def get_active_addresses(
        self, 
        from_block: int, 
        to_block: int
    ) -> int:
        """Get number of unique active addresses in block range"""
        logs = await self.get_c_chain_logs(from_block, to_block)
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
        """Get total transaction volume in AVAX"""
        logs = await self.get_c_chain_logs(from_block, to_block)
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
