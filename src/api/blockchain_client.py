import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
from config.settings import settings
from enum import Enum

class BlockchainType(str, Enum):
    """Supported blockchain types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    COSMOS = "cosmos"
    FANTOM = "fantom"
    TRON = "tron"
    LITECOIN = "litecoin"
    CHAINLINK = "chainlink"
    NEAR = "near"
    ALGORAND = "algorand"
    TEZOS = "tezos"
    EOS = "eos"
    WAVES = "waves"
    NEO = "neo"
    VECHAIN = "vechain"
    HEDERA = "hedera"
    ELROND = "elrond"
    HARMONY = "harmony"
    KLAYTN = "klaytn"
    CRONOS = "cronos"
    GNOSIS = "gnosis"
    CELO = "celo"
    MOONBEAM = "moonbeam"
    AURORA = "aurora"
    EVMOS = "evmos"
    KAVA = "kava"
    INJECTIVE = "injective"
    OSMOSIS = "osmosis"
    JUNO = "juno"
    SECRET = "secret"
    TERRA_CLASSIC = "terra-classic"
    TERRA_2 = "terra-2"
    APTOS = "aptos"
    SUI = "sui"
    SEI = "sei"
    BASE = "base"
    LINEA = "linea"
    SCROLL = "scroll"
    MANTLE = "mantle"
    BLAST = "blast"

class BlockchainClient:
    """Universal blockchain client supporting multiple networks"""
    
    def __init__(self, blockchain_type: BlockchainType):
        self.blockchain_type = blockchain_type
        self.session = None
        self.rate_limit_delay = 0.1  # Delay between requests
        
        # Blockchain-specific configurations
        self.config = self._get_blockchain_config(blockchain_type)
    
    def _get_blockchain_config(self, blockchain_type: BlockchainType) -> Dict[str, Any]:
        """Get blockchain-specific configuration"""
        configs = {
            BlockchainType.ETHEREUM: {
                "name": "Ethereum",
                "symbol": "ETH",
                "chain_id": 1,
                "rpc_url": settings.QUICKNODE_HTTP_ENDPOINT if "ethereum" in settings.QUICKNODE_HTTP_ENDPOINT else "",
                "explorer_url": "https://etherscan.io",
                "block_time": 12,  # seconds
                "native_token": "ETH",
                "decimals": 18
            },
            BlockchainType.POLYGON: {
                "name": "Polygon",
                "symbol": "MATIC",
                "chain_id": 137,
                "rpc_url": settings.QUICKNODE_HTTP_ENDPOINT,
                "explorer_url": "https://polygonscan.com",
                "block_time": 2,  # seconds
                "native_token": "MATIC",
                "decimals": 18
            },
            BlockchainType.BSC: {
                "name": "Binance Smart Chain",
                "symbol": "BNB",
                "chain_id": 56,
                "rpc_url": "https://bsc-dataseed.binance.org",
                "explorer_url": "https://bscscan.com",
                "block_time": 3,  # seconds
                "native_token": "BNB",
                "decimals": 18
            },
            BlockchainType.ARBITRUM: {
                "name": "Arbitrum One",
                "symbol": "ETH",
                "chain_id": 42161,
                "rpc_url": "https://arb1.arbitrum.io/rpc",
                "explorer_url": "https://arbiscan.io",
                "block_time": 0.25,  # seconds
                "native_token": "ETH",
                "decimals": 18
            },
            BlockchainType.OPTIMISM: {
                "name": "Optimism",
                "symbol": "ETH",
                "chain_id": 10,
                "rpc_url": "https://mainnet.optimism.io",
                "explorer_url": "https://optimistic.etherscan.io",
                "block_time": 2,  # seconds
                "native_token": "ETH",
                "decimals": 18
            },
            BlockchainType.AVALANCHE: {
                "name": "Avalanche",
                "symbol": "AVAX",
                "chain_id": 43114,
                "rpc_url": "https://api.avax.network/ext/bc/C/rpc",
                "explorer_url": "https://snowtrace.io",
                "block_time": 2,  # seconds
                "native_token": "AVAX",
                "decimals": 18
            },
            BlockchainType.SOLANA: {
                "name": "Solana",
                "symbol": "SOL",
                "chain_id": 101,
                "rpc_url": "https://api.mainnet-beta.solana.com",
                "explorer_url": "https://explorer.solana.com",
                "block_time": 0.4,  # seconds
                "native_token": "SOL",
                "decimals": 9
            }
        }
        
        return configs.get(blockchain_type, {
            "name": blockchain_type.value.title(),
            "symbol": blockchain_type.value.upper(),
            "chain_id": 0,
            "rpc_url": "",
            "explorer_url": "",
            "block_time": 10,
            "native_token": blockchain_type.value.upper(),
            "decimals": 18
        })
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self, 
        method: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to blockchain RPC"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        if not self.config.get("rpc_url"):
            logger.warning(f"No RPC URL configured for {self.blockchain_type}")
            return {}
        
        default_headers = {
            "Content-Type": "application/json"
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.request(
                method, self.config["rpc_url"], json=data, headers=default_headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Blockchain API error for {self.blockchain_type}: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Blockchain API request failed for {self.blockchain_type}: {e}")
            return {}
    
    async def get_block_number(self) -> int:
        """Get latest block number"""
        if self.blockchain_type == BlockchainType.SOLANA:
            # Solana uses different RPC methods
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "getSlot",
                "params": [],
                "id": 1
            })
            if result and "result" in result:
                return result["result"]
        else:
            # EVM-compatible chains
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
        if self.blockchain_type == BlockchainType.SOLANA:
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "getBlock",
                "params": [block_number, {"encoding": "json"}],
                "id": 1
            })
        else:
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(block_number), True],
                "id": 1
            })
        
        if result and "result" in result:
            return result["result"]
        return {}
    
    async def get_balance(self, address: str) -> float:
        """Get balance for address"""
        if self.blockchain_type == BlockchainType.SOLANA:
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "getBalance",
                "params": [address],
                "id": 1
            })
            if result and "result" in result:
                lamports = result["result"]["value"]
                return lamports / 10**9  # Convert lamports to SOL
        else:
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [address, "latest"],
                "id": 1
            })
            if result and "result" in result:
                wei_balance = int(result["result"], 16)
                return wei_balance / 10**self.config["decimals"]
        
        return 0.0
    
    async def get_transaction_count(self, address: str) -> int:
        """Get transaction count for address"""
        if self.blockchain_type == BlockchainType.SOLANA:
            # Solana doesn't have transaction count in the same way
            return 0
        else:
            result = await self._make_request("POST", {
                "jsonrpc": "2.0",
                "method": "eth_getTransactionCount",
                "params": [address, "latest"],
                "id": 1
            })
            if result and "result" in result:
                return int(result["result"], 16)
        
        return 0
    
    async def get_gas_price(self) -> float:
        """Get current gas price"""
        if self.blockchain_type == BlockchainType.SOLANA:
            # Solana doesn't use gas prices
            return 0.0
        else:
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
            "blockchain": self.blockchain_type.value,
            "current_block": current_block,
            "gas_price_gwei": gas_price,
            "block_time": self.config["block_time"],
            "native_token": self.config["native_token"],
            "chain_id": self.config["chain_id"]
        }
        
        if block_info:
            if self.blockchain_type == BlockchainType.SOLANA:
                stats.update({
                    "block_timestamp": block_info.get("blockTime"),
                    "block_size": len(block_info.get("transactions", [])),
                    "parent_slot": block_info.get("parentSlot")
                })
            else:
                stats.update({
                    "block_timestamp": block_info.get("timestamp", "0x0"),
                    "block_size": len(block_info.get("transactions", [])),
                    "gas_limit": int(block_info.get("gasLimit", "0x0"), 16),
                    "gas_used": int(block_info.get("gasUsed", "0x0"), 16)
                })
        
        return stats
    
    async def get_token_balance(
        self, 
        contract_address: str, 
        wallet_address: str,
        decimals: int = 18
    ) -> float:
        """Get ERC-20 token balance (EVM chains only)"""
        if self.blockchain_type == BlockchainType.SOLANA:
            # Solana uses SPL tokens, different implementation needed
            logger.warning("Token balance not implemented for Solana yet")
            return 0.0
        
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
    
    async def get_logs(
        self, 
        from_block: int, 
        to_block: int, 
        address: Optional[str] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get event logs (EVM chains only)"""
        if self.blockchain_type == BlockchainType.SOLANA:
            logger.warning("Logs not available for Solana")
            return []
        
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
    
    async def get_active_addresses(
        self, 
        from_block: int, 
        to_block: int
    ) -> int:
        """Get number of unique active addresses in block range"""
        if self.blockchain_type == BlockchainType.SOLANA:
            # Solana implementation would be different
            logger.warning("Active addresses not implemented for Solana yet")
            return 0
        
        logs = await self.get_logs(from_block, to_block)
        unique_addresses = set()
        
        for log in logs:
            if "address" in log:
                unique_addresses.add(log["address"])
            if "topics" in log and len(log["topics"]) > 1:
                for topic in log["topics"][1:]:
                    if len(topic) >= 42:
                        addr = "0x" + topic[-40:]
                        unique_addresses.add(addr)
        
        return len(unique_addresses)
    
    async def get_transaction_volume(
        self, 
        from_block: int, 
        to_block: int
    ) -> float:
        """Get total transaction volume"""
        if self.blockchain_type == BlockchainType.SOLANA:
            logger.warning("Transaction volume not implemented for Solana yet")
            return 0.0
        
        logs = await self.get_logs(from_block, to_block)
        total_volume = 0.0
        
        for log in logs:
            if "data" in log and len(log["data"]) > 2:
                try:
                    value_hex = log["data"][2:]
                    if len(value_hex) >= 64:
                        value = int(value_hex[:64], 16)
                        total_volume += value / 10**self.config["decimals"]
                except:
                    continue
        
        return total_volume
    
    async def get_contract_interactions(
        self, 
        contract_address: str, 
        from_block: int, 
        to_block: int
    ) -> List[Dict]:
        """Get all interactions with a specific contract"""
        if self.blockchain_type == BlockchainType.SOLANA:
            logger.warning("Contract interactions not implemented for Solana yet")
            return []
        
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
        """Get ERC-20 token transfers (EVM chains only)"""
        if self.blockchain_type == BlockchainType.SOLANA:
            logger.warning("Token transfers not implemented for Solana yet")
            return []
        
        # ERC-20 Transfer event signature
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
    
    async def get_historical_data(
        self, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Get historical blockchain data"""
        current_block = await self.get_block_number()
        if current_block == 0:
            return {}
        
        # Calculate block range
        blocks_per_day = 24 * 60 * 60 // self.config["block_time"]
        from_block = max(0, current_block - (blocks_per_day * days))
        
        # Get various metrics
        active_addresses = await self.get_active_addresses(from_block, current_block)
        transaction_volume = await self.get_transaction_volume(from_block, current_block)
        gas_price = await self.get_gas_price()
        
        return {
            "blockchain": self.blockchain_type.value,
            "period_days": days,
            "from_block": from_block,
            "to_block": current_block,
            "active_addresses": active_addresses,
            "transaction_volume": transaction_volume,
            "gas_price_gwei": gas_price,
            "blocks_analyzed": current_block - from_block,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get blockchain configuration"""
        return self.config.copy()
    
    @classmethod
    def get_supported_blockchains(cls) -> List[BlockchainType]:
        """Get list of supported blockchains"""
        return list(BlockchainType)
    
    @classmethod
    def create_client(cls, blockchain_type: Union[str, BlockchainType]) -> 'BlockchainClient':
        """Create blockchain client by type"""
        if isinstance(blockchain_type, str):
            try:
                blockchain_type = BlockchainType(blockchain_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported blockchain type: {blockchain_type}")
        
        return cls(blockchain_type)
