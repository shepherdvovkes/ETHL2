import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from config.settings import settings

class EtherscanClient:
    """Client for Etherscan API integration"""
    
    def __init__(self):
        self.api_key = settings.ETHERSCAN_API_KEY
        self.base_url = "https://api.polygonscan.com/api"  # Polygon scan API
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make HTTP request to Etherscan API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Add API key to params
        params["apikey"] = self.api_key
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Etherscan API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Etherscan API request failed: {e}")
            return {}
    
    async def get_account_balance(self, address: str) -> float:
        """Get account balance in MATIC"""
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest"
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            wei_balance = int(result.get("result", "0"))
            return wei_balance / 10**18  # Convert wei to MATIC
        return 0.0
    
    async def get_token_balance(
        self, 
        contract_address: str, 
        wallet_address: str
    ) -> float:
        """Get ERC-20 token balance"""
        params = {
            "module": "account",
            "action": "tokenbalance",
            "contractaddress": contract_address,
            "address": wallet_address,
            "tag": "latest"
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            balance = int(result.get("result", "0"))
            return balance / 10**18  # Assuming 18 decimals
        return 0.0
    
    async def get_token_info(self, contract_address: str) -> Dict[str, Any]:
        """Get token information"""
        # Get token name
        name_params = {
            "module": "token",
            "action": "tokeninfo",
            "contractaddress": contract_address
        }
        
        result = await self._make_request(name_params)
        if result and result.get("status") == "1":
            token_data = result.get("result", [])
            if token_data and len(token_data) > 0:
                return {
                    "name": token_data[0].get("tokenName", "Unknown"),
                    "symbol": token_data[0].get("symbol", "Unknown"),
                    "decimals": int(token_data[0].get("divisor", "18")),
                    "total_supply": token_data[0].get("totalSupply", "0")
                }
        
        return {}
    
    async def get_transaction_list(
        self, 
        address: str, 
        start_block: int = 0, 
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """Get transaction list for address"""
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "desc"
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return result.get("result", [])
        return []
    
    async def get_token_transfers(
        self, 
        contract_address: str, 
        address: Optional[str] = None,
        start_block: int = 0, 
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """Get ERC-20 token transfers"""
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "desc"
        }
        
        if address:
            params["address"] = address
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return result.get("result", [])
        return []
    
    async def get_contract_source_code(self, contract_address: str) -> Dict[str, Any]:
        """Get contract source code"""
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            contract_data = result.get("result", [])
            if contract_data and len(contract_data) > 0:
                return {
                    "source_code": contract_data[0].get("SourceCode", ""),
                    "abi": contract_data[0].get("ABI", ""),
                    "contract_name": contract_data[0].get("ContractName", ""),
                    "compiler_version": contract_data[0].get("CompilerVersion", ""),
                    "optimization_used": contract_data[0].get("OptimizationUsed", ""),
                    "runs": contract_data[0].get("Runs", ""),
                    "constructor_arguments": contract_data[0].get("ConstructorArguments", ""),
                    "evm_version": contract_data[0].get("EVMVersion", ""),
                    "library": contract_data[0].get("Library", ""),
                    "license_type": contract_data[0].get("LicenseType", ""),
                    "proxy": contract_data[0].get("Proxy", ""),
                    "implementation": contract_data[0].get("Implementation", ""),
                    "swarm_source": contract_data[0].get("SwarmSource", "")
                }
        return {}
    
    async def get_contract_creation(self, contract_address: str) -> Dict[str, Any]:
        """Get contract creation information"""
        params = {
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": contract_address
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            creation_data = result.get("result", [])
            if creation_data and len(creation_data) > 0:
                return {
                    "contract_address": creation_data[0].get("contractAddress", ""),
                    "contract_creator": creation_data[0].get("contractCreator", ""),
                    "tx_hash": creation_data[0].get("txHash", "")
                }
        return {}
    
    async def get_gas_price(self) -> Dict[str, Any]:
        """Get current gas price"""
        params = {
            "module": "gastracker",
            "action": "gasoracle"
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            gas_data = result.get("result", {})
            return {
                "safe_gas_price": gas_data.get("SafeGasPrice", "0"),
                "propose_gas_price": gas_data.get("ProposeGasPrice", "0"),
                "fast_gas_price": gas_data.get("FastGasPrice", "0"),
                "suggest_base_fee": gas_data.get("suggestBaseFee", "0"),
                "gas_used_ratio": gas_data.get("gasUsedRatio", "0")
            }
        return {}
    
    async def get_block_reward(self, block_number: int) -> Dict[str, Any]:
        """Get block reward information"""
        params = {
            "module": "block",
            "action": "getblockreward",
            "blockno": block_number
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            reward_data = result.get("result", {})
            return {
                "block_number": reward_data.get("blockNumber", ""),
                "time_stamp": reward_data.get("timeStamp", ""),
                "block_miner": reward_data.get("blockMiner", ""),
                "block_reward": reward_data.get("blockReward", ""),
                "uncle_inclusion_reward": reward_data.get("uncleInclusionReward", ""),
                "uncle_rewards": reward_data.get("uncleRewards", [])
            }
        return {}
    
    async def get_contract_events(
        self, 
        contract_address: str, 
        from_block: int, 
        to_block: int,
        topic0: Optional[str] = None
    ) -> List[Dict]:
        """Get contract events"""
        params = {
            "module": "logs",
            "action": "getLogs",
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": contract_address
        }
        
        if topic0:
            params["topic0"] = topic0
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return result.get("result", [])
        return []
    
    async def get_token_holders(
        self, 
        contract_address: str, 
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """Get token holders"""
        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": contract_address,
            "page": page,
            "offset": offset
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return result.get("result", [])
        return []
    
    async def get_token_supply(self, contract_address: str) -> float:
        """Get token total supply"""
        params = {
            "module": "stats",
            "action": "tokensupply",
            "contractaddress": contract_address
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            supply = int(result.get("result", "0"))
            return supply / 10**18  # Assuming 18 decimals
        return 0.0
    
    async def get_account_balance_multi(self, addresses: List[str]) -> List[Dict]:
        """Get balances for multiple addresses"""
        params = {
            "module": "account",
            "action": "balancemulti",
            "address": ",".join(addresses),
            "tag": "latest"
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return result.get("result", [])
        return []
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status"""
        params = {
            "module": "transaction",
            "action": "gettxreceiptstatus",
            "txhash": tx_hash
        }
        
        result = await self._make_request(params)
        if result and result.get("status") == "1":
            return {
                "status": result.get("result", {}).get("status", "0"),
                "tx_hash": tx_hash
            }
        return {}
    
    async def get_contract_verification_status(self, contract_address: str) -> bool:
        """Check if contract is verified"""
        source_code = await self.get_contract_source_code(contract_address)
        return bool(source_code.get("source_code"))
    
    async def analyze_contract_security(self, contract_address: str) -> Dict[str, Any]:
        """Analyze contract security (basic analysis)"""
        security_analysis = {
            "is_verified": False,
            "has_source_code": False,
            "has_abi": False,
            "compiler_version": None,
            "optimization_used": False,
            "proxy_contract": False,
            "security_score": 0
        }
        
        # Get contract source code
        contract_info = await self.get_contract_source_code(contract_address)
        
        if contract_info:
            security_analysis["is_verified"] = True
            security_analysis["has_source_code"] = bool(contract_info.get("source_code"))
            security_analysis["has_abi"] = bool(contract_info.get("abi"))
            security_analysis["compiler_version"] = contract_info.get("compiler_version")
            security_analysis["optimization_used"] = contract_info.get("optimization_used") == "1"
            security_analysis["proxy_contract"] = contract_info.get("proxy") == "1"
            
            # Calculate basic security score
            score = 0
            if security_analysis["is_verified"]:
                score += 30
            if security_analysis["has_source_code"]:
                score += 25
            if security_analysis["has_abi"]:
                score += 20
            if security_analysis["optimization_used"]:
                score += 15
            if not security_analysis["proxy_contract"]:
                score += 10
            
            security_analysis["security_score"] = score
        
        return security_analysis
