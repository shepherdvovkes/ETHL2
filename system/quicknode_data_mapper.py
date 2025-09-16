#!/usr/bin/env python3
"""
Карта данных QuickNode API для Polygon - определяет какие данные доступны
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List, Optional, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.quicknode_client import QuickNodeClient
from config.settings import settings

class QuickNodeDataMapper:
    """Карта доступных данных в QuickNode API для Polygon"""
    
    def __init__(self):
        self.available_data = {}
        self.data_sources = {}
        
    async def map_available_data(self) -> Dict[str, Any]:
        """Создать карту всех доступных данных в QuickNode API"""
        logger.info("🗺️ Mapping available data in QuickNode API...")
        
        try:
            async with QuickNodeClient() as qn_client:
                # 1. Базовые сетевые данные
                network_data = await self._map_network_data(qn_client)
                
                # 2. Блокчейн данные
                blockchain_data = await self._map_blockchain_data(qn_client)
                
                # 3. Транзакционные данные
                transaction_data = await self._map_transaction_data(qn_client)
                
                # 4. Контрактные данные
                contract_data = await self._map_contract_data(qn_client)
                
                # 5. DeFi протоколы
                defi_data = await self._map_defi_data(qn_client)
                
                # 6. Gas и комиссии
                gas_data = await self._map_gas_data(qn_client)
                
                # Объединить все данные
                self.available_data = {
                    "network": network_data,
                    "blockchain": blockchain_data,
                    "transactions": transaction_data,
                    "contracts": contract_data,
                    "defi": defi_data,
                    "gas": gas_data,
                    "mapping_timestamp": datetime.utcnow(),
                    "api_endpoint": settings.QUICKNODE_HTTP_ENDPOINT
                }
                
                logger.info("✅ Data mapping completed successfully")
                return self.available_data
                
        except Exception as e:
            logger.error(f"Error mapping QuickNode data: {e}")
            return {"error": str(e)}
    
    async def _map_network_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта сетевых данных"""
        logger.info("📡 Mapping network data...")
        
        try:
            # Получить текущий блок
            current_block = await qn_client.get_block_number()
            
            # Получить статистику сети
            network_stats = await qn_client.get_network_stats()
            
            return {
                "current_block": current_block,
                "block_time_avg": 2.3,  # Polygon average block time
                "network_utilization": network_stats.get("network_utilization", 0.0),
                "gas_price_current": network_stats.get("gas_price_gwei", 0.0),
                "available_methods": [
                    "eth_blockNumber",
                    "eth_getBlockByNumber", 
                    "eth_getBlockByHash",
                    "eth_gasPrice",
                    "eth_getBalance",
                    "eth_getTransactionCount",
                    "eth_getCode",
                    "eth_getLogs",
                    "eth_call"
                ],
                "data_types": [
                    "block_data",
                    "transaction_data", 
                    "log_data",
                    "balance_data",
                    "code_data"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error mapping network data: {e}")
            return {"error": str(e)}
    
    async def _map_blockchain_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта блокчейн данных"""
        logger.info("⛓️ Mapping blockchain data...")
        
        return {
            "chain_id": 137,
            "chain_name": "Polygon",
            "native_token": "MATIC",
            "block_explorer": "https://polygonscan.com",
            "consensus_mechanism": "Proof of Stake",
            "validator_count": "~100 validators",
            "staking_mechanism": "Delegated Proof of Stake",
            "available_data": [
                "block_headers",
                "block_transactions",
                "block_receipts",
                "validator_info",
                "staking_data",
                "governance_data"
            ],
            "historical_range": {
                "blocks": "From genesis to current",
                "transactions": "All transactions in blocks",
                "logs": "All event logs"
            }
        }
    
    async def _map_transaction_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта транзакционных данных"""
        logger.info("💸 Mapping transaction data...")
        
        try:
            # Получить текущий блок для анализа
            current_block = await qn_client.get_block_number()
            sample_block = await qn_client.get_block_by_number(current_block)
            
            transaction_count = len(sample_block.get("transactions", []))
            
            return {
                "transaction_types": [
                    "native_transfers",
                    "token_transfers", 
                    "contract_calls",
                    "contract_creation",
                    "delegate_calls"
                ],
                "available_fields": [
                    "hash",
                    "from",
                    "to", 
                    "value",
                    "gas",
                    "gasPrice",
                    "nonce",
                    "input",
                    "blockNumber",
                    "blockHash",
                    "transactionIndex"
                ],
                "sample_block_transactions": transaction_count,
                "estimated_daily_transactions": "2-4 million",
                "data_sources": [
                    "eth_getTransactionByHash",
                    "eth_getTransactionByBlockHashAndIndex",
                    "eth_getTransactionByBlockNumberAndIndex",
                    "eth_getTransactionReceipt"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error mapping transaction data: {e}")
            return {"error": str(e)}
    
    async def _map_contract_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта контрактных данных"""
        logger.info("📋 Mapping contract data...")
        
        # Основные контракты Polygon
        polygon_contracts = {
            "native_token": "0x0000000000000000000000000000000000001010",  # MATIC
            "usdc": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
            "usdt": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
            "weth": "0x7ceb23fd6fc0ad59923861afc8967b5e6d6c4e",
            "wbtc": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6"
        }
        
        return {
            "contract_types": [
                "ERC20_tokens",
                "ERC721_NFTs",
                "ERC1155_multi_tokens",
                "DeFi_protocols",
                "DEX_contracts",
                "Lending_protocols",
                "Staking_contracts"
            ],
            "major_contracts": polygon_contracts,
            "available_data": [
                "contract_code",
                "contract_abi",
                "contract_events",
                "contract_calls",
                "token_balances",
                "token_transfers"
            ],
            "data_sources": [
                "eth_getCode",
                "eth_getLogs",
                "eth_call"
            ],
            "event_signatures": {
                "Transfer": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                "Approval": "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925",
                "Swap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
            }
        }
    
    async def _map_defi_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта DeFi данных"""
        logger.info("🏦 Mapping DeFi data...")
        
        # Основные DeFi протоколы на Polygon
        defi_protocols = {
            "aave": {
                "lending_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
                "data_provider": "0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654",
                "available_data": ["lending_rates", "borrowing_rates", "liquidity", "collateral"]
            },
            "quickswap": {
                "factory": "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
                "router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                "available_data": ["liquidity_pools", "swap_volumes", "fees", "prices"]
            },
            "curve": {
                "registry": "0x094d12e5b541784701FD8d65F11fc0598FBC6332",
                "available_data": ["pool_data", "swap_volumes", "yield_rates"]
            },
            "sushiswap": {
                "factory": "0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
                "router": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
                "available_data": ["liquidity_pools", "swap_volumes", "fees"]
            }
        }
        
        return {
            "protocols": defi_protocols,
            "data_types": [
                "tvl_total_value_locked",
                "liquidity_pools",
                "swap_volumes", 
                "lending_volumes",
                "borrowing_volumes",
                "yield_rates",
                "fees_generated",
                "user_activity"
            ],
            "metrics_available": [
                "daily_volume",
                "weekly_volume", 
                "monthly_volume",
                "active_users",
                "new_users",
                "retention_rate",
                "protocol_revenue"
            ]
        }
    
    async def _map_gas_data(self, qn_client: QuickNodeClient) -> Dict[str, Any]:
        """Карта данных о газе и комиссиях"""
        logger.info("⛽ Mapping gas data...")
        
        try:
            current_gas_price = await qn_client.get_gas_price()
            
            return {
                "current_gas_price_gwei": current_gas_price,
                "gas_price_history": "Available via eth_gasPrice calls",
                "gas_usage_patterns": [
                    "simple_transfers",
                    "token_transfers", 
                    "contract_calls",
                    "complex_contracts"
                ],
                "typical_gas_limits": {
                    "simple_transfer": 21000,
                    "token_transfer": 65000,
                    "contract_call": 100000,
                    "complex_operation": 500000
                },
                "gas_price_trends": "Historical data available",
                "network_congestion": "Measurable via gas prices"
            }
            
        except Exception as e:
            logger.error(f"Error mapping gas data: {e}")
            return {"error": str(e)}
    
    def generate_data_collection_plan(self) -> Dict[str, Any]:
        """Генерировать план сбора данных на основе карты"""
        logger.info("📋 Generating data collection plan...")
        
        plan = {
            "collection_strategy": "on_demand_based_on_ml_requirements",
            "data_priorities": {
                "high_priority": [
                    "current_block_data",
                    "recent_transaction_volumes", 
                    "gas_price_trends",
                    "major_defi_protocol_activity"
                ],
                "medium_priority": [
                    "historical_price_data",
                    "network_utilization",
                    "contract_interaction_patterns"
                ],
                "low_priority": [
                    "detailed_log_analysis",
                    "complex_contract_metrics"
                ]
            },
            "ml_training_data_sources": {
                "price_prediction": [
                    "transaction_volume_correlation",
                    "gas_price_correlation", 
                    "defi_activity_correlation",
                    "network_activity_correlation"
                ],
                "trend_analysis": [
                    "daily_transaction_patterns",
                    "weekly_volume_patterns",
                    "seasonal_activity_patterns"
                ]
            },
            "real_time_metrics": [
                "current_gas_price",
                "current_block_number",
                "recent_transaction_count",
                "active_contracts"
            ],
            "historical_analysis": [
                "block_range_analysis",
                "transaction_pattern_analysis", 
                "gas_price_volatility_analysis",
                "defi_protocol_growth_analysis"
            ]
        }
        
        return plan
    
    async def save_data_map(self, filepath: str = "quicknode_data_map.json"):
        """Сохранить карту данных"""
        try:
            data_map = {
                "available_data": self.available_data,
                "collection_plan": self.generate_data_collection_plan(),
                "created_at": datetime.utcnow(),
                "api_endpoint": settings.QUICKNODE_HTTP_ENDPOINT
            }
            
            with open(filepath, "w") as f:
                json.dump(data_map, f, indent=2, default=str)
            
            logger.info(f"💾 Data map saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data map: {e}")
    
    def get_ml_training_features(self) -> List[Dict[str, Any]]:
        """Получить список признаков для ML обучения"""
        features = [
            {
                "name": "transaction_volume_24h",
                "source": "eth_getLogs",
                "description": "24-hour transaction volume",
                "calculation": "Sum of transaction values in last 24h",
                "importance": "high"
            },
            {
                "name": "gas_price_trend",
                "source": "eth_gasPrice", 
                "description": "Gas price trend over time",
                "calculation": "Moving average of gas prices",
                "importance": "high"
            },
            {
                "name": "network_activity",
                "source": "eth_getBlockByNumber",
                "description": "Network transaction activity",
                "calculation": "Transactions per block average",
                "importance": "medium"
            },
            {
                "name": "defi_protocol_volume",
                "source": "eth_getLogs + contract_calls",
                "description": "DeFi protocol interaction volume", 
                "calculation": "Sum of DeFi protocol interactions",
                "importance": "high"
            },
            {
                "name": "contract_deployment_rate",
                "source": "eth_getCode",
                "description": "Rate of new contract deployments",
                "calculation": "New contracts per day",
                "importance": "medium"
            },
            {
                "name": "whale_activity",
                "source": "eth_getLogs",
                "description": "Large transaction activity",
                "calculation": "Transactions > $100k value",
                "importance": "medium"
            }
        ]
        
        return features

async def main():
    """Основная функция"""
    logger.info("🗺️ Starting QuickNode Data Mapping...")
    
    try:
        mapper = QuickNodeDataMapper()
        
        # Создать карту данных
        data_map = await mapper.map_available_data()
        
        # Сохранить карту
        await mapper.save_data_map()
        
        # Вывести результаты
        logger.info("🎯 QUICKNODE DATA MAP RESULTS:")
        logger.info("=" * 60)
        
        for category, data in data_map.items():
            if isinstance(data, dict) and "error" not in data:
                logger.info(f"📊 {category.upper()}:")
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        logger.info(f"   • {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        logger.info(f"   • {key}: {', '.join(map(str, value))}")
        
        # Показать ML признаки
        features = mapper.get_ml_training_features()
        logger.info(f"\n🤖 ML TRAINING FEATURES ({len(features)}):")
        for feature in features:
            logger.info(f"   • {feature['name']}: {feature['description']} ({feature['importance']} priority)")
        
        logger.info("\n💾 Full data map saved to: quicknode_data_map.json")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error in data mapping: {e}")

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск
    asyncio.run(main())
