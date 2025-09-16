#!/usr/bin/env python3
"""
Полный список Layer 2 сетей поверх Ethereum
Включает все основные категории: Optimistic Rollups, ZK Rollups, Sidechains, State Channels, Validium, Plasma
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class L2Type(Enum):
    OPTIMISTIC_ROLLUP = "Optimistic Rollup"
    ZK_ROLLUP = "ZK Rollup"
    VALIDIUM = "Validium"
    PLASMA = "Plasma"
    SIDECHAIN = "Sidechain"
    STATE_CHANNEL = "State Channel"
    HYBRID = "Hybrid"

class SecurityModel(Enum):
    ETHEREUM_SECURITY = "Ethereum Security"
    OWN_VALIDATORS = "Own Validators"
    HYBRID = "Hybrid"

@dataclass
class L2Network:
    name: str
    type: L2Type
    security_model: SecurityModel
    launch_date: str
    tvl_usd: Optional[float] = None
    tps: Optional[int] = None
    finality_time: Optional[str] = None
    withdrawal_time: Optional[str] = None
    native_token: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    status: str = "Active"

# Полный список Layer 2 сетей Ethereum
ETHEREUM_L2_NETWORKS = [
    # === OPTIMISTIC ROLLUPS ===
    L2Network(
        name="Arbitrum One",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-08-31",
        tvl_usd=2.5e9,  # ~$2.5B
        tps=4000,
        finality_time="~1 minute",
        withdrawal_time="7 days",
        native_token="ARB",
        website="https://arbitrum.io",
        description="Первый и крупнейший Optimistic Rollup, совместимый с EVM"
    ),
    
    L2Network(
        name="Arbitrum Nova",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2022-07-20",
        tvl_usd=50e6,  # ~$50M
        tps=4000,
        finality_time="~1 minute",
        withdrawal_time="7 days",
        native_token="ARB",
        website="https://nova.arbitrum.io",
        description="Оптимизирован для игр и социальных приложений"
    ),
    
    L2Network(
        name="Optimism",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-12-16",
        tvl_usd=800e6,  # ~$800M
        tps=2000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token="OP",
        website="https://optimism.io",
        description="Optimistic Rollup с фокусом на децентрализацию"
    ),
    
    L2Network(
        name="Base",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-07-13",
        tvl_usd=1.2e9,  # ~$1.2B
        tps=2000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token=None,
        website="https://base.org",
        description="L2 от Coinbase, построенный на OP Stack"
    ),
    
    L2Network(
        name="Boba Network",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-09-20",
        tvl_usd=5e6,  # ~$5M
        tps=1000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token="BOBA",
        website="https://boba.network",
        description="Optimistic Rollup с расширенными возможностями"
    ),
    
    L2Network(
        name="Metis",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-11-19",
        tvl_usd=10e6,  # ~$10M
        tps=1000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token="METIS",
        website="https://metis.io",
        description="Optimistic Rollup для децентрализованных автономных компаний"
    ),
    
    # === ZK ROLLUPS ===
    L2Network(
        name="zkSync Era",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-03-24",
        tvl_usd=600e6,  # ~$600M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="ZK",
        website="https://zksync.io",
        description="ZK Rollup с полной совместимостью с EVM"
    ),
    
    L2Network(
        name="StarkNet",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-11-29",
        tvl_usd=50e6,  # ~$50M
        tps=10000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="STRK",
        website="https://starknet.io",
        description="ZK Rollup с собственной виртуальной машиной Cairo"
    ),
    
    L2Network(
        name="Polygon zkEVM",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-03-27",
        tvl_usd=100e6,  # ~$100M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="MATIC",
        website="https://polygon.technology",
        description="ZK Rollup от Polygon с полной совместимостью с EVM"
    ),
    
    L2Network(
        name="Scroll",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-10-17",
        tvl_usd=80e6,  # ~$80M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="SCROLL",
        website="https://scroll.io",
        description="ZK Rollup с нативной совместимостью с EVM"
    ),
    
    L2Network(
        name="Linea",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-07-11",
        tvl_usd=200e6,  # ~$200M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="LXP",
        website="https://linea.build",
        description="ZK Rollup от ConsenSys с полной совместимостью с EVM"
    ),
    
    L2Network(
        name="Taiko",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2024-01-10",
        tvl_usd=20e6,  # ~$20M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="TKO",
        website="https://taiko.xyz",
        description="ZK Rollup с эквивалентностью Ethereum"
    ),
    
    # === VALIDIUM ===
    L2Network(
        name="Immutable X",
        type=L2Type.VALIDIUM,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2021-04-20",
        tvl_usd=30e6,  # ~$30M
        tps=9000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="IMX",
        website="https://immutable.com",
        description="Validium для NFT и игр, нулевые комиссии за газ"
    ),
    
    L2Network(
        name="Polygon Miden",
        type=L2Type.VALIDIUM,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2024-01-01",
        tvl_usd=5e6,  # ~$5M
        tps=10000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="MATIC",
        website="https://polygon.technology",
        description="Validium с поддержкой приватности"
    ),
    
    # === SIDECHAINS ===
    L2Network(
        name="Polygon PoS",
        type=L2Type.SIDECHAIN,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2020-05-30",
        tvl_usd=1e9,  # ~$1B
        tps=7000,
        finality_time="~2 seconds",
        withdrawal_time="~30 minutes",
        native_token="MATIC",
        website="https://polygon.technology",
        description="Proof of Stake сайдчейн с высокой производительностью"
    ),
    
    L2Network(
        name="Gnosis Chain",
        type=L2Type.SIDECHAIN,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2018-10-08",
        tvl_usd=100e6,  # ~$100M
        tps=1000,
        finality_time="~5 seconds",
        withdrawal_time="~30 minutes",
        native_token="GNO",
        website="https://gnosis.io",
        description="Сайдчейн с фокусом на децентрализованные приложения"
    ),
    
    L2Network(
        name="BSC (Binance Smart Chain)",
        type=L2Type.SIDECHAIN,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2020-09-01",
        tvl_usd=5e9,  # ~$5B
        tps=100,
        finality_time="~3 seconds",
        withdrawal_time="~30 minutes",
        native_token="BNB",
        website="https://bscscan.com",
        description="Сайдчейн от Binance с высокой совместимостью с EVM"
    ),
    
    # === PLASMA ===
    L2Network(
        name="Polygon Plasma",
        type=L2Type.PLASMA,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2019-05-30",
        tvl_usd=1e6,  # ~$1M
        tps=1000,
        finality_time="~1 minute",
        withdrawal_time="7 days",
        native_token="MATIC",
        website="https://polygon.technology",
        description="Plasma решение для быстрых платежей"
    ),
    
    # === STATE CHANNELS ===
    L2Network(
        name="Lightning Network",
        type=L2Type.STATE_CHANNEL,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2017-01-01",
        tvl_usd=100e6,  # ~$100M
        tps=1000000,
        finality_time="Instant",
        withdrawal_time="Instant",
        native_token="ETH",
        website="https://lightning.network",
        description="State channels для мгновенных микроплатежей"
    ),
    
    L2Network(
        name="Raiden Network",
        type=L2Type.STATE_CHANNEL,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2018-05-31",
        tvl_usd=1e6,  # ~$1M
        tps=1000000,
        finality_time="Instant",
        withdrawal_time="Instant",
        native_token="RDN",
        website="https://raiden.network",
        description="State channels для Ethereum"
    ),
    
    # === HYBRID SOLUTIONS ===
    L2Network(
        name="Polygon Avail",
        type=L2Type.HYBRID,
        security_model=SecurityModel.OWN_VALIDATORS,
        launch_date="2024-01-01",
        tvl_usd=0,
        tps=10000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="MATIC",
        website="https://polygon.technology",
        description="Модульная сеть для доступности данных"
    ),
    
    L2Network(
        name="Mantle",
        type=L2Type.HYBRID,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-07-17",
        tvl_usd=150e6,  # ~$150M
        tps=2000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token="MNT",
        website="https://mantle.xyz",
        description="Модульная L2 с отдельными уровнями выполнения и данных"
    ),
    
    # === ДРУГИЕ РЕШЕНИЯ ===
    L2Network(
        name="Loopring",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2019-12-03",
        tvl_usd=50e6,  # ~$50M
        tps=2000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="LRC",
        website="https://loopring.org",
        description="ZK Rollup для децентрализованных бирж"
    ),
    
    L2Network(
        name="Aztec",
        type=L2Type.ZK_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2021-03-15",
        tvl_usd=10e6,  # ~$10M
        tps=1000,
        finality_time="~10 minutes",
        withdrawal_time="~10 minutes",
        native_token="AZTEC",
        website="https://aztec.network",
        description="ZK Rollup с фокусом на приватность"
    ),
    
    L2Network(
        name="Fuel",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2024-01-01",
        tvl_usd=5e6,  # ~$5M
        tps=5000,
        finality_time="~1 minute",
        withdrawal_time="7 days",
        native_token="FUEL",
        website="https://fuel.network",
        description="Optimistic Rollup с UTXO моделью"
    ),
    
    L2Network(
        name="OP Stack Chains",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-01-01",
        tvl_usd=100e6,  # ~$100M
        tps=2000,
        finality_time="~2 minutes",
        withdrawal_time="7 days",
        native_token="Various",
        website="https://optimism.io",
        description="Семейство L2 сетей на базе OP Stack"
    ),
    
    L2Network(
        name="Arbitrum Orbit",
        type=L2Type.OPTIMISTIC_ROLLUP,
        security_model=SecurityModel.ETHEREUM_SECURITY,
        launch_date="2023-01-01",
        tvl_usd=50e6,  # ~$50M
        tps=4000,
        finality_time="~1 minute",
        withdrawal_time="7 days",
        native_token="ARB",
        website="https://arbitrum.io",
        description="Семейство L3 сетей на базе Arbitrum"
    ),
]

def get_networks_by_type(l2_type: L2Type) -> List[L2Network]:
    """Получить все сети определенного типа"""
    return [network for network in ETHEREUM_L2_NETWORKS if network.type == l2_type]

def get_networks_by_security_model(security_model: SecurityModel) -> List[L2Network]:
    """Получить все сети с определенной моделью безопасности"""
    return [network for network in ETHEREUM_L2_NETWORKS if network.security_model == security_model]

def get_networks_by_tvl_range(min_tvl: float, max_tvl: float) -> List[L2Network]:
    """Получить сети в определенном диапазоне TVL"""
    return [network for network in ETHEREUM_L2_NETWORKS 
            if network.tvl_usd and min_tvl <= network.tvl_usd <= max_tvl]

def get_network_stats() -> Dict:
    """Получить статистику по всем L2 сетям"""
    stats = {
        "total_networks": len(ETHEREUM_L2_NETWORKS),
        "by_type": {},
        "by_security_model": {},
        "total_tvl": 0,
        "active_networks": 0
    }
    
    for network in ETHEREUM_L2_NETWORKS:
        # По типам
        if network.type.value not in stats["by_type"]:
            stats["by_type"][network.type.value] = 0
        stats["by_type"][network.type.value] += 1
        
        # По моделям безопасности
        if network.security_model.value not in stats["by_security_model"]:
            stats["by_security_model"][network.security_model.value] = 0
        stats["by_security_model"][network.security_model.value] += 1
        
        # TVL
        if network.tvl_usd:
            stats["total_tvl"] += network.tvl_usd
        
        # Активные сети
        if network.status == "Active":
            stats["active_networks"] += 1
    
    return stats

def print_network_summary():
    """Вывести краткую сводку по всем сетям"""
    stats = get_network_stats()
    
    print("=== ПОЛНЫЙ СПИСОК LAYER 2 СЕТЕЙ ETHEREUM ===\n")
    print(f"Всего сетей: {stats['total_networks']}")
    print(f"Активных сетей: {stats['active_networks']}")
    print(f"Общий TVL: ${stats['total_tvl']/1e9:.2f}B\n")
    
    print("По типам:")
    for l2_type, count in stats['by_type'].items():
        print(f"  {l2_type}: {count}")
    
    print("\nПо моделям безопасности:")
    for security_model, count in stats['by_security_model'].items():
        print(f"  {security_model}: {count}")
    
    print("\n=== ТОП-10 ПО TVL ===")
    sorted_networks = sorted([n for n in ETHEREUM_L2_NETWORKS if n.tvl_usd], 
                           key=lambda x: x.tvl_usd, reverse=True)[:10]
    
    for i, network in enumerate(sorted_networks, 1):
        print(f"{i:2d}. {network.name:<20} ${network.tvl_usd/1e9:.2f}B ({network.type.value})")

if __name__ == "__main__":
    print_network_summary()
