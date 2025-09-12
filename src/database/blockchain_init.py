#!/usr/bin/env python3
"""
Инициализация блокчейнов для DEFIMON системы
Поддерживает 50+ блокчейнов
"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models_v2 import Base, Blockchain, BlockchainType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://defimon:password@localhost:5432/defimon_db"
)

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_blockchains():
    """Инициализация всех поддерживаемых блокчейнов"""
    
    blockchains_data = [
        # Layer 1 Blockchains
        {
            "name": "Ethereum",
            "symbol": "ETH",
            "chain_id": 1,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet.infura.io/v3/",
            "explorer_url": "https://etherscan.io",
            "native_token": "ETH",
            "description": "The world's leading smart contract platform"
        },
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://blockstream.info",
            "native_token": "BTC",
            "description": "The first and largest cryptocurrency"
        },
        {
            "name": "Binance Smart Chain",
            "symbol": "BNB",
            "chain_id": 56,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://bsc-dataseed.binance.org",
            "explorer_url": "https://bscscan.com",
            "native_token": "BNB",
            "description": "Binance's smart contract blockchain"
        },
        {
            "name": "Polygon",
            "symbol": "MATIC",
            "chain_id": 137,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://polygon-rpc.com",
            "explorer_url": "https://polygonscan.com",
            "native_token": "MATIC",
            "description": "Ethereum scaling solution"
        },
        {
            "name": "Avalanche",
            "symbol": "AVAX",
            "chain_id": 43114,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://api.avax.network/ext/bc/C/rpc",
            "explorer_url": "https://snowtrace.io",
            "native_token": "AVAX",
            "description": "High-performance smart contracts platform"
        },
        {
            "name": "Solana",
            "symbol": "SOL",
            "chain_id": 101,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://api.mainnet-beta.solana.com",
            "explorer_url": "https://explorer.solana.com",
            "native_token": "SOL",
            "description": "High-speed blockchain for DeFi and NFTs"
        },
        {
            "name": "Cardano",
            "symbol": "ADA",
            "chain_id": 1,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://cardano-mainnet.blockfrost.io/api/v0",
            "explorer_url": "https://cardanoscan.io",
            "native_token": "ADA",
            "description": "Research-driven blockchain platform"
        },
        {
            "name": "Polkadot",
            "symbol": "DOT",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "wss://rpc.polkadot.io",
            "explorer_url": "https://polkascan.io",
            "native_token": "DOT",
            "description": "Multi-chain blockchain platform"
        },
        {
            "name": "Cosmos",
            "symbol": "ATOM",
            "chain_id": "cosmoshub-4",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.cosmos.network",
            "explorer_url": "https://www.mintscan.io/cosmos",
            "native_token": "ATOM",
            "description": "Internet of blockchains"
        },
        {
            "name": "Fantom",
            "symbol": "FTM",
            "chain_id": 250,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.ftm.tools",
            "explorer_url": "https://ftmscan.com",
            "native_token": "FTM",
            "description": "Fast, scalable, and secure smart contract platform"
        },
        
        # Layer 2 Solutions
        {
            "name": "Arbitrum One",
            "symbol": "ETH",
            "chain_id": 42161,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://arb1.arbitrum.io/rpc",
            "explorer_url": "https://arbiscan.io",
            "native_token": "ETH",
            "description": "Optimistic rollup scaling solution for Ethereum"
        },
        {
            "name": "Optimism",
            "symbol": "ETH",
            "chain_id": 10,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://mainnet.optimism.io",
            "explorer_url": "https://optimistic.etherscan.io",
            "native_token": "ETH",
            "description": "Optimistic rollup for Ethereum"
        },
        {
            "name": "Polygon zkEVM",
            "symbol": "ETH",
            "chain_id": 1101,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://zkevm-rpc.com",
            "explorer_url": "https://zkevm.polygonscan.com",
            "native_token": "ETH",
            "description": "Zero-knowledge rollup for Ethereum"
        },
        {
            "name": "zkSync Era",
            "symbol": "ETH",
            "chain_id": 324,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://mainnet.era.zksync.io",
            "explorer_url": "https://explorer.zksync.io",
            "native_token": "ETH",
            "description": "Zero-knowledge rollup for Ethereum"
        },
        {
            "name": "StarkNet",
            "symbol": "ETH",
            "chain_id": "SN_MAIN",
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://starknet-mainnet.infura.io/v3/",
            "explorer_url": "https://starkscan.co",
            "native_token": "ETH",
            "description": "Zero-knowledge rollup for Ethereum"
        },
        
        # Other Major Blockchains
        {
            "name": "Tron",
            "symbol": "TRX",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://api.trongrid.io",
            "explorer_url": "https://tronscan.org",
            "native_token": "TRX",
            "description": "Decentralized content platform"
        },
        {
            "name": "Litecoin",
            "symbol": "LTC",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://blockchair.com/litecoin",
            "native_token": "LTC",
            "description": "Digital silver to Bitcoin's gold"
        },
        {
            "name": "Chainlink",
            "symbol": "LINK",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x514910771af9ca656af840dff83e8264ecf986ca",
            "native_token": "LINK",
            "description": "Decentralized oracle network"
        },
        {
            "name": "Near Protocol",
            "symbol": "NEAR",
            "chain_id": "mainnet",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.mainnet.near.org",
            "explorer_url": "https://explorer.near.org",
            "native_token": "NEAR",
            "description": "Developer-friendly blockchain"
        },
        {
            "name": "Algorand",
            "symbol": "ALGO",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet-api.algonode.cloud",
            "explorer_url": "https://algoexplorer.io",
            "native_token": "ALGO",
            "description": "Pure proof-of-stake blockchain"
        },
        
        # DeFi Focused Chains
        {
            "name": "Uniswap",
            "symbol": "UNI",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "native_token": "UNI",
            "description": "Decentralized exchange protocol"
        },
        {
            "name": "Aave",
            "symbol": "AAVE",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
            "native_token": "AAVE",
            "description": "Decentralized lending protocol"
        },
        {
            "name": "Compound",
            "symbol": "COMP",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0xc00e94cb662c3520282e6f5717214004a7f26888",
            "native_token": "COMP",
            "description": "Decentralized lending protocol"
        },
        {
            "name": "Curve",
            "symbol": "CRV",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0xd533a949740bb3306d119cc777fa900ba034cd52",
            "native_token": "CRV",
            "description": "Stablecoin exchange protocol"
        },
        {
            "name": "SushiSwap",
            "symbol": "SUSHI",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",
            "native_token": "SUSHI",
            "description": "Decentralized exchange protocol"
        },
        
        # Gaming & NFT Chains
        {
            "name": "Immutable X",
            "symbol": "IMX",
            "chain_id": 0,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://rpc.immutable.com",
            "explorer_url": "https://immutascan.io",
            "native_token": "IMX",
            "description": "NFT-focused Layer 2 for Ethereum"
        },
        {
            "name": "Axie Infinity",
            "symbol": "AXS",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0xbb0e17ef65f82ab018d8edd776e8dd940327b28b",
            "native_token": "AXS",
            "description": "Play-to-earn gaming platform"
        },
        {
            "name": "The Sandbox",
            "symbol": "SAND",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x3845badade8e6dd04c6e4c7a4b4c8fdfa55af43d",
            "native_token": "SAND",
            "description": "Virtual world and gaming platform"
        },
        {
            "name": "Decentraland",
            "symbol": "MANA",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://etherscan.io/token/0x0f5d2fb29fb7d3cfee444a200298f468908cc942",
            "native_token": "MANA",
            "description": "Virtual reality platform"
        },
        
        # Privacy Coins
        {
            "name": "Monero",
            "symbol": "XMR",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://xmrchain.net",
            "native_token": "XMR",
            "description": "Privacy-focused cryptocurrency"
        },
        {
            "name": "Zcash",
            "symbol": "ZEC",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://explorer.zcha.in",
            "native_token": "ZEC",
            "description": "Privacy-focused cryptocurrency"
        },
        {
            "name": "Dash",
            "symbol": "DASH",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": "https://explorer.dash.org",
            "native_token": "DASH",
            "description": "Digital cash with privacy features"
        },
        
        # Enterprise Blockchains
        {
            "name": "Hyperledger Fabric",
            "symbol": "HLF",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": None,
            "native_token": "HLF",
            "description": "Enterprise blockchain framework"
        },
        {
            "name": "Corda",
            "symbol": "CORDA",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": None,
            "explorer_url": None,
            "native_token": "CORDA",
            "description": "Enterprise blockchain platform"
        },
        
        # Additional Layer 1s
        {
            "name": "Tezos",
            "symbol": "XTZ",
            "chain_id": "mainnet",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet.tezos.marigold.dev",
            "explorer_url": "https://tzkt.io",
            "native_token": "XTZ",
            "description": "Self-amending blockchain"
        },
        {
            "name": "EOS",
            "symbol": "EOS",
            "chain_id": "aca376f206b8fc25a6ed44dbdc66547c36c6c33e3a119ffbeaef943642f0e906",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://eos.greymass.com",
            "explorer_url": "https://eosq.app",
            "native_token": "EOS",
            "description": "High-performance blockchain platform"
        },
        {
            "name": "Waves",
            "symbol": "WAVES",
            "chain_id": "W",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://nodes.wavesnodes.com",
            "explorer_url": "https://wavesexplorer.com",
            "native_token": "WAVES",
            "description": "Custom blockchain platform"
        },
        {
            "name": "NEO",
            "symbol": "NEO",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://seed1.neo.org:10331",
            "explorer_url": "https://neoscan.io",
            "native_token": "NEO",
            "description": "Smart economy platform"
        },
        {
            "name": "VeChain",
            "symbol": "VET",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet.veblocks.net",
            "explorer_url": "https://explore.vechain.org",
            "native_token": "VET",
            "description": "Enterprise-focused blockchain"
        },
        {
            "name": "Hedera",
            "symbol": "HBAR",
            "chain_id": 0,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet-public.mirrornode.hedera.com",
            "explorer_url": "https://hashscan.io",
            "native_token": "HBAR",
            "description": "Enterprise-grade public network"
        },
        {
            "name": "Elrond",
            "symbol": "EGLD",
            "chain_id": 1,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://api.elrond.com",
            "explorer_url": "https://explorer.elrond.com",
            "native_token": "EGLD",
            "description": "High-throughput blockchain"
        },
        {
            "name": "Harmony",
            "symbol": "ONE",
            "chain_id": 1666600000,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://api.harmony.one",
            "explorer_url": "https://explorer.harmony.one",
            "native_token": "ONE",
            "description": "Fast and open blockchain"
        },
        {
            "name": "Klaytn",
            "symbol": "KLAY",
            "chain_id": 8217,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://public-node-api.klaytnapi.com/v1/cypress",
            "explorer_url": "https://scope.klaytn.com",
            "native_token": "KLAY",
            "description": "Metaverse blockchain"
        },
        {
            "name": "Cronos",
            "symbol": "CRO",
            "chain_id": 25,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://evm.cronos.org",
            "explorer_url": "https://cronoscan.com",
            "native_token": "CRO",
            "description": "Crypto.com's blockchain"
        },
        {
            "name": "Gnosis Chain",
            "symbol": "GNO",
            "chain_id": 100,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.gnosischain.com",
            "explorer_url": "https://gnosisscan.io",
            "native_token": "GNO",
            "description": "Ethereum sidechain"
        },
        {
            "name": "Celo",
            "symbol": "CELO",
            "chain_id": 42220,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://forno.celo.org",
            "explorer_url": "https://explorer.celo.org",
            "native_token": "CELO",
            "description": "Mobile-first blockchain"
        },
        {
            "name": "Moonbeam",
            "symbol": "GLMR",
            "chain_id": 1284,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.api.moonbeam.network",
            "explorer_url": "https://moonbeam.moonscan.io",
            "native_token": "GLMR",
            "description": "Ethereum-compatible parachain"
        },
        {
            "name": "Aurora",
            "symbol": "ETH",
            "chain_id": 1313161554,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://mainnet.aurora.dev",
            "explorer_url": "https://aurorascan.dev",
            "native_token": "ETH",
            "description": "Ethereum-compatible blockchain on NEAR"
        },
        {
            "name": "Evmos",
            "symbol": "EVMOS",
            "chain_id": 9001,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://eth.bd.evmos.org:8545",
            "explorer_url": "https://evm.evmos.org",
            "native_token": "EVMOS",
            "description": "Ethereum-compatible Cosmos chain"
        },
        {
            "name": "Kava",
            "symbol": "KAVA",
            "chain_id": 2222,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://evm.kava.io",
            "explorer_url": "https://explorer.kava.io",
            "native_token": "KAVA",
            "description": "DeFi hub for Cosmos"
        },
        {
            "name": "Injective",
            "symbol": "INJ",
            "chain_id": "injective-1",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://tm.injective.network",
            "explorer_url": "https://explorer.injective.network",
            "native_token": "INJ",
            "description": "DeFi-focused Cosmos chain"
        },
        {
            "name": "Osmosis",
            "symbol": "OSMO",
            "chain_id": "osmosis-1",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.osmosis.zone",
            "explorer_url": "https://www.mintscan.io/osmosis",
            "native_token": "OSMO",
            "description": "Cosmos DEX chain"
        },
        {
            "name": "Juno",
            "symbol": "JUNO",
            "chain_id": "juno-1",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc-juno.itastakers.com",
            "explorer_url": "https://www.mintscan.io/juno",
            "native_token": "JUNO",
            "description": "Cosmos smart contract platform"
        },
        {
            "name": "Secret Network",
            "symbol": "SCRT",
            "chain_id": "secret-4",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.secret.express",
            "explorer_url": "https://www.mintscan.io/secret",
            "native_token": "SCRT",
            "description": "Privacy-preserving smart contracts"
        },
        {
            "name": "Terra Classic",
            "symbol": "LUNC",
            "chain_id": "columbus-5",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://terra-classic-rpc.publicnode.com",
            "explorer_url": "https://finder.terra.money/classic",
            "native_token": "LUNC",
            "description": "Algorithmic stablecoin platform (Classic)"
        },
        {
            "name": "Terra 2.0",
            "symbol": "LUNA",
            "chain_id": "phoenix-1",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://phoenix-rpc.terra.dev",
            "explorer_url": "https://finder.terra.money",
            "native_token": "LUNA",
            "description": "Rebuilt Terra blockchain"
        },
        {
            "name": "Aptos",
            "symbol": "APT",
            "chain_id": 1,
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://fullnode.mainnet.aptoslabs.com",
            "explorer_url": "https://explorer.aptoslabs.com",
            "native_token": "APT",
            "description": "Move-based blockchain"
        },
        {
            "name": "Sui",
            "symbol": "SUI",
            "chain_id": "sui-mainnet",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://fullnode.mainnet.sui.io:443",
            "explorer_url": "https://explorer.sui.io",
            "native_token": "SUI",
            "description": "Move-based blockchain"
        },
        {
            "name": "Sei",
            "symbol": "SEI",
            "chain_id": "pacific-1",
            "blockchain_type": BlockchainType.MAINNET,
            "rpc_url": "https://rpc.sei-apis.com",
            "explorer_url": "https://www.mintscan.io/sei",
            "native_token": "SEI",
            "description": "Trading-focused blockchain"
        },
        {
            "name": "Base",
            "symbol": "ETH",
            "chain_id": 8453,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://mainnet.base.org",
            "explorer_url": "https://basescan.org",
            "native_token": "ETH",
            "description": "Coinbase's Layer 2"
        },
        {
            "name": "Linea",
            "symbol": "ETH",
            "chain_id": 59144,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://rpc.linea.build",
            "explorer_url": "https://lineascan.build",
            "native_token": "ETH",
            "description": "ConsenSys Layer 2"
        },
        {
            "name": "Scroll",
            "symbol": "ETH",
            "chain_id": 534352,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://rpc.scroll.io",
            "explorer_url": "https://scrollscan.com",
            "native_token": "ETH",
            "description": "Native zkEVM Layer 2"
        },
        {
            "name": "Mantle",
            "symbol": "MNT",
            "chain_id": 5000,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://rpc.mantle.xyz",
            "explorer_url": "https://mantlescan.info",
            "native_token": "MNT",
            "description": "Modular Layer 2"
        },
        {
            "name": "Blast",
            "symbol": "ETH",
            "chain_id": 81457,
            "blockchain_type": BlockchainType.LAYER2,
            "rpc_url": "https://rpc.blast.io",
            "explorer_url": "https://blastscan.io",
            "native_token": "ETH",
            "description": "Native yield Layer 2"
        }
    ]
    
    db = SessionLocal()
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        # Insert blockchains
        for blockchain_data in blockchains_data:
            existing = db.query(Blockchain).filter(
                Blockchain.name == blockchain_data["name"]
            ).first()
            
            if not existing:
                blockchain = Blockchain(**blockchain_data)
                db.add(blockchain)
                print(f"✅ Added blockchain: {blockchain_data['name']}")
            else:
                print(f"⚠️  Blockchain already exists: {blockchain_data['name']}")
        
        db.commit()
        print(f"\n🎉 Successfully initialized {len(blockchains_data)} blockchains!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error initializing blockchains: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 DEFIMON - Blockchain Initialization")
    print("=" * 50)
    init_blockchains()
