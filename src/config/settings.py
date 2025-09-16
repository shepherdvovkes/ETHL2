import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from config.env
load_dotenv("config.env")

class Settings:
    # API Keys
    QUICKNODE_API_KEY: str = os.getenv("QUICKNODE_API_KEY", "")
    QUICKNODE_HTTP_ENDPOINT: str = os.getenv("QUICKNODE_HTTP_ENDPOINT", "")
    QUICKNODE_WSS_ENDPOINT: str = os.getenv("QUICKNODE_WSS_ENDPOINT", "")
    
    # Avalanche QuickNode Endpoints
    QUICKNODE_AVALANCHE_C_CHAIN_ENDPOINT: str = os.getenv("QUICKNODE_AVALANCHE_C_CHAIN_ENDPOINT", "")
    QUICKNODE_AVALANCHE_C_CHAIN_WSS_ENDPOINT: str = os.getenv("QUICKNODE_AVALANCHE_C_CHAIN_WSS_ENDPOINT", "")
    QUICKNODE_AVALANCHE_P_CHAIN_ENDPOINT: str = os.getenv("QUICKNODE_AVALANCHE_P_CHAIN_ENDPOINT", "")
    QUICKNODE_AVALANCHE_X_CHAIN_ENDPOINT: str = os.getenv("QUICKNODE_AVALANCHE_X_CHAIN_ENDPOINT", "")
    
    # TRON QuickNode Endpoints
    QUICKNODE_TRON_HTTP_ENDPOINT: str = os.getenv("QUICKNODE_TRON_HTTP_ENDPOINT", "")
    QUICKNODE_TRON_WSS_ENDPOINT: str = os.getenv("QUICKNODE_TRON_WSS_ENDPOINT", "")
    
    # Polkadot QuickNode Endpoints
    POLKADOT_RPC_ENDPOINT: str = os.getenv("POLKADOT_RPC_ENDPOINT", "")
    POLKADOT_WS_ENDPOINT: str = os.getenv("POLKADOT_WS_ENDPOINT", "")
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
    INFURA_API_KEY: str = os.getenv("INFURA_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI: str = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")
    COINMARKETCAP_API_KEY: str = os.getenv("COINMARKETCAP_API_KEY", "")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://defimon:password@localhost:5432/defimon_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    
    # Data Collection
    COLLECTION_INTERVAL: int = int(os.getenv("COLLECTION_INTERVAL", "3600"))  # seconds
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    
    # ML Configuration
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./models")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium")
    
    # Supported Blockchains (50+ blockchains)
    SUPPORTED_BLOCKCHAINS: List[str] = os.getenv("SUPPORTED_BLOCKCHAINS", 
        "ethereum,polygon,bsc,arbitrum,optimism,avalanche,solana,cardano,polkadot,cosmos,fantom,"
        "tron,litecoin,chainlink,near,algorand,uniswap,aave,compound,curve,sushiswap,"
        "immutable-x,axie-infinity,sandbox,decentraland,monero,zcash,dash,"
        "tezos,eos,waves,neo,vechain,hedera,elrond,harmony,klaytn,cronos,gnosis,celo,"
        "moonbeam,aurora,evmos,kava,injective,osmosis,juno,secret,terra-classic,terra-2,"
        "aptos,sui,sei,base,linea,scroll,mantle,blast"
    ).split(",")
    
    # Default Assets to Track (Polygon DeFi tokens)
    DEFAULT_ASSETS: List[dict] = [
        {"symbol": "MATIC", "name": "Polygon", "contract": "0x0000000000000000000000000000000000001010", "blockchain": "polygon"},
        {"symbol": "USDC", "name": "USD Coin", "contract": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174", "blockchain": "polygon"},
        {"symbol": "USDT", "name": "Tether USD", "contract": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f", "blockchain": "polygon"},
        {"symbol": "WETH", "name": "Wrapped Ether", "contract": "0x7ceb23fd6fc0ad59923861afc8967b5e6d6c4e", "blockchain": "polygon"},
        {"symbol": "WBTC", "name": "Wrapped Bitcoin", "contract": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6", "blockchain": "polygon"},
        {"symbol": "AAVE", "name": "Aave Token", "contract": "0xd6df932a45c0f255f85145f286ea0b292b21c90b", "blockchain": "polygon"},
        {"symbol": "CRV", "name": "Curve DAO Token", "contract": "0x172370d5cd63279efa6d502dab29171933a610af", "blockchain": "polygon"},
        {"symbol": "SUSHI", "name": "SushiToken", "contract": "0x0b3f868e0be5597d5db7feb59e1cadb0c3da0f9c", "blockchain": "polygon"},
        {"symbol": "QUICK", "name": "Quickswap", "contract": "0x831753dd7087cac61ab5644b308642cc1c33dc13", "blockchain": "polygon"},
        {"symbol": "BAL", "name": "Balancer", "contract": "0x9a71012b13ca4d3d0cdc72a177df3ef03b0e76a3", "blockchain": "polygon"},
    ]
    
    # Competitor Analysis
    COMPETITOR_PLATFORMS: List[dict] = [
        {
            "name": "DeFiPulse",
            "url": "https://defipulse.com",
            "type": "analytics",
            "features": ["TVL", "protocol_rankings", "yield_farming"]
        },
        {
            "name": "DeFiLlama",
            "url": "https://defillama.com",
            "type": "analytics",
            "features": ["TVL", "protocol_analytics", "yield_tracking"]
        },
        {
            "name": "Token Terminal",
            "url": "https://tokenterminal.com",
            "type": "analytics",
            "features": ["revenue_metrics", "P/E_ratios", "protocol_comparison"]
        },
        {
            "name": "Dune Analytics",
            "url": "https://dune.com",
            "type": "analytics",
            "features": ["custom_queries", "dashboard_creation", "data_visualization"]
        },
        {
            "name": "Nansen",
            "url": "https://nansen.ai",
            "type": "analytics",
            "features": ["wallet_analysis", "smart_money_tracking", "market_intelligence"]
        }
    ]

settings = Settings()
