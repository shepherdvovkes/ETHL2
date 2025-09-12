# Ethereum L2 Analytics System Setup

This document provides setup instructions for the Ethereum L2 Analytics System.

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Docker (optional)

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# API Keys
QUICKNODE_API_KEY=your_quicknode_api_key_here
QUICKNODE_HTTP_ENDPOINT=your_quicknode_http_endpoint_here
QUICKNODE_WSS_ENDPOINT=your_quicknode_wss_endpoint_here

# Etherscan API (already configured)
ETHERSCAN_API_KEY=753BZTQQDZ1B6TYNDUPQAZHPDWSMWXUXGQ

# Hugging Face (already configured)
HF_TOKEN=your_huggingface_token_here

# Database
DATABASE_URL=postgresql://defimon:password@localhost:5432/defimon_db
REDIS_URL=redis://localhost:6379
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up the database: `python setup_database_v2.py`
4. Run the application: `python run.py`

## Usage

The system provides various analysis tools for Ethereum L2 networks:

- Real-time monitoring
- Investment metrics collection
- Price prediction
- Network analysis

For detailed usage instructions, see the individual README files in the project.
