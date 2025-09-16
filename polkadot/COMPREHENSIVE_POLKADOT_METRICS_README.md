# Comprehensive Polkadot Metrics System

A comprehensive monitoring and analytics system for the Polkadot network and its parachain ecosystem. This system provides real-time metrics, historical data analysis, and interactive dashboards for monitoring network health, economic indicators, and parachain performance.

## 🚀 Features

### Network Metrics
- **Block Metrics**: Current block, block time, transaction throughput
- **Validator Metrics**: Validator count, active validators, staking distribution
- **Performance Metrics**: Network latency, consensus time, finalization time
- **Security Metrics**: Nakamoto coefficient, decentralization index, security incidents

### Staking Metrics
- **Staking Amounts**: Total staked, staking ratio, ideal staking rate
- **Era Information**: Active era, era progress, era length
- **Validator Metrics**: Validator count, nominator count, commission rates
- **Nomination Pools**: Pool count, members, staked amounts
- **Rewards**: Block rewards, validator rewards, era rewards
- **Inflation**: Inflation rate, annual inflation, deflation rate

### Governance Metrics
- **Democracy**: Active proposals, referendums, voting participation
- **Council**: Council members, motions, votes
- **Treasury**: Treasury balance, proposals, spend rate
- **Voting**: Participation rate, direct voters, delegated voters

### Economic Metrics
- **Treasury**: Treasury balance, spend rate, burn rate
- **Tokenomics**: Total supply, circulating supply, inflation rate
- **Market Data**: Market cap, price, volume, price changes
- **Transaction Fees**: Average fees, total fees, fee burn

### Parachain Metrics
- **Block Metrics**: Current block, block time, production rate
- **Transaction Metrics**: Daily transactions, volume, success rate
- **User Activity**: Active addresses, new addresses, unique users
- **Token Metrics**: Supply, circulation, price, market cap
- **Network Health**: Validator count, collator count, uptime

### Cross-Chain Metrics
- **HRMP**: Channels, messages, volume, fees
- **XCMP**: Channels, messages, volume, fees
- **XCM**: Messages, volume, fees
- **Bridges**: Volume, transactions, TVL
- **Liquidity**: Cross-chain liquidity, arbitrage opportunities

### DeFi Metrics (Parachain-specific)
- **TVL**: Total value locked, TVL changes
- **DEX**: Volume, trades, liquidity pools
- **Lending**: TVL, borrowed amounts, APY rates
- **Staking**: Liquid staking TVL, staking APY
- **Yield Farming**: TVL, APY, active farms
- **Derivatives**: TVL, volume, options, futures

### Developer Metrics
- **Activity**: Total developers, commits, stars, forks
- **Projects**: Active projects, new launches, funding
- **Documentation**: Updates, tutorial views, community questions

### Security Metrics
- **Incidents**: Security incidents, vulnerabilities, audits
- **Network Security**: Validator uptime, geographic distribution
- **Smart Contracts**: Vulnerabilities, audits, upgrades

## 📊 Database Schema

The system uses a comprehensive database schema with the following main tables:

### Core Tables
- `polkadot_networks` - Polkadot network information
- `parachains` - Parachain information and metadata
- `token_market_data` - Token price and market data
- `validator_info` - Validator information and metrics

### Metrics Tables
- `polkadot_network_metrics` - Network performance metrics
- `polkadot_staking_metrics` - Staking-related metrics
- `polkadot_governance_metrics` - Governance metrics
- `polkadot_economic_metrics` - Economic indicators
- `polkadot_performance_metrics` - Performance metrics
- `polkadot_security_metrics` - Security metrics
- `polkadot_developer_metrics` - Developer activity metrics

### Parachain Metrics Tables
- `parachain_metrics` - General parachain metrics
- `parachain_cross_chain_metrics` - Cross-chain messaging metrics
- `parachain_defi_metrics` - DeFi-specific metrics
- `parachain_performance_metrics` - Performance metrics
- `parachain_security_metrics` - Security metrics
- `parachain_developer_metrics` - Developer metrics

### Ecosystem Tables
- `polkadot_ecosystem_metrics` - Overall ecosystem metrics

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL or SQLite
- Node.js (for dashboard)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database
```bash
python setup_comprehensive_polkadot_metrics.py
```

### 3. Collect Initial Data
```bash
python collect_comprehensive_polkadot_data.py
```

### 4. Start Metrics Server
```bash
python polkadot_comprehensive_metrics_server.py
```

### 5. Access Dashboard
Open your browser and navigate to: `http://localhost:8008`

## 📈 API Endpoints

### Network Endpoints
- `GET /api/network/overview` - Comprehensive network overview
- `GET /api/network/metrics` - Network metrics history
- `GET /api/staking/metrics` - Staking metrics history
- `GET /api/governance/metrics` - Governance metrics history
- `GET /api/economic/metrics` - Economic metrics history

### Parachain Endpoints
- `GET /api/parachains` - All parachains
- `GET /api/parachains/{id}/metrics` - Specific parachain metrics
- `GET /api/parachains/categories` - Parachains by category

### Cross-Chain Endpoints
- `GET /api/cross-chain/metrics` - Cross-chain messaging metrics

### Token Endpoints
- `GET /api/tokens/market-data` - Token market data

### Validator Endpoints
- `GET /api/validators` - Validator information

### Ecosystem Endpoints
- `GET /api/ecosystem/metrics` - Ecosystem-wide metrics

### Data Collection Endpoints
- `POST /api/collect` - Trigger manual data collection

## 🎯 Supported Parachains

The system monitors 20+ active parachains across different categories:

### DeFi Parachains
- **Moonbeam** (2004) - Ethereum-compatible smart contracts
- **Acala** (2000) - DeFi hub with stablecoins and lending
- **Astar** (2006) - Multi-VM smart contract platform
- **HydraDX** (2034) - Omnipool AMM
- **Bifrost** (2030) - Liquid staking protocol
- **Parallel** (2012) - DeFi infrastructure
- **Clover** (2002) - DeFi infrastructure
- **Equilibrium** (2011) - DeFi protocol
- **Composable** (2019) - DeFi infrastructure

### Infrastructure Parachains
- **AssetHub** (1000) - Asset management
- **Frequency** (2091) - Social media protocol

### Identity Parachains
- **Litentry** (2013) - Identity aggregation
- **KILT Protocol** (2085) - Identity infrastructure

### Computing Parachains
- **Phala Network** (2035) - Privacy-preserving cloud computing
- **NeuroWeb** (2046) - AI and machine learning

### IoT Parachains
- **Nodle** (2026) - IoT connectivity

### Privacy Parachains
- **Manta** (2104) - Privacy-preserving DeFi

### NFT Parachains
- **Efinity** (2121) - NFT infrastructure

### Gaming Parachains
- **SubDAO** (2018) - Gaming and governance

### Prediction Markets
- **Zeitgeist** (2092) - Prediction markets

## 📊 Dashboard Features

### Overview Tab
- Network health indicators
- Key metrics summary
- Real-time network status

### Network Tab
- Staking metrics charts
- Economic indicators
- Governance activity
- Performance metrics

### Parachains Tab
- Parachain categories
- Individual parachain cards
- Cross-chain activity

### Tokens Tab
- Token market data
- Price charts
- Market cap rankings

### Validators Tab
- Validator distribution
- Performance metrics
- Geographic distribution

## 🔄 Data Collection

The system automatically collects data every 10 minutes with the following schedule:

1. **Network Metrics** - Every 10 minutes
2. **Parachain Metrics** - Every 10 minutes
3. **Cross-Chain Metrics** - Every 10 minutes
4. **Token Market Data** - Every 30 minutes
5. **Validator Information** - Every hour
6. **Ecosystem Metrics** - Every hour

## 🚨 Monitoring & Alerts

The system provides monitoring for:

- Network health and uptime
- Validator performance
- Cross-chain message failures
- Economic indicators
- Security incidents
- Developer activity

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/polkadot_metrics

# API Configuration
API_HOST=0.0.0.0
API_PORT=8008

# RPC Endpoints
POLKADOT_RPC_ENDPOINT=https://rpc.polkadot.io
POLKADOT_WS_ENDPOINT=wss://rpc.polkadot.io
```

### Custom Parachain Configuration
Add new parachains by updating the `active_parachains` dictionary in `polkadot_comprehensive_client.py`.

## 📝 Usage Examples

### Collect Data Manually
```python
from collect_comprehensive_polkadot_data import ComprehensivePolkadotDataCollector

collector = ComprehensivePolkadotDataCollector()
await collector.initialize()
await collector.collect_all_metrics()
await collector.cleanup()
```

### Query Metrics
```python
from database.database import SessionLocal
from database.polkadot_comprehensive_models import PolkadotNetworkMetrics

db = SessionLocal()
latest_metrics = db.query(PolkadotNetworkMetrics).order_by(
    PolkadotNetworkMetrics.timestamp.desc()
).first()
```

### API Usage
```bash
# Get network overview
curl http://localhost:8008/api/network/overview

# Get parachain metrics
curl http://localhost:8008/api/parachains/2004/metrics

# Trigger data collection
curl -X POST http://localhost:8008/api/collect
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Join our Discord community
- Check the documentation

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Core metrics collection
- ✅ Basic dashboard
- ✅ API endpoints

### Phase 2 (Next)
- 🔄 Advanced analytics
- 🔄 Machine learning predictions
- 🔄 Custom alerts

### Phase 3 (Future)
- 📋 Mobile app
- 📋 Advanced visualizations
- 📋 Integration with external tools

---

**Built with ❤️ for the Polkadot ecosystem**
