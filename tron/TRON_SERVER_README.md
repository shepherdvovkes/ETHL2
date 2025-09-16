# 🚀 TRON Network Real-Time Metrics Server

A comprehensive real-time monitoring and analytics system for the TRON blockchain, continuously gathering data from QuickNode endpoints and providing detailed insights into network performance, economic metrics, DeFi ecosystem, and more.

## 📊 Overview

This server provides real-time monitoring of **12 key parameter categories** for the TRON network:

1. **Network Performance** - TPS, block time, finality, utilization
2. **Economic Metrics** - Price, volume, market cap, supply
3. **DeFi Ecosystem** - TVL, protocols, yields, volumes
4. **Smart Contracts** - Deployments, TRC-20 tokens, contract calls
5. **Staking & Governance** - Validators, staking ratios, proposals
6. **User Activity** - Addresses, retention, adoption metrics
7. **Network Health** - Security, decentralization, latency
8. **Token Metrics** - Major tokens (TRX, USDT, USDC, BTT)
9. **Protocol Analysis** - Individual DeFi protocol performance
10. **Risk Assessment** - Centralization, technical, economic risks
11. **Cross-Chain Metrics** - Bridge volumes, interoperability
12. **Comprehensive Scoring** - Overall network health assessment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 TRON Metrics Server                          │
├─────────────────────────────────────────────────────────────┤
│  Real-Time Data Collector  │  Monitoring System  │  API Server │
│  • Network Performance     │  • Alert Rules      │  • FastAPI   │
│  • Economic Data          │  • Notifications    │  • Endpoints │
│  • DeFi Metrics           │  • Thresholds       │  • Documentation │
│  • Smart Contract Data    │  • Cooldowns        │  • Health Check │
│  • User Activity          │  • Multi-channel    │  • Historical Data │
│  • Network Health         │  • Dashboard        │  • Real-time Updates │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Quick Start

### 1. Prerequisites

- Python 3.8+
- PostgreSQL database
- QuickNode TRON endpoint (provided)
- Redis (optional, for caching)

### 2. Installation

```bash
# Clone and navigate to the project
cd /home/vovkes/ETHL2

# Run the setup script
python setup_tron_server.py
```

### 3. Configuration

Update `tron_config.env` with your actual values:

```bash
# Required: Your QuickNode TRON endpoint
QUICKNODE_TRON_HTTP_ENDPOINT=https://wandering-distinguished-telescope.tron-mainnet.quiknode.pro/334584c6bb3a0655bd946cbae25fbd6594bcd8b5/jsonrpc

# Database configuration
TRON_DATABASE_URL=postgresql://user:password@localhost:5432/tron_metrics_db

# API keys (optional but recommended)
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
```

### 4. Start the Server

```bash
# Option 1: Direct start
./start_tron_server.sh

# Option 2: Using PM2 (recommended for production)
pm2 start ecosystem.tron.config.js

# Option 3: Manual start
python tron_metrics_server.py
```

### 5. Access the Dashboard

- **Dashboard**: http://localhost:8008/dashboard
- **API Documentation**: http://localhost:8008/docs
- **Health Check**: http://localhost:8008/health

## 📈 Key Metrics Collected

### Network Performance
- **Transaction Throughput**: Real-time TPS measurement
- **Block Time**: Average time between blocks
- **Finality Time**: Time to finalize transactions
- **Network Utilization**: Current network capacity usage
- **Active Nodes**: Number of active network nodes
- **Consensus Participation**: Validator participation rate

### Economic Metrics
- **TRX Price**: Real-time price in USD/BTC
- **Market Capitalization**: Total market value
- **Trading Volume**: 24h trading volume
- **Supply Metrics**: Total, circulating, and burned supply
- **Network Revenue**: Transaction fees and revenue
- **Market Dominance**: Share of total crypto market

### DeFi Ecosystem
- **Total Value Locked (TVL)**: Total assets locked in DeFi
- **Protocol Count**: Number of active DeFi protocols
- **DEX Volume**: Daily exchange volume
- **Lending Metrics**: TVL, borrowed amounts, utilization rates
- **Yield Farming**: APY rates and farming TVL
- **Bridge Activity**: Cross-chain transfer volumes

### Smart Contracts
- **Contract Deployments**: New contracts per day/week
- **TRC-20 Tokens**: Token count and activity
- **Major Tokens**: USDT, USDC, BTT supply and activity
- **NFT Metrics**: Collections, transactions, volumes
- **Contract Interactions**: Daily contract calls and energy usage

### Staking & Governance
- **Staking Amounts**: Total staked TRX and ratios
- **Validator Metrics**: Active validators and participation
- **Governance Activity**: Proposals and voting participation
- **Resource Management**: Energy and bandwidth freezing
- **APY Rates**: Current staking rewards

### User Activity
- **Active Addresses**: Daily active unique addresses
- **New Addresses**: Address creation rate
- **User Retention**: Long-term user engagement
- **DApp Usage**: DeFi and NFT user activity
- **Geographic Distribution**: Regional activity patterns

### Network Health
- **Security Score**: Overall network security assessment
- **Decentralization Index**: Network decentralization level
- **Latency Metrics**: Average network response times
- **Risk Assessment**: Centralization, technical, and economic risks
- **Incident Tracking**: Security incidents and error rates

## 🔌 API Endpoints

### Core Metrics
- `GET /metrics` - Summary of all metrics
- `GET /metrics/network` - Network performance metrics
- `GET /metrics/economic` - Economic and market data
- `GET /metrics/defi` - DeFi ecosystem metrics
- `GET /metrics/smart-contracts` - Smart contract metrics
- `GET /metrics/staking` - Staking and governance data
- `GET /metrics/user-activity` - User activity metrics
- `GET /metrics/network-health` - Network health assessment
- `GET /metrics/comprehensive` - Comprehensive analysis

### Management
- `GET /health` - Server health check
- `POST /collect` - Trigger manual data collection
- `GET /dashboard` - Interactive dashboard

### Example API Response

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "overall_score": 85.2,
  "risk_level": "low",
  "network_performance": {
    "block_number": 65432100,
    "tps": 125.5,
    "block_time": 2.98,
    "uptime": 99.9
  },
  "economic": {
    "market_cap": 11000000000,
    "volume_24h": 450000000,
    "price_change_24h": 2.5
  }
}
```

## 🗄️ Database Schema

The system uses PostgreSQL with the following main tables:

- `tron_network_metrics` - Network performance data
- `tron_economic_metrics` - Economic and market data
- `tron_defi_metrics` - DeFi ecosystem metrics
- `tron_smart_contract_metrics` - Smart contract data
- `tron_staking_metrics` - Staking and governance
- `tron_user_activity_metrics` - User activity data
- `tron_network_health_metrics` - Network health assessment
- `tron_protocol_metrics` - Individual protocol data
- `tron_token_metrics` - Token-specific metrics
- `tron_comprehensive_metrics` - Aggregated analysis

## 📊 Dashboard Features

### Real-Time Monitoring
- **Live Updates**: Auto-refresh every 30 seconds
- **Interactive Charts**: Network performance, economic trends
- **Risk Indicators**: Visual risk level assessment
- **Trend Analysis**: Performance trends and patterns

### Comprehensive Analysis
- **Overall Scoring**: 0-100 network health score
- **Multi-Dimensional View**: Performance across all categories
- **Risk Assessment**: Centralization and technical risks
- **Recommendations**: Actionable insights and suggestions

### Key Visualizations
- **Network Performance**: TPS, block time, utilization charts
- **Economic Trends**: Price, volume, market cap graphs
- **DeFi Ecosystem**: TVL, protocol count, yield farming
- **User Activity**: Address growth, retention rates
- **Security Metrics**: Decentralization, risk scores
- **Comprehensive Radar**: Multi-dimensional network health

## ⚙️ Configuration Options

### Data Collection Intervals
```bash
NETWORK_PERFORMANCE_INTERVAL=60      # 1 minute
ECONOMIC_DATA_INTERVAL=300           # 5 minutes
DEFI_METRICS_INTERVAL=600            # 10 minutes
SMART_CONTRACT_INTERVAL=900          # 15 minutes
STAKING_GOVERNANCE_INTERVAL=1800     # 30 minutes
USER_ACTIVITY_INTERVAL=120           # 2 minutes
NETWORK_HEALTH_INTERVAL=300          # 5 minutes
```

### Server Configuration
```bash
API_HOST=0.0.0.0
API_PORT=8008
API_WORKERS=4
COLLECTION_INTERVAL=300              # 5 minutes
BATCH_SIZE=100
METRICS_RETENTION_DAYS=90
```

### TRON-Specific Settings
```bash
TRON_NETWORK=mainnet
TRON_CHAIN_ID=728
TRON_NATIVE_TOKEN=TRX
TRON_DECIMALS=6
TRON_ENERGY_COST=420
TRON_BANDWIDTH_COST=1
```

## 🚀 Production Deployment

### Using PM2 (Recommended)

```bash
# Start with PM2
pm2 start ecosystem.tron.config.js

# Monitor
pm2 monit

# View logs
pm2 logs tron-metrics-server

# Restart
pm2 restart tron-metrics-server

# Stop
pm2 stop tron-metrics-server
```

### Using Docker

```bash
# Build image
docker build -t tron-metrics-server .

# Run container
docker run -d \
  --name tron-metrics-server \
  -p 8008:8008 \
  -v $(pwd)/tron_config.env:/app/tron_config.env \
  -v $(pwd)/logs:/app/logs \
  tron-metrics-server
```

### Using Systemd

```bash
# Create service file
sudo nano /etc/systemd/system/tron-metrics-server.service

# Enable and start
sudo systemctl enable tron-metrics-server
sudo systemctl start tron-metrics-server
```

## 📈 Monitoring and Alerts

### Health Checks
- **Endpoint Monitoring**: `/health` endpoint for uptime checks
- **Database Connectivity**: Connection status monitoring
- **API Response Times**: Performance monitoring
- **Error Rate Tracking**: Failure rate monitoring

### Logging
- **Structured Logging**: JSON-formatted logs
- **Log Rotation**: Automatic log rotation and retention
- **Error Tracking**: Detailed error logging and stack traces
- **Performance Metrics**: Request timing and resource usage

### Alerting (Optional)
```bash
ALERT_ENABLED=true
ALERT_WEBHOOK_URL=https://your-webhook-url
ALERT_EMAIL=admin@yourdomain.com
ALERT_THRESHOLD_ERROR_RATE=0.1
ALERT_THRESHOLD_RESPONSE_TIME=5.0
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database URL and credentials
   psql -h localhost -U defimon -d tron_metrics_db
   ```

2. **QuickNode Endpoint Issues**
   ```bash
   # Test endpoint connectivity
   curl -X POST https://your-endpoint/jsonrpc \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
   ```

3. **Port Already in Use**
   ```bash
   # Check port usage
   sudo netstat -tlnp | grep :8008
   
   # Kill process or change port
   export API_PORT=8009
   ```

4. **Memory Issues**
   ```bash
   # Monitor memory usage
   pm2 monit
   
   # Increase memory limit
   pm2 restart tron-metrics-server --max-memory-restart 2G
   ```

### Performance Optimization

1. **Database Optimization**
   - Enable connection pooling
   - Add database indexes
   - Regular VACUUM and ANALYZE

2. **API Optimization**
   - Enable response caching
   - Implement rate limiting
   - Use async processing

3. **Resource Management**
   - Monitor memory usage
   - Set appropriate batch sizes
   - Configure log retention

## 📚 Additional Resources

- **TRON Documentation**: https://developers.tron.network/
- **QuickNode TRON API**: https://www.quicknode.com/docs/tron
- **TRON Explorer**: https://tronscan.org/
- **TRON DeFi Protocols**: https://defillama.com/chain/Tron

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs in `logs/` directory
3. Verify configuration in `tron_config.env`
4. Test API endpoints manually

## 📄 License

This project is part of the ETHL2 monitoring system and follows the same licensing terms.

---

**🚀 Happy Monitoring! Track TRON network performance with precision and insight.**
