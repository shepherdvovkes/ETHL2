# 🚀 Avalanche Network Real-Time Metrics Server

A comprehensive real-time monitoring and analytics system for the Avalanche network, continuously gathering data from external APIs and providing detailed insights into network performance, economic metrics, DeFi ecosystem, and more.

## 📊 Overview

This server provides real-time monitoring of **12 key parameter categories** for the Avalanche network:

1. **Network Performance** - TPS, gas prices, finality, utilization
2. **Economic Metrics** - Price, volume, market cap, supply
3. **DeFi Ecosystem** - TVL, protocols, yields, volumes
4. **Subnet Analysis** - Count, activity, validators
5. **Security Metrics** - Validators, staking, audits
6. **Development Activity** - GitHub, contracts, launches
7. **User Behavior** - Whales, retail vs institutional
8. **Competitive Position** - Market share, performance
9. **Technical Health** - RPC, uptime, infrastructure
10. **Risk Assessment** - Centralization, technical, market
11. **Macro Environment** - Market conditions, regulations
12. **Ecosystem Health** - Community, partnerships

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Avalanche Metrics Server                    │
├─────────────────────────────────────────────────────────────┤
│  Real-Time Data Collector  │  Monitoring System  │  API Server │
│  • Network Performance     │  • Alert Rules      │  • FastAPI   │
│  • Economic Data          │  • Notifications    │  • Endpoints │
│  • DeFi Metrics           │  • Thresholds       │  • Documentation │
│  • Subnet Data            │  • Cooldowns        │  • Health Check │
│  • Security Status        │  • Multi-channel    │  • Historical Data │
│  • Development Activity   │  • Email/Slack/Telegram │  • Real-time Data │
│  • User Behavior          │  • Webhooks         │  • Export/Import │
│  • Competitive Analysis   │  • Risk Assessment  │  • Authentication │
│  • Technical Health       │  • Performance      │  • Rate Limiting │
│  • Risk Indicators        │  • Reliability      │  • CORS Support │
│  • Macro Environment      │  • Scalability      │  • Error Handling │
│  • Ecosystem Health       │  • Monitoring       │  • Logging │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Data Sources                    │
├─────────────────────────────────────────────────────────────┤
│  • Avalanche RPC APIs      │  • CoinGecko API              │
│  • Snowtrace Explorer      │  • DeFiLlama API              │
│  • P-Chain & X-Chain       │  • GitHub API                 │
│  • QuickNode API           │  • Social Media APIs          │
│  • Etherscan API           │  • News & Sentiment APIs      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Storage                          │
├─────────────────────────────────────────────────────────────┤
│  • PostgreSQL Database     │  • Redis Cache                │
│  • Time-series Data        │  • Real-time Metrics          │
│  • Historical Data         │  • Session Storage            │
│  • Alert History           │  • Rate Limiting              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6+ (optional but recommended)
- API keys for external services

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ETHL2

# Install dependencies
pip install -r requirements.avalanche.txt

# Run setup script
python setup_avalanche_server.py
```

### 3. Configuration

```bash
# Copy configuration template
cp avalanche_config.env config.env

# Edit configuration with your API keys
nano config.env
```

**Required API Keys:**
- CoinGecko API key (for market data)
- QuickNode API key (for blockchain data)
- Email credentials (for alerts)
- Optional: Slack, Telegram, webhook URLs

### 4. Start the Server

```bash
# Development mode (all components)
python run_avalanche_server.py --mode full

# Production mode with systemd
sudo systemctl start avalanche-metrics

# Docker mode
docker-compose -f docker-compose.avalanche.yml up -d
```

### 5. Access the Server

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics Summary**: http://localhost:8000/metrics/summary

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Description | Collection Interval |
|----------|-------------|-------------------|
| `/metrics/summary` | Key metrics overview | Real-time |
| `/metrics/network-performance` | TPS, gas prices, finality | 30 seconds |
| `/metrics/economic` | Price, volume, market cap | 1 minute |
| `/metrics/defi` | DeFi TVL, protocols, yields | 2 minutes |
| `/metrics/subnets` | Subnet count, activity | 5 minutes |
| `/metrics/security` | Validators, staking, audits | 10 minutes |
| `/metrics/development` | GitHub activity, contracts | 30 minutes |
| `/metrics/user-behavior` | Whales, retail vs institutional | 5 minutes |
| `/metrics/competitive` | Market share, performance | 1 hour |
| `/metrics/technical` | RPC health, uptime | 1 minute |
| `/metrics/risk` | Risk indicators, assessment | 30 minutes |
| `/metrics/macro` | Market conditions, regulations | 30 minutes |
| `/metrics/ecosystem` | Community, partnerships | 1 hour |
| `/metrics/all` | Complete metrics dataset | Real-time |

### Utility Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Server health status |
| `/collect` | Trigger manual data collection |
| `/historical/{hours}` | Historical data (max 168 hours) |

### Example API Calls

```bash
# Get network performance metrics
curl http://localhost:8000/metrics/network-performance

# Get economic data
curl http://localhost:8000/metrics/economic

# Get all metrics
curl http://localhost:8000/metrics/all

# Get historical data (last 24 hours)
curl http://localhost:8000/historical/24

# Trigger manual collection
curl -X POST http://localhost:8000/collect
```

## 🔍 Data Collection Intervals

| Metric Category | Collection Interval | Description |
|----------------|-------------------|-------------|
| Network Performance | 30 seconds | TPS, gas prices, block times |
| Economic Data | 1 minute | Price, volume, market cap |
| DeFi Metrics | 2 minutes | TVL, protocol activity |
| Subnet Data | 5 minutes | Subnet count, validator status |
| Security Status | 10 minutes | Validator count, staking ratio |
| Development Activity | 30 minutes | GitHub commits, contract deployments |
| User Behavior | 5 minutes | Transaction patterns, whale activity |
| Competitive Position | 1 hour | Market share, competitor analysis |
| Technical Health | 1 minute | RPC performance, endpoint status |
| Risk Indicators | 30 minutes | Risk assessment, volatility |
| Macro Environment | 30 minutes | Market conditions, regulations |
| Ecosystem Health | 1 hour | Community growth, partnerships |

## 🚨 Monitoring & Alerting

### Alert Types

- **Network Performance**: High gas prices, low throughput, high utilization
- **Economic Metrics**: High volatility, extreme price drops, low volume
- **Security Issues**: Low validator count, low staking ratio
- **Technical Failures**: Slow RPC response, low health score
- **Market Anomalies**: Unusual price movements, volume spikes
- **DeFi Risks**: TVL drops, protocol issues
- **Subnet Issues**: Subnet failures, validator problems
- **API Failures**: External API downtime, data collection errors

### Alert Channels

- **Email**: SMTP notifications with detailed reports
- **Slack**: Rich formatted messages with metrics
- **Telegram**: Bot notifications with inline keyboards
- **Webhooks**: Custom HTTP endpoints for integration
- **Logs**: Structured logging for monitoring systems

### Alert Configuration

```python
# Example alert rule
MonitoringRule(
    name="high_gas_price",
    metric_type="network_performance",
    metric_name="gas_price_avg",
    condition="greater_than",
    threshold=100.0,  # 100 Gwei
    alert_level=AlertLevel.WARNING,
    alert_type=AlertType.NETWORK_PERFORMANCE,
    cooldown_minutes=30
)
```

## 🗄️ Database Schema

### Core Tables

- **`blockchains`** - Supported blockchain networks
- **`network_metrics`** - Network performance data
- **`economic_metrics`** - Economic and market data
- **`security_metrics`** - Security and validator data
- **`ecosystem_metrics`** - Ecosystem and community data
- **`alert_history`** - Alert and notification history

### Data Retention

- **Real-time data**: 7 days
- **Hourly aggregates**: 30 days
- **Daily aggregates**: 1 year
- **Alert history**: 90 days

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
COINGECKO_API_KEY=your_key_here
QUICKNODE_API_KEY=your_key_here
ETHERSCAN_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379

# Server
API_HOST=0.0.0.0
API_PORT=8000

# Alerting
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SLACK_WEBHOOK=https://hooks.slack.com/services/...
TELEGRAM_BOT_TOKEN=your_bot_token
```

### Collection Intervals

```bash
# Adjust collection intervals (in seconds)
NETWORK_PERFORMANCE_INTERVAL=30
ECONOMIC_DATA_INTERVAL=60
DEFI_METRICS_INTERVAL=120
# ... etc
```

### Alert Thresholds

```bash
# Network performance thresholds
HIGH_GAS_PRICE_THRESHOLD=100.0
LOW_THROUGHPUT_THRESHOLD=1000
HIGH_NETWORK_UTILIZATION_THRESHOLD=90.0

# Economic thresholds
HIGH_PRICE_VOLATILITY_THRESHOLD=20.0
EXTREME_PRICE_DROP_THRESHOLD=-30.0
LOW_VOLUME_THRESHOLD=100000000
```

## 🐳 Docker Deployment

### Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.avalanche.yml up -d

# View logs
docker-compose -f docker-compose.avalanche.yml logs -f

# Stop services
docker-compose -f docker-compose.avalanche.yml down
```

### Dockerfile

```bash
# Build image
docker build -f Dockerfile.avalanche -t avalanche-metrics .

# Run container
docker run -d \
  --name avalanche-metrics \
  -p 8000:8000 \
  -v $(pwd)/config.env:/app/config.env \
  -v $(pwd)/logs:/app/logs \
  avalanche-metrics
```

## 🔧 Production Deployment

### Systemd Service

```bash
# Install service
sudo cp avalanche-metrics.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable avalanche-metrics
sudo systemctl start avalanche-metrics

# Check status
sudo systemctl status avalanche-metrics

# View logs
sudo journalctl -u avalanche-metrics -f
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring Dashboard

### Key Metrics to Monitor

1. **Server Health**
   - API response times
   - Database connection status
   - Memory and CPU usage
   - Error rates

2. **Data Collection**
   - Collection success rates
   - API response times
   - Data freshness
   - Queue sizes

3. **Alerting**
   - Alert frequency
   - Notification delivery rates
   - Alert resolution times
   - False positive rates

### Recommended Monitoring Tools

- **Prometheus + Grafana**: Metrics collection and visualization
- **ELK Stack**: Log aggregation and analysis
- **Uptime Robot**: External monitoring
- **PagerDuty**: Incident management

## 🔒 Security Considerations

### API Security

- Rate limiting on all endpoints
- CORS configuration
- Input validation and sanitization
- Error handling without information leakage

### Data Security

- Encrypted database connections
- Secure API key storage
- Regular security updates
- Access logging and monitoring

### Network Security

- Firewall configuration
- VPN access for sensitive operations
- Regular security audits
- Incident response procedures

## 🚀 Performance Optimization

### Database Optimization

- Proper indexing on time-series data
- Connection pooling
- Query optimization
- Regular maintenance

### API Optimization

- Response caching
- Compression
- Pagination for large datasets
- Async processing

### Resource Management

- Memory usage monitoring
- CPU optimization
- Network bandwidth management
- Storage optimization

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Test connection
   psql -h localhost -U defimon -d defimon_db
   ```

2. **API Key Issues**
   ```bash
   # Verify API keys in config.env
   grep -E "API_KEY|TOKEN" config.env
   
   # Test API connectivity
   curl -H "x-cg-pro-api-key: YOUR_KEY" https://api.coingecko.com/api/v3/ping
   ```

3. **High Memory Usage**
   ```bash
   # Monitor memory usage
   htop
   
   # Check for memory leaks
   python -m memory_profiler run_avalanche_server.py
   ```

4. **Collection Failures**
   ```bash
   # Check logs
   tail -f logs/avalanche_server_*.log
   
   # Test individual collectors
   python -c "from avalanche_realtime_server import RealTimeDataCollector; import asyncio; asyncio.run(RealTimeDataCollector().collect_network_performance())"
   ```

### Log Analysis

```bash
# View recent errors
grep "ERROR" logs/avalanche_server_*.log | tail -20

# Monitor real-time logs
tail -f logs/avalanche_server_$(date +%Y-%m-%d).log

# Analyze collection performance
grep "Collection completed" logs/avalanche_server_*.log | tail -10
```

## 📈 Scaling Considerations

### Horizontal Scaling

- Load balancer configuration
- Database replication
- Redis clustering
- Microservices architecture

### Vertical Scaling

- CPU and memory upgrades
- SSD storage
- Network bandwidth
- Connection pooling

### Data Archiving

- Time-series data compression
- Historical data archiving
- Backup strategies
- Disaster recovery

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.avalanche.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

### Adding New Metrics

1. Add metric collection method to `RealTimeDataCollector`
2. Update database schema if needed
3. Add API endpoint to `avalanche_api_server.py`
4. Add monitoring rules if applicable
5. Update documentation

### Adding New Alert Channels

1. Implement notification method in `AlertManager`
2. Add configuration options
3. Update setup script
4. Add tests and documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation

- API Documentation: http://localhost:8000/docs
- Code Documentation: Inline comments and docstrings
- Configuration Guide: This README

### Community

- GitHub Issues: For bug reports and feature requests
- Discussions: For questions and community support
- Wiki: For additional documentation and guides

### Professional Support

For enterprise support, custom development, or consulting services, please contact the development team.

---

**🎯 Ready to monitor Avalanche network in real-time!**

Start the server and begin collecting comprehensive metrics about the Avalanche ecosystem. The system will continuously gather data, provide insights, and alert you to important changes in the network.
