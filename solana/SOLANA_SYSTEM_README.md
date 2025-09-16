# Solana Blockchain Data Collection System

A comprehensive Solana blockchain data collection and metrics system with 10 concurrent workers, complete historical archive, and real-time monitoring capabilities.

## 🚀 Features

- **10 Concurrent Workers**: High-performance data collection with parallel processing
- **Complete Historical Archive**: Full blockchain data from genesis to present
- **Real-time Data Collection**: Live blockchain monitoring and metrics
- **REST API**: Comprehensive API for accessing all data
- **WebSocket Support**: Real-time data streaming
- **Prometheus Metrics**: Monitoring and alerting integration
- **Database Optimization**: SQLite with WAL mode and indexing
- **Rate Limiting**: Intelligent request throttling
- **Error Handling**: Robust retry logic and error recovery

## 📁 System Components

### Core Files

- `solana_config.env` - Configuration file with Linea Mainnet endpoints
- `solana_database_schema.py` - Database schema creation
- `solana_archive_collector.py` - Historical data collector (10 workers)
- `solana_comprehensive_server.py` - Main metrics server
- `test_solana_system.py` - System test suite

### Management Scripts

- `start_solana_system.sh` - Start all services
- `stop_solana_system.sh` - Stop all services
- `check_solana_status.sh` - Check system status

### Database Files

- `solana_data.db` - Main database (real-time data)
- `solana_archive_data.db` - Archive database (historical data)

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7+
- Required packages: `aiohttp`, `sqlite3` (built-in)

### Quick Start

1. **Install dependencies**:
   ```bash
   pip3 install aiohttp
   ```

2. **Initialize databases**:
   ```bash
   python3 solana_database_schema.py
   ```

3. **Start the system**:
   ```bash
   ./start_solana_system.sh
   ```

4. **Check status**:
   ```bash
   ./check_solana_status.sh
   ```

5. **Run tests**:
   ```bash
   python3 test_solana_system.py
   ```

## 🌐 API Endpoints

### Main API (Port 8001)

- `GET /` - Service information and available endpoints
- `GET /api/blocks` - Get blocks (supports `?limit=10&archive=true`)
- `GET /api/transactions` - Get transactions (supports `?limit=100&archive=true`)
- `GET /api/network_metrics` - Get network metrics
- `GET /api/validators` - Get validator information
- `GET /api/programs` - Get program information
- `GET /api/archive` - Archive data access
- `GET /api/stats` - System statistics
- `GET /ws` - WebSocket endpoint

### Metrics API (Port 9091)

- `GET /metrics` - Prometheus metrics

## 📊 Database Schema

### Main Tables

- `solana_blocks` - Block data
- `solana_transactions` - Transaction data
- `solana_accounts` - Account information
- `solana_token_accounts` - Token account data
- `solana_tokens` - Token metadata
- `solana_programs` - Program information
- `solana_validators` - Validator data
- `solana_network_metrics` - Network performance metrics

### Archive Tables

- `solana_archive_blocks` - Historical block data
- `solana_archive_transactions` - Historical transaction data
- `solana_archive_network_metrics` - Historical network metrics
- `solana_archive_progress` - Collection progress tracking

## ⚙️ Configuration

The system is configured via `solana_config.env`:

```env
# Solana RPC Configuration (Using Linea Mainnet endpoints)
SOLANA_RPC_URL=https://dry-special-card.linea-mainnet.quiknode.pro/1a758dced4242faa7d8c05445418dda72ba6e403/
SOLANA_WSS_URL=wss://dry-special-card.linea-mainnet.quiknode.pro/1a758dced4242faa7d8c05445418dda72ba6e403/

# Database Configuration
SOLANA_DATABASE_PATH=solana_data.db
SOLANA_ARCHIVE_DATABASE_PATH=solana_archive_data.db

# Archive collection settings
ARCHIVE_START_SLOT=0
ARCHIVE_BATCH_SIZE=1000
ARCHIVE_CONCURRENT_WORKERS=10
ARCHIVE_MAX_RETRIES=3
ARCHIVE_RETRY_DELAY=5

# Server configuration
API_HOST=0.0.0.0
API_PORT=8001
```

## 🔄 Data Collection Process

### Archive Collection (10 Workers)

1. **Worker Distribution**: Slots are distributed evenly across 10 workers
2. **Batch Processing**: Each worker processes blocks in batches of 1000
3. **Rate Limiting**: Intelligent throttling to respect API limits
4. **Error Handling**: Automatic retry with exponential backoff
5. **Progress Tracking**: Real-time progress monitoring

### Real-time Collection

- Continuous monitoring of latest blocks
- Network metrics collection
- Validator information updates
- Token and program data tracking

## 📈 Monitoring & Metrics

### Prometheus Metrics

- `solana_blocks_total` - Total blocks collected
- `solana_transactions_total` - Total transactions collected
- `solana_accounts_total` - Total accounts collected
- `solana_validators_total` - Total validators
- `solana_latest_slot` - Latest slot number
- `solana_server_uptime_seconds` - Server uptime
- `solana_requests_served_total` - Total requests served
- `solana_websocket_connections_active` - Active WebSocket connections

### WebSocket Streaming

Real-time data streaming for:
- Latest blocks
- Network metrics
- Archive progress
- System statistics

## 🧪 Testing

The system includes comprehensive tests:

```bash
python3 test_solana_system.py
```

Tests cover:
- Database connectivity
- API endpoints
- WebSocket connections
- Metrics endpoints
- Archive data access
- Data collection simulation

## 📝 Logging

Logs are stored in the `logs/` directory:
- `solana_server.log` - Main server logs
- `solana_archive.log` - Archive collector logs

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   ./stop_solana_system.sh
   ./start_solana_system.sh
   ```

2. **Database locked**:
   ```bash
   pkill -f solana
   rm -f solana_data.db-wal solana_data.db-shm
   ```

3. **Out of memory**:
   - Reduce `ARCHIVE_BATCH_SIZE` in config
   - Reduce `ARCHIVE_CONCURRENT_WORKERS`

### Status Check

```bash
./check_solana_status.sh
```

## 📊 Performance

### Expected Performance

- **Archive Collection**: ~1000 blocks/second with 10 workers
- **Real-time Collection**: ~100 blocks/second
- **API Response Time**: <100ms for most endpoints
- **Database Size**: ~1GB per 1M blocks

### Optimization Tips

1. **Database**: Use SSD storage for better I/O performance
2. **Memory**: Allocate at least 4GB RAM for large datasets
3. **Network**: Use high-bandwidth connection for archive collection
4. **CPU**: Multi-core system recommended for concurrent workers

## 🔒 Security

- Rate limiting to prevent API abuse
- Input validation on all endpoints
- SQL injection protection
- CORS headers for web access

## 📚 API Examples

### Get Latest Blocks

```bash
curl "http://localhost:8001/api/blocks?limit=10"
```

### Get Archive Blocks

```bash
curl "http://localhost:8001/api/archive?type=blocks&start_slot=1000&end_slot=2000"
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');
ws.send(JSON.stringify({command: 'subscribe_blocks'}));
```

### Get System Stats

```bash
curl "http://localhost:8001/api/stats"
```

## 🎯 Use Cases

- **Blockchain Analytics**: Historical data analysis
- **Network Monitoring**: Real-time performance tracking
- **DeFi Research**: Transaction pattern analysis
- **Validator Monitoring**: Staking and performance metrics
- **Token Tracking**: Supply and distribution analysis

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Run `./check_solana_status.sh` for system status
3. Run `python3 test_solana_system.py` for diagnostics

## 🔄 Updates

To update the system:
1. Stop services: `./stop_solana_system.sh`
2. Update code
3. Restart services: `./start_solana_system.sh`

---

**Note**: This system uses Linea Mainnet endpoints as specified. The archive collection will pull the complete Solana blockchain history using 10 concurrent workers for maximum efficiency.


