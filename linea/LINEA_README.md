# LINEA Blockchain Data Collection System

A comprehensive system for collecting, storing, and serving LINEA blockchain data with real-time monitoring and archive collection capabilities.

## 🚀 Features

- **Real-time Data Collection**: 10 concurrent workers collecting blocks, transactions, accounts, contracts, tokens, and DeFi data
- **Archive Collection**: Complete blockchain history from genesis to current block
- **REST API Server**: FastAPI-based metrics server with comprehensive endpoints
- **Database Storage**: SQLite databases for both real-time and archive data
- **Progress Tracking**: Real-time progress monitoring and statistics
- **Error Handling**: Robust error handling with retry logic and rate limiting

## 📁 System Components

### 1. Configuration
- `linea_config.env` - Environment configuration with RPC endpoints and settings

### 2. Database Schema
- `linea_database_schema.py` - Creates comprehensive database schema for LINEA data
- `linea_data.db` - Real-time data database
- `linea_archive_data.db` - Archive data database

### 3. Data Collectors
- `linea_data_collector.py` - Real-time data collector with 10 concurrent workers
- `linea_archive_collector.py` - Archive data collector for complete chain history

### 4. API Server
- `linea_metrics_server.py` - FastAPI server serving blockchain data and metrics

### 5. System Runner
- `run_linea_system.py` - Orchestrates the complete system

## 🛠️ Setup Instructions

### Prerequisites
```bash
pip install asyncio aiohttp web3 fastapi uvicorn sqlite3
```

### 1. Configure Environment
The system uses LINEA QuickNode endpoints:
- HTTP: `https://dry-special-card.linea-mainnet.quiknode.pro/1a758dced4242faa7d8c05445418dda72ba6e403/`
- WebSocket: `wss://dry-special-card.linea-mainnet.quiknode.pro/1a758dced4242faa7d8c05445418dda72ba6e403/`

### 2. Setup Database
```bash
python linea_database_schema.py
```

### 3. Run Complete System
```bash
python run_linea_system.py
```

### 4. Run Individual Components

#### Real-time Data Collector
```bash
python linea_data_collector.py
```

#### Archive Data Collector
```bash
python linea_archive_collector.py
```

#### Metrics Server
```bash
python linea_metrics_server.py --host 0.0.0.0 --port 8008
```

## 📊 API Endpoints

### Base URL: `http://localhost:8008`

#### System Endpoints
- `GET /` - System information and available endpoints
- `GET /api/health` - API health check
- `GET /api/network/status` - Network status

#### Block Endpoints
- `GET /api/blocks/latest` - Latest block information
- `GET /api/blocks/{block_number}` - Get specific block
- `GET /api/blocks` - List blocks with pagination

#### Transaction Endpoints
- `GET /api/transactions/{tx_hash}` - Get transaction details
- `GET /api/transactions` - List transactions with filters

#### Account Endpoints
- `GET /api/accounts/{address}` - Get account information

#### Contract & Token Endpoints
- `GET /api/contracts` - List contracts
- `GET /api/tokens` - List tokens

#### DeFi Endpoints
- `GET /api/defi/protocols` - List DeFi protocols

#### Metrics Endpoints
- `GET /api/metrics/network` - Network metrics
- `GET /api/metrics/summary` - Metrics summary
- `GET /api/stats` - Collection statistics

## 📈 Data Collection

### Real-time Collection (10 Workers)
1. **Workers 1-3**: Block data collection
2. **Workers 4-6**: Account data collection
3. **Workers 7-8**: Token data collection
4. **Workers 9-10**: DeFi protocol data collection

### Archive Collection
- Collects complete blockchain history from genesis
- Uses 10 concurrent workers for parallel processing
- Progress tracking and error handling
- Batch processing with configurable batch sizes

## 🗄️ Database Schema

### Real-time Tables
- `linea_blocks` - Block data
- `linea_transactions` - Transaction data
- `linea_transaction_receipts` - Transaction receipts
- `linea_network_metrics` - Network metrics
- `linea_accounts` - Account data
- `linea_contracts` - Contract data
- `linea_tokens` - Token data
- `linea_defi_protocols` - DeFi protocol data
- `linea_bridge_transactions` - Bridge transactions

### Archive Tables
- Same structure with `linea_archive_` prefix
- `linea_archive_progress` - Collection progress tracking

## ⚙️ Configuration Options

### Collection Intervals
- Block collection: 2 seconds
- Transaction collection: 1 second
- Account collection: 5 seconds
- Contract collection: 10 seconds
- Token collection: 15 seconds
- DeFi collection: 30 seconds

### Rate Limiting
- RPC rate limit: 100 requests/second
- WebSocket rate limit: 50 requests/second

### Archive Settings
- Batch size: 1000 blocks
- Concurrent workers: 10
- Max retries: 3
- Retry delay: 5 seconds

## 📊 Monitoring

### Logs
- `linea_collector.log` - Real-time collector logs
- `linea_archive_collector.log` - Archive collector logs

### Statistics
Real-time statistics available via API:
- Blocks collected
- Transactions collected
- Accounts collected
- Contracts collected
- Tokens collected
- DeFi protocols collected
- Error counts
- Progress percentage

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure SQLite files are writable
   - Check disk space

2. **RPC Connection Errors**
   - Verify LINEA RPC endpoints
   - Check network connectivity
   - Monitor rate limiting

3. **Memory Issues**
   - Adjust batch sizes
   - Monitor system resources
   - Consider database optimization

### Performance Tuning

1. **Increase Workers**
   - Modify `CONCURRENT_WORKERS` in config
   - Monitor system resources

2. **Adjust Batch Sizes**
   - Increase `ARCHIVE_BATCH_SIZE` for faster processing
   - Decrease for lower memory usage

3. **Rate Limiting**
   - Adjust `LINEA_RPC_RATE_LIMIT` based on provider limits
   - Monitor for rate limit errors

## 📚 Documentation

- API Documentation: `http://localhost:8008/docs`
- ReDoc Documentation: `http://localhost:8008/redoc`

## 🔒 Security

- Rate limiting to prevent API abuse
- Input validation on all endpoints
- Error handling without sensitive data exposure
- CORS configuration for web access

## 📞 Support

For issues or questions:
1. Check logs for error messages
2. Verify configuration settings
3. Monitor system resources
4. Review API documentation

## 🚀 Getting Started

1. **Quick Start**
   ```bash
   # Setup database
   python linea_database_schema.py
   
   # Run complete system
   python run_linea_system.py
   ```

2. **Access API**
   - Open browser to `http://localhost:8008`
   - View API docs at `http://localhost:8008/docs`

3. **Monitor Progress**
   - Check logs for collection progress
   - Use API endpoints for real-time statistics

## 📈 Performance Expectations

- **Real-time Collection**: ~2-5 blocks per second
- **Archive Collection**: ~100-500 blocks per second (depending on system)
- **API Response Time**: <100ms for most endpoints
- **Database Size**: ~1-10GB per million blocks (depending on data density)

## 🔄 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LINEA RPC     │    │  Data Collector │    │   SQLite DB     │
│   Endpoints     │◄───┤   (10 Workers)  │◄───┤  (Real-time)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Archive Collector│    │   SQLite DB     │
                       │   (10 Workers)   │◄───┤   (Archive)     │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Metrics Server │
                       │   (FastAPI)     │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   REST API      │
                       │  (Port 8008)    │
                       └─────────────────┘
```

This system provides a complete solution for LINEA blockchain data collection, storage, and serving with high performance and reliability.
