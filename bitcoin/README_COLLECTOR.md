# Bitcoin Chain Collector

A high-performance Bitcoin blockchain data collector that uses QuickNode API to collect the complete Bitcoin blockchain data locally using 10 parallel workers.

## 🚀 Features

- **10 Parallel Workers**: Collects blockchain data using 10 concurrent workers for maximum speed
- **QuickNode API Integration**: Uses your QuickNode endpoint for reliable, fast data access
- **Local SQLite Database**: Stores all blockchain data in a local SQLite database
- **Real-time Progress Monitoring**: Track collection progress with detailed statistics
- **Error Handling**: Robust error handling with automatic retries
- **Resume Capability**: Can resume collection from the last collected block
- **Complete Data**: Collects both blocks and transactions with full metadata

## 📋 Requirements

- Python 3.8+
- Internet connection
- ~500GB free disk space (for full blockchain)
- QuickNode API endpoint

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r collector_requirements.txt
```

### 2. Start Collection
```bash
# Run the collection script
./start_collection.sh
```

Or run directly:
```bash
python3 run_collector.py
```

### 3. Monitor Progress
```bash
# View real-time progress
python3 monitor_collector.py

# Continuous monitoring
python3 monitor_collector.py --continuous 30

# Save progress to JSON
python3 monitor_collector.py --json progress.json
```

## 📁 Files Created

- `bitcoin_chain.db` - SQLite database with all blockchain data
- `bitcoin_collector.log` - Collection logs
- `collection_progress.json` - Progress data (if saved)

## 🗄️ Database Schema

### Blocks Table
- `height` - Block height
- `hash` - Block hash
- `previous_hash` - Previous block hash
- `timestamp` - Block timestamp
- `size` - Block size in bytes
- `weight` - Block weight
- `version` - Block version
- `nonce` - Block nonce
- `bits` - Block bits
- `difficulty` - Block difficulty
- `merkle_root` - Merkle root
- `tx_count` - Number of transactions
- `raw_data` - Complete raw block data (JSON)

### Transactions Table
- `txid` - Transaction ID
- `block_height` - Block height
- `block_hash` - Block hash
- `size` - Transaction size
- `weight` - Transaction weight
- `fee` - Transaction fee
- `input_count` - Number of inputs
- `output_count` - Number of outputs
- `raw_data` - Complete raw transaction data (JSON)

### Collection Progress Table
- `worker_id` - Worker identifier
- `start_height` - Starting block height
- `end_height` - Ending block height
- `status` - Collection status
- `blocks_collected` - Number of blocks collected
- `transactions_collected` - Number of transactions collected

## ⚙️ Configuration

Edit `collector_config.py` to customize:

```python
# Number of parallel workers
NUM_WORKERS = 10

# Database path
DATABASE_PATH = "bitcoin_chain.db"

# Start height (0 = genesis block)
START_HEIGHT = 0

# Rate limiting (seconds between requests)
RATE_LIMIT_DELAY = 0.1

# Progress update interval
PROGRESS_UPDATE_INTERVAL = 30
```

## 📊 Performance

### Expected Performance
- **Collection Speed**: 1000+ blocks/hour per worker
- **Total Speed**: 10,000+ blocks/hour with 10 workers
- **Full Chain**: ~3-5 days for complete Bitcoin blockchain
- **Database Size**: ~400-500GB for full blockchain
- **Memory Usage**: ~2-4GB during collection

### Optimization Tips
1. **SSD Storage**: Use SSD for faster database writes
2. **Sufficient RAM**: 8GB+ recommended
3. **Stable Internet**: Ensure stable connection to QuickNode
4. **Monitor Resources**: Watch CPU and memory usage

## 🔧 Management Commands

### Start Collection
```bash
./start_collection.sh
```

### Monitor Progress
```bash
# Basic progress report
python3 monitor_collector.py

# Continuous monitoring every 30 seconds
python3 monitor_collector.py --continuous 30

# Save progress to file
python3 monitor_collector.py --json progress.json
```

### Database Queries
```bash
# Connect to SQLite database
sqlite3 bitcoin_chain.db

# Example queries
SELECT COUNT(*) FROM blocks;
SELECT COUNT(*) FROM transactions;
SELECT MAX(height) FROM blocks;
SELECT * FROM collection_progress;
```

## 🚨 Troubleshooting

### Common Issues

#### Collection Stops
- Check internet connection
- Verify QuickNode endpoint is working
- Check logs for error messages
- Ensure sufficient disk space

#### Slow Performance
- Check system resources (CPU, RAM, disk)
- Verify network speed
- Consider reducing number of workers
- Check for disk I/O bottlenecks

#### Database Errors
- Ensure sufficient disk space
- Check file permissions
- Verify SQLite installation
- Check for disk corruption

### Log Analysis
```bash
# View recent logs
tail -f bitcoin_collector.log

# Search for errors
grep -i error bitcoin_collector.log

# Check worker progress
grep "Worker" bitcoin_collector.log
```

## 📈 Monitoring

### Real-time Statistics
- Blocks collected per worker
- Transactions collected per worker
- Collection rate (blocks/hour)
- Database size
- System resources

### Progress Tracking
- Worker status (started/completed/failed)
- Height ranges assigned to workers
- Time estimates for completion
- Error counts and types

## 🔒 Security

- All data stored locally
- No sensitive data transmitted
- Uses HTTPS for API calls
- Database file permissions controlled

## 📚 API Reference

### QuickNodeClient
- `get_blockchain_info()` - Get blockchain information
- `get_block_hash(height)` - Get block hash by height
- `get_block(hash, verbosity)` - Get block data
- `get_raw_transaction(txid)` - Get transaction data

### BitcoinDatabase
- `save_block(block_data)` - Save block to database
- `save_transaction(tx_data)` - Save transaction to database
- `get_collection_progress()` - Get progress statistics

### BitcoinWorker
- `collect_range()` - Collect data for assigned height range
- `stop()` - Stop the worker

## 🆘 Support

For issues:
1. Check the logs: `bitcoin_collector.log`
2. Run the monitor: `python3 monitor_collector.py`
3. Check system resources
4. Verify QuickNode endpoint
5. Check disk space

## 📄 License

This collector is provided as-is for educational and development purposes.

---

**⚠️ Important Notes:**
- Initial collection can take several days
- Ensure sufficient disk space (500GB+)
- Monitor system resources during collection
- Keep QuickNode endpoint active
- Consider backup strategies for collected data
