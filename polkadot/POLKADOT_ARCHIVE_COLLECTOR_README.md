# Polkadot Archive Data Collector

A comprehensive archive data collector for Polkadot using QuickNode endpoints with 30 parallel workers for efficient historical data collection.

## Features

### 🚀 High-Performance Collection
- **30 Parallel Workers**: Optimized for maximum throughput
- **Batch Processing**: Efficient data collection in batches
- **Rate Limiting**: Respects API limits with configurable delays
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive error handling and logging

### 📊 Comprehensive Data Collection
- **Historical Block Data**: Complete block information with sampling
- **Staking Metrics**: Validator and nominator information
- **Parachain Data**: Parachain status and metrics
- **Governance Data**: Proposals, referendums, and council information
- **Cross-Chain Metrics**: HRMP channels and XCM data
- **Network Health**: System health and peer information

### 🗄️ Data Storage
- **SQLite Database**: Efficient local storage
- **Structured Tables**: Optimized schema for time-series data
- **Indexes**: Fast query performance
- **Data Validation**: Quality checks and validation

### ⚙️ Flexible Configuration
- **Predefined Configs**: Quick test, monthly, quarterly, yearly, comprehensive
- **Custom Parameters**: Override any configuration option
- **Command Line Interface**: Easy to use CLI
- **Environment Variables**: Support for environment-based configuration

## Quick Start

### 1. Basic Usage

```bash
# Run with default yearly configuration
python run_polkadot_archive_collector.py

# Run with specific configuration
python run_polkadot_archive_collector.py --config monthly

# Run with custom parameters
python run_polkadot_archive_collector.py --config yearly --days 30 --workers 20
```

### 2. Available Configurations

| Configuration | Days Back | Sample Rate | Workers | Use Case |
|---------------|-----------|-------------|---------|----------|
| `quick_test` | 7 | 100 | 10 | Testing and development |
| `monthly` | 30 | 5 | 20 | Monthly analysis |
| `quarterly` | 90 | 3 | 25 | Quarterly reports |
| `yearly` | 365 | 10 | 30 | Annual analysis |
| `comprehensive` | 365 | 1 | 30 | Complete historical data |

### 3. Command Line Options

```bash
python run_polkadot_archive_collector.py --help

Options:
  --config {quick_test,monthly,quarterly,yearly,comprehensive}
                        Predefined configuration to use
  --days DAYS           Number of days back to collect
  --workers WORKERS     Number of parallel workers
  --sample-rate RATE    Block sampling rate
  --database PATH       Database path
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
  --quicknode-url URL   QuickNode endpoint URL
```

## Architecture

### Components

1. **QuickNodePolkadotArchiveClient**: Enhanced QuickNode client with parallel processing
2. **PolkadotArchiveDatabase**: Database manager with optimized schema
3. **PolkadotArchiveCollector**: Main collector orchestrating the process
4. **CollectionConfig**: Configuration management system

### Data Flow

```
QuickNode API → Parallel Workers → Data Processing → Database Storage
     ↓              ↓                    ↓              ↓
  RPC Calls    Batch Processing    Validation    SQLite Tables
```

### Database Schema

#### Block Metrics Table
```sql
CREATE TABLE block_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_number INTEGER UNIQUE,
    timestamp TEXT,
    extrinsics_count INTEGER,
    events_count INTEGER,
    block_size INTEGER,
    validator_count INTEGER,
    finalization_time REAL,
    parachain_blocks INTEGER,
    cross_chain_messages INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### Staking Data Table
```sql
CREATE TABLE staking_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    validators_count INTEGER,
    nominators_count INTEGER,
    active_era INTEGER,
    total_staked REAL,
    inflation_rate REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### Parachain Data Table
```sql
CREATE TABLE parachain_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    parachains_count INTEGER,
    hrmp_channels_count INTEGER,
    active_parachains INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### Governance Data Table
```sql
CREATE TABLE governance_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    proposals_count INTEGER,
    referendums_count INTEGER,
    council_members_count INTEGER,
    treasury_proposals_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Performance Optimization

### Parallel Processing
- **30 Concurrent Workers**: Maximum parallelization
- **Batch Processing**: Process blocks in batches of 100
- **Semaphore Control**: Prevents overwhelming the API
- **Rate Limiting**: Configurable delays between requests

### Memory Management
- **Streaming Processing**: Process data in chunks
- **Connection Pooling**: Reuse HTTP connections
- **Garbage Collection**: Automatic cleanup of processed data

### Error Handling
- **Retry Logic**: 3 attempts with exponential backoff
- **Exception Handling**: Comprehensive error catching
- **Progress Tracking**: Real-time progress updates
- **Failure Recovery**: Continue processing after failures

## Configuration Examples

### Custom Configuration
```python
from polkadot_archive_config import create_custom_config

# Create custom configuration
config = create_custom_config(
    days_back=180,
    max_workers=25,
    sample_rate=5,
    batch_size=75,
    rate_limit_delay=0.05
)
```

### Environment Variables
```bash
export QUICKNODE_URL="https://your-quicknode-endpoint.com"
export MAX_WORKERS=30
export DAYS_BACK=365
export SAMPLE_RATE=10
```

## Data Analysis

### Querying the Database
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('polkadot_archive_data.db')

# Query block metrics
df = pd.read_sql_query("""
    SELECT 
        block_number,
        extrinsics_count,
        events_count,
        block_size,
        timestamp
    FROM block_metrics 
    WHERE block_number > 10000000
    ORDER BY block_number
""", conn)

# Analyze data
print(df.describe())
```

### Exporting Data
```python
# Export to CSV
df.to_csv('polkadot_block_metrics.csv', index=False)

# Export to Parquet
df.to_parquet('polkadot_block_metrics.parquet', index=False)
```

## Monitoring and Logging

### Log Files
- **Console Output**: Real-time progress and status
- **File Logging**: Detailed logs saved to `logs/` directory
- **Log Rotation**: Automatic rotation at 100MB
- **Retention**: 7 days of log history

### Progress Tracking
```
Progress: 45.2% - Collected: 45230, Failed: 123
Processing batch 23/50 (100 blocks)
```

### Performance Metrics
- **Collection Rate**: Blocks per second
- **Success Rate**: Percentage of successful collections
- **Error Rate**: Failed requests percentage
- **Duration**: Total collection time

## Troubleshooting

### Common Issues

1. **Rate Limiting Errors**
   - Increase `rate_limit_delay` in configuration
   - Reduce `max_workers` if needed
   - Check QuickNode plan limits

2. **Memory Issues**
   - Reduce `batch_size`
   - Increase `sample_rate` to collect fewer blocks
   - Monitor system memory usage

3. **Connection Errors**
   - Check QuickNode endpoint URL
   - Verify network connectivity
   - Increase `retry_attempts`

### Debug Mode
```bash
python run_polkadot_archive_collector.py --log-level DEBUG
```

## Best Practices

### Performance
- Use appropriate sample rates for your use case
- Monitor system resources during collection
- Use SSD storage for better database performance
- Consider network bandwidth limitations

### Data Quality
- Validate collected data regularly
- Monitor error rates and adjust configuration
- Keep backups of important datasets
- Use appropriate retention policies

### Security
- Secure your QuickNode API keys
- Use environment variables for sensitive data
- Regularly rotate API credentials
- Monitor API usage and costs

## API Reference

### QuickNodePolkadotArchiveClient
```python
async def make_rpc_call(method: str, params: List = None) -> Dict
async def get_current_block() -> int
async def collect_block_data(block_number: int) -> Dict
async def collect_staking_data() -> Dict
async def collect_parachain_data() -> Dict
async def collect_governance_data() -> Dict
```

### PolkadotArchiveDatabase
```python
def store_block_metrics(metrics: Dict)
def store_staking_data(data: Dict)
def store_parachain_data(data: Dict)
def store_governance_data(data: Dict)
```

### PolkadotArchiveCollector
```python
async def collect_historical_blocks(start_block: int, end_block: int) -> List[Dict]
async def collect_network_metrics()
async def run_comprehensive_collection()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Open an issue on GitHub
- Contact the development team

## Changelog

### Version 1.0.0
- Initial release
- 30 parallel workers support
- Comprehensive data collection
- SQLite database storage
- Command line interface
- Multiple configuration options
