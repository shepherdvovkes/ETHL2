# Grafana Setup Instructions

## Manual Setup Steps

### 1. Access Grafana
- URL: http://localhost:3000
- Username: admin
- Password: admin

### 2. Add Prometheus Data Source
1. Go to Configuration → Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL: `http://192.168.0.247:9091`
5. Click "Save & Test"

### 3. Import Dashboards
1. Go to "+" → Import
2. Copy and paste the dashboard JSON from the files:
   - `grafana-dashboard-comprehensive.json` - Main Bitcoin monitoring dashboard
   - Or create custom dashboards using the provided queries

### 4. Key Metrics to Monitor

#### Bitcoin Network Metrics
- `bitcoin_block_height` - Current Bitcoin block height
- `bitcoin_difficulty` - Network difficulty
- `bitcoin_mempool_size` - Mempool size
- `bitcoin_network_connections` - Network connections

#### Data Collection Metrics
- `bitcoin_collection_success_total` - Successful collections
- `bitcoin_collection_failures_total` - Failed collections
- `rate(bitcoin_collection_success_total[5m])` - Collection rate
- `bitcoin_collection_duration_seconds` - Collection timing

#### Data Storage Metrics
- `bitcoin_data_metrics_stored` - Total metrics stored
- `bitcoin_data_blocks_stored` - Total blocks stored
- `bitcoin_data_transactions_stored` - Total transactions stored
- `rate(bitcoin_data_stored_total[5m])` - Data throughput

#### Performance Metrics
- `bitcoin_rpc_requests_total` - RPC request counts
- `bitcoin_rpc_duration_seconds` - RPC call timing
- `bitcoin_database_write_duration_seconds` - Database write timing
- `bitcoin_errors_total` - Error counts by type

#### System Metrics
- `jvm_memory_used_bytes / jvm_memory_max_bytes` - Memory usage
- `rate(process_cpu_seconds_total[5m])` - CPU usage
- `(disk_total_bytes - disk_free_bytes) / disk_total_bytes` - Disk usage

### 5. Alerting Rules (Already Configured in Prometheus)

#### Critical Alerts
- **BitcoinCollectionDown**: No collection for 2+ minutes
- **DiskSpaceLow**: Less than 10% disk space free

#### Warning Alerts
- **BitcoinCollectionHighFailureRate**: >10% failure rate
- **BitcoinCollectionSlow**: 95th percentile >5s
- **BitcoinRPCHighErrorRate**: >0.1 errors/sec
- **BitcoinDatabaseHighErrorRate**: >0.05 errors/sec
- **BitcoinNoNewBlocks**: No blocks collected in 10 minutes
- **BitcoinNetworkConnectionsLow**: <5 connections

#### Info Alerts
- **BitcoinMempoolSizeHigh**: >10,000 transactions

### 6. Useful Queries for Dashboards

#### Collection Success Rate
```
rate(bitcoin_collection_success_total[5m]) / (rate(bitcoin_collection_success_total[5m]) + rate(bitcoin_collection_failures_total[5m])) * 100
```

#### Data Throughput
```
rate(bitcoin_data_stored_total[5m])
```

#### Last Collection Time
```
time() - bitcoin_collection_last_timestamp / 1000
```

#### Memory Usage Percentage
```
jvm_memory_used_bytes / jvm_memory_max_bytes * 100
```

### 7. Dashboard Panels to Create

1. **Bitcoin Network Status** (Stat panels)
   - Current block height
   - Network difficulty
   - Mempool size
   - Network connections

2. **Data Collection Performance** (Graph panels)
   - Collection success/failure rate
   - Collection duration percentiles
   - RPC request rates

3. **Data Storage Growth** (Graph panels)
   - Metrics stored over time
   - Blocks stored over time
   - Transactions stored over time

4. **System Health** (Graph panels)
   - Memory usage
   - CPU usage
   - Disk usage
   - Thread count

5. **Error Tracking** (Graph panels)
   - Error rates by type
   - Database performance
   - RPC performance

### 8. Current Status
- **Prometheus**: Running on port 9091
- **Grafana**: Running on port 3000
- **Bitcoin Service**: Running on port 8200
- **Alerts**: 12 rules configured and active
- **Metrics**: 385+ metrics collected and growing
