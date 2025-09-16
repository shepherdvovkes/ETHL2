# Polkadot Metrics API Server

A comprehensive API server for collecting and serving metrics from the Polkadot network and its top 20 most active parachains over the past 3 months.

## Features

### Network Metrics Collection
- **Polkadot Main Network**: Block information, validator count, runtime version, consensus metrics
- **Staking Metrics**: Total staked amount, validator/nominator counts, era information, inflation rates
- **Governance Metrics**: Active proposals, referendums, council members, voting participation
- **Economic Metrics**: Treasury balance, token supply, inflation/deflation rates, transaction fees

### Parachain Monitoring
- **Top 20 Most Active Parachains**: Based on Q2 2024 activity data
- **Individual Parachain Metrics**: Block production, transaction volumes, user activity
- **Cross-Chain Metrics**: HRMP/XCMP messaging, bridge volumes, cross-chain liquidity

### Real-time Data Collection
- **Automated Collection**: Every 5 minutes via background tasks
- **Manual Triggers**: On-demand data collection via API endpoints
- **Historical Data**: 90-day retention with configurable collection intervals

## Architecture

### Components

1. **Polkadot Client** (`src/api/polkadot_client.py`)
   - RPC client for Polkadot network interaction
   - Support for multiple RPC endpoints with failover
   - Comprehensive metric collection methods

2. **Database Models** (`src/database/polkadot_models.py`)
   - Structured data models for all metric types
   - Relationships between networks, parachains, and metrics
   - Optimized for time-series data storage

3. **API Server** (`polkadot_metrics_server.py`)
   - FastAPI-based REST API
   - Real-time metric endpoints
   - Background data collection tasks

4. **PM2 Configuration** (`ecosystem.config.js`)
   - Process management and monitoring
   - Automatic restarts and logging
   - Production-ready deployment

## Supported Parachains

Based on Q2 2024 activity metrics:

| Parachain ID | Name | Symbol | Status |
|--------------|------|--------|--------|
| 2004 | Moonbeam | GLMR | Active |
| 2026 | Nodle | NODL | Active |
| 2035 | Phala Network | PHA | Active |
| 2091 | Frequency | FRQCY | Active |
| 2046 | NeuroWeb | NEURO | Active |
| 2034 | HydraDX | HDX | Active |
| 2030 | Bifrost | BNC | Active |
| 1000 | AssetHub | DOT | Active |
| 2006 | Astar | ASTR | Active |
| 2104 | Manta | MANTA | Active |
| 2000 | Acala | ACA | Active |
| 2012 | Parallel | PARA | Active |
| 2002 | Clover | CLV | Active |
| 2013 | Litentry | LIT | Active |
| 2011 | Equilibrium | EQ | Active |
| 2018 | SubDAO | GOV | Active |
| 2092 | Zeitgeist | ZTG | Active |
| 2121 | Efinity | EFI | Active |
| 2019 | Composable | LAYR | Active |
| 2085 | KILT Protocol | KILT | Active |

## API Endpoints

### Network Endpoints

- `GET /` - Server status and information
- `GET /health` - Health check with Polkadot client status
- `GET /network/info` - Polkadot network information
- `GET /network/metrics` - Comprehensive network metrics
- `GET /staking/metrics` - Staking-related metrics
- `GET /governance/metrics` - Governance metrics
- `GET /economic/metrics` - Economic metrics

### Parachain Endpoints

- `GET /parachains` - List all supported parachains
- `GET /parachains/{name}/metrics` - Metrics for specific parachain
- `GET /parachains/metrics` - Metrics for all parachains
- `GET /cross-chain/metrics` - Cross-chain messaging metrics

### Data Management

- `GET /historical/{days}` - Historical data for specified period
- `POST /collect` - Trigger manual data collection
- `GET /database/parachains` - Parachains from database
- `GET /database/network-metrics` - Network metrics from database

## Installation & Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- PM2 (Node.js process manager)
- Git

### 1. Database Setup

```bash
# Run the database setup script
python setup_polkadot_database.py
```

This will:
- Create all necessary database tables
- Initialize Polkadot network entry
- Add all 20 supported parachains

### 2. Configuration

Copy and customize the configuration:

```bash
cp polkadot_config.env .env
# Edit .env with your specific settings
```

Key configuration options:
- `POLKADOT_RPC_ENDPOINT`: Main RPC endpoint
- `DATA_COLLECTION_INTERVAL`: Collection frequency (default: 300s)
- `METRICS_RETENTION_DAYS`: Data retention period (default: 90 days)

### 3. Start the Server

#### Using PM2 (Recommended)

```bash
# Start all servers including Polkadot
pm2 start ecosystem.config.js

# Start only Polkadot server
pm2 start ecosystem.config.js --only polkadot-metrics

# Check status
pm2 status

# View logs
pm2 logs polkadot-metrics
```

#### Manual Start

```bash
# Direct Python execution
python polkadot_metrics_server.py

# Using uvicorn
uvicorn polkadot_metrics_server:app --host 0.0.0.0 --port 8007
```

### 4. Data Collection

#### Automatic Collection
- Runs every 5 minutes in background
- Configurable via `DATA_COLLECTION_INTERVAL`
- Monitored via PM2 logs

#### Manual Collection

```bash
# Run standalone collection script
python collect_polkadot_data.py

# Trigger via API
curl -X POST http://localhost:8007/collect
```

## Monitoring & Maintenance

### Health Monitoring

```bash
# Check server health
curl http://localhost:8007/health

# Check PM2 status
pm2 status

# View real-time logs
pm2 logs polkadot-metrics --lines 100
```

### Log Files

- `logs/polkadot-metrics.log` - Main application logs
- `logs/polkadot-metrics-out.log` - Standard output
- `logs/polkadot-metrics-error.log` - Error logs

### Database Maintenance

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename LIKE 'polkadot%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Clean old metrics (older than 90 days)
DELETE FROM polkadot_network_metrics 
WHERE timestamp < NOW() - INTERVAL '90 days';
```

## API Usage Examples

### Get Network Status

```bash
curl http://localhost:8007/network/info
```

### Get Parachain Metrics

```bash
# All parachains
curl http://localhost:8007/parachains/metrics

# Specific parachain
curl http://localhost:8007/parachains/moonbeam/metrics
```

### Get Historical Data

```bash
# Last 7 days
curl http://localhost:8007/historical/7

# Last 30 days
curl http://localhost:8007/historical/30
```

### Database Queries

```bash
# Get stored parachains
curl http://localhost:8007/database/parachains

# Get recent network metrics
curl http://localhost:8007/database/network-metrics?limit=50
```

## Performance & Scaling

### Resource Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 10GB+ for 90 days of data
- **Network**: Stable connection to Polkadot RPC

### Optimization Tips

1. **RPC Endpoints**: Use multiple endpoints for redundancy
2. **Database**: Regular VACUUM and ANALYZE operations
3. **Collection Frequency**: Adjust based on your needs (5-15 minutes)
4. **Retention**: Monitor disk usage and adjust retention period

### Scaling Options

- **Horizontal**: Multiple server instances with load balancer
- **Database**: Read replicas for query performance
- **Caching**: Redis for frequently accessed data
- **Storage**: Partition tables by time periods

## Troubleshooting

### Common Issues

1. **RPC Connection Errors**
   - Check network connectivity
   - Verify RPC endpoint URLs
   - Try backup endpoints

2. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check connection credentials
   - Ensure database exists

3. **PM2 Process Issues**
   - Check PM2 status: `pm2 status`
   - Restart process: `pm2 restart polkadot-metrics`
   - View logs: `pm2 logs polkadot-metrics`

4. **Data Collection Failures**
   - Check RPC endpoint availability
   - Verify database permissions
   - Review error logs

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Restart server
pm2 restart polkadot-metrics
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the ETHL2 ecosystem. See the main project license for details.

## Support

For issues and questions:
- Check the logs first
- Review this documentation
- Create an issue in the repository
- Contact the development team

---

**Note**: This server is designed for monitoring and analytics purposes. Always verify data accuracy and consider rate limits when using RPC endpoints.
