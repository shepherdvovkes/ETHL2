# 🚀 Ethereum Archive Node Setup

This directory contains a complete Docker-based Ethereum archive node setup with Geth and Lighthouse.

## 📁 Directory Structure

```
ethereum/fullnode/
├── docker-compose.yml          # Main Docker Compose configuration
├── prometheus.yml              # Prometheus monitoring configuration
├── start-ethereum-node.sh      # Startup script
├── monitor-node.sh             # Monitoring script
├── README.md                   # This file
├── logs/                       # Application logs
├── prometheus-data/            # Prometheus metrics storage
└── grafana-data/               # Grafana dashboards and data
```

## 🗄️ Data Storage

The Ethereum chain data is stored on the 20TB SATA drive at `/mnt/ethereum-data/ethereum/`:

- **Geth Data**: `/mnt/ethereum-data/ethereum/geth-data/` (46GB)
- **Lighthouse Data**: `/mnt/ethereum-data/ethereum/lighthouse-hot/` (48GB)

## 🚀 Quick Start

### 1. Start the Archive Node

```bash
cd /home/vovkes/ETHL2/ethereum/fullnode
./start-ethereum-node.sh
```

### 2. Monitor the Node

```bash
./monitor-node.sh
```

### 3. View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f geth
docker-compose logs -f lighthouse
```

## 🔧 Configuration

### Geth Configuration

- **Mode**: Archive node (`--gcmode=archive`)
- **Sync Mode**: Full sync (`--syncmode=full`)
- **Database Engine**: Pebble (faster than LevelDB)
- **Cache**: 4GB total cache allocation
- **Max Peers**: 100
- **RPC Endpoints**: HTTP (8545) and WebSocket (8546)

### Lighthouse Configuration

- **Network**: Mainnet
- **Checkpoint Sync**: Enabled (faster initial sync)
- **Execution Endpoint**: Connected to Geth
- **Max Peers**: 50
- **API Endpoint**: HTTP (5052)

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Geth HTTP RPC | http://localhost:8545 | JSON-RPC API |
| Geth WebSocket | ws://localhost:8546 | WebSocket API |
| Lighthouse API | http://localhost:5052 | Beacon chain API |
| Prometheus | http://localhost:9090 | Metrics (if enabled) |
| Grafana | http://localhost:3000 | Dashboards (if enabled) |

## 📊 Monitoring

### Enable Monitoring Stack

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Grafana
# URL: http://localhost:3000
# Username: admin
# Password: ethereum123
```

### Key Metrics to Monitor

- **Sync Status**: `eth.syncing` (should be `false` when synced)
- **Block Height**: `eth.blockNumber`
- **Peer Count**: `net.peerCount`
- **Disk Usage**: Monitor `/mnt/ethereum-data` usage
- **Memory Usage**: Geth and Lighthouse memory consumption

## 🛠️ Management Commands

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d geth
docker-compose up -d lighthouse
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop geth
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart geth
```

### Check Sync Status

```bash
# Check Geth sync status
docker-compose exec geth geth attach --exec "eth.syncing"

# Check current block
docker-compose exec geth geth attach --exec "eth.blockNumber"

# Check peer count
docker-compose exec geth geth attach --exec "net.peerCount"
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Disk Space**
   ```bash
   # Check disk usage
   df -h /mnt/ethereum-data
   
   # Clean up old logs
   docker system prune -f
   ```

2. **Sync Stuck**
   ```bash
   # Restart Geth
   docker-compose restart geth
   
   # Check for corrupted data
   docker-compose exec geth geth attach --exec "eth.syncing"
   ```

3. **High Memory Usage**
   ```bash
   # Reduce cache size in docker-compose.yml
   --cache=2048  # Reduce from 4096
   ```

4. **Connection Issues**
   ```bash
   # Check firewall
   sudo ufw status
   
   # Check port availability
   netstat -tuln | grep -E "(8545|8546|9000|5052)"
   ```

### Log Analysis

```bash
# View recent errors
docker-compose logs geth | grep -i error
docker-compose logs lighthouse | grep -i error

# Monitor sync progress
docker-compose logs -f geth | grep -i "imported\|synced"
```

## 📈 Performance Optimization

### System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ recommended (8GB for Geth, 4GB for Lighthouse)
- **Storage**: 2TB+ free space (archive node grows ~1GB/day)
- **Network**: Stable internet connection

### Optimization Tips

1. **Use SSD for logs** (if available)
2. **Increase file descriptors**: `ulimit -n 65536`
3. **Optimize Docker settings**:
   ```bash
   # Increase Docker memory limit
   # Edit /etc/docker/daemon.json
   {
     "default-ulimits": {
       "nofile": {
         "Hard": 65536,
         "Name": "nofile",
         "Soft": 65536
       }
     }
   }
   ```

## 🔐 Security Considerations

1. **Firewall**: Only expose necessary ports
2. **RPC Access**: Restrict RPC access to trusted networks
3. **JWT Secret**: Keep JWT secret secure
4. **Updates**: Regularly update Docker images

## 📚 Additional Resources

- [Geth Documentation](https://geth.ethereum.org/docs)
- [Lighthouse Documentation](https://lighthouse-book.sigmaprime.io/)
- [Ethereum Archive Node Guide](https://ethereum.org/en/developers/docs/nodes-and-clients/run-a-node/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🆘 Support

For issues or questions:

1. Check the logs: `docker-compose logs -f`
2. Run the monitor script: `./monitor-node.sh`
3. Check system resources: `htop`, `df -h`
4. Verify network connectivity: `ping 8.8.8.8`

---

**Note**: This is an archive node that stores the complete Ethereum blockchain history. It requires significant storage space and will continue to grow over time.
