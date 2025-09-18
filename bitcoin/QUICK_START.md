# Bitcoin Ultrafast Archive Node - Quick Start Guide

Get your Bitcoin archive node running in minutes!

## 🚀 5-Minute Setup

### Step 1: Prerequisites Check
```bash
# Check if Docker is installed
docker --version
docker-compose --version

# If not installed:
sudo apt update
sudo apt install docker.io docker-compose jq bc
sudo systemctl start docker
```

### Step 2: Start the Node
```bash
cd /home/vovkes/ETHL2/bitcoin
./scripts/start_bitcoin_node.sh
```

### Step 3: Monitor Progress
```bash
# In another terminal:
./scripts/monitor_sync_progress.sh
```

## ✅ That's It!

Your Bitcoin archive node is now running and syncing. The initial sync will take several days, but you can use the node for queries once it's partially synced.

## 🔧 Quick Commands

```bash
# Check if running
docker ps | grep bitcoin

# View logs
docker-compose logs -f bitcoin-core

# Stop node
./scripts/stop_bitcoin_node.sh

# Health check
./scripts/check_node_health.sh
```

## 📊 RPC Access

- **URL**: `http://localhost:8332`
- **User**: `bitcoin`
- **Password**: `ultrafast_archive_node_2024`

```bash
# Test RPC
docker exec bitcoin-archive-node bitcoin-cli getblockchaininfo
```

## 🆘 Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Run `./scripts/check_node_health.sh` for diagnostics
3. Check logs: `docker-compose logs bitcoin-core`

## ⏱️ Expected Timeline

- **Setup**: 5 minutes
- **Initial Sync**: 3-7 days
- **Full Archive**: 400GB+ disk space

---

**Happy Bitcoin node running! 🎉**

