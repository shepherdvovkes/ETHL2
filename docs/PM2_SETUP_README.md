# Comprehensive Polkadot Metrics System - PM2 Setup Guide

This guide explains how to set up and manage the Comprehensive Polkadot Metrics System using PM2 for process management.

## 🚀 Quick Start

### 1. Complete Automated Setup
```bash
./setup_comprehensive_system.sh
```

This script will:
- Install all dependencies (Node.js, PM2, Python packages)
- Set up the database
- Configure PM2
- Start all services
- Run system tests

### 2. Manual Setup (Step by Step)

#### Install Dependencies
```bash
# Install Node.js and PM2
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g pm2

# Install Python dependencies
pip3 install -r requirements.txt
```

#### Setup PM2 Environment
```bash
./setup_pm2_environment.sh
```

#### Setup Database
```bash
python3 setup_comprehensive_polkadot_metrics.py
```

#### Start Services
```bash
./start_comprehensive_services.sh
```

## 📊 PM2 Services

The system runs the following services via PM2:

| Service Name | Script | Port | Description |
|--------------|--------|------|-------------|
| `polkadot-comprehensive-metrics` | `polkadot_comprehensive_metrics_server.py` | 8008 | Main comprehensive metrics API |
| `polkadot-data-collector` | `run_comprehensive_data_collector.py` | - | Continuous data collection |
| `polkadot-legacy-metrics` | `polkadot_metrics_server.py` | 8007 | Legacy Polkadot metrics API |
| `avalanche-metrics` | `avalanche_metrics_server.py` | 8006 | Avalanche metrics API |
| `l2-monitoring-dashboard` | `l2_monitoring_dashboard.py` | 8005 | L2 monitoring dashboard |

## 🛠️ Management Commands

### Service Management
```bash
# Start all services
./start_comprehensive_services.sh

# Stop all services
./stop_comprehensive_services.sh

# Restart all services
./restart_comprehensive_services.sh

# Monitor services
./pm2_monitor.sh
```

### PM2 Commands
```bash
# Show service status
pm2 status

# Show logs
pm2 logs

# Show logs for specific service
pm2 logs polkadot-comprehensive-metrics

# Restart specific service
pm2 restart polkadot-comprehensive-metrics

# Stop specific service
pm2 stop polkadot-comprehensive-metrics

# Start specific service
pm2 start polkadot-comprehensive-metrics

# Monitor resources
pm2 monit

# Save current PM2 configuration
pm2 save

# Reload PM2 configuration
pm2 reload ecosystem.comprehensive.config.js
```

### Advanced PM2 Commands
```bash
# Show detailed information about a service
pm2 show polkadot-comprehensive-metrics

# Reset restart counter
pm2 reset polkadot-comprehensive-metrics

# Flush logs
pm2 flush

# Delete all services
pm2 delete all

# Start services from ecosystem file
pm2 start ecosystem.comprehensive.config.js

# Stop services from ecosystem file
pm2 stop ecosystem.comprehensive.config.js
```

## 📁 Configuration Files

### PM2 Ecosystem Configuration
- **File**: `ecosystem.comprehensive.config.js`
- **Purpose**: Defines all services, their configurations, and environment variables

### Service Scripts
- **Start**: `start_comprehensive_services.sh`
- **Stop**: `stop_comprehensive_services.sh`
- **Restart**: `restart_comprehensive_services.sh`
- **Monitor**: `pm2_monitor.sh`

## 📊 Monitoring & Logs

### Log Files
All logs are stored in the `logs/` directory:

```
logs/
├── polkadot-comprehensive-metrics.log
├── polkadot-comprehensive-metrics-out.log
├── polkadot-comprehensive-metrics-error.log
├── polkadot-data-collector.log
├── polkadot-data-collector-out.log
├── polkadot-data-collector-error.log
├── avalanche-metrics.log
├── avalanche-metrics-out.log
├── avalanche-metrics-error.log
├── l2-monitoring-dashboard.log
├── l2-monitoring-dashboard-out.log
└── l2-monitoring-dashboard-error.log
```

### Monitoring Commands
```bash
# Real-time monitoring
pm2 monit

# Show service status
pm2 status

# Show logs in real-time
pm2 logs --follow

# Show logs for specific service
pm2 logs polkadot-comprehensive-metrics --follow

# Show system information
pm2 info polkadot-comprehensive-metrics
```

## 🔧 Environment Variables

### API Configuration
```bash
API_HOST=0.0.0.0
API_PORT=8008
NODE_ENV=production
```

### Database Configuration
```bash
DATABASE_URL=postgresql://user:password@localhost/polkadot_metrics
```

### RPC Endpoints
```bash
POLKADOT_RPC_ENDPOINT=https://rpc.polkadot.io
POLKADOT_WS_ENDPOINT=wss://rpc.polkadot.io
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
pm2 logs polkadot-comprehensive-metrics

# Check if port is in use
netstat -tlnp | grep :8008

# Restart service
pm2 restart polkadot-comprehensive-metrics
```

#### 2. Data Collection Issues
```bash
# Check data collector logs
pm2 logs polkadot-data-collector

# Restart data collector
pm2 restart polkadot-data-collector

# Test data collection manually
python3 collect_comprehensive_polkadot_data.py
```

#### 3. Database Connection Issues
```bash
# Check database connection
python3 -c "from src.database.database import engine; print(engine.url)"

# Test database setup
python3 setup_comprehensive_polkadot_metrics.py
```

#### 4. PM2 Issues
```bash
# Reset PM2
pm2 kill
pm2 start ecosystem.comprehensive.config.js

# Check PM2 status
pm2 status

# Show PM2 version
pm2 --version
```

### Performance Issues

#### High Memory Usage
```bash
# Check memory usage
pm2 monit

# Restart services with memory limit
pm2 restart all --max-memory-restart 1G
```

#### High CPU Usage
```bash
# Check CPU usage
pm2 monit

# Reduce collection frequency
# Edit ecosystem.comprehensive.config.js
```

## 🔄 Auto-Startup

### Enable PM2 Startup
```bash
# Generate startup script
pm2 startup

# Run the command shown (usually requires sudo)
sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u $USER --hp $HOME

# Save current PM2 configuration
pm2 save
```

### Disable PM2 Startup
```bash
pm2 unstartup
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale specific service
pm2 scale polkadot-comprehensive-metrics 2

# Scale all services
pm2 scale ecosystem.comprehensive.config.js 2
```

### Vertical Scaling
Edit `ecosystem.comprehensive.config.js` to adjust:
- `max_memory_restart`
- `instances`
- `cron_restart`

## 🔒 Security

### Firewall Configuration
```bash
# Allow specific ports
sudo ufw allow 8005
sudo ufw allow 8006
sudo ufw allow 8007
sudo ufw allow 8008

# Check firewall status
sudo ufw status
```

### Process Isolation
```bash
# Run as specific user
pm2 start ecosystem.comprehensive.config.js --uid nodejs

# Set process limits
pm2 start ecosystem.comprehensive.config.js --max-memory-restart 1G
```

## 📋 Maintenance

### Regular Maintenance Tasks

#### Daily
```bash
# Check service status
pm2 status

# Check logs for errors
pm2 logs --err
```

#### Weekly
```bash
# Restart services
pm2 restart all

# Clean old logs
pm2 flush

# Update dependencies
pip3 install -r requirements.txt --upgrade
```

#### Monthly
```bash
# Backup database
pg_dump polkadot_metrics > backup_$(date +%Y%m%d).sql

# Clean old log files
find logs/ -name "*.log" -mtime +30 -delete
```

## 🆘 Support

### Getting Help
1. Check logs: `pm2 logs`
2. Check status: `pm2 status`
3. Monitor resources: `pm2 monit`
4. Test manually: `python3 test_comprehensive_metrics.py`

### Useful Resources
- [PM2 Documentation](https://pm2.keymetrics.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Polkadot Documentation](https://polkadot.network/docs/)

---

**Built with ❤️ for the Polkadot ecosystem using PM2**
