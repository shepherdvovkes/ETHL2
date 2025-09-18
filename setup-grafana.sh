#!/bin/bash

# Setup Grafana for Bitcoin Monitoring
echo "Setting up Grafana for Bitcoin monitoring..."

# Wait for Grafana to be ready
echo "Waiting for Grafana to be ready..."
sleep 10

# Create Prometheus data source
echo "Creating Prometheus data source..."
curl -X POST \
  http://admin:admin@localhost:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://192.168.0.247:9091",
    "access": "proxy",
    "isDefault": true,
    "jsonData": {
      "httpMethod": "POST"
    }
  }'

echo -e "\nData source created!"

# Import Bitcoin monitoring dashboard
echo "Importing Bitcoin monitoring dashboard..."
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @/home/vovkes/ETHL2/grafana-dashboard-comprehensive.json

echo -e "\nDashboard imported!"

# Create additional useful dashboards
echo "Creating system overview dashboard..."

# System Overview Dashboard
cat > /tmp/system-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "System Overview",
    "tags": ["system", "overview"],
    "style": "dark",
    "timezone": "browser",
    "refresh": "10s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "JVM Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "jvm_memory_used_bytes / jvm_memory_max_bytes * 100",
            "legendFormat": "Memory Usage %",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
            "legendFormat": "CPU Usage %",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(disk_total_bytes - disk_free_bytes) / disk_total_bytes * 100",
            "legendFormat": "Disk Usage %",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Thread Count",
        "type": "graph",
        "targets": [
          {
            "expr": "jvm_threads_states_threads",
            "legendFormat": "{{state}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Threads",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ]
  }
}
EOF

curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @/tmp/system-dashboard.json

echo -e "\nSystem dashboard created!"

# Create Bitcoin Network Health Dashboard
echo "Creating Bitcoin network health dashboard..."

cat > /tmp/network-health-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Bitcoin Network Health",
    "tags": ["bitcoin", "network", "health"],
    "style": "dark",
    "timezone": "browser",
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Block Height Growth",
        "type": "graph",
        "targets": [
          {
            "expr": "bitcoin_block_height",
            "legendFormat": "Block Height",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Block Height",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Mempool Activity",
        "type": "graph",
        "targets": [
          {
            "expr": "bitcoin_mempool_size",
            "legendFormat": "Mempool Size",
            "refId": "A"
          },
          {
            "expr": "bitcoin_mempool_bytes",
            "legendFormat": "Mempool Bytes",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "Count/Bytes",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Network Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "bitcoin_network_connections",
            "legendFormat": "Connections",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Connections",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Difficulty Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "bitcoin_difficulty",
            "legendFormat": "Difficulty",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Difficulty",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ]
  }
}
EOF

curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @/tmp/network-health-dashboard.json

echo -e "\nNetwork health dashboard created!"

echo -e "\nGrafana setup complete!"
echo "Access Grafana at: http://localhost:3000"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "Available dashboards:"
echo "1. Bitcoin Data Pipeline - Comprehensive Monitoring"
echo "2. System Overview"
echo "3. Bitcoin Network Health"
