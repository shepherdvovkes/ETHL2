# 🧪 Avalanche Server Test Suite

Comprehensive test suite for the Avalanche Network Real-Time Metrics Server that validates real data collection without mocks.

## 📋 Overview

This test suite ensures that the Avalanche server can successfully:
- Connect to real external APIs (CoinGecko, Avalanche RPC, DeFiLlama)
- Collect real-time data from all 12 metric categories
- Process and validate data correctly
- Handle errors gracefully
- Maintain performance standards
- Provide accurate monitoring and alerting

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Install test requirements
pip install -r requirements.test.txt

# Or install specific packages
pip install pytest pytest-asyncio aiohttp fastapi httpx
```

### 2. Setup Test Environment

```bash
# Copy test configuration
cp test_config.env config.env

# Create test database (optional)
createdb defimon_test_db

# Create logs directory
mkdir -p logs
```

### 3. Run Tests

```bash
# Run all tests
python run_tests.py --type comprehensive

# Run specific test categories
python run_tests.py --type connectivity
python run_tests.py --type data_collection
python run_tests.py --type monitoring

# Run individual tests
python run_tests.py --test TestAvalancheDataCollection::test_network_performance_collection

# Run with coverage
python run_tests.py --type all --coverage
```

## 🧪 Test Categories

### 1. **Connectivity Tests** (`TestExternalAPIConnectivity`)
Tests real connectivity to external APIs:
- **CoinGecko API**: Market data, AVAX price, global crypto data
- **Avalanche RPC**: C-Chain, P-Chain, X-Chain connectivity
- **DeFiLlama API**: DeFi TVL and protocol data
- **Snowtrace API**: Blockchain explorer data

```python
# Example test
async def test_coingecko_api_connectivity():
    async with CoinGeckoClient() as cg:
        global_data = await cg.get_global_data()
        assert global_data is not None
        assert "data" in global_data
```

### 2. **Data Collection Tests** (`TestAvalancheDataCollection`)
Tests real data collection from all 12 metric categories:

#### Network Performance
- Block time, TPS, finality time
- Gas prices, network utilization
- Current block information

#### Economic Data
- AVAX price, market cap, volume
- Price changes, volatility
- Supply metrics

#### DeFi Metrics
- Total TVL, protocol counts
- Individual protocol data
- Yield farming metrics

#### Subnet Data
- Subnet counts, activity
- Validator information
- Custom VM usage

#### Security Status
- Validator counts, staking ratios
- Security scores, audit counts
- Risk assessments

#### User Behavior
- Whale activity, transaction patterns
- Retail vs institutional distribution
- Address concentration

#### Competitive Position
- Market rank, market share
- Competitor analysis
- Performance comparisons

#### Technical Health
- RPC performance, endpoint status
- Network uptime, health scores
- Infrastructure monitoring

#### Risk Indicators
- Price volatility, risk levels
- Centralization, technical risks
- Market and regulatory risks

#### Macro Environment
- Global market conditions
- Bitcoin/Ethereum dominance
- Economic indicators

#### Ecosystem Health
- Community growth metrics
- Media coverage, partnerships
- Developer experience

```python
# Example test
async def test_network_performance_collection(data_collector):
    metrics = await data_collector.collect_network_performance()
    
    assert isinstance(metrics, dict)
    assert "block_time" in metrics
    assert "transaction_throughput" in metrics
    assert "gas_price_avg" in metrics
    assert metrics["block_time"] > 0
    assert metrics["transaction_throughput"] >= 0
```

### 3. **Monitoring System Tests** (`TestMonitoringSystem`)
Tests the monitoring and alerting system:
- Rule initialization and configuration
- Alert manager functionality
- Rule evaluation with real data
- Alert creation and management

```python
# Example test
async def test_rule_evaluation_with_real_data(monitoring_system):
    test_data = {
        "network_performance": {
            "gas_price_avg": 150.0,  # Should trigger alert
            "transaction_throughput": 500  # Should trigger alert
        }
    }
    
    await monitoring_system.evaluate_rules(test_data)
    active_alerts = monitoring_system.get_active_alerts()
    assert len(active_alerts) > 0
```

### 4. **API Server Tests** (`TestAPIServer`)
Tests the FastAPI server endpoints:
- Health check endpoint
- Metrics endpoints
- Historical data endpoints
- Error handling

```python
# Example test
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
```

### 5. **Data Validation Tests** (`TestDataValidation`)
Tests data quality and consistency:
- Data freshness validation
- Data consistency across collections
- Error handling and recovery
- Data type validation

```python
# Example test
async def test_data_freshness():
    collector = RealTimeDataCollector()
    await collector.__aenter__()
    
    try:
        metrics = await collector.collect_network_performance()
        if metrics and "timestamp" in metrics:
            timestamp = datetime.fromisoformat(metrics["timestamp"])
            time_diff = (datetime.utcnow() - timestamp).total_seconds()
            assert time_diff < 60  # Data should be fresh
    finally:
        await collector.__aexit__(None, None, None)
```

### 6. **Performance Tests** (`TestPerformance`)
Tests performance characteristics:
- Collection speed validation
- Concurrent collection testing
- Resource usage monitoring
- Timeout handling

```python
# Example test
async def test_collection_performance():
    collector = RealTimeDataCollector()
    await collector.__aenter__()
    
    try:
        start_time = time.time()
        metrics = await collector.collect_network_performance()
        end_time = time.time()
        
        collection_time = end_time - start_time
        assert collection_time < 30  # Should complete within 30 seconds
    finally:
        await collector.__aexit__(None, None, None)
```

## 🔧 Test Configuration

### Test Environment Variables

```bash
# Test-specific settings
ENVIRONMENT=test
DEBUG=True
LOG_LEVEL=DEBUG

# Faster collection intervals for testing
NETWORK_PERFORMANCE_INTERVAL=5
ECONOMIC_DATA_INTERVAL=10

# Test database
DATABASE_URL=postgresql://user:pass@localhost:5432/test_db

# Disabled alerting for testing
DISABLE_ALERTING=True
```

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = .
python_files = test_*.py
addopts = -v --tb=short --timeout=300
asyncio_mode = auto
markers =
    asyncio: marks tests as async
    real_data: marks tests that use real external APIs
    network: marks tests that require network access
```

## 📊 Test Execution

### Running Specific Tests

```bash
# Run all tests
pytest test_avalanche_server.py -v

# Run specific test class
pytest test_avalanche_server.py::TestAvalancheDataCollection -v

# Run specific test method
pytest test_avalanche_server.py::TestAvalancheDataCollection::test_network_performance_collection -v

# Run tests with markers
pytest -m "real_data" -v
pytest -m "not slow" -v

# Run tests in parallel
pytest -n auto -v
```

### Test Runner Script

```bash
# Comprehensive test suite
python run_tests.py --type comprehensive

# Quick connectivity test
python run_tests.py --type connectivity

# Data collection validation
python run_tests.py --type data_collection

# Performance testing
python run_tests.py --type performance

# With coverage report
python run_tests.py --type all --coverage
```

## 🎯 Test Scenarios

### 1. **Happy Path Testing**
- All APIs are accessible
- Data collection succeeds
- Metrics are within expected ranges
- Alerts trigger correctly

### 2. **Error Handling Testing**
- API timeouts and failures
- Invalid data responses
- Network connectivity issues
- Database connection problems

### 3. **Performance Testing**
- Collection speed validation
- Concurrent request handling
- Memory usage monitoring
- Resource utilization

### 4. **Data Quality Testing**
- Data freshness validation
- Data consistency checks
- Type validation
- Range validation

### 5. **Integration Testing**
- End-to-end data flow
- API endpoint functionality
- Database operations
- Alert system integration

## 📈 Test Metrics

### Success Criteria

- **Connectivity**: 100% of external APIs accessible
- **Data Collection**: 90%+ success rate for all metric categories
- **Performance**: All collections complete within 30 seconds
- **Data Quality**: 95%+ of data passes validation
- **API Response**: All endpoints respond within 5 seconds

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Speed and resource testing
- **Security Tests**: Vulnerability and security testing

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```bash
   # Solution: Add delays between requests
   RATE_LIMIT_DELAY=1.0
   ```

2. **Network Timeouts**
   ```bash
   # Solution: Increase timeout values
   REQUEST_TIMEOUT=60
   CONNECTION_TIMEOUT=30
   ```

3. **Database Connection Issues**
   ```bash
   # Solution: Check database status
   sudo systemctl status postgresql
   psql -h localhost -U defimon -d defimon_test_db
   ```

4. **Missing API Keys**
   ```bash
   # Solution: Add API keys to config.env
   COINGECKO_API_KEY=your_key_here
   ```

### Debug Mode

```bash
# Run tests with debug logging
LOG_LEVEL=DEBUG python run_tests.py --type connectivity

# Run specific test with verbose output
pytest test_avalanche_server.py::TestAvalancheDataCollection::test_network_performance_collection -v -s

# Run with coverage and detailed output
pytest --cov=src --cov-report=html --cov-report=term-missing -v
```

## 📊 Test Reports

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Performance Report
```bash
# Generate performance report
pytest --durations=10 -v
```

### Test Results
```bash
# Generate JSON test report
pytest --json-report --json-report-file=test_results.json
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Avalanche Server Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.test.txt
    - name: Run tests
      run: |
        python run_tests.py --type comprehensive
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📝 Test Documentation

### Adding New Tests

1. **Create test method** in appropriate test class
2. **Use real data collection** (no mocks)
3. **Validate data types and ranges**
4. **Test error conditions**
5. **Document test purpose**

```python
async def test_new_metric_collection(self, data_collector):
    """Test collection of new metric type"""
    metrics = await data_collector.collect_new_metric()
    
    # Validate structure
    assert isinstance(metrics, dict)
    assert "timestamp" in metrics
    
    # Validate data types
    assert isinstance(metrics["value"], (int, float))
    
    # Validate ranges
    assert metrics["value"] >= 0
```

### Test Best Practices

1. **Use real APIs** - No mocks for data collection tests
2. **Validate data quality** - Check types, ranges, freshness
3. **Test error handling** - Ensure graceful failure
4. **Performance validation** - Check collection speed
5. **Comprehensive coverage** - Test all metric categories

## 🎯 Success Metrics

### Test Pass Rate
- **Target**: 95%+ test pass rate
- **Current**: Monitor via CI/CD pipeline
- **Action**: Fix failing tests immediately

### Data Collection Success
- **Target**: 90%+ successful data collection
- **Current**: Monitor via test results
- **Action**: Investigate and fix collection failures

### Performance Benchmarks
- **Target**: All collections < 30 seconds
- **Current**: Monitor via performance tests
- **Action**: Optimize slow collections

### API Response Times
- **Target**: All endpoints < 5 seconds
- **Current**: Monitor via API tests
- **Action**: Optimize slow endpoints

---

**🧪 Ready to test the Avalanche server!**

Run the comprehensive test suite to validate that your Avalanche network metrics server is working correctly with real data collection from external APIs.
