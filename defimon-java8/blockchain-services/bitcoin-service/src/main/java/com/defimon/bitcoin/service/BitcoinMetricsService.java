package com.defimon.bitcoin.service;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Bitcoin Metrics Service
 * Provides comprehensive metrics for monitoring the Bitcoin data pipeline
 */
@Service
public class BitcoinMetricsService {

    private static final Logger logger = LoggerFactory.getLogger(BitcoinMetricsService.class);

    @Autowired
    private MeterRegistry meterRegistry;

    // Counters for tracking events
    private Counter blockCountRequests;
    private Counter difficultyRequests;
    private Counter mempoolRequests;
    private Counter networkInfoRequests;
    private Counter feeEstimateRequests;
    private Counter blockDataRequests;
    private Counter transactionRequests;
    
    // Counters for successful operations
    private Counter successfulCollections;
    private Counter failedCollections;
    private Counter dataStored;
    private Counter dataRetrieved;
    
    // Counters for errors
    private Counter rpcErrors;
    private Counter databaseErrors;
    private Counter networkErrors;
    
    // Timers for measuring performance
    private Timer collectionTimer;
    private Timer rpcCallTimer;
    private Timer databaseWriteTimer;
    private Timer databaseReadTimer;
    
    // Gauges for current state
    private AtomicLong lastBlockHeight;
    private AtomicLong currentDifficulty;
    private AtomicLong mempoolSize;
    private AtomicLong mempoolBytes;
    private AtomicLong networkConnections;
    private AtomicLong totalMetricsStored;
    private AtomicLong totalBlocksStored;
    private AtomicLong totalTransactionsStored;
    
    // Collection status
    private AtomicLong collectionCycleCount;
    private AtomicLong lastCollectionTimestamp;

    @PostConstruct
    public void initializeMetrics() {
        logger.info("Initializing Bitcoin metrics...");
        
        // Initialize counters
        blockCountRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of block count RPC requests")
                .tag("method", "getblockcount")
                .register(meterRegistry);
                
        difficultyRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of difficulty RPC requests")
                .tag("method", "getdifficulty")
                .register(meterRegistry);
                
        mempoolRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of mempool RPC requests")
                .tag("method", "getmempoolinfo")
                .register(meterRegistry);
                
        networkInfoRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of network info RPC requests")
                .tag("method", "getnetworkinfo")
                .register(meterRegistry);
                
        feeEstimateRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of fee estimate RPC requests")
                .tag("method", "estimatesmartfee")
                .register(meterRegistry);
                
        blockDataRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of block data RPC requests")
                .tag("method", "getblock")
                .register(meterRegistry);
                
        transactionRequests = Counter.builder("bitcoin.rpc.requests")
                .description("Number of transaction RPC requests")
                .tag("method", "getrawtransaction")
                .register(meterRegistry);

        // Success/failure counters
        successfulCollections = Counter.builder("bitcoin.collection.success")
                .description("Number of successful data collections")
                .register(meterRegistry);
                
        failedCollections = Counter.builder("bitcoin.collection.failures")
                .description("Number of failed data collections")
                .register(meterRegistry);
                
        dataStored = Counter.builder("bitcoin.data.stored")
                .description("Number of data records stored")
                .register(meterRegistry);
                
        dataRetrieved = Counter.builder("bitcoin.data.retrieved")
                .description("Number of data records retrieved")
                .register(meterRegistry);

        // Error counters
        rpcErrors = Counter.builder("bitcoin.errors")
                .description("Number of RPC errors")
                .tag("type", "rpc")
                .register(meterRegistry);
                
        databaseErrors = Counter.builder("bitcoin.errors")
                .description("Number of database errors")
                .tag("type", "database")
                .register(meterRegistry);
                
        networkErrors = Counter.builder("bitcoin.errors")
                .description("Number of network errors")
                .tag("type", "network")
                .register(meterRegistry);

        // Initialize timers
        collectionTimer = Timer.builder("bitcoin.collection.duration")
                .description("Time taken for data collection cycles")
                .register(meterRegistry);
                
        rpcCallTimer = Timer.builder("bitcoin.rpc.duration")
                .description("Time taken for RPC calls")
                .register(meterRegistry);
                
        databaseWriteTimer = Timer.builder("bitcoin.database.write.duration")
                .description("Time taken for database write operations")
                .register(meterRegistry);
                
        databaseReadTimer = Timer.builder("bitcoin.database.read.duration")
                .description("Time taken for database read operations")
                .register(meterRegistry);

        // Initialize atomic longs for gauges
        lastBlockHeight = new AtomicLong(0);
        currentDifficulty = new AtomicLong(0);
        mempoolSize = new AtomicLong(0);
        mempoolBytes = new AtomicLong(0);
        networkConnections = new AtomicLong(0);
        totalMetricsStored = new AtomicLong(0);
        totalBlocksStored = new AtomicLong(0);
        totalTransactionsStored = new AtomicLong(0);
        collectionCycleCount = new AtomicLong(0);
        lastCollectionTimestamp = new AtomicLong(0);

        // Register gauges using simpler approach
        meterRegistry.gauge("bitcoin.block.height", lastBlockHeight);
        meterRegistry.gauge("bitcoin.difficulty", currentDifficulty);
        meterRegistry.gauge("bitcoin.mempool.size", mempoolSize);
        meterRegistry.gauge("bitcoin.mempool.bytes", mempoolBytes);
        meterRegistry.gauge("bitcoin.network.connections", networkConnections);
        meterRegistry.gauge("bitcoin.data.metrics.stored", totalMetricsStored);
        meterRegistry.gauge("bitcoin.data.blocks.stored", totalBlocksStored);
        meterRegistry.gauge("bitcoin.data.transactions.stored", totalTransactionsStored);
        meterRegistry.gauge("bitcoin.collection.cycles", collectionCycleCount);
        meterRegistry.gauge("bitcoin.collection.last.timestamp", lastCollectionTimestamp);

        logger.info("Bitcoin metrics initialized successfully");
    }

    // RPC request tracking
    public void recordBlockCountRequest() {
        blockCountRequests.increment();
    }

    public void recordDifficultyRequest() {
        difficultyRequests.increment();
    }

    public void recordMempoolRequest() {
        mempoolRequests.increment();
    }

    public void recordNetworkInfoRequest() {
        networkInfoRequests.increment();
    }

    public void recordFeeEstimateRequest() {
        feeEstimateRequests.increment();
    }

    public void recordBlockDataRequest() {
        blockDataRequests.increment();
    }

    public void recordTransactionRequest() {
        transactionRequests.increment();
    }

    // Success/failure tracking
    public void recordSuccessfulCollection() {
        successfulCollections.increment();
    }

    public void recordFailedCollection() {
        failedCollections.increment();
    }

    public void recordDataStored() {
        dataStored.increment();
    }

    public void recordDataRetrieved() {
        dataRetrieved.increment();
    }

    // Error tracking
    public void recordRpcError() {
        rpcErrors.increment();
    }

    public void recordDatabaseError() {
        databaseErrors.increment();
    }

    public void recordNetworkError() {
        networkErrors.increment();
    }

    // Timer tracking
    public Timer.Sample startCollectionTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordCollectionDuration(Timer.Sample sample) {
        sample.stop(collectionTimer);
    }

    public Timer.Sample startRpcCallTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordRpcCallDuration(Timer.Sample sample) {
        sample.stop(rpcCallTimer);
    }

    public Timer.Sample startDatabaseWriteTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordDatabaseWriteDuration(Timer.Sample sample) {
        sample.stop(databaseWriteTimer);
    }

    public Timer.Sample startDatabaseReadTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordDatabaseReadDuration(Timer.Sample sample) {
        sample.stop(databaseReadTimer);
    }

    // Gauge updates
    public void updateLastBlockHeight(long height) {
        lastBlockHeight.set(height);
    }

    public void updateCurrentDifficulty(long difficulty) {
        currentDifficulty.set(difficulty);
    }

    public void updateMempoolSize(long size) {
        mempoolSize.set(size);
    }

    public void updateMempoolBytes(long bytes) {
        mempoolBytes.set(bytes);
    }

    public void updateNetworkConnections(long connections) {
        networkConnections.set(connections);
    }

    public void updateTotalMetricsStored(long count) {
        totalMetricsStored.set(count);
    }

    public void updateTotalBlocksStored(long count) {
        totalBlocksStored.set(count);
    }

    public void updateTotalTransactionsStored(long count) {
        totalTransactionsStored.set(count);
    }

    public void incrementCollectionCycle() {
        collectionCycleCount.incrementAndGet();
        lastCollectionTimestamp.set(System.currentTimeMillis());
    }

    // Utility methods
    public long getCollectionCycleCount() {
        return collectionCycleCount.get();
    }

    public long getLastCollectionTimestamp() {
        return lastCollectionTimestamp.get();
    }
}
