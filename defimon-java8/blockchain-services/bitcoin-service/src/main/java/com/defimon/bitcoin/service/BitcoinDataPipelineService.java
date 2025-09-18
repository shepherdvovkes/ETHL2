package com.defimon.bitcoin.service;

import com.defimon.bitcoin.client.QuickNodeBitcoinClient;
import com.defimon.bitcoin.model.*;
import com.defimon.bitcoin.repository.BitcoinMetricsRepository;
import com.defimon.bitcoin.repository.BitcoinBlockRepository;
import com.defimon.bitcoin.repository.BitcoinTransactionRepository;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.concurrent.CompletableFuture;

/**
 * Bitcoin Data Pipeline Service
 * Handles scheduled data collection and storage
 */
@Service
public class BitcoinDataPipelineService {

    private static final Logger logger = LoggerFactory.getLogger(BitcoinDataPipelineService.class);

    @Autowired
    private QuickNodeBitcoinClient bitcoinClient;

    @Autowired
    private BitcoinMetricsRepository metricsRepository;

    @Autowired
    private BitcoinBlockRepository blockRepository;

    @Autowired
    private BitcoinTransactionRepository transactionRepository;

    @Autowired
    private BitcoinMetricsService metricsService;

    private volatile boolean isCollecting = false;
    private volatile long lastCollectedBlock = 0;

    /**
     * Collect Bitcoin data every 30 seconds
     */
    @Scheduled(fixedRate = 30000)
    public void collectBitcoinData() {
        if (isCollecting) {
            logger.debug("Data collection already in progress, skipping...");
            return;
        }

        isCollecting = true;
        Timer.Sample sample = metricsService.startCollectionTimer();
        
        try {
            logger.info("Starting Bitcoin data collection cycle...");
            
            // Collect current metrics
            collectCurrentMetrics();
            
            // Collect latest blocks
            collectLatestBlocks();
            
            // Update metrics
            metricsService.recordSuccessfulCollection();
            metricsService.incrementCollectionCycle();
            
            // Update stored data counts
            updateStoredDataCounts();
            
            logger.info("Bitcoin data collection cycle completed successfully");
        } catch (Exception e) {
            logger.error("Error during Bitcoin data collection: {}", e.getMessage(), e);
            metricsService.recordFailedCollection();
        } finally {
            isCollecting = false;
            metricsService.recordCollectionDuration(sample);
        }
    }

    /**
     * Collect current Bitcoin network metrics
     */
    private void collectCurrentMetrics() {
        try {
            // Collect block count
            bitcoinClient.getBlockCount()
                .subscribe(blockCount -> {
                    Timer.Sample sample = metricsService.startDatabaseWriteTimer();
                    try {
                        metricsService.recordBlockCountRequest();
                        BitcoinMetrics metrics = BitcoinMetrics.builder()
                            .timestamp(System.currentTimeMillis())
                            .blockCount(blockCount)
                            .metricType("BLOCK_COUNT")
                            .build();
                        metricsRepository.save(metrics);
                        metricsService.recordDataStored();
                        metricsService.updateLastBlockHeight(blockCount);
                        logger.debug("Saved block count: {}", blockCount);
                    } catch (Exception e) {
                        metricsService.recordDatabaseError();
                        logger.error("Error saving block count: {}", e.getMessage());
                    } finally {
                        metricsService.recordDatabaseWriteDuration(sample);
                    }
                });

            // Collect difficulty
            bitcoinClient.getDifficulty()
                .subscribe(difficulty -> {
                    BitcoinMetrics metrics = BitcoinMetrics.builder()
                        .timestamp(System.currentTimeMillis())
                        .difficulty(difficulty.doubleValue())
                        .metricType("DIFFICULTY")
                        .build();
                    metricsRepository.save(metrics);
                    logger.debug("Saved difficulty: {}", difficulty);
                });

            // Collect mempool info
            bitcoinClient.getMempoolInfo()
                .subscribe(mempoolInfo -> {
                    BitcoinMetrics metrics = BitcoinMetrics.builder()
                        .timestamp(System.currentTimeMillis())
                        .mempoolSize(mempoolInfo.getSize())
                        .mempoolBytes(mempoolInfo.getBytes())
                        .mempoolUsage(mempoolInfo.getUsage())
                        .metricType("MEMPOOL_INFO")
                        .build();
                    metricsRepository.save(metrics);
                    logger.debug("Saved mempool info: size={}, bytes={}", 
                        mempoolInfo.getSize(), mempoolInfo.getBytes());
                });

            // Collect fee estimates
            bitcoinClient.estimateSmartFee(6)
                .subscribe(feeEstimate -> {
                    BitcoinMetrics metrics = BitcoinMetrics.builder()
                        .timestamp(System.currentTimeMillis())
                        .feeEstimate6Blocks(feeEstimate.doubleValue())
                        .metricType("FEE_ESTIMATE_6_BLOCKS")
                        .build();
                    metricsRepository.save(metrics);
                    logger.debug("Saved fee estimate (6 blocks): {}", feeEstimate);
                });

            // Collect network info
            bitcoinClient.getNetworkInfo()
                .subscribe(networkInfo -> {
                    BitcoinMetrics metrics = BitcoinMetrics.builder()
                        .timestamp(System.currentTimeMillis())
                        .networkConnections(networkInfo.getConnections())
                        .networkVersion(networkInfo.getVersion())
                        .metricType("NETWORK_INFO")
                        .build();
                    metricsRepository.save(metrics);
                    logger.debug("Saved network info: connections={}, version={}", 
                        networkInfo.getConnections(), networkInfo.getVersion());
                });

        } catch (Exception e) {
            logger.error("Error collecting current metrics: {}", e.getMessage(), e);
        }
    }

    /**
     * Collect latest Bitcoin blocks
     */
    private void collectLatestBlocks() {
        try {
            bitcoinClient.getBlockCount()
                .subscribe(currentBlockCount -> {
                    // Collect last 5 blocks
                    long startBlock = Math.max(1, currentBlockCount - 4);
                    for (long blockHeight = startBlock; blockHeight <= currentBlockCount; blockHeight++) {
                        if (blockHeight > lastCollectedBlock) {
                            collectBlock(blockHeight);
                            lastCollectedBlock = blockHeight;
                        }
                    }
                });
        } catch (Exception e) {
            logger.error("Error collecting latest blocks: {}", e.getMessage(), e);
        }
    }

    /**
     * Collect specific block data
     */
    private void collectBlock(long blockHeight) {
        try {
            bitcoinClient.getBlockHash(blockHeight)
                .flatMap(blockHash -> bitcoinClient.getBlock(blockHash))
                .subscribe(block -> {
                    if (block != null) {
                        // Check if block already exists
                        if (!blockRepository.existsByHash(block.getHash())) {
                            blockRepository.save(block);
                            logger.debug("Saved block: height={}, hash={}", 
                                block.getHeight(), block.getHash());
                        }
                    }
                });
        } catch (Exception e) {
            logger.error("Error collecting block {}: {}", blockHeight, e.getMessage(), e);
        }
    }

    /**
     * Get collection statistics
     */
    public CompletableFuture<CollectionStats> getCollectionStats() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                long totalMetrics = metricsRepository.count();
                long totalBlocks = blockRepository.count();
                long totalTransactions = transactionRepository.count();
                
                return CollectionStats.builder()
                    .totalMetrics(totalMetrics)
                    .totalBlocks(totalBlocks)
                    .totalTransactions(totalTransactions)
                    .lastCollectedBlock(lastCollectedBlock)
                    .isCollecting(isCollecting)
                    .build();
            } catch (Exception e) {
                logger.error("Error getting collection stats: {}", e.getMessage(), e);
                return CollectionStats.builder()
                    .totalMetrics(0L)
                    .totalBlocks(0L)
                    .totalTransactions(0L)
                    .lastCollectedBlock(0L)
                    .isCollecting(false)
                    .build();
            }
        });
    }

    /**
     * Start data collection manually
     */
    public void startCollection() {
        if (!isCollecting) {
            collectBitcoinData();
        }
    }

    /**
     * Get current collection status
     */
    public boolean isCollecting() {
        return isCollecting;
    }

    /**
     * Update stored data counts for metrics
     */
    private void updateStoredDataCounts() {
        try {
            long totalMetrics = metricsRepository.count();
            long totalBlocks = blockRepository.count();
            long totalTransactions = transactionRepository.count();
            
            metricsService.updateTotalMetricsStored(totalMetrics);
            metricsService.updateTotalBlocksStored(totalBlocks);
            metricsService.updateTotalTransactionsStored(totalTransactions);
            
            logger.debug("Updated stored data counts - Metrics: {}, Blocks: {}, Transactions: {}", 
                totalMetrics, totalBlocks, totalTransactions);
        } catch (Exception e) {
            logger.error("Error updating stored data counts: {}", e.getMessage());
            metricsService.recordDatabaseError();
        }
    }
}
