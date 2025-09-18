package com.defimon.collector.service;

import com.defimon.bitcoin.client.QuickNodeBitcoinClient;
import com.defimon.bitcoin.model.*;
import com.defimon.common.model.Asset;
import com.defimon.common.util.CollectionResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Bitcoin-specific data collector service using QuickNode API
 */
@Service
public class BitcoinDataCollectorService {
    
    private static final Logger logger = LoggerFactory.getLogger(BitcoinDataCollectorService.class);
    
    @Autowired
    private QuickNodeBitcoinClient bitcoinClient;
    
    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;
    
    @Autowired
    private MeterRegistry meterRegistry;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Value("${bitcoin.collector.batch-size:100}")
    private int batchSize;
    
    @Value("${bitcoin.collector.start-height:0}")
    private long startHeight;
    
    @Value("${bitcoin.collector.end-height:#{null}}")
    private Long endHeight;
    
    @Value("${bitcoin.collector.resume-from-last:true}")
    private boolean resumeFromLast;
    
    @Value("${bitcoin.collector.progress-update-interval:30}")
    private int progressUpdateInterval;
    
    @Value("${bitcoin.collector.max-retries:3}")
    private int maxRetries;
    
    @Value("${bitcoin.collector.retry-delay:5000}")
    private long retryDelay;
    
    @Value("${bitcoin.collector.continue-on-error:true}")
    private boolean continueOnError;
    
    private ExecutorService executorService;
    
    // Metrics
    private Counter bitcoinBlocksCollected;
    private Counter bitcoinTransactionsCollected;
    private Counter bitcoinErrors;
    private Timer bitcoinCollectionTimer;
    private Counter bitcoinNetworkInfoCollected;
    
    @PostConstruct
    public void init() {
        this.executorService = Executors.newFixedThreadPool(10);
        
        // Initialize metrics
        this.bitcoinBlocksCollected = Counter.builder("bitcoin.collection.blocks")
            .description("Bitcoin blocks collected")
            .register(meterRegistry);
        this.bitcoinTransactionsCollected = Counter.builder("bitcoin.collection.transactions")
            .description("Bitcoin transactions collected")
            .register(meterRegistry);
        this.bitcoinErrors = Counter.builder("bitcoin.collection.errors")
            .description("Bitcoin collection errors")
            .register(meterRegistry);
        this.bitcoinCollectionTimer = Timer.builder("bitcoin.collection.duration")
            .description("Bitcoin data collection processing time")
            .register(meterRegistry);
        this.bitcoinNetworkInfoCollected = Counter.builder("bitcoin.collection.network.info")
            .description("Bitcoin network info collected")
            .register(meterRegistry);
        
        logger.info("Bitcoin Data Collector Service initialized");
    }
    
    /**
     * Scheduled Bitcoin data collection
     */
    @Scheduled(fixedRateString = "${bitcoin.collector.interval:60000}")
    public void scheduleBitcoinDataCollection() {
        logger.info("Starting scheduled Bitcoin data collection...");
        
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            collectBitcoinNetworkData()
                .whenComplete((result, throwable) -> {
                    sample.stop(bitcoinCollectionTimer);
                    
                    if (throwable != null) {
                        logger.error("Bitcoin data collection failed", throwable);
                        bitcoinErrors.increment();
                    } else {
                        logger.info("Bitcoin data collection completed successfully");
                    }
                });
                
        } catch (Exception e) {
            sample.stop(bitcoinCollectionTimer);
            logger.error("Error starting Bitcoin data collection", e);
            bitcoinErrors.increment();
        }
    }
    
    /**
     * Collect Bitcoin network data
     */
    @Async
    @CircuitBreaker(name = "bitcoin-quicknode")
    @RateLimiter(name = "bitcoin-api")
    public CompletableFuture<Map<String, Object>> collectBitcoinNetworkData() {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> networkData = new HashMap<>();
            Instant start = Instant.now();
            
            try {
                logger.info("Collecting Bitcoin network data...");
                
                // Collect current block count
                Long blockCount = bitcoinClient.getBlockCount().block();
                if (blockCount != null) {
                    networkData.put("currentBlockCount", blockCount);
                    logger.info("Current Bitcoin block count: {}", blockCount);
                }
                
                // Collect network info
                NetworkInfo networkInfo = bitcoinClient.getNetworkInfo().block();
                if (networkInfo != null) {
                    networkData.put("networkInfo", networkInfo);
                    bitcoinNetworkInfoCollected.increment();
                    logger.info("Bitcoin network info collected - Version: {}, Connections: {}", 
                               networkInfo.getVersion(), networkInfo.getConnections());
                }
                
                // Collect mempool info
                MempoolInfo mempoolInfo = bitcoinClient.getMempoolInfo().block();
                if (mempoolInfo != null) {
                    networkData.put("mempoolInfo", mempoolInfo);
                    logger.info("Bitcoin mempool info - Size: {}, Bytes: {}", 
                               mempoolInfo.getSize(), mempoolInfo.getBytes());
                }
                
                // Collect difficulty
                BigDecimal difficulty = bitcoinClient.getDifficulty().block();
                if (difficulty != null) {
                    networkData.put("difficulty", difficulty);
                    logger.info("Bitcoin difficulty: {}", difficulty);
                }
                
                // Collect fee estimates
                BigDecimal feeEstimate1 = bitcoinClient.estimateSmartFee(1).block();
                BigDecimal feeEstimate6 = bitcoinClient.estimateSmartFee(6).block();
                if (feeEstimate1 != null) {
                    networkData.put("feeEstimate1Block", feeEstimate1);
                }
                if (feeEstimate6 != null) {
                    networkData.put("feeEstimate6Blocks", feeEstimate6);
                }
                
                networkData.put("collectionTime", start);
                networkData.put("duration", Duration.between(start, Instant.now()).toMillis());
                networkData.put("successful", true);
                
                // Publish to Kafka
                publishBitcoinNetworkData(networkData);
                
                return networkData;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin network data", e);
                bitcoinErrors.increment();
                
                networkData.put("collectionTime", start);
                networkData.put("duration", Duration.between(start, Instant.now()).toMillis());
                networkData.put("successful", false);
                networkData.put("error", e.getMessage());
                
                return networkData;
            }
        }, executorService);
    }
    
    /**
     * Collect Bitcoin block data
     */
    @Async
    @CircuitBreaker(name = "bitcoin-quicknode")
    @RateLimiter(name = "bitcoin-api")
    public CompletableFuture<BitcoinBlock> collectBitcoinBlock(long blockHeight) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                logger.info("Collecting Bitcoin block at height: {}", blockHeight);
                
                // Get block hash first
                String blockHash = bitcoinClient.getBlockHash(blockHeight).block();
                if (blockHash == null || blockHash.isEmpty()) {
                    logger.warn("Could not get block hash for height: {}", blockHeight);
                    return null;
                }
                
                // Get full block data
                BitcoinBlock block = bitcoinClient.getBlock(blockHash).block();
                if (block != null) {
                    bitcoinBlocksCollected.increment();
                    logger.info("Successfully collected block {} at height {}", blockHash, blockHeight);
                    
                    // Publish block data to Kafka
                    publishBitcoinBlockData(block);
                }
                
                return block;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin block at height {}: {}", blockHeight, e.getMessage());
                bitcoinErrors.increment();
                return null;
            }
        }, executorService);
    }
    
    /**
     * Collect Bitcoin transaction data
     */
    @Async
    @CircuitBreaker(name = "bitcoin-transaction")
    @RateLimiter(name = "bitcoin-api")
    public CompletableFuture<BitcoinTransaction> collectBitcoinTransaction(String txHash) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                logger.info("Collecting Bitcoin transaction: {}", txHash);
                
                // Get raw transaction
                String rawTx = bitcoinClient.getRawTransaction(txHash).block();
                if (rawTx == null || rawTx.isEmpty()) {
                    logger.warn("Could not get raw transaction for hash: {}", txHash);
                    return null;
                }
                
                // Decode transaction
                BitcoinTransaction transaction = bitcoinClient.decodeRawTransaction(rawTx).block();
                if (transaction != null) {
                    bitcoinTransactionsCollected.increment();
                    logger.info("Successfully collected transaction: {}", txHash);
                    
                    // Publish transaction data to Kafka
                    publishBitcoinTransactionData(transaction);
                }
                
                return transaction;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin transaction {}: {}", txHash, e.getMessage());
                bitcoinErrors.increment();
                return null;
            }
        }, executorService);
    }
    
    /**
     * Collect Bitcoin address data
     */
    @Async
    @CircuitBreaker(name = "bitcoin-quicknode")
    @RateLimiter(name = "bitcoin-api")
    public CompletableFuture<Map<String, Object>> collectBitcoinAddressData(String address) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> addressData = new HashMap<>();
            Instant start = Instant.now();
            
            try {
                logger.info("Collecting Bitcoin address data: {}", address);
                
                // Get address balance
                BigDecimal balance = bitcoinClient.getAddressBalance(address).block();
                if (balance != null) {
                    addressData.put("balance", balance);
                    logger.info("Address {} balance: {}", address, balance);
                }
                
                addressData.put("address", address);
                addressData.put("collectionTime", start);
                addressData.put("duration", Duration.between(start, Instant.now()).toMillis());
                addressData.put("successful", true);
                
                // Publish address data to Kafka
                publishBitcoinAddressData(addressData);
                
                return addressData;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin address data for {}: {}", address, e.getMessage());
                bitcoinErrors.increment();
                
                addressData.put("address", address);
                addressData.put("collectionTime", start);
                addressData.put("duration", Duration.between(start, Instant.now()).toMillis());
                addressData.put("successful", false);
                addressData.put("error", e.getMessage());
                
                return addressData;
            }
        }, executorService);
    }
    
    /**
     * Batch collect Bitcoin blocks
     */
    @Async
    public CompletableFuture<Map<String, Object>> collectBitcoinBlocksBatch(long startHeight, long endHeight) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> batchResult = new HashMap<>();
            Instant start = Instant.now();
            int successCount = 0;
            int errorCount = 0;
            
            try {
                logger.info("Starting batch collection of Bitcoin blocks from {} to {}", startHeight, endHeight);
                
                for (long height = startHeight; height <= endHeight; height++) {
                    try {
                        BitcoinBlock block = collectBitcoinBlock(height).get();
                        if (block != null) {
                            successCount++;
                        } else {
                            errorCount++;
                        }
                        
                        // Rate limiting
                        Thread.sleep(100); // 100ms delay between requests
                        
                    } catch (Exception e) {
                        logger.error("Error collecting block at height {}: {}", height, e.getMessage());
                        errorCount++;
                        
                        if (!continueOnError) {
                            break;
                        }
                    }
                }
                
                batchResult.put("startHeight", startHeight);
                batchResult.put("endHeight", endHeight);
                batchResult.put("successCount", successCount);
                batchResult.put("errorCount", errorCount);
                batchResult.put("totalBlocks", endHeight - startHeight + 1);
                batchResult.put("collectionTime", start);
                batchResult.put("duration", Duration.between(start, Instant.now()).toMillis());
                batchResult.put("successful", true);
                
                logger.info("Batch collection completed - Success: {}, Errors: {}, Total: {}", 
                           successCount, errorCount, endHeight - startHeight + 1);
                
                return batchResult;
                
            } catch (Exception e) {
                logger.error("Error in batch collection", e);
                bitcoinErrors.increment();
                
                batchResult.put("startHeight", startHeight);
                batchResult.put("endHeight", endHeight);
                batchResult.put("successCount", successCount);
                batchResult.put("errorCount", errorCount);
                batchResult.put("collectionTime", start);
                batchResult.put("duration", Duration.between(start, Instant.now()).toMillis());
                batchResult.put("successful", false);
                batchResult.put("error", e.getMessage());
                
                return batchResult;
            }
        }, executorService);
    }
    
    private void publishBitcoinNetworkData(Map<String, Object> networkData) {
        try {
            String jsonData = objectMapper.writeValueAsString(networkData);
            kafkaTemplate.send("bitcoin-network-data", "network", jsonData);
            logger.debug("Published Bitcoin network data to Kafka");
        } catch (Exception e) {
            logger.error("Error publishing Bitcoin network data to Kafka", e);
        }
    }
    
    private void publishBitcoinBlockData(BitcoinBlock block) {
        try {
            String jsonData = objectMapper.writeValueAsString(block);
            kafkaTemplate.send("bitcoin-block-data", block.getHash(), jsonData);
            logger.debug("Published Bitcoin block data to Kafka: {}", block.getHash());
        } catch (Exception e) {
            logger.error("Error publishing Bitcoin block data to Kafka", e);
        }
    }
    
    private void publishBitcoinTransactionData(BitcoinTransaction transaction) {
        try {
            String jsonData = objectMapper.writeValueAsString(transaction);
            kafkaTemplate.send("bitcoin-transaction-data", transaction.getHash(), jsonData);
            logger.debug("Published Bitcoin transaction data to Kafka: {}", transaction.getHash());
        } catch (Exception e) {
            logger.error("Error publishing Bitcoin transaction data to Kafka", e);
        }
    }
    
    private void publishBitcoinAddressData(Map<String, Object> addressData) {
        try {
            String jsonData = objectMapper.writeValueAsString(addressData);
            kafkaTemplate.send("bitcoin-address-data", (String) addressData.get("address"), jsonData);
            logger.debug("Published Bitcoin address data to Kafka: {}", addressData.get("address"));
        } catch (Exception e) {
            logger.error("Error publishing Bitcoin address data to Kafka", e);
        }
    }
    
    /**
     * Get Bitcoin collection statistics
     */
    public Map<String, Object> getBitcoinCollectionStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("blocksCollected", bitcoinBlocksCollected.count());
        stats.put("transactionsCollected", bitcoinTransactionsCollected.count());
        stats.put("networkInfoCollected", bitcoinNetworkInfoCollected.count());
        stats.put("errors", bitcoinErrors.count());
        stats.put("averageCollectionTime", bitcoinCollectionTimer.mean(java.util.concurrent.TimeUnit.MILLISECONDS));
        stats.put("lastUpdate", Instant.now());
        return stats;
    }
}
