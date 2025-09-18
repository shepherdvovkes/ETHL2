package com.defimon.bitcoin.service;

import com.defimon.bitcoin.client.QuickNodeBitcoinClient;
import com.defimon.bitcoin.model.*;
import com.fasterxml.jackson.databind.ObjectMapper;
// import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
// import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
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

/**
 * Bitcoin Data Collector Service - Java 8 + Spring Boot
 * Collects Bitcoin data from QuickNode API
 */
@Service
public class BitcoinDataCollectorService {
    
    private static final Logger logger = LoggerFactory.getLogger(BitcoinDataCollectorService.class);
    
    @Autowired
    private QuickNodeBitcoinClient bitcoinClient;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Value("${bitcoin.collector.interval:60000}")
    private long collectionInterval;
    
    @Value("${bitcoin.collector.enabled:true}")
    private boolean collectionEnabled;
    
    @PostConstruct
    public void init() {
        logger.info("Bitcoin Data Collector Service initialized");
        logger.info("Collection interval: {} ms", collectionInterval);
        logger.info("Collection enabled: {}", collectionEnabled);
    }
    
    /**
     * Scheduled Bitcoin data collection
     */
    @Scheduled(fixedRateString = "${bitcoin.collector.interval:60000}")
    public void scheduleBitcoinDataCollection() {
        if (!collectionEnabled) {
            logger.debug("Bitcoin data collection is disabled");
            return;
        }
        
        logger.info("Starting scheduled Bitcoin data collection...");
        
        try {
            collectBitcoinNetworkData()
                .whenComplete((result, throwable) -> {
                    if (throwable != null) {
                        logger.error("Bitcoin data collection failed", throwable);
                    } else {
                        logger.info("Bitcoin data collection completed successfully");
                        logger.info("Collected data: {}", result);
                    }
                });
                
        } catch (Exception e) {
            logger.error("Error starting Bitcoin data collection", e);
        }
    }
    
    /**
     * Collect Bitcoin network data
     */
    @Async
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
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
                
                return networkData;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin network data", e);
                
                networkData.put("collectionTime", start);
                networkData.put("duration", Duration.between(start, Instant.now()).toMillis());
                networkData.put("successful", false);
                networkData.put("error", e.getMessage());
                
                return networkData;
            }
        });
    }
    
    /**
     * Collect Bitcoin block data
     */
    @Async
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
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
                    logger.info("Successfully collected block {} at height {}", blockHash, blockHeight);
                }
                
                return block;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin block at height {}: {}", blockHeight, e.getMessage());
                return null;
            }
        });
    }
    
    /**
     * Collect Bitcoin transaction data
     */
    @Async
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
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
                    logger.info("Successfully collected transaction: {}", txHash);
                }
                
                return transaction;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin transaction {}: {}", txHash, e.getMessage());
                return null;
            }
        });
    }
    
    /**
     * Collect Bitcoin address data
     */
    @Async
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
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
                
                return addressData;
                
            } catch (Exception e) {
                logger.error("Error collecting Bitcoin address data for {}: {}", address, e.getMessage());
                
                addressData.put("address", address);
                addressData.put("collectionTime", start);
                addressData.put("duration", Duration.between(start, Instant.now()).toMillis());
                addressData.put("successful", false);
                addressData.put("error", e.getMessage());
                
                return addressData;
            }
        });
    }
    
    /**
     * Manual trigger for data collection
     */
    public CompletableFuture<Map<String, Object>> collectDataNow() {
        logger.info("Manual Bitcoin data collection triggered");
        return collectBitcoinNetworkData();
    }
    
    /**
     * Get service status
     */
    public Map<String, Object> getServiceStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("service", "BitcoinDataCollectorService");
        status.put("enabled", collectionEnabled);
        status.put("interval", collectionInterval);
        status.put("status", "RUNNING");
        status.put("timestamp", Instant.now());
        return status;
    }
}
