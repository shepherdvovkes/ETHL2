package com.defimon.collector.service;

import com.defimon.common.model.Asset;
import com.defimon.common.model.CollectionStats;
import com.defimon.common.util.CollectionResult;
import com.defimon.collector.client.BlockchainClientManager;
import com.defimon.collector.config.CollectorConfig;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

/**
 * Enterprise-grade data collector service with high-performance parallel processing
 */
@Service
public class EnterpriseDataCollectorService {
    
    private static final Logger logger = LoggerFactory.getLogger(EnterpriseDataCollectorService.class);
    
    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;
    
    @Autowired
    private BlockchainClientManager clientManager;
    
    @Autowired
    private CollectorConfig config;
    
    @Autowired
    private MeterRegistry meterRegistry;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    private ExecutorService executorService;
    private ForkJoinPool forkJoinPool;
    
    // Metrics
    private Counter successCounter;
    private Counter errorCounter;
    private Timer collectionTimer;
    private Counter assetsProcessedCounter;
    
    @PostConstruct
    public void init() {
        // Initialize thread pools optimized for I/O operations
        this.executorService = Executors.newFixedThreadPool(
            config.getMaxConcurrentCollections());
        this.forkJoinPool = new ForkJoinPool(
            Runtime.getRuntime().availableProcessors() * 2);
        
        // Initialize metrics
        this.successCounter = Counter.builder("data.collection.success")
            .description("Successful data collections")
            .register(meterRegistry);
        this.errorCounter = Counter.builder("data.collection.errors")
            .description("Failed data collections")
            .register(meterRegistry);
        this.collectionTimer = Timer.builder("data.collection.duration")
            .description("Data collection processing time")
            .register(meterRegistry);
        this.assetsProcessedCounter = Counter.builder("data.collection.assets.processed")
            .description("Total assets processed")
            .register(meterRegistry);
        
        logger.info("Enterprise Data Collector Service initialized with {} threads", 
                   config.getMaxConcurrentCollections());
    }
    
    @Scheduled(fixedRateString = "${defimon.collector.interval:60000}")
    public void scheduleDataCollection() {
        logger.info("Starting scheduled data collection cycle...");
        
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            List<Asset> assets = getActiveAssets();
            collectDataForAllAssets(assets)
                .whenComplete((results, throwable) -> {
                    sample.stop(collectionTimer);
                    
                    if (throwable != null) {
                        logger.error("Data collection cycle failed", throwable);
                        errorCounter.increment();
                    } else {
                        long successful = results.stream()
                            .mapToLong(CollectionResult::getSuccessfulCollections)
                            .sum();
                        long failed = results.stream()
                            .mapToLong(CollectionResult::getFailedCollections)
                            .sum();
                        
                        successCounter.increment(successful);
                        errorCounter.increment(failed);
                        assetsProcessedCounter.increment(assets.size());
                        
                        logger.info("Data collection cycle completed. Success: {}, Failed: {}, Assets: {}", 
                                   successful, failed, assets.size());
                    }
                });
                
        } catch (Exception e) {
            sample.stop(collectionTimer);
            logger.error("Error starting data collection cycle", e);
            errorCounter.increment();
        }
    }
    
    @Async
    public CompletableFuture<List<CollectionResult>> collectDataForAllAssets(List<Asset> assets) {
        return CompletableFuture.supplyAsync(() -> {
            // Use parallel streams for CPU-intensive operations
            return assets.parallelStream()
                .map(this::collectAssetDataSync)
                .collect(Collectors.toList());
        }, forkJoinPool);
    }
    
    @CircuitBreaker(name = "asset-collection", fallbackMethod = "fallbackCollectAssetData")
    @Retry(name = "asset-collection")
    private CollectionResult collectAssetDataSync(Asset asset) {
        Instant start = Instant.now();
        
        try {
            logger.debug("Collecting data for asset: {}", asset.getSymbol());
            
            // Collect from multiple sources in parallel
            CompletableFuture<String> onChainData = CompletableFuture
                .supplyAsync(() -> clientManager.collectOnChainMetrics(asset), executorService);
            
            CompletableFuture<String> priceData = CompletableFuture
                .supplyAsync(() -> clientManager.collectPriceData(asset), executorService);
            
            CompletableFuture<String> socialData = CompletableFuture
                .supplyAsync(() -> clientManager.collectSocialMetrics(asset), executorService);
            
            // Wait for all data collection to complete
            CompletableFuture<Void> allData = CompletableFuture.allOf(
                onChainData, priceData, socialData);
            
            allData.get(); // Wait for completion
            
            // Combine results
            CollectionResult result = CollectionResult.builder()
                .asset(asset)
                .onChainData(onChainData.get())
                .priceData(priceData.get())
                .socialData(socialData.get())
                .collectionTime(start)
                .duration(Duration.between(start, Instant.now()))
                .successful(true)
                .build();
            
            // Publish to Kafka for further processing
            publishCollectionResult(result);
            
            return result;
            
        } catch (Exception e) {
            logger.error("Error collecting data for asset {}: {}", asset.getSymbol(), e.getMessage());
            
            return CollectionResult.builder()
                .asset(asset)
                .collectionTime(start)
                .duration(Duration.between(start, Instant.now()))
                .successful(false)
                .error(e.getMessage())
                .build();
        }
    }
    
    private CollectionResult fallbackCollectAssetData(Asset asset, Exception ex) {
        logger.warn("Fallback triggered for asset {}: {}", asset.getSymbol(), ex.getMessage());
        
        return CollectionResult.builder()
            .asset(asset)
            .collectionTime(Instant.now())
            .duration(Duration.ZERO)
            .successful(false)
            .error("Circuit breaker activated: " + ex.getMessage())
            .build();
    }
    
    private void publishCollectionResult(CollectionResult result) {
        try {
            String jsonResult = objectMapper.writeValueAsString(result);
            kafkaTemplate.send("asset-collection-results", result.getAsset().getSymbol(), jsonResult);
            
            logger.debug("Published data to Kafka for asset: {}", result.getAsset().getSymbol());
            
        } catch (Exception e) {
            logger.error("Error publishing collection result to Kafka", e);
        }
    }
    
    private List<Asset> getActiveAssets() {
        // Implementation to fetch active assets from database
        // This would integrate with Asset Management Service
        return clientManager.getActiveAssets();
    }
    
    /**
     * Manual trigger for data collection
     */
    public CompletableFuture<List<CollectionResult>> collectDataForAssets(List<String> symbols) {
        logger.info("Manual data collection triggered for symbols: {}", symbols);
        
        List<Asset> assets = symbols.stream()
            .map(clientManager::getAssetBySymbol)
            .filter(asset -> asset != null)
            .collect(Collectors.toList());
        
        return collectDataForAllAssets(assets);
    }
    
    /**
     * Collect data for a specific asset
     */
    public CompletableFuture<CollectionResult> collectDataForAsset(String symbol) {
        logger.info("Collecting data for specific asset: {}", symbol);
        
        Asset asset = clientManager.getAssetBySymbol(symbol);
        if (asset == null) {
            return CompletableFuture.completedFuture(
                CollectionResult.failure(null, "Asset not found: " + symbol));
        }
        
        return CompletableFuture.supplyAsync(() -> collectAssetDataSync(asset), executorService);
    }
    
    /**
     * Get collection statistics
     */
    public CollectionStats getCollectionStats() {
        return CollectionStats.builder()
            .totalSuccess(successCounter.count())
            .totalErrors(errorCounter.count())
            .totalAssetsProcessed(assetsProcessedCounter.count())
            .averageCollectionTime(collectionTimer.mean(java.util.concurrent.TimeUnit.MILLISECONDS))
            .maxConcurrentCollections(config.getMaxConcurrentCollections())
            .lastCollectionTime(Instant.now())
            .build();
    }
}
