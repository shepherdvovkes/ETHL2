package com.defimon.collector.service;

import com.defimon.collector.client.BlockchainClientManager;
import com.defimon.collector.config.CollectorConfig;
import com.defimon.common.model.Asset;
import com.defimon.common.util.CollectionResult;
import com.defimon.common.util.CollectionStats;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.MeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Unit tests for EnterpriseDataCollectorService
 */
@ExtendWith(MockitoExtension.class)
class EnterpriseDataCollectorServiceTest {
    
    @Mock
    private KafkaTemplate<String, String> kafkaTemplate;
    
    @Mock
    private BlockchainClientManager clientManager;
    
    @Mock
    private CollectorConfig config;
    
    @Mock
    private MeterRegistry meterRegistry;
    
    @Mock
    private ObjectMapper objectMapper;
    
    @InjectMocks
    private EnterpriseDataCollectorService dataCollectorService;
    
    private Asset testAsset;
    
    @BeforeEach
    void setUp() {
        testAsset = Asset.builder()
            .symbol("BTC")
            .name("Bitcoin")
            .blockchain("bitcoin")
            .isActive(true)
            .build();
        
        when(config.getMaxConcurrentCollections()).thenReturn(10);
    }
    
    @Test
    void testCollectDataForAsset() throws Exception {
        // Given
        when(clientManager.getAssetBySymbol("BTC")).thenReturn(testAsset);
        when(clientManager.collectOnChainMetrics(testAsset)).thenReturn("{\"price\":50000}");
        when(clientManager.collectPriceData(testAsset)).thenReturn("{\"volume\":1000000}");
        when(clientManager.collectSocialMetrics(testAsset)).thenReturn("{\"sentiment\":0.8}");
        when(objectMapper.writeValueAsString(any())).thenReturn("{\"test\":\"data\"}");
        
        // When
        CompletableFuture<CollectionResult> future = dataCollectorService.collectDataForAsset("BTC");
        CollectionResult result = future.get();
        
        // Then
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertEquals("{\"price\":50000}", result.getOnChainData());
        assertEquals("{\"volume\":1000000}", result.getPriceData());
        assertEquals("{\"sentiment\":0.8}", result.getSocialData());
        assertTrue(result.isSuccessful());
        
        // Verify client calls
        verify(clientManager).getAssetBySymbol("BTC");
        verify(clientManager).collectOnChainMetrics(testAsset);
        verify(clientManager).collectPriceData(testAsset);
        verify(clientManager).collectSocialMetrics(testAsset);
    }
    
    @Test
    void testCollectDataForAssetNotFound() throws Exception {
        // Given
        when(clientManager.getAssetBySymbol("INVALID")).thenReturn(null);
        
        // When
        CompletableFuture<CollectionResult> future = dataCollectorService.collectDataForAsset("INVALID");
        CollectionResult result = future.get();
        
        // Then
        assertNotNull(result);
        assertNull(result.getAsset());
        assertFalse(result.isSuccessful());
        assertEquals("Asset not found: INVALID", result.getError());
    }
    
    @Test
    void testCollectDataForAssets() throws Exception {
        // Given
        List<String> symbols = Arrays.asList("BTC", "ETH");
        List<Asset> assets = Arrays.asList(
            testAsset,
            Asset.builder().symbol("ETH").name("Ethereum").blockchain("ethereum").isActive(true).build()
        );
        
        when(clientManager.getAssetBySymbol("BTC")).thenReturn(testAsset);
        when(clientManager.getAssetBySymbol("ETH")).thenReturn(assets.get(1));
        when(clientManager.collectOnChainMetrics(any())).thenReturn("{\"data\":\"test\"}");
        when(clientManager.collectPriceData(any())).thenReturn("{\"data\":\"test\"}");
        when(clientManager.collectSocialMetrics(any())).thenReturn("{\"data\":\"test\"}");
        when(objectMapper.writeValueAsString(any())).thenReturn("{\"test\":\"data\"}");
        
        // When
        CompletableFuture<List<CollectionResult>> future = dataCollectorService.collectDataForAssets(symbols);
        List<CollectionResult> results = future.get();
        
        // Then
        assertNotNull(results);
        assertEquals(2, results.size());
        assertTrue(results.stream().allMatch(CollectionResult::isSuccessful));
        
        // Verify client calls
        verify(clientManager, times(2)).collectOnChainMetrics(any());
        verify(clientManager, times(2)).collectPriceData(any());
        verify(clientManager, times(2)).collectSocialMetrics(any());
    }
    
    @Test
    void testCollectDataForAllAssets() throws Exception {
        // Given
        List<Asset> assets = Arrays.asList(testAsset);
        when(clientManager.getActiveAssets()).thenReturn(assets);
        when(clientManager.collectOnChainMetrics(any())).thenReturn("{\"data\":\"test\"}");
        when(clientManager.collectPriceData(any())).thenReturn("{\"data\":\"test\"}");
        when(clientManager.collectSocialMetrics(any())).thenReturn("{\"data\":\"test\"}");
        when(objectMapper.writeValueAsString(any())).thenReturn("{\"test\":\"data\"}");
        
        // When
        CompletableFuture<List<CollectionResult>> future = dataCollectorService.collectDataForAllAssets(assets);
        List<CollectionResult> results = future.get();
        
        // Then
        assertNotNull(results);
        assertEquals(1, results.size());
        assertTrue(results.get(0).isSuccessful());
    }
    
    @Test
    void testGetCollectionStats() {
        // When
        CollectionStats stats = dataCollectorService.getCollectionStats();
        
        // Then
        assertNotNull(stats);
        assertEquals(10, stats.getMaxConcurrentCollections());
        assertNotNull(stats.getLastCollectionTime());
    }
    
    @Test
    void testFallbackCollectAssetData() {
        // Given
        Exception exception = new RuntimeException("Service unavailable");
        
        // When
        CollectionResult result = dataCollectorService.fallbackCollectAssetData(testAsset, exception);
        
        // Then
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertFalse(result.isSuccessful());
        assertTrue(result.getError().contains("Circuit breaker activated"));
    }
    
    @Test
    void testServiceInitialization() {
        // Test that the service initializes properly
        assertNotNull(dataCollectorService);
        
        // Verify that the service has the correct configuration
        // This would be tested through the @PostConstruct method
    }
    
    @Test
    void testKafkaPublishing() throws Exception {
        // Given
        when(clientManager.getAssetBySymbol("BTC")).thenReturn(testAsset);
        when(clientManager.collectOnChainMetrics(testAsset)).thenReturn("{\"price\":50000}");
        when(clientManager.collectPriceData(testAsset)).thenReturn("{\"volume\":1000000}");
        when(clientManager.collectSocialMetrics(testAsset)).thenReturn("{\"sentiment\":0.8}");
        when(objectMapper.writeValueAsString(any())).thenReturn("{\"test\":\"data\"}");
        
        // When
        CompletableFuture<CollectionResult> future = dataCollectorService.collectDataForAsset("BTC");
        CollectionResult result = future.get();
        
        // Then
        assertTrue(result.isSuccessful());
        
        // Verify Kafka publishing
        verify(kafkaTemplate).send(eq("asset-collection-results"), eq("BTC"), anyString());
    }
}
