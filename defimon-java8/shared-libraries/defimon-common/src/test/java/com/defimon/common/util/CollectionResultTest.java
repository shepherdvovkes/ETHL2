package com.defimon.common.util;

import com.defimon.common.model.Asset;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;

/**
 * Unit tests for CollectionResult
 */
class CollectionResultTest {
    
    private Asset testAsset;
    
    @BeforeEach
    void setUp() {
        testAsset = Asset.builder()
            .symbol("BTC")
            .name("Bitcoin")
            .blockchain("bitcoin")
            .build();
    }
    
    @Test
    void testSuccessCreation() {
        CollectionResult result = CollectionResult.success(
            testAsset,
            "{\"price\":50000}",
            "{\"volume\":1000000}",
            "{\"sentiment\":0.8}"
        );
        
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertEquals("{\"price\":50000}", result.getOnChainData());
        assertEquals("{\"volume\":1000000}", result.getPriceData());
        assertEquals("{\"sentiment\":0.8}", result.getSocialData());
        assertTrue(result.isSuccessful());
        assertEquals(1, result.getSuccessfulCollections());
        assertEquals(0, result.getFailedCollections());
    }
    
    @Test
    void testFailureCreation() {
        CollectionResult result = CollectionResult.failure(testAsset, "Connection timeout");
        
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertEquals("Connection timeout", result.getError());
        assertFalse(result.isSuccessful());
        assertEquals(0, result.getSuccessfulCollections());
        assertEquals(1, result.getFailedCollections());
    }
    
    @Test
    void testFailureWithException() {
        Exception exception = new RuntimeException("Network error");
        CollectionResult result = CollectionResult.failure(testAsset, exception);
        
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertEquals("Network error", result.getError());
        assertFalse(result.isSuccessful());
    }
    
    @Test
    void testHasWarnings() {
        CollectionResult result = CollectionResult.success(testAsset, "", "", "");
        
        // Initially no warnings
        assertFalse(result.hasWarnings());
        
        // Add a warning
        result.addWarning("Rate limit approaching");
        assertTrue(result.hasWarnings());
        assertEquals(1, result.getWarnings().size());
        assertEquals("Rate limit approaching", result.getWarnings().get(0));
    }
    
    @Test
    void testDurationCalculations() {
        Instant start = Instant.now().minusSeconds(5);
        CollectionResult result = CollectionResult.builder()
            .asset(testAsset)
            .collectionTime(start)
            .duration(Duration.between(start, Instant.now()))
            .successful(true)
            .build();
        
        assertTrue(result.getDurationMillis() > 0);
        assertTrue(result.getDurationSeconds() > 0);
        assertTrue(result.getDurationSeconds() < 10); // Should be around 5 seconds
    }
    
    @Test
    void testBuilderPattern() {
        CollectionResult result = CollectionResult.builder()
            .asset(testAsset)
            .onChainData("{\"data\":\"test\"}")
            .priceData("{\"price\":100}")
            .socialData("{\"sentiment\":0.5}")
            .collectionTime(Instant.now())
            .duration(Duration.ofSeconds(2))
            .successful(true)
            .successfulCollections(1)
            .failedCollections(0)
            .build();
        
        assertNotNull(result);
        assertEquals(testAsset, result.getAsset());
        assertTrue(result.isSuccessful());
        assertEquals(1, result.getSuccessfulCollections());
        assertEquals(0, result.getFailedCollections());
    }
}
