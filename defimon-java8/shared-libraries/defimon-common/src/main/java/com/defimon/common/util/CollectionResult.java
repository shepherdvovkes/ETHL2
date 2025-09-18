package com.defimon.common.util;

import com.defimon.common.model.Asset;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/**
 * Result of data collection operation
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CollectionResult {
    
    private Asset asset;
    private String onChainData;
    private String priceData;
    private String socialData;
    private Instant collectionTime;
    private Duration duration;
    private boolean successful;
    private String error;
    private List<String> warnings;
    private int successfulCollections;
    private int failedCollections;
    
    /**
     * Create a successful collection result
     */
    public static CollectionResult success(Asset asset, String onChainData, 
                                         String priceData, String socialData) {
        return CollectionResult.builder()
            .asset(asset)
            .onChainData(onChainData)
            .priceData(priceData)
            .socialData(socialData)
            .collectionTime(Instant.now())
            .successful(true)
            .successfulCollections(1)
            .failedCollections(0)
            .build();
    }
    
    /**
     * Create a failed collection result
     */
    public static CollectionResult failure(Asset asset, String error) {
        return CollectionResult.builder()
            .asset(asset)
            .collectionTime(Instant.now())
            .successful(false)
            .error(error)
            .successfulCollections(0)
            .failedCollections(1)
            .build();
    }
    
    /**
     * Create a failed collection result with exception
     */
    public static CollectionResult failure(Asset asset, Exception exception) {
        return CollectionResult.builder()
            .asset(asset)
            .collectionTime(Instant.now())
            .successful(false)
            .error(exception.getMessage())
            .successfulCollections(0)
            .failedCollections(1)
            .build();
    }
    
    /**
     * Check if collection has warnings
     */
    public boolean hasWarnings() {
        return warnings != null && !warnings.isEmpty();
    }
    
    /**
     * Add a warning to the result
     */
    public void addWarning(String warning) {
        if (warnings == null) {
            warnings = new java.util.ArrayList<>();
        }
        warnings.add(warning);
    }
    
    /**
     * Get collection duration in milliseconds
     */
    public long getDurationMillis() {
        return duration != null ? duration.toMillis() : 0;
    }
    
    /**
     * Get collection duration in seconds
     */
    public double getDurationSeconds() {
        return duration != null ? duration.toMillis() / 1000.0 : 0.0;
    }
}
