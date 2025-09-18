package com.defimon.common.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/**
 * Collection statistics for monitoring data collection performance
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CollectionStats {
    
    private double totalSuccess;
    private double totalErrors;
    private double totalAssetsProcessed;
    private double averageCollectionTime;
    private int maxConcurrentCollections;
    
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private Instant lastCollectionTime;
    
    private double successRate;
    private double errorRate;
    private double throughput;
    
    /**
     * Calculate success rate
     */
    public double getSuccessRate() {
        double total = totalSuccess + totalErrors;
        return total > 0 ? (totalSuccess / total) * 100 : 0.0;
    }
    
    /**
     * Calculate error rate
     */
    public double getErrorRate() {
        double total = totalSuccess + totalErrors;
        return total > 0 ? (totalErrors / total) * 100 : 0.0;
    }
    
    /**
     * Calculate throughput (assets per minute)
     */
    public double getThroughput() {
        if (lastCollectionTime != null) {
            long minutesElapsed = java.time.Duration.between(
                lastCollectionTime.minusSeconds(3600), // Last hour
                Instant.now()
            ).toMinutes();
            return minutesElapsed > 0 ? totalAssetsProcessed / minutesElapsed : 0.0;
        }
        return 0.0;
    }
}
