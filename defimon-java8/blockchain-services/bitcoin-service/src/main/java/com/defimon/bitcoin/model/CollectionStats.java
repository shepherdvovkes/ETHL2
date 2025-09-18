package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Collection statistics model
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CollectionStats {
    
    private Long totalMetrics;
    private Long totalBlocks;
    private Long totalTransactions;
    private Long lastCollectedBlock;
    private Boolean isCollecting;
    private String status;
    private Long timestamp;
    
    public CollectionStats(Long totalMetrics, Long totalBlocks, Long totalTransactions, 
                          Long lastCollectedBlock, Boolean isCollecting) {
        this.totalMetrics = totalMetrics;
        this.totalBlocks = totalBlocks;
        this.totalTransactions = totalTransactions;
        this.lastCollectedBlock = lastCollectedBlock;
        this.isCollecting = isCollecting;
        this.status = isCollecting ? "COLLECTING" : "IDLE";
        this.timestamp = System.currentTimeMillis();
    }
}
