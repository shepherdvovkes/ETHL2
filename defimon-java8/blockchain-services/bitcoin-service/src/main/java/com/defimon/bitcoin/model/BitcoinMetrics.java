package com.defimon.bitcoin.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Bitcoin network metrics
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "bitcoin_metrics")
public class BitcoinMetrics {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private Long blockCount;
    private Long blockHeight;
    private String blockHash;
    private Double difficulty;
    private BigDecimal networkHashRate;
    private Long mempoolSize;
    private Long mempoolBytes;
    private Long mempoolUsage;
    private BigDecimal averageFeeRate;
    private Double feeEstimate6Blocks;
    private Integer networkConnections;
    private Integer networkVersion;
    private String metricType;
    private Long timestamp;
    private String error;
    
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (timestamp == null) {
            timestamp = System.currentTimeMillis();
        }
    }
}
