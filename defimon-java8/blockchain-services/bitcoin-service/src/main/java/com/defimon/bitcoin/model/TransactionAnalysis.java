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
 * Bitcoin transaction analysis result
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TransactionAnalysis {
    
    private String txHash;
    private int inputCount;
    private int outputCount;
    private BigDecimal totalValue;
    private BigDecimal fee;
    private boolean isSegwit;
    private int confirmations;
    private Long blockHeight;
    private Long timestamp;
    private String error;
    
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}
