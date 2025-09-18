package com.defimon.common.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * On-chain metrics for cryptocurrency assets
 */
@Entity
@Table(name = "onchain_metrics", indexes = {
    @Index(name = "idx_onchain_asset_timestamp", columnList = "asset_id, timestamp"),
    @Index(name = "idx_onchain_timestamp", columnList = "timestamp")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class OnChainMetrics {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @NotNull
    @Column(name = "asset_id", nullable = false)
    private Long assetId;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "asset_id", insertable = false, updatable = false)
    private Asset asset;
    
    @NotNull
    @Column(name = "timestamp", nullable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime timestamp;
    
    // Price and Market Data
    @Column(name = "price_usd", precision = 20, scale = 8)
    private BigDecimal priceUsd;
    
    @Column(name = "price_btc", precision = 20, scale = 8)
    private BigDecimal priceBtc;
    
    @Column(name = "price_eth", precision = 20, scale = 8)
    private BigDecimal priceEth;
    
    @Column(name = "market_cap", precision = 20, scale = 2)
    private BigDecimal marketCap;
    
    @Column(name = "fully_diluted_valuation", precision = 20, scale = 2)
    private BigDecimal fullyDilutedValuation;
    
    // Volume Data
    @Column(name = "volume_24h", precision = 20, scale = 2)
    private BigDecimal volume24h;
    
    @Column(name = "volume_change_24h", precision = 10, scale = 4)
    private BigDecimal volumeChange24h;
    
    // Price Changes
    @Column(name = "price_change_1h", precision = 10, scale = 4)
    private BigDecimal priceChange1h;
    
    @Column(name = "price_change_24h", precision = 10, scale = 4)
    private BigDecimal priceChange24h;
    
    @Column(name = "price_change_7d", precision = 10, scale = 4)
    private BigDecimal priceChange7d;
    
    @Column(name = "price_change_30d", precision = 10, scale = 4)
    private BigDecimal priceChange30d;
    
    // High/Low Data
    @Column(name = "high_24h", precision = 20, scale = 8)
    private BigDecimal high24h;
    
    @Column(name = "low_24h", precision = 20, scale = 8)
    private BigDecimal low24h;
    
    @Column(name = "high_7d", precision = 20, scale = 8)
    private BigDecimal high7d;
    
    @Column(name = "low_7d", precision = 20, scale = 8)
    private BigDecimal low7d;
    
    @Column(name = "high_30d", precision = 20, scale = 8)
    private BigDecimal high30d;
    
    @Column(name = "low_30d", precision = 20, scale = 8)
    private BigDecimal low30d;
    
    // Blockchain Specific Metrics
    @Column(name = "active_addresses_24h")
    private Long activeAddresses24h;
    
    @Column(name = "active_addresses_7d")
    private Long activeAddresses7d;
    
    @Column(name = "active_addresses_30d")
    private Long activeAddresses30d;
    
    @Column(name = "transaction_count_24h")
    private Long transactionCount24h;
    
    @Column(name = "transaction_volume_24h", precision = 20, scale = 8)
    private BigDecimal transactionVolume24h;
    
    @Column(name = "average_transaction_value", precision = 20, scale = 8)
    private BigDecimal averageTransactionValue;
    
    // Network Metrics
    @Column(name = "network_hash_rate", precision = 20, scale = 2)
    private BigDecimal networkHashRate;
    
    @Column(name = "difficulty", precision = 20, scale = 2)
    private BigDecimal difficulty;
    
    @Column(name = "block_height")
    private Long blockHeight;
    
    @Column(name = "block_time_avg", precision = 10, scale = 4)
    private BigDecimal blockTimeAvg;
    
    @Column(name = "gas_price_avg", precision = 20, scale = 8)
    private BigDecimal gasPriceAvg;
    
    @Column(name = "gas_used_avg")
    private Long gasUsedAvg;
    
    // DeFi Specific Metrics
    @Column(name = "tvl", precision = 20, scale = 2)
    private BigDecimal tvl; // Total Value Locked
    
    @Column(name = "tvl_change_24h", precision = 10, scale = 4)
    private BigDecimal tvlChange24h;
    
    @Column(name = "yield_farming_apy", precision = 10, scale = 4)
    private BigDecimal yieldFarmingApy;
    
    @Column(name = "liquidity_pools_count")
    private Integer liquidityPoolsCount;
    
    @Column(name = "dex_volume_24h", precision = 20, scale = 2)
    private BigDecimal dexVolume24h;
    
    // Supply Metrics
    @Column(name = "supply_inflation_rate", precision = 10, scale = 6)
    private BigDecimal supplyInflationRate;
    
    @Column(name = "burn_rate_24h", precision = 20, scale = 8)
    private BigDecimal burnRate24h;
    
    @Column(name = "staked_amount", precision = 20, scale = 8)
    private BigDecimal stakedAmount;
    
    @Column(name = "staking_apy", precision = 10, scale = 4)
    private BigDecimal stakingApy;
    
    // Additional Metrics
    @Column(name = "whale_transactions_24h")
    private Integer whaleTransactions24h;
    
    @Column(name = "exchange_inflows_24h", precision = 20, scale = 8)
    private BigDecimal exchangeInflows24h;
    
    @Column(name = "exchange_outflows_24h", precision = 20, scale = 8)
    private BigDecimal exchangeOutflows24h;
    
    @Column(name = "exchange_netflow_24h", precision = 20, scale = 8)
    private BigDecimal exchangeNetflow24h;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (timestamp == null) {
            timestamp = LocalDateTime.now();
        }
    }
    
    /**
     * Calculate price volatility based on 24h high/low
     */
    public BigDecimal calculateVolatility24h() {
        if (high24h == null || low24h == null || priceUsd == null || priceUsd.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        BigDecimal range = high24h.subtract(low24h);
        return range.divide(priceUsd, 6, BigDecimal.ROUND_HALF_UP);
    }
    
    /**
     * Calculate market dominance if market cap is available
     */
    public BigDecimal calculateMarketDominance(BigDecimal totalMarketCap) {
        if (marketCap == null || totalMarketCap == null || totalMarketCap.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        return marketCap.divide(totalMarketCap, 6, BigDecimal.ROUND_HALF_UP)
                       .multiply(BigDecimal.valueOf(100));
    }
    
    /**
     * Check if metrics are recent (within specified minutes)
     */
    public boolean isRecent(int minutesThreshold) {
        if (timestamp == null) {
            return false;
        }
        return timestamp.isAfter(LocalDateTime.now().minusMinutes(minutesThreshold));
    }
}
