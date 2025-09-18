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
 * Financial metrics for cryptocurrency assets
 */
@Entity
@Table(name = "financial_metrics", indexes = {
    @Index(name = "idx_financial_asset_timestamp", columnList = "asset_id, timestamp"),
    @Index(name = "idx_financial_timestamp", columnList = "timestamp")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class FinancialMetrics {
    
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
    
    // Fundamental Analysis
    @Column(name = "price_to_earnings_ratio", precision = 10, scale = 4)
    private BigDecimal priceToEarningsRatio;
    
    @Column(name = "price_to_sales_ratio", precision = 10, scale = 4)
    private BigDecimal priceToSalesRatio;
    
    @Column(name = "price_to_book_ratio", precision = 10, scale = 4)
    private BigDecimal priceToBookRatio;
    
    @Column(name = "debt_to_equity_ratio", precision = 10, scale = 4)
    private BigDecimal debtToEquityRatio;
    
    @Column(name = "current_ratio", precision = 10, scale = 4)
    private BigDecimal currentRatio;
    
    @Column(name = "quick_ratio", precision = 10, scale = 4)
    private BigDecimal quickRatio;
    
    // Revenue and Profitability
    @Column(name = "revenue_24h", precision = 20, scale = 2)
    private BigDecimal revenue24h;
    
    @Column(name = "revenue_7d", precision = 20, scale = 2)
    private BigDecimal revenue7d;
    
    @Column(name = "revenue_30d", precision = 20, scale = 2)
    private BigDecimal revenue30d;
    
    @Column(name = "revenue_growth_rate", precision = 10, scale = 4)
    private BigDecimal revenueGrowthRate;
    
    @Column(name = "gross_profit_margin", precision = 10, scale = 4)
    private BigDecimal grossProfitMargin;
    
    @Column(name = "net_profit_margin", precision = 10, scale = 4)
    private BigDecimal netProfitMargin;
    
    @Column(name = "operating_margin", precision = 10, scale = 4)
    private BigDecimal operatingMargin;
    
    // Cash Flow
    @Column(name = "operating_cash_flow", precision = 20, scale = 2)
    private BigDecimal operatingCashFlow;
    
    @Column(name = "investing_cash_flow", precision = 20, scale = 2)
    private BigDecimal investingCashFlow;
    
    @Column(name = "financing_cash_flow", precision = 20, scale = 2)
    private BigDecimal financingCashFlow;
    
    @Column(name = "free_cash_flow", precision = 20, scale = 2)
    private BigDecimal freeCashFlow;
    
    // Balance Sheet
    @Column(name = "total_assets", precision = 20, scale = 2)
    private BigDecimal totalAssets;
    
    @Column(name = "total_liabilities", precision = 20, scale = 2)
    private BigDecimal totalLiabilities;
    
    @Column(name = "total_equity", precision = 20, scale = 2)
    private BigDecimal totalEquity;
    
    @Column(name = "working_capital", precision = 20, scale = 2)
    private BigDecimal workingCapital;
    
    @Column(name = "cash_and_equivalents", precision = 20, scale = 2)
    private BigDecimal cashAndEquivalents;
    
    @Column(name = "short_term_debt", precision = 20, scale = 2)
    private BigDecimal shortTermDebt;
    
    @Column(name = "long_term_debt", precision = 20, scale = 2)
    private BigDecimal longTermDebt;
    
    // Valuation Metrics
    @Column(name = "enterprise_value", precision = 20, scale = 2)
    private BigDecimal enterpriseValue;
    
    @Column(name = "ev_to_revenue", precision = 10, scale = 4)
    private BigDecimal evToRevenue;
    
    @Column(name = "ev_to_ebitda", precision = 10, scale = 4)
    private BigDecimal evToEbitda;
    
    @Column(name = "price_to_cash_flow", precision = 10, scale = 4)
    private BigDecimal priceToCashFlow;
    
    @Column(name = "price_to_free_cash_flow", precision = 10, scale = 4)
    private BigDecimal priceToFreeCashFlow;
    
    // Growth Metrics
    @Column(name = "earnings_growth_rate", precision = 10, scale = 4)
    private BigDecimal earningsGrowthRate;
    
    @Column(name = "book_value_growth_rate", precision = 10, scale = 4)
    private BigDecimal bookValueGrowthRate;
    
    @Column(name = "dividend_yield", precision = 10, scale = 4)
    private BigDecimal dividendYield;
    
    @Column(name = "payout_ratio", precision = 10, scale = 4)
    private BigDecimal payoutRatio;
    
    // Risk Metrics
    @Column(name = "beta", precision = 10, scale = 4)
    private BigDecimal beta;
    
    @Column(name = "volatility_30d", precision = 10, scale = 4)
    private BigDecimal volatility30d;
    
    @Column(name = "volatility_90d", precision = 10, scale = 4)
    private BigDecimal volatility90d;
    
    @Column(name = "sharpe_ratio", precision = 10, scale = 4)
    private BigDecimal sharpeRatio;
    
    @Column(name = "sortino_ratio", precision = 10, scale = 4)
    private BigDecimal sortinoRatio;
    
    @Column(name = "max_drawdown", precision = 10, scale = 4)
    private BigDecimal maxDrawdown;
    
    // Liquidity Metrics
    @Column(name = "bid_ask_spread", precision = 10, scale = 6)
    private BigDecimal bidAskSpread;
    
    @Column(name = "market_depth", precision = 20, scale = 2)
    private BigDecimal marketDepth;
    
    @Column(name = "liquidity_score", precision = 10, scale = 4)
    private BigDecimal liquidityScore;
    
    // Additional Financial Indicators
    @Column(name = "roa", precision = 10, scale = 4)
    private BigDecimal returnOnAssets;
    
    @Column(name = "roe", precision = 10, scale = 4)
    private BigDecimal returnOnEquity;
    
    @Column(name = "roic", precision = 10, scale = 4)
    private BigDecimal returnOnInvestedCapital;
    
    @Column(name = "asset_turnover", precision = 10, scale = 4)
    private BigDecimal assetTurnover;
    
    @Column(name = "inventory_turnover", precision = 10, scale = 4)
    private BigDecimal inventoryTurnover;
    
    @Column(name = "receivables_turnover", precision = 10, scale = 4)
    private BigDecimal receivablesTurnover;
    
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
     * Calculate return on investment
     */
    public BigDecimal calculateROI(BigDecimal initialInvestment, BigDecimal currentValue) {
        if (initialInvestment == null || currentValue == null || 
            initialInvestment.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        return currentValue.subtract(initialInvestment)
                         .divide(initialInvestment, 6, BigDecimal.ROUND_HALF_UP)
                         .multiply(BigDecimal.valueOf(100));
    }
    
    /**
     * Calculate debt-to-assets ratio
     */
    public BigDecimal calculateDebtToAssetsRatio() {
        if (totalDebt() == null || totalAssets == null || 
            totalAssets.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        return totalDebt().divide(totalAssets, 6, BigDecimal.ROUND_HALF_UP);
    }
    
    /**
     * Get total debt (short term + long term)
     */
    public BigDecimal totalDebt() {
        BigDecimal shortTerm = shortTermDebt != null ? shortTermDebt : BigDecimal.ZERO;
        BigDecimal longTerm = longTermDebt != null ? longTermDebt : BigDecimal.ZERO;
        return shortTerm.add(longTerm);
    }
    
    /**
     * Calculate interest coverage ratio
     */
    public BigDecimal calculateInterestCoverageRatio(BigDecimal interestExpense) {
        if (operatingCashFlow == null || interestExpense == null || 
            interestExpense.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        
        return operatingCashFlow.divide(interestExpense, 6, BigDecimal.ROUND_HALF_UP);
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
