package com.defimon.common.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.DecimalMax;
import javax.validation.constraints.DecimalMin;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Machine Learning predictions for cryptocurrency assets
 */
@Entity
@Table(name = "ml_predictions", indexes = {
    @Index(name = "idx_ml_asset_timestamp", columnList = "asset_id, created_at"),
    @Index(name = "idx_ml_model", columnList = "model_name"),
    @Index(name = "idx_ml_horizon", columnList = "prediction_horizon"),
    @Index(name = "idx_ml_confidence", columnList = "confidence_score")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class MLPrediction {
    
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
    @Column(name = "model_name", nullable = false, length = 100)
    private String modelName;
    
    @NotNull
    @Column(name = "model_version", nullable = false, length = 50)
    private String modelVersion;
    
    @NotNull
    @Column(name = "prediction_type", nullable = false, length = 50)
    private String predictionType;
    
    @NotNull
    @Column(name = "prediction_horizon", nullable = false, length = 20)
    private String predictionHorizon; // 1h, 24h, 7d, 30d, etc.
    
    @NotNull
    @Column(name = "prediction_value", nullable = false, precision = 20, scale = 8)
    private BigDecimal predictionValue;
    
    @NotNull
    @DecimalMin(value = "0.0", message = "Confidence score must be between 0 and 1")
    @DecimalMax(value = "1.0", message = "Confidence score must be between 0 and 1")
    @Column(name = "confidence_score", nullable = false, precision = 5, scale = 4)
    private BigDecimal confidenceScore;
    
    @Column(name = "prediction_interval_lower", precision = 20, scale = 8)
    private BigDecimal predictionIntervalLower;
    
    @Column(name = "prediction_interval_upper", precision = 20, scale = 8)
    private BigDecimal predictionIntervalUpper;
    
    @Column(name = "feature_importance", columnDefinition = "TEXT")
    private String featureImportance; // JSON string of feature importance
    
    @Column(name = "model_metadata", columnDefinition = "TEXT")
    private String modelMetadata; // JSON string of model parameters
    
    @Column(name = "training_data_size")
    private Long trainingDataSize;
    
    @Column(name = "training_accuracy", precision = 5, scale = 4)
    private BigDecimal trainingAccuracy;
    
    @Column(name = "validation_accuracy", precision = 5, scale = 4)
    private BigDecimal validationAccuracy;
    
    @Column(name = "test_accuracy", precision = 5, scale = 4)
    private BigDecimal testAccuracy;
    
    @Column(name = "rmse", precision = 20, scale = 8)
    private BigDecimal rmse; // Root Mean Square Error
    
    @Column(name = "mae", precision = 20, scale = 8)
    private BigDecimal mae; // Mean Absolute Error
    
    @Column(name = "mape", precision = 10, scale = 4)
    private BigDecimal mape; // Mean Absolute Percentage Error
    
    @Column(name = "r_squared", precision = 5, scale = 4)
    private BigDecimal rSquared;
    
    @Column(name = "sharpe_ratio", precision = 10, scale = 4)
    private BigDecimal sharpeRatio;
    
    @Column(name = "max_drawdown", precision = 10, scale = 4)
    private BigDecimal maxDrawdown;
    
    @Column(name = "volatility", precision = 10, scale = 4)
    private BigDecimal volatility;
    
    @Column(name = "skewness", precision = 10, scale = 4)
    private BigDecimal skewness;
    
    @Column(name = "kurtosis", precision = 10, scale = 4)
    private BigDecimal kurtosis;
    
    @Column(name = "var_95", precision = 20, scale = 8)
    private BigDecimal var95; // Value at Risk 95%
    
    @Column(name = "var_99", precision = 20, scale = 8)
    private BigDecimal var99; // Value at Risk 99%
    
    @Column(name = "expected_shortfall", precision = 20, scale = 8)
    private BigDecimal expectedShortfall;
    
    @Column(name = "calmar_ratio", precision = 10, scale = 4)
    private BigDecimal calmarRatio;
    
    @Column(name = "sortino_ratio", precision = 10, scale = 4)
    private BigDecimal sortinoRatio;
    
    @Column(name = "information_ratio", precision = 10, scale = 4)
    private BigDecimal informationRatio;
    
    @Column(name = "treynor_ratio", precision = 10, scale = 4)
    private BigDecimal treynorRatio;
    
    @Column(name = "jensen_alpha", precision = 10, scale = 4)
    private BigDecimal jensenAlpha;
    
    @Column(name = "tracking_error", precision = 10, scale = 4)
    private BigDecimal trackingError;
    
    @Column(name = "beta", precision = 10, scale = 4)
    private BigDecimal beta;
    
    @Column(name = "correlation_market", precision = 5, scale = 4)
    private BigDecimal correlationMarket;
    
    @Column(name = "correlation_btc", precision = 5, scale = 4)
    private BigDecimal correlationBtc;
    
    @Column(name = "correlation_eth", precision = 5, scale = 4)
    private BigDecimal correlationEth;
    
    @Column(name = "sentiment_score", precision = 5, scale = 4)
    private BigDecimal sentimentScore;
    
    @Column(name = "technical_score", precision = 5, scale = 4)
    private BigDecimal technicalScore;
    
    @Column(name = "fundamental_score", precision = 5, scale = 4)
    private BigDecimal fundamentalScore;
    
    @Column(name = "social_score", precision = 5, scale = 4)
    private BigDecimal socialScore;
    
    @Column(name = "overall_score", precision = 5, scale = 4)
    private BigDecimal overallScore;
    
    @Column(name = "risk_score", precision = 5, scale = 4)
    private BigDecimal riskScore;
    
    @Column(name = "investment_recommendation", length = 20)
    private String investmentRecommendation; // BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    
    @Column(name = "target_price", precision = 20, scale = 8)
    private BigDecimal targetPrice;
    
    @Column(name = "stop_loss", precision = 20, scale = 8)
    private BigDecimal stopLoss;
    
    @Column(name = "take_profit", precision = 20, scale = 8)
    private BigDecimal takeProfit;
    
    @Column(name = "position_size", precision = 10, scale = 4)
    private BigDecimal positionSize;
    
    @Column(name = "expected_return", precision = 10, scale = 4)
    private BigDecimal expectedReturn;
    
    @Column(name = "expected_volatility", precision = 10, scale = 4)
    private BigDecimal expectedVolatility;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @Column(name = "expires_at")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime expiresAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (expiresAt == null) {
            // Set expiration based on prediction horizon
            expiresAt = calculateExpirationTime();
        }
    }
    
    /**
     * Calculate expiration time based on prediction horizon
     */
    private LocalDateTime calculateExpirationTime() {
        LocalDateTime now = LocalDateTime.now();
        
        switch (predictionHorizon.toLowerCase()) {
            case "1h":
            case "1hour":
                return now.plusHours(1);
            case "4h":
            case "4hours":
                return now.plusHours(4);
            case "24h":
            case "1d":
            case "1day":
                return now.plusDays(1);
            case "7d":
            case "7days":
            case "1w":
            case "1week":
                return now.plusWeeks(1);
            case "30d":
            case "30days":
            case "1m":
            case "1month":
                return now.plusMonths(1);
            case "90d":
            case "90days":
            case "3m":
            case "3months":
                return now.plusMonths(3);
            case "1y":
            case "1year":
                return now.plusYears(1);
            default:
                return now.plusHours(24); // Default to 24 hours
        }
    }
    
    /**
     * Check if prediction is still valid
     */
    public boolean isValid() {
        return expiresAt == null || LocalDateTime.now().isBefore(expiresAt);
    }
    
    /**
     * Check if prediction is expired
     */
    public boolean isExpired() {
        return !isValid();
    }
    
    /**
     * Get prediction age in minutes
     */
    public long getAgeInMinutes() {
        if (createdAt == null) {
            return 0;
        }
        return java.time.Duration.between(createdAt, LocalDateTime.now()).toMinutes();
    }
    
    /**
     * Calculate prediction accuracy if actual value is provided
     */
    public BigDecimal calculateAccuracy(BigDecimal actualValue) {
        if (actualValue == null || predictionValue == null) {
            return BigDecimal.ZERO;
        }
        
        BigDecimal error = actualValue.subtract(predictionValue).abs();
        return BigDecimal.ONE.subtract(error.divide(actualValue, 6, BigDecimal.ROUND_HALF_UP));
    }
    
    /**
     * Get confidence level as string
     */
    public String getConfidenceLevel() {
        if (confidenceScore == null) {
            return "UNKNOWN";
        }
        
        if (confidenceScore.compareTo(BigDecimal.valueOf(0.9)) >= 0) {
            return "VERY_HIGH";
        } else if (confidenceScore.compareTo(BigDecimal.valueOf(0.8)) >= 0) {
            return "HIGH";
        } else if (confidenceScore.compareTo(BigDecimal.valueOf(0.7)) >= 0) {
            return "MEDIUM";
        } else if (confidenceScore.compareTo(BigDecimal.valueOf(0.6)) >= 0) {
            return "LOW";
        } else {
            return "VERY_LOW";
        }
    }
    
    /**
     * Check if prediction is within confidence interval
     */
    public boolean isWithinInterval(BigDecimal actualValue) {
        if (actualValue == null || predictionIntervalLower == null || predictionIntervalUpper == null) {
            return false;
        }
        
        return actualValue.compareTo(predictionIntervalLower) >= 0 && 
               actualValue.compareTo(predictionIntervalUpper) <= 0;
    }
}
