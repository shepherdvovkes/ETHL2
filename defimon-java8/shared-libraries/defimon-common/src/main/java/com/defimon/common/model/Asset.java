package com.defimon.common.model;

import com.defimon.common.enums.RiskLevel;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.PositiveOrZero;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Core Asset model representing cryptocurrency assets tracked by DEFIMON
 */
@Entity
@Table(name = "assets", indexes = {
    @Index(name = "idx_asset_symbol", columnList = "symbol"),
    @Index(name = "idx_asset_blockchain", columnList = "blockchain"),
    @Index(name = "idx_asset_contract", columnList = "contract_address"),
    @Index(name = "idx_asset_active", columnList = "is_active")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class Asset {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @NotBlank(message = "Symbol is required")
    @Column(name = "symbol", nullable = false, unique = true, length = 20)
    private String symbol;
    
    @NotBlank(message = "Name is required")
    @Column(name = "name", nullable = false, length = 100)
    private String name;
    
    @Column(name = "contract_address", length = 42)
    private String contractAddress;
    
    @NotBlank(message = "Blockchain is required")
    @Column(name = "blockchain", nullable = false, length = 50)
    private String blockchain;
    
    @Column(name = "category", length = 50)
    private String category;
    
    @Column(name = "decimals")
    private Integer decimals;
    
    @Column(name = "total_supply", precision = 36, scale = 18)
    private BigDecimal totalSupply;
    
    @Column(name = "circulating_supply", precision = 36, scale = 18)
    private BigDecimal circulatingSupply;
    
    @Column(name = "max_supply", precision = 36, scale = 18)
    private BigDecimal maxSupply;
    
    @Column(name = "github_repo", length = 200)
    private String githubRepo;
    
    @Column(name = "website", length = 200)
    private String website;
    
    @Column(name = "description", columnDefinition = "TEXT")
    private String description;
    
    @Column(name = "logo_url", length = 500)
    private String logoUrl;
    
    @Column(name = "is_active")
    @Builder.Default
    private Boolean isActive = true;
    
    @Column(name = "is_verified")
    @Builder.Default
    private Boolean isVerified = false;
    
    @Column(name = "security_score")
    @PositiveOrZero
    private BigDecimal securityScore;
    
    @Column(name = "risk_level")
    @Enumerated(EnumType.STRING)
    private RiskLevel riskLevel;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime updatedAt;
    
    @Column(name = "last_data_collection")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime lastDataCollection;
    
    // Relationships
    @OneToMany(mappedBy = "asset", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<OnChainMetrics> onChainMetrics;
    
    @OneToMany(mappedBy = "asset", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<FinancialMetrics> financialMetrics;
    
    @OneToMany(mappedBy = "asset", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<MLPrediction> mlPredictions;
    
    @OneToMany(mappedBy = "asset", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<SocialMetrics> socialMetrics;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
    
    /**
     * Update last data collection timestamp
     */
    public void updateDataCollectionTime() {
        this.lastDataCollection = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * Check if asset needs data collection based on last collection time
     */
    public boolean needsDataCollection(int intervalMinutes) {
        if (lastDataCollection == null) {
            return true;
        }
        return lastDataCollection.isBefore(LocalDateTime.now().minusMinutes(intervalMinutes));
    }
    
    /**
     * Get display name for the asset
     */
    public String getDisplayName() {
        return String.format("%s (%s)", name, symbol);
    }
    
    /**
     * Check if asset is a token (has contract address)
     */
    public boolean isToken() {
        return contractAddress != null && !contractAddress.isEmpty();
    }
    
    /**
     * Check if asset is native to its blockchain
     */
    public boolean isNative() {
        return !isToken();
    }
}
