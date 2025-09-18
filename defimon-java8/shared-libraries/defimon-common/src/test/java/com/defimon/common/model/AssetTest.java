package com.defimon.common.model;

import com.defimon.common.enums.RiskLevel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Unit tests for Asset model
 */
class AssetTest {
    
    private Asset asset;
    
    @BeforeEach
    void setUp() {
        asset = Asset.builder()
            .symbol("BTC")
            .name("Bitcoin")
            .contractAddress(null)
            .blockchain("bitcoin")
            .category("cryptocurrency")
            .decimals(8)
            .totalSupply(new BigDecimal("21000000"))
            .circulatingSupply(new BigDecimal("19500000"))
            .maxSupply(new BigDecimal("21000000"))
            .githubRepo("https://github.com/bitcoin/bitcoin")
            .website("https://bitcoin.org")
            .description("Digital currency")
            .logoUrl("https://bitcoin.org/logo.png")
            .isActive(true)
            .isVerified(true)
            .securityScore(new BigDecimal("9.5"))
            .riskLevel(RiskLevel.LOW)
            .createdAt(LocalDateTime.now())
            .updatedAt(LocalDateTime.now())
            .build();
    }
    
    @Test
    void testAssetCreation() {
        assertNotNull(asset);
        assertEquals("BTC", asset.getSymbol());
        assertEquals("Bitcoin", asset.getSymbol());
        assertEquals("bitcoin", asset.getBlockchain());
        assertTrue(asset.getIsActive());
        assertTrue(asset.getIsVerified());
    }
    
    @Test
    void testIsToken() {
        // Bitcoin is not a token (no contract address)
        assertFalse(asset.isToken());
        
        // Create a token asset
        Asset tokenAsset = Asset.builder()
            .symbol("USDC")
            .name("USD Coin")
            .contractAddress("0xA0b86a33E6441b8c4C8C0C4C0C4C0C4C0C4C0C4C")
            .blockchain("ethereum")
            .build();
        
        assertTrue(tokenAsset.isToken());
    }
    
    @Test
    void testIsNative() {
        // Bitcoin is native to its blockchain
        assertTrue(asset.isNative());
        
        // Create a token asset
        Asset tokenAsset = Asset.builder()
            .symbol("USDC")
            .name("USD Coin")
            .contractAddress("0xA0b86a33E6441b8c4C8C0C4C0C4C0C4C0C4C0C4C")
            .blockchain("ethereum")
            .build();
        
        assertFalse(tokenAsset.isNative());
    }
    
    @Test
    void testGetDisplayName() {
        assertEquals("Bitcoin (BTC)", asset.getDisplayName());
    }
    
    @Test
    void testNeedsDataCollection() {
        // Asset with no last collection time should need collection
        assertTrue(asset.needsDataCollection(60));
        
        // Update last collection time to recent
        asset.setLastDataCollection(LocalDateTime.now().minusMinutes(30));
        assertFalse(asset.needsDataCollection(60));
        
        // Update last collection time to old
        asset.setLastDataCollection(LocalDateTime.now().minusMinutes(90));
        assertTrue(asset.needsDataCollection(60));
    }
    
    @Test
    void testUpdateDataCollectionTime() {
        LocalDateTime before = asset.getUpdatedAt();
        
        asset.updateDataCollectionTime();
        
        assertNotNull(asset.getLastDataCollection());
        assertTrue(asset.getUpdatedAt().isAfter(before));
    }
    
    @Test
    void testPrePersist() {
        Asset newAsset = Asset.builder()
            .symbol("ETH")
            .name("Ethereum")
            .blockchain("ethereum")
            .build();
        
        // Simulate @PrePersist
        newAsset.onCreate();
        
        assertNotNull(newAsset.getCreatedAt());
        assertNotNull(newAsset.getUpdatedAt());
        assertEquals(newAsset.getCreatedAt(), newAsset.getUpdatedAt());
    }
    
    @Test
    void testPreUpdate() {
        LocalDateTime originalCreatedAt = asset.getCreatedAt();
        LocalDateTime originalUpdatedAt = asset.getUpdatedAt();
        
        // Simulate @PreUpdate
        asset.onUpdate();
        
        assertEquals(originalCreatedAt, asset.getCreatedAt());
        assertTrue(asset.getUpdatedAt().isAfter(originalUpdatedAt));
    }
}
