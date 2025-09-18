package com.defimon.collector.client;

import com.defimon.common.model.Asset;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;

/**
 * Blockchain client manager for data collection
 */
@Component
public class BlockchainClientManager {
    
    public String collectOnChainMetrics(Asset asset) {
        // Mock implementation
        return String.format("{\"asset\":\"%s\",\"price\":100.0,\"volume\":1000000}", asset.getSymbol());
    }
    
    public String collectPriceData(Asset asset) {
        // Mock implementation
        return String.format("{\"asset\":\"%s\",\"price\":100.0,\"change24h\":5.2}", asset.getSymbol());
    }
    
    public String collectSocialMetrics(Asset asset) {
        // Mock implementation
        return String.format("{\"asset\":\"%s\",\"sentiment\":0.7,\"mentions\":1000}", asset.getSymbol());
    }
    
    public List<Asset> getActiveAssets() {
        // Mock implementation - return default assets
        return Arrays.asList(
            Asset.builder()
                .symbol("BTC")
                .name("Bitcoin")
                .blockchain("bitcoin")
                .isActive(true)
                .build(),
            Asset.builder()
                .symbol("ETH")
                .name("Ethereum")
                .blockchain("ethereum")
                .isActive(true)
                .build(),
            Asset.builder()
                .symbol("MATIC")
                .name("Polygon")
                .blockchain("polygon")
                .isActive(true)
                .build()
        );
    }
    
    public Asset getAssetBySymbol(String symbol) {
        return getActiveAssets().stream()
            .filter(asset -> asset.getSymbol().equals(symbol))
            .findFirst()
            .orElse(null);
    }
}
