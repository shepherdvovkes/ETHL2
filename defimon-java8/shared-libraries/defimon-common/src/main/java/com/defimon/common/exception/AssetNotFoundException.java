package com.defimon.common.exception;

/**
 * Exception thrown when an asset is not found
 */
public class AssetNotFoundException extends DefimonException {
    
    public AssetNotFoundException(String symbol) {
        super("ASSET_NOT_FOUND", "Asset with symbol '%s' not found", symbol);
    }
    
    public AssetNotFoundException(Long assetId) {
        super("ASSET_NOT_FOUND", "Asset with ID '%d' not found", assetId);
    }
}
