package com.defimon.common.exception;

/**
 * Exception thrown when blockchain operations fail
 */
public class BlockchainException extends DefimonException {
    
    public BlockchainException(String blockchain, String operation, String message) {
        super("BLOCKCHAIN_ERROR", "Blockchain '%s' operation '%s' failed: %s", 
              blockchain, operation, message);
    }
    
    public BlockchainException(String blockchain, String operation, Throwable cause) {
        super("BLOCKCHAIN_ERROR", "Blockchain '%s' operation '%s' failed", 
              cause, blockchain, operation);
    }
    
    public BlockchainException(String message) {
        super("BLOCKCHAIN_ERROR", message);
    }
    
    public BlockchainException(String message, Throwable cause) {
        super("BLOCKCHAIN_ERROR", message, cause);
    }
}
