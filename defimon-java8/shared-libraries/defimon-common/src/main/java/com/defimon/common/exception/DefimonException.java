package com.defimon.common.exception;

import lombok.Getter;

/**
 * Base exception for DEFIMON platform
 */
@Getter
public class DefimonException extends RuntimeException {
    
    private final String errorCode;
    private final String errorMessage;
    private final Object[] args;
    
    public DefimonException(String errorCode, String errorMessage) {
        super(errorMessage);
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.args = new Object[0];
    }
    
    public DefimonException(String errorCode, String errorMessage, Object... args) {
        super(String.format(errorMessage, args));
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.args = args;
    }
    
    public DefimonException(String errorCode, String errorMessage, Throwable cause) {
        super(errorMessage, cause);
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.args = new Object[0];
    }
    
    public DefimonException(String errorCode, String errorMessage, Throwable cause, Object... args) {
        super(String.format(errorMessage, args), cause);
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.args = args;
    }
}
