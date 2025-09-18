package com.defimon.collector.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Data collector configuration
 */
@Data
@Component
@ConfigurationProperties(prefix = "defimon.collector")
public class CollectorConfig {
    
    private int maxConcurrentCollections = 10;
    private int interval = 60000; // 1 minute
    private int batchSize = 100;
    private int retryAttempts = 3;
    private int timeoutSeconds = 30;
    
    // Rate limiting
    private int rateLimitPerSecond = 10;
    private int rateLimitBurst = 20;
    
    // Circuit breaker
    private int circuitBreakerFailureThreshold = 5;
    private int circuitBreakerTimeout = 30000;
    private int circuitBreakerRetryTimeout = 60000;
}
