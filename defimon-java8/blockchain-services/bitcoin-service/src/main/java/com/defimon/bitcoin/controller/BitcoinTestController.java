package com.defimon.bitcoin.controller;

import com.defimon.bitcoin.service.BitcoinDataCollectorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Simple REST Controller for testing Bitcoin data collection
 */
@RestController
@RequestMapping("/api/bitcoin/test")
@CrossOrigin(origins = "*")
public class BitcoinTestController {
    
    private static final Logger logger = LoggerFactory.getLogger(BitcoinTestController.class);
    
    @Autowired
    private BitcoinDataCollectorService bitcoinDataCollectorService;
    
    /**
     * Test endpoint to trigger data collection
     */
    @GetMapping("/collect")
    public ResponseEntity<Map<String, Object>> collectData() {
        try {
            logger.info("Manual data collection triggered via REST API");
            
            CompletableFuture<Map<String, Object>> future = bitcoinDataCollectorService.collectDataNow();
            Map<String, Object> result = future.get(); // Wait for completion
            
            return ResponseEntity.ok(result);
            
        } catch (Exception e) {
            logger.error("Error in manual data collection", e);
            
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", e.getMessage());
            errorResponse.put("successful", false);
            errorResponse.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.status(500).body(errorResponse);
        }
    }
    
    /**
     * Get service status
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        try {
            Map<String, Object> status = bitcoinDataCollectorService.getServiceStatus();
            return ResponseEntity.ok(status);
            
        } catch (Exception e) {
            logger.error("Error getting service status", e);
            
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", e.getMessage());
            errorResponse.put("successful", false);
            errorResponse.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.status(500).body(errorResponse);
        }
    }
    
    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "UP");
        response.put("service", "bitcoin-data-collector");
        response.put("timestamp", System.currentTimeMillis());
        return ResponseEntity.ok(response);
    }
}
