package com.defimon.bitcoin.controller;

import com.defimon.bitcoin.client.QuickNodeBitcoinClient;
import com.defimon.bitcoin.model.*;
// import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
// import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

/**
 * REST Controller for Bitcoin data collection and API endpoints
 */
@RestController
@RequestMapping("/api/bitcoin")
@CrossOrigin(origins = "*")
public class BitcoinDataController {
    
    private static final Logger logger = LoggerFactory.getLogger(BitcoinDataController.class);
    
    @Autowired
    private QuickNodeBitcoinClient bitcoinClient;
    
    /**
     * Get current Bitcoin block count
     */
    @GetMapping("/block-count")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getBlockCount() {
        return bitcoinClient.getBlockCount()
            .map(blockCount -> {
                Map<String, Object> response = new HashMap<>();
                response.put("blockCount", blockCount);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get block count")));
    }
    
    /**
     * Get Bitcoin block by height
     */
    @GetMapping("/block/{height}")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getBlockByHeight(@PathVariable long height) {
        return bitcoinClient.getBlockHash(height)
            .flatMap(blockHash -> bitcoinClient.getBlock(blockHash))
            .map(block -> {
                Map<String, Object> response = new HashMap<>();
                response.put("block", block);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get block at height " + height)));
    }
    
    /**
     * Get Bitcoin block by hash
     */
    @GetMapping("/block/hash/{hash}")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getBlockByHash(@PathVariable String hash) {
        return bitcoinClient.getBlock(hash)
            .map(block -> {
                Map<String, Object> response = new HashMap<>();
                response.put("block", block);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get block with hash " + hash)));
    }
    
    /**
     * Get Bitcoin network information
     */
    @GetMapping("/network-info")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getNetworkInfo() {
        return bitcoinClient.getNetworkInfo()
            .map(networkInfo -> {
                Map<String, Object> response = new HashMap<>();
                response.put("networkInfo", networkInfo);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get network info")));
    }
    
    /**
     * Get Bitcoin mempool information
     */
    @GetMapping("/mempool-info")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getMempoolInfo() {
        return bitcoinClient.getMempoolInfo()
            .map(mempoolInfo -> {
                Map<String, Object> response = new HashMap<>();
                response.put("mempoolInfo", mempoolInfo);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get mempool info")));
    }
    
    /**
     * Get Bitcoin difficulty
     */
    @GetMapping("/difficulty")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getDifficulty() {
        return bitcoinClient.getDifficulty()
            .map(difficulty -> {
                Map<String, Object> response = new HashMap<>();
                response.put("difficulty", difficulty);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get difficulty")));
    }
    
    /**
     * Get Bitcoin fee estimates
     */
    @GetMapping("/fee-estimates")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getFeeEstimates() {
        return Mono.zip(
            bitcoinClient.estimateSmartFee(1),
            bitcoinClient.estimateSmartFee(6),
            bitcoinClient.estimateSmartFee(12)
        ).map(tuple -> {
            Map<String, Object> response = new HashMap<>();
            response.put("feeEstimate1Block", tuple.getT1());
            response.put("feeEstimate6Blocks", tuple.getT2());
            response.put("feeEstimate12Blocks", tuple.getT3());
            response.put("timestamp", System.currentTimeMillis());
            response.put("success", true);
            return ResponseEntity.ok(response);
        }).onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get fee estimates")));
    }
    
    /**
     * Get Bitcoin transaction by hash
     */
    @GetMapping("/transaction/{txHash}")
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getTransaction(@PathVariable String txHash) {
        return bitcoinClient.getRawTransaction(txHash)
            .flatMap(rawTx -> bitcoinClient.decodeRawTransaction(rawTx))
            .map(transaction -> {
                Map<String, Object> response = new HashMap<>();
                response.put("transaction", transaction);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get transaction " + txHash)));
    }
    
    /**
     * Get Bitcoin transaction confirmations
     */
    @GetMapping("/transaction/{txHash}/confirmations")
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getTransactionConfirmations(@PathVariable String txHash) {
        return bitcoinClient.getTransactionConfirmations(txHash)
            .map(confirmations -> {
                Map<String, Object> response = new HashMap<>();
                response.put("txHash", txHash);
                response.put("confirmations", confirmations);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get confirmations for transaction " + txHash)));
    }
    
    /**
     * Get Bitcoin address balance
     */
    @GetMapping("/address/{address}/balance")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getAddressBalance(@PathVariable String address) {
        return bitcoinClient.getAddressBalance(address)
            .map(balance -> {
                Map<String, Object> response = new HashMap<>();
                response.put("address", address);
                response.put("balance", balance);
                response.put("timestamp", System.currentTimeMillis());
                response.put("success", true);
                return ResponseEntity.ok(response);
            })
            .onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get balance for address " + address)));
    }
    
    /**
     * Get comprehensive Bitcoin network status
     */
    @GetMapping("/status")
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<ResponseEntity<Map<String, Object>>> getBitcoinStatus() {
        return Mono.zip(
            bitcoinClient.getBlockCount(),
            bitcoinClient.getNetworkInfo(),
            bitcoinClient.getMempoolInfo(),
            bitcoinClient.getDifficulty(),
            bitcoinClient.estimateSmartFee(6)
        ).map(tuple -> {
            Map<String, Object> response = new HashMap<>();
            response.put("blockCount", tuple.getT1());
            response.put("networkInfo", tuple.getT2());
            response.put("mempoolInfo", tuple.getT3());
            response.put("difficulty", tuple.getT4());
            response.put("feeEstimate6Blocks", tuple.getT5());
            response.put("timestamp", System.currentTimeMillis());
            response.put("success", true);
            return ResponseEntity.ok(response);
        }).onErrorReturn(ResponseEntity.status(500).body(createErrorResponse("Failed to get Bitcoin status")));
    }
    
    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "UP");
        response.put("service", "bitcoin-service");
        response.put("timestamp", System.currentTimeMillis());
        return ResponseEntity.ok(response);
    }
    
    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> response = new HashMap<>();
        response.put("error", message);
        response.put("timestamp", System.currentTimeMillis());
        response.put("success", false);
        return response;
    }
}
