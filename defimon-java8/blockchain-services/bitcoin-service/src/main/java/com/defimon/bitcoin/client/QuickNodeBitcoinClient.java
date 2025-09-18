package com.defimon.bitcoin.client;

import com.defimon.bitcoin.model.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
// import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
// import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

import java.math.BigDecimal;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * QuickNode Bitcoin API client with real RPC calls
 */
@Component
public class QuickNodeBitcoinClient {
    
    private static final Logger logger = LoggerFactory.getLogger(QuickNodeBitcoinClient.class);
    
    @Value("${bitcoin.quicknode.rpc.url}")
    private String rpcUrl;
    
    @Value("${bitcoin.quicknode.rpc.user}")
    private String rpcUser;
    
    @Value("${bitcoin.quicknode.rpc.password}")
    private String rpcPassword;
    
    private final WebClient webClient;
    private final ObjectMapper objectMapper;
    
    public QuickNodeBitcoinClient() {
        this.webClient = WebClient.builder()
            .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
            .build();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Make authenticated RPC call to Bitcoin QuickNode
     */
    private Mono<JsonNode> makeRpcCall(String method, Object... params) {
        try {
            Map<String, Object> request = new HashMap<>();
            request.put("jsonrpc", "1.0");
            request.put("id", "defimon-bitcoin-client");
            request.put("method", method);
            request.put("params", params);
            
            String auth = rpcUser + ":" + rpcPassword;
            String encodedAuth = Base64.getEncoder().encodeToString(auth.getBytes());
            
            return webClient.post()
                .uri(rpcUrl)
                .header(HttpHeaders.AUTHORIZATION, "Basic " + encodedAuth)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(JsonNode.class)
                .doOnError(error -> logger.error("RPC call failed for method {}: {}", method, error.getMessage()))
                .onErrorResume(WebClientResponseException.class, ex -> {
                    logger.error("HTTP error {} for method {}: {}", ex.getStatusCode(), method, ex.getResponseBodyAsString());
                    return Mono.empty();
                });
        } catch (Exception e) {
            logger.error("Error making RPC call for method {}: {}", method, e.getMessage());
            return Mono.empty();
        }
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<Long> getBlockCount() {
        return makeRpcCall("getblockcount")
            .map(response -> {
                if (response.has("result")) {
                    return response.get("result").asLong();
                }
                return 0L;
            })
            .doOnSuccess(count -> logger.info("Current block count: {}", count))
            .doOnError(error -> logger.error("Failed to get block count: {}", error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<String> getBlockHash(long blockHeight) {
        return makeRpcCall("getblockhash", blockHeight)
            .map(response -> {
                if (response.has("result")) {
                    return response.get("result").asText();
                }
                return "";
            })
            .doOnSuccess(hash -> logger.info("Block hash for height {}: {}", blockHeight, hash))
            .doOnError(error -> logger.error("Failed to get block hash for height {}: {}", blockHeight, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<BitcoinBlock> getBlock(String blockHash) {
        return makeRpcCall("getblock", blockHash, 2) // verbosity level 2 for full block data
            .map(response -> {
                if (response.has("result")) {
                    JsonNode blockData = response.get("result");
                    return BitcoinBlock.builder()
                        .hash(blockData.get("hash").asText())
                        .height(blockData.get("height").asLong())
                        .timestamp(blockData.get("time").asLong())
                        .size(blockData.get("size").asLong())
                        .weight(blockData.get("weight").asLong())
                        .version(blockData.get("version").asInt())
                        .merkleRoot(blockData.get("merkleroot").asText())
                        .nonce(blockData.get("nonce").asLong())
                        .bits(blockData.get("bits").asText())
                        .difficulty(blockData.get("difficulty").asDouble())
                        .chainwork(blockData.get("chainwork").asText())
                        .previousBlockHash(blockData.get("previousblockhash").asText())
                        .nextBlockHash(blockData.has("nextblockhash") ? blockData.get("nextblockhash").asText() : null)
                        .build();
                }
                return null;
            })
            .doOnSuccess(block -> {
                if (block != null) {
                    logger.info("Retrieved block: {} at height {}", block.getHash(), block.getHeight());
                }
            })
            .doOnError(error -> logger.error("Failed to get block {}: {}", blockHash, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<BigDecimal> getDifficulty() {
        return makeRpcCall("getdifficulty")
            .map(response -> {
                if (response.has("result")) {
                    return new BigDecimal(response.get("result").asText());
                }
                return BigDecimal.ZERO;
            })
            .doOnSuccess(difficulty -> logger.info("Current difficulty: {}", difficulty))
            .doOnError(error -> logger.error("Failed to get difficulty: {}", error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<MempoolInfo> getMempoolInfo() {
        return makeRpcCall("getmempoolinfo")
            .map(response -> {
                if (response.has("result")) {
                    JsonNode mempoolData = response.get("result");
                    return MempoolInfo.builder()
                        .size(mempoolData.get("size").asLong())
                        .bytes(mempoolData.get("bytes").asLong())
                        .usage(mempoolData.get("usage").asLong())
                        .maxMempool(mempoolData.get("maxmempool").asLong())
                        .mempoolMinFee(mempoolData.get("mempoolminfee").asDouble())
                        .minRelayTxFee(mempoolData.get("minrelaytxfee").asDouble())
                        .build();
                }
                return null;
            })
            .doOnSuccess(info -> {
                if (info != null) {
                    logger.info("Mempool info - Size: {}, Bytes: {}", info.getSize(), info.getBytes());
                }
            })
            .doOnError(error -> logger.error("Failed to get mempool info: {}", error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<BigDecimal> estimateSmartFee(int blocks) {
        return makeRpcCall("estimatesmartfee", blocks)
            .map(response -> {
                if (response.has("result")) {
                    JsonNode feeData = response.get("result");
                    if (feeData.has("feerate")) {
                        return new BigDecimal(feeData.get("feerate").asText());
                    }
                }
                return BigDecimal.ZERO;
            })
            .doOnSuccess(fee -> logger.info("Estimated smart fee for {} blocks: {}", blocks, fee))
            .doOnError(error -> logger.error("Failed to estimate smart fee for {} blocks: {}", blocks, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<String> getRawTransaction(String txHash) {
        return makeRpcCall("getrawtransaction", txHash)
            .map(response -> {
                if (response.has("result")) {
                    return response.get("result").asText();
                }
                return "";
            })
            .doOnSuccess(rawTx -> logger.info("Retrieved raw transaction: {} ({} chars)", txHash, rawTx.length()))
            .doOnError(error -> logger.error("Failed to get raw transaction {}: {}", txHash, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<BitcoinTransaction> decodeRawTransaction(String rawTx) {
        return makeRpcCall("decoderawtransaction", rawTx)
            .map(response -> {
                if (response.has("result")) {
                    JsonNode txData = response.get("result");
                    return BitcoinTransaction.builder()
                        .hash(txData.get("txid").asText())
                        .version(txData.get("version").asInt())
                        .lockTime(txData.get("locktime").asLong())
                        .size(txData.get("size").asLong())
                        .vSize(txData.get("vsize").asLong())
                        .weight(txData.get("weight").asLong())
                        .build();
                }
                return null;
            })
            .doOnSuccess(tx -> {
                if (tx != null) {
                    logger.info("Decoded transaction: {}", tx.getHash());
                }
            })
            .doOnError(error -> logger.error("Failed to decode raw transaction: {}", error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-transaction")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<Integer> getTransactionConfirmations(String txHash) {
        return makeRpcCall("gettransaction", txHash)
            .map(response -> {
                if (response.has("result")) {
                    JsonNode txData = response.get("result");
                    return txData.get("confirmations").asInt();
                }
                return 0;
            })
            .doOnSuccess(confirmations -> logger.info("Transaction {} has {} confirmations", txHash, confirmations))
            .doOnError(error -> logger.error("Failed to get confirmations for transaction {}: {}", txHash, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<BigDecimal> getAddressBalance(String address) {
        return makeRpcCall("getreceivedbyaddress", address, 0) // 0 confirmations
            .map(response -> {
                if (response.has("result")) {
                    return new BigDecimal(response.get("result").asText());
                }
                return BigDecimal.ZERO;
            })
            .doOnSuccess(balance -> logger.info("Address {} balance: {}", address, balance))
            .doOnError(error -> logger.error("Failed to get balance for address {}: {}", address, error.getMessage()));
    }
    
    // @CircuitBreaker(name = "bitcoin-quicknode")
    // @RateLimiter(name = "bitcoin-api")
    public Mono<NetworkInfo> getNetworkInfo() {
        return makeRpcCall("getnetworkinfo")
            .map(response -> {
                if (response.has("result")) {
                    JsonNode networkData = response.get("result");
                    return NetworkInfo.builder()
                        .version(networkData.get("version").asInt())
                        .subversion(networkData.get("subversion").asText())
                        .protocolVersion(networkData.get("protocolversion").asInt())
                        .localServices(networkData.get("localservices").asText())
                        .localRelay(networkData.get("localrelay").asBoolean())
                        .timeOffset(networkData.get("timeoffset").asLong())
                        .networkActive(networkData.get("networkactive").asBoolean())
                        .connections(networkData.get("connections").asInt())
                        .build();
                }
                return null;
            })
            .doOnSuccess(info -> {
                if (info != null) {
                    logger.info("Network info - Version: {}, Connections: {}", info.getVersion(), info.getConnections());
                }
            })
            .doOnError(error -> logger.error("Failed to get network info: {}", error.getMessage()));
    }
}
