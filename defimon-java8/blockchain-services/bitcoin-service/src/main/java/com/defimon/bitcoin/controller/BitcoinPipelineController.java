package com.defimon.bitcoin.controller;

import com.defimon.bitcoin.model.BitcoinBlock;
import com.defimon.bitcoin.model.BitcoinMetrics;
import com.defimon.bitcoin.model.CollectionStats;
import com.defimon.bitcoin.repository.BitcoinBlockRepository;
import com.defimon.bitcoin.repository.BitcoinMetricsRepository;
import com.defimon.bitcoin.service.BitcoinDataPipelineService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Bitcoin Data Pipeline Controller
 * Provides endpoints for data collection management and statistics
 */
@RestController
@RequestMapping("/api/bitcoin/pipeline")
public class BitcoinPipelineController {

    private static final Logger logger = LoggerFactory.getLogger(BitcoinPipelineController.class);

    @Autowired
    private BitcoinDataPipelineService pipelineService;

    @Autowired
    private BitcoinMetricsRepository metricsRepository;

    @Autowired
    private BitcoinBlockRepository blockRepository;

    /**
     * Get collection statistics
     */
    @GetMapping("/stats")
    public CompletableFuture<ResponseEntity<CollectionStats>> getCollectionStats() {
        return pipelineService.getCollectionStats()
            .thenApply(stats -> ResponseEntity.ok(stats))
            .exceptionally(throwable -> {
                logger.error("Error getting collection stats: {}", throwable.getMessage());
                return ResponseEntity.status(500).body(null);
            });
    }

    /**
     * Start data collection manually
     */
    @PostMapping("/start")
    public ResponseEntity<Map<String, Object>> startCollection() {
        try {
            pipelineService.startCollection();
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Data collection started");
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error starting collection: {}", e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to start collection: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }

    /**
     * Get collection status
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getCollectionStatus() {
        Map<String, Object> response = new HashMap<>();
        response.put("isCollecting", pipelineService.isCollecting());
        response.put("status", pipelineService.isCollecting() ? "COLLECTING" : "IDLE");
        response.put("timestamp", System.currentTimeMillis());
        return ResponseEntity.ok(response);
    }

    /**
     * Get latest metrics
     */
    @GetMapping("/metrics/latest")
    public ResponseEntity<Map<String, Object>> getLatestMetrics() {
        try {
            Map<String, Object> response = new HashMap<>();
            
            // Get latest block count
            List<BitcoinMetrics> blockCountMetrics = metricsRepository.findLatestBlockCount();
            if (!blockCountMetrics.isEmpty()) {
                response.put("latestBlockCount", blockCountMetrics.get(0).getBlockCount());
            }
            
            // Get latest difficulty
            List<BitcoinMetrics> difficultyMetrics = metricsRepository.findLatestDifficulty();
            if (!difficultyMetrics.isEmpty()) {
                response.put("latestDifficulty", difficultyMetrics.get(0).getDifficulty());
            }
            
            // Get latest mempool info
            List<BitcoinMetrics> mempoolMetrics = metricsRepository.findLatestMempoolInfo();
            if (!mempoolMetrics.isEmpty()) {
                BitcoinMetrics mempool = mempoolMetrics.get(0);
                response.put("latestMempoolSize", mempool.getMempoolSize());
                response.put("latestMempoolBytes", mempool.getMempoolBytes());
            }
            
            response.put("success", true);
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error getting latest metrics: {}", e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get latest metrics: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }

    /**
     * Get metrics by type
     */
    @GetMapping("/metrics/{metricType}")
    public ResponseEntity<Map<String, Object>> getMetricsByType(
            @PathVariable String metricType,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        try {
            Pageable pageable = PageRequest.of(page, size);
            Page<BitcoinMetrics> metricsPage = metricsRepository.findByMetricType(metricType, pageable);
            
            Map<String, Object> response = new HashMap<>();
            response.put("metrics", metricsPage.getContent());
            response.put("totalElements", metricsPage.getTotalElements());
            response.put("totalPages", metricsPage.getTotalPages());
            response.put("currentPage", page);
            response.put("pageSize", size);
            response.put("success", true);
            response.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error getting metrics by type {}: {}", metricType, e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get metrics: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }

    /**
     * Get latest blocks
     */
    @GetMapping("/blocks/latest")
    public ResponseEntity<Map<String, Object>> getLatestBlocks(
            @RequestParam(defaultValue = "10") int limit) {
        try {
            Pageable pageable = PageRequest.of(0, limit);
            Page<BitcoinBlock> blocksPage = blockRepository.findLatestBlocks(pageable);
            
            Map<String, Object> response = new HashMap<>();
            response.put("blocks", blocksPage.getContent());
            response.put("totalElements", blocksPage.getTotalElements());
            response.put("limit", limit);
            response.put("success", true);
            response.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error getting latest blocks: {}", e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get latest blocks: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }

    /**
     * Get block by height
     */
    @GetMapping("/blocks/height/{height}")
    public ResponseEntity<Map<String, Object>> getBlockByHeight(@PathVariable Long height) {
        try {
            BitcoinBlock block = blockRepository.findByHeight(height).orElse(null);
            
            Map<String, Object> response = new HashMap<>();
            response.put("block", block);
            response.put("success", block != null);
            response.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error getting block by height {}: {}", height, e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get block: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }

    /**
     * Get data collection summary
     */
    @GetMapping("/summary")
    public ResponseEntity<Map<String, Object>> getDataSummary() {
        try {
            Map<String, Object> response = new HashMap<>();
            
            // Get counts
            long totalMetrics = metricsRepository.count();
            long totalBlocks = blockRepository.count();
            
            // Get latest block height
            Long maxHeight = blockRepository.findMaxHeight().orElse(0L);
            Long minHeight = blockRepository.findMinHeight().orElse(0L);
            
            response.put("totalMetrics", totalMetrics);
            response.put("totalBlocks", totalBlocks);
            response.put("maxBlockHeight", maxHeight);
            response.put("minBlockHeight", minHeight);
            response.put("blockRange", maxHeight - minHeight + 1);
            response.put("isCollecting", pipelineService.isCollecting());
            response.put("success", true);
            response.put("timestamp", System.currentTimeMillis());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error getting data summary: {}", e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get data summary: " + e.getMessage());
            response.put("timestamp", System.currentTimeMillis());
            return ResponseEntity.status(500).body(response);
        }
    }
}
