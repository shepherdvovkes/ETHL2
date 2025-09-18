package com.defimon.bitcoin.repository;

import com.defimon.bitcoin.model.BitcoinMetrics;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Repository for Bitcoin metrics data
 */
@Repository
public interface BitcoinMetricsRepository extends JpaRepository<BitcoinMetrics, Long> {

    /**
     * Find metrics by type
     */
    List<BitcoinMetrics> findByMetricType(String metricType);

    /**
     * Find metrics within time range
     */
    List<BitcoinMetrics> findByTimestampBetween(LocalDateTime startTime, LocalDateTime endTime);

    /**
     * Find latest metrics by type
     */
    @Query("SELECT m FROM BitcoinMetrics m WHERE m.metricType = :metricType ORDER BY m.timestamp DESC")
    List<BitcoinMetrics> findLatestByMetricType(@Param("metricType") String metricType);

    /**
     * Find latest block count
     */
    @Query("SELECT m FROM BitcoinMetrics m WHERE m.metricType = 'BLOCK_COUNT' ORDER BY m.timestamp DESC")
    List<BitcoinMetrics> findLatestBlockCount();

    /**
     * Find latest difficulty
     */
    @Query("SELECT m FROM BitcoinMetrics m WHERE m.metricType = 'DIFFICULTY' ORDER BY m.timestamp DESC")
    List<BitcoinMetrics> findLatestDifficulty();

    /**
     * Find latest mempool info
     */
    @Query("SELECT m FROM BitcoinMetrics m WHERE m.metricType = 'MEMPOOL_INFO' ORDER BY m.timestamp DESC")
    List<BitcoinMetrics> findLatestMempoolInfo();

    /**
     * Find metrics by type with pagination
     */
    org.springframework.data.domain.Page<BitcoinMetrics> findByMetricType(String metricType, org.springframework.data.domain.Pageable pageable);

    /**
     * Count metrics by type
     */
    long countByMetricType(String metricType);

    /**
     * Delete old metrics (older than specified days)
     */
    @Query("DELETE FROM BitcoinMetrics m WHERE m.timestamp < :cutoffTime")
    void deleteOldMetrics(@Param("cutoffTime") LocalDateTime cutoffTime);
}
