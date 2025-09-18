package com.defimon.bitcoin.repository;

import com.defimon.bitcoin.model.BitcoinBlock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository for Bitcoin block data
 */
@Repository
public interface BitcoinBlockRepository extends JpaRepository<BitcoinBlock, Long> {

    /**
     * Find block by hash
     */
    Optional<BitcoinBlock> findByHash(String hash);

    /**
     * Find block by height
     */
    Optional<BitcoinBlock> findByHeight(Long height);

    /**
     * Check if block exists by hash
     */
    boolean existsByHash(String hash);

    /**
     * Find blocks within height range
     */
    List<BitcoinBlock> findByHeightBetween(Long startHeight, Long endHeight);

    /**
     * Find latest blocks
     */
    @Query("SELECT b FROM BitcoinBlock b ORDER BY b.height DESC")
    List<BitcoinBlock> findLatestBlocks();

    /**
     * Find latest blocks with pagination
     */
    @Query("SELECT b FROM BitcoinBlock b ORDER BY b.height DESC")
    org.springframework.data.domain.Page<BitcoinBlock> findLatestBlocks(org.springframework.data.domain.Pageable pageable);

    /**
     * Find blocks by timestamp range
     */
    @Query("SELECT b FROM BitcoinBlock b WHERE b.timestamp BETWEEN :startTime AND :endTime ORDER BY b.height")
    List<BitcoinBlock> findByTimestampBetween(@Param("startTime") Long startTime, @Param("endTime") Long endTime);

    /**
     * Find highest block height
     */
    @Query("SELECT MAX(b.height) FROM BitcoinBlock b")
    Optional<Long> findMaxHeight();

    /**
     * Find lowest block height
     */
    @Query("SELECT MIN(b.height) FROM BitcoinBlock b")
    Optional<Long> findMinHeight();

    /**
     * Count blocks in height range
     */
    long countByHeightBetween(Long startHeight, Long endHeight);

    /**
     * Find blocks with specific difficulty range
     */
    @Query("SELECT b FROM BitcoinBlock b WHERE b.difficulty BETWEEN :minDifficulty AND :maxDifficulty ORDER BY b.height")
    List<BitcoinBlock> findByDifficultyBetween(@Param("minDifficulty") Double minDifficulty, @Param("maxDifficulty") Double maxDifficulty);
}
