package com.defimon.bitcoin.repository;

import com.defimon.bitcoin.model.BitcoinTransaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository for Bitcoin transaction data
 */
@Repository
public interface BitcoinTransactionRepository extends JpaRepository<BitcoinTransaction, Long> {

    /**
     * Find transaction by hash
     */
    Optional<BitcoinTransaction> findByHash(String hash);

    /**
     * Check if transaction exists by hash
     */
    boolean existsByHash(String hash);

    /**
     * Find transactions by version
     */
    List<BitcoinTransaction> findByVersion(Integer version);

    /**
     * Find transactions by size range
     */
    List<BitcoinTransaction> findBySizeBetween(Long minSize, Long maxSize);

    /**
     * Find transactions by weight range
     */
    List<BitcoinTransaction> findByWeightBetween(Long minWeight, Long maxWeight);

    /**
     * Find latest transactions
     */
    @Query("SELECT t FROM BitcoinTransaction t ORDER BY t.id DESC")
    List<BitcoinTransaction> findLatestTransactions();

    /**
     * Count transactions by version
     */
    long countByVersion(Integer version);

    /**
     * Find transactions with specific size
     */
    @Query("SELECT t FROM BitcoinTransaction t WHERE t.size = :size ORDER BY t.id DESC")
    List<BitcoinTransaction> findByExactSize(@Param("size") Long size);

    /**
     * Find transactions with specific weight
     */
    @Query("SELECT t FROM BitcoinTransaction t WHERE t.weight = :weight ORDER BY t.id DESC")
    List<BitcoinTransaction> findByExactWeight(@Param("weight") Long weight);
}
