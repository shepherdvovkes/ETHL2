package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.math.BigDecimal;

/**
 * Bitcoin block representation
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "bitcoin_blocks")
public class BitcoinBlock {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(unique = true, nullable = false)
    private String hash;
    
    @Column(unique = true, nullable = false)
    private Long height;
    
    private Long timestamp;
    private Long size;
    private Long weight;
    private Integer version;
    private String merkleRoot;
    private Long nonce;
    private String bits;
    private Double difficulty;
    private String chainwork;
    private String previousBlockHash;
    private String nextBlockHash;
}
