package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.util.List;

/**
 * Bitcoin transaction representation
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "bitcoin_transactions")
public class BitcoinTransaction {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(unique = true, nullable = false)
    private String hash;
    
    private Integer version;
    private Long lockTime;
    private Long size;
    private Long vSize;
    private Long weight;
    
    @Transient
    private List<BitcoinInput> inputs;
    
    @Transient
    private List<BitcoinOutput> outputs;
    
    private Long blockHeight;
    private Long timestamp;
    private boolean hasWitness;
}
