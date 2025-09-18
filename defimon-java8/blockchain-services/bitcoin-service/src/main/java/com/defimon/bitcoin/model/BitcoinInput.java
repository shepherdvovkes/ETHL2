package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Bitcoin transaction input
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BitcoinInput {
    private String previousTxHash;
    private Integer previousOutputIndex;
    private BitcoinOutput previousOutput;
    private String scriptSig;
    private Long sequence;
}
