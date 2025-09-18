package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

/**
 * Bitcoin transaction output
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BitcoinOutput {
    private BigDecimal value;
    private String scriptPubKey;
    private String address;
    private Integer index;
}
