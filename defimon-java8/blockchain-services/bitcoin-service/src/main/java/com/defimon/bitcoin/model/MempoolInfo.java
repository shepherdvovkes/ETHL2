package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Bitcoin mempool information
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MempoolInfo {
    private Long size;
    private Long bytes;
    private Long usage;
    private Long maxMempool;
    private Double mempoolMinFee;
    private Double minRelayTxFee;
}
