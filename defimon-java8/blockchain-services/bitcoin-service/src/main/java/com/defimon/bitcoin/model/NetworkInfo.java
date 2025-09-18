package com.defimon.bitcoin.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Bitcoin network information
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NetworkInfo {
    private Integer version;
    private String subversion;
    private Integer protocolVersion;
    private String localServices;
    private Boolean localRelay;
    private Long timeOffset;
    private Integer connections;
    private Boolean networkActive;
    private String relayFee;
    private String incrementalfee;
    private String localAddresses;
}
