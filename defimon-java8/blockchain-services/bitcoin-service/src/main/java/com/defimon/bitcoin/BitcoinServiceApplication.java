package com.defimon.bitcoin;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * DEFIMON Bitcoin Service Application
 * 
 * Bitcoin blockchain integration service with QuickNode API support.
 * Provides Bitcoin network data, transaction analysis, and metrics.
 */
@SpringBootApplication
@EnableEurekaClient
@EnableScheduling
public class BitcoinServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(BitcoinServiceApplication.class, args);
    }
}
