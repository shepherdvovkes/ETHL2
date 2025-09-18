package com.defimon.collector;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * DEFIMON Data Collector Service Application
 * 
 * High-performance data collection service for blockchain and market data.
 * Collects data from multiple sources and publishes to Kafka for processing.
 */
@SpringBootApplication
@EnableEurekaClient
@EnableKafka
@EnableScheduling
@EnableAsync
public class DataCollectorApplication {
    public static void main(String[] args) {
        SpringApplication.run(DataCollectorApplication.class, args);
    }
}
