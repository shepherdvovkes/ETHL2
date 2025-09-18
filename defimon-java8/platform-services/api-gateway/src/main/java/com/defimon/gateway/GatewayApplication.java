package com.defimon.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;

/**
 * DEFIMON API Gateway Application
 * 
 * Central API Gateway for the DEFIMON microservices platform.
 * Provides routing, load balancing, security, and monitoring for all services.
 */
@SpringBootApplication
@EnableEurekaClient
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
