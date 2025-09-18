package com.defimon.gateway.config;

import org.springframework.cloud.gateway.filter.ratelimit.KeyResolver;
import org.springframework.cloud.gateway.filter.ratelimit.RedisRateLimiter;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;
import reactor.core.publisher.Mono;

/**
 * Gateway configuration for DEFIMON API Gateway
 */
@Configuration
public class GatewayConfig {

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
                // Core Business Services
                .route("asset-management", r -> r
                    .path("/api/v1/assets/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("asset-management-cb")
                            .setFallbackUri("forward:/fallback/assets"))
                        .retry(3)
                        .requestRateLimiter(config -> config
                            .setRateLimiter(redisRateLimiter())
                            .setKeyResolver(userKeyResolver())))
                    .uri("lb://asset-management-service"))
                
                .route("blockchain-integration", r -> r
                    .path("/api/v1/blockchain/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("blockchain-cb")
                            .setFallbackUri("forward:/fallback/blockchain"))
                        .retry(2)
                        .requestRateLimiter(config -> config
                            .setRateLimiter(redisRateLimiter())
                            .setKeyResolver(userKeyResolver())))
                    .uri("lb://blockchain-integration-service"))
                
                .route("analytics-engine", r -> r
                    .path("/api/v1/analytics/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("analytics-cb")
                            .setFallbackUri("forward:/fallback/analytics"))
                        .requestRateLimiter(config -> config
                            .setRateLimiter(redisRateLimiter())
                            .setKeyResolver(userKeyResolver())))
                    .uri("lb://analytics-engine-service"))
                
                .route("ml-inference", r -> r
                    .path("/api/v1/ml/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("ml-cb")
                            .setFallbackUri("forward:/fallback/ml"))
                        .requestRateLimiter(config -> config
                            .setRateLimiter(redisRateLimiter())
                            .setKeyResolver(userKeyResolver())))
                    .uri("lb://ml-inference-service"))
                
                // Data Services
                .route("data-collector", r -> r
                    .path("/api/v1/collector/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("data-collector-cb")
                            .setFallbackUri("forward:/fallback/collector")))
                    .uri("lb://data-collector-service"))
                
                .route("stream-processor", r -> r
                    .path("/api/v1/stream/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("stream-processor-cb")
                            .setFallbackUri("forward:/fallback/stream")))
                    .uri("lb://stream-processing-service"))
                
                // Blockchain-Specific Services
                .route("bitcoin-service", r -> r
                    .path("/api/v1/bitcoin/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("bitcoin-cb")
                            .setFallbackUri("forward:/fallback/bitcoin")))
                    .uri("lb://bitcoin-service"))
                
                .route("ethereum-service", r -> r
                    .path("/api/v1/ethereum/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("ethereum-cb")
                            .setFallbackUri("forward:/fallback/ethereum")))
                    .uri("lb://ethereum-service"))
                
                .route("polygon-service", r -> r
                    .path("/api/v1/polygon/**")
                    .filters(f -> f
                        .circuitBreaker(config -> config
                            .setName("polygon-cb")
                            .setFallbackUri("forward:/fallback/polygon")))
                    .uri("lb://polygon-service"))
                
                // Admin and Monitoring
                .route("admin-server", r -> r
                    .path("/admin/**")
                    .uri("lb://admin-server"))
                
                .route("prometheus", r -> r
                    .path("/prometheus/**")
                    .uri("http://prometheus:9090"))
                
                .route("grafana", r -> r
                    .path("/grafana/**")
                    .uri("http://grafana:3000"))
                
                .build();
    }

    @Bean
    public CorsWebFilter corsWebFilter() {
        CorsConfiguration corsConfig = new CorsConfiguration();
        corsConfig.setAllowCredentials(true);
        corsConfig.addAllowedOriginPattern("*");
        corsConfig.addAllowedHeader("*");
        corsConfig.addAllowedMethod("*");

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", corsConfig);

        return new CorsWebFilter(source);
    }

    @Bean
    public RedisRateLimiter redisRateLimiter() {
        return new RedisRateLimiter(100, 200, 1); // 100 requests per second, burst 200
    }

    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> exchange.getRequest().getHeaders()
            .getFirst("X-User-ID") != null 
            ? Mono.just(exchange.getRequest().getHeaders().getFirst("X-User-ID"))
            : Mono.just("anonymous");
    }
}
