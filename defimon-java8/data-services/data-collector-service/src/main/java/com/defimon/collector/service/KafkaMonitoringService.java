package com.defimon.collector.service;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.ListTopicsResult;
import org.apache.kafka.clients.admin.TopicDescription;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.TopicPartition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Kafka Monitoring Service
 * Provides comprehensive metrics for monitoring Kafka operations
 */
@Service
public class KafkaMonitoringService {

    private static final Logger logger = LoggerFactory.getLogger(KafkaMonitoringService.class);

    @Autowired
    private MeterRegistry meterRegistry;

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Value("${spring.kafka.bootstrap-servers}")
    private String bootstrapServers;

    // Counters for Kafka operations
    private Counter messagesPublished;
    private Counter messagesPublishedSuccess;
    private Counter messagesPublishedFailed;
    private Counter messagesConsumed;
    private Counter kafkaErrors;
    
    // Counters by topic
    private Counter assetCollectionResultsPublished;
    private Counter bitcoinNetworkDataPublished;
    private Counter bitcoinBlockDataPublished;
    private Counter bitcoinTransactionDataPublished;
    private Counter bitcoinAddressDataPublished;
    
    // Timers for performance monitoring
    private Timer kafkaPublishTimer;
    private Timer kafkaConsumeTimer;
    
    // Gauges for current state
    private AtomicLong totalMessagesPublished;
    private AtomicLong totalMessagesConsumed;
    private AtomicLong kafkaConnectionStatus;
    private AtomicLong topicCount;
    private AtomicLong partitionCount;
    
    // Kafka admin client for monitoring
    private AdminClient adminClient;
    private KafkaConsumer<String, String> consumer;

    @PostConstruct
    public void initializeMetrics() {
        logger.info("Initializing Kafka monitoring metrics...");
        
        // Initialize counters
        messagesPublished = Counter.builder("kafka.messages.published")
                .description("Total number of messages published to Kafka")
                .register(meterRegistry);
                
        messagesPublishedSuccess = Counter.builder("kafka.messages.published.success")
                .description("Number of successfully published messages")
                .register(meterRegistry);
                
        messagesPublishedFailed = Counter.builder("kafka.messages.published.failed")
                .description("Number of failed message publications")
                .register(meterRegistry);
                
        messagesConsumed = Counter.builder("kafka.messages.consumed")
                .description("Total number of messages consumed from Kafka")
                .register(meterRegistry);
                
        kafkaErrors = Counter.builder("kafka.errors")
                .description("Number of Kafka errors")
                .register(meterRegistry);

        // Topic-specific counters
        assetCollectionResultsPublished = Counter.builder("kafka.messages.published")
                .description("Messages published to asset-collection-results topic")
                .tag("topic", "asset-collection-results")
                .register(meterRegistry);
                
        bitcoinNetworkDataPublished = Counter.builder("kafka.messages.published")
                .description("Messages published to bitcoin-network-data topic")
                .tag("topic", "bitcoin-network-data")
                .register(meterRegistry);
                
        bitcoinBlockDataPublished = Counter.builder("kafka.messages.published")
                .description("Messages published to bitcoin-block-data topic")
                .tag("topic", "bitcoin-block-data")
                .register(meterRegistry);
                
        bitcoinTransactionDataPublished = Counter.builder("kafka.messages.published")
                .description("Messages published to bitcoin-transaction-data topic")
                .tag("topic", "bitcoin-transaction-data")
                .register(meterRegistry);
                
        bitcoinAddressDataPublished = Counter.builder("kafka.messages.published")
                .description("Messages published to bitcoin-address-data topic")
                .tag("topic", "bitcoin-address-data")
                .register(meterRegistry);

        // Initialize timers
        kafkaPublishTimer = Timer.builder("kafka.publish.duration")
                .description("Time taken to publish messages to Kafka")
                .register(meterRegistry);
                
        kafkaConsumeTimer = Timer.builder("kafka.consume.duration")
                .description("Time taken to consume messages from Kafka")
                .register(meterRegistry);

        // Initialize atomic longs for gauges
        totalMessagesPublished = new AtomicLong(0);
        totalMessagesConsumed = new AtomicLong(0);
        kafkaConnectionStatus = new AtomicLong(0);
        topicCount = new AtomicLong(0);
        partitionCount = new AtomicLong(0);

        // Register gauges
        Gauge.builder("kafka.messages.total.published")
                .description("Total messages published to Kafka")
                .register(meterRegistry, totalMessagesPublished, AtomicLong::get);
                
        Gauge.builder("kafka.messages.total.consumed")
                .description("Total messages consumed from Kafka")
                .register(meterRegistry, totalMessagesConsumed, AtomicLong::get);
                
        Gauge.builder("kafka.connection.status")
                .description("Kafka connection status (1=connected, 0=disconnected)")
                .register(meterRegistry, kafkaConnectionStatus, AtomicLong::get);
                
        Gauge.builder("kafka.topics.count")
                .description("Number of Kafka topics")
                .register(meterRegistry, topicCount, AtomicLong::get);
                
        Gauge.builder("kafka.partitions.count")
                .description("Total number of Kafka partitions")
                .register(meterRegistry, partitionCount, AtomicLong::get);

        // Initialize Kafka admin client
        initializeKafkaAdminClient();
        
        logger.info("Kafka monitoring metrics initialized successfully");
    }

    private void initializeKafkaAdminClient() {
        try {
            Properties props = new Properties();
            props.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
            props.put(AdminClientConfig.REQUEST_TIMEOUT_MS_CONFIG, 5000);
            props.put(AdminClientConfig.CONNECTIONS_MAX_IDLE_MS_CONFIG, 10000);
            
            adminClient = AdminClient.create(props);
            
            // Test connection
            testKafkaConnection();
            
        } catch (Exception e) {
            logger.error("Failed to initialize Kafka admin client: {}", e.getMessage());
            kafkaConnectionStatus.set(0);
        }
    }

    private void testKafkaConnection() {
        try {
            ListTopicsResult topics = adminClient.listTopics();
            Set<String> topicNames = topics.names().get(5, java.util.concurrent.TimeUnit.SECONDS);
            
            kafkaConnectionStatus.set(1);
            topicCount.set(topicNames.size());
            
            // Count total partitions
            int totalPartitions = 0;
            for (String topicName : topicNames) {
                try {
                    TopicDescription description = adminClient.describeTopics(Collections.singleton(topicName))
                            .values().get(topicName).get(5, java.util.concurrent.TimeUnit.SECONDS);
                    totalPartitions += description.partitions().size();
                } catch (Exception e) {
                    logger.warn("Could not get partition count for topic {}: {}", topicName, e.getMessage());
                }
            }
            partitionCount.set(totalPartitions);
            
            logger.info("Kafka connection test successful - Topics: {}, Partitions: {}", 
                topicNames.size(), totalPartitions);
                
        } catch (Exception e) {
            logger.error("Kafka connection test failed: {}", e.getMessage());
            kafkaConnectionStatus.set(0);
            kafkaErrors.increment();
        }
    }

    // Message publishing tracking
    public void recordMessagePublished(String topic) {
        messagesPublished.increment();
        totalMessagesPublished.incrementAndGet();
        
        // Record topic-specific metrics
        switch (topic) {
            case "asset-collection-results":
                assetCollectionResultsPublished.increment();
                break;
            case "bitcoin-network-data":
                bitcoinNetworkDataPublished.increment();
                break;
            case "bitcoin-block-data":
                bitcoinBlockDataPublished.increment();
                break;
            case "bitcoin-transaction-data":
                bitcoinTransactionDataPublished.increment();
                break;
            case "bitcoin-address-data":
                bitcoinAddressDataPublished.increment();
                break;
        }
    }

    public void recordMessagePublishedSuccess() {
        messagesPublishedSuccess.increment();
    }

    public void recordMessagePublishedFailed() {
        messagesPublishedFailed.increment();
        kafkaErrors.increment();
    }

    public void recordMessageConsumed() {
        messagesConsumed.increment();
        totalMessagesConsumed.incrementAndGet();
    }

    public void recordKafkaError() {
        kafkaErrors.increment();
    }

    // Timer tracking
    public Timer.Sample startPublishTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordPublishDuration(Timer.Sample sample) {
        sample.stop(kafkaPublishTimer);
    }

    public Timer.Sample startConsumeTimer() {
        return Timer.start(meterRegistry);
    }

    public void recordConsumeDuration(Timer.Sample sample) {
        sample.stop(kafkaConsumeTimer);
    }

    // Connection monitoring
    public void updateConnectionStatus(boolean connected) {
        kafkaConnectionStatus.set(connected ? 1 : 0);
    }

    // Periodic monitoring
    public void refreshKafkaMetrics() {
        testKafkaConnection();
    }

    // Utility methods
    public long getTotalMessagesPublished() {
        return totalMessagesPublished.get();
    }

    public long getTotalMessagesConsumed() {
        return totalMessagesConsumed.get();
    }

    public boolean isKafkaConnected() {
        return kafkaConnectionStatus.get() == 1;
    }

    public long getTopicCount() {
        return topicCount.get();
    }

    public long getPartitionCount() {
        return partitionCount.get();
    }
}
