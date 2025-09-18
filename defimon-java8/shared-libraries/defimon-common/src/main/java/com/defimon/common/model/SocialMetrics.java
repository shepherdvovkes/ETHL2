package com.defimon.common.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Social metrics for cryptocurrency assets
 */
@Entity
@Table(name = "social_metrics", indexes = {
    @Index(name = "idx_social_asset_timestamp", columnList = "asset_id, timestamp"),
    @Index(name = "idx_social_timestamp", columnList = "timestamp")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class SocialMetrics {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @NotNull
    @Column(name = "asset_id", nullable = false)
    private Long assetId;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "asset_id", insertable = false, updatable = false)
    private Asset asset;
    
    @NotNull
    @Column(name = "timestamp", nullable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime timestamp;
    
    // GitHub Metrics
    @Column(name = "github_stars")
    private Long githubStars;
    
    @Column(name = "github_forks")
    private Long githubForks;
    
    @Column(name = "github_watchers")
    private Long githubWatchers;
    
    @Column(name = "github_commits_24h")
    private Long githubCommits24h;
    
    @Column(name = "github_commits_7d")
    private Long githubCommits7d;
    
    @Column(name = "github_commits_30d")
    private Long githubCommits30d;
    
    @Column(name = "github_contributors")
    private Long githubContributors;
    
    @Column(name = "github_issues_open")
    private Long githubIssuesOpen;
    
    @Column(name = "github_issues_closed")
    private Long githubIssuesClosed;
    
    @Column(name = "github_pull_requests_open")
    private Long githubPullRequestsOpen;
    
    @Column(name = "github_pull_requests_merged")
    private Long githubPullRequestsMerged;
    
    @Column(name = "github_languages", columnDefinition = "TEXT")
    private String githubLanguages; // JSON string of languages
    
    @Column(name = "github_last_commit")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime githubLastCommit;
    
    // Twitter/X Metrics
    @Column(name = "twitter_followers")
    private Long twitterFollowers;
    
    @Column(name = "twitter_following")
    private Long twitterFollowing;
    
    @Column(name = "twitter_tweets_24h")
    private Long twitterTweets24h;
    
    @Column(name = "twitter_tweets_7d")
    private Long twitterTweets7d;
    
    @Column(name = "twitter_mentions_24h")
    private Long twitterMentions24h;
    
    @Column(name = "twitter_mentions_7d")
    private Long twitterMentions7d;
    
    @Column(name = "twitter_retweets_24h")
    private Long twitterRetweets24h;
    
    @Column(name = "twitter_likes_24h")
    private Long twitterLikes24h;
    
    @Column(name = "twitter_sentiment_score", precision = 5, scale = 4)
    private BigDecimal twitterSentimentScore;
    
    @Column(name = "twitter_sentiment_positive", precision = 5, scale = 4)
    private BigDecimal twitterSentimentPositive;
    
    @Column(name = "twitter_sentiment_negative", precision = 5, scale = 4)
    private BigDecimal twitterSentimentNegative;
    
    @Column(name = "twitter_sentiment_neutral", precision = 5, scale = 4)
    private BigDecimal twitterSentimentNeutral;
    
    // Reddit Metrics
    @Column(name = "reddit_subscribers")
    private Long redditSubscribers;
    
    @Column(name = "reddit_active_users")
    private Long redditActiveUsers;
    
    @Column(name = "reddit_posts_24h")
    private Long redditPosts24h;
    
    @Column(name = "reddit_comments_24h")
    private Long redditComments24h;
    
    @Column(name = "reddit_upvotes_24h")
    private Long redditUpvotes24h;
    
    @Column(name = "reddit_downvotes_24h")
    private Long redditDownvotes24h;
    
    @Column(name = "reddit_sentiment_score", precision = 5, scale = 4)
    private BigDecimal redditSentimentScore;
    
    // Telegram Metrics
    @Column(name = "telegram_members")
    private Long telegramMembers;
    
    @Column(name = "telegram_messages_24h")
    private Long telegramMessages24h;
    
    @Column(name = "telegram_messages_7d")
    private Long telegramMessages7d;
    
    @Column(name = "telegram_sentiment_score", precision = 5, scale = 4)
    private BigDecimal telegramSentimentScore;
    
    // Discord Metrics
    @Column(name = "discord_members")
    private Long discordMembers;
    
    @Column(name = "discord_online_members")
    private Long discordOnlineMembers;
    
    @Column(name = "discord_messages_24h")
    private Long discordMessages24h;
    
    @Column(name = "discord_sentiment_score", precision = 5, scale = 4)
    private BigDecimal discordSentimentScore;
    
    // YouTube Metrics
    @Column(name = "youtube_subscribers")
    private Long youtubeSubscribers;
    
    @Column(name = "youtube_videos_30d")
    private Long youtubeVideos30d;
    
    @Column(name = "youtube_views_30d")
    private Long youtubeViews30d;
    
    @Column(name = "youtube_likes_30d")
    private Long youtubeLikes30d;
    
    @Column(name = "youtube_dislikes_30d")
    private Long youtubeDislikes30d;
    
    @Column(name = "youtube_sentiment_score", precision = 5, scale = 4)
    private BigDecimal youtubeSentimentScore;
    
    // News and Media Metrics
    @Column(name = "news_mentions_24h")
    private Long newsMentions24h;
    
    @Column(name = "news_mentions_7d")
    private Long newsMentions7d;
    
    @Column(name = "news_sentiment_score", precision = 5, scale = 4)
    private BigDecimal newsSentimentScore;
    
    @Column(name = "news_sentiment_positive", precision = 5, scale = 4)
    private BigDecimal newsSentimentPositive;
    
    @Column(name = "news_sentiment_negative", precision = 5, scale = 4)
    private BigDecimal newsSentimentNegative;
    
    @Column(name = "news_sentiment_neutral", precision = 5, scale = 4)
    private BigDecimal newsSentimentNeutral;
    
    // Search Metrics
    @Column(name = "google_trends_score", precision = 5, scale = 4)
    private BigDecimal googleTrendsScore;
    
    @Column(name = "google_search_volume")
    private Long googleSearchVolume;
    
    @Column(name = "bing_search_volume")
    private Long bingSearchVolume;
    
    @Column(name = "yahoo_search_volume")
    private Long yahooSearchVolume;
    
    // Influencer Metrics
    @Column(name = "influencer_mentions_24h")
    private Long influencerMentions24h;
    
    @Column(name = "influencer_sentiment_score", precision = 5, scale = 4)
    private BigDecimal influencerSentimentScore;
    
    @Column(name = "influencer_reach")
    private Long influencerReach;
    
    // Community Health Metrics
    @Column(name = "community_growth_rate", precision = 10, scale = 4)
    private BigDecimal communityGrowthRate;
    
    @Column(name = "engagement_rate", precision = 5, scale = 4)
    private BigDecimal engagementRate;
    
    @Column(name = "activity_score", precision = 5, scale = 4)
    private BigDecimal activityScore;
    
    @Column(name = "buzz_score", precision = 5, scale = 4)
    private BigDecimal buzzScore;
    
    @Column(name = "fear_greed_index", precision = 5, scale = 4)
    private BigDecimal fearGreedIndex;
    
    // Overall Social Scores
    @Column(name = "social_volume_score", precision = 5, scale = 4)
    private BigDecimal socialVolumeScore;
    
    @Column(name = "social_sentiment_score", precision = 5, scale = 4)
    private BigDecimal socialSentimentScore;
    
    @Column(name = "social_engagement_score", precision = 5, scale = 4)
    private BigDecimal socialEngagementScore;
    
    @Column(name = "social_influence_score", precision = 5, scale = 4)
    private BigDecimal socialInfluenceScore;
    
    @Column(name = "overall_social_score", precision = 5, scale = 4)
    private BigDecimal overallSocialScore;
    
    @Column(name = "created_at", nullable = false, updatable = false)
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        if (timestamp == null) {
            timestamp = LocalDateTime.now();
        }
    }
    
    /**
     * Calculate overall social score from individual components
     */
    public BigDecimal calculateOverallSocialScore() {
        BigDecimal volume = socialVolumeScore != null ? socialVolumeScore : BigDecimal.ZERO;
        BigDecimal sentiment = socialSentimentScore != null ? socialSentimentScore : BigDecimal.ZERO;
        BigDecimal engagement = socialEngagementScore != null ? socialEngagementScore : BigDecimal.ZERO;
        BigDecimal influence = socialInfluenceScore != null ? socialInfluenceScore : BigDecimal.ZERO;
        
        return volume.add(sentiment).add(engagement).add(influence)
                   .divide(BigDecimal.valueOf(4), 4, BigDecimal.ROUND_HALF_UP);
    }
    
    /**
     * Calculate engagement rate
     */
    public BigDecimal calculateEngagementRate() {
        Long totalInteractions = getTotalSocialInteractions();
        Long totalFollowers = getTotalFollowers();
        
        if (totalFollowers == null || totalFollowers == 0) {
            return BigDecimal.ZERO;
        }
        
        return BigDecimal.valueOf(totalInteractions)
                       .divide(BigDecimal.valueOf(totalFollowers), 6, BigDecimal.ROUND_HALF_UP);
    }
    
    /**
     * Get total social interactions across all platforms
     */
    public Long getTotalSocialInteractions() {
        long interactions = 0;
        
        if (twitterLikes24h != null) interactions += twitterLikes24h;
        if (twitterRetweets24h != null) interactions += twitterRetweets24h;
        if (redditUpvotes24h != null) interactions += redditUpvotes24h;
        if (youtubeLikes30d != null) interactions += youtubeLikes30d;
        
        return interactions;
    }
    
    /**
     * Get total followers across all platforms
     */
    public Long getTotalFollowers() {
        long followers = 0;
        
        if (twitterFollowers != null) followers += twitterFollowers;
        if (redditSubscribers != null) followers += redditSubscribers;
        if (telegramMembers != null) followers += telegramMembers;
        if (discordMembers != null) followers += discordMembers;
        if (youtubeSubscribers != null) followers += youtubeSubscribers;
        
        return followers;
    }
    
    /**
     * Calculate community growth rate
     */
    public BigDecimal calculateCommunityGrowthRate(SocialMetrics previousMetrics) {
        if (previousMetrics == null) {
            return BigDecimal.ZERO;
        }
        
        Long currentFollowers = getTotalFollowers();
        Long previousFollowers = previousMetrics.getTotalFollowers();
        
        if (previousFollowers == null || previousFollowers == 0) {
            return BigDecimal.ZERO;
        }
        
        return BigDecimal.valueOf(currentFollowers - previousFollowers)
                       .divide(BigDecimal.valueOf(previousFollowers), 6, BigDecimal.ROUND_HALF_UP)
                       .multiply(BigDecimal.valueOf(100));
    }
    
    /**
     * Check if metrics are recent (within specified minutes)
     */
    public boolean isRecent(int minutesThreshold) {
        if (timestamp == null) {
            return false;
        }
        return timestamp.isAfter(LocalDateTime.now().minusMinutes(minutesThreshold));
    }
}
