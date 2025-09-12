#!/usr/bin/env python3
"""
Анализатор трендов и паттернов для Polygon
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics
)
from config.settings import settings

@dataclass
class TrendAnalysis:
    """Результат анализа тренда"""
    metric_name: str
    trend_direction: str  # "bullish", "bearish", "sideways"
    trend_strength: float  # 0-1
    change_percent: float
    period: str
    confidence: float
    pattern_type: str
    recommendation: str

@dataclass
class PatternDetection:
    """Обнаруженный паттерн"""
    pattern_name: str
    pattern_type: str  # "seasonal", "cyclical", "anomaly", "correlation"
    strength: float  # 0-1
    frequency: str
    description: str
    impact_on_price: str  # "positive", "negative", "neutral"

class PolygonTrendAnalyzer:
    """Анализатор трендов и паттернов для Polygon"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.polygon_asset_id = None
        self.analysis_results = {}
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def setup_polygon_asset(self):
        """Настроить актив Polygon"""
        try:
            matic_asset = self.session.query(CryptoAsset).filter(
                CryptoAsset.symbol == "MATIC"
            ).first()
            
            if not matic_asset:
                raise ValueError("MATIC asset not found in database")
            
            self.polygon_asset_id = matic_asset.id
            logger.info(f"Polygon asset setup complete. Asset ID: {self.polygon_asset_id}")
            
        except Exception as e:
            logger.error(f"Error setting up Polygon asset: {e}")
            raise
    
    async def analyze_price_trends(self, days_back: int = 90) -> List[TrendAnalysis]:
        """Анализ ценовых трендов"""
        try:
            logger.info("Analyzing price trends...")
            
            # Получить финансовые метрики
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            if not financial_metrics:
                logger.warning("No financial metrics found")
                return []
            
            # Создать DataFrame
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'price': float(m.price_usd or 0),
                'volume': float(m.volume_24h or 0),
                'market_cap': float(m.market_cap or 0),
                'volatility': float(m.volatility_24h or 0),
                'price_change_24h': float(m.price_change_24h or 0),
                'price_change_7d': float(m.price_change_7d or 0),
                'price_change_30d': float(m.price_change_30d or 0)
            } for m in financial_metrics])
            
            trends = []
            
            # Анализ тренда цены
            if len(df) > 1:
                price_trend = self._analyze_metric_trend(
                    df['price'], "Price", "7d", df['timestamp']
                )
                trends.append(price_trend)
                
                # Анализ тренда объема
                volume_trend = self._analyze_metric_trend(
                    df['volume'], "Volume", "7d", df['timestamp']
                )
                trends.append(volume_trend)
                
                # Анализ тренда волатильности
                volatility_trend = self._analyze_metric_trend(
                    df['volatility'], "Volatility", "7d", df['timestamp']
                )
                trends.append(volatility_trend)
                
                # Анализ тренда рыночной капитализации
                market_cap_trend = self._analyze_metric_trend(
                    df['market_cap'], "Market Cap", "7d", df['timestamp']
                )
                trends.append(market_cap_trend)
            
            logger.info(f"Analyzed {len(trends)} price trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing price trends: {e}")
            return []
    
    async def analyze_onchain_trends(self, days_back: int = 90) -> List[TrendAnalysis]:
        """Анализ on-chain трендов"""
        try:
            logger.info("Analyzing on-chain trends...")
            
            # Получить on-chain метрики
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            onchain_metrics = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == self.polygon_asset_id,
                OnChainMetrics.timestamp >= start_date,
                OnChainMetrics.timestamp <= end_date
            ).order_by(OnChainMetrics.timestamp).all()
            
            if not onchain_metrics:
                logger.warning("No on-chain metrics found")
                return []
            
            # Создать DataFrame
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'tvl': float(m.tvl or 0),
                'tvl_change_24h': float(m.tvl_change_24h or 0),
                'tvl_change_7d': float(m.tvl_change_7d or 0),
                'daily_transactions': float(m.daily_transactions or 0),
                'active_addresses': float(m.active_addresses_24h or 0),
                'transaction_volume': float(m.transaction_volume_24h or 0),
                'gas_price': float(m.gas_price_avg or 0),
                'contract_interactions': float(m.contract_interactions_24h or 0),
                'liquidity_pools_count': float(m.liquidity_pools_count or 0),
                'liquidity_pools_tvl': float(m.liquidity_pools_tvl or 0)
            } for m in onchain_metrics])
            
            trends = []
            
            # Анализ различных метрик
            metrics_to_analyze = [
                ('tvl', 'TVL'),
                ('daily_transactions', 'Daily Transactions'),
                ('active_addresses', 'Active Addresses'),
                ('transaction_volume', 'Transaction Volume'),
                ('gas_price', 'Gas Price'),
                ('contract_interactions', 'Contract Interactions'),
                ('liquidity_pools_count', 'Liquidity Pools Count'),
                ('liquidity_pools_tvl', 'Liquidity Pools TVL')
            ]
            
            for metric_col, metric_name in metrics_to_analyze:
                if metric_col in df.columns and len(df[metric_col].dropna()) > 1:
                    trend = self._analyze_metric_trend(
                        df[metric_col], metric_name, "7d", df['timestamp']
                    )
                    trends.append(trend)
            
            logger.info(f"Analyzed {len(trends)} on-chain trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing on-chain trends: {e}")
            return []
    
    async def analyze_ecosystem_trends(self, days_back: int = 90) -> List[TrendAnalysis]:
        """Анализ трендов экосистемы"""
        try:
            logger.info("Analyzing ecosystem trends...")
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            trends = []
            
            # GitHub метрики
            github_metrics = self.session.query(GitHubMetrics).filter(
                GitHubMetrics.asset_id == self.polygon_asset_id,
                GitHubMetrics.timestamp >= start_date,
                GitHubMetrics.timestamp <= end_date
            ).order_by(GitHubMetrics.timestamp).all()
            
            if github_metrics:
                df_github = pd.DataFrame([{
                    'timestamp': m.timestamp,
                    'commits_24h': float(m.commits_24h or 0),
                    'commits_7d': float(m.commits_7d or 0),
                    'commits_30d': float(m.commits_30d or 0),
                    'active_contributors': float(m.active_contributors_30d or 0),
                    'stars': float(m.stars or 0),
                    'forks': float(m.forks or 0),
                    'open_issues': float(m.open_issues or 0),
                    'open_prs': float(m.open_prs or 0),
                    'code_quality_score': float(m.code_quality_score or 0)
                } for m in github_metrics])
                
                github_metrics_to_analyze = [
                    ('commits_7d', 'GitHub Commits'),
                    ('active_contributors', 'Active Contributors'),
                    ('stars', 'GitHub Stars'),
                    ('forks', 'GitHub Forks'),
                    ('code_quality_score', 'Code Quality Score')
                ]
                
                for metric_col, metric_name in github_metrics_to_analyze:
                    if metric_col in df_github.columns and len(df_github[metric_col].dropna()) > 1:
                        trend = self._analyze_metric_trend(
                            df_github[metric_col], metric_name, "7d", df_github['timestamp']
                        )
                        trends.append(trend)
            
            # Community метрики
            community_metrics = self.session.query(CommunityMetrics).filter(
                CommunityMetrics.asset_id == self.polygon_asset_id,
                CommunityMetrics.timestamp >= start_date,
                CommunityMetrics.timestamp <= end_date
            ).order_by(CommunityMetrics.timestamp).all()
            
            if community_metrics:
                df_community = pd.DataFrame([{
                    'timestamp': m.timestamp,
                    'twitter_followers': float(m.twitter_followers or 0),
                    'telegram_members': float(m.telegram_members or 0),
                    'discord_members': float(m.discord_members or 0),
                    'social_engagement_rate': float(m.social_engagement_rate or 0),
                    'brand_awareness_score': float(m.brand_awareness_score or 0)
                } for m in community_metrics])
                
                community_metrics_to_analyze = [
                    ('twitter_followers', 'Twitter Followers'),
                    ('telegram_members', 'Telegram Members'),
                    ('discord_members', 'Discord Members'),
                    ('social_engagement_rate', 'Social Engagement Rate'),
                    ('brand_awareness_score', 'Brand Awareness Score')
                ]
                
                for metric_col, metric_name in community_metrics_to_analyze:
                    if metric_col in df_community.columns and len(df_community[metric_col].dropna()) > 1:
                        trend = self._analyze_metric_trend(
                            df_community[metric_col], metric_name, "7d", df_community['timestamp']
                        )
                        trends.append(trend)
            
            logger.info(f"Analyzed {len(trends)} ecosystem trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing ecosystem trends: {e}")
            return []
    
    def _analyze_metric_trend(
        self, 
        values: pd.Series, 
        metric_name: str, 
        period: str,
        timestamps: pd.Series
    ) -> TrendAnalysis:
        """Анализ тренда для конкретной метрики"""
        try:
            # Очистить данные
            clean_values = values.dropna()
            if len(clean_values) < 2:
                return TrendAnalysis(
                    metric_name=metric_name,
                    trend_direction="sideways",
                    trend_strength=0.0,
                    change_percent=0.0,
                    period=period,
                    confidence=0.0,
                    pattern_type="insufficient_data",
                    recommendation="Collect more data"
                )
            
            # Рассчитать изменения
            first_value = clean_values.iloc[0]
            last_value = clean_values.iloc[-1]
            change_percent = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
            
            # Определить направление тренда
            if change_percent > 5:
                trend_direction = "bullish"
            elif change_percent < -5:
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"
            
            # Рассчитать силу тренда
            trend_strength = min(abs(change_percent) / 20, 1.0)  # Нормализация к 0-1
            
            # Рассчитать уверенность
            volatility = clean_values.std() / clean_values.mean() if clean_values.mean() != 0 else 1
            confidence = max(0, 1 - volatility)
            
            # Определить тип паттерна
            pattern_type = self._detect_pattern_type(clean_values)
            
            # Создать рекомендацию
            recommendation = self._generate_recommendation(
                metric_name, trend_direction, change_percent, pattern_type
            )
            
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_percent=change_percent,
                period=period,
                confidence=confidence,
                pattern_type=pattern_type,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="sideways",
                trend_strength=0.0,
                change_percent=0.0,
                period=period,
                confidence=0.0,
                pattern_type="error",
                recommendation="Analysis failed"
            )
    
    def _detect_pattern_type(self, values: pd.Series) -> str:
        """Обнаружить тип паттерна в данных"""
        try:
            if len(values) < 7:
                return "insufficient_data"
            
            # Проверить на сезонность (еженедельные паттерны)
            if len(values) >= 14:
                weekly_avg = values.rolling(window=7).mean()
                if weekly_avg.std() / weekly_avg.mean() < 0.1:
                    return "seasonal"
            
            # Проверить на тренд
            x = np.arange(len(values))
            correlation = np.corrcoef(x, values)[0, 1]
            
            if correlation > 0.7:
                return "strong_uptrend"
            elif correlation < -0.7:
                return "strong_downtrend"
            elif correlation > 0.3:
                return "weak_uptrend"
            elif correlation < -0.3:
                return "weak_downtrend"
            else:
                return "sideways"
            
        except Exception as e:
            logger.error(f"Error detecting pattern: {e}")
            return "unknown"
    
    def _generate_recommendation(
        self, 
        metric_name: str, 
        trend_direction: str, 
        change_percent: float,
        pattern_type: str
    ) -> str:
        """Сгенерировать рекомендацию на основе анализа"""
        recommendations = {
            "Price": {
                "bullish": f"Price is trending upward ({change_percent:.1f}%). Consider holding or accumulating.",
                "bearish": f"Price is trending downward ({change_percent:.1f}%). Consider reducing position or waiting for better entry.",
                "sideways": f"Price is moving sideways ({change_percent:.1f}%). Wait for breakout or accumulation opportunity."
            },
            "Volume": {
                "bullish": "High volume confirms bullish momentum. Strong buying interest.",
                "bearish": "High volume on decline suggests selling pressure. Monitor for capitulation.",
                "sideways": "Low volume suggests consolidation. Wait for volume breakout."
            },
            "TVL": {
                "bullish": "Growing TVL indicates increasing DeFi adoption. Positive for long-term price.",
                "bearish": "Declining TVL suggests reduced DeFi activity. Monitor for recovery.",
                "sideways": "Stable TVL indicates consistent DeFi usage. Neutral signal."
            },
            "Active Addresses": {
                "bullish": "Growing active addresses show network adoption. Bullish for ecosystem.",
                "bearish": "Declining active addresses suggest reduced usage. Bearish signal.",
                "sideways": "Stable active addresses indicate consistent usage. Neutral."
            }
        }
        
        if metric_name in recommendations and trend_direction in recommendations[metric_name]:
            return recommendations[metric_name][trend_direction]
        
        # Общие рекомендации
        if trend_direction == "bullish":
            return f"{metric_name} showing bullish trend. Monitor for continuation."
        elif trend_direction == "bearish":
            return f"{metric_name} showing bearish trend. Watch for reversal signals."
        else:
            return f"{metric_name} moving sideways. Wait for directional breakout."
    
    async def detect_seasonal_patterns(self, days_back: int = 365) -> List[PatternDetection]:
        """Обнаружить сезонные паттерны"""
        try:
            logger.info("Detecting seasonal patterns...")
            
            # Получить данные за более длительный период
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            if not financial_metrics or len(financial_metrics) < 30:
                logger.warning("Insufficient data for seasonal analysis")
                return []
            
            # Создать DataFrame
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'price': float(m.price_usd or 0),
                'volume': float(m.volume_24h or 0),
                'price_change_24h': float(m.price_change_24h or 0)
            } for m in financial_metrics])
            
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            
            patterns = []
            
            # Анализ дневных паттернов
            daily_pattern = self._analyze_daily_patterns(df)
            if daily_pattern:
                patterns.append(daily_pattern)
            
            # Анализ недельных паттернов
            weekly_pattern = self._analyze_weekly_patterns(df)
            if weekly_pattern:
                patterns.append(weekly_pattern)
            
            # Анализ месячных паттернов
            monthly_pattern = self._analyze_monthly_patterns(df)
            if monthly_pattern:
                patterns.append(monthly_pattern)
            
            logger.info(f"Detected {len(patterns)} seasonal patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {e}")
            return []
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Optional[PatternDetection]:
        """Анализ дневных паттернов"""
        try:
            if 'day_of_week' not in df.columns:
                return None
            
            # Группировка по дням недели
            daily_stats = df.groupby('day_of_week').agg({
                'price_change_24h': ['mean', 'std'],
                'volume': 'mean'
            }).round(4)
            
            # Проверить на значимые различия
            price_changes = daily_stats[('price_change_24h', 'mean')]
            if price_changes.std() > 1.0:  # Значимое отклонение
                best_day = price_changes.idxmax()
                worst_day = price_changes.idxmin()
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                return PatternDetection(
                    pattern_name="Daily Price Pattern",
                    pattern_type="seasonal",
                    strength=min(price_changes.std() / 2, 1.0),
                    frequency="daily",
                    description=f"Best performance on {day_names[best_day]}, worst on {day_names[worst_day]}",
                    impact_on_price="neutral"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing daily patterns: {e}")
            return None
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Optional[PatternDetection]:
        """Анализ недельных паттернов"""
        try:
            if len(df) < 14:
                return None
            
            # Рассчитать недельные изменения
            df['week'] = df['timestamp'].dt.isocalendar().week
            weekly_changes = df.groupby('week')['price_change_24h'].mean()
            
            if len(weekly_changes) > 4 and weekly_changes.std() > 2.0:
                return PatternDetection(
                    pattern_name="Weekly Price Pattern",
                    pattern_type="cyclical",
                    strength=min(weekly_changes.std() / 5, 1.0),
                    frequency="weekly",
                    description=f"Weekly price changes show {weekly_changes.std():.2f}% standard deviation",
                    impact_on_price="neutral"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing weekly patterns: {e}")
            return None
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Optional[PatternDetection]:
        """Анализ месячных паттернов"""
        try:
            if 'month' not in df.columns or len(df) < 60:
                return None
            
            # Группировка по месяцам
            monthly_stats = df.groupby('month')['price_change_24h'].mean()
            
            if len(monthly_stats) > 6 and monthly_stats.std() > 3.0:
                best_month = monthly_stats.idxmax()
                worst_month = monthly_stats.idxmin()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                return PatternDetection(
                    pattern_name="Monthly Price Pattern",
                    pattern_type="seasonal",
                    strength=min(monthly_stats.std() / 10, 1.0),
                    frequency="monthly",
                    description=f"Best performance in {month_names[best_month-1]}, worst in {month_names[worst_month-1]}",
                    impact_on_price="neutral"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing monthly patterns: {e}")
            return None
    
    async def detect_anomalies(self, days_back: int = 90) -> List[PatternDetection]:
        """Обнаружить аномалии в данных"""
        try:
            logger.info("Detecting anomalies...")
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            if not financial_metrics:
                return []
            
            # Создать DataFrame
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'price': float(m.price_usd or 0),
                'volume': float(m.volume_24h or 0),
                'price_change_24h': float(m.price_change_24h or 0),
                'volatility': float(m.volatility_24h or 0)
            } for m in financial_metrics])
            
            anomalies = []
            
            # Обнаружить аномалии в изменении цены
            price_changes = df['price_change_24h'].dropna()
            if len(price_changes) > 0:
                mean_change = price_changes.mean()
                std_change = price_changes.std()
                
                # Аномалии: изменения больше 2 стандартных отклонений
                threshold = 2 * std_change
                price_anomalies = df[abs(df['price_change_24h'] - mean_change) > threshold]
                
                for _, anomaly in price_anomalies.iterrows():
                    severity = "high" if abs(anomaly['price_change_24h']) > 20 else "medium"
                    direction = "positive" if anomaly['price_change_24h'] > 0 else "negative"
                    
                    anomalies.append(PatternDetection(
                        pattern_name=f"Price Anomaly ({anomaly['timestamp'].strftime('%Y-%m-%d')})",
                        pattern_type="anomaly",
                        strength=min(abs(anomaly['price_change_24h']) / 30, 1.0),
                        frequency="irregular",
                        description=f"{direction} price spike of {anomaly['price_change_24h']:.2f}%",
                        impact_on_price=direction
                    ))
            
            # Обнаружить аномалии в объеме
            volumes = df['volume'].dropna()
            if len(volumes) > 0:
                mean_volume = volumes.mean()
                std_volume = volumes.std()
                
                volume_anomalies = df[df['volume'] > mean_volume + 3 * std_volume]
                
                for _, anomaly in volume_anomalies.iterrows():
                    volume_ratio = anomaly['volume'] / mean_volume
                    
                    anomalies.append(PatternDetection(
                        pattern_name=f"Volume Anomaly ({anomaly['timestamp'].strftime('%Y-%m-%d')})",
                        pattern_type="anomaly",
                        strength=min(volume_ratio / 5, 1.0),
                        frequency="irregular",
                        description=f"Volume spike: {volume_ratio:.1f}x normal volume",
                        impact_on_price="neutral"
                    ))
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def analyze_correlations(self, days_back: int = 90) -> Dict[str, float]:
        """Анализ корреляций между метриками"""
        try:
            logger.info("Analyzing correlations...")
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Получить все метрики
            financial_metrics = self.session.query(FinancialMetrics).filter(
                FinancialMetrics.asset_id == self.polygon_asset_id,
                FinancialMetrics.timestamp >= start_date,
                FinancialMetrics.timestamp <= end_date
            ).order_by(FinancialMetrics.timestamp).all()
            
            onchain_metrics = self.session.query(OnChainMetrics).filter(
                OnChainMetrics.asset_id == self.polygon_asset_id,
                OnChainMetrics.timestamp >= start_date,
                OnChainMetrics.timestamp <= end_date
            ).order_by(OnChainMetrics.timestamp).all()
            
            if not financial_metrics or not onchain_metrics:
                return {}
            
            # Создать объединенный DataFrame
            financial_df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'price': float(m.price_usd or 0),
                'volume': float(m.volume_24h or 0),
                'market_cap': float(m.market_cap or 0),
                'volatility': float(m.volatility_24h or 0)
            } for m in financial_metrics])
            
            onchain_df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'tvl': float(m.tvl or 0),
                'daily_transactions': float(m.daily_transactions or 0),
                'active_addresses': float(m.active_addresses_24h or 0),
                'transaction_volume': float(m.transaction_volume_24h or 0),
                'gas_price': float(m.gas_price_avg or 0)
            } for m in onchain_metrics])
            
            # Объединить данные
            merged_df = pd.merge(financial_df, onchain_df, on='timestamp', how='inner')
            
            if len(merged_df) < 10:
                return {}
            
            # Рассчитать корреляции с ценой
            correlations = {}
            price = merged_df['price']
            
            for column in merged_df.columns:
                if column not in ['timestamp', 'price']:
                    try:
                        corr = price.corr(merged_df[column])
                        if not np.isnan(corr):
                            correlations[f"price_vs_{column}"] = corr
                    except:
                        continue
            
            logger.info(f"Calculated {len(correlations)} correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    async def generate_comprehensive_analysis(self, days_back: int = 90) -> Dict[str, Any]:
        """Генерировать комплексный анализ"""
        try:
            logger.info("Generating comprehensive trend analysis...")
            
            # Выполнить все анализы
            price_trends = await self.analyze_price_trends(days_back)
            onchain_trends = await self.analyze_onchain_trends(days_back)
            ecosystem_trends = await self.analyze_ecosystem_trends(days_back)
            seasonal_patterns = await self.detect_seasonal_patterns(days_back * 4)  # Больше данных для сезонности
            anomalies = await self.detect_anomalies(days_back)
            correlations = await self.analyze_correlations(days_back)
            
            # Объединить все тренды
            all_trends = price_trends + onchain_trends + ecosystem_trends
            
            # Рассчитать общий тренд
            bullish_trends = len([t for t in all_trends if t.trend_direction == "bullish"])
            bearish_trends = len([t for t in all_trends if t.trend_direction == "bearish"])
            sideways_trends = len([t for t in all_trends if t.trend_direction == "sideways"])
            
            total_trends = len(all_trends)
            if total_trends > 0:
                bullish_percentage = (bullish_trends / total_trends) * 100
                bearish_percentage = (bearish_trends / total_trends) * 100
                sideways_percentage = (sideways_trends / total_trends) * 100
            else:
                bullish_percentage = bearish_percentage = sideways_percentage = 0
            
            # Определить общий тренд
            if bullish_percentage > 60:
                overall_trend = "bullish"
            elif bearish_percentage > 60:
                overall_trend = "bearish"
            else:
                overall_trend = "sideways"
            
            # Создать сводку
            analysis = {
                "analysis_timestamp": datetime.utcnow(),
                "analysis_period_days": days_back,
                "overall_trend": overall_trend,
                "trend_distribution": {
                    "bullish_percentage": bullish_percentage,
                    "bearish_percentage": bearish_percentage,
                    "sideways_percentage": sideways_percentage
                },
                "trends": {
                    "price_trends": [self._trend_to_dict(t) for t in price_trends],
                    "onchain_trends": [self._trend_to_dict(t) for t in onchain_trends],
                    "ecosystem_trends": [self._trend_to_dict(t) for t in ecosystem_trends]
                },
                "patterns": {
                    "seasonal_patterns": [self._pattern_to_dict(p) for p in seasonal_patterns],
                    "anomalies": [self._pattern_to_dict(p) for p in anomalies]
                },
                "correlations": correlations,
                "summary": {
                    "total_metrics_analyzed": total_trends,
                    "patterns_detected": len(seasonal_patterns),
                    "anomalies_found": len(anomalies),
                    "strong_correlations": len([c for c in correlations.values() if abs(c) > 0.7])
                },
                "recommendations": self._generate_overall_recommendations(
                    overall_trend, all_trends, seasonal_patterns, anomalies
                )
            }
            
            self.analysis_results = analysis
            logger.info("Comprehensive analysis completed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return {"error": str(e)}
    
    def _trend_to_dict(self, trend: TrendAnalysis) -> Dict[str, Any]:
        """Преобразовать тренд в словарь"""
        return {
            "metric_name": trend.metric_name,
            "trend_direction": trend.trend_direction,
            "trend_strength": trend.trend_strength,
            "change_percent": trend.change_percent,
            "period": trend.period,
            "confidence": trend.confidence,
            "pattern_type": trend.pattern_type,
            "recommendation": trend.recommendation
        }
    
    def _pattern_to_dict(self, pattern: PatternDetection) -> Dict[str, Any]:
        """Преобразовать паттерн в словарь"""
        return {
            "pattern_name": pattern.pattern_name,
            "pattern_type": pattern.pattern_type,
            "strength": pattern.strength,
            "frequency": pattern.frequency,
            "description": pattern.description,
            "impact_on_price": pattern.impact_on_price
        }
    
    def _generate_overall_recommendations(
        self, 
        overall_trend: str, 
        trends: List[TrendAnalysis],
        patterns: List[PatternDetection],
        anomalies: List[PatternDetection]
    ) -> List[str]:
        """Сгенерировать общие рекомендации"""
        recommendations = []
        
        # Рекомендации на основе общего тренда
        if overall_trend == "bullish":
            recommendations.append("Overall bullish trend detected. Consider holding or accumulating MATIC.")
        elif overall_trend == "bearish":
            recommendations.append("Overall bearish trend detected. Consider reducing position or waiting for better entry.")
        else:
            recommendations.append("Mixed signals detected. Wait for clearer directional movement.")
        
        # Рекомендации на основе аномалий
        if anomalies:
            high_impact_anomalies = [a for a in anomalies if a.strength > 0.7]
            if high_impact_anomalies:
                recommendations.append(f"High-impact anomalies detected. Monitor for potential price movements.")
        
        # Рекомендации на основе паттернов
        if patterns:
            seasonal_patterns = [p for p in patterns if p.pattern_type == "seasonal"]
            if seasonal_patterns:
                recommendations.append("Seasonal patterns detected. Consider timing entries/exits based on historical patterns.")
        
        # Рекомендации на основе силы трендов
        strong_trends = [t for t in trends if t.trend_strength > 0.7]
        if strong_trends:
            recommendations.append(f"{len(strong_trends)} strong trends detected. High confidence in directional movement.")
        
        return recommendations

async def main():
    """Основная функция"""
    logger.info("🚀 Starting Polygon trend analysis...")
    
    try:
        # Инициализировать базу данных
        init_db()
        
        async with PolygonTrendAnalyzer() as analyzer:
            # Настроить актив Polygon
            await analyzer.setup_polygon_asset()
            
            # Выполнить комплексный анализ
            analysis = await analyzer.generate_comprehensive_analysis(days_back=90)
            
            # Вывести результаты
            logger.info("🎯 TREND ANALYSIS RESULTS:")
            logger.info(f"Overall Trend: {analysis['overall_trend'].upper()}")
            logger.info(f"Trend Distribution: {analysis['trend_distribution']['bullish_percentage']:.1f}% bullish, {analysis['trend_distribution']['bearish_percentage']:.1f}% bearish, {analysis['trend_distribution']['sideways_percentage']:.1f}% sideways")
            
            logger.info("📊 KEY TRENDS:")
            for trend in analysis['trends']['price_trends'][:5]:  # Показать топ-5 ценовых трендов
                logger.info(f"  {trend['metric_name']}: {trend['trend_direction']} ({trend['change_percent']:.1f}%)")
            
            logger.info("🔍 PATTERNS DETECTED:")
            for pattern in analysis['patterns']['seasonal_patterns']:
                logger.info(f"  {pattern['pattern_name']}: {pattern['description']}")
            
            logger.info("⚠️ ANOMALIES FOUND:")
            for anomaly in analysis['patterns']['anomalies'][:3]:  # Показать топ-3 аномалии
                logger.info(f"  {anomaly['pattern_name']}: {anomaly['description']}")
            
            logger.info("💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                logger.info(f"  • {rec}")
            
            # Сохранить результаты
            with open("polygon_trend_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info("📁 Analysis saved to polygon_trend_analysis.json")
        
        logger.info("🎉 Polygon trend analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in main process: {e}")

if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск
    asyncio.run(main())
