#!/usr/bin/env python3
"""
Комплексный сбор инвестиционных метрик для принятия решений по MATIC
Собирает данные из всех доступных источников и сохраняет в БД
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import SessionLocal, init_db
from database.models_v2 import (
    Blockchain, CryptoAsset, OnChainMetrics, FinancialMetrics, 
    GitHubMetrics, TokenomicsMetrics, SecurityMetrics, 
    CommunityMetrics, NetworkMetrics, TrendingMetrics, MLPrediction
)
from api.coingecko_client import CoinGeckoClient
from api.quicknode_client import QuickNodeClient
from api.github_client import GitHubClient
from api.metrics_mapper import MetricsMapper
from ml.ml_pipeline import CryptoMLPipeline
from config.settings import settings

class InvestmentMetricsCollector:
    """Сборщик инвестиционных метрик для принятия решений"""
    
    def __init__(self):
        self.session = None
        self.metrics_mapper = MetricsMapper()
        self.ml_pipeline = None
        
    async def __aenter__(self):
        self.session = SessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def collect_all_metrics_for_asset(self, asset_id: int) -> Dict[str, Any]:
        """Собрать все метрики для конкретного актива"""
        try:
            # Получить актив из БД
            asset = self.session.query(CryptoAsset).filter(CryptoAsset.id == asset_id).first()
            if not asset:
                logger.error(f"Asset with ID {asset_id} not found")
                return {}
            
            logger.info(f"Collecting metrics for {asset.symbol} ({asset.name})")
            
            results = {
                "asset_id": asset_id,
                "symbol": asset.symbol,
                "timestamp": datetime.utcnow(),
                "metrics": {}
            }
            
            # 1. Финансовые метрики (CoinGecko)
            if asset.coingecko_id:
                financial_metrics = await self._collect_financial_metrics(asset)
                results["metrics"]["financial"] = financial_metrics
            
            # 2. On-chain метрики (QuickNode)
            if asset.contract_address:
                onchain_metrics = await self._collect_onchain_metrics(asset)
                results["metrics"]["onchain"] = onchain_metrics
            
            # 3. GitHub метрики
            if asset.github_repo:
                github_metrics = await self._collect_github_metrics(asset)
                results["metrics"]["github"] = github_metrics
            
            # 4. Токеномика
            tokenomics_metrics = await self._collect_tokenomics_metrics(asset)
            results["metrics"]["tokenomics"] = tokenomics_metrics
            
            # 5. Метрики безопасности
            security_metrics = await self._collect_security_metrics(asset)
            results["metrics"]["security"] = security_metrics
            
            # 6. Метрики сообщества
            community_metrics = await self._collect_community_metrics(asset)
            results["metrics"]["community"] = community_metrics
            
            # 7. Трендовые метрики
            trending_metrics = await self._collect_trending_metrics(asset)
            results["metrics"]["trending"] = trending_metrics
            
            # 8. ML предсказания
            ml_predictions = await self._generate_ml_predictions(asset_id)
            results["metrics"]["ml_predictions"] = ml_predictions
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting metrics for asset {asset_id}: {e}")
            return {}
    
    async def _collect_financial_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать финансовые метрики из CoinGecko"""
        try:
            async with CoinGeckoClient() as cg_client:
                # Получить детальные данные по монете
                coin_data = await cg_client.get_coin_data(asset.coingecko_id)
                
                if not coin_data:
                    return {}
                
                market_data = coin_data.get("market_data", {})
                
                # Извлечь ключевые финансовые метрики
                financial_metrics = {
                    "price_usd": market_data.get("current_price", {}).get("usd", 0),
                    "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                    "fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get("usd", 0),
                    "volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                    "volume_7d": 0,  # Нужно вычислить
                    "volatility_24h": 0,  # Нужно вычислить
                    "volatility_7d": 0,  # Нужно вычислить
                    "volatility_30d": 0,  # Нужно вычислить
                    "price_change_1h": market_data.get("price_change_percentage_1h_in_currency", {}).get("usd", 0),
                    "price_change_24h": market_data.get("price_change_percentage_24h_in_currency", {}).get("usd", 0),
                    "price_change_7d": market_data.get("price_change_percentage_7d_in_currency", {}).get("usd", 0),
                    "price_change_30d": market_data.get("price_change_percentage_30d_in_currency", {}).get("usd", 0),
                    "circulating_supply": market_data.get("circulating_supply", 0),
                    "total_supply": market_data.get("total_supply", 0),
                    "max_supply": market_data.get("max_supply", 0),
                    "bid_ask_spread": 0,  # Нужно получить с бирж
                    "order_book_depth": 0,  # Нужно получить с бирж
                }
                
                # Сохранить в БД
                db_metrics = FinancialMetrics(
                    asset_id=asset.id,
                    **financial_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected financial metrics for {asset.symbol}")
                return financial_metrics
                
        except Exception as e:
            logger.error(f"Error collecting financial metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_onchain_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать on-chain метрики из QuickNode"""
        try:
            async with QuickNodeClient() as qn_client:
                # Получить текущий блок
                current_block = await qn_client.get_block_number()
                blocks_24h_ago = current_block - (24 * 60 * 60 / 2)  # Примерно 24 часа назад
                blocks_7d_ago = current_block - (7 * 24 * 60 * 60 / 2)  # Примерно 7 дней назад
                
                # Получить статистику сети
                network_stats = await qn_client.get_network_stats()
                
                # Получить активные адреса за 24 часа
                active_addresses_24h = await qn_client.get_active_addresses(blocks_24h_ago, current_block)
                
                # Получить объем транзакций за 24 часа
                transaction_volume_24h = await qn_client.get_transaction_volume(blocks_24h_ago, current_block)
                
                # Получить взаимодействия с контрактом
                contract_interactions = await qn_client.get_contract_interactions(
                    asset.contract_address, blocks_24h_ago, current_block
                )
                
                # Получить газ цену
                gas_price = await qn_client.get_gas_price()
                
                onchain_metrics = {
                    "tvl": 0,  # Нужно вычислить из DeFi протоколов
                    "tvl_change_24h": 0,
                    "tvl_change_7d": 0,
                    "daily_transactions": len(contract_interactions),
                    "transaction_volume_24h": transaction_volume_24h,
                    "avg_transaction_fee": gas_price * 21000 / 10**18,  # Примерная комиссия
                    "active_addresses_24h": active_addresses_24h,
                    "new_addresses_24h": 0,  # Нужно вычислить
                    "unique_users_7d": 0,  # Нужно вычислить
                    "new_contracts_deployed": 0,  # Нужно вычислить
                    "contract_interactions_24h": len(contract_interactions),
                    "block_time_avg": 2.0,  # Polygon блок время
                    "gas_price_avg": gas_price,
                    "network_utilization": 0.0,  # Нужно вычислить
                }
                
                # Сохранить в БД
                db_metrics = OnChainMetrics(
                    asset_id=asset.id,
                    **onchain_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected on-chain metrics for {asset.symbol}")
                return onchain_metrics
                
        except Exception as e:
            logger.error(f"Error collecting on-chain metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_github_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать GitHub метрики"""
        try:
            async with GitHubClient() as gh_client:
                # Получить метрики репозитория
                github_data = await gh_client.get_github_metrics(asset.github_repo)
                
                if not github_data:
                    return {}
                
                activity_metrics = github_data.get("activity_metrics", {})
                development_metrics = github_data.get("development_metrics", {})
                community_metrics = github_data.get("community_metrics", {})
                
                github_metrics = {
                    "commits_24h": activity_metrics.get("commits_24h", 0),
                    "commits_7d": activity_metrics.get("commits_7d", 0),
                    "commits_30d": activity_metrics.get("commits_30d", 0),
                    "open_prs": development_metrics.get("open_prs", 0),
                    "merged_prs_7d": development_metrics.get("merged_prs_7d", 0),
                    "closed_prs_7d": development_metrics.get("closed_prs_7d", 0),
                    "open_issues": development_metrics.get("open_issues", 0),
                    "closed_issues_7d": development_metrics.get("closed_issues_7d", 0),
                    "active_contributors_30d": activity_metrics.get("active_contributors_30d", 0),
                    "total_contributors": activity_metrics.get("total_contributors", 0),
                    "stars": activity_metrics.get("stars", 0),
                    "forks": activity_metrics.get("forks", 0),
                    "stars_change_7d": community_metrics.get("stars_change_7d", 0),
                    "primary_language": community_metrics.get("primary_language", ""),
                    "languages_distribution": community_metrics.get("languages_distribution", {}),
                    "code_quality_score": development_metrics.get("code_quality_score", 0),
                }
                
                # Сохранить в БД
                db_metrics = GitHubMetrics(
                    asset_id=asset.id,
                    **github_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected GitHub metrics for {asset.symbol}")
                return github_metrics
                
        except Exception as e:
            logger.error(f"Error collecting GitHub metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_tokenomics_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать токеномические метрики"""
        try:
            # Получить данные из CoinGecko
            async with CoinGeckoClient() as cg_client:
                coin_data = await cg_client.get_coin_data(asset.coingecko_id)
                
                if not coin_data:
                    return {}
                
                market_data = coin_data.get("market_data", {})
                
                circulating_supply = market_data.get("circulating_supply", 0)
                total_supply = market_data.get("total_supply", 0)
                max_supply = market_data.get("max_supply", 0)
                
                # Вычислить инфляцию
                inflation_rate = 0
                if total_supply and circulating_supply:
                    inflation_rate = ((total_supply - circulating_supply) / total_supply) * 100
                
                tokenomics_metrics = {
                    "circulating_supply": circulating_supply,
                    "total_supply": total_supply,
                    "max_supply": max_supply,
                    "inflation_rate": inflation_rate,
                    "burn_rate": 0,  # Нужно вычислить из сжигания токенов
                    "staking_ratio": 0,  # Нужно получить из стейкинга
                    "vesting_schedule": {},  # Нужно получить из контрактов
                    "token_distribution": {},  # Нужно получить из контрактов
                }
                
                # Сохранить в БД
                db_metrics = TokenomicsMetrics(
                    asset_id=asset.id,
                    **tokenomics_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected tokenomics metrics for {asset.symbol}")
                return tokenomics_metrics
                
        except Exception as e:
            logger.error(f"Error collecting tokenomics metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_security_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать метрики безопасности"""
        try:
            # Проверить верификацию контракта
            contract_verified = False
            if asset.contract_address:
                async with QuickNodeClient() as qn_client:
                    contract_code = await qn_client.get_contract_code(asset.contract_address)
                    contract_verified = len(contract_code) > 2  # Больше чем "0x"
            
            security_metrics = {
                "audit_status": "unknown",  # Нужно проверить в базах аудитов
                "audit_score": 0,  # Нужно получить из аудитов
                "contract_verified": contract_verified,
                "vulnerability_score": 0,  # Нужно получить из сканеров
                "multisig_wallets": False,  # Нужно проверить в контрактах
                "timelock_contracts": False,  # Нужно проверить в контрактах
                "proxy_contracts": False,  # Нужно проверить в контрактах
                "upgradeable_contracts": False,  # Нужно проверить в контрактах
            }
            
            # Сохранить в БД
            db_metrics = SecurityMetrics(
                asset_id=asset.id,
                **security_metrics
            )
            self.session.add(db_metrics)
            
            logger.info(f"Collected security metrics for {asset.symbol}")
            return security_metrics
            
        except Exception as e:
            logger.error(f"Error collecting security metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_community_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать метрики сообщества"""
        try:
            # Получить данные из CoinGecko
            async with CoinGeckoClient() as cg_client:
                coin_data = await cg_client.get_coin_data(asset.coingecko_id)
                
                if not coin_data:
                    return {}
                
                community_data = coin_data.get("community_data", {})
                
                community_metrics = {
                    "twitter_followers": community_data.get("twitter_followers", 0),
                    "telegram_members": 0,  # Нужно получить из Telegram API
                    "discord_members": 0,  # Нужно получить из Discord API
                    "reddit_subscribers": community_data.get("reddit_subscribers", 0),
                    "reddit_active_users": community_data.get("reddit_accounts_active_48h", 0),
                    "social_engagement_rate": 0,  # Нужно вычислить
                    "sentiment_score": 0,  # Нужно получить из анализа соцсетей
                    "influence_score": 0,  # Нужно вычислить
                }
                
                # Сохранить в БД
                db_metrics = CommunityMetrics(
                    asset_id=asset.id,
                    **community_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected community metrics for {asset.symbol}")
                return community_metrics
                
        except Exception as e:
            logger.error(f"Error collecting community metrics for {asset.symbol}: {e}")
            return {}
    
    async def _collect_trending_metrics(self, asset: CryptoAsset) -> Dict[str, Any]:
        """Собрать трендовые метрики"""
        try:
            # Получить данные из CoinGecko
            async with CoinGeckoClient() as cg_client:
                # Получить исторические данные для вычисления импульса
                historical_data = await cg_client.get_coin_market_chart(asset.coingecko_id, days=30)
                
                if not historical_data:
                    return {}
                
                prices = historical_data.get("prices", [])
                
                # Вычислить импульс
                momentum_score = 0
                if len(prices) >= 7:
                    current_price = prices[-1][1]
                    week_ago_price = prices[-7][1]
                    momentum_score = ((current_price - week_ago_price) / week_ago_price) * 100
                
                # Определить направление тренда
                trend_direction = "sideways"
                if momentum_score > 5:
                    trend_direction = "bullish"
                elif momentum_score < -5:
                    trend_direction = "bearish"
                
                trending_metrics = {
                    "momentum_score": momentum_score,
                    "trend_direction": trend_direction,
                    "fear_greed_index": 50,  # Нужно получить из внешнего API
                    "social_sentiment": 0,  # Нужно получить из анализа соцсетей
                    "search_volume": 0,  # Нужно получить из Google Trends
                    "news_sentiment": 0,  # Нужно получить из анализа новостей
                    "whale_activity": 0,  # Нужно вычислить из больших транзакций
                }
                
                # Сохранить в БД
                db_metrics = TrendingMetrics(
                    asset_id=asset.id,
                    **trending_metrics
                )
                self.session.add(db_metrics)
                
                logger.info(f"Collected trending metrics for {asset.symbol}")
                return trending_metrics
                
        except Exception as e:
            logger.error(f"Error collecting trending metrics for {asset.symbol}: {e}")
            return {}
    
    async def _generate_ml_predictions(self, asset_id: int) -> Dict[str, Any]:
        """Генерировать ML предсказания"""
        try:
            if not self.ml_pipeline:
                self.ml_pipeline = CryptoMLPipeline()
                await self.ml_pipeline.__aenter__()
            
            # Получить предсказание инвестиционного скора
            prediction_result = await self.ml_pipeline.predict_investment_score(asset_id)
            
            ml_predictions = {
                "investment_score": prediction_result.prediction_value,
                "confidence_score": prediction_result.confidence_score,
                "model_name": prediction_result.model_name,
                "prediction_horizon": prediction_result.prediction_horizon,
                "feature_importance": prediction_result.feature_importance,
                "created_at": prediction_result.created_at,
            }
            
            # Сохранить в БД
            db_prediction = MLPrediction(
                asset_id=asset_id,
                model_name=prediction_result.model_name,
                prediction_type="investment_score",
                prediction_value=prediction_result.prediction_value,
                confidence_score=prediction_result.confidence_score,
                prediction_horizon=prediction_result.prediction_horizon,
                features_used=list(prediction_result.feature_importance.keys()),
                model_version="1.0"
            )
            self.session.add(db_prediction)
            
            logger.info(f"Generated ML predictions for asset {asset_id}")
            return ml_predictions
            
        except Exception as e:
            logger.error(f"Error generating ML predictions for asset {asset_id}: {e}")
            return {}
    
    async def save_metrics_to_database(self):
        """Сохранить все метрики в базу данных"""
        try:
            self.session.commit()
            logger.info("All metrics saved to database successfully")
        except Exception as e:
            logger.error(f"Error saving metrics to database: {e}")
            self.session.rollback()
    
    async def collect_metrics_for_all_assets(self) -> Dict[str, Any]:
        """Собрать метрики для всех активов"""
        try:
            # Получить все активные активы
            assets = self.session.query(CryptoAsset).filter(CryptoAsset.is_active == True).all()
            
            results = {
                "total_assets": len(assets),
                "successful_collections": 0,
                "failed_collections": 0,
                "collections": []
            }
            
            for asset in assets:
                try:
                    collection_result = await self.collect_all_metrics_for_asset(asset.id)
                    if collection_result:
                        results["successful_collections"] += 1
                        results["collections"].append(collection_result)
                    else:
                        results["failed_collections"] += 1
                    
                    # Небольшая задержка между активами
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error collecting metrics for {asset.symbol}: {e}")
                    results["failed_collections"] += 1
                    continue
            
            # Сохранить все метрики
            await self.save_metrics_to_database()
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting metrics for all assets: {e}")
            return {}
    
    async def collect_metrics_for_matic(self) -> Dict[str, Any]:
        """Собрать метрики специально для MATIC"""
        try:
            # Найти MATIC в базе данных
            matic_asset = self.session.query(CryptoAsset).filter(
                CryptoAsset.symbol == "MATIC"
            ).first()
            
            if not matic_asset:
                logger.error("MATIC asset not found in database")
                return {}
            
            logger.info(f"Collecting comprehensive metrics for MATIC (ID: {matic_asset.id})")
            
            # Собрать все метрики
            matic_metrics = await self.collect_all_metrics_for_asset(matic_asset.id)
            
            # Сохранить в базу данных
            await self.save_metrics_to_database()
            
            # Создать инвестиционный отчет
            investment_report = await self._generate_investment_report(matic_asset.id, matic_metrics)
            
            return {
                "asset": {
                    "id": matic_asset.id,
                    "symbol": matic_asset.symbol,
                    "name": matic_asset.name
                },
                "metrics": matic_metrics,
                "investment_report": investment_report,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics for MATIC: {e}")
            return {}
    
    async def _generate_investment_report(self, asset_id: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Генерировать инвестиционный отчет"""
        try:
            financial = metrics.get("metrics", {}).get("financial", {})
            onchain = metrics.get("metrics", {}).get("onchain", {})
            github = metrics.get("metrics", {}).get("github", {})
            security = metrics.get("metrics", {}).get("security", {})
            ml_predictions = metrics.get("metrics", {}).get("ml_predictions", {})
            
            # Вычислить инвестиционный скоринг
            investment_score = 0
            risk_score = 0
            
            # Финансовые факторы (40%)
            if financial:
                price_change_7d = financial.get("price_change_7d", 0)
                volume_24h = financial.get("volume_24h", 0)
                market_cap = financial.get("market_cap", 0)
                
                if price_change_7d > 10:
                    investment_score += 20
                elif price_change_7d > 0:
                    investment_score += 10
                
                if volume_24h > market_cap * 0.05:  # Высокая ликвидность
                    investment_score += 20
            
            # On-chain факторы (30%)
            if onchain:
                active_addresses = onchain.get("active_addresses_24h", 0)
                daily_transactions = onchain.get("daily_transactions", 0)
                
                if active_addresses > 100000:
                    investment_score += 15
                elif active_addresses > 50000:
                    investment_score += 10
                
                if daily_transactions > 1000000:
                    investment_score += 15
                elif daily_transactions > 500000:
                    investment_score += 10
            
            # Разработка (20%)
            if github:
                commits_30d = github.get("commits_30d", 0)
                active_contributors = github.get("active_contributors_30d", 0)
                
                if commits_30d > 100:
                    investment_score += 10
                elif commits_30d > 50:
                    investment_score += 5
                
                if active_contributors > 10:
                    investment_score += 10
                elif active_contributors > 5:
                    investment_score += 5
            
            # Безопасность (10%)
            if security:
                if security.get("contract_verified", False):
                    investment_score += 5
                
                if security.get("audit_status") == "audited":
                    investment_score += 5
            
            # ML предсказания
            if ml_predictions:
                ml_score = ml_predictions.get("investment_score", 0)
                confidence = ml_predictions.get("confidence_score", 0)
                
                if confidence > 0.8:
                    investment_score += ml_score * 20
            
            # Вычислить риск
            if financial:
                volatility = financial.get("volatility_24h", 0)
                if volatility > 10:
                    risk_score += 30
                elif volatility > 5:
                    risk_score += 20
                else:
                    risk_score += 10
            
            if not security.get("contract_verified", False):
                risk_score += 20
            
            if github and github.get("commits_30d", 0) < 10:
                risk_score += 15
            
            # Нормализовать скоринг
            investment_score = min(investment_score, 100)
            risk_score = min(risk_score, 100)
            
            # Определить рекомендацию
            if investment_score >= 70 and risk_score <= 30:
                recommendation = "STRONG_BUY"
            elif investment_score >= 60 and risk_score <= 40:
                recommendation = "BUY"
            elif investment_score >= 50 and risk_score <= 50:
                recommendation = "HOLD"
            elif investment_score >= 40:
                recommendation = "WEAK_HOLD"
            else:
                recommendation = "SELL"
            
            return {
                "investment_score": investment_score,
                "risk_score": risk_score,
                "recommendation": recommendation,
                "confidence": ml_predictions.get("confidence_score", 0.5),
                "key_factors": {
                    "financial_health": "good" if financial.get("price_change_7d", 0) > 0 else "poor",
                    "development_activity": "high" if github.get("commits_30d", 0) > 50 else "low",
                    "security_status": "verified" if security.get("contract_verified", False) else "unverified",
                    "community_engagement": "active" if onchain.get("active_addresses_24h", 0) > 50000 else "inactive"
                },
                "price_targets": {
                    "conservative": financial.get("price_usd", 0) * 1.2,
                    "moderate": financial.get("price_usd", 0) * 1.5,
                    "optimistic": financial.get("price_usd", 0) * 2.0
                },
                "time_horizons": {
                    "short_term": "1-3 months",
                    "medium_term": "3-12 months", 
                    "long_term": "1-3 years"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating investment report: {e}")
            return {}

async def main():
    """Основная функция"""
    logger.info("🚀 Starting comprehensive investment metrics collection...")
    
    try:
        # Инициализировать базу данных
        init_db()
        logger.info("Database initialized")
        
        # Создать сборщик метрик
        async with InvestmentMetricsCollector() as collector:
            # Собрать метрики для MATIC
            matic_results = await collector.collect_metrics_for_matic()
            
            if matic_results:
                logger.info("✅ Successfully collected MATIC investment metrics")
                
                # Вывести краткий отчет
                investment_report = matic_results.get("investment_report", {})
                logger.info(f"📊 Investment Score: {investment_report.get('investment_score', 0)}/100")
                logger.info(f"⚠️ Risk Score: {investment_report.get('risk_score', 0)}/100")
                logger.info(f"💡 Recommendation: {investment_report.get('recommendation', 'UNKNOWN')}")
                
                # Сохранить результаты в файл
                import json
                with open("matic_investment_analysis.json", "w") as f:
                    json.dump(matic_results, f, indent=2, default=str)
                
                logger.info("📄 Results saved to matic_investment_analysis.json")
            else:
                logger.error("❌ Failed to collect MATIC metrics")
        
        logger.info("🎉 Investment metrics collection completed!")
        
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
