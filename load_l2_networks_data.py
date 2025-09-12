#!/usr/bin/env python3
"""
Скрипт для загрузки данных Layer 2 сетей в базу данных
Интеграция с существующей системой мониторинга
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import our L2 data
from ethereum_l2_networks_complete_list import ETHEREUM_L2_NETWORKS
from l2_networks_detailed_analysis import DETAILED_L2_NETWORKS

# Import models
from src.database.l2_models import (
    L2Network, L2PerformanceMetrics, L2EconomicMetrics, 
    L2SecurityMetrics, L2EcosystemMetrics, L2ComparisonMetrics,
    L2RiskAssessment, L2TrendingMetrics, L2CrossChainMetrics
)
from src.database.models_v2 import Blockchain, Base

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'defimon_db',
    'user': 'defimon',
    'password': 'password'
}

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

def get_sqlalchemy_session():
    """Get SQLAlchemy session"""
    engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    Session = sessionmaker(bind=engine)
    return Session()

async def load_l2_networks():
    """Загрузить основные данные L2 сетей"""
    logger.info("Loading L2 networks data...")
    
    session = get_sqlalchemy_session()
    
    try:
        # Получить Ethereum blockchain ID
        ethereum = session.query(Blockchain).filter_by(name='Ethereum').first()
        if not ethereum:
            logger.error("Ethereum blockchain not found in database")
            return
        
        logger.info(f"Found Ethereum blockchain with ID: {ethereum.id}")
        
        # Загрузить L2 сети
        for l2_data in ETHEREUM_L2_NETWORKS:
            # Проверить, существует ли уже сеть
            existing_network = session.query(L2Network).filter_by(name=l2_data.name).first()
            
            if existing_network:
                logger.info(f"Updating existing L2 network: {l2_data.name}")
                # Обновить существующую сеть
                existing_network.symbol = l2_data.native_token
                existing_network.l2_type = l2_data.type.value
                existing_network.security_model = l2_data.security_model.value
                existing_network.launch_date = datetime.strptime(l2_data.launch_date, '%Y-%m-%d') if l2_data.launch_date else None
                existing_network.website = l2_data.website
                existing_network.description = l2_data.description
                existing_network.status = l2_data.status
                existing_network.updated_at = datetime.utcnow()
            else:
                logger.info(f"Creating new L2 network: {l2_data.name}")
                # Создать новую сеть
                new_network = L2Network(
                    name=l2_data.name,
                    symbol=l2_data.native_token,
                    l2_type=l2_data.type.value,
                    security_model=l2_data.security_model.value,
                    launch_date=datetime.strptime(l2_data.launch_date, '%Y-%m-%d') if l2_data.launch_date else None,
                    website=l2_data.website,
                    description=l2_data.description,
                    status=l2_data.status,
                    parent_blockchain_id=ethereum.id,
                    evm_compatibility=True,  # Большинство L2 совместимы с EVM
                    created_at=datetime.utcnow()
                )
                session.add(new_network)
        
        session.commit()
        logger.info("✅ L2 networks loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading L2 networks: {e}")
        session.rollback()
    finally:
        session.close()

async def load_detailed_l2_metrics():
    """Загрузить детальные метрики L2 сетей"""
    logger.info("Loading detailed L2 metrics...")
    
    session = get_sqlalchemy_session()
    
    try:
        for detailed_network in DETAILED_L2_NETWORKS:
            # Найти L2 сеть в базе
            l2_network = session.query(L2Network).filter_by(name=detailed_network.basic_info.name).first()
            if not l2_network:
                logger.warning(f"L2 network not found: {detailed_network.basic_info.name}")
                continue
            
            timestamp = datetime.utcnow()
            
            # Загрузить технические характеристики
            l2_network.consensus_mechanism = detailed_network.technical_specs.consensus_mechanism
            l2_network.block_time = detailed_network.technical_specs.block_time
            l2_network.gas_limit = detailed_network.technical_specs.gas_limit
            l2_network.evm_compatibility = detailed_network.technical_specs.evm_compatibility
            l2_network.programming_language = detailed_network.technical_specs.programming_language
            l2_network.virtual_machine = detailed_network.technical_specs.virtual_machine
            l2_network.data_availability = detailed_network.technical_specs.data_availability
            l2_network.fraud_proofs = detailed_network.technical_specs.fraud_proofs
            l2_network.zero_knowledge_proofs = detailed_network.technical_specs.zero_knowledge_proofs
            
            # Загрузить метрики производительности
            perf_metrics = L2PerformanceMetrics(
                l2_network_id=l2_network.id,
                timestamp=timestamp,
                transactions_per_second=detailed_network.performance.transactions_per_second,
                finality_time=detailed_network.performance.finality_time,
                withdrawal_time=detailed_network.performance.withdrawal_time,
                gas_fee_reduction=detailed_network.performance.gas_fee_reduction,
                throughput_improvement=detailed_network.performance.throughput_improvement,
                latency=detailed_network.performance.latency,
                daily_transactions=int(detailed_network.economics.daily_volume / 1000) if detailed_network.economics.daily_volume else None,
                active_addresses_24h=detailed_network.economics.active_users_24h,
                new_addresses_24h=int(detailed_network.economics.active_users_24h * 0.1) if detailed_network.economics.active_users_24h else None,
                unique_users_7d=int(detailed_network.economics.active_users_24h * 7) if detailed_network.economics.active_users_24h else None
            )
            session.add(perf_metrics)
            
            # Загрузить экономические метрики
            econ_metrics = L2EconomicMetrics(
                l2_network_id=l2_network.id,
                timestamp=timestamp,
                total_value_locked=detailed_network.economics.total_value_locked,
                daily_volume=detailed_network.economics.daily_volume,
                active_users_24h=detailed_network.economics.active_users_24h,
                transaction_fees_24h=detailed_network.economics.transaction_fees_24h,
                revenue_24h=detailed_network.economics.revenue_24h,
                market_cap=detailed_network.economics.market_cap
            )
            session.add(econ_metrics)
            
            # Загрузить метрики безопасности
            sec_metrics = L2SecurityMetrics(
                l2_network_id=l2_network.id,
                timestamp=timestamp,
                validator_count=detailed_network.security.validator_count,
                slashing_mechanism=detailed_network.security.slashing_mechanism,
                multisig_required=detailed_network.security.multisig_required,
                upgrade_mechanism=detailed_network.security.upgrade_mechanism,
                time_to_finality=detailed_network.security.time_to_finality,
                audit_count=detailed_network.security.audit_count,
                bug_bounty_program=detailed_network.security.bug_bounty_program
            )
            session.add(sec_metrics)
            
            # Загрузить метрики экосистемы
            eco_metrics = L2EcosystemMetrics(
                l2_network_id=l2_network.id,
                timestamp=timestamp,
                defi_protocols_count=detailed_network.ecosystem.get('defi_protocols', 0),
                nft_marketplaces=detailed_network.ecosystem.get('nft_marketplaces', 0),
                games=detailed_network.ecosystem.get('games', 0),
                bridges_count=detailed_network.ecosystem.get('bridges', 0),
                wallets_support=detailed_network.ecosystem.get('wallets', 0)
            )
            session.add(eco_metrics)
            
            # Загрузить оценку рисков
            risk_factors = []
            risk_level = "Low"
            
            if detailed_network.risks:
                for risk in detailed_network.risks:
                    if "централизация" in risk.lower() or "centralization" in risk.lower():
                        risk_factors.append({"type": "centralization", "description": risk, "severity": "High"})
                        risk_level = "Medium"
                    elif "безопасность" in risk.lower() or "security" in risk.lower():
                        risk_factors.append({"type": "security", "description": risk, "severity": "Medium"})
                    elif "ликвидность" in risk.lower() or "liquidity" in risk.lower():
                        risk_factors.append({"type": "liquidity", "description": risk, "severity": "Medium"})
                    elif "технические" in risk.lower() or "technical" in risk.lower():
                        risk_factors.append({"type": "technical", "description": risk, "severity": "High"})
                        risk_level = "High"
            
            risk_assessment = L2RiskAssessment(
                l2_network_id=l2_network.id,
                timestamp=timestamp,
                centralization_risk=8.0 if any("централизация" in risk.lower() for risk in detailed_network.risks) else 3.0,
                security_risk=6.0 if detailed_network.security.audit_count < 3 else 3.0,
                liquidity_risk=7.0 if detailed_network.economics.total_value_locked < 100e6 else 3.0,
                technical_risk=8.0 if not detailed_network.technical_specs.evm_compatibility else 3.0,
                overall_risk_score=6.5,
                risk_level=risk_level,
                risk_factors=risk_factors
            )
            session.add(risk_assessment)
            
            logger.info(f"✅ Loaded detailed metrics for {detailed_network.basic_info.name}")
        
        session.commit()
        logger.info("✅ Detailed L2 metrics loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading detailed L2 metrics: {e}")
        session.rollback()
    finally:
        session.close()

async def load_l2_comparison_metrics():
    """Загрузить сравнительные метрики L2 сетей"""
    logger.info("Loading L2 comparison metrics...")
    
    session = get_sqlalchemy_session()
    
    try:
        # Получить все L2 сети
        l2_networks = session.query(L2Network).all()
        
        # Сортировать по TVL для ранжирования
        networks_with_tvl = []
        for network in l2_networks:
            latest_econ = session.query(L2EconomicMetrics).filter_by(l2_network_id=network.id).order_by(L2EconomicMetrics.timestamp.desc()).first()
            if latest_econ and latest_econ.total_value_locked:
                networks_with_tvl.append((network, latest_econ.total_value_locked))
        
        # Сортировать по TVL
        networks_with_tvl.sort(key=lambda x: x[1], reverse=True)
        
        timestamp = datetime.utcnow()
        
        for rank, (network, tvl) in enumerate(networks_with_tvl, 1):
            # Получить последние метрики
            latest_perf = session.query(L2PerformanceMetrics).filter_by(l2_network_id=network.id).order_by(L2PerformanceMetrics.timestamp.desc()).first()
            latest_econ = session.query(L2EconomicMetrics).filter_by(l2_network_id=network.id).order_by(L2EconomicMetrics.timestamp.desc()).first()
            latest_sec = session.query(L2SecurityMetrics).filter_by(l2_network_id=network.id).order_by(L2SecurityMetrics.timestamp.desc()).first()
            
            # Рассчитать сравнительные метрики
            tps_vs_ethereum = latest_perf.transactions_per_second / 15 if latest_perf and latest_perf.transactions_per_second else 1.0  # Ethereum ~15 TPS
            fee_reduction = latest_perf.gas_fee_reduction if latest_perf and latest_perf.gas_fee_reduction else 0.0
            
            # Рассчитать рыночную долю
            total_tvl = sum(tvl for _, tvl in networks_with_tvl)
            market_share_tvl = (tvl / total_tvl * 100) if total_tvl > 0 else 0.0
            
            comparison_metrics = L2ComparisonMetrics(
                l2_network_id=network.id,
                timestamp=timestamp,
                overall_rank=rank,
                tps_rank=rank,  # Упрощенное ранжирование
                tvl_rank=rank,
                security_rank=rank,
                tps_vs_ethereum=tps_vs_ethereum,
                fee_reduction_vs_ethereum=fee_reduction,
                market_share_tvl=market_share_tvl,
                growth_rate_30d=5.0,  # Оценочное значение
                adoption_rate_30d=3.0,  # Оценочное значение
                innovation_score=7.0,  # Оценочное значение
                risk_score=6.0  # Оценочное значение
            )
            session.add(comparison_metrics)
            
            logger.info(f"✅ Loaded comparison metrics for {network.name} (Rank: {rank})")
        
        session.commit()
        logger.info("✅ L2 comparison metrics loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading L2 comparison metrics: {e}")
        session.rollback()
    finally:
        session.close()

async def load_l2_trending_metrics():
    """Загрузить трендовые метрики L2 сетей"""
    logger.info("Loading L2 trending metrics...")
    
    session = get_sqlalchemy_session()
    
    try:
        l2_networks = session.query(L2Network).all()
        timestamp = datetime.utcnow()
        
        for network in l2_networks:
            # Получить последние экономические метрики
            latest_econ = session.query(L2EconomicMetrics).filter_by(l2_network_id=network.id).order_by(L2EconomicMetrics.timestamp.desc()).first()
            
            # Рассчитать трендовые метрики
            momentum_score = 5.0  # Базовое значение
            trend_direction = "sideways"
            trend_strength = 0.5
            
            if latest_econ:
                if latest_econ.tvl_change_24h and latest_econ.tvl_change_24h > 5:
                    momentum_score = 8.0
                    trend_direction = "bullish"
                    trend_strength = 0.8
                elif latest_econ.tvl_change_24h and latest_econ.tvl_change_24h < -5:
                    momentum_score = 2.0
                    trend_direction = "bearish"
                    trend_strength = 0.8
            
            trending_metrics = L2TrendingMetrics(
                l2_network_id=network.id,
                timestamp=timestamp,
                momentum_score=momentum_score,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonality_score=4.0,
                cyclical_patterns={"daily": 0.1, "weekly": 0.3, "monthly": 0.6},
                anomaly_score=0.0,
                anomaly_type="none",
                anomaly_severity="low",
                fear_greed_index=50.0,
                social_sentiment=0.5,
                news_sentiment=0.5,
                developer_sentiment=0.6,
                user_growth_rate=5.0,
                volume_growth_rate=3.0,
                tvl_growth_rate=4.0,
                adoption_growth_rate=2.0
            )
            session.add(trending_metrics)
            
            logger.info(f"✅ Loaded trending metrics for {network.name}")
        
        session.commit()
        logger.info("✅ L2 trending metrics loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading L2 trending metrics: {e}")
        session.rollback()
    finally:
        session.close()

async def create_l2_database_tables():
    """Создать таблицы для L2 данных"""
    logger.info("Creating L2 database tables...")
    
    try:
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        
        # Create tables for both existing and L2 models
        Base.metadata.create_all(engine)
        logger.info("✅ L2 database tables created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating L2 database tables: {e}")

async def main():
    """Основная функция"""
    logger.info("🚀 Loading L2 networks data...")
    
    try:
        # Создать таблицы
        await create_l2_database_tables()
        
        # Загрузить данные
        await load_l2_networks()
        await load_detailed_l2_metrics()
        await load_l2_comparison_metrics()
        await load_l2_trending_metrics()
        
        logger.info("🎉 L2 networks data loading completed successfully!")
        
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
