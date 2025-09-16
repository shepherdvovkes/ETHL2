#!/usr/bin/env python3
"""
Analyze Astar data quality and completeness for ML training
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_data_quality():
    """Analyze the quality and completeness of collected Astar data"""
    
    print("🔍 Astar Data Quality Analysis")
    print("=" * 50)
    
    # Connect to database
    conn = sqlite3.connect('astar_multithreaded_data.db')
    
    try:
        # 1. Basic Statistics
        print("\n📊 Basic Data Statistics:")
        print("-" * 30)
        
        # Block statistics
        blocks_query = """
        SELECT 
            COUNT(*) as total_blocks,
            MIN(block_number) as min_block,
            MAX(block_number) as max_block,
            AVG(gas_used) as avg_gas_used,
            AVG(transaction_count) as avg_tx_per_block
        FROM astar_blocks
        """
        
        blocks_stats = pd.read_sql_query(blocks_query, conn)
        print(f"Total Blocks: {blocks_stats['total_blocks'].iloc[0]:,}")
        print(f"Block Range: {blocks_stats['min_block'].iloc[0]:,} - {blocks_stats['max_block'].iloc[0]:,}")
        print(f"Average Gas Used: {blocks_stats['avg_gas_used'].iloc[0]:.0f}")
        print(f"Average TX per Block: {blocks_stats['avg_tx_per_block'].iloc[0]:.2f}")
        
        # Transaction statistics
        tx_query = """
        SELECT 
            COUNT(*) as total_transactions,
            COUNT(DISTINCT from_address) as unique_senders,
            COUNT(DISTINCT to_address) as unique_receivers,
            AVG(gas) as avg_tx_gas,
            SUM(CASE WHEN CAST(value AS INTEGER) > 0 THEN 1 ELSE 0 END) as value_transfers
        FROM astar_transactions
        """
        
        tx_stats = pd.read_sql_query(tx_query, conn)
        print(f"\nTotal Transactions: {tx_stats['total_transactions'].iloc[0]:,}")
        print(f"Unique Senders: {tx_stats['unique_senders'].iloc[0]:,}")
        print(f"Unique Receivers: {tx_stats['unique_receivers'].iloc[0]:,}")
        print(f"Average TX Gas: {tx_stats['avg_tx_gas'].iloc[0]:.0f}")
        print(f"Value Transfers: {tx_stats['value_transfers'].iloc[0]:,}")
        
        # 2. Data Completeness Analysis
        print("\n📈 Data Completeness Analysis:")
        print("-" * 35)
        
        # Check for missing blocks
        block_range_query = """
        WITH RECURSIVE block_sequence AS (
            SELECT MIN(block_number) as block_num FROM astar_blocks
            UNION ALL
            SELECT block_num + 1 FROM block_sequence 
            WHERE block_num < (SELECT MAX(block_number) FROM astar_blocks)
        )
        SELECT 
            COUNT(*) as expected_blocks,
            (SELECT COUNT(*) FROM astar_blocks) as actual_blocks,
            COUNT(*) - (SELECT COUNT(*) FROM astar_blocks) as missing_blocks
        FROM block_sequence
        """
        
        completeness = pd.read_sql_query(block_range_query, conn)
        expected = completeness['expected_blocks'].iloc[0]
        actual = completeness['actual_blocks'].iloc[0]
        missing = completeness['missing_blocks'].iloc[0]
        
        print(f"Expected Blocks: {expected:,}")
        print(f"Actual Blocks: {actual:,}")
        print(f"Missing Blocks: {missing:,}")
        print(f"Completeness: {(actual/expected)*100:.2f}%")
        
        # 3. Time Analysis
        print("\n⏰ Time Analysis:")
        print("-" * 20)
        
        time_query = """
        SELECT 
            MIN(timestamp) as earliest_block,
            MAX(timestamp) as latest_block
        FROM astar_blocks
        """
        
        time_stats = pd.read_sql_query(time_query, conn)
        earliest = pd.to_datetime(time_stats['earliest_block'].iloc[0])
        latest = pd.to_datetime(time_stats['latest_block'].iloc[0])
        total_hours = (latest - earliest).total_seconds() / 3600
        blocks_per_hour = actual / total_hours if total_hours > 0 else 0
        
        print(f"Earliest Block: {earliest}")
        print(f"Latest Block: {latest}")
        print(f"Total Time Span: {total_hours:.1f} hours")
        print(f"Blocks per Hour: {blocks_per_hour:.1f}")
        
        # 4. Feature Analysis for ML
        print("\n🤖 ML Feature Analysis:")
        print("-" * 25)
        
        # Network activity features
        activity_query = """
        SELECT 
            AVG(transaction_count) as avg_daily_tx,
            AVG(gas_used) as avg_daily_gas
        FROM astar_blocks
        """
        
        activity_stats = pd.read_sql_query(activity_query, conn)
        print(f"Average Daily Transactions: {activity_stats['avg_daily_tx'].iloc[0]:.1f}")
        print(f"Average Daily Gas Used: {activity_stats['avg_daily_gas'].iloc[0]:.0f}")
        
        # 5. Data Quality Score
        print("\n🎯 Data Quality Assessment:")
        print("-" * 30)
        
        # Calculate quality metrics
        completeness_score = (actual/expected) * 100 if expected > 0 else 0
        time_span_days = total_hours / 24
        data_density = actual / time_span_days if time_span_days > 0 else 0
        
        # Quality scoring
        quality_score = 0
        if completeness_score > 95:
            quality_score += 30
        elif completeness_score > 90:
            quality_score += 20
        elif completeness_score > 80:
            quality_score += 10
            
        if time_span_days > 30:
            quality_score += 25
        elif time_span_days > 14:
            quality_score += 15
        elif time_span_days > 7:
            quality_score += 10
            
        if data_density > 1000:
            quality_score += 25
        elif data_density > 500:
            quality_score += 15
        elif data_density > 100:
            quality_score += 10
            
        if tx_stats['total_transactions'].iloc[0] > 10000:
            quality_score += 20
        elif tx_stats['total_transactions'].iloc[0] > 5000:
            quality_score += 10
        
        print(f"Completeness Score: {completeness_score:.1f}%")
        print(f"Time Span: {time_span_days:.1f} days")
        print(f"Data Density: {data_density:.0f} blocks/day")
        print(f"Overall Quality Score: {quality_score}/100")
        
        # Recommendations
        print("\n💡 Recommendations:")
        print("-" * 20)
        
        if quality_score >= 80:
            print("✅ Excellent data quality! Ready for ML training.")
        elif quality_score >= 60:
            print("✅ Good data quality. Consider collecting more data for better accuracy.")
        elif quality_score >= 40:
            print("⚠️  Moderate data quality. Need more data collection.")
        else:
            print("❌ Poor data quality. Significant data collection needed.")
            
        if completeness_score < 95:
            print(f"📊 Consider filling {missing:,} missing blocks for better completeness.")
            
        if time_span_days < 30:
            print(f"⏰ Collect more historical data (currently {time_span_days:.1f} days).")
            
        if data_density < 1000:
            print(f"📈 Low data density ({data_density:.0f} blocks/day). Consider higher frequency collection.")
        
        # Save analysis results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "total_blocks": int(actual),
            "total_transactions": int(tx_stats['total_transactions'].iloc[0]),
            "block_range": f"{blocks_stats['min_block'].iloc[0]}-{blocks_stats['max_block'].iloc[0]}",
            "time_span_days": float(time_span_days),
            "completeness_score": float(completeness_score),
            "data_density": float(data_density),
            "quality_score": int(quality_score),
            "missing_blocks": int(missing),
            "recommendations": {
                "ready_for_ml": bool(quality_score >= 60),
                "needs_more_data": bool(quality_score < 80),
                "missing_blocks": bool(missing > 0),
                "time_span_sufficient": bool(time_span_days >= 14)
            }
        }
        
        with open('astar_data_quality_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
        print(f"\n💾 Analysis saved to: astar_data_quality_analysis.json")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_data_quality()
