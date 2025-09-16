#!/usr/bin/env python3
"""
Polkadot Real-Time System Monitor
================================

Monitor the real-time fine-tuning system status and performance.
"""

import asyncio
import sqlite3
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PolkadotRealtimeMonitor:
    """Monitor for the real-time fine-tuning system"""
    
    def __init__(self, database_path: str = "polkadot_archive_data.db"):
        self.database_path = database_path
        self.realtime_table = "realtime_block_metrics"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get latest data
            cursor.execute(f"""
                SELECT COUNT(*) as total_blocks,
                       MAX(block_number) as latest_block,
                       MAX(created_at) as latest_timestamp
                FROM {self.realtime_table}
            """)
            
            result = cursor.fetchone()
            total_blocks, latest_block, latest_timestamp = result
            
            # Get recent activity (last hour)
            cursor.execute(f"""
                SELECT COUNT(*) as recent_blocks
                FROM {self.realtime_table}
                WHERE created_at > datetime('now', '-1 hour')
            """)
            
            recent_blocks = cursor.fetchone()[0]
            
            # Get fraud detection stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_samples,
                    AVG(extrinsics_count) as avg_extrinsics,
                    AVG(events_count) as avg_events,
                    AVG(block_size) as avg_block_size
                FROM {self.realtime_table}
                WHERE created_at > datetime('now', '-24 hours')
            """)
            
            stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_blocks_collected': total_blocks,
                'latest_block': latest_block,
                'latest_timestamp': latest_timestamp,
                'recent_blocks_1h': recent_blocks,
                'avg_extrinsics_24h': stats[1] if stats[1] else 0,
                'avg_events_24h': stats[2] if stats[2] else 0,
                'avg_block_size_24h': stats[3] if stats[3] else 0,
                'system_status': 'RUNNING' if recent_blocks > 0 else 'IDLE'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'system_status': 'ERROR'
            }
    
    def get_fraud_analysis(self) -> Dict[str, Any]:
        """Analyze fraud patterns in recent data"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Get recent data
            query = f"""
                SELECT 
                    block_number,
                    timestamp,
                    extrinsics_count,
                    events_count,
                    block_size,
                    cross_chain_messages,
                    parachain_blocks
                FROM {self.realtime_table}
                WHERE created_at > datetime('now', '-24 hours')
                ORDER BY block_number
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {'no_data': True}
            
            # Calculate fraud indicators
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Transaction velocity spikes
            df['tx_velocity'] = df['extrinsics_count'].rolling(6, min_periods=1).mean()
            df['tx_velocity_std'] = df['extrinsics_count'].rolling(6, min_periods=1).std()
            df['tx_spike'] = (df['extrinsics_count'] > df['tx_velocity'] + 3 * df['tx_velocity_std']).astype(int)
            
            # Block size anomalies
            df['block_size_anomaly'] = (df['block_size'] > df['block_size'].quantile(0.95)).astype(int)
            
            # Event spikes
            df['event_spike'] = (df['events_count'] > df['events_count'].rolling(6, min_periods=1).mean() + 2 * df['events_count'].rolling(6, min_periods=1).std()).astype(int)
            
            # Cross-chain anomalies
            df['cross_chain_anomaly'] = (df['cross_chain_messages'] > df['cross_chain_messages'].quantile(0.9)).astype(int)
            
            # Calculate risk scores
            df['risk_score'] = (
                df['tx_spike'] * 0.3 +
                df['block_size_anomaly'] * 0.2 +
                df['event_spike'] * 0.2 +
                df['cross_chain_anomaly'] * 0.3
            )
            
            # Risk level classification
            df['risk_level'] = 'LOW'
            df.loc[df['risk_score'] > 0.7, 'risk_level'] = 'HIGH'
            df.loc[(df['risk_score'] > 0.4) & (df['risk_score'] <= 0.7), 'risk_level'] = 'MEDIUM'
            
            # Summary statistics
            risk_distribution = df['risk_level'].value_counts().to_dict()
            high_risk_samples = df[df['risk_level'] == 'HIGH']
            
            return {
                'total_samples': len(df),
                'risk_distribution': risk_distribution,
                'high_risk_count': len(high_risk_samples),
                'avg_risk_score': float(df['risk_score'].mean()),
                'max_risk_score': float(df['risk_score'].max()),
                'recent_anomalies': {
                    'tx_spikes': int(df['tx_spike'].sum()),
                    'block_size_anomalies': int(df['block_size_anomaly'].sum()),
                    'event_spikes': int(df['event_spike'].sum()),
                    'cross_chain_anomalies': int(df['cross_chain_anomaly'].sum())
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_monitoring_dashboard(self):
        """Create monitoring dashboard"""
        print("📊 Creating Polkadot Real-Time Monitoring Dashboard...")
        
        # Get system status
        status = self.get_system_status()
        fraud_analysis = self.get_fraud_analysis()
        
        # Create dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'System Status', 'Risk Distribution',
                'Recent Activity', 'Fraud Indicators',
                'Block Metrics', 'Network Health'
            ),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # System status indicator
        if status.get('system_status') == 'RUNNING':
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=status.get('recent_blocks_1h', 0),
                    title={'text': "Blocks Collected (1h)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "green"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=0,
                    title={'text': "System Status"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "red"}}
                ),
                row=1, col=1
            )
        
        # Risk distribution pie chart
        if 'risk_distribution' in fraud_analysis:
            risk_dist = fraud_analysis['risk_distribution']
            fig.add_trace(
                go.Pie(
                    labels=list(risk_dist.keys()),
                    values=list(risk_dist.values()),
                    name="Risk Distribution"
                ),
                row=1, col=2
            )
        
        # Recent activity
        if 'recent_anomalies' in fraud_analysis:
            anomalies = fraud_analysis['recent_anomalies']
            fig.add_trace(
                go.Bar(
                    x=list(anomalies.keys()),
                    y=list(anomalies.values()),
                    name="Recent Anomalies"
                ),
                row=2, col=1
            )
        
        # Fraud indicators
        if 'recent_anomalies' in fraud_analysis:
            anomalies = fraud_analysis['recent_anomalies']
            fig.add_trace(
                go.Bar(
                    x=list(anomalies.keys()),
                    y=list(anomalies.values()),
                    name="Fraud Indicators",
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Polkadot Real-Time Fine-Tuning System Monitor",
            height=900,
            showlegend=False
        )
        
        # Save dashboard
        fig.write_html("polkadot_realtime_dashboard.html")
        print("✅ Dashboard saved to polkadot_realtime_dashboard.html")
        
        return fig
    
    def print_status_report(self):
        """Print status report to console"""
        print("\n" + "="*60)
        print("🚀 POLKADOT REAL-TIME FINE-TUNING SYSTEM STATUS")
        print("="*60)
        
        # System status
        status = self.get_system_status()
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   Status: {status.get('system_status', 'UNKNOWN')}")
        print(f"   Total Blocks Collected: {status.get('total_blocks_collected', 0)}")
        print(f"   Latest Block: {status.get('latest_block', 'N/A')}")
        print(f"   Recent Blocks (1h): {status.get('recent_blocks_1h', 0)}")
        print(f"   Latest Timestamp: {status.get('latest_timestamp', 'N/A')}")
        
        # Fraud analysis
        fraud_analysis = self.get_fraud_analysis()
        if 'error' not in fraud_analysis:
            print(f"\n🔍 FRAUD DETECTION ANALYSIS:")
            print(f"   Total Samples (24h): {fraud_analysis.get('total_samples', 0)}")
            print(f"   High Risk Count: {fraud_analysis.get('high_risk_count', 0)}")
            print(f"   Average Risk Score: {fraud_analysis.get('avg_risk_score', 0):.4f}")
            print(f"   Max Risk Score: {fraud_analysis.get('max_risk_score', 0):.4f}")
            
            if 'risk_distribution' in fraud_analysis:
                print(f"\n📈 RISK DISTRIBUTION:")
                for level, count in fraud_analysis['risk_distribution'].items():
                    print(f"   {level}: {count} samples")
            
            if 'recent_anomalies' in fraud_analysis:
                print(f"\n🚨 RECENT ANOMALIES (24h):")
                for anomaly, count in fraud_analysis['recent_anomalies'].items():
                    print(f"   {anomaly}: {count}")
        else:
            print(f"\n❌ Error in fraud analysis: {fraud_analysis['error']}")
        
        print(f"\n💾 FILES:")
        print(f"   Model: polkadot_simple_comprehensive_model.pth")
        print(f"   Logs: polkadot_realtime_finetuner.log")
        print(f"   Dashboard: polkadot_realtime_dashboard.html")
        
        print("\n" + "="*60)

def main():
    """Main function"""
    monitor = PolkadotRealtimeMonitor()
    
    # Print status report
    monitor.print_status_report()
    
    # Create dashboard
    monitor.create_monitoring_dashboard()
    
    print("\n🎉 Monitoring complete!")

if __name__ == "__main__":
    main()

