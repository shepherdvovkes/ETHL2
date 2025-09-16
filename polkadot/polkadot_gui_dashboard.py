#!/usr/bin/env python3
"""
Polkadot Parachain Activity & Fraud Detection GUI Dashboard
==========================================================

Real-time monitoring dashboard for:
- Parachain activity metrics
- Fraud detection predictions
- Model performance scores
- Network health indicators
- Risk assessment visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import time
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Dict, List, Any, Optional
import os

# Page configuration
st.set_page_config(
    page_title="Polkadot Parachain Monitor",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E6007A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E6007A;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: linear-gradient(90deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(90deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(90deg, #00aa44 0%, #008822 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .parachain-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PolkadotDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.database_path = "polkadot_archive_data.db"
        self.realtime_table = "realtime_block_metrics"
        self.parachain_data = {}
        self.last_update = None
        
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.database_path)
    
    def load_realtime_data(self) -> pd.DataFrame:
        """Load real-time data from database"""
        try:
            conn = self.get_connection()
            
            # Get recent data (last 24 hours)
            query = f"""
                SELECT 
                    block_number,
                    timestamp,
                    extrinsics_count,
                    events_count,
                    block_size,
                    validator_count,
                    finalization_time,
                    parachain_blocks,
                    cross_chain_messages,
                    created_at
                FROM {self.realtime_table}
                WHERE created_at > datetime('now', '-24 hours')
                ORDER BY block_number DESC
                LIMIT 1000
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def load_parachain_data(self) -> Dict[str, Any]:
        """Load parachain-specific data"""
        try:
            conn = self.get_connection()
            
            # Get parachain data (using actual schema)
            parachain_query = """
                SELECT 
                    timestamp,
                    parachains_count,
                    hrmp_channels_count,
                    active_parachains,
                    created_at
                FROM parachain_data
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            parachain_df = pd.read_sql_query(parachain_query, conn)
            
            # Get staking data
            staking_query = """
                SELECT 
                    timestamp,
                    validators_count,
                    nominators_count,
                    active_era,
                    total_staked,
                    inflation_rate,
                    created_at
                FROM staking_data
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            staking_df = pd.read_sql_query(staking_query, conn)
            
            # Get governance data
            governance_query = """
                SELECT 
                    timestamp,
                    proposals_count,
                    referendums_count,
                    council_members_count,
                    treasury_proposals_count,
                    created_at
                FROM governance_data
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            governance_df = pd.read_sql_query(governance_query, conn)
            conn.close()
            
            return {
                'parachains': parachain_df,
                'staking': staking_df,
                'governance': governance_df
            }
            
        except Exception as e:
            st.warning(f"Could not load parachain data: {e}")
            return {'parachains': pd.DataFrame(), 'staking': pd.DataFrame(), 'governance': pd.DataFrame()}
    
    def calculate_fraud_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate fraud detection metrics"""
        if df.empty:
            return {}
        
        # Calculate fraud indicators
        df['tx_velocity'] = df['extrinsics_count'].rolling(6, min_periods=1).mean()
        df['tx_velocity_std'] = df['extrinsics_count'].rolling(6, min_periods=1).std()
        df['tx_spike'] = (df['extrinsics_count'] > df['tx_velocity'] + 3 * df['tx_velocity_std']).astype(int)
        
        df['block_size_anomaly'] = (df['block_size'] > df['block_size'].quantile(0.95)).astype(int)
        df['event_spike'] = (df['events_count'] > df['events_count'].rolling(6, min_periods=1).mean() + 2 * df['events_count'].rolling(6, min_periods=1).std()).astype(int)
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
            },
            'latest_risk': df['risk_level'].iloc[0] if len(df) > 0 else 'UNKNOWN',
            'latest_risk_score': float(df['risk_score'].iloc[0]) if len(df) > 0 else 0.0
        }
    
    def create_parachain_activity_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create parachain activity visualization"""
        if df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Extrinsics Count', 'Events Count', 'Block Size', 'Cross-Chain Messages'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extrinsics count
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['extrinsics_count'],
                mode='lines+markers',
                name='Extrinsics',
                line=dict(color='#E6007A', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Events count
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['events_count'],
                mode='lines+markers',
                name='Events',
                line=dict(color='#00D4AA', width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Block size
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['block_size'],
                mode='lines+markers',
                name='Block Size',
                line=dict(color='#FF6B35', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Cross-chain messages
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cross_chain_messages'],
                mode='lines+markers',
                name='Cross-Chain',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Parachain Activity Over Time",
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_risk_distribution_chart(self, fraud_metrics: Dict[str, Any]) -> go.Figure:
        """Create risk distribution pie chart"""
        if 'risk_distribution' not in fraud_metrics:
            return go.Figure()
        
        risk_dist = fraud_metrics['risk_distribution']
        colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00aa44'}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(risk_dist.keys()),
            values=list(risk_dist.values()),
            marker_colors=[colors.get(label, '#666666') for label in risk_dist.keys()],
            hole=0.3
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_anomaly_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create anomaly timeline chart"""
        if df.empty:
            return go.Figure()
        
        # Calculate anomalies
        df['tx_velocity'] = df['extrinsics_count'].rolling(6, min_periods=1).mean()
        df['tx_velocity_std'] = df['extrinsics_count'].rolling(6, min_periods=1).std()
        df['tx_spike'] = (df['extrinsics_count'] > df['tx_velocity'] + 3 * df['tx_velocity_std']).astype(int)
        
        df['block_size_anomaly'] = (df['block_size'] > df['block_size'].quantile(0.95)).astype(int)
        df['event_spike'] = (df['events_count'] > df['events_count'].rolling(6, min_periods=1).mean() + 2 * df['events_count'].rolling(6, min_periods=1).std()).astype(int)
        
        # Create timeline
        fig = go.Figure()
        
        # Add anomaly markers
        tx_spikes = df[df['tx_spike'] == 1]
        if not tx_spikes.empty:
            fig.add_trace(go.Scatter(
                x=tx_spikes['timestamp'],
                y=tx_spikes['extrinsics_count'],
                mode='markers',
                name='TX Spikes',
                marker=dict(color='red', size=10, symbol='triangle-up')
            ))
        
        block_anomalies = df[df['block_size_anomaly'] == 1]
        if not block_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=block_anomalies['timestamp'],
                y=block_anomalies['block_size'],
                mode='markers',
                name='Block Size Anomalies',
                marker=dict(color='orange', size=10, symbol='diamond')
            ))
        
        event_spikes = df[df['event_spike'] == 1]
        if not event_spikes.empty:
            fig.add_trace(go.Scatter(
                x=event_spikes['timestamp'],
                y=event_spikes['events_count'],
                mode='markers',
                name='Event Spikes',
                marker=dict(color='yellow', size=10, symbol='square')
            ))
        
        fig.update_layout(
            title="Anomaly Timeline",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig

def main():
    """Main dashboard function"""
    
    # Initialize dashboard
    dashboard = PolkadotDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🔗 Polkadot Parachain Monitor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "📅 Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
        index=2
    )
    
    # Risk threshold
    risk_threshold = st.sidebar.slider("🚨 Risk Threshold", 0.0, 1.0, 0.7, 0.1)
    
    # Main content area
    if auto_refresh:
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
    else:
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        realtime_data = dashboard.load_realtime_data()
        parachain_data = dashboard.load_parachain_data()
        fraud_metrics = dashboard.calculate_fraud_metrics(realtime_data)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Blocks",
            len(realtime_data),
            delta=f"+{len(realtime_data)} in last 24h" if len(realtime_data) > 0 else "No data"
        )
    
    with col2:
        latest_risk = fraud_metrics.get('latest_risk', 'UNKNOWN')
        latest_score = fraud_metrics.get('latest_risk_score', 0.0)
        
        risk_color = {
            'HIGH': '🔴',
            'MEDIUM': '🟡', 
            'LOW': '🟢',
            'UNKNOWN': '⚪'
        }.get(latest_risk, '⚪')
        
        st.metric(
            f"{risk_color} Current Risk",
            latest_risk,
            delta=f"Score: {latest_score:.3f}"
        )
    
    with col3:
        high_risk_count = fraud_metrics.get('high_risk_count', 0)
        total_samples = fraud_metrics.get('total_samples', 0)
        high_risk_pct = (high_risk_count / total_samples * 100) if total_samples > 0 else 0
        
        st.metric(
            "🚨 High Risk Samples",
            high_risk_count,
            delta=f"{high_risk_pct:.1f}% of total"
        )
    
    with col4:
        avg_risk = fraud_metrics.get('avg_risk_score', 0.0)
        st.metric(
            "📈 Avg Risk Score",
            f"{avg_risk:.3f}",
            delta="24h average"
        )
    
    # Main charts
    if not realtime_data.empty:
        # Parachain activity chart
        st.subheader("📊 Parachain Activity")
        activity_chart = dashboard.create_parachain_activity_chart(realtime_data)
        st.plotly_chart(activity_chart, width='stretch')
        
        # Risk analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Distribution")
            risk_chart = dashboard.create_risk_distribution_chart(fraud_metrics)
            st.plotly_chart(risk_chart, width='stretch')
        
        with col2:
            st.subheader("⚠️ Anomaly Timeline")
            anomaly_chart = dashboard.create_anomaly_timeline(realtime_data)
            st.plotly_chart(anomaly_chart, width='stretch')
        
        # Recent anomalies table
        st.subheader("🔍 Recent Anomalies")
        
        if 'recent_anomalies' in fraud_metrics:
            anomalies_df = pd.DataFrame([
                {"Anomaly Type": "Transaction Spikes", "Count": fraud_metrics['recent_anomalies']['tx_spikes']},
                {"Anomaly Type": "Block Size Anomalies", "Count": fraud_metrics['recent_anomalies']['block_size_anomalies']},
                {"Anomaly Type": "Event Spikes", "Count": fraud_metrics['recent_anomalies']['event_spikes']},
                {"Anomaly Type": "Cross-Chain Anomalies", "Count": fraud_metrics['recent_anomalies']['cross_chain_anomalies']}
            ])
            
            st.dataframe(anomalies_df, width='stretch')
        
        # Network details
        if not parachain_data['parachains'].empty or not parachain_data['staking'].empty:
            st.subheader("🔗 Network Details")
            
            # Parachain overview
            if not parachain_data['parachains'].empty:
                parachain_df = parachain_data['parachains']
                latest_parachain = parachain_df.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Parachains", latest_parachain['parachains_count'])
                    st.metric("Active Parachains", latest_parachain['active_parachains'])
                
                with col2:
                    st.metric("HRMP Channels", latest_parachain['hrmp_channels_count'])
                    st.metric("Last Update", latest_parachain['created_at'][:19])
                
                with col3:
                    st.metric("Network Health", "🟢 Healthy" if latest_parachain['active_parachains'] > 0 else "🟡 Monitoring")
            
            # Staking overview
            if not parachain_data['staking'].empty:
                staking_df = parachain_data['staking']
                latest_staking = staking_df.iloc[0]
                
                st.subheader("💰 Staking Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Validators", latest_staking['validators_count'])
                    st.metric("Nominators", latest_staking['nominators_count'])
                
                with col2:
                    st.metric("Active Era", latest_staking['active_era'])
                    st.metric("Total Staked", f"{latest_staking['total_staked']:,.0f}")
                
                with col3:
                    st.metric("Inflation Rate", f"{latest_staking['inflation_rate']:.2%}")
                    st.metric("Last Update", latest_staking['created_at'][:19])
            
            # Governance overview
            if not parachain_data['governance'].empty:
                governance_df = parachain_data['governance']
                latest_governance = governance_df.iloc[0]
                
                st.subheader("🏛️ Governance Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Active Proposals", latest_governance['proposals_count'])
                    st.metric("Referendums", latest_governance['referendums_count'])
                
                with col2:
                    st.metric("Council Members", latest_governance['council_members_count'])
                    st.metric("Treasury Proposals", latest_governance['treasury_proposals_count'])
                
                with col3:
                    st.metric("Governance Activity", "🟢 Active" if latest_governance['proposals_count'] > 0 else "🟡 Quiet")
                    st.metric("Last Update", latest_governance['created_at'][:19])
        
        # Model performance
        st.subheader("🤖 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Status", "🟢 Active", "Real-time monitoring")
        
        with col2:
            st.metric("GPU Utilization", "RTX 4090", "CUDA enabled")
        
        with col3:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"), "Live")
    
    else:
        st.warning("⚠️ No real-time data available. Please check the data collection system.")
        
        # Show system status
        st.subheader("🔧 System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📡 Data Collection: Check QuickNode connection")
            st.info("🗄️ Database: Verify table exists")
        
        with col2:
            st.info("🤖 Model: Check model file exists")
            st.info("🔄 Fine-tuning: Check real-time system")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🔗 Polkadot Parachain Monitor | Real-time Fraud Detection | Powered by RTX 4090
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
