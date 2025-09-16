#!/usr/bin/env python3
"""
Avalanche Network Monitoring and Alerting System
Real-time monitoring with advanced alerting capabilities
"""

import asyncio
import json
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from loguru import logger
import os
import sys
from enum import Enum
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from avalanche_realtime_server import RealTimeDataCollector
from config.settings import settings

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of alerts"""
    NETWORK_PERFORMANCE = "network_performance"
    ECONOMIC_METRICS = "economic_metrics"
    SECURITY_ISSUE = "security_issue"
    TECHNICAL_FAILURE = "technical_failure"
    MARKET_ANOMALY = "market_anomaly"
    DEFI_RISK = "defi_risk"
    SUBNET_ISSUE = "subnet_issue"
    API_FAILURE = "api_failure"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metric_name: str
    current_value: Any
    threshold_value: Any
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class MonitoringRule:
    """Monitoring rule configuration"""
    name: str
    metric_type: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: Any
    alert_level: AlertLevel
    alert_type: AlertType
    enabled: bool = True
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = {
            "email": self.send_email_alert,
            "webhook": self.send_webhook_alert,
            "slack": self.send_slack_alert,
            "telegram": self.send_telegram_alert
        }
        
        # Notification settings
        self.email_config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("EMAIL_USERNAME", ""),
            "password": os.getenv("EMAIL_PASSWORD", ""),
            "from_email": os.getenv("FROM_EMAIL", ""),
            "to_emails": os.getenv("TO_EMAILS", "").split(",")
        }
        
        self.webhook_url = os.getenv("WEBHOOK_URL", "")
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    async def create_alert(self, rule: MonitoringRule, current_value: Any, metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            type=rule.alert_type,
            level=rule.alert_level,
            title=f"Avalanche {rule.alert_type.value.replace('_', ' ').title()} Alert",
            message=self._generate_alert_message(rule, current_value),
            timestamp=datetime.utcnow(),
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self.send_notifications(alert)
        
        logger.warning(f"🚨 Alert created: {alert.title} - {alert.message}")
        return alert
    
    def _generate_alert_message(self, rule: MonitoringRule, current_value: Any) -> str:
        """Generate alert message"""
        condition_text = {
            "greater_than": "exceeded",
            "less_than": "dropped below",
            "equals": "equals",
            "not_equals": "does not equal"
        }
        
        return (
            f"Avalanche {rule.metric_name} has {condition_text.get(rule.condition, 'changed')} "
            f"threshold. Current: {current_value}, Threshold: {rule.threshold}"
        )
    
    async def send_notifications(self, alert: Alert):
        """Send notifications through all configured channels"""
        for channel, send_func in self.notification_channels.items():
            try:
                await send_func(alert)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
    
    async def send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.email_config["username"] or not self.email_config["password"]:
            logger.warning("Email configuration not set, skipping email alert")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(self.email_config["to_emails"])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Avalanche Network Alert
            
            Alert Type: {alert.type.value.replace('_', ' ').title()}
            Severity: {alert.level.value.upper()}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Message: {alert.message}
            
            Current Value: {alert.current_value}
            Threshold: {alert.threshold}
            
            Please check the Avalanche network status immediately.
            
            Best regards,
            Avalanche Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            text = msg.as_string()
            server.sendmail(self.email_config["from_email"], self.email_config["to_emails"], text)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.id}")
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        if not self.webhook_url:
            logger.warning("Webhook URL not configured, skipping webhook alert")
            return
        
        try:
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "metadata": alert.metadata
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.id}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured, skipping Slack alert")
            return
        
        try:
            color = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ffaa00",
                AlertLevel.CRITICAL: "#ff6600",
                AlertLevel.EMERGENCY: "#ff0000"
            }.get(alert.level, "#36a64f")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Alert Type",
                                "value": alert.type.value.replace('_', ' ').title(),
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.level.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": str(alert.current_value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold_value),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": False
                            }
                        ],
                        "footer": "Avalanche Monitoring System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.id}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram configuration not set, skipping Telegram alert")
            return
        
        try:
            emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🔥"
            }.get(alert.level, "ℹ️")
            
            message = f"""
{emoji} *{alert.title}*

*Alert Type:* {alert.type.value.replace('_', ' ').title()}
*Severity:* {alert.level.value.upper()}
*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

*Message:* {alert.message}

*Current Value:* {alert.current_value}
*Threshold:* {alert.threshold_value}

Please check the Avalanche network status immediately.
            """
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent for {alert.id}")
        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

class AvalancheMonitoringSystem:
    """Main monitoring system for Avalanche network"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.monitoring_rules: List[MonitoringRule] = []
        self.data_collector = None
        self.running = False
        
        # Initialize default monitoring rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default monitoring rules"""
        default_rules = [
            # Network Performance Rules
            MonitoringRule(
                name="high_gas_price",
                metric_type="network_performance",
                metric_name="gas_price_avg",
                condition="greater_than",
                threshold=100.0,  # 100 Gwei
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.NETWORK_PERFORMANCE,
                cooldown_minutes=30
            ),
            MonitoringRule(
                name="low_throughput",
                metric_type="network_performance",
                metric_name="transaction_throughput",
                condition="less_than",
                threshold=1000,  # 1000 TPS
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.NETWORK_PERFORMANCE,
                cooldown_minutes=15
            ),
            MonitoringRule(
                name="high_network_utilization",
                metric_type="network_performance",
                metric_name="network_utilization",
                condition="greater_than",
                threshold=90.0,  # 90%
                alert_level=AlertLevel.CRITICAL,
                alert_type=AlertType.NETWORK_PERFORMANCE,
                cooldown_minutes=10
            ),
            
            # Economic Metrics Rules
            MonitoringRule(
                name="high_price_volatility",
                metric_type="economic_data",
                metric_name="price_change_24h",
                condition="greater_than",
                threshold=20.0,  # 20% change
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.MARKET_ANOMALY,
                cooldown_minutes=60
            ),
            MonitoringRule(
                name="extreme_price_drop",
                metric_type="economic_data",
                metric_name="price_change_24h",
                condition="less_than",
                threshold=-30.0,  # -30% change
                alert_level=AlertLevel.CRITICAL,
                alert_type=AlertType.MARKET_ANOMALY,
                cooldown_minutes=30
            ),
            MonitoringRule(
                name="low_volume",
                metric_type="economic_data",
                metric_name="daily_volume",
                condition="less_than",
                threshold=100000000,  # $100M
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.MARKET_ANOMALY,
                cooldown_minutes=120
            ),
            
            # Security Rules
            MonitoringRule(
                name="low_validator_count",
                metric_type="security_status",
                metric_name="validator_count",
                condition="less_than",
                threshold=1000,  # 1000 validators
                alert_level=AlertLevel.CRITICAL,
                alert_type=AlertType.SECURITY_ISSUE,
                cooldown_minutes=60
            ),
            MonitoringRule(
                name="low_staking_ratio",
                metric_type="security_status",
                metric_name="staking_ratio",
                condition="less_than",
                threshold=50.0,  # 50%
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.SECURITY_ISSUE,
                cooldown_minutes=120
            ),
            
            # Technical Health Rules
            MonitoringRule(
                name="slow_rpc_response",
                metric_type="technical_health",
                metric_name="rpc_performance.response_time_ms",
                condition="greater_than",
                threshold=5000.0,  # 5 seconds
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.TECHNICAL_FAILURE,
                cooldown_minutes=15
            ),
            MonitoringRule(
                name="low_health_score",
                metric_type="technical_health",
                metric_name="overall_health_score",
                condition="less_than",
                threshold=80.0,  # 80%
                alert_level=AlertLevel.CRITICAL,
                alert_type=AlertType.TECHNICAL_FAILURE,
                cooldown_minutes=30
            ),
            
            # DeFi Rules
            MonitoringRule(
                name="defi_tvl_drop",
                metric_type="defi_metrics",
                metric_name="total_tvl",
                condition="less_than",
                threshold=1000000000,  # $1B
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.DEFI_RISK,
                cooldown_minutes=60
            )
        ]
        
        self.monitoring_rules.extend(default_rules)
        logger.info(f"Initialized {len(default_rules)} default monitoring rules")
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a new monitoring rule"""
        self.monitoring_rules.append(rule)
        logger.info(f"Added monitoring rule: {rule.name}")
    
    def remove_monitoring_rule(self, rule_name: str):
        """Remove a monitoring rule"""
        self.monitoring_rules = [r for r in self.monitoring_rules if r.name != rule_name]
        logger.info(f"Removed monitoring rule: {rule_name}")
    
    def get_monitoring_rules(self) -> List[MonitoringRule]:
        """Get all monitoring rules"""
        return self.monitoring_rules.copy()
    
    async def evaluate_rules(self, data: Dict[str, Any]):
        """Evaluate all monitoring rules against current data"""
        for rule in self.monitoring_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if rule.last_triggered:
                time_since_triggered = datetime.utcnow() - rule.last_triggered
                if time_since_triggered.total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Get metric value
            metric_data = data.get(rule.metric_type, {})
            if not metric_data:
                continue
            
            # Navigate nested metric names (e.g., "rpc_performance.response_time_ms")
            current_value = metric_data
            for key in rule.metric_name.split('.'):
                if isinstance(current_value, dict) and key in current_value:
                    current_value = current_value[key]
                else:
                    current_value = None
                    break
            
            if current_value is None:
                continue
            
            # Evaluate condition
            should_alert = False
            try:
                if rule.condition == "greater_than":
                    should_alert = float(current_value) > float(rule.threshold)
                elif rule.condition == "less_than":
                    should_alert = float(current_value) < float(rule.threshold)
                elif rule.condition == "equals":
                    should_alert = current_value == rule.threshold
                elif rule.condition == "not_equals":
                    should_alert = current_value != rule.threshold
            except (ValueError, TypeError):
                logger.warning(f"Could not evaluate rule {rule.name}: invalid value or threshold")
                continue
            
            if should_alert:
                # Create alert
                alert = await self.alert_manager.create_alert(
                    rule, 
                    current_value, 
                    {"metric_data": metric_data}
                )
                
                # Update rule last triggered time
                rule.last_triggered = datetime.utcnow()
                
                logger.warning(f"Rule {rule.name} triggered: {current_value} {rule.condition} {rule.threshold}")
    
    async def start_monitoring(self, data_collector: RealTimeDataCollector):
        """Start monitoring with data collector"""
        logger.info("🔍 Starting Avalanche monitoring system")
        
        self.data_collector = data_collector
        self.running = True
        
        # Start monitoring loop
        while self.running:
            try:
                # Get latest data
                latest_data = self.data_collector.get_latest_data()
                
                if latest_data:
                    # Evaluate all rules
                    await self.evaluate_rules(latest_data)
                
                # Wait before next evaluation
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("🛑 Stopping Avalanche monitoring system")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "running": self.running,
            "monitoring_rules_count": len(self.monitoring_rules),
            "active_alerts_count": len(self.alert_manager.get_active_alerts()),
            "total_alerts_24h": len(self.alert_manager.get_alert_history(24)),
            "enabled_rules": len([r for r in self.monitoring_rules if r.enabled])
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        return self.alert_manager.get_alert_history(hours)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        self.alert_manager.resolve_alert(alert_id)

# Global monitoring system instance
monitoring_system = None

async def main():
    """Main function to run monitoring system"""
    global monitoring_system
    
    logger.info("🚀 Starting Avalanche Monitoring and Alerting System")
    
    # Initialize monitoring system
    monitoring_system = AvalancheMonitoringSystem()
    
    # Initialize data collector
    async with RealTimeDataCollector() as data_collector:
        # Start data collection
        collection_task = asyncio.create_task(data_collector.start())
        
        # Start monitoring
        monitoring_task = asyncio.create_task(
            monitoring_system.start_monitoring(data_collector)
        )
        
        try:
            # Wait for both tasks
            await asyncio.gather(collection_task, monitoring_task)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            # Stop monitoring
            monitoring_system.stop_monitoring()
            
            # Stop data collection
            await data_collector.stop()

if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/avalanche_monitoring_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the monitoring system
    asyncio.run(main())
