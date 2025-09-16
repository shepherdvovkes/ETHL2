#!/usr/bin/env python3
"""
Setup script for Astar collection notifications
Configures and starts the notification monitoring system
"""

import os
import subprocess
import time
from loguru import logger

def setup_notifications():
    """Setup and start notification monitoring"""
    
    print("🔔 Astar Collection Notification Setup")
    print("=" * 50)
    
    # Check if collection is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "start_full_astar_collection.py"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print("✅ Astar collection process detected")
            print(f"📊 Process ID: {result.stdout.strip()}")
        else:
            print("⚠️  No Astar collection process found")
            print("   Make sure the collection is running first")
            return
    except Exception as e:
        print(f"❌ Error checking process: {e}")
        return
    
    # Check notification dependencies
    print("\n🔍 Checking notification dependencies...")
    
    # Check for notify-send (Linux desktop notifications)
    try:
        subprocess.run(["which", "notify-send"], check=True, capture_output=True)
        print("✅ Desktop notifications: Available (notify-send)")
    except subprocess.CalledProcessError:
        print("⚠️  Desktop notifications: notify-send not found")
        print("   Install with: sudo apt install libnotify-bin")
    
    # Check for email configuration
    print("\n📧 Email Notifications:")
    print("   To enable email notifications, edit astar_completion_notifier.py")
    print("   and configure the email_config dictionary")
    
    # Start the notification monitor
    print("\n🚀 Starting notification monitor...")
    
    try:
        # Start the notifier in background
        cmd = ["python3", "astar_completion_notifier.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Notification monitor started (PID: {process.pid})")
        print("📝 Monitor will check every 30 seconds")
        print("🔔 You'll get notified when collection completes")
        
        # Save PID for reference
        with open("notification_monitor.pid", "w") as f:
            f.write(str(process.pid))
        
        print("\n📊 Notification Monitor Status:")
        print("=" * 30)
        print("🔔 Desktop notifications: ENABLED")
        print("📧 Email notifications: DISABLED (configure to enable)")
        print("⏰ Check interval: 30 seconds")
        print("🎯 Target: Block 10,227,287")
        print("📝 Log: Monitor logs to console")
        
        print("\n🛑 To stop monitoring:")
        print("   kill $(cat notification_monitor.pid)")
        print("   or Ctrl+C if running in foreground")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start notification monitor: {e}")
        return None

def check_notification_status():
    """Check if notification monitor is running"""
    try:
        if os.path.exists("notification_monitor.pid"):
            with open("notification_monitor.pid", "r") as f:
                pid = f.read().strip()
            
            # Check if process is still running
            result = subprocess.run(
                ["pgrep", "-f", "astar_completion_notifier.py"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                print(f"✅ Notification monitor running (PID: {pid})")
                return True
            else:
                print("❌ Notification monitor not running")
                return False
        else:
            print("❌ No notification monitor PID file found")
            return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

def stop_notifications():
    """Stop notification monitoring"""
    try:
        if os.path.exists("notification_monitor.pid"):
            with open("notification_monitor.pid", "r") as f:
                pid = f.read().strip()
            
            subprocess.run(["kill", pid], check=False)
            os.remove("notification_monitor.pid")
            print(f"✅ Stopped notification monitor (PID: {pid})")
        else:
            print("❌ No notification monitor PID file found")
    except Exception as e:
        print(f"❌ Error stopping notifications: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            setup_notifications()
        elif command == "status":
            check_notification_status()
        elif command == "stop":
            stop_notifications()
        else:
            print("Usage: python setup_notifications.py [start|status|stop]")
    else:
        setup_notifications()
