#!/usr/bin/env python3
"""
Test script for Astar notification system
Tests desktop notifications and system functionality
"""

import subprocess
import time
import os
from loguru import logger

def test_desktop_notification():
    """Test desktop notification functionality"""
    print("🔔 Testing Desktop Notifications")
    print("=" * 40)
    
    try:
        # Test notify-send
        subprocess.run([
            "notify-send",
            "-i", "applications-development",
            "-u", "critical",
            "🔔 Astar Notification Test",
            "This is a test notification from the Astar collection system!"
        ], check=True)
        
        print("✅ Desktop notification test sent!")
        print("   You should see a notification popup")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Desktop notification test failed")
        print("   Make sure notify-send is installed:")
        print("   sudo apt install libnotify-bin")
        return False
    except Exception as e:
        print(f"❌ Error testing notifications: {e}")
        return False

def test_database_access():
    """Test database access for monitoring"""
    print("\n💾 Testing Database Access")
    print("=" * 30)
    
    db_path = "astar_multithreaded_data.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test basic queries
        cursor.execute("SELECT COUNT(*) FROM astar_blocks")
        block_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(block_number) FROM astar_blocks")
        max_block = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database access successful")
        print(f"   📊 Blocks in database: {block_count:,}")
        print(f"   🎯 Latest block: {max_block:,}")
        return True
        
    except Exception as e:
        print(f"❌ Database access failed: {e}")
        return False

def test_process_monitoring():
    """Test process monitoring functionality"""
    print("\n🔍 Testing Process Monitoring")
    print("=" * 35)
    
    try:
        # Check for Astar collection process
        result = subprocess.run(
            ["pgrep", "-f", "start_full_astar_collection.py"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pid = result.stdout.strip()
            print(f"✅ Collection process found (PID: {pid})")
            return True
        else:
            print("⚠️  No Astar collection process found")
            print("   This is normal if collection hasn't started yet")
            return False
            
    except Exception as e:
        print(f"❌ Process monitoring test failed: {e}")
        return False

def test_notification_system():
    """Test the complete notification system"""
    print("\n🧪 Testing Notification System")
    print("=" * 35)
    
    try:
        # Import the notifier
        from astar_completion_notifier import AstarCompletionNotifier
        
        # Create a test instance
        notifier = AstarCompletionNotifier(
            db_path="astar_multithreaded_data.db",
            target_block=10227287,
            check_interval=30
        )
        
        # Test basic functionality
        current_blocks = notifier._get_current_block_count()
        latest_block = notifier._get_latest_block_number()
        process_status = notifier._get_process_status()
        
        print(f"✅ Notification system initialized")
        print(f"   📊 Current blocks: {current_blocks:,}")
        print(f"   🎯 Latest block: {latest_block:,}")
        print(f"   🔄 Process running: {process_status['running']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Notification system test failed: {e}")
        return False

def run_all_tests():
    """Run all notification tests"""
    print("🧪 Astar Notification System Tests")
    print("=" * 50)
    
    tests = [
        ("Desktop Notifications", test_desktop_notification),
        ("Database Access", test_database_access),
        ("Process Monitoring", test_process_monitoring),
        ("Notification System", test_notification_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Notification system is ready.")
    elif passed >= total // 2:
        print("⚠️  Some tests failed, but basic functionality should work.")
    else:
        print("❌ Multiple tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
