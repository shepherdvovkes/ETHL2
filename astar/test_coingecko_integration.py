#!/usr/bin/env python3
"""
Test CoinGecko Integration for Astar
===================================

Simple test script to verify CoinGecko API integration works correctly.
"""

import asyncio
import sys
import os
from loguru import logger

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from astar_coingecko_integration import AstarCoinGeckoIntegration

async def test_coingecko_integration():
    """Test the CoinGecko integration"""
    print("🧪 Testing Astar CoinGecko Integration")
    print("=" * 50)
    
    try:
        async with AstarCoinGeckoIntegration() as integration:
            # Test 1: Get current price data
            print("\n📊 Test 1: Getting current price data...")
            price_data = await integration.get_current_price_data()
            
            if price_data:
                print("✅ Price data retrieved successfully!")
                print(f"   💰 Price: ${price_data.get('price_usd', 0):.4f}")
                print(f"   📊 Market Cap: ${price_data.get('market_cap_usd', 0):,.0f}")
                print(f"   📈 24h Volume: ${price_data.get('volume_24h_usd', 0):,.0f}")
                print(f"   📉 24h Change: {price_data.get('price_change_24h', 0):.2f}%")
            else:
                print("❌ Failed to retrieve price data")
                return False
            
            # Test 2: Get detailed market data
            print("\n📊 Test 2: Getting detailed market data...")
            detailed_data = await integration.get_detailed_market_data()
            
            if detailed_data:
                print("✅ Detailed market data retrieved successfully!")
                print(f"   💰 Price: ${detailed_data.get('price_usd', 0):.4f}")
                print(f"   📊 Market Cap: ${detailed_data.get('market_cap_usd', 0):,.0f}")
                print(f"   📈 24h Volume: ${detailed_data.get('volume_24h_usd', 0):,.0f}")
                print(f"   📉 7d Change: {detailed_data.get('price_change_7d', 0):.2f}%")
                print(f"   📊 Volatility: {detailed_data.get('price_volatility_24h', 0):.2f}%")
            else:
                print("❌ Failed to retrieve detailed market data")
                return False
            
            # Test 3: Get historical price data
            print("\n📊 Test 3: Getting historical price data (7 days)...")
            historical_data = await integration.get_historical_price_data(days=7)
            
            if historical_data:
                print(f"✅ Historical data retrieved successfully! ({len(historical_data)} points)")
                if historical_data:
                    latest = historical_data[-1]
                    oldest = historical_data[0]
                    print(f"   📅 Date range: {oldest['timestamp'].strftime('%Y-%m-%d')} to {latest['timestamp'].strftime('%Y-%m-%d')}")
                    print(f"   💰 Latest price: ${latest['price_usd']:.4f}")
                    print(f"   💰 Oldest price: ${oldest['price_usd']:.4f}")
                    price_change = ((latest['price_usd'] - oldest['price_usd']) / oldest['price_usd']) * 100
                    print(f"   📈 7d price change: {price_change:.2f}%")
            else:
                print("❌ Failed to retrieve historical data")
                return False
            
            # Test 4: Get comprehensive market data
            print("\n📊 Test 4: Getting comprehensive market data...")
            comprehensive_data = await integration.get_comprehensive_market_data()
            
            if comprehensive_data:
                print("✅ Comprehensive market data retrieved successfully!")
                print(f"   💰 Price: ${comprehensive_data.get('price_usd', 0):.4f}")
                print(f"   📊 Market Cap: ${comprehensive_data.get('market_cap_usd', 0):,.0f}")
                print(f"   📈 24h Volume: ${comprehensive_data.get('volume_24h_usd', 0):,.0f}")
                print(f"   📉 24h Change: {comprehensive_data.get('price_change_24h', 0):.2f}%")
                print(f"   📉 7d Change: {comprehensive_data.get('price_change_7d', 0):.2f}%")
                print(f"   📊 Volatility: {comprehensive_data.get('price_volatility', 0):.2f}%")
                print(f"   🚀 Momentum: {comprehensive_data.get('price_momentum', 0):.2f}%")
                print(f"   📈 Volume Trend: {comprehensive_data.get('volume_trend', 0):.2f}%")
                print(f"   📊 Data Source: {comprehensive_data.get('data_source', 'unknown')}")
                print(f"   📅 Last Updated: {comprehensive_data.get('timestamp')}")
            else:
                print("❌ Failed to retrieve comprehensive market data")
                return False
            
            # Test 5: Test database saving
            print("\n💾 Test 5: Testing database saving...")
            integration.save_market_data(comprehensive_data)
            print("✅ Market data saved to database successfully!")
            
            print("\n🎉 All tests passed! CoinGecko integration is working correctly.")
            return True
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

async def main():
    """Main function"""
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run tests
    success = await test_coingecko_integration()
    
    if success:
        print("\n✅ CoinGecko integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ CoinGecko integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
