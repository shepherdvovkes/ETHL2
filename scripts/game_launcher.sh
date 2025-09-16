#!/bin/bash

# RTX 4090 Open Source Game Launcher
# This script helps you launch games that can showcase your RTX 4090 performance

echo "🎮 RTX 4090 Open Source Game Launcher 🎮"
echo "========================================"
echo ""

# Check GPU status
echo "🔍 Checking GPU Status..."
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU not detected or drivers not installed"
echo ""

echo "Available Games:"
echo "================"
echo "1. SuperTuxKart - 3D Kart Racing (Great for GPU testing)"
echo "2. FlightGear - Flight Simulator (Very demanding)"
echo "3. Minetest - Minecraft-like sandbox (Can be modded for performance)"
echo "4. OpenMW - Morrowind Engine (Requires game files)"
echo "5. Xonotic - Fast FPS (Via Snap)"
echo "6. Wesnoth - Turn-based Strategy"
echo "7. OpenTTD - Transport Simulation"
echo "8. Frozen Bubble - Puzzle Game"
echo "9. Pingus - Lemmings-style Puzzle"
echo ""

read -p "Enter game number (1-9) or 'q' to quit: " choice

case $choice in
    1)
        echo "Launching SuperTuxKart..."
        supertuxkart
        ;;
    2)
        echo "Launching FlightGear..."
        flightgear
        ;;
    3)
        echo "Launching Minetest..."
        minetest
        ;;
    4)
        echo "Launching OpenMW Launcher..."
        openmw-launcher
        ;;
    5)
        echo "Launching Xonotic..."
        xonotic
        ;;
    6)
        echo "Launching Wesnoth..."
        wesnoth
        ;;
    7)
        echo "Launching OpenTTD..."
        openttd
        ;;
    8)
        echo "Launching Frozen Bubble..."
        frozen-bubble
        ;;
    9)
        echo "Launching Pingus..."
        pingus
        ;;
    q|Q)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac
