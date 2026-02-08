"""
Test Abbey Road Video Analytics System
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

print("="*70)
print("🎥 ABBEY ROAD VIDEO ANALYTICS - SYSTEM TEST")
print("="*70)
print()

# Test 1: Check dependencies
print("1️⃣ Checking dependencies...")
print("-" * 70)

try:
    import cv2
    print("✅ OpenCV installed:", cv2.__version__)
except ImportError:
    print("❌ OpenCV not installed")
    print("   Install with: pip install opencv-python")
    cv2 = None

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO installed")
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ YOLO not installed")
    print("   Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

try:
    import torch
    print("✅ PyTorch installed:", torch.__version__)
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch torchvision")

print()

# Test 2: Load module
print("2️⃣ Loading video analytics module...")
print("-" * 70)

try:
    from data_collection.video_analytics import AbbeyRoadCollector, AbbeyRoadAnalytics
    print("✅ Video analytics module loaded successfully")
except Exception as e:
    print(f"❌ Failed to load module: {e}")
    sys.exit(1)

print()

# Test 3: Initialize collector
print("3️⃣ Initializing Abbey Road collector...")
print("-" * 70)

try:
    collector = AbbeyRoadCollector()
    print("✅ Collector initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test 4: Test collection
print("4️⃣ Testing data collection...")
print("-" * 70)

try:
    data = collector.collect()
    print("✅ Collection method works")
    print(f"   Location: {data.get('location')}")
    print(f"   Coordinates: ({data.get('lat')}, {data.get('lon')})")
    print(f"   Source: {data.get('source')}")
except Exception as e:
    print(f"❌ Collection failed: {e}")

print()

# Test 5: Check YOLO model
if YOLO_AVAILABLE:
    print("5️⃣ Testing YOLO model...")
    print("-" * 70)
    
    try:
        analytics = AbbeyRoadAnalytics()
        if analytics.model:
            print("✅ YOLO model loaded successfully")
            print("   Model can detect:")
            print("   - 🚗 Vehicles (cars, buses, motorcycles, trucks)")
            print("   - 🚶 Pedestrians")
            print("   - 📊 Direction tracking (N/S/E/W)")
        else:
            print("⚠️  YOLO model not loaded (dependencies missing)")
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
else:
    print("5️⃣ Skipping YOLO test (not installed)")
    print("-" * 70)
    print("   Install YOLO to enable AI detection:")
    print("   pip install ultralytics torch torchvision")

print()

# Test 6: Check data directory
print("6️⃣ Checking data storage...")
print("-" * 70)

data_dir = Path("owl_data")
if data_dir.exists():
    print(f"✅ Data directory exists: {data_dir.absolute()}")
    
    # Check for video analytics subdirs
    today_folders = list(data_dir.glob("*/video_analytics"))
    if today_folders:
        print(f"   Found {len(today_folders)} day(s) with video analytics")
        for folder in today_folders:
            files = list(folder.glob("*.json"))
            print(f"   - {folder.parent.name}: {len(files)} analytics files")
    else:
        print("   No analytics data collected yet")
else:
    print(f"⚠️  Data directory will be created: {data_dir.absolute()}")

print()

# Summary
print("="*70)
print("📋 SYSTEM STATUS SUMMARY")
print("="*70)

all_ok = cv2 is not None and YOLO_AVAILABLE

if all_ok:
    print("✅ System is READY for video analytics!")
    print()
    print("📝 Next steps:")
    print("   1. Run continuous collector:")
    print("      python continuous_collector.py")
    print()
    print("   2. Or run dedicated video analysis:")
    print("      from data_collection.video_analytics import AbbeyRoadAnalytics")
    print("      analytics = AbbeyRoadAnalytics()")
    print("      analytics.run_continuous_analysis(duration_minutes=60)")
    print()
    print("   3. View results in Palantir dashboard:")
    print("      streamlit run palantir_dashboard.py")
    print("      → Check 'Abbey Road' tab")
else:
    print("⚠️  System needs dependencies installed")
    print()
    print("📦 Installation commands:")
    if cv2 is None:
        print("   pip install opencv-python")
    if not YOLO_AVAILABLE:
        print("   pip install ultralytics")
    if 'torch' not in sys.modules:
        print("   pip install torch torchvision")
    print()
    print("   Or install all at once:")
    print("   pip install opencv-python ultralytics torch torchvision")

print("="*70)
print()

print("📖 Full documentation: ABBEY_ROAD_ANALYTICS.md")
print()
