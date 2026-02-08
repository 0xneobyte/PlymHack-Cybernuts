"""
Test real-time Abbey Road detection
"""

from data_collection.video_analytics import AbbeyRoadCollector
import logging

logging.basicConfig(level=logging.INFO)

print("="*70)
print("  TESTING REAL-TIME ABBEY ROAD DETECTION")
print("="*70)
print()

# Create collector with real-time mode (demo_mode=False)
collector = AbbeyRoadCollector(demo_mode=False)

print("🎥 Attempting to capture frame from YouTube live stream...")
print("📹 Running YOLO detection...")
print()

# Run collection (will try YouTube capture, fallback to demo if needed)
result = collector.collect()

print()
print("="*70)
print("  DETECTION RESULTS")
print("="*70)
print(f"  Vehicles: {result.get('vehicle_count', 0)}")
print(f"  People: {result.get('person_count', 0)}")
print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
print()

if 'detections' in result:
    print(f"  Objects detected: {len(result.get('detections', []))}")
    for obj in result.get('detections', [])[:5]:  # Show first 5
        print(f"    - {obj.get('class')} ({obj.get('confidence', 0):.2f})")

print()
print("="*70)
print("💾 Data saved to owl_data/[date]/video_analytics/")
print("🖼️ Annotated frames saved to frames/ subfolder")
print("="*70)
