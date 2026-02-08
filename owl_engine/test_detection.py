"""Test Abbey Road detection functionality"""

from data_collection.video_analytics import AbbeyRoadCollector
import json

print("Testing Abbey Road video analytics...")

collector = AbbeyRoadCollector()
results = collector.collect()

if results and results.get('status') == 'success':
    print(f"✓ Detection successful!")
    print(f"  Status: {results['status']}")
    print(f"  Data collected: {len(results.get('detections', []))} frames")
    if results.get('detections'):
        latest = results['detections'][0]
        print(f"  Latest detection:")
        print(f"    - Vehicles: {latest.get('vehicle_count', 0)}")
        print(f"    - People: {latest.get('person_count', 0)}")
        print(f"    - Objects: {len(latest.get('objects', []))}")
    print(f"\n✅ Abbey Road analytics is ready to use in the dashboard!")
else:
    print(f"⚠️  Detection returned: {json.dumps(results, indent=2)}")
