"""Test geocoding implementation"""
from data_manager import DataManager

dm = DataManager()
timeline = dm.create_unified_timeline()

print(f"Total events: {len(timeline)}")
print(f"\nEvents with coordinates: {timeline[['lat', 'lon']].notna().all(axis=1).sum()}")

print("\n=== SAMPLE EVENTS WITH LOCATIONS ===")
print(timeline[['timestamp', 'event_type', 'location', 'lat', 'lon']].head(15).to_string())

print("\n=== TRAFFIC ROUTES ===")
traffic = timeline[timeline['event_type'] == 'traffic']
if not traffic.empty:
    print(f"Traffic routes with coordinates: {len(traffic)}")
    print(traffic[['location', 'lat', 'lon']].head().to_string())

print("\n=== FLOOD WARNINGS ===")
floods = timeline[timeline['event_type'] == 'flood']
if not floods.empty:
    print(f"Flood warnings with coordinates: {len(floods)}")
    print(floods[['location', 'lat', 'lon']].head().to_string())
