"""
Generate demo Abbey Road data for different times of day
"""
from data_collection.video_analytics import AbbeyRoadCollector
from datetime import datetime, timedelta
import time

print("🎬 Generating Abbey Road Demo Data - 24 Hour Simulation")
print("=" * 60)

collector = AbbeyRoadCollector(demo_mode=True)

# Generate data for different hours to show patterns
hours = [7, 8, 9, 12, 14, 17, 18, 19, 22]  # Various times including rush hours

print(f"\nGenerating data for {len(hours)} different time periods...")
print("This simulates a full day of traffic patterns\n")

for hour in hours:
    # Temporarily override hour for demo
    original_hour = datetime.now().hour
    
    print(f"⏰ Simulating hour: {hour:02d}:00")
    
    # Generate multiple samples per hour
    for i in range(2):
        data = collector.collect()
        vehicles = data.get('vehicle_count', 0)
        people = data.get('person_count', 0)
        rush = "🔥 RUSH HOUR" if data.get('is_rush_hour') else ""
        tourist = "📸 TOURIST PEAK" if data.get('is_tourist_peak') else ""
        
        print(f"  Sample {i+1}: {vehicles} vehicles, {people} people {rush} {tourist}")
        time.sleep(0.1)  # Small delay
    
    print()

print("✅ Demo data generated!")
print(f"📁 Saved to: owl_data/{datetime.now().strftime('%Y-%m-%d')}/video_analytics/")
print("\n🎯 View in dashboard:")
print("   streamlit run palantir_dashboard.py")
print("   → Check 'Abbey Road' tab for peak time charts!")
print("=" * 60)
