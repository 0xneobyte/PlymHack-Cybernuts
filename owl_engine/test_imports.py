"""
Test script to verify all OWL Engine modules can be imported
"""

print("Testing OWL Engine imports...")

try:
    print("✓ Testing data collectors...")
    from data_collection.flood_monitoring import FloodMonitoringCollector
    from data_collection.environmental_monitor import EnvironmentalMonitor
    from data_collection.social_monitor import SocialThreatMonitor
    from data_collection.infrastructure_monitor import InfrastructureMonitor
    print("  ✓ All collectors imported successfully")
    
    print("✓ Testing intelligence modules...")
    from intelligence.threat_correlator import ThreatCorrelator
    from intelligence.predictive_engine import PredictiveEngine
    print("  ✓ Intelligence modules imported successfully")
    
    print("✓ Testing correlation layers...")
    from layer3_link import EntityGraph, EntityLinker
    from layer4_correlate import CorrelationEngine
    from layer6_7_infer_alert import BaselineModel, EventDetector, AlertSystem
    print("  ✓ Correlation layers imported successfully")
    
    print("\n✅ ALL MODULES IMPORTED SUCCESSFULLY!")
    print("\n🦉 OWL Engine is ready to launch.")
    print("\nRun: python owl_engine\\main.py")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
