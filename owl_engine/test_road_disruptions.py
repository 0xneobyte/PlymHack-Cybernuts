"""
Test Road Disruptions Integration
Verify:
1. Collector fetches disruptions from TfL
2. Data gets saved to owl_data/
3. DataManager loads disruptions
4. Timeline includes disruptions with geocoding
"""
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test')

def test_collector():
    """Test road disruptions collector"""
    logger.info("="*70)
    logger.info("TEST 1: Road Disruptions Collection")
    logger.info("="*70)
    
    from data_collection.road_disruptions import RoadDisruptionsCollector
    
    collector = RoadDisruptionsCollector()
    disruptions = collector.collect()
    
    logger.info(f"✓ Collected {len(disruptions)} disruptions")
    
    if disruptions:
        sample = disruptions[0]
        logger.info(f"  Sample: {sample.get('road_name')} - {sample.get('category')}")
        logger.info(f"  Location: {sample.get('location')}")
        logger.info(f"  Severity: {sample.get('severity')}")
    
    return len(disruptions) > 0

def test_data_loading():
    """Test DataManager loads disruptions"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: DataManager Loading")
    logger.info("="*70)
    
    from data_manager import DataManager
    
    dm = DataManager()
    disruptions_df = dm.load_all_road_disruptions()
    
    logger.info(f"✓ Loaded {len(disruptions_df)} disruptions from disk")
    
    if not disruptions_df.empty:
        logger.info(f"  Columns: {list(disruptions_df.columns)}")
        logger.info(f"  Categories: {disruptions_df['category'].unique().tolist()}")
    
    return not disruptions_df.empty

def test_timeline_integration():
    """Test disruptions appear in unified timeline"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Timeline Integration")
    logger.info("="*70)
    
    from data_manager import DataManager
    
    dm = DataManager()
    timeline = dm.create_unified_timeline()
    
    # Filter disruption events
    disruption_events = timeline[timeline['event_type'] == 'road_disruption']
    
    logger.info(f"✓ Timeline has {len(timeline)} total events")
    logger.info(f"  - {len(disruption_events)} road disruptions")
    
    if not disruption_events.empty:
        # Check geocoding
        geocoded = disruption_events[disruption_events['lat'].notna()]
        logger.info(f"  - {len(geocoded)}/{len(disruption_events)} geocoded")
        
        # Show sample
        sample = disruption_events.iloc[0]
        logger.info(f"\n  Sample disruption:")
        logger.info(f"    Road: {sample['description']}")
        logger.info(f"    Location: {sample['location']}")
        logger.info(f"    Coordinates: ({sample['lat']:.4f}, {sample['lon']:.4f})")
        logger.info(f"    Severity: {sample['severity']}")
        logger.info(f"    Temporal Weight: {sample['temporal_weight']:.3f}")
        logger.info(f"    Domain: {sample['domain']}")
    
    return not disruption_events.empty

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🦉 OWL ENGINE - ROAD DISRUPTIONS INTEGRATION TEST")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: Collection
    try:
        results.append(("Collection", test_collector()))
    except Exception as e:
        logger.error(f"✗ Collection test failed: {e}", exc_info=True)
        results.append(("Collection", False))
    
    # Test 2: Loading
    try:
        results.append(("Data Loading", test_data_loading()))
    except Exception as e:
        logger.error(f"✗ Loading test failed: {e}", exc_info=True)
        results.append(("Data Loading", False))
    
    # Test 3: Timeline
    try:
        results.append(("Timeline Integration", test_timeline_integration()))
    except Exception as e:
        logger.error(f"✗ Timeline test failed: {e}", exc_info=True)
        results.append(("Timeline Integration", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Road disruptions fully integrated!")
    else:
        print("✗ SOME TESTS FAILED - Check logs above")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())
