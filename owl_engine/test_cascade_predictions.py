"""
Test Advanced Cascade Prediction System
Verifies network effect modeling and traffic redistribution predictions
"""
import logging
from intelligence.predictive_engine import PredictiveEngine
from data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test')

def test_cascade_predictions():
    """Test cascade predictions with real data"""
    logger.info("="*70)
    logger.info("🔮 TESTING ADVANCED CASCADE PREDICTION SYSTEM")
    logger.info("="*70)
    
    # Load real events
    dm = DataManager()
    timeline = dm.create_unified_timeline()
    
    # Convert to list of dicts
    events = timeline.to_dict('records')
    
    logger.info(f"\n📊 Loaded {len(events)} events for analysis")
    
    # Initialize predictor
    predictor = PredictiveEngine()
    
    # Generate predictions
    predictions = predictor.generate_all_predictions(events)
    
    # Filter cascade predictions
    cascade_preds = [p for p in predictions if p.get('prediction_type') == 'traffic_cascade']
    
    logger.info(f"\n🚨 GENERATED {len(cascade_preds)} CASCADE PREDICTIONS\n")
    
    if cascade_preds:
        for i, pred in enumerate(cascade_preds[:5], 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDICTION #{i}")
            logger.info(f"{'='*70}")
            logger.info(f"Trigger: {pred.get('trigger_event', 'Unknown')}")
            logger.info(f"Trigger Location: {pred.get('trigger_location', 'Unknown')}")
            logger.info(f"")
            logger.info(f"Affected Route: {pred.get('affected_route', 'Unknown')}")
            logger.info(f"Affected Location: {pred.get('affected_location', 'Unknown')}")
            logger.info(f"")
            logger.info(f"Risk Score: {pred.get('risk_score', 0):.0%}")
            logger.info(f"Confidence: {pred.get('confidence', 0):.0%}")
            logger.info(f"Time to Impact: {pred.get('time_to_impact', 'Unknown')}")
            logger.info(f"")
            logger.info(f"Expected Delay Increase: {pred.get('expected_delay_increase', 'N/A')}")
            logger.info(f"Capacity Overflow: {pred.get('capacity_overflow', 'N/A')}")
            logger.info(f"")
            logger.info(f"PREDICTION:")
            logger.info(f"  {pred.get('prediction', 'Unknown')}")
            logger.info(f"")
            logger.info(f"RECOMMENDATION:")
            logger.info(f"  {pred.get('recommendation', 'Unknown')}")
            logger.info(f"")
            if pred.get('factors'):
                logger.info(f"FACTORS:")
                for factor in pred['factors']:
                    logger.info(f"  • {factor}")
    else:
        logger.info("ℹ️  No cascade predictions generated")
        logger.info("   This may be because:")
        logger.info("   - No road disruptions with closures detected")
        logger.info("   - No alternative routes identified within 3km radius")
        logger.info("   - Current traffic data insufficient for cascade modeling")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ TESTING COMPLETE")
    logger.info(f"{'='*70}\n")
    
    return len(cascade_preds) > 0

if __name__ == '__main__':
    success = test_cascade_predictions()
    exit(0 if success else 1)
