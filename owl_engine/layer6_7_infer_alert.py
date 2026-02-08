"""
OWL ENGINE - Layer 6 & 7: INFER + ALERT (Pattern Detection & Live Event Awareness)

Purpose: Detect patterns, anomalies, and generate real-time alerts.
Learn baseline behaviors and flag deviations.

Philosophy: Intelligence emerges from patterns. Alert on anomalies, not just thresholds.

Math:
- Baseline: μ ± σ (mean and standard deviation)
- Anomaly Score: (x - μ) / σ (z-score)
- Alert Confidence: sigmoid(anomaly_score × correlation_strength)
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
from layer3_link import EntityGraph
from layer4_correlate import CorrelationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('INFER_ALERT_LAYER')


class BaselineModel:
    """Learn baseline patterns from historical data"""
    
    def __init__(self):
        self.route_baselines = {}  # route_name -> {mean, std, samples}
        
    def learn_traffic_baselines(self, graph: EntityGraph):
        """Learn normal traffic patterns"""
        logger.info("[LEARN] Learning traffic baselines...")
        
        route_durations = defaultdict(list)
        
        # Collect durations per route
        for entity_id, entity in graph.entities.items():
            if entity['type'] == 'traffic_route':
                route_name = entity['data'].get('route_name')
                duration = entity['data'].get('duration_minutes')
                
                if route_name and duration:
                    route_durations[route_name].append(duration)
        
        # Calculate statistics
        for route_name, durations in route_durations.items():
            if len(durations) > 0:
                self.route_baselines[route_name] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations) if len(durations) > 1 else 5.0,
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'samples': len(durations)
                }
        
        logger.info(f"✓ Learned baselines for {len(self.route_baselines)} routes")
        
        for route, stats in self.route_baselines.items():
            logger.info(f"  {route}: {stats['mean']:.1f} ± {stats['std']:.1f} min (n={stats['samples']})")
    
    def compute_anomaly_score(self, route_name: str, duration_minutes: int) -> float:
        """
        Compute anomaly score using z-score.
        
        Z-score = (x - μ) / σ
        
        Returns: Standard deviations from mean (positive = slower than normal)
        """
        if route_name not in self.route_baselines:
            return 0.0  # No baseline, can't determine anomaly
        
        baseline = self.route_baselines[route_name]
        mean = baseline['mean']
        std = baseline['std']
        
        if std == 0:
            return 0.0
        
        z_score = (duration_minutes - mean) / std
        return z_score
    
    def is_anomaly(self, route_name: str, duration_minutes: int, threshold: float = 2.0) -> bool:
        """Check if duration is anomalous (>2σ from mean)"""
        z_score = self.compute_anomaly_score(route_name, duration_minutes)
        return abs(z_score) > threshold


class EventDetector:
    """Detect live events from data patterns"""
    
    def __init__(self, graph: EntityGraph, correlation_engine: CorrelationEngine, baseline_model: BaselineModel):
        self.graph = graph
        self.correlation_engine = correlation_engine
        self.baseline_model = baseline_model
        self.events = []
    
    def detect_flood_impact_events(self, min_confidence: float = 0.5) -> List[Dict]:
        """
        Detect events where floods are impacting traffic.
        
        Event Confidence:
        C_event = sigmoid(z_score × corr_strength × flood_severity)
        
        where sigmoid(x) = 1 / (1 + e^(-x))
        """
        logger.info("[DETECT] Detecting flood impact events...")
        
        events = []
        
        # Get correlations
        correlations = self.correlation_engine.correlations
        
        for corr in correlations:
            traffic_id = corr['traffic_id']
            flood_id = corr['flood_id']
            
            # Get entities
            traffic_entity = self.graph.get_entity(traffic_id)
            flood_entity = self.graph.get_entity(flood_id)
            
            if not traffic_entity or not flood_entity:
                continue
            
            # Check if traffic is anomalous
            route_name = traffic_entity['data'].get('route_name')
            duration = traffic_entity['data'].get('duration_minutes')
            
            z_score = self.baseline_model.compute_anomaly_score(route_name, duration)
            
            # Only flag positive anomalies (slower than normal)
            if z_score < 1.5:  # Less than 1.5 std above mean
                continue
            
            # Calculate event confidence using sigmoid
            # Higher z-score + higher correlation = higher confidence
            raw_score = z_score * corr['combined_score'] * 2
            confidence = self._sigmoid(raw_score)
            
            if confidence < min_confidence:
                continue
            
            # Create event
            event = {
                'event_type': 'flood_traffic_impact',
                'confidence': round(confidence, 3),
                'severity': self._calculate_severity(z_score, corr['flood_severity']),
                'timestamp': datetime.now().isoformat(),
                'flood_id': flood_id,
                'traffic_id': traffic_id,
                'flood_area': flood_entity['data'].get('description'),
                'flood_severity': corr['flood_severity'],
                'affected_route': route_name,
                'current_duration': f"{duration} min",
                'expected_duration': f"{self.baseline_model.route_baselines.get(route_name, {}).get('mean', 0):.0f} min",
                'delay_factor': round(z_score, 2),
                'spatial_correlation': corr['spatial_correlation'],
                'temporal_correlation': corr['temporal_correlation'],
                'lag_hours': corr['lag_hours'],
                'message': self._generate_event_message(
                    flood_entity['data'].get('description'),
                    route_name,
                    duration,
                    z_score,
                    confidence
                )
            }
            
            events.append(event)
        
        # Sort by confidence
        events.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"✓ Detected {len(events)} flood impact events")
        
        self.events = events
        return events
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function: 1 / (1 + e^(-x))"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _calculate_severity(self, z_score: float, flood_severity: float) -> str:
        """Calculate event severity level"""
        combined = z_score * flood_severity
        
        if combined > 3.0:
            return "CRITICAL"
        elif combined > 2.0:
            return "HIGH"
        elif combined > 1.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_event_message(self, flood_area: str, route_name: str, 
                                duration: int, z_score: float, confidence: float) -> str:
        """Generate human-readable event message"""
        delay_pct = int(z_score * 50)  # Rough % above normal
        
        return (f"TRAFFIC IMPACT DETECTED (Confidence: {confidence:.0%})\n"
                f"Flooding in {flood_area} is likely causing delays on {route_name}. "
                f"Current travel time: {duration} min (~{delay_pct}% above normal). "
                f"Consider alternative routes.")


class AlertSystem:
    """Generate and manage real-time alerts"""
    
    def __init__(self, event_detector: EventDetector):
        self.event_detector = event_detector
        self.alerts = []
        self.alert_history = []
    
    def generate_alerts(self, min_severity: str = "MEDIUM") -> List[Dict]:
        """Generate alerts from detected events"""
        logger.info("[ALERT] Generating alerts...")
        
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        min_level = severity_order.get(min_severity, 1)
        
        alerts = []
        
        for event in self.event_detector.events:
            event_level = severity_order.get(event['severity'], 0)
            
            if event_level >= min_level:
                alert = {
                    'alert_id': f"ALERT_{len(self.alert_history) + len(alerts) + 1:04d}",
                    'timestamp': datetime.now().isoformat(),
                    'severity': event['severity'],
                    'confidence': event['confidence'],
                    'title': f"Flood Impact on {event['affected_route']}",
                    'message': event['message'],
                    'event_data': event,
                    'status': 'ACTIVE'
                }
                alerts.append(alert)
        
        self.alerts = alerts
        self.alert_history.extend(alerts)
        
        logger.info(f"✓ Generated {len(alerts)} alerts (severity >= {min_severity})")
        
        return alerts
    
    def display_alerts(self):
        """Display active alerts"""
        if not self.alerts:
            logger.info("✅ No active alerts")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("[ALERTS] ACTIVE ALERTS")
        logger.info("=" * 70)
        
        for alert in self.alerts:
            logger.info(f"\n[{alert['alert_id']}] {alert['severity']} - Confidence: {alert['confidence']:.0%}")
            logger.info(f"Title: {alert['title']}")
            logger.info(f"Time: {alert['timestamp']}")
            logger.info(f"\n{alert['message']}")
            logger.info("-" * 70)
    
    def export_alerts(self, filepath: str = 'owl_alerts.json'):
        """Export alerts to file"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'active_alerts': self.alerts,
            'alert_count': len(self.alerts),
            'severity_breakdown': self._get_severity_breakdown()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Alerts exported to {filepath}")
    
    def _get_severity_breakdown(self) -> Dict[str, int]:
        """Get count of alerts by severity"""
        breakdown = defaultdict(int)
        for alert in self.alerts:
            breakdown[alert['severity']] += 1
        return dict(breakdown)


def run_intelligence_pipeline():
    """Run complete intelligence pipeline (Layers 3-7)"""
    from layer3_link import build_entity_graph
    
    logger.info("=" * 70)
    logger.info("[OWL] OWL ENGINE - INTELLIGENCE PIPELINE")
    logger.info("=" * 70)
    
    # Layer 3: Build entity graph
    logger.info("\n[LINK] Layer 3: LINK - Building entity graph...")
    graph = build_entity_graph()
    
    # Layer 4: Find correlations
    logger.info("\n[CORRELATE] Layer 4: CORRELATE - Finding relationships...")
    corr_engine = CorrelationEngine(graph)
    correlations = corr_engine.find_flood_traffic_correlations(
        min_spatial_threshold=0.05,
        min_temporal_threshold=0.2
    )
    
    # Layer 6: Learn baselines and infer
    logger.info("\n📊 Layer 6: INFER - Learning baselines...")
    baseline = BaselineModel()
    baseline.learn_traffic_baselines(graph)
    
    # Layer 7: Detect events
    logger.info("\n🚨 Layer 7: ALERT - Detecting events...")
    detector = EventDetector(graph, corr_engine, baseline)
    events = detector.detect_flood_impact_events(min_confidence=0.4)
    
    # Generate alerts
    alert_system = AlertSystem(detector)
    alerts = alert_system.generate_alerts(min_severity="MEDIUM")
    
    # Display results
    alert_system.display_alerts()
    
    # Export
    alert_system.export_alerts()
    
    # Export events
    with open('owl_events.json', 'w', encoding='utf-8') as f:
        json.dump({
            'detected_at': datetime.now().isoformat(),
            'event_count': len(events),
            'events': events
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Events exported to owl_events.json")
    logger.info("\n" + "=" * 70)
    logger.info("🎯 INTELLIGENCE PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    return alert_system


if __name__ == '__main__':
    run_intelligence_pipeline()
