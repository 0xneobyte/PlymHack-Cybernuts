"""
OWL ENGINE - Layer 4: CORRELATE (Correlation & Relationship Layer)

Purpose: Find temporal, spatial, and causal correlations between entities.
Detect patterns like "floods causing traffic delays" through geo-temporal analysis.

Philosophy: Everything influences everything. Measure all correlations, even weak ones.
Use exponential decay and lead-lag analysis.

Math:
- Spatial correlation: Jaccard similarity of location sets
- Temporal correlation: Lead-lag cross-correlation with exponential decay
- Combined score: weighted sum of spatial × temporal × severity
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
from layer3_link import EntityGraph, build_entity_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CORRELATE_LAYER')


class CorrelationEngine:
    """Find correlations between entities in the knowledge graph"""
    
    def __init__(self, graph: EntityGraph):
        self.graph = graph
        self.correlations = []  # List of correlation records
        
    def compute_spatial_correlation(self, entity1_id: str, entity2_id: str) -> float:
        """
        Compute spatial correlation using Jaccard similarity of location sets.
        
        Jaccard(A,B) = |A ∩ B| / |A ∪ B|
        
        Returns: 0.0 to 1.0 (0 = no overlap, 1 = perfect overlap)
        """
        e1 = self.graph.get_entity(entity1_id)
        e2 = self.graph.get_entity(entity2_id)
        
        if not e1 or not e2:
            return 0.0
        
        # Normalize locations
        locs1 = set(self.graph._normalize_location(loc) for loc in e1.get('locations', []))
        locs2 = set(self.graph._normalize_location(loc) for loc in e2.get('locations', []))
        
        if not locs1 or not locs2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(locs1 & locs2)
        union = len(locs1 | locs2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_temporal_correlation(self, entity1_id: str, entity2_id: str, decay_rate: float = 0.1) -> Tuple[float, int]:
        """
        Compute temporal correlation with exponential decay.
        
        Temporal Score = exp(-λ × Δt)
        where Δt is time difference in hours, λ is decay rate
        
        Returns: (correlation_score, lag_hours)
        - correlation_score: 0.0 to 1.0
        - lag_hours: time difference (positive if e2 after e1)
        """
        e1 = self.graph.get_entity(entity1_id)
        e2 = self.graph.get_entity(entity2_id)
        
        if not e1 or not e2:
            return 0.0, 0
        
        ts1_str = e1.get('timestamp')
        ts2_str = e2.get('timestamp')
        
        if not ts1_str or not ts2_str:
            return 0.0, 0
        
        # Parse timestamps
        ts1 = datetime.fromisoformat(ts1_str.replace('Z', '+00:00'))
        ts2 = datetime.fromisoformat(ts2_str.replace('Z', '+00:00'))
        
        # Calculate time difference in hours
        delta = ts2 - ts1
        lag_hours = delta.total_seconds() / 3600
        
        # Exponential decay: closer in time = higher correlation
        # exp(-λ × |Δt|)
        time_diff_abs = abs(lag_hours)
        temporal_score = np.exp(-decay_rate * time_diff_abs)
        
        return temporal_score, int(lag_hours)
    
    def compute_severity_weight(self, entity_id: str) -> float:
        """
        Compute severity/importance weight for an entity.
        
        For floods: Based on severity level (1-4)
        For traffic: Based on delay (duration vs expected baseline)
        
        Returns: 0.0 to 1.0
        """
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return 0.5
        
        entity_type = entity.get('type')
        data = entity.get('data', {})
        
        if entity_type == 'flood_warning':
            # Severity levels: 1=Severe, 2=Warning, 3=Alert, 4=No longer in force
            severity_level = data.get('severity_level', 3)
            # Invert: lower number = higher severity
            return 1.0 - (severity_level - 1) / 3.0
        
        elif entity_type == 'traffic_route':
            # Duration-based: normalize to 0-1 range
            # Assume 15 min = normal (0.5), 60+ min = severe (1.0)
            duration = data.get('duration_minutes', 15)
            normalized = min((duration - 15) / 45.0, 1.0)  # 15-60 min range
            return max(0.0, normalized)
        
        return 0.5
    
    def find_flood_traffic_correlations(self, min_spatial_threshold: float = 0.1, 
                                       min_temporal_threshold: float = 0.3) -> List[Dict]:
        """
        Find correlations between flood warnings and traffic delays.
        
        Combined Correlation Score:
        C = α × S_spatial × S_temporal × (W_flood + W_traffic) / 2
        
        where:
        - S_spatial: Jaccard similarity of locations
        - S_temporal: Exponential decay based on time difference
        - W_flood, W_traffic: Severity weights
        - α: scaling factor (default 1.0)
        """
        logger.info("🔍 Finding flood-traffic correlations...")
        
        floods = [eid for eid, e in self.graph.entities.items() if e['type'] == 'flood_warning']
        traffic_routes = [eid for eid, e in self.graph.entities.items() if e['type'] == 'traffic_route']
        
        logger.info(f"  Analyzing {len(floods)} floods × {len(traffic_routes)} traffic routes")
        
        correlations = []
        
        for flood_id in floods:
            for traffic_id in traffic_routes:
                # Compute spatial correlation
                spatial_corr = self.compute_spatial_correlation(flood_id, traffic_id)
                
                if spatial_corr < min_spatial_threshold:
                    continue  # Skip if no spatial overlap
                
                # Compute temporal correlation
                temporal_corr, lag_hours = self.compute_temporal_correlation(flood_id, traffic_id)
                
                if temporal_corr < min_temporal_threshold:
                    continue  # Skip if too far apart in time
                
                # Get severity weights
                flood_severity = self.compute_severity_weight(flood_id)
                traffic_severity = self.compute_severity_weight(traffic_id)
                
                # Combined correlation score
                avg_severity = (flood_severity + traffic_severity) / 2
                combined_score = spatial_corr * temporal_corr * avg_severity
                
                # Store correlation
                flood_entity = self.graph.get_entity(flood_id)
                traffic_entity = self.graph.get_entity(traffic_id)
                
                correlation = {
                    'flood_id': flood_id,
                    'traffic_id': traffic_id,
                    'spatial_correlation': round(spatial_corr, 3),
                    'temporal_correlation': round(temporal_corr, 3),
                    'lag_hours': lag_hours,
                    'flood_severity': round(flood_severity, 3),
                    'traffic_severity': round(traffic_severity, 3),
                    'combined_score': round(combined_score, 3),
                    'flood_description': flood_entity['data'].get('description'),
                    'flood_area': flood_entity['data'].get('area'),
                    'traffic_route': traffic_entity['data'].get('route_name'),
                    'traffic_duration': traffic_entity['data'].get('duration_text'),
                    'inference': self._generate_inference(spatial_corr, temporal_corr, lag_hours, flood_severity, traffic_severity)
                }
                
                correlations.append(correlation)
                
                # Add relationship to graph
                self.graph.add_relationship(
                    flood_id, 
                    traffic_id, 
                    'potential_impact', 
                    strength=combined_score
                )
        
        # Sort by combined score
        correlations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"✓ Found {len(correlations)} significant correlations")
        
        self.correlations = correlations
        return correlations
    
    def _generate_inference(self, spatial: float, temporal: float, lag_hours: int, 
                           flood_sev: float, traffic_sev: float) -> str:
        """Generate human-readable inference about the correlation"""
        
        if spatial > 0.5 and temporal > 0.7:
            if lag_hours > 0:
                return f"HIGH: Flood likely caused traffic delay ({abs(lag_hours)}h later)"
            else:
                return f"HIGH: Concurrent flood and traffic congestion"
        elif spatial > 0.3 and temporal > 0.5:
            return f"MODERATE: Possible flood impact on traffic"
        elif spatial > 0.1:
            return f"LOW: Geographic proximity, weak temporal link"
        else:
            return "WEAK: Minimal spatial/temporal correlation"
    
    def get_top_correlations(self, n: int = 10) -> List[Dict]:
        """Get top N correlations by score"""
        return self.correlations[:n]
    
    def export_correlations(self) -> Dict:
        """Export all correlations for storage"""
        return {
            'computed_at': datetime.now().isoformat(),
            'total_correlations': len(self.correlations),
            'correlations': self.correlations
        }


def run_correlation_analysis():
    """Run complete correlation analysis"""
    logger.info("=" * 60)
    logger.info("OWL ENGINE - Layer 4: CORRELATE")
    logger.info("=" * 60)
    
    # Build graph from Layer 3
    graph = build_entity_graph()
    
    # Run correlation engine
    engine = CorrelationEngine(graph)
    correlations = engine.find_flood_traffic_correlations(
        min_spatial_threshold=0.05,  # Lower threshold to catch more
        min_temporal_threshold=0.2
    )
    
    # Display top correlations
    logger.info("\n🎯 Top 10 Flood-Traffic Correlations:")
    logger.info("=" * 60)
    
    for i, corr in enumerate(engine.get_top_correlations(10), 1):
        logger.info(f"\n{i}. Score: {corr['combined_score']:.3f}")
        logger.info(f"   Flood: {corr['flood_description']}")
        logger.info(f"   Area: {corr['flood_area']}")
        logger.info(f"   Traffic: {corr['traffic_route']} ({corr['traffic_duration']})")
        logger.info(f"   Spatial: {corr['spatial_correlation']:.2f} | Temporal: {corr['temporal_correlation']:.2f} | Lag: {corr['lag_hours']}h")
        logger.info(f"   ⚡ {corr['inference']}")
    
    # Save correlations
    output_path = Path('owl_correlations.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(engine.export_correlations(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Correlations saved to {output_path}")
    
    return engine


if __name__ == '__main__':
    run_correlation_analysis()
