"""
Predictive Intelligence Engine
Forecasts threats before they fully materialize
Uses machine learning to predict cascading events

CORE PHILOSOPHY: Learn from ALL history with temporal weighting
- Train on complete historical dataset (nothing deleted)
- Recent patterns weighted higher for prediction accuracy
- Ancient patterns still inform long-term trend analysis
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.ensemble import IsolationForest
import json

logger = logging.getLogger('PREDICTIVE_ENGINE')


class PredictiveEngine:
    """
    Predict future threats using pattern recognition
    
    Predictions:
    1. Traffic cascade (one jam triggers others)
    2. Flood escalation (minor alert → severe)
    3. Environmental degradation (air quality worsening)
    4. Infrastructure failures (cascading outages)
    """
    
    def __init__(self):
        self.anomaly_detector = None
        self.historical_patterns = defaultdict(list)
        
        # Temporal weighting for predictions (NO DELETION)
        self.decay_half_life_hours = 48  # Longer half-life for pattern learning
        self.min_weight = 0.1  # Even old patterns keep 10% influence
    
    def calculate_pattern_weight(self, pattern_timestamp):
        """
        Calculate temporal weight for pattern matching
        Recent patterns weighted higher, but historical patterns preserved
        """
        now = datetime.now()
        if isinstance(pattern_timestamp, str):
            pattern_dt = datetime.fromisoformat(pattern_timestamp.replace('Z', '+00:00'))
        else:
            pattern_dt = pattern_timestamp
        
        if pattern_dt.tzinfo:
            pattern_dt = pattern_dt.replace(tzinfo=None)
        
        age_hours = (now - pattern_dt).total_seconds() / 3600
        decay_factor = 0.5 ** (age_hours / self.decay_half_life_hours)
        weight = max(self.min_weight, decay_factor)
        
        return weight
        
    def train_anomaly_detector(self, historical_events: List[Dict]):
        """
        Train Isolation Forest on historical data
        Detects anomalous patterns that often precede major incidents
        """
        logger.info("🧠 Training anomaly detector...")
        
        if len(historical_events) < 10:
            logger.warning("Not enough historical data for training")
            return
        
        # Feature engineering
        features = []
        for event in historical_events:
            feature_vector = self._extract_features(event)
            if feature_vector:
                features.append(feature_vector)
        
        if len(features) < 10:
            return
        
        X = np.array(features)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        self.anomaly_detector.fit(X)
        
        logger.info(f"✓ Anomaly detector trained on {len(features)} samples")
    
    def _extract_features(self, event: Dict) -> List[float]:
        """
        Extract numerical features from event
        
        Features:
        - Temporal: hour of day, day of week
        - Severity level
        - Confidence score
        - Domain encoding
        """
        try:
            timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            
            # Temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Event features
            severity = event.get('severity', 3)
            confidence = event.get('confidence', 0.5)
            
            # Domain one-hot encoding
            domain_map = {
                'flood': 0, 'environmental': 1, 'infrastructure': 2,
                'social': 3, 'traffic': 4
            }
            domain_code = domain_map.get(event.get('domain', 'social'), 3)
            
            return [hour, day_of_week, severity, confidence, domain_code]
        except:
            return None
    
    def detect_anomalies(self, current_events: List[Dict]) -> List[Dict]:
        """
        Find anomalous events in current data
        Anomalies often indicate emerging threats
        """
        if not self.anomaly_detector:
            logger.warning("Anomaly detector not trained")
            return []
        
        anomalies = []
        
        for event in current_events:
            features = self._extract_features(event)
            if features:
                prediction = self.anomaly_detector.predict([features])[0]
                
                if prediction == -1:  # Anomaly
                    anomaly_score = self.anomaly_detector.score_samples([features])[0]
                    
                    anomalies.append({
                        'event': event,
                        'anomaly_score': float(anomaly_score),
                        'prediction': 'Anomalous pattern detected'
                    })
        
        return anomalies
    
    def predict_traffic_cascade(self, current_traffic: List[Dict], 
                                 other_events: List[Dict]) -> List[Dict]:
        """
        Predict traffic cascade events with network effect modeling
        
        Advanced Logic:
        - Blocked road → traffic redistributes to parallel routes
        - Calculate overflow capacity on alternative routes
        - Predict secondary congestion points
        - Model time-delayed cascade effects
        """
        predictions = []
        
        # Build road network model
        road_network = self._build_road_network(current_traffic, other_events)
        
        # Find all blocked/disrupted roads
        disruptions = [e for e in other_events if e.get('event_type') == 'road_disruption']
        # Include disruptions with closures OR severity <= 3 (moderate or worse)
        blocked_roads = [d for d in disruptions if d.get('has_closures') or d.get('severity', 5) <= 3]
        
        logger.info(f"📊 Analyzing {len(blocked_roads)} significant road disruptions for cascade effects...")
        
        for disruption in blocked_roads:
            # Identify blocked road - use location as fallback
            blocked_road = disruption.get('road_name', disruption.get('location', 'Unknown'))
            if blocked_road == 'Unknown' and disruption.get('location'):
                blocked_road = disruption.get('location')[:50]
            
            blocked_location = disruption.get('location', '')
            category = disruption.get('category', 'Unknown')
            
            # Find alternative routes that will receive diverted traffic
            alternatives = self._find_alternative_routes(disruption, road_network, disruptions)
            
            logger.info(f"  {blocked_road[:40]}: Found {len(alternatives)} alternative routes")
            
            if alternatives:
                # Calculate overflow impact
                cascade_impact = self._calculate_cascade_impact(disruption, alternatives, current_traffic)
                
                # Generate prediction for each affected alternative route
                for alt in cascade_impact['affected_routes']:
                    predictions.append({
                        'prediction_type': 'traffic_cascade',
                        'trigger_event': f"{category} on {blocked_road}",
                        'trigger_location': blocked_location,
                        'affected_route': alt['route_name'],
                        'affected_location': alt['location'],
                        'risk_score': alt['overflow_risk'],
                        'confidence': alt['confidence'],
                        'expected_delay_increase': alt['expected_delay'],
                        'capacity_overflow': alt['capacity_overflow_pct'],
                        'prediction': alt['prediction_text'],
                        'recommendation': alt['recommendation'],
                        'time_to_impact': alt['time_to_impact'],
                        'factors': alt['factors'],
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Check for existing traffic triggering cascades
        for traffic_event in current_traffic:
            cascade_risk = 0.0
            factors = []
            
            # Check for nearby environmental threats
            for other in other_events:
                if other.get('domain') == 'environmental':
                    if self._is_nearby(traffic_event, other, threshold_km=2):
                        cascade_risk += 0.3
                        factors.append('environmental_threat_nearby')
                
                # Check for infrastructure disruption
                if other.get('domain') == 'infrastructure':
                    if self._is_nearby(traffic_event, other, threshold_km=1):
                        cascade_risk += 0.4
                        factors.append('infrastructure_disruption')
                
                # Check for flood impact
                if other.get('threat_type') == 'flood':
                    if self._is_nearby(traffic_event, other, threshold_km=3):
                        cascade_risk += 0.5
                        factors.append('flood_risk')
            
            # If cascade risk above threshold, create prediction
            if cascade_risk >= 0.5:
                predictions.append({
                    'prediction_type': 'traffic_cascade',
                    'affected_route': traffic_event.get('location', 'Unknown'),
                    'location': traffic_event.get('location'),
                    'risk_score': min(cascade_risk, 1.0),
                    'confidence': 0.6 + (0.2 * len(factors)),
                    'factors': factors,
                    'prediction': f"Traffic cascade likely in next 30-60 minutes",
                    'recommendation': 'Reroute traffic preemptively',
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"✓ Generated {len(predictions)} cascade predictions")
        return predictions
    
    def _build_road_network(self, traffic_events: List[Dict], all_events: List[Dict]) -> Dict:
        """Build simple road network graph from available data"""
        network = {
            'roads': {},
            'connections': defaultdict(list)
        }
        
        # Map known roads
        for event in all_events:
            if event.get('event_type') == 'road_disruption':
                road_id = event.get('road_name', event.get('location', 'unknown'))
                # Use location as fallback if road_name not available
                if road_id == 'unknown' and event.get('location'):
                    road_id = event.get('location')[:50]  # Use first part of location
                
                if road_id not in network['roads']:
                    network['roads'][road_id] = {
                        'name': road_id,
                        'location': event.get('location', ''),
                        'lat': event.get('lat'),
                        'lon': event.get('lon'),
                        'disruptions': []
                    }
                network['roads'][road_id]['disruptions'].append(event)
        
        # Add traffic routes
        for traffic in traffic_events:
            route_name = traffic.get('location', traffic.get('description', 'unknown'))
            if route_name not in network['roads']:
                network['roads'][route_name] = {
                    'name': route_name,
                    'location': traffic.get('location', ''),
                    'lat': traffic.get('lat'),
                    'lon': traffic.get('lon'),
                    'current_traffic': traffic
                }
        
        logger.info(f"  Built network with {len(network['roads'])} roads")
        
        # Build connections based on proximity (simplified - real implementation would use map data)
        road_list = list(network['roads'].items())
        connections_count = 0
        for i, (road_id1, road1) in enumerate(road_list):
            for road_id2, road2 in road_list[i+1:]:
                # If roads are within 5km, consider them connected/parallel
                if road1.get('lat') and road2.get('lat'):
                    dist = self._haversine_distance(
                        road1['lat'], road1['lon'],
                        road2['lat'], road2['lon']
                    )
                    if dist <= 5.0:  # 5km threshold for alternative routes
                        network['connections'][road_id1].append(road_id2)
                        network['connections'][road_id2].append(road_id1)
                        connections_count += 1
        
        logger.info(f"  Identified {connections_count} road connections")
        
        return network
    
    def _find_alternative_routes(self, disruption: Dict, network: Dict, 
                                 all_disruptions: List[Dict]) -> List[Dict]:
        """Find alternative routes that will receive diverted traffic"""
        blocked_road = disruption.get('road_name', disruption.get('location', 'unknown'))
        if blocked_road == 'unknown' and disruption.get('location'):
            blocked_road = disruption.get('location')[:50]
        
        alternatives = []
        
        # Get connected roads from network
        connected_roads = network['connections'].get(blocked_road, [])
        
        for alt_road_id in connected_roads:
            alt_road = network['roads'].get(alt_road_id, {})
            
            # Check if alternative is also significantly disrupted
            # Allow minor disruptions, only exclude if severe or closed
            is_blocked = any(
                (d.get('road_name', d.get('location', ''))[:50] == alt_road_id) and 
                (d.get('has_closures') or d.get('severity', 5) <= 1)
                for d in all_disruptions
            )
            
            if not is_blocked:
                alternatives.append({
                    'road_id': alt_road_id,
                    'name': alt_road.get('name', alt_road_id),
                    'location': alt_road.get('location', ''),
                    'lat': alt_road.get('lat'),
                    'lon': alt_road.get('lon'),
                    'current_traffic': alt_road.get('current_traffic')
                })
        
        return alternatives
    
    def _calculate_cascade_impact(self, disruption: Dict, alternatives: List[Dict],
                                  current_traffic: List[Dict]) -> Dict:
        """Calculate expected traffic overflow impact on alternative routes"""
        
        affected_routes = []
        
        for alt in alternatives:
            # Estimate diverted traffic (simplified model)
            # Real implementation would use actual traffic volume data
            diversion_factor = 0.3 + (0.4 if disruption.get('has_closures') else 0.2)
            
            # Check current traffic on alternative
            current_delay = 0
            if alt.get('current_traffic'):
                current_delay = alt['current_traffic'].get('value', 0)
            
            # Predict overflow
            expected_additional_delay = current_delay * diversion_factor
            total_expected_delay = current_delay + expected_additional_delay
            capacity_overflow = (diversion_factor * 100)  # Percentage increase
            
            # Risk score based on overflow
            overflow_risk = min(diversion_factor + (0.3 if current_delay > 30 else 0), 1.0)
            
            # Time to impact (minutes)
            time_to_impact = "5-15 minutes" if disruption.get('has_closures') else "15-30 minutes"
            
            # Confidence based on certainty of diversion
            confidence = 0.75 if disruption.get('has_closures') else 0.60
            
            # Build prediction text
            if capacity_overflow > 50:
                severity = "SEVERE"
                prediction_text = f"SEVERE congestion expected on {alt['name']} due to traffic diversion from {disruption.get('road_name')}. Expect delays to increase by ~{expected_additional_delay:.0f} minutes."
                recommendation = f"URGENT: Activate traffic management on {alt['name']}. Consider additional diversions. Alert drivers via navigation apps."
            elif capacity_overflow > 30:
                severity = "HIGH"
                prediction_text = f"HIGH traffic increase expected on {alt['name']} as drivers avoid {disruption.get('road_name')}. Delays may increase by ~{expected_additional_delay:.0f} minutes."
                recommendation = f"Monitor {alt['name']} closely. Prepare traffic control measures. Update journey time estimates."
            else:
                severity = "MODERATE"
                prediction_text = f"MODERATE traffic increase expected on {alt['name']} due to nearby disruption. Minor delays anticipated."
                recommendation = f"Continue monitoring {alt['name']}. Inform drivers of alternative route availability."
            
            affected_routes.append({
                'route_name': alt['name'],
                'location': alt['location'],
                'overflow_risk': overflow_risk,
                'confidence': confidence,
                'expected_delay': f"{expected_additional_delay:.0f} minutes",
                'total_delay': f"{total_expected_delay:.0f} minutes",
                'capacity_overflow_pct': f"{capacity_overflow:.0f}%",
                'severity': severity,
                'prediction_text': prediction_text,
                'recommendation': recommendation,
                'time_to_impact': time_to_impact,
                'factors': [
                    f"{disruption.get('category', 'Disruption')} on {disruption.get('road_name')}",
                    f"Closure: {'Yes' if disruption.get('has_closures') else 'No'}",
                    f"Diversion factor: {diversion_factor:.0%}",
                    f"Current delay: {current_delay:.0f} min"
                ]
            })
        
        return {
            'disruption_location': disruption.get('location'),
            'affected_routes': affected_routes,
            'total_affected': len(affected_routes)
        }
    
    def predict_flood_escalation(self, flood_events: List[Dict], 
                                  weather_data: List[Dict]) -> List[Dict]:
        """
        Predict flood escalation
        
        Logic:
        - Current flood alert + heavy rain forecast → severe flood risk
        - River level trend analysis (rising vs falling)
        """
        predictions = []
        
        for flood in flood_events:
            escalation_risk = 0.0
            factors = []
            
            # Current severity
            current_severity = flood.get('severity', 3)
            
            # If already at alert level (3), check for escalation
            if current_severity >= 2:
                # Check weather
                for weather in weather_data:
                    if self._is_nearby(flood, weather, threshold_km=10):
                        # Heavy rain indicator
                        if 'rain' in str(weather.get('raw_data', {})).lower():
                            escalation_risk += 0.4
                            factors.append('heavy_rain_forecast')
                
                # Time-based: floods tend to worsen overnight
                hour = datetime.now().hour
                if 18 <= hour or hour <= 6:
                    escalation_risk += 0.2
                    factors.append('overnight_worsening')
                
                if escalation_risk >= 0.4:
                    predictions.append({
                        'prediction_type': 'flood_escalation',
                        'location': flood.get('location_name'),
                        'current_severity': current_severity,
                        'predicted_severity': max(current_severity - 1, 1),  # Lower number = worse
                        'risk_score': min(escalation_risk, 1.0),
                        'confidence': 0.5 + (0.3 * len(factors)),
                        'factors': factors,
                        'prediction': 'Flood likely to escalate to higher severity',
                        'recommendation': 'Issue preemptive evacuation warnings',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return predictions
    
    def predict_compound_threat_formation(self, all_events: List[Dict]) -> List[Dict]:
        """
        Predict when separate events will combine into compound threat
        
        Logic:
        - Multiple weak threats converging spatially → compound threat forming
        - Social media velocity increase → incident about to break
        """
        predictions = []
        
        # Group events by proximity
        clusters = self._simple_spatial_cluster(all_events, threshold_km=2)
        
        for cluster in clusters:
            if len(cluster) >= 2:
                domains = set(e['domain'] for e in cluster)
                
                # If different domains clustering
                if len(domains) >= 2:
                    compound_risk = len(domains) * 0.25 + (len(cluster) * 0.1)
                    
                    predictions.append({
                        'prediction_type': 'compound_threat_forming',
                        'location': cluster[0].get('location_name'),
                        'domains_involved': list(domains),
                        'event_count': len(cluster),
                        'risk_score': min(compound_risk, 1.0),
                        'confidence': 0.4 + (len(domains) * 0.15),
                        'prediction': f'Compound threat forming: {len(domains)} domains converging',
                        'recommendation': 'Increase monitoring and prepare coordinated response',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return predictions
    
    def _is_nearby(self, event1: Dict, event2: Dict, threshold_km: float = 2.0) -> bool:
        """Check if two events are geographically nearby"""
        lat1 = event1.get('lat')
        lon1 = event1.get('lon')
        lat2 = event2.get('lat')
        lon2 = event2.get('lon')
        
        if not all([lat1, lon1, lat2, lon2]):
            return False
        
        distance = self._haversine_distance(lat1, lon1, lat2, lon2)
        return distance <= threshold_km
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points on Earth
        Returns distance in kilometers
        """
        # Haversine distance
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _simple_spatial_cluster(self, events: List[Dict], threshold_km: float = 2.0) -> List[List[Dict]]:
        """Simple spatial clustering"""
        clusters = []
        used = set()
        
        for i, event1 in enumerate(events):
            if i in used:
                continue
            
            cluster = [event1]
            used.add(i)
            
            for j, event2 in enumerate(events):
                if j <= i or j in used:
                    continue
                
                if self._is_nearby(event1, event2, threshold_km):
                    cluster.append(event2)
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def generate_all_predictions(self, events: List[Dict]) -> List[Dict]:
        """
        Generate all types of predictions
        """
        logger.info("🔮 Generating predictive intelligence...")
        
        # Separate by domain
        traffic_events = [e for e in events if e.get('domain') == 'traffic']
        flood_events = [e for e in events if e.get('domain') == 'flood']
        environmental_events = [e for e in events if e.get('domain') == 'environmental']
        
        all_predictions = []
        
        # Traffic cascade predictions
        traffic_predictions = self.predict_traffic_cascade(traffic_events, events)
        all_predictions.extend(traffic_predictions)
        
        # Flood escalation predictions
        flood_predictions = self.predict_flood_escalation(flood_events, environmental_events)
        all_predictions.extend(flood_predictions)
        
        # Compound threat predictions
        compound_predictions = self.predict_compound_threat_formation(events)
        all_predictions.extend(compound_predictions)
        
        # Anomaly detection
        if len(events) >= 10:
            self.train_anomaly_detector(events)
            anomalies = self.detect_anomalies(events[-20:])  # Check recent events
            all_predictions.extend(anomalies)
        
        logger.info(f"✓ Generated {len(all_predictions)} predictions")
        
        return all_predictions
    
    def export_predictions(self, predictions: List[Dict], filepath: str = 'owl_predictions.json'):
        """Export predictions to file"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'prediction_count': len(predictions),
            'predictions': predictions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Predictions exported to {filepath}")


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🔮 PREDICTIVE ENGINE - Threat Forecasting")
    logger.info("=" * 70)
    
    # This would be integrated with real events
    engine = PredictiveEngine()
    
    logger.info("✓ Predictive engine initialized")
