"""
Multi-Domain Threat Correlator - The Palantir Brain
Fuses environmental, social, infrastructure, traffic, and flood data
to detect compound threats

CORE PHILOSOPHY: Temporal Data Weighting
- Historical events preserved indefinitely
- Recent events weighted higher in correlation analysis
- Aging events contribute to pattern recognition but with reduced priority
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.cluster import DBSCAN
from pathlib import Path
import json

logger = logging.getLogger('THREAT_CORRELATOR')


class ThreatCorrelator:
    """
    Advanced multi-domain correlation engine
    Finds hidden connections across:
    - Environmental (air quality, weather)
    - Traffic (delays, accidents)
    - Social (citizen reports)
    - Infrastructure (outages, disruptions)
    - Floods (water levels, warnings)
    """
    
    def __init__(self):
        self.events = []
        self.threat_clusters = []
        
        # Temporal weighting (NO DELETION)
        self.decay_half_life_hours = 24  # Match DataManager
        self.min_weight = 0.05
    
    def calculate_event_weight(self, event_timestamp):
        """
        Calculate temporal weight for correlation priority
        Older events get lower weight but are NEVER deleted
        """
        now = datetime.now()
        if isinstance(event_timestamp, str):
            event_dt = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
        else:
            event_dt = event_timestamp
        
        # Remove timezone for calculation
        if event_dt.tzinfo:
            event_dt = event_dt.replace(tzinfo=None)
        
        age_hours = (now - event_dt).total_seconds() / 3600
        decay_factor = 0.5 ** (age_hours / self.decay_half_life_hours)
        weight = max(self.min_weight, decay_factor)
        
        return weight
        
    def load_all_events(self, owl_data_path: str = 'owl_data', 
                        traffic_path: str = 'traffic_data_collected') -> List[Dict]:
        """
        Load all events from all domains
        Returns unified event list with standardized format
        """
        events = []
        
        # Load floods
        flood_path = Path(owl_data_path)
        for flood_file in flood_path.glob('**/flood_warnings/*.json'):
            try:
                with open(flood_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    event = self._parse_flood_event(data)
                    if event:
                        events.append(event)
            except:
                pass
        
        # Load environmental data
        for env_file in flood_path.glob('**/environmental_*/*.json'):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    event = self._parse_environmental_event(data)
                    if event:
                        events.append(event)
            except:
                pass
        
        # Load infrastructure
        for infra_file in flood_path.glob('**/infrastructure_*/*.json'):
            try:
                with open(infra_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    event = self._parse_infrastructure_event(data)
                    if event:
                        events.append(event)
            except:
                pass
        
        # Load social threats
        for social_file in flood_path.glob('**/social_*/*.json'):
            try:
                with open(social_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    event = self._parse_social_event(data)
                    if event:
                        events.append(event)
            except:
                pass
        
        # Load traffic
        traffic_p = Path(traffic_path)
        for traffic_file in traffic_p.glob('*.json'):
            try:
                with open(traffic_file, 'r', encoding='utf-8') as f:
                    traffic_list = json.load(f)
                    for traffic_data in traffic_list:
                        event = self._parse_traffic_event(traffic_data)
                        if event:
                            events.append(event)
            except:
                pass
        
        logger.info(f"📊 Loaded {len(events)} total events across all domains")
        self.events = events
        return events
    
    def _parse_flood_event(self, data: Dict) -> Dict:
        """Parse flood warning into standard event format"""
        raw = data.get('raw_data', {})
        metadata = data.get('metadata', {})
        
        # Extract location (approximate center of flood area)
        # In production, would geocode flood area properly
        location_name = raw.get('description', 'Unknown')
        
        return {
            'event_id': f"flood_{metadata.get('data_id', 'unknown')}",
            'domain': 'flood',
            'threat_type': 'flood',
            'timestamp': metadata.get('timestamp'),
            'location_name': location_name,
            'lat': 51.5 + np.random.uniform(-0.3, 0.3),  # Approximate London
            'lon': -0.1 + np.random.uniform(-0.3, 0.3),
            'severity': raw.get('severityLevel', 3),
            'confidence': metadata.get('current_confidence', 0.9),
            'source_type': 'government',
            'text': raw.get('message', ''),
            'raw_data': raw
        }
    
    def _parse_environmental_event(self, data: Dict) -> Dict:
        """Parse environmental data into event format"""
        raw = data.get('raw_data', {})
        
        coords = raw.get('coordinates', {})
        
        return {
            'event_id': f"env_{raw.get('location', 'unknown')}_{data.get('metadata', {}).get('timestamp', '')}",
            'domain': 'environmental',
            'threat_type': raw.get('threat_type', 'air_quality'),
            'timestamp': data.get('metadata', {}).get('timestamp'),
            'location_name': raw.get('location', 'Unknown'),
            'lat': coords.get('latitude') or coords.get('lat'),
            'lon': coords.get('longitude') or coords.get('lon'),
            'severity': 1 if raw.get('threat_level') == 'hazardous' else 2,
            'confidence': data.get('metadata', {}).get('current_confidence', 0.85),
            'source_type': 'sensor',
            'parameter': raw.get('parameter'),
            'value': raw.get('value'),
            'raw_data': raw
        }
    
    def _parse_infrastructure_event(self, data: Dict) -> Dict:
        """Parse infrastructure disruption into event format"""
        raw = data.get('raw_data', {})
        
        return {
            'event_id': f"infra_{raw.get('line', 'road')}_{data.get('metadata', {}).get('timestamp', '')}",
            'domain': 'infrastructure',
            'threat_type': 'infrastructure_disruption',
            'timestamp': data.get('metadata', {}).get('timestamp'),
            'location_name': raw.get('location', raw.get('line', 'Unknown')),
            'lat': 51.5 + np.random.uniform(-0.3, 0.3),
            'lon': -0.1 + np.random.uniform(-0.3, 0.3),
            'severity': 1 if raw.get('threat_level') == 'severe' else 2,
            'confidence': data.get('metadata', {}).get('current_confidence', 0.9),
            'source_type': 'government',
            'infrastructure_type': raw.get('infrastructure_type'),
            'raw_data': raw
        }
    
    def _parse_social_event(self, data: Dict) -> Dict:
        """Parse social media report into event format"""
        raw = data.get('raw_data', {})
        
        return {
            'event_id': f"social_{raw.get('timestamp', '')}",
            'domain': 'social',
            'threat_type': raw.get('threat_type', 'unknown'),
            'timestamp': raw.get('timestamp'),
            'location_name': raw.get('location', 'Unknown'),
            'lat': raw.get('lat'),
            'lon': raw.get('lon'),
            'severity': 1 if raw.get('severity') == 'high' else 2,
            'confidence': raw.get('confidence', 0.4),
            'source_type': 'social',
            'text': raw.get('text', ''),
            'raw_data': raw
        }
    
    def _parse_traffic_event(self, data: Dict) -> Dict:
        """Parse traffic data into event format"""
        # Approximate coordinates for London traffic routes
        coords_map = {
            'Hammersmith': (51.4927, -0.2339),
            'Camden': (51.5390, -0.1426),
            'Stratford': (51.5416, -0.0022),
            'Clapham': (51.4618, -0.1383),
        }
        
        origin = data.get('origin', '')
        dest = data.get('destination', '')
        
        # Try to extract location
        for area, coords in coords_map.items():
            if area in origin or area in dest:
                lat, lon = coords
                break
        else:
            lat, lon = 51.5074, -0.1278
        
        return {
            'event_id': f"traffic_{data.get('timestamp', '')}",
            'domain': 'traffic',
            'threat_type': 'traffic',
            'timestamp': data.get('timestamp'),
            'location_name': origin,
            'lat': lat,
            'lon': lon,
            'severity': 2,
            'confidence': 0.8,
            'source_type': 'sensor',
            'route': data.get('route_name'),
            'duration': data.get('duration'),
            'raw_data': data
        }
    
    def spatial_clustering(self, events: List[Dict], eps_km: float = 2.0, min_samples: int = 2) -> np.ndarray:
        """
        Find geo-spatial event clusters using DBSCAN
        
        Args:
            eps_km: Events within this distance (km) are considered neighbors
            min_samples: Minimum events to form a cluster
        
        Returns:
            Array of cluster labels (-1 = noise, 0+ = cluster ID)
        """
        # Filter events with coordinates
        valid_events = [e for e in events if e.get('lat') and e.get('lon')]
        
        if len(valid_events) < min_samples:
            return np.array([-1] * len(events))
        
        # Extract coordinates
        coords = np.array([[e['lat'], e['lon']] for e in valid_events])
        
        # Convert to radians
        coords_rad = np.radians(coords)
        
        # DBSCAN with haversine metric (great-circle distance)
        clustering = DBSCAN(
            eps=eps_km / 6371.0,  # Earth radius in km
            min_samples=min_samples,
            metric='haversine'
        ).fit(coords_rad)
        
        return clustering.labels_
    
    def detect_compound_threats(self, min_domains: int = 2, time_window_hours: int = 6) -> List[Dict]:
        """
        Detect compound threats: multiple domains reporting in same area+time
        
        TEMPORAL WEIGHTING APPLIED:
        - Recent events weighted higher in threat calculation
        - Old events contribute to pattern but not immediate alert
        - All events preserved for historical analysis
        
        Example compound threat:
        - Air quality spike (environmental)
        - Traffic jam (traffic)
        - "Smoke" reports (social)
        - Road closure (infrastructure)
        → Inference: Building fire or industrial incident
        """
        logger.info("🔍 Detecting compound threats...")
        
        if not self.events:
            return []
        
        # Filter events with temporal weighting (NO DELETION)
        now = datetime.now()
        weighted_events = []
        
        for event in self.events:
            try:
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                hours_ago = (now - timestamp).total_seconds() / 3600
                
                # Calculate temporal weight
                event_weight = self.calculate_event_weight(event['timestamp'])
                
                # Include events in time window OR with sufficient weight
                if hours_ago <= time_window_hours or event_weight > 0.1:
                    event_with_weight = event.copy()
                    event_with_weight['temporal_weight'] = event_weight
                    event_with_weight['hours_ago'] = hours_ago
                    weighted_events.append(event_with_weight)
            except:
                pass
        
        logger.info(f"  {len(weighted_events)} events analyzed (weighted)")
        
        # Separate high-priority recent events from historical context
        recent_events = [e for e in weighted_events if e['temporal_weight'] > 0.5]
        context_events = [e for e in weighted_events if e['temporal_weight'] <= 0.5]
        
        logger.info(f"  {len(recent_events)} high-priority, {len(context_events)} historical context")
        
        # Spatial clustering on recent events
        if len(recent_events) < 2:
            return []
        
        cluster_labels = self.spatial_clustering(recent_events, eps_km=2.0, min_samples=2)
        
        # Analyze each cluster
        compound_threats = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise
                continue
            
            # Get events in this cluster
            cluster_events = [
                e for i, e in enumerate(recent_events) 
                if i < len(cluster_labels) and cluster_labels[i] == cluster_id
            ]
            
            # Count unique domains
            domains = set(e['domain'] for e in cluster_events)
            
            # Compound threat if multiple domains
            if len(domains) >= min_domains:
                # Add historical context from same location
                cluster_with_context = self._add_historical_context(
                    cluster_events, 
                    context_events
                )
                
                threat = self._analyze_compound_threat(cluster_with_context, cluster_id)
                compound_threats.append(threat)
        
        # Sort by weighted threat score
        compound_threats.sort(key=lambda x: x['threat_score'], reverse=True)
        
        logger.info(f"✓ Detected {len(compound_threats)} compound threats")
        
        self.threat_clusters = compound_threats
        return compound_threats
    
    def _add_historical_context(self, cluster_events: List[Dict], 
                                context_events: List[Dict]) -> List[Dict]:
        """
        Add relevant historical events to cluster for pattern analysis
        Weighted lower but provides crucial context
        """
        if not cluster_events:
            return cluster_events
        
        # Get cluster centroid
        cluster_lats = [e['lat'] for e in cluster_events if e.get('lat')]
        cluster_lons = [e['lon'] for e in cluster_events if e.get('lon')]
        
        if not cluster_lats:
            return cluster_events
        
        center_lat = np.mean(cluster_lats)
        center_lon = np.mean(cluster_lons)
        
        # Find historical events near this location
        enhanced = cluster_events.copy()
        
        for hist_event in context_events:
            if hist_event.get('lat') and hist_event.get('lon'):
                # Simple distance check (rough approximation)
                dist_lat = abs(hist_event['lat'] - center_lat)
                dist_lon = abs(hist_event['lon'] - center_lon)
                
                # Within ~2km
                if dist_lat < 0.02 and dist_lon < 0.02:
                    enhanced.append(hist_event)
        
        return enhanced
    
    def _analyze_compound_threat(self, events: List[Dict], cluster_id: int) -> Dict:
        """
        Analyze a cluster of events to determine compound threat
        """
        domains = list(set(e['domain'] for e in events))
        threat_types = list(set(e['threat_type'] for e in events))
        
        # Calculate centroid
        lats = [e['lat'] for e in events if e.get('lat')]
        lons = [e['lon'] for e in events if e.get('lon')]
        
        center_lat = np.mean(lats) if lats else None
        center_lon = np.mean(lons) if lons else None
        
        # Calculate threat score
        threat_score = self._calculate_threat_score(events)
        
        # Infer threat category
        threat_category = self._infer_threat_category(events)
        
        # Generate description
        description = self._generate_threat_description(events, domains, threat_category)
        
        return {
            'cluster_id': cluster_id,
            'threat_category': threat_category,
            'threat_score': threat_score,
            'severity': self._determine_severity(threat_score),
            'event_count': len(events),
            'domains': domains,
            'domain_count': len(domains),
            'threat_types': threat_types,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'location_name': self._get_representative_location(events),
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'events': events,
            'confidence': np.mean([e.get('confidence', 0.5) for e in events])
        }
    
    def _calculate_threat_score(self, events: List[Dict]) -> float:
        """
        Calculate composite threat score
        
        Score = Σ(source_reliability × domain_weight × severity_weight × temporal_decay)
        """
        total_score = 0
        now = datetime.now()
        
        source_weights = {
            'government': 1.0,
            'sensor': 0.9,
            'social': 0.5
        }
        
        domain_weights = {
            'flood': 1.2,
            'environmental': 1.0,
            'infrastructure': 1.3,
            'social': 0.7,
            'traffic': 0.8
        }
        
        for event in events:
            # Source reliability
            source_weight = source_weights.get(event.get('source_type', 'social'), 0.5)
            
            # Domain importance
            domain_weight = domain_weights.get(event.get('domain', 'social'), 0.7)
            
            # Severity (lower = worse)
            severity = event.get('severity', 3)
            severity_weight = 1.0 / severity if severity > 0 else 1.0
            
            # Time decay (fresher = higher score)
            try:
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                hours_ago = (now - timestamp).total_seconds() / 3600
                temporal_decay = np.exp(-hours_ago / 6)  # 6-hour half-life
            except:
                temporal_decay = 0.5
            
            # Confidence
            confidence = event.get('confidence', 0.5)
            
            score = source_weight * domain_weight * severity_weight * temporal_decay * confidence
            total_score += score
        
        return min(total_score, 10.0)  # Cap at 10
    
    def _determine_severity(self, threat_score: float) -> str:
        """Map threat score to severity level"""
        if threat_score >= 7.0:
            return 'CRITICAL'
        elif threat_score >= 5.0:
            return 'HIGH'
        elif threat_score >= 3.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _infer_threat_category(self, events: List[Dict]) -> str:
        """
        Infer threat category from event combination
        """
        domains = set(e['domain'] for e in events)
        threat_types = [e.get('threat_type', '') for e in events]
        event_types = [e.get('event_type', '') for e in events]
        
        # Count specific event types
        has_floods = any(e.get('event_type') == 'flood' or e.get('threat_type') == 'flood' for e in events)
        has_traffic = any(e.get('event_type') == 'traffic' for e in events)
        has_disruptions = any(e.get('event_type') == 'road_disruption' for e in events)
        disruption_count = len([e for e in events if e.get('event_type') == 'road_disruption'])
        
        # Pattern matching with enhanced road disruptions detection
        if has_floods and (has_traffic or has_disruptions):
            return 'flood_impact'
        
        if has_disruptions and disruption_count >= 3:
            # Multiple road disruptions = infrastructure crisis
            return 'infrastructure_disruption'
        
        if 'environmental' in domains and 'social' in domains:
            # Check for fire/chemical keywords
            texts = ' '.join([e.get('text', '') for e in events]).lower()
            if 'fire' in texts or 'smoke' in texts:
                return 'fire_incident'
            if 'chemical' in texts or 'gas' in texts:
                return 'chemical_hazard'
            return 'environmental_hazard'
        
        if 'transport' in domains and len([e for e in events if e.get('event_type') == 'road_disruption']) >= 2:
            return 'infrastructure_disruption'
        
        if 'social' in domains and len([t for t in threat_types if 'safety' in t]) > 0:
            return 'public_safety_incident'
        
        return 'compound_threat'
    
    def _generate_threat_description(self, events: List[Dict], domains: List[str], category: str) -> str:
        """Generate detailed, actionable threat description"""
        location = self._get_representative_location(events)
        
        # Build specific threat details
        details = []
        impacts = []
        recommendations = []
        
        # Group events by type
        floods = [e for e in events if e.get('threat_type') == 'flood' or e.get('event_type') == 'flood']
        traffic = [e for e in events if e.get('domain') == 'transport' and e.get('event_type') == 'traffic']
        disruptions = [e for e in events if e.get('event_type') == 'road_disruption']
        
        # Flood details
        if floods:
            flood_count = len(floods)
            severe_floods = [f for f in floods if f.get('severity', 5) <= 2]
            flood_areas = list(set([f.get('location_name', f.get('location', 'Unknown')) for f in floods[:3]]))
            
            if severe_floods:
                details.append(f"🌊 {len(severe_floods)} SEVERE flood warnings active")
            else:
                details.append(f"🌊 {flood_count} flood warnings")
            details.append(f"   Areas: {', '.join(flood_areas)}")
            impacts.append("Flooded roads and impassable routes")
            recommendations.append("Avoid low-lying areas and flood zones")
        
        # Road disruption details
        if disruptions:
            collisions = [d for d in disruptions if d.get('category') == 'Collisions']
            roadworks = [d for d in disruptions if d.get('category') == 'Works']
            closures = [d for d in disruptions if d.get('has_closures')]
            
            if collisions:
                roads = list(set([c.get('location', 'Unknown road')[:30] for c in collisions[:2]]))
                details.append(f"🚗 {len(collisions)} traffic collisions reported")
                details.append(f"   Locations: {', '.join(roads)}")
                impacts.append("Blocked lanes and emergency response delays")
                recommendations.append("Expect delays and use alternative routes")
            
            if roadworks:
                details.append(f"🚧 {len(roadworks)} active roadworks zones")
            
            if closures:
                details.append(f"⚠️ {len(closures)} roads with FULL or PARTIAL CLOSURES")
                impacts.append("Road network capacity severely reduced")
                recommendations.append("Check TfL before traveling")
        
        # Traffic congestion details
        if traffic:
            routes_affected = list(set([t.get('location', t.get('description', 'Unknown'))[:40] for t in traffic[:3]]))
            durations = [t.get('value') for t in traffic if t.get('value')]
            avg_duration = np.mean(durations) if durations else 0
            
            details.append(f"🚦 {len(traffic)} traffic routes with delays")
            details.append(f"   Routes: {', '.join(routes_affected)}")
            if avg_duration > 0:
                details.append(f"   Average delay: {avg_duration:.0f} minutes")
            impacts.append("Significant travel time increases")
            recommendations.append("Allow extra travel time")
        
        # Build category-specific descriptions
        if category == 'flood_impact':
            headline = f"FLOODING DISRUPTING TRANSPORT NETWORK IN {location.upper()}"
            context = "Active flood warnings are causing cascading failures in road infrastructure and traffic flow."
        elif category == 'infrastructure_disruption':
            headline = f"CRITICAL INFRASTRUCTURE DISRUPTION - {location.upper()}"
            context = "Multiple road incidents and disruptions are creating compound transport failures."
        elif category == 'compound_threat':
            headline = f"⚠️ MULTIPLE CONVERGING THREATS - {location.upper()}"
            context = f"{len(domains)} domains affected: {', '.join(domains)}"
        else:
            headline = f"{category.upper().replace('_', ' ')} - {location.upper()}"
            context = f"Compound threat involving {', '.join(domains)}"
        
        # Assemble full description
        description_parts = [
            headline,
            "",
            context,
            "",
            "CURRENT SITUATION:",
        ]
        description_parts.extend(details)
        
        if impacts:
            description_parts.append("")
            description_parts.append("IMPACTS:")
            description_parts.extend([f"• {imp}" for imp in impacts])
        
        if recommendations:
            description_parts.append("")
            description_parts.append("RECOMMENDED ACTIONS:")
            description_parts.extend([f"• {rec}" for rec in recommendations])
        
        description_parts.append("")
        description_parts.append(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        return "\n".join(description_parts)
    
    def _get_representative_location(self, events: List[Dict]) -> str:
        """Get most common or representative location name"""
        locations = [e.get('location_name', 'Unknown') for e in events if e.get('location_name')]
        
        if not locations:
            return 'Unknown'
        
        # Return most common
        from collections import Counter
        return Counter(locations).most_common(1)[0][0]
    
    def export_threats(self, filepath: str = 'owl_compound_threats.json'):
        """Export compound threats to file"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'threat_count': len(self.threat_clusters),
            'threats': self.threat_clusters
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Threats exported to {filepath}")


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🧠 THREAT CORRELATOR - Multi-Domain Analysis")
    logger.info("=" * 70)
    
    correlator = ThreatCorrelator()
    events = correlator.load_all_events()
    threats = correlator.detect_compound_threats(min_domains=2, time_window_hours=24)
    
    if threats:
        logger.info(f"\n🚨 Top Compound Threats:\n")
        for i, threat in enumerate(threats[:5], 1):
            logger.info(f"{i}. {threat['severity']} - Score: {threat['threat_score']:.2f}")
            logger.info(f"   {threat['description']}")
            logger.info(f"   Domains: {', '.join(threat['domains'])}")
            logger.info(f"   Events: {threat['event_count']}")
            logger.info("")
        
        correlator.export_threats()
