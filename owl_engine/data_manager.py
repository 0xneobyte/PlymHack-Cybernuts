"""
Data Manager - Unified data storage and timeline analysis
Handles: Traffic, Floods, Weather, Air Quality, Health Data

CORE PHILOSOPHY: No data gets deleted, only weighted by temporal relevance
- All historical data preserved for pattern analysis
- Temporal decay function calculates current relevance (0.0-1.0)
- Recent events = high weight, old events = low weight but still queryable
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# London location database for geocoding
LONDON_LOCATIONS = {
    # Central London
    'trafalgar square': (51.5080, -0.1281),
    'city of london': (51.5155, -0.0922),
    'westminster': (51.4975, -0.1357),
    'kensington': (51.4991, -0.1938),
    
    # North London
    'camden town': (51.5390, -0.1426),
    'finchley': (51.5975, -0.1882),
    'barnet': (51.6252, -0.1517),
    'wembley': (51.5536, -0.2817),
    
    # East London
    'stratford': (51.5416, -0.0022),
    'ilford': (51.5588, 0.0883),
    'woolwich': (51.4892, 0.0648),
    'greenwich': (51.4826, -0.0077),
    
    # South London  
    'clapham common': (51.4618, -0.1383),
    'clapham': (51.4618, -0.1383),
    'croydon': (51.3762, -0.0982),
    'dulwich': (51.4500, -0.0900),
    
    # West London
    'hammersmith': (51.4927, -0.2339),
    'chiswick': (51.4821, -0.2679),
    'kew': (51.4878, -0.2850),
    'heathrow': (51.4700, -0.4543),
    'heathrow airport': (51.4700, -0.4543),
    
    # Flood-affected areas (Thames Valley)
    'denham': (51.5768, -0.4968),
    'west drayton': (51.5070, -0.4700),
    'yiewsley': (51.5119, -0.4781),
    'staines': (51.4343, -0.5092),
    'wraysbury': (51.4571, -0.5534),
    'iver': (51.5209, -0.5092),
    'colnbrook': (51.4873, -0.5260),
    'shepperton': (51.3953, -0.4574),
    'molesey': (51.4001, -0.3617),
    'teddington': (51.4254, -0.3377),
    'hampton': (51.4167, -0.3683),
    'thames ditton': (51.3884, -0.3321),
    'putney': (51.4646, -0.2140),
    'purfleet': (51.4864, 0.2384),
    
    # Rivers and waterways
    'river thames': (51.5074, -0.1278),
    'river colne': (51.5400, -0.4800),
    'colne brook': (51.5000, -0.5000),
    
    # Major London Roads (TfL strategic network)
    'a13': (51.5000, 0.1000),  # A13 Thames Gateway
    'a40': (51.5200, -0.2000),  # A40 Westway
    'a12': (51.5500, 0.0500),  # A12 Eastern Avenue
    'a406': (51.5800, -0.1200),  # A406 North Circular
    'a205': (51.4400, -0.1000),  # A205 South Circular
    'a4': (51.5000, -0.3000),  # A4 Great West Road
    'a105': (51.6100, -0.1100),  # A105 Green Lanes (Enfield)
    'a4020': (51.5150, -0.3100),  # A4020 Uxbridge Road (Ealing)
    'a23': (51.4000, -0.1200),  # A23 Brighton Road
    'a10': (51.5500, -0.0600),  # A10 Great Cambridge Road
    'a2': (51.4700, 0.0500),  # A2 Dover Road
    
    # London Boroughs for road context
    'havering': (51.5779, 0.2120),
    'ealing': (51.5130, -0.3089),
    'enfield': (51.6522, -0.0808),
    'hammersmith & fulham': (51.4927, -0.2339),
    'kensington & chelsea': (51.4991, -0.1938),
    
    # Default London
    'london': (51.5074, -0.1278),
    'hertfordshire': (51.8090, -0.2376),
    'buckinghamshire': (51.8133, -0.8084),
}

def geocode_location(location_str):
    """
    Convert location string to lat/lon coordinates
    Returns (lat, lon) or (None, None) if not found
    """
    if not location_str or not isinstance(location_str, str):
        return None, None
    
    # Clean up location string
    location = location_str.lower().strip()
    location = location.replace(', london', '').replace(' london', '')
    location = location.replace(',', '').strip()
    
    # Direct lookup
    if location in LONDON_LOCATIONS:
        return LONDON_LOCATIONS[location]
    
    # Partial match (check if any key is in the location string)
    for key, coords in LONDON_LOCATIONS.items():
        if key in location or location in key:
            return coords
    
    # Extract first recognizable location from comma-separated list
    parts = location_str.lower().split(',')
    for part in parts:
        part = part.strip()
        if part in LONDON_LOCATIONS:
            return LONDON_LOCATIONS[part]
    
    # Default to central London
    return 51.5074, -0.1278


def extract_flood_location(description, message, area_name):
    """
    Extract specific location from flood description and message
    Returns (lat, lon) tuple
    """
    # Try description first (e.g., "Lower River Colne and Frays River")
    if description:
        desc_lower = description.lower()
        # Check for specific location keywords in description
        # Sort by length to prioritize longer/more specific matches
        sorted_locations = sorted(LONDON_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for location, coords in sorted_locations:
            # Use word boundaries to avoid partial matches (e.g., "iver" in "river")
            if f' {location} ' in f' {desc_lower} ' or desc_lower.startswith(location) or desc_lower.endswith(location):
                return coords
    
    # Parse message for specific locations (comma-separated list)
    if message:
        msg_lower = message.lower()
        # Extract location names from message
        # Sort by length to prioritize longer/more specific matches
        sorted_locations = sorted(LONDON_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for location, coords in sorted_locations:
            # Use word boundaries to avoid partial matches
            if f' {location},' in msg_lower or f' {location} ' in f' {msg_lower} ' or msg_lower.startswith(location):
                return coords
    
    # Fallback to area name with better regional defaults
    area_lower = area_name.lower() if area_name else ''
    
    # Regional defaults based on area names
    if 'hertfordshire' in area_lower and 'north london' in area_lower:
        return 51.7500, -0.3362  # North London / Herts border
    elif 'thames valley' in area_lower or 'buckinghamshire' in area_lower:
        return 51.5000, -0.6000  # Thames Valley
    elif 'kent' in area_lower and ('south london' in area_lower or 'east sussex' in area_lower):
        return 51.3500, 0.0000  # Southeast London / Kent
    elif 'east anglia' in area_lower:
        return 52.0500, 1.1500  # East Anglia
    elif 'surrey' in area_lower:
        return 51.2500, -0.4000  # Surrey
    elif 'essex' in area_lower:
        return 51.7500, 0.4700  # Essex
    elif 'thames' in area_lower or 'river thames' in area_lower:
        return 51.4500, -0.3500  # Thames River area
    elif 'hertfordshire' in area_lower:
        return 51.8090, -0.2376  # Hertfordshire
    elif 'kent' in area_lower:
        return 51.3500, 0.0000  # Kent
    
    # Default to central London
    return 51.5074, -0.1278


class DataManager:
    """Centralized data storage and retrieval for timeline analysis"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.traffic_dir = self.base_path / "traffic_data_collected"
        self.owl_data_dir = self.base_path / "owl_data"
        self.owl_extracted_dir = self.base_path / "owl_extracted"
        self.events_dir = self.base_path / "intelligence_events"
        self.events_dir.mkdir(exist_ok=True)
        
        # Temporal weighting parameters (NO DELETION POLICY)
        self.decay_half_life_hours = 24  # Events decay to 50% relevance after 24h
        self.min_weight = 0.05  # Even very old events retain 5% weight
    
    def calculate_temporal_weight(self, event_timestamp, reference_time=None):
        """
        Calculate temporal relevance weight using exponential decay
        
        Philosophy: No data deleted - older events weighted lower but preserved
        Weight range: [min_weight, 1.0]
        
        Args:
            event_timestamp: When event occurred
            reference_time: Current time (default: now)
        
        Returns:
            float: Temporal weight (0.05-1.0)
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        # Convert to datetime if needed
        if isinstance(event_timestamp, str):
            event_timestamp = pd.to_datetime(event_timestamp)
        if isinstance(reference_time, str):
            reference_time = pd.to_datetime(reference_time)
        
        # Ensure timezone-aware comparison
        if event_timestamp.tzinfo is None:
            event_timestamp = event_timestamp.replace(tzinfo=None)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=None)
        else:
            reference_time = reference_time.replace(tzinfo=None)
        
        # Calculate age in hours
        age_hours = (reference_time - event_timestamp).total_seconds() / 3600
        
        # Exponential decay: weight = 0.5^(age/half_life)
        decay_factor = 0.5 ** (age_hours / self.decay_half_life_hours)
        
        # Apply minimum weight floor
        weight = max(self.min_weight, decay_factor)
        
        return weight
    
    def normalize_traffic_data(self, traffic_json):
        """Convert traffic data to DataFrame with proper types"""
        records = []
        
        for record in traffic_json:
            # Parse duration: "26 min" -> 26
            duration_str = record.get('duration', '0 min')
            duration_match = re.search(r'(\d+)', duration_str)
            duration_minutes = int(duration_match.group(1)) if duration_match else 0
            
            normalized = {
                'timestamp': pd.to_datetime(record['timestamp']),
                'route_id': record['route_name'].lower().replace(' ', '_'),
                'route_name': record['route_name'],
                'origin': record['origin'],
                'destination': record['destination'],
                'duration_minutes': duration_minutes,
                'duration_raw': duration_str,
                'status': record['status'],
                'url': record.get('url', '')
            }
            records.append(normalized)
        
        return pd.DataFrame(records) if records else pd.DataFrame()
    
    def load_all_traffic_history(self):
        """Load ALL traffic collections"""
        all_data = []
        
        if not self.traffic_dir.exists():
            return pd.DataFrame()
        
        for json_file in self.traffic_dir.glob("traffic_collection_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    df = self.normalize_traffic_data(data)
                    if not df.empty:
                        all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values('timestamp')
            return combined
        return pd.DataFrame()
    
    def load_all_flood_warnings(self):
        """Load ALL flood warnings with proper timestamps"""
        all_floods = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        # Search all date folders
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            flood_dir = date_folder / "flood_warnings"
            if not flood_dir.exists():
                continue
            
            for json_file in flood_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Extract from metadata structure
                        metadata = data.get('metadata', {})
                        raw_data = data.get('raw_data', {})
                        
                        timestamp = metadata.get('timestamp') or raw_data.get('timeRaised') or raw_data.get('timeMessageChanged')
                        
                        if timestamp:
                            all_floods.append({
                                'timestamp': pd.to_datetime(timestamp),
                                'area': raw_data.get('eaAreaName', 'Unknown'),
                                'severity': raw_data.get('severityLevel', 3),
                                'severity_text': raw_data.get('severity', 'Unknown'),
                                'description': raw_data.get('description', ''),
                                'message': raw_data.get('message', '')[:200],
                                'flood_area_id': raw_data.get('floodAreaID', ''),
                                'source': 'UK Environment Agency'
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse flood data {json_file}: {e}")
        
        return pd.DataFrame(all_floods) if all_floods else pd.DataFrame()
    
    def load_all_weather_data(self):
        """Load weather data from Met Office collections"""
        all_weather = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            weather_dir = date_folder / "weather"
            if weather_dir.exists():
                for json_file in weather_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            metadata = data.get('metadata', {})
                            raw_data = data.get('raw_data', {})
                            
                            all_weather.append({
                                'timestamp': pd.to_datetime(metadata.get('timestamp')),
                                'temperature': raw_data.get('temperature'),
                                'feels_like': raw_data.get('feels_like'),
                                'humidity': raw_data.get('humidity'),
                                'pressure': raw_data.get('pressure'),
                                'wind_speed': raw_data.get('wind_speed'),
                                'description': raw_data.get('description', ''),
                                'visibility': raw_data.get('visibility'),
                                'source': 'Met Office'
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse weather data: {e}")
        
        return pd.DataFrame(all_weather) if all_weather else pd.DataFrame()
    
    def load_all_air_quality_data(self):
        """Load air quality data from environmental monitor"""
        all_air_quality = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            air_quality_dir = date_folder / "air_quality"
            if air_quality_dir.exists():
                for json_file in air_quality_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            metadata = data.get('metadata', {})
                            raw_data = data.get('raw_data', {})
                            
                            all_air_quality.append({
                                'timestamp': pd.to_datetime(metadata.get('timestamp')),
                                'pm25': raw_data.get('pm25'),
                                'pm10': raw_data.get('pm10'),
                                'no2': raw_data.get('no2'),
                                'o3': raw_data.get('o3'),
                                'co': raw_data.get('co'),
                                'aqi': raw_data.get('aqi'),
                                'threat_level': raw_data.get('threat_level', 'unknown'),
                                'location': raw_data.get('location', 'London'),
                                'source': raw_data.get('source', 'OpenAQ')
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse air quality data: {e}")
        
        return pd.DataFrame(all_air_quality) if all_air_quality else pd.DataFrame()
    
    def load_all_health_data(self):
        """Load health data (hospitals, A&E capacity, etc.)"""
        all_health = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            health_dir = date_folder / "health"
            if health_dir.exists():
                for json_file in health_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            metadata = data.get('metadata', {})
                            raw_data = data.get('raw_data', {})
                            
                            all_health.append({
                                'timestamp': pd.to_datetime(metadata.get('timestamp')),
                                'hospital': raw_data.get('hospital_name', 'Unknown'),
                                'ae_capacity': raw_data.get('ae_capacity'),
                                'wait_time': raw_data.get('average_wait_time'),
                                'bed_availability': raw_data.get('bed_availability'),
                                'status': raw_data.get('status', 'unknown'),
                                'source': raw_data.get('source', 'NHS Digital')
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse health data: {e}")
        
        return pd.DataFrame(all_health) if all_health else pd.DataFrame()
    
    def load_all_road_disruptions(self):
        """Load road disruptions from TfL (collisions, roadworks, incidents)"""
        all_disruptions = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            disruptions_dir = date_folder / "road_disruptions"
            if disruptions_dir.exists():
                for json_file in disruptions_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            metadata = data.get('metadata', {})
                            raw_data = data.get('raw_data', {})
                            
                            all_disruptions.append({
                                'timestamp': pd.to_datetime(raw_data.get('timestamp')),
                                'disruption_id': raw_data.get('disruption_id', 'unknown'),
                                'road_name': raw_data.get('road_name', 'Unknown'),
                                'category': raw_data.get('category', 'Unknown'),
                                'sub_category': raw_data.get('sub_category', ''),
                                'severity': raw_data.get('severity', 3),
                                'location': raw_data.get('location', 'London'),
                                'comments': raw_data.get('comments', ''),
                                'current_update': raw_data.get('current_update', ''),
                                'status': raw_data.get('status', 'Active'),
                                'has_closures': raw_data.get('has_closures', False),
                                'source': raw_data.get('source', 'TfL')
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse road disruption data: {e}")
        
        return pd.DataFrame(all_disruptions) if all_disruptions else pd.DataFrame()
    
    def load_all_video_analytics(self):
        """Load Abbey Road video analytics data"""
        all_analytics = []
        
        if not self.owl_data_dir.exists():
            return pd.DataFrame()
        
        for date_folder in self.owl_data_dir.glob("*"):
            if not date_folder.is_dir():
                continue
                
            video_dir = date_folder / "video_analytics"
            if video_dir.exists():
                for json_file in video_dir.glob("abbey_road_*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # Extract hourly stats if available
                            if 'hourly_stats' in data:
                                stats = data['hourly_stats']
                                
                                # Parse vehicle counts
                                vehicle_counts = stats.get('vehicle_counts', {})
                                
                                all_analytics.append({
                                    'timestamp': pd.to_datetime(stats.get('timestamp')),
                                    'hour': stats.get('hour'),
                                    'frame_count': stats.get('frame_count', 0),
                                    'avg_crowd_density': stats.get('avg_crowd_density', 0),
                                    # Vehicle counts by direction
                                    'cars_north': vehicle_counts.get('car', {}).get('north', 0),
                                    'cars_south': vehicle_counts.get('car', {}).get('south', 0),
                                    'cars_east': vehicle_counts.get('car', {}).get('east', 0),
                                    'cars_west': vehicle_counts.get('car', {}).get('west', 0),
                                    'buses_north': vehicle_counts.get('bus', {}).get('north', 0),
                                    'buses_south': vehicle_counts.get('bus', {}).get('south', 0),
                                    'motorcycles_north': vehicle_counts.get('motorcycle', {}).get('north', 0),
                                    'motorcycles_south': vehicle_counts.get('motorcycle', {}).get('south', 0),
                                    'location': 'Abbey Road Crossing',
                                    'source': 'EarthCam Abbey Road'
                                })
                            
                            # Also process individual detections if present
                            if 'detections' in data:
                                for detection in data['detections']:
                                    timestamp = pd.to_datetime(detection.get('timestamp'))
                                    all_analytics.append({
                                        'timestamp': timestamp,
                                        'vehicle_count': detection.get('vehicle_count', 0),
                                        'person_count': detection.get('person_count', 0),
                                        'location': 'Abbey Road Crossing',
                                        'source': 'EarthCam Abbey Road',
                                        'type': 'snapshot'
                                    })
                    except Exception as e:
                        logger.warning(f"Failed to parse video analytics: {e}")
        
        return pd.DataFrame(all_analytics) if all_analytics else pd.DataFrame()
    
    def create_unified_timeline(self):
        """
        Combine ALL events into ONE chronological timeline
        Sources: Traffic, Floods, Weather, Air Quality, Health
        
        INCLUDES TEMPORAL WEIGHTING: No data deleted, only weighted by age
        """
        timeline = []
        reference_time = datetime.now()
        
        # Traffic events
        traffic_df = self.load_all_traffic_history()
        if not traffic_df.empty:
            for _, row in traffic_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                
                # Geocode origin and destination
                origin_lat, origin_lon = geocode_location(row['origin'])
                dest_lat, dest_lon = geocode_location(row['destination'])
                
                # Use origin as primary location, store destination in data
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'traffic',
                    'domain': 'transport',
                    'severity': self._calculate_traffic_severity(row['duration_minutes']),
                    'description': f"{row['route_name']}: {row['duration_minutes']} min",
                    'location': f"{row['origin']} → {row['destination']}",
                    'lat': origin_lat,
                    'lon': origin_lon,
                    'dest_lat': dest_lat,
                    'dest_lon': dest_lon,
                    'value': row['duration_minutes'],
                    'unit': 'minutes',
                    'source': 'Google Maps',
                    'temporal_weight': weight,  # NEVER DELETED - decay only
                    'data': row.to_dict()
                })
        
        # Flood events
        flood_df = self.load_all_flood_warnings()
        if not flood_df.empty:
            for _, row in flood_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                
                # Extract specific flood location from description and message
                lat, lon = extract_flood_location(
                    row.get('description', ''),
                    row.get('message', ''),
                    row.get('area', '')
                )
                
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'flood',
                    'domain': 'environmental',
                    'severity': row['severity'],
                    'description': row['description'],
                    'location': row['area'],
                    'lat': lat,
                    'lon': lon,
                    'value': row['severity'],
                    'unit': 'severity_level',
                    'source': row['source'],
                    'temporal_weight': weight,  # Historical floods preserved
                    'data': row.to_dict()
                })
        
        # Weather events
        weather_df = self.load_all_weather_data()
        if not weather_df.empty:
            for _, row in weather_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'weather',
                    'domain': 'environmental',
                    'severity': self._calculate_weather_severity(row),
                    'description': f"{row.get('description', 'Weather')}: {row.get('temperature', 'N/A')}°C",
                    'location': 'London',
                    'value': row.get('temperature'),
                    'unit': 'celsius',
                    'source': row['source'],
                    'temporal_weight': weight,  # Weather history maintained
                    'data': row.to_dict()
                })
        
        # Air Quality events
        air_df = self.load_all_air_quality_data()
        if not air_df.empty:
            for _, row in air_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'air_quality',
                    'domain': 'health',
                    'severity': self._calculate_air_quality_severity(row),
                    'description': f"AQI: {row.get('aqi', 'N/A')}, PM2.5: {row.get('pm25', 'N/A')}",
                    'location': row.get('location', 'London'),
                    'value': row.get('aqi'),
                    'unit': 'AQI',
                    'source': row['source'],
                    'temporal_weight': weight,  # Air quality trends preserved
                    'data': row.to_dict()
                })
        
        # Health events
        health_df = self.load_all_health_data()
        if not health_df.empty:
            for _, row in health_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'health',
                    'domain': 'health',
                    'severity': self._calculate_health_severity(row),
                    'description': f"{row['hospital']}: {row.get('wait_time', 'N/A')} min wait",
                    'location': row['hospital'],
                    'value': row.get('wait_time'),
                    'unit': 'minutes',
                    'source': row['source'],
                    'temporal_weight': weight,  # Health pattern history
                    'data': row.to_dict()
                })
        
        # Road Disruptions
        disruptions_df = self.load_all_road_disruptions()
        if not disruptions_df.empty:
            for _, row in disruptions_df.iterrows():
                weight = self.calculate_temporal_weight(row['timestamp'], reference_time)
                
                # Geocode disruption location
                lat, lon = geocode_location(row['location'])
                
                # Build description from details
                desc_parts = [row['road_name'], row['category']]
                if row.get('sub_category'):
                    desc_parts.append(row['sub_category'])
                description = ': '.join(desc_parts)
                
                # Add details from comments if available
                if row.get('current_update'):
                    description += f" - {row['current_update'][:100]}"
                
                timeline.append({
                    'timestamp': row['timestamp'],
                    'event_type': 'road_disruption',
                    'domain': 'transport',
                    'severity': row['severity'],
                    'description': description,
                    'location': row['location'],
                    'lat': lat,
                    'lon': lon,
                    'value': row['severity'],
                    'unit': 'severity_level',
                    'source': row['source'],
                    'temporal_weight': weight,  # Disruption history preserved
                    'category': row['category'],
                    'has_closures': row.get('has_closures', False),
                    'data': row.to_dict()
                })
        
        # Convert to DataFrame and sort by time
        if timeline:
            timeline_df = pd.DataFrame(timeline)
            timeline_df = timeline_df.sort_values('timestamp')
            
            # Log weighting stats
            if 'temporal_weight' in timeline_df.columns:
                logger.info(f"📊 Temporal weighting: avg={timeline_df['temporal_weight'].mean():.3f}, "
                           f"min={timeline_df['temporal_weight'].min():.3f}, "
                           f"max={timeline_df['temporal_weight'].max():.3f}")
            
            return timeline_df
        
        return pd.DataFrame()
    
    def _calculate_traffic_severity(self, duration_minutes):
        """Convert traffic duration to severity (1-5)"""
        if duration_minutes < 20:
            return 1  # Normal
        elif duration_minutes < 30:
            return 2  # Slight delay
        elif duration_minutes < 45:
            return 3  # Moderate delay
        elif duration_minutes < 60:
            return 4  # Heavy delay
        else:
            return 5  # Severe delay
    
    def _calculate_weather_severity(self, weather_row):
        """Calculate weather severity based on conditions"""
        severity = 1
        
        temp = weather_row.get('temperature')
        wind = weather_row.get('wind_speed')
        
        if temp is not None:
            if temp < 0 or temp > 35:
                severity = max(severity, 4)
            elif temp < 5 or temp > 30:
                severity = max(severity, 3)
        
        if wind is not None:
            if wind > 50:  # km/h
                severity = max(severity, 5)
            elif wind > 30:
                severity = max(severity, 3)
        
        return severity
    
    def _calculate_air_quality_severity(self, air_row):
        """Calculate air quality severity"""
        pm25 = air_row.get('pm25')
        aqi = air_row.get('aqi')
        
        if pm25:
            if pm25 > 150:
                return 5  # Hazardous
            elif pm25 > 75:
                return 4  # Unhealthy
            elif pm25 > 35:
                return 3  # Moderate
            elif pm25 > 12:
                return 2  # Fair
            else:
                return 1  # Good
        
        if aqi:
            if aqi > 200:
                return 5
            elif aqi > 150:
                return 4
            elif aqi > 100:
                return 3
            elif aqi > 50:
                return 2
            else:
                return 1
        
        return 2  # Default moderate
    
    def _calculate_health_severity(self, health_row):
        """Calculate health system severity"""
        wait_time = health_row.get('wait_time')
        capacity = health_row.get('ae_capacity')
        
        severity = 1
        
        if wait_time:
            if wait_time > 240:  # 4 hours
                severity = max(severity, 5)
            elif wait_time > 120:
                severity = max(severity, 4)
            elif wait_time > 60:
                severity = max(severity, 3)
        
        if capacity:
            if capacity > 95:
                severity = max(severity, 5)
            elif capacity > 85:
                severity = max(severity, 4)
            elif capacity > 75:
                severity = max(severity, 3)
        
        return severity
    
    def export_timeline(self, output_path="timeline_export.json"):
        """Export complete timeline as JSON"""
        timeline = self.create_unified_timeline()
        
        if not timeline.empty:
            # Convert timestamps to ISO strings
            timeline_dict = timeline.copy()
            timeline_dict['timestamp'] = timeline_dict['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Convert data dict to JSON-serializable format
            timeline_dict['data'] = timeline_dict['data'].apply(lambda x: str(x) if pd.notna(x) else {})
            
            output_file = self.base_path / output_path
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(timeline_dict.to_dict('records'), f, indent=2)
            
            logger.info(f"✅ Timeline exported to {output_file}")
            return str(output_file)
        
        return None
    
    def get_summary_stats(self):
        """Get summary statistics for all data sources"""
        timeline = self.create_unified_timeline()
        
        if timeline.empty:
            return {}
        
        stats = {
            'total_events': len(timeline),
            'date_range': {
                'start': timeline['timestamp'].min().isoformat(),
                'end': timeline['timestamp'].max().isoformat()
            },
            'by_type': timeline['event_type'].value_counts().to_dict(),
            'by_domain': timeline['domain'].value_counts().to_dict(),
            'by_source': timeline['source'].value_counts().to_dict(),
            'avg_severity': timeline['severity'].mean(),
            'high_severity_count': len(timeline[timeline['severity'] >= 4])
        }
        
        return stats


if __name__ == "__main__":
    print("="*70)
    print("OWL ENGINE - DATA MANAGER TEST")
    print("="*70)
    
    dm = DataManager()
    
    # Test each data source
    print("\n🚗 TRAFFIC DATA")
    traffic = dm.load_all_traffic_history()
    print(f"   Records: {len(traffic)}")
    if not traffic.empty:
        print(f"   Date range: {traffic['timestamp'].min()} to {traffic['timestamp'].max()}")
    
    print("\n🌊 FLOOD DATA")
    floods = dm.load_all_flood_warnings()
    print(f"   Records: {len(floods)}")
    if not floods.empty:
        print(f"   Date range: {floods['timestamp'].min()} to {floods['timestamp'].max()}")
    
    print("\n🌤️ WEATHER DATA")
    weather = dm.load_all_weather_data()
    print(f"   Records: {len(weather)}")
    
    print("\n🌫️ AIR QUALITY DATA")
    air = dm.load_all_air_quality_data()
    print(f"   Records: {len(air)}")
    
    print("\n🏥 HEALTH DATA")
    health = dm.load_all_health_data()
    print(f"   Records: {len(health)}")
    
    print("\n📊 UNIFIED TIMELINE")
    timeline = dm.create_unified_timeline()
    print(f"   Total events: {len(timeline)}")
    
    if not timeline.empty:
        print(f"   Event types: {timeline['event_type'].unique()}")
        print(f"   Date range: {timeline['timestamp'].min()} to {timeline['timestamp'].max()}")
        
        print("\n   Sample events:")
        print(timeline[['timestamp', 'event_type', 'severity', 'description']].head(10))
        
        # Export
        export_path = dm.export_timeline()
        print(f"\n✅ Timeline exported to: {export_path}")
        
        # Stats
        stats = dm.get_summary_stats()
        print(f"\n📈 SUMMARY STATISTICS:")
        print(json.dumps(stats, indent=2))
