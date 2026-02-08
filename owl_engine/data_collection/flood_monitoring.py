"""
Flood Monitoring Data Collector
Source: UK Environment Agency Real-Time Flood-Monitoring API

Collects:
- Flood warnings/alerts for London
- River levels (Thames)
- Rainfall data
"""

import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger('FloodMonitoring')


class FloodMonitoringCollector:
    """Collects flood monitoring data from UK Environment Agency"""
    
    BASE_URL = "https://environment.data.gov.uk/flood-monitoring"
    
    def __init__(self, database):
        """Initialize collector with database reference"""
        self.db = database
        self.session = requests.Session()
        logger.info("✓ Flood Monitoring Collector initialized")
    
    def collect(self):
        """Main collection method"""
        try:
            # Collect flood warnings
            warnings = self._get_flood_warnings()
            if warnings:
                self.db.store_batch('flood_warnings', warnings, confidence=0.9)
            
            # Collect river levels for London stations
            river_levels = self._get_river_levels()
            if river_levels:
                self.db.store_batch('river_levels', river_levels, confidence=0.85)
            
            logger.info(f"✓ Flood monitoring: {len(warnings)} warnings, {len(river_levels)} river readings")
            
        except Exception as e:
            logger.error(f"✗ Flood monitoring collection failed: {e}")
    
    def _get_flood_warnings(self) -> List[Dict]:
        """Get current flood warnings for Greater London"""
        try:
            # Filter for Greater London county
            url = f"{self.BASE_URL}/id/floods"
            params = {
                'county': 'Greater London',
                '_limit': 100
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            logger.debug(f"  → Retrieved {len(items)} flood warnings")
            return items
            
        except Exception as e:
            logger.error(f"  ✗ Failed to get flood warnings: {e}")
            return []
    
    def _get_river_levels(self) -> List[Dict]:
        """Get river level data for Thames and London stations"""
        try:
            # Get stations in London area
            stations_url = f"{self.BASE_URL}/id/stations"
            params = {
                'search': 'Thames',
                '_limit': 50
            }
            
            response = self.session.get(stations_url, params=params, timeout=30)
            response.raise_for_status()
            
            stations_data = response.json()
            stations = stations_data.get('items', [])
            
            # Get latest readings for each station
            readings = []
            for station in stations[:10]:  # Limit to 10 stations for efficiency
                station_id = station.get('stationReference')
                measures = station.get('measures', [])
                
                for measure in measures[:2]:  # Get first 2 measures per station
                    measure_id = measure.get('@id', '').split('/')[-1]
                    
                    # Get latest reading
                    reading_url = f"{self.BASE_URL}/id/measures/{measure_id}/readings"
                    reading_params = {'_limit': 1, '_sorted': True}
                    
                    try:
                        reading_response = self.session.get(reading_url, params=reading_params, timeout=15)
                        reading_response.raise_for_status()
                        reading_data = reading_response.json()
                        
                        reading_items = reading_data.get('items', [])
                        if reading_items:
                            reading_items[0]['station'] = station.get('label', 'Unknown')
                            reading_items[0]['stationReference'] = station_id
                            readings.append(reading_items[0])
                    except Exception as e:
                        logger.debug(f"  → Failed to get reading for {measure_id}: {e}")
                        continue
            
            logger.debug(f"  → Retrieved {len(readings)} river level readings")
            return readings
            
        except Exception as e:
            logger.error(f"  ✗ Failed to get river levels: {e}")
            return []
