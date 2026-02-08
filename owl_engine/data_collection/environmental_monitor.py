"""
Environmental Threat Monitor
Collects air quality, weather extremes, radiation, noise pollution
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List
import json

logger = logging.getLogger('ENVIRONMENTAL_MONITOR')


class EnvironmentalMonitor:
    """Monitor environmental threats for urban intelligence"""
    
    def __init__(self, db=None):
        self.db = db
        self.sources = {
            'openaq': 'https://api.openaq.org/v2/latest',
            'openweather': 'https://api.openweathermap.org/data/2.5/weather',
            'uk_air': 'https://api.erg.ic.ac.uk/AirQuality/Daily/MonitoringIndex/Latest/GroupName=London/Json'
        }
    
    def collect(self):
        """Main collection method"""
        logger.info("📡 Collecting environmental data...")
        
        try:
            air_quality_data = self.collect_air_quality()
            weather_data = self.collect_weather_threats()
            
            # Store with high confidence (sensor data)
            if self.db and air_quality_data:
                for record in air_quality_data:
                    self.db.store(
                        source_type='environmental_air_quality',
                        data=record,
                        initial_confidence=0.85
                    )
            
            if self.db and weather_data:
                for record in weather_data:
                    self.db.store(
                        source_type='environmental_weather',
                        data=record,
                        initial_confidence=0.9
                    )
            
            logger.info(f"✓ Environmental: {len(air_quality_data)} air quality + {len(weather_data)} weather records")
            
        except Exception as e:
            logger.error(f"Environmental collection error: {e}")
    
    def collect_air_quality(self) -> List[Dict]:
        """
        Collect air quality data for threat detection
        High PM2.5/PM10/NO2 = health threat + traffic correlation
        """
        records = []
        
        try:
            # OpenAQ API - Global air quality
            params = {
                'city': 'London',
                'limit': 100
            }
            
            response = requests.get(self.sources['openaq'], params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for result in data.get('results', []):
                    for measurement in result.get('measurements', []):
                        # Calculate threat level
                        parameter = measurement['parameter']
                        value = measurement['value']
                        threat_level = self._assess_air_quality_threat(parameter, value)
                        
                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'location': result['location'],
                            'city': result.get('city', 'London'),
                            'country': result.get('country', 'GB'),
                            'coordinates': result.get('coordinates', {}),
                            'parameter': parameter,
                            'value': value,
                            'unit': measurement['unit'],
                            'threat_level': threat_level,
                            'threat_type': 'environmental',
                            'domain': 'environmental',
                            'source': 'OpenAQ',
                            'source_type': 'sensor'
                        }
                        records.append(record)
            
            # Try UK Air Quality API as backup
            try:
                uk_response = requests.get(self.sources['uk_air'], timeout=15)
                if uk_response.status_code == 200:
                    uk_data = uk_response.json()
                    # Parse UK air quality format
                    for site in uk_data.get('DailyAirQualityIndex', {}).get('LocalAuthority', []):
                        if isinstance(site, dict):
                            records.append({
                                'timestamp': datetime.now().isoformat(),
                                'location': site.get('LocalAuthorityName', 'London'),
                                'air_quality_band': site.get('@AirQualityBand', 'Unknown'),
                                'air_quality_index': site.get('@AirQualityIndex', 0),
                                'threat_type': 'environmental',
                                'domain': 'environmental',
                                'source': 'UK_Air',
                                'source_type': 'government'
                            })
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Air quality collection failed: {e}")
        
        return records
    
    def _assess_air_quality_threat(self, parameter: str, value: float) -> str:
        """
        Assess threat level based on WHO/UK standards
        """
        # WHO Air Quality Guidelines
        thresholds = {
            'pm25': {'good': 15, 'moderate': 35, 'unhealthy': 55, 'hazardous': 150},
            'pm10': {'good': 45, 'moderate': 100, 'unhealthy': 150, 'hazardous': 250},
            'no2': {'good': 40, 'moderate': 100, 'unhealthy': 200, 'hazardous': 400},
            'o3': {'good': 60, 'moderate': 120, 'unhealthy': 180, 'hazardous': 240},
            'so2': {'good': 40, 'moderate': 80, 'unhealthy': 120, 'hazardous': 200}
        }
        
        param = parameter.lower()
        if param not in thresholds:
            return 'unknown'
        
        t = thresholds[param]
        
        if value <= t['good']:
            return 'good'
        elif value <= t['moderate']:
            return 'moderate'
        elif value <= t['unhealthy']:
            return 'unhealthy'
        else:
            return 'hazardous'
    
    def collect_weather_threats(self) -> List[Dict]:
        """
        Collect extreme weather data
        Storms, heavy rain, heat waves, cold snaps
        """
        records = []
        
        try:
            # London coordinates
            params = {
                'lat': 51.5074,
                'lon': -0.1278,
                'appid': 'demo'  # Replace with actual API key
            }
            
            # Note: This will fail without real API key - placeholder for structure
            # response = requests.get(self.sources['openweather'], params=params, timeout=15)
            
            # Simulated weather threat detection
            # In production, parse actual weather API for:
            # - Wind speed > 50mph (storm)
            # - Temperature > 30°C or < 0°C (extreme)
            # - Heavy rainfall > 25mm/hour (flood risk)
            # - Severe weather warnings
            
            record = {
                'timestamp': datetime.now().isoformat(),
                'location': 'London',
                'coordinates': {'lat': 51.5074, 'lon': -0.1278},
                'weather_status': 'monitoring',
                'threat_type': 'environmental',
                'domain': 'environmental',
                'source': 'OpenWeather',
                'source_type': 'sensor',
                'note': 'Add OpenWeather API key for live data'
            }
            records.append(record)
            
        except Exception as e:
            logger.warning(f"Weather collection failed: {e}")
        
        return records


if __name__ == '__main__':
    # Test the collector
    monitor = EnvironmentalMonitor()
    data = monitor.collect_air_quality()
    print(f"Collected {len(data)} air quality records")
    
    if data:
        print("\nSample record:")
        print(json.dumps(data[0], indent=2))
