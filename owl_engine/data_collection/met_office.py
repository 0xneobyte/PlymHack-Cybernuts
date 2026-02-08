"""
Met Office Weather Data Collector
Source: UK Met Office DataPoint API

Collects:
- Weather observations for London
- Severe weather warnings
- Forecast data
"""

import requests
import logging
from typing import List, Dict
import os

logger = logging.getLogger('MetOffice')


class MetOfficeCollector:
    """Collects weather data from UK Met Office"""
    
    BASE_URL = "http://datapoint.metoffice.gov.uk/public/data"
    
    # London/Heathrow site ID
    LONDON_SITE_ID = "3772"  # Heathrow
    
    def __init__(self, database):
        """Initialize collector with database reference"""
        self.db = database
        self.api_key = os.getenv('MET_OFFICE_API_KEY', '')
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("⚠ Met Office API key not set. Set MET_OFFICE_API_KEY environment variable.")
        
        logger.info("✓ Met Office Collector initialized")
    
    def collect(self):
        """Main collection method"""
        if not self.api_key:
            logger.warning("⚠ Skipping Met Office collection (no API key)")
            return
        
        try:
            # Collect weather observations
            observations = self._get_observations()
            if observations:
                self.db.store_batch('weather_observations', observations, confidence=0.9)
            
            # Collect severe warnings
            warnings = self._get_warnings()
            if warnings:
                self.db.store_batch('weather_warnings', warnings, confidence=0.95)
            
            logger.info(f"✓ Met Office: {len(observations)} observations, {len(warnings)} warnings")
            
        except Exception as e:
            logger.error(f"✗ Met Office collection failed: {e}")
    
    def _get_observations(self) -> List[Dict]:
        """Get weather observations for London"""
        try:
            url = f"{self.BASE_URL}/val/wxobs/all/json/{self.LONDON_SITE_ID}"
            params = {
                'res': 'hourly',
                'key': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract observation periods
            observations = []
            siteRep = data.get('SiteRep', {})
            dv = siteRep.get('DV', {})
            location = dv.get('Location', {})
            
            if isinstance(location, dict):
                periods = location.get('Period', [])
                
                for period in periods:
                    date = period.get('value', '')
                    reps = period.get('Rep', [])
                    
                    for rep in reps:
                        obs = {
                            'date': date,
                            'location': location.get('name', 'London'),
                            **rep
                        }
                        observations.append(obs)
            
            logger.debug(f"  → Retrieved {len(observations)} weather observations")
            return observations
            
        except Exception as e:
            logger.error(f"  ✗ Failed to get observations: {e}")
            return []
    
    def _get_warnings(self) -> List[Dict]:
        """Get severe weather warnings for UK"""
        try:
            url = f"{self.BASE_URL}/txt/wxwarning/uk/json"
            params = {'key': self.api_key}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract warnings
            warnings = []
            warnings_data = data.get('Warnings', {})
            
            if warnings_data:
                # Check different warning categories
                for category in ['Forecast', 'Warning']:
                    if category in warnings_data:
                        items = warnings_data[category]
                        if isinstance(items, list):
                            warnings.extend(items)
                        elif isinstance(items, dict):
                            warnings.append(items)
            
            logger.debug(f"  → Retrieved {len(warnings)} weather warnings")
            return warnings
            
        except Exception as e:
            logger.error(f"  ✗ Failed to get warnings: {e}")
            return []
