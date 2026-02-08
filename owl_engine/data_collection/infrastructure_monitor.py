"""
Critical Infrastructure Monitor
Track power grid, water, transport, hospitals, emergency services
"""

import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger('INFRASTRUCTURE_MONITOR')


class InfrastructureMonitor:
    """Monitor critical infrastructure for threat detection"""
    
    def __init__(self, db=None):
        self.db = db
        
        # UK/London infrastructure APIs
        self.sources = {
            'tfl': 'https://api.tfl.gov.uk',
            'national_rail': 'https://lite.realtime.nationalrail.co.uk/OpenLDBWS/',
            'nhs': None,  # NHS API for hospital capacity
            'power': None  # Power grid status
        }
    
    def collect(self):
        """Main collection method"""
        logger.info("🏗️ Collecting infrastructure status...")
        
        try:
            transport_data = self.collect_transport_disruptions()
            
            if self.db and transport_data:
                for record in transport_data:
                    self.db.store(
                        source_type='infrastructure_transport',
                        data=record,
                        initial_confidence=0.9  # Official sources
                    )
            
            logger.info(f"✓ Infrastructure: {len(transport_data)} transport status records")
            
        except Exception as e:
            logger.error(f"Infrastructure collection error: {e}")
    
    def collect_transport_disruptions(self) -> List[Dict]:
        """
        Collect TfL (Transport for London) disruptions
        Tube closures, bus diversions, road works
        """
        records = []
        
        try:
            # TfL Line Status API
            url = f"{self.sources['tfl']}/Line/Mode/tube,dlr,overground,elizabeth-line/Status"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                lines = response.json()
                
                for line in lines:
                    line_name = line.get('name')
                    line_statuses = line.get('lineStatuses', [])
                    
                    for status in line_statuses:
                        severity = status.get('statusSeverity', 10)
                        
                        # Only record disruptions (severity < 10 = good service)
                        if severity < 10:
                            record = {
                                'timestamp': datetime.now().isoformat(),
                                'infrastructure_type': 'transport',
                                'line': line_name,
                                'mode': line.get('modeName', 'tube'),
                                'status': status.get('statusSeverityDescription', 'Unknown'),
                                'severity': severity,
                                'reason': status.get('reason', ''),
                                'disruption_category': status.get('disruption', {}).get('category', 'Unknown'),
                                'threat_level': self._assess_transport_threat(severity),
                                'domain': 'infrastructure',
                                'source': 'TfL',
                                'source_type': 'government'
                            }
                            records.append(record)
            
            # Road disruptions
            road_url = f"{self.sources['tfl']}/Road/all/Disruption"
            road_response = requests.get(road_url, timeout=15)
            
            if road_response.status_code == 200:
                disruptions = road_response.json()
                
                for disruption in disruptions[:50]:  # Limit to 50
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'infrastructure_type': 'road',
                        'location': disruption.get('location', 'Unknown'),
                        'corridor': disruption.get('corridorIds', []),
                        'category': disruption.get('category', 'Unknown'),
                        'sub_category': disruption.get('subCategory', ''),
                        'comments': disruption.get('comments', ''),
                        'severity': disruption.get('severity', 'Unknown'),
                        'domain': 'infrastructure',
                        'source': 'TfL',
                        'source_type': 'government'
                    }
                    records.append(record)
                    
        except Exception as e:
            logger.warning(f"Transport collection failed: {e}")
        
        return records
    
    def _assess_transport_threat(self, severity: int) -> str:
        """
        Assess threat level from TfL severity codes
        
        TfL Severity Scale:
        1-3: Severe disruption
        4-6: Moderate disruption
        7-9: Minor delays
        10: Good service
        """
        if severity <= 3:
            return 'severe'
        elif severity <= 6:
            return 'moderate'
        elif severity <= 9:
            return 'minor'
        else:
            return 'normal'
    
    def collect_power_grid(self) -> List[Dict]:
        """
        Monitor power outages
        In production: Integrate with National Grid API or local provider
        """
        records = []
        
        # Placeholder for power grid monitoring
        # Would integrate with:
        # - National Grid ESO API
        # - Local electricity provider APIs
        # - Citizen reports via social media
        
        return records
    
    def collect_water_supply(self) -> List[Dict]:
        """
        Monitor water main breaks, supply issues
        """
        records = []
        
        # Placeholder for water infrastructure
        # Would integrate with:
        # - Thames Water API
        # - Other water companies
        # - Emergency service reports
        
        return records
    
    def collect_hospital_capacity(self) -> List[Dict]:
        """
        Monitor NHS hospital A&E wait times, bed availability
        Useful for: Mass casualty events, disease outbreaks
        """
        records = []
        
        # Placeholder for NHS data
        # Would integrate with:
        # - NHS Digital API
        # - Hospital trust status feeds
        # - Ambulance service data
        
        return records
    
    def collect_emergency_services(self) -> List[Dict]:
        """
        Monitor fire, police, ambulance deployments
        High activity = potential incident
        """
        records = []
        
        # Placeholder for emergency service monitoring
        # Would use:
        # - Public incident logs
        # - Radio scanner data (where legal)
        # - Social media reports of sirens/activity
        
        return records


if __name__ == '__main__':
    # Test the collector
    monitor = InfrastructureMonitor()
    data = monitor.collect_transport_disruptions()
    
    print(f"Collected {len(data)} infrastructure records")
    
    if data:
        print("\nSample record:")
        import json
        print(json.dumps(data[0], indent=2))
