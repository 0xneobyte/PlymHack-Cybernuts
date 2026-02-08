"""
London Road Disruptions Collector
Collects real-time road closures, incidents, and disruptions affecting London

Data source: Transport for London (TfL) Road Disruptions API
Collects:
- Road closures
- Traffic incidents
- Planned roadworks
- Emergency disruptions
- Location and severity information
"""

import logging
import requests
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger('RoadDisruptions')


class RoadDisruptionsCollector:
    """Collects London road disruption data from TfL API"""
    
    # TfL Unified API endpoint for road disruptions
    TFL_API_BASE = "https://api.tfl.gov.uk"
    
    def __init__(self):
        """Initialize road disruptions collector"""
        self.data_dir = Path(__file__).parent.parent / "owl_data"
        logger.info("✓ Road Disruptions Collector initialized")
    
    def collect(self) -> List[Dict]:
        """
        Main collection method - fetches current road disruptions
        Returns list of disruption events
        """
        try:
            disruptions = []
            
            # Collect from TfL Road Disruptions API
            tfl_disruptions = self._collect_tfl_disruptions()
            if tfl_disruptions:
                disruptions.extend(tfl_disruptions)
            
            # Save to disk
            if disruptions:
                self._save_disruptions(disruptions)
                logger.info(f"✓ Road Disruptions: {len(disruptions)} incidents collected")
            else:
                logger.warning("⚠ No road disruptions found")
            
            return disruptions
            
        except Exception as e:
            logger.error(f"✗ Road disruptions collection failed: {e}")
            return []
    
    def _collect_tfl_disruptions(self) -> List[Dict]:
        """
        Collect road disruptions from TfL Unified API
        Endpoint: /Road/all/Disruption
        """
        try:
            # TfL Road Disruptions endpoint
            url = f"{self.TFL_API_BASE}/Road/all/Disruption"
            
            headers = {
                'User-Agent': 'OWL-Intelligence-Platform/1.0',
                'Accept': 'application/json'
            }
            
            logger.debug(f"Fetching disruptions from TfL API...")
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_tfl_disruptions(data)
            else:
                logger.warning(f"TfL API returned status {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch TfL disruptions: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse TfL response: {e}")
            return []
    
    def _parse_tfl_disruptions(self, data: List[Dict]) -> List[Dict]:
        """
        Parse TfL API response into standardized disruption events
        
        TfL returns list of disruption objects directly
        Each object represents one road disruption incident
        """
        disruptions = []
        
        try:
            # Debug: Check structure of first item
            if data and len(data) > 0:
                logger.debug(f"Sample disruption keys: {list(data[0].keys())}")
            
            # Each item in data IS a disruption
            for disruption in data:
                # Extract corridor/road information
                corridor_ids = disruption.get('corridorIds', [])
                road_name = corridor_ids[0] if corridor_ids else 'Unknown Road'
                
                # Build standardized disruption event
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'road_id': road_name,
                    'road_name': road_name,
                    'disruption_id': disruption.get('id', 'unknown'),
                    'category': disruption.get('category', 'Unknown'),
                    'sub_category': disruption.get('subCategory', ''),
                    'severity': self._map_severity(disruption.get('severity', 'Minimal')),
                    'location': disruption.get('location', 'London'),
                    'comments': disruption.get('comments', ''),
                    'current_update': disruption.get('currentUpdate', ''),
                    'start_time': disruption.get('startDateTime', ''),
                    'end_time': disruption.get('endDateTime', ''),
                    'status': disruption.get('status', 'Active'),
                    'has_closures': disruption.get('hasClosures', False),
                    'level_of_interest': disruption.get('levelOfInterest', 'Unknown'),
                    'geography': disruption.get('geography', {}),
                    'impact_areas': disruption.get('roadDisruptionImpactAreas', []),
                    'source': 'TfL Road Disruptions API',
                    'confidence': 0.95  # High confidence from official source
                }
                
                disruptions.append(event)
                    
            logger.debug(f"Parsed {len(disruptions)} disruptions")
            return disruptions
            
        except Exception as e:
            logger.error(f"Error parsing TfL disruptions: {e}", exc_info=True)
            return []
    
    def _map_severity(self, tfl_severity: str) -> int:
        """
        Map TfL severity to OWL severity scale (1-5, where 1=critical)
        
        TfL severities: Minimal, Moderate, Serious, Severe
        """
        severity_map = {
            'Minimal': 4,
            'Moderate': 3,
            'Serious': 2,
            'Severe': 1,
            'Critical': 1
        }
        return severity_map.get(tfl_severity, 3)
    
    def _map_status_to_severity(self, status: str) -> int:
        """
        Map TfL road status to OWL severity scale (1-5, where 1=critical)
        
        TfL statuses: Good, MinorDelays, ModerateDelays, SevereDelays, Blocked, Closed
        """
        status_map = {
            'Good': 5,
            'MinorDelays': 4,
            'ModerateDelays': 3,
            'SevereDelays': 2,
            'Blocked': 1,
            'Closed': 1,
            'PartClosed': 2
        }
        return status_map.get(status, 3)
    
    def _save_disruptions(self, disruptions: List[Dict]):
        """Save disruptions to owl_data directory"""
        try:
            # Create date-based directory structure
            today = datetime.now().strftime('%Y-%m-%d')
            disruption_dir = self.data_dir / today / "road_disruptions"
            disruption_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each disruption as separate JSON file
            for disruption in disruptions:
                timestamp_str = datetime.now().strftime('%H-%M-%S-%f')
                disruption_id = disruption.get('disruption_id', 'unknown')[:8]
                
                filename = f"{timestamp_str}_{disruption_id}.json"
                filepath = disruption_dir / filename
                
                # Wrap in OWL metadata structure
                owl_data = {
                    'metadata': {
                        'timestamp': disruption['timestamp'],
                        'source_type': 'road_disruptions',
                        'initial_confidence': disruption.get('confidence', 0.95),
                        'current_confidence': disruption.get('confidence', 0.95),
                        'decay_applied': False,
                        'ingestion_layer': 'COLLECT',
                        'data_id': f"road_disruption_{disruption_id}",
                        'file_path': str(filepath)
                    },
                    'raw_data': disruption
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(owl_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(disruptions)} disruptions to {disruption_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save disruptions: {e}")


def main():
    """Test the road disruptions collector"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    collector = RoadDisruptionsCollector()
    disruptions = collector.collect()
    
    print(f"\n{'='*70}")
    print(f"ROAD DISRUPTIONS COLLECTION TEST")
    print(f"{'='*70}")
    print(f"Total disruptions collected: {len(disruptions)}")
    
    if disruptions:
        print(f"\nSample disruptions:")
        for i, disruption in enumerate(disruptions[:5], 1):
            print(f"\n{i}. {disruption.get('road_name', 'Unknown')}")
            print(f"   Category: {disruption.get('category', 'Unknown')}")
            print(f"   Severity: {disruption.get('severity', 'N/A')}")
            print(f"   Location: {disruption.get('location', 'N/A')}")
            print(f"   Status: {disruption.get('status', 'N/A')}")
            if disruption.get('comments'):
                print(f"   Details: {disruption.get('comments', '')[:100]}...")


if __name__ == '__main__':
    main()
