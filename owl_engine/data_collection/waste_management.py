"""
Waste Management Data Collector
Source: UK WasteDataFlow / data.gov.uk

Collects:
- London borough waste collection data
- Recycling rates
- Landfill metrics
"""

import requests
import logging
from typing import List, Dict

logger = logging.getLogger('WasteManagement')


class WasteManagementCollector:
    """Collects waste management data for London"""
    
    # Using data.gov.uk as documented source
    BASE_URL = "https://data.london.gov.uk/api/3"
    
    def __init__(self, database):
        """Initialize collector with database reference"""
        self.db = database
        self.session = requests.Session()
        logger.info("✓ Waste Management Collector initialized")
    
    def collect(self):
        """Main collection method"""
        try:
            # Search for waste datasets
            datasets = self._search_waste_datasets()
            
            if datasets:
                self.db.store_batch('waste_datasets', datasets, confidence=0.75)
            
            logger.info(f"✓ Waste Management: {len(datasets)} datasets found")
            
        except Exception as e:
            logger.error(f"✗ Waste management collection failed: {e}")
    
    def _search_waste_datasets(self) -> List[Dict]:
        """Search for waste management datasets"""
        try:
            url = f"{self.BASE_URL}/action/package_search"
            params = {
                'q': 'waste management',
                'fq': 'organization:greater-london-authority',
                'rows': 30
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                logger.warning("  → API returned success=false")
                return []
            
            result = data.get('result', {})
            results = result.get('results', [])
            
            # Extract relevant information
            datasets = []
            for dataset in results:
                # Look for waste-related datasets
                title = dataset.get('title', '').lower()
                if any(keyword in title for keyword in ['waste', 'recycling', 'landfill', 'disposal']):
                    simplified = {
                        'id': dataset.get('id'),
                        'name': dataset.get('name'),
                        'title': dataset.get('title'),
                        'notes': dataset.get('notes', '')[:500],
                        'organization': dataset.get('organization', {}).get('title', 'Unknown'),
                        'resources': [
                            {
                                'name': r.get('name'),
                                'format': r.get('format'),
                                'url': r.get('url'),
                                'created': r.get('created')
                            }
                            for r in dataset.get('resources', [])[:5]  # First 5 resources
                        ],
                        'metadata_modified': dataset.get('metadata_modified')
                    }
                    datasets.append(simplified)
            
            logger.debug(f"  → Found {len(datasets)} waste datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"  ✗ Failed to search waste datasets: {e}")
            return []
