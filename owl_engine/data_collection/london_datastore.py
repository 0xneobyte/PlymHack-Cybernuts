"""
London Datastore Collector
Source: London Datastore (GLA) CKAN API

Collects:
- SDG indicator datasets
- Policy documents
- Urban planning data
"""

import requests
import logging
from typing import List, Dict

logger = logging.getLogger('LondonDatastore')


class LondonDatastoreCollector:
    """Collects open data from London Datastore"""
    
    BASE_URL = "https://data.london.gov.uk/api/3"
    
    # Topics of interest for SDG monitoring
    TOPICS = [
        'climate+adaptation',
        'flood',
        'waste+management',
        'air+quality',
        'sustainable+development'
    ]
    
    def __init__(self, database):
        """Initialize collector with database reference"""
        self.db = database
        self.session = requests.Session()
        logger.info("✓ London Datastore Collector initialized")
    
    def collect(self):
        """Main collection method"""
        try:
            all_datasets = []
            
            # Search for datasets on different topics
            for topic in self.TOPICS:
                datasets = self._search_datasets(topic)
                all_datasets.extend(datasets)
            
            if all_datasets:
                self.db.store_batch('london_datasets', all_datasets, confidence=0.7)
            
            logger.info(f"✓ London Datastore: {len(all_datasets)} datasets found")
            
        except Exception as e:
            logger.error(f"✗ London Datastore collection failed: {e}")
    
    def _search_datasets(self, query: str) -> List[Dict]:
        """Search for datasets on a specific topic"""
        try:
            url = f"{self.BASE_URL}/action/package_search"
            params = {
                'q': query,
                'rows': 20  # Limit results per topic
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                logger.warning(f"  → API returned success=false for query: {query}")
                return []
            
            result = data.get('result', {})
            results = result.get('results', [])
            
            # Simplify dataset info
            datasets = []
            for dataset in results:
                simplified = {
                    'id': dataset.get('id'),
                    'name': dataset.get('name'),
                    'title': dataset.get('title'),
                    'notes': dataset.get('notes', '')[:500],  # Truncate description
                    'organization': dataset.get('organization', {}).get('title', 'Unknown'),
                    'resources_count': len(dataset.get('resources', [])),
                    'metadata_modified': dataset.get('metadata_modified'),
                    'topic_query': query
                }
                datasets.append(simplified)
            
            logger.debug(f"  → Found {len(datasets)} datasets for '{query}'")
            return datasets
            
        except Exception as e:
            logger.error(f"  ✗ Failed to search for '{query}': {e}")
            return []
