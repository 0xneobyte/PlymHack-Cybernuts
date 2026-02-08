"""
OWL ENGINE - Layer 3: LINK (Entity Linking & Graph Layer)

Purpose: Build a knowledge graph connecting entities, locations, and events.
Create relationships between floods, traffic routes, and geographic areas.

Philosophy: Everything is connected. Store all relationships, even weak ones.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LINK_LAYER')


class EntityGraph:
    """Knowledge graph for OWL Engine entities"""
    
    def __init__(self):
        self.entities = {}  # entity_id -> entity data
        self.relationships = defaultdict(list)  # entity_id -> [(related_id, relationship_type, strength)]
        self.location_index = defaultdict(set)  # location_name -> set of entity_ids
        self.temporal_index = defaultdict(list)  # date -> list of entity_ids
        
    def add_entity(self, entity_id: str, entity_type: str, data: Dict, locations: List[str] = None, timestamp: str = None):
        """Add entity to graph"""
        self.entities[entity_id] = {
            'type': entity_type,
            'data': data,
            'locations': locations or [],
            'timestamp': timestamp,
            'created': datetime.now().isoformat()
        }
        
        # Index by location
        if locations:
            for loc in locations:
                normalized_loc = self._normalize_location(loc)
                self.location_index[normalized_loc].add(entity_id)
        
        # Index by time
        if timestamp:
            date = timestamp.split('T')[0]
            self.temporal_index[date].append(entity_id)
    
    def add_relationship(self, entity1_id: str, entity2_id: str, relationship_type: str, strength: float = 1.0):
        """Add bidirectional relationship between entities"""
        self.relationships[entity1_id].append((entity2_id, relationship_type, strength))
        self.relationships[entity2_id].append((entity1_id, relationship_type, strength))
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location names for matching"""
        # Remove common suffixes and normalize
        location = location.lower().strip()
        location = re.sub(r',.*$', '', location)  # Remove everything after comma
        location = re.sub(r'\s+(london|uk|united kingdom)$', '', location)
        return location.strip()
    
    def find_spatial_overlap(self, entity_id: str) -> List[Tuple[str, float]]:
        """Find entities that share geographic locations"""
        if entity_id not in self.entities:
            return []
        
        entity_locations = self.entities[entity_id].get('locations', [])
        overlapping = defaultdict(int)
        
        for loc in entity_locations:
            normalized = self._normalize_location(loc)
            # Find all entities at this location
            for other_id in self.location_index[normalized]:
                if other_id != entity_id:
                    overlapping[other_id] += 1
        
        # Convert to list with overlap strength
        results = []
        for other_id, count in overlapping.items():
            strength = count / len(entity_locations) if entity_locations else 0
            results.append((other_id, strength))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def find_temporal_neighbors(self, entity_id: str, time_window_hours: int = 24) -> List[str]:
        """Find entities within time window"""
        if entity_id not in self.entities:
            return []
        
        entity_ts = self.entities[entity_id].get('timestamp')
        if not entity_ts:
            return []
        
        # Get entities from same date (can be enhanced for time window)
        date = entity_ts.split('T')[0]
        neighbors = [eid for eid in self.temporal_index[date] if eid != entity_id]
        return neighbors
    
    def get_entity(self, entity_id: str) -> Dict:
        """Retrieve entity data"""
        return self.entities.get(entity_id)
    
    def get_relationships(self, entity_id: str) -> List[Tuple]:
        """Get all relationships for an entity"""
        return self.relationships.get(entity_id, [])
    
    def export_graph(self) -> Dict:
        """Export graph for storage"""
        return {
            'entities': self.entities,
            'relationships': dict(self.relationships),
            'location_index': {k: list(v) for k, v in self.location_index.items()},
            'temporal_index': dict(self.temporal_index)
        }
    
    def import_graph(self, data: Dict):
        """Import previously saved graph"""
        self.entities = data.get('entities', {})
        self.relationships = defaultdict(list, data.get('relationships', {}))
        self.location_index = defaultdict(set, {
            k: set(v) for k, v in data.get('location_index', {}).items()
        })
        self.temporal_index = defaultdict(list, data.get('temporal_index', {}))


class EntityLinker:
    """Links raw data to entities in the knowledge graph"""
    
    def __init__(self, graph: EntityGraph):
        self.graph = graph
    
    def link_flood_warning(self, flood_data: Dict) -> str:
        """Create entity from flood warning and link to graph"""
        raw = flood_data.get('raw_data', {})
        metadata = flood_data.get('metadata', {})
        
        # Extract locations from message
        message = raw.get('message', '')
        locations = self._extract_flood_locations(message, raw)
        
        entity_id = f"flood_{metadata.get('data_id', 'unknown')}"
        
        # Add to graph
        self.graph.add_entity(
            entity_id=entity_id,
            entity_type='flood_warning',
            data={
                'severity': raw.get('severity'),
                'severity_level': raw.get('severityLevel'),
                'description': raw.get('description'),
                'message': message,
                'area': raw.get('eaAreaName'),
                'counties': raw.get('floodArea', {}).get('county', ''),
                'river': raw.get('floodArea', {}).get('riverOrSea')
            },
            locations=locations,
            timestamp=metadata.get('timestamp')
        )
        
        logger.info(f"✓ Linked flood entity: {entity_id} ({len(locations)} locations)")
        return entity_id
    
    def link_traffic_data(self, traffic_data: Dict) -> str:
        """Create entity from traffic data and link to graph"""
        route_name = traffic_data.get('route_name', 'unknown')
        entity_id = f"traffic_{route_name.replace(' ', '_')}_{traffic_data.get('timestamp', '')[:19]}"
        
        # Extract locations
        locations = [
            traffic_data.get('origin', ''),
            traffic_data.get('destination', '')
        ]
        
        # Parse duration to minutes
        duration_str = traffic_data.get('duration', '0 min')
        duration_min = self._parse_duration(duration_str)
        
        # Add to graph
        self.graph.add_entity(
            entity_id=entity_id,
            entity_type='traffic_route',
            data={
                'route_name': route_name,
                'origin': traffic_data.get('origin'),
                'destination': traffic_data.get('destination'),
                'duration_minutes': duration_min,
                'duration_text': duration_str,
                'url': traffic_data.get('url')
            },
            locations=locations,
            timestamp=traffic_data.get('timestamp')
        )
        
        logger.info(f"✓ Linked traffic entity: {route_name} ({duration_min} min)")
        return entity_id
    
    def _extract_flood_locations(self, message: str, raw_data: Dict) -> List[str]:
        """Extract location names from flood warning"""
        locations = []
        
        # Get main area
        if 'description' in raw_data:
            locations.append(raw_data['description'])
        
        # Extract from counties
        if 'floodArea' in raw_data and 'county' in raw_data['floodArea']:
            counties = raw_data['floodArea']['county'].split(', ')
            locations.extend(counties)
        
        # Parse locations from message (look for capitalized place names)
        # Common pattern: "around Location1, Location2, Location3"
        import re
        around_match = re.search(r'around ([A-Z][^.]+?)\.', message)
        if around_match:
            location_text = around_match.group(1)
            place_names = [p.strip() for p in location_text.split(',')]
            locations.extend(place_names)
        
        return list(set(locations))  # Remove duplicates
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to minutes"""
        import re
        match = re.search(r'(\d+)\s*min', duration_str)
        if match:
            return int(match.group(1))
        return 0


def build_entity_graph(owl_data_path: str = 'owl_data', traffic_data_path: str = 'traffic_data_collected') -> EntityGraph:
    """Build complete entity graph from collected data"""
    logger.info("🔗 Building Entity Graph (Layer 3: LINK)")
    
    graph = EntityGraph()
    linker = EntityLinker(graph)
    
    # Load flood warnings
    flood_count = 0
    owl_path = Path(owl_data_path)
    for flood_file in owl_path.glob('**/flood_warnings/*.json'):
        with open(flood_file, 'r', encoding='utf-8') as f:
            flood_data = json.load(f)
            linker.link_flood_warning(flood_data)
            flood_count += 1
    
    # Load traffic data
    traffic_count = 0
    traffic_path = Path(traffic_data_path)
    for traffic_file in traffic_path.glob('*.json'):
        with open(traffic_file, 'r', encoding='utf-8') as f:
            traffic_list = json.load(f)
            for traffic_data in traffic_list:
                linker.link_traffic_data(traffic_data)
                traffic_count += 1
    
    logger.info(f"✓ Graph built: {flood_count} floods, {traffic_count} traffic routes")
    logger.info(f"✓ Total entities: {len(graph.entities)}")
    logger.info(f"✓ Location index: {len(graph.location_index)} unique locations")
    
    return graph


if __name__ == '__main__':
    # Test the entity linker
    graph = build_entity_graph()
    
    # Save graph
    output_path = Path('owl_engine_graph.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph.export_graph(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Graph saved to {output_path}")
