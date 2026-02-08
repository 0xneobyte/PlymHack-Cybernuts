"""
OWL ENGINE - Layer 2: EXTRACT (Extraction / Parsing Layer)

Purpose: Clean and structure raw data without altering or discarding content.
Extract meaningful elements to prepare for entity linking.

Philosophy: Only reclassify (add uncertainty); no deletion. Store relationships early.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re
from concurrent.futures import ThreadPoolExecutor

# Add UTF-8 support for Windows
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('owl_extract.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('EXTRACT_LAYER')


class ExtractedData:
    """Container for extracted/structured data"""
    
    def __init__(self, original_data_id: str, source_type: str):
        self.original_data_id = original_data_id
        self.source_type = source_type
        self.extracted_timestamp = datetime.now().isoformat()
        self.entities_mentioned = []
        self.structured_fields = {}
        self.confidence_adjustment = 0.0
        self.uncertainty = 0.0
        self.normalized_timestamp = None
        self.language = 'en'
        self.extraction_quality = 1.0
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'original_data_id': self.original_data_id,
            'source_type': self.source_type,
            'extracted_timestamp': self.extracted_timestamp,
            'entities_mentioned': self.entities_mentioned,
            'structured_fields': self.structured_fields,
            'confidence_adjustment': self.confidence_adjustment,
            'uncertainty': self.uncertainty,
            'normalized_timestamp': self.normalized_timestamp,
            'language': self.language,
            'extraction_quality': self.extraction_quality
        }


class BaseExtractor:
    """Base class for all extractors"""
    
    def extract(self, raw_data: Dict, metadata: Dict) -> ExtractedData:
        """Extract and structure raw data"""
        raise NotImplementedError


class FloodWarningExtractor(BaseExtractor):
    """Extract structured data from flood warnings"""
    
    def extract(self, raw_data: Dict, metadata: Dict) -> ExtractedData:
        """Extract flood warning information"""
        extracted = ExtractedData(
            original_data_id=metadata.get('data_id'),
            source_type='flood_warnings'
        )
        
        # Extract entities
        entities = []
        
        # Location entities
        description = raw_data.get('description', '')
        if description:
            entities.append({
                'type': 'location',
                'name': description,
                'context': 'flood_area'
            })
        
        # County/region entities
        flood_area = raw_data.get('floodArea', {})
        county = flood_area.get('county', '')
        if county:
            # Split multiple counties
            for c in county.split(','):
                entities.append({
                    'type': 'location',
                    'name': c.strip(),
                    'context': 'administrative_area'
                })
        
        river = flood_area.get('riverOrSea', '')
        if river:
            entities.append({
                'type': 'infrastructure',
                'name': river,
                'context': 'water_body'
            })
        
        extracted.entities_mentioned = entities
        
        # Extract structured fields
        extracted.structured_fields = {
            'severity': raw_data.get('severity'),
            'severity_level': raw_data.get('severityLevel'),
            'is_tidal': raw_data.get('isTidal', False),
            'area_code': raw_data.get('floodAreaID'),
            'message': raw_data.get('message', ''),
            'ea_area': raw_data.get('eaAreaName'),
            'event_type': 'flood_alert'
        }
        
        # Normalize timestamp
        time_raised = raw_data.get('timeRaised')
        if time_raised:
            try:
                # Already in ISO format, just normalize
                extracted.normalized_timestamp = time_raised
            except:
                extracted.normalized_timestamp = datetime.now().isoformat()
        
        # Adjust confidence based on severity
        severity_level = raw_data.get('severityLevel', 3)
        if severity_level == 1:  # Severe warning
            extracted.confidence_adjustment = 0.05  # Increase confidence
        elif severity_level == 4:  # Alert removed
            extracted.confidence_adjustment = -0.1  # Decrease confidence
        
        # Calculate uncertainty (1 - confidence)
        original_confidence = metadata.get('current_confidence', 0.9)
        new_confidence = original_confidence + extracted.confidence_adjustment
        extracted.uncertainty = 1.0 - new_confidence
        
        # Extraction quality (high for structured government data)
        extracted.extraction_quality = 0.95
        
        return extracted


class RiverLevelExtractor(BaseExtractor):
    """Extract structured data from river level readings"""
    
    def extract(self, raw_data: Dict, metadata: Dict) -> ExtractedData:
        """Extract river level information"""
        extracted = ExtractedData(
            original_data_id=metadata.get('data_id'),
            source_type='river_levels'
        )
        
        # Extract entities
        entities = []
        
        station_name = raw_data.get('station', '')
        if station_name:
            entities.append({
                'type': 'infrastructure',
                'name': station_name,
                'context': 'monitoring_station'
            })
        
        station_ref = raw_data.get('stationReference', '')
        if station_ref:
            entities.append({
                'type': 'asset',
                'name': station_ref,
                'context': 'station_id'
            })
        
        extracted.entities_mentioned = entities
        
        # Extract structured fields
        value = raw_data.get('value')
        unit = raw_data.get('unitName', '')
        
        extracted.structured_fields = {
            'measurement_value': value,
            'unit': unit,
            'station': station_name,
            'station_reference': station_ref,
            'event_type': 'river_measurement'
        }
        
        # Normalize timestamp
        date_time = raw_data.get('dateTime')
        if date_time:
            extracted.normalized_timestamp = date_time
        
        # Confidence based on measurement availability
        if value is not None:
            extracted.confidence_adjustment = 0.0
            extracted.extraction_quality = 0.9
        else:
            extracted.confidence_adjustment = -0.05
            extracted.extraction_quality = 0.7
        
        original_confidence = metadata.get('current_confidence', 0.85)
        new_confidence = original_confidence + extracted.confidence_adjustment
        extracted.uncertainty = 1.0 - new_confidence
        
        return extracted


class TrafficConditionExtractor(BaseExtractor):
    """Extract structured data from traffic conditions"""
    
    def extract(self, raw_data: Dict, metadata: Dict) -> ExtractedData:
        """Extract traffic information"""
        extracted = ExtractedData(
            original_data_id=metadata.get('data_id'),
            source_type='traffic_conditions'
        )
        
        # Extract entities
        entities = []
        
        route_name = raw_data.get('route_name', '')
        origin = raw_data.get('origin', '')
        destination = raw_data.get('destination', '')
        
        if route_name:
            entities.append({
                'type': 'infrastructure',
                'name': route_name,
                'context': 'traffic_route'
            })
        
        if origin:
            entities.append({
                'type': 'location',
                'name': origin,
                'context': 'origin_point'
            })
        
        if destination:
            entities.append({
                'type': 'location',
                'name': destination,
                'context': 'destination_point'
            })
        
        extracted.entities_mentioned = entities
        
        # Extract structured fields
        extracted.structured_fields = {
            'route_name': route_name,
            'origin': origin,
            'destination': destination,
            'status': raw_data.get('status', 'pending'),
            'duration': raw_data.get('duration'),
            'distance': raw_data.get('distance'),
            'event_type': 'traffic_observation'
        }
        
        # Normalize timestamp
        timestamp = raw_data.get('collection_timestamp') or raw_data.get('timestamp')
        if timestamp:
            extracted.normalized_timestamp = timestamp
        
        # Adjust confidence based on data completeness
        status = raw_data.get('status', '')
        if status == 'pending_browser_automation':
            extracted.confidence_adjustment = -0.2  # Lower for placeholder
            extracted.extraction_quality = 0.5
        elif status == 'collected' and raw_data.get('duration'):
            extracted.confidence_adjustment = 0.0
            extracted.extraction_quality = 0.9
        else:
            extracted.confidence_adjustment = -0.1
            extracted.extraction_quality = 0.7
        
        original_confidence = metadata.get('current_confidence', 0.8)
        new_confidence = original_confidence + extracted.confidence_adjustment
        extracted.uncertainty = 1.0 - new_confidence
        
        return extracted


class LondonDatasetExtractor(BaseExtractor):
    """Extract structured data from London datasets"""
    
    def extract(self, raw_data: Dict, metadata: Dict) -> ExtractedData:
        """Extract London dataset information"""
        extracted = ExtractedData(
            original_data_id=metadata.get('data_id'),
            source_type='london_datasets'
        )
        
        # Extract entities
        entities = []
        
        title = raw_data.get('title', '')
        organization = raw_data.get('organization', '')
        topic = raw_data.get('topic_query', '')
        
        if organization:
            entities.append({
                'type': 'organization',
                'name': organization,
                'context': 'data_provider'
            })
        
        # Extract location mention (London)
        entities.append({
            'type': 'location',
            'name': 'London',
            'context': 'geographic_scope'
        })
        
        extracted.entities_mentioned = entities
        
        # Extract structured fields
        extracted.structured_fields = {
            'title': title,
            'dataset_id': raw_data.get('id'),
            'organization': organization,
            'topic': topic,
            'resources_count': raw_data.get('resources_count', 0),
            'event_type': 'dataset_availability'
        }
        
        # Normalize timestamp
        modified = raw_data.get('metadata_modified')
        if modified:
            extracted.normalized_timestamp = modified
        
        # Confidence based on organization and freshness
        extracted.confidence_adjustment = 0.0
        extracted.extraction_quality = 0.85
        
        original_confidence = metadata.get('current_confidence', 0.7)
        new_confidence = original_confidence + extracted.confidence_adjustment
        extracted.uncertainty = 1.0 - new_confidence
        
        return extracted


class ExtractionEngine:
    """Main extraction engine for Layer 2"""
    
    def __init__(self, raw_data_path: str = "owl_data", extracted_data_path: str = "owl_extracted"):
        self.raw_data_path = Path(raw_data_path)
        self.extracted_data_path = Path(extracted_data_path)
        self.extracted_data_path.mkdir(parents=True, exist_ok=True)
        
        # Register extractors for each source type
        self.extractors = {
            'flood_warnings': FloodWarningExtractor(),
            'river_levels': RiverLevelExtractor(),
            'traffic_conditions': TrafficConditionExtractor(),
            'london_datasets': LondonDatasetExtractor(),
            'waste_datasets': LondonDatasetExtractor(),  # Same structure
        }
        
        logger.info(f"Extraction Engine initialized with {len(self.extractors)} extractors")
    
    def extract_file(self, filepath: Path) -> Dict:
        """Extract data from a single raw JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            raw_data = data.get('raw_data', {})
            source_type = metadata.get('source_type')
            
            # Get appropriate extractor
            extractor = self.extractors.get(source_type)
            if not extractor:
                logger.warning(f"No extractor for source type: {source_type}")
                return None
            
            # Extract structured data
            extracted = extractor.extract(raw_data, metadata)
            
            # Create output structure
            output = {
                'extraction_metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'extraction_layer': 'EXTRACT',
                    'original_file': str(filepath),
                    'original_data_id': metadata.get('data_id'),
                    'extraction_quality': extracted.extraction_quality
                },
                'original_metadata': metadata,  # Preserve original
                'extracted_data': extracted.to_dict()
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error extracting {filepath}: {e}")
            return None
    
    def save_extracted(self, extracted_data: Dict, source_type: str, date_str: str):
        """Save extracted data"""
        # Create directory structure
        date_dir = self.extracted_data_path / date_str
        source_dir = date_dir / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%H-%M-%S-%f")
        data_id = extracted_data.get('extraction_metadata', {}).get('original_data_id', 'unknown')
        filename = f"{timestamp}_{data_id}.json"
        filepath = source_dir / filename
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def extract_all(self, date_str: str = None):
        """Extract all raw data for a specific date"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        date_dir = self.raw_data_path / date_str
        
        if not date_dir.exists():
            logger.error(f"No data directory for {date_str}")
            return
        
        logger.info(f"Starting extraction for {date_str}")
        
        # Find all JSON files
        json_files = list(date_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} files to extract")
        
        extracted_count = 0
        
        # Process with thread pool for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.extract_file, filepath) for filepath in json_files]
            
            for future in futures:
                result = future.result()
                if result:
                    source_type = result['original_metadata'].get('source_type')
                    saved_path = self.save_extracted(result, source_type, date_str)
                    extracted_count += 1
                    logger.debug(f"Extracted to {saved_path}")
        
        logger.info(f"✓ Extraction complete: {extracted_count}/{len(json_files)} files processed")
        return extracted_count
    
    def get_statistics(self):
        """Get extraction statistics"""
        stats = {}
        
        for date_dir in self.extracted_data_path.glob("*"):
            if not date_dir.is_dir():
                continue
            
            for source_dir in date_dir.glob("*"):
                if not source_dir.is_dir():
                    continue
                
                source_type = source_dir.name
                file_count = len(list(source_dir.glob("*.json")))
                
                if source_type in stats:
                    stats[source_type] += file_count
                else:
                    stats[source_type] = file_count
        
        return stats


def main():
    """Main entry point for extraction"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              🦉 OWL ENGINE - Layer 2: EXTRACT            ║
    ║                 Extraction & Parsing Layer               ║
    ║                                                          ║
    ║         Philosophy: Structure Without Deletion          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    engine = ExtractionEngine()
    
    import sys
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nExtracting data for: {date_str}")
    print(f"Raw data source: owl_data/{date_str}/")
    print(f"Extracted output: owl_extracted/{date_str}/\n")
    
    count = engine.extract_all(date_str)
    
    print(f"\n✓ Extraction complete: {count} files processed")
    
    stats = engine.get_statistics()
    print("\n📊 Extracted Data Statistics:")
    for source, count in sorted(stats.items()):
        print(f"  {source}: {count} records")


if __name__ == "__main__":
    main()
