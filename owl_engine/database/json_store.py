"""
JSON-based Time-Series Database for Owl Engine
Stores all collected data in timestamped JSON files

Philosophy: Never delete. Only add and evolve confidence scores.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import hashlib

logger = logging.getLogger('JSONDatabase')


class JSONDatabase:
    """
    Time-based JSON storage for the COLLECT layer
    
    Structure:
    owl_data/
        ├── YYYY-MM-DD/
        │   ├── flood_monitoring/
        │   │   ├── HH-MM-SS.json
        │   ├── met_office/
        │   ├── traffic/
        │   └── ...
    """
    
    def __init__(self, base_path: str = "owl_data"):
        """Initialize the database"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Database initialized at {self.base_path}")
    
    def store(self, source_type: str, data: Dict[Any, Any], confidence: float = 0.5):
        """
        Store raw data with metadata
        
        Args:
            source_type: Type of data source (e.g., 'flood_monitoring', 'traffic')
            data: Raw data to store
            confidence: Initial confidence score (0.0-1.0)
        """
        # Create timestamp-based path
        now = datetime.now()
        date_dir = self.base_path / now.strftime("%Y-%m-%d")
        source_dir = date_dir / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp and hash for uniqueness
        timestamp = now.strftime("%H-%M-%S-%f")
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
        filename = f"{timestamp}_{data_hash}.json"
        filepath = source_dir / filename
        
        # Wrap data with metadata following Owl philosophy
        wrapped_data = {
            "metadata": {
                "timestamp": now.isoformat(),
                "source_type": source_type,
                "initial_confidence": confidence,
                "current_confidence": confidence,
                "decay_applied": False,
                "ingestion_layer": "COLLECT",
                "data_id": f"{source_type}_{data_hash}",
                "file_path": str(filepath)
            },
            "raw_data": data
        }
        
        # Write to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(wrapped_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✓ Stored {source_type} data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to store {source_type} data: {e}")
            return False
    
    def store_batch(self, source_type: str, data_list: List[Dict], confidence: float = 0.5):
        """
        Store multiple records at once
        
        Args:
            source_type: Type of data source
            data_list: List of data records to store
            confidence: Initial confidence score
        """
        success_count = 0
        
        for data in data_list:
            if self.store(source_type, data, confidence):
                success_count += 1
        
        logger.info(f"✓ Stored {success_count}/{len(data_list)} records for {source_type}")
        return success_count
    
    def get_latest(self, source_type: str, count: int = 10) -> List[Dict]:
        """
        Retrieve latest N records for a source type
        
        Args:
            source_type: Type of data source
            count: Number of latest records to retrieve
        
        Returns:
            List of data records
        """
        records = []
        
        # Search in date directories (newest first)
        date_dirs = sorted(self.base_path.glob("*"), reverse=True)
        
        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue
            
            source_dir = date_dir / source_type
            if not source_dir.exists():
                continue
            
            # Get files sorted by timestamp (newest first)
            files = sorted(source_dir.glob("*.json"), reverse=True)
            
            for filepath in files:
                if len(records) >= count:
                    break
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        records.append(data)
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
            
            if len(records) >= count:
                break
        
        return records[:count]
    
    def get_by_date_range(self, source_type: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Retrieve records within a date range
        
        Args:
            source_type: Type of data source
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            List of data records
        """
        records = []
        
        # Iterate through date range
        date_dirs = sorted(self.base_path.glob("*"))
        
        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue
            
            date_str = date_dir.name
            if date_str < start_date or date_str > end_date:
                continue
            
            source_dir = date_dir / source_type
            if not source_dir.exists():
                continue
            
            # Read all files in this directory
            for filepath in source_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        records.append(data)
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
        
        return records
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with counts per source type
        """
        stats = {}
        
        for date_dir in self.base_path.glob("*"):
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
    
    def update_confidence(self, data_id: str, new_confidence: float):
        """
        Update confidence score for a specific record
        (Used in later layers)
        
        Args:
            data_id: Unique data identifier
            new_confidence: New confidence score
        """
        # Search for the file with this data_id
        for filepath in self.base_path.rglob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get('metadata', {}).get('data_id') == data_id:
                    # Update confidence
                    data['metadata']['current_confidence'] = new_confidence
                    
                    # Write back
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"✓ Updated confidence for {data_id} to {new_confidence}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error updating {filepath}: {e}")
        
        logger.warning(f"Data ID {data_id} not found")
        return False
