"""
OWL ENGINE - Main Orchestrator
Layer 1: COLLECT (Data Collection & Ingestion)

Philosophy: Never discard data. Everything gets in.
This is the entry point for the entire memory system.
"""

import asyncio
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
import schedule
import time
from concurrent.futures import ThreadPoolExecutor

# Import all data collection modules
from data_collection.flood_monitoring import FloodMonitoringCollector
from data_collection.met_office import MetOfficeCollector
from data_collection.london_datastore import LondonDatastoreCollector
from data_collection.waste_management import WasteManagementCollector
from data_collection.traffic_monitor import TrafficMonitorCollector
from data_collection.environmental_monitor import EnvironmentalMonitor
from data_collection.social_monitor import SocialThreatMonitor
from data_collection.infrastructure_monitor import InfrastructureMonitor
from database.json_store import JSONDatabase

# Configure logging with UTF-8 encoding for Windows compatibility
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for console
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('owl_engine.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('OWL_ENGINE')


class OwlEngine:
    """
    Main Owl Engine Orchestrator
    Coordinates all data collection layers and manages the ingestion pipeline
    """
    
    def __init__(self):
        """Initialize the Owl Engine with all collectors and database"""
        logger.info("🦉 Initializing Owl Engine - COLLECT Layer")
        
        # Initialize database
        self.db = JSONDatabase(base_path="owl_data")
        
        # Initialize all collectors
        self.collectors = {
            'flood_monitoring': FloodMonitoringCollector(self.db),
            'met_office': MetOfficeCollector(self.db),
            'london_datastore': LondonDatastoreCollector(self.db),
            'waste_management': WasteManagementCollector(self.db),
            'traffic_monitor': TrafficMonitorCollector(self.db),
            'environmental': EnvironmentalMonitor(self.db),
            'social_threats': SocialThreatMonitor(self.db),
            'infrastructure': InfrastructureMonitor(self.db)
        }
        
        logger.info(f"✓ Initialized {len(self.collectors)} data collectors")
        
    def collect_all(self):
        """Run all collectors in parallel"""
        logger.info("📡 Starting data collection cycle...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for name, collector in self.collectors.items():
                future = executor.submit(self._safe_collect, name, collector)
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        logger.info("✓ Collection cycle complete")
    
    def _safe_collect(self, name, collector):
        """Safely run a collector with error handling"""
        try:
            logger.info(f"  → Collecting from {name}...")
            collector.collect()
            logger.info(f"  ✓ {name} collection successful")
        except Exception as e:
            logger.error(f"  ✗ {name} collection failed: {e}")
    
    def start_realtime_collection(self):
        """Start real-time collection with scheduling"""
        logger.info("🚀 Starting real-time collection mode")
        
        # Schedule different collectors at different intervals
        
        # High-frequency collectors (every 5 minutes)
        schedule.every(5).minutes.do(self.collectors['flood_monitoring'].collect)
        schedule.every(5).minutes.do(self.collectors['met_office'].collect)
        schedule.every(5).minutes.do(self.collectors['traffic_monitor'].collect)
        
        # Medium-frequency collectors (every 30 minutes)
        schedule.every(30).minutes.do(self.collectors['london_datastore'].collect)
        
        # Low-frequency collectors (every 6 hours)
        schedule.every(6).hours.do(self.collectors['waste_management'].collect)
        
        # Run initial collection
        self.collect_all()
        
        # Keep running
        logger.info("⏰ Scheduler active. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logger.info("🛑 Owl Engine stopped by user")
    
    def collect_once(self):
        """Run a single collection cycle (useful for testing)"""
        logger.info("🔄 Running single collection cycle...")
        self.collect_all()
        logger.info("✓ Single collection complete")
    
    def status(self):
        """Get system status and statistics"""
        logger.info("📊 Owl Engine Status")
        stats = self.db.get_statistics()
        
        for source, count in stats.items():
            logger.info(f"  {source}: {count} records")
        
        return stats


def main():
    """Main entry point - Launches Palantir Intelligence Dashboard"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                   OWL ENGINE v2.0                          ║
    ║            Urban Intelligence Platform (Palantir-Level)      ║
    ║                                                              ║
    ║  ✓ Layer 1: COLLECT  (Multi-Domain Ingestion)               ║
    ║  ✓ Layer 2: EXTRACT  (Entity Parsing)                       ║
    ║  ✓ Layer 3: LINK     (Knowledge Graph)                      ║
    ║  ✓ Layer 4: CORRELATE (Pattern Detection)                   ║
    ║  ✓ Layer 5: VECTORIZE (Semantic Embeddings)                 ║
    ║  ✓ Layer 6: INFER    (Baseline Learning)                    ║
    ║  ✓ Layer 7: ALERT    (Live Event Awareness)                 ║
    ║  ✓ Layer 8: PREDICT  (Threat Forecasting)                   ║
    ║                                                              ║
    ║  🔗 Multi-Domain: Floods • Traffic • Environmental          ║
    ║                   Social • Infrastructure                    ║
    ║                                                              ║
    ║         Philosophy: Everything Gets In, Nothing Lost         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Get the path to Palantir dashboard
    dashboard_path = Path(__file__).parent / "palantir_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"ERROR: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    print("\n🚀 Starting OWL Palantir Intelligence Platform...")
    print("📊 Multi-domain threat correlation...")
    print("🧠 Machine learning pattern detection...")
    print("🔮 Predictive threat forecasting...")
    print("🚨 Real-time compound threat detection...\n")
    
    # Launch Streamlit dashboard
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down OWL Engine...")
    except Exception as e:
        print(f"ERROR launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
