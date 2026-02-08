"""
Continuous Data Collection - Build historical datasets
Collects: Traffic, Weather, Air Quality automatically
"""
import time
import schedule
from datetime import datetime
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousCollector:
    """
    Continuously collect data from all sources to build timeline
    """
    
    def __init__(self, interval_minutes=10):
        self.interval = interval_minutes
        self.base_path = Path(__file__).parent
        self.traffic_dir = self.base_path / "traffic_data_collected"
        self.owl_data_dir = self.base_path / "owl_data"
        
        # Create directories
        self.traffic_dir.mkdir(exist_ok=True)
        self.owl_data_dir.mkdir(exist_ok=True)
        
        # Import collectors
        try:
            from browser_traffic_collector import TrafficBrowserCollector
            self.traffic_collector = TrafficBrowserCollector()
            logger.info("[OK] Traffic collector loaded")
        except Exception as e:
            logger.warning(f"[WARN] Traffic collector not available: {e}")
            self.traffic_collector = None
        
        try:
            from data_collection.environmental_monitor import EnvironmentalMonitor
            self.env_monitor = EnvironmentalMonitor()
            logger.info("[OK] Environmental monitor loaded")
        except Exception as e:
            logger.warning(f"[WARN] Environmental monitor not available: {e}")
            self.env_monitor = None
        
        try:
            from data_collection.road_disruptions import RoadDisruptionsCollector
            self.disruptions_collector = RoadDisruptionsCollector()
            logger.info("[OK] Road disruptions collector loaded")
        except Exception as e:
            logger.warning(f"[WARN] Road disruptions collector not available: {e}")
            self.disruptions_collector = None
        
        try:
            from data_collection.video_analytics import AbbeyRoadCollector
            self.video_collector = AbbeyRoadCollector()
            logger.info("[OK] Abbey Road video analytics loaded")
        except Exception as e:
            logger.warning(f"[WARN] Video analytics not available: {e}")
            self.video_collector = None
    
    def collect_traffic(self):
        """Collect traffic data"""
        if not self.traffic_collector:
            logger.warning("Traffic collector not available")
            return
        
        try:
            logger.info("[TRAFFIC] Collecting traffic data...")
            results = self.traffic_collector.collect_all_routes()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.traffic_dir / f"traffic_collection_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"[OK] Collected {len(results)} traffic routes → {output_file.name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Traffic collection failed: {e}")
    
    def collect_weather_and_air(self):
        """Collect weather and air quality data"""
        if not self.env_monitor:
            logger.warning("Environmental monitor not available")
            return
        
        try:
            # Create today's folder
            today = datetime.now().strftime("%Y-%m-%d")
            today_folder = self.owl_data_dir / today
            today_folder.mkdir(exist_ok=True)
            
            # Collect air quality
            logger.info("[AIR] Collecting air quality data...")
            air_quality_dir = today_folder / "air_quality"
            air_quality_dir.mkdir(exist_ok=True)
            
            air_data = self.env_monitor.collect_air_quality()
            
            if air_data:
                timestamp = datetime.now().strftime("%H-%M-%S-%f")
                filename = f"{timestamp}_{hash(str(air_data)) % 100000000:08x}.json"
                output_file = air_quality_dir / filename
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(air_data, f, indent=2)
                
                logger.info(f"[OK] Air quality collected → {output_file.name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Environmental collection failed: {e}")
    
    def collect_road_disruptions(self):
        """Collect road disruptions from TfL"""
        if not self.disruptions_collector:
            logger.warning("Road disruptions collector not available")
            return
        
        try:
            logger.info("[ROADS] Collecting road disruptions...")
            disruptions = self.disruptions_collector.collect()
            
            if disruptions:
                logger.info(f"[OK] Collected {len(disruptions)} road disruptions")
            
        except Exception as e:
            logger.error(f"[ERROR] Road disruptions collection failed: {e}")
    
    def collect_video_analytics(self):
        """Collect Abbey Road camera analytics"""
        if not self.video_collector:
            logger.warning("Video analytics collector not available")
            return
        
        try:
            logger.info("[VIDEO] Collecting Abbey Road video analytics...")
            data = self.video_collector.collect()
            
            if data:
                logger.info(f"[OK] Abbey Road analytics collected")
            
        except Exception as e:
            logger.error(f"[ERROR] Video analytics collection failed: {e}")
    
    def collect_all(self):
        """Single collection run - all sources"""
        logger.info(f"\n{'='*60}")
        logger.info(f"[COLLECT] Collection run at {datetime.now()}")
        logger.info(f"{'='*60}")
        
        # Collect from all sources
        self.collect_traffic()
        self.collect_weather_and_air()
        self.collect_road_disruptions()
        self.collect_video_analytics()
        
        logger.info(f"{'='*60}")
        logger.info("[OK] Collection run complete")
        logger.info(f"{'='*60}\n")
    
    def run_continuous(self):
        """
        Run collection every X minutes indefinitely
        Press Ctrl+C to stop
        """
        print("\n" + "="*70)
        print("🦉 OWL ENGINE - CONTINUOUS DATA COLLECTION")
        print("="*70)
        print(f"Collecting every {self.interval} minutes")
        print(f"Traffic data → {self.traffic_dir}")
        print(f"Environmental data → {self.owl_data_dir}")
        print("\nPress Ctrl+C to stop\n")
        print("="*70 + "\n")
        
        # Schedule the job
        schedule.every(self.interval).minutes.do(self.collect_all)
        
        # Run first collection immediately
        self.collect_all()
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("⏹️  Collection stopped by user")
            print("="*70 + "\n")


if __name__ == "__main__":
    # Collect every 10 minutes by default
    # Can be changed: ContinuousCollector(interval_minutes=5)
    collector = ContinuousCollector(interval_minutes=10)
    collector.run_continuous()
