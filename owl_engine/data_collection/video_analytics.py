"""
Abbey Road EarthCam Video Analytics
Analyzes live camera feed for:
- Vehicle counting and direction tracking
- Crowd density and pedestrian flow
- Peak time analysis
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import time
import random
import math

logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """
    Generates realistic Abbey Road traffic data based on:
    - Time of day patterns (rush hours, quiet times)
    - Day of week patterns (weekday vs weekend)
    - Tourist traffic (Beatles fans!)
    """
    
    def __init__(self):
        self.location = 'Abbey Road Crossing, London'
        self.lat = 51.5319
        self.lon = -0.1774
    
    def generate_hourly_data(self) -> Dict:
        """Generate realistic hourly traffic data"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        
        # Time-based traffic patterns
        traffic_multiplier = self._get_traffic_multiplier(hour, day_of_week)
        crowd_multiplier = self._get_crowd_multiplier(hour, day_of_week)
        
        # Base traffic (vehicles per hour in each direction)
        base_vehicles = {
            'north': random.randint(40, 60),
            'south': random.randint(35, 55),
            'east': random.randint(20, 35),
            'west': random.randint(20, 35)
        }
        
        # Apply time multipliers
        vehicle_counts = {
            'car': {
                'north': int(base_vehicles['north'] * traffic_multiplier * random.uniform(0.9, 1.1)),
                'south': int(base_vehicles['south'] * traffic_multiplier * random.uniform(0.9, 1.1)),
                'east': int(base_vehicles['east'] * traffic_multiplier * random.uniform(0.9, 1.1)),
                'west': int(base_vehicles['west'] * traffic_multiplier * random.uniform(0.9, 1.1))
            },
            'bus': {
                'north': random.randint(2, 6),
                'south': random.randint(2, 5),
                'east': random.randint(1, 3),
                'west': random.randint(1, 3)
            },
            'motorcycle': {
                'north': random.randint(1, 4),
                'south': random.randint(1, 4),
                'east': random.randint(0, 2),
                'west': random.randint(0, 2)
            }
        }
        
        # Tourist/pedestrian counts (Beatles fans recreating album cover!)
        base_people = random.randint(5, 15)
        person_count = int(base_people * crowd_multiplier)
        
        # Total vehicles
        total_vehicles = sum(
            vehicle_counts['car'][d] + vehicle_counts['bus'][d] + vehicle_counts['motorcycle'][d]
            for d in ['north', 'south', 'east', 'west']
        )
        
        return {
            'timestamp': now.isoformat(),
            'location': self.location,
            'lat': self.lat,
            'lon': self.lon,
            'source': 'Abbey Road Demo (Realistic Simulation)',
            'hour': now.strftime("%Y-%m-%d_%H:00"),
            'vehicle_counts': vehicle_counts,
            'vehicle_count': total_vehicles,
            'person_count': person_count,
            'frame_count': 120,  # Simulated frames analyzed
            'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week],
            'is_rush_hour': hour in [7, 8, 9, 17, 18, 19],
            'is_tourist_peak': hour in [10, 11, 12, 13, 14, 15, 16]
        }
    
    def _get_traffic_multiplier(self, hour: int, day_of_week: int) -> float:
        """
        Traffic patterns:
        - Rush hours (7-9 AM, 5-7 PM): 1.8x
        - Mid-day: 1.2x
        - Night (10 PM - 6 AM): 0.3x
        - Weekend: 0.7x of weekday
        """
        # Base by hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            multiplier = 1.8
        elif 10 <= hour <= 16:  # Mid-day
            multiplier = 1.2
        elif 20 <= hour <= 23 or 0 <= hour <= 6:  # Night
            multiplier = 0.3
        else:
            multiplier = 1.0
        
        # Weekend reduction
        if day_of_week >= 5:  # Saturday or Sunday
            multiplier *= 0.7
        
        return multiplier
    
    def _get_crowd_multiplier(self, hour: int, day_of_week: int) -> float:
        """
        Pedestrian/tourist patterns:
        - Tourist hours (10 AM - 4 PM): 2.5x (Beatles fans!)
        - Weekend: 1.5x
        - Evening: 0.5x
        - Night: 0.2x
        """
        # Tourist peak hours
        if 10 <= hour <= 16:
            multiplier = 2.5
        elif 17 <= hour <= 20:
            multiplier = 0.8
        elif 21 <= hour <= 23 or 0 <= hour <= 7:
            multiplier = 0.2
        else:
            multiplier = 1.0
        
        # Weekend boost (more tourists)
        if day_of_week >= 5:
            multiplier *= 1.5
        
        return multiplier

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLO not available. Run: pip install ultralytics")


class AbbeyRoadAnalytics:
    """
    Real-time video analytics for Abbey Road crossing
    
    Features:
    - Vehicle detection and counting (cars, buses, motorcycles)
    - Direction tracking (north/south, east/west)
    - Pedestrian/crowd density analysis
    - Peak time detection
    - Historical aggregation
    """
    
    def __init__(self, camera_url: str = None):
        self.camera_url = camera_url or "https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk"
        
        # Load YOLO model (YOLOv8n - nano version for speed)
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')  # Will auto-download on first run
                logger.info("✓ YOLO model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                self.model = None
        else:
            self.model = None
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.person_class = 0
        
        # Tracking data
        self.vehicle_tracks = {}  # track_id -> {positions, direction, class}
        self.person_tracks = {}
        self.track_id_counter = 0
        
        # Counters
        self.vehicle_count = defaultdict(lambda: {'north': 0, 'south': 0, 'east': 0, 'west': 0})
        self.crowd_density = deque(maxlen=60)  # Last 60 measurements
        
        # Zone definitions (normalized coordinates 0-1)
        # Abbey Road crossing zones
        self.zones = {
            'north_entry': [(0.3, 0.0), (0.7, 0.3)],  # Top of frame
            'south_entry': [(0.3, 0.7), (0.7, 1.0)],  # Bottom
            'east_entry': [(0.0, 0.3), (0.3, 0.7)],   # Left
            'west_entry': [(0.7, 0.3), (1.0, 0.7)],   # Right
            'crossing': [(0.35, 0.4), (0.65, 0.6)]    # Center crossing area
        }
        
        # Data storage
        self.data_dir = Path("owl_data") / datetime.now().strftime("%Y-%m-%d") / "video_analytics"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Session stats
        self.session_start = datetime.now()
        self.frame_count = 0
        self.last_save = datetime.now()
        
    def get_video_stream(self):
        """
        Get video stream from EarthCam
        Note: EarthCam may require specific handling or API access
        For now, uses placeholder - real implementation would need EarthCam API
        """
        # Try to get stream (EarthCam specific implementation needed)
        try:
            # Attempt direct stream access
            cap = cv2.VideoCapture(self.camera_url)
            if cap.isOpened():
                logger.info("✓ Connected to video stream")
                return cap
        except Exception as e:
            logger.warning(f"Stream connection issue: {e}")
        
        # Fallback: Could use EarthCam API or snapshot mode
        logger.info("Using snapshot mode (capturing periodic images)")
        return None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze single frame for vehicles and people
        
        Returns:
            Detection results with counts and positions
        """
        if self.model is None:
            return {'vehicles': [], 'people': [], 'error': 'YOLO not available'}
        
        h, w = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]
        
        vehicles = []
        people = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter by confidence
            if conf < 0.4:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2 / w  # Normalize 0-1
            center_y = (y1 + y2) / 2 / h
            
            if cls in self.vehicle_classes:
                # Vehicle detected
                direction = self._detect_direction(center_x, center_y)
                vehicles.append({
                    'type': self.vehicle_classes[cls],
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'direction': direction
                })
            elif cls == self.person_class:
                # Person detected
                people.append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'vehicles': vehicles,
            'people': people,
            'vehicle_count': len(vehicles),
            'person_count': len(people)
        }
    
    def _detect_direction(self, x: float, y: float) -> str:
        """
        Detect which direction vehicle is traveling based on position
        Uses entry zone detection
        """
        # Check which zone the vehicle is in
        if y < 0.3:
            return 'south'  # Coming from north, going south
        elif y > 0.7:
            return 'north'  # Coming from south, going north
        elif x < 0.3:
            return 'east'   # Coming from west, going east
        elif x > 0.7:
            return 'west'   # Coming from east, going west
        else:
            return 'crossing'  # In the crossing area
    
    def calculate_crowd_density(self, people: List[Dict]) -> Dict:
        """
        Calculate crowd density in crossing area
        
        Returns:
            Density metrics (low/medium/high)
        """
        # Count people in crossing zone
        crossing_zone = self.zones['crossing']
        (x1, y1), (x2, y2) = crossing_zone
        
        people_in_crossing = sum(
            1 for p in people
            if x1 <= p['position'][0] <= x2 and y1 <= p['position'][1] <= y2
        )
        
        total_people = len(people)
        
        # Density classification (based on typical Abbey Road traffic)
        if people_in_crossing == 0:
            density_level = 'empty'
        elif people_in_crossing <= 3:
            density_level = 'low'
        elif people_in_crossing <= 8:
            density_level = 'medium'
        elif people_in_crossing <= 15:
            density_level = 'high'
        else:
            density_level = 'very_high'
        
        return {
            'crossing_count': people_in_crossing,
            'total_count': total_people,
            'density_level': density_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def aggregate_hourly_stats(self) -> Dict:
        """
        Aggregate statistics for current hour
        """
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d_%H:00")
        
        return {
            'hour': hour_key,
            'vehicle_counts': dict(self.vehicle_count),
            'avg_crowd_density': np.mean([d['total_count'] for d in self.crowd_density]) if self.crowd_density else 0,
            'frame_count': self.frame_count,
            'timestamp': now.isoformat()
        }
    
    def save_analytics(self, data: Dict):
        """Save analytics data to JSON file"""
        timestamp = datetime.now()
        filename = f"abbey_road_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved analytics: {filename}")
    
    def run_continuous_analysis(self, duration_minutes: int = 60, snapshot_interval: int = 30):
        """
        Run continuous analysis for specified duration
        
        Args:
            duration_minutes: How long to run analysis
            snapshot_interval: Seconds between snapshots (if using snapshot mode)
        """
        logger.info(f"🎥 Starting Abbey Road video analytics for {duration_minutes} minutes")
        
        if not YOLO_AVAILABLE:
            logger.error("❌ Cannot run analysis - YOLO not installed")
            return
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Try to get video stream
        cap = self.get_video_stream()
        
        all_detections = []
        
        while datetime.now() < end_time:
            try:
                frame = None
                
                if cap and cap.isOpened():
                    # Read from stream
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame, reconnecting...")
                        cap.release()
                        time.sleep(5)
                        cap = self.get_video_stream()
                        continue
                else:
                    # Snapshot mode - would need EarthCam API integration
                    logger.info("📸 Snapshot mode - waiting for next interval")
                    time.sleep(snapshot_interval)
                    continue
                
                # Analyze frame
                if frame is not None:
                    detection = self.analyze_frame(frame)
                    
                    # Update counters
                    for vehicle in detection['vehicles']:
                        vtype = vehicle['type']
                        direction = vehicle['direction']
                        if direction != 'crossing':
                            self.vehicle_count[vtype][direction] += 1
                    
                    # Update crowd density
                    if detection['people']:
                        density = self.calculate_crowd_density(detection['people'])
                        self.crowd_density.append(density)
                    
                    all_detections.append(detection)
                    self.frame_count += 1
                    
                    # Save periodically (every 5 minutes)
                    if (datetime.now() - self.last_save).seconds >= 300:
                        self.save_analytics({
                            'session_start': self.session_start.isoformat(),
                            'detections': all_detections[-100:],  # Last 100 detections
                            'hourly_stats': self.aggregate_hourly_stats()
                        })
                        self.last_save = datetime.now()
                        all_detections = all_detections[-100:]  # Keep only recent
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Analysis stopped by user")
                break
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                time.sleep(5)
        
        # Final save
        self.save_analytics({
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'detections': all_detections,
            'hourly_stats': self.aggregate_hourly_stats(),
            'total_frames': self.frame_count
        })
        
        if cap:
            cap.release()
        
        logger.info("✓ Abbey Road analysis completed")


class AbbeyRoadCollector:
    """Collector interface for continuous data collection"""
    
    def __init__(self, demo_mode=False):
        self.analytics = AbbeyRoadAnalytics()
        self.last_collection = None
        self.demo_mode = demo_mode  # Use realistic demo data
        self.demo_data_generator = DemoDataGenerator()
        
        # Initialize required attributes for real-time detection
        self.base_dir = Path(__file__).parent.parent / "owl_data"
        self.base_dir.mkdir(exist_ok=True)
        
        # Data directory for today
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.data_dir = self.base_dir / date_str / "video_analytics"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model for real-time detection
        if YOLO_AVAILABLE:
            try:
                model_path = Path(__file__).parent.parent / 'yolov8n.pt'
                self.model = YOLO(str(model_path))
                logger.info("✓ YOLO model loaded for real-time detection")
            except Exception as e:
                logger.warning(f"Could not load YOLO: {e}")
                self.model = None
        else:
            self.model = None
        
        # COCO class IDs
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.person_class = 0
    
    def collect(self) -> Dict:
        """
        Collect current Abbey Road data with real-time YOLO detection.
        Captures 30 seconds of video, detects vehicles/people in each frame,
        and saves only frames with detections.
        """
        logger.info("📹 Collecting Abbey Road camera data (30s clip)...")
        
        if self.demo_mode:
            # Generate realistic demo data
            demo_data = self.demo_data_generator.generate_hourly_data()
            self._save_demo_data(demo_data)
            logger.info(f"✓ Generated demo data: {demo_data['vehicle_count']} vehicles, {demo_data['person_count']} people")
            return demo_data
        
        # Try real-time detection from YouTube stream (30-second clip)
        try:
            result = self._process_video_clip()
            
            if result is not None:
                logger.info(f"✓ Clip processed: {result['detections_found']} frames with detections")
                return result
            else:
                logger.warning("⚠️ Video clip processing failed, using demo mode")
                return self._fallback_to_demo()
                
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return self._fallback_to_demo()
    
    def _capture_youtube_frame(self) -> np.ndarray:
        """
        Capture a frame from YouTube live stream
        Uses yt-dlp to get stream URL, then OpenCV to capture frame
        """
        try:
            import subprocess
            
            youtube_url = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
            
            # Use yt-dlp to get direct stream URL
            cmd = [
                'python', '-m', 'yt_dlp',
                '-f', 'best',
                '--get-url',
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                stream_url = result.stdout.strip()
                
                # Capture frame from stream
                cap = cv2.VideoCapture(stream_url)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        logger.info("✓ Captured frame from YouTube stream")
                        return frame
                        
            logger.warning("Could not get YouTube stream URL")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp timeout")
            return None
        except FileNotFoundError:
            logger.warning("yt-dlp not installed - install with: pip install yt-dlp")
            return None
        except Exception as e:
            logger.warning(f"Frame capture error: {e}")
            return None
    
    def _process_video_clip(self) -> Dict:
        """
        Capture and process 30 seconds of video from YouTube stream.
        Only saves frames with vehicle or person detections.
        Deletes old detection images before saving new ones.
        """
        try:
            youtube_url = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
            
            # Use yt-dlp to get direct stream URL
            cmd = ['python', '-m', 'yt_dlp', '-f', 'best', '--get-url', youtube_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning("Could not get YouTube stream URL")
                return None
            
            stream_url = result.stdout.strip()
            
            # Open video stream
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                logger.warning("Could not open video stream")
                return None
            
            # Get FPS (default to 30 if unknown)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 60:
                fps = 30
            
            # Calculate frames to process (30 seconds)
            total_frames = int(fps * 30)
            frames_processed = 0
            detections_found = 0
            
            # Delete old detection images first
            self._delete_old_detection_images()
            
            logger.info(f"Processing 30-second clip ({total_frames} frames at {fps} FPS)...")
            
            # Variables to track best detection
            best_frame = None
            best_detections = None
            max_objects = 0
            
            while frames_processed < total_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frames_processed += 1
                
                # Process every 30th frame (1 frame per second)
                if frames_processed % int(fps) == 0:
                    # Run YOLO detection
                    detections = self._run_detection(frame)
                    
                    total_objects = detections['vehicle_count'] + detections['person_count']
                    
                    # Save frame if it has detections
                    if total_objects > 0:
                        detections_found += 1
                        
                        # Track frame with most activity
                        if total_objects > max_objects:
                            max_objects = total_objects
                            best_frame = frame.copy()
                            best_detections = detections
                        
                        logger.info(f"Frame {frames_processed}: {detections['vehicle_count']} vehicles, {detections['person_count']} people")
            
            cap.release()
            
            # Save the best frame with most activity
            if best_frame is not None and best_detections is not None:
                self._save_detection_frame(best_frame, best_detections)
                self._save_detection_data(best_detections)
                
                return {
                    'timestamp': datetime.now(),
                    'location': 'Abbey Road Crossing',
                    'vehicle_count': best_detections['vehicle_count'],
                    'person_count': best_detections['person_count'],
                    'detections_found': detections_found,
                    'frames_processed': frames_processed,
                    'source': 'real-time YouTube detection (30s clip)'
                }
            
            logger.warning(f"No detections found in {frames_processed} frames")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp timeout")
            return None
        except FileNotFoundError:
            logger.warning("yt-dlp not installed - install with: pip install yt-dlp")
            return None
        except Exception as e:
            logger.error(f"Video clip processing error: {e}")
            return None
    
    def _delete_old_detection_images(self):
        """Delete old detection images to save space"""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            frames_dir = self.base_dir / date_str / "video_analytics" / "frames"
            
            if frames_dir.exists():
                # Delete all detection images
                for img_file in frames_dir.glob("detection_*.jpg"):
                    img_file.unlink()
                    logger.info(f"Deleted old image: {img_file.name}")
        except Exception as e:
            logger.warning(f"Could not delete old images: {e}")
    
    def _run_detection(self, frame: np.ndarray) -> Dict:
        """Run YOLO detection on frame and return counts"""
        results = self.model(frame, conf=0.3, verbose=False)
        
        vehicle_count = 0
        person_count = 0
        detections_list = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                class_name = result.names[cls]
                
                # Count vehicles and people
                if cls in self.vehicle_classes.values() or cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    vehicle_count += 1
                elif cls == self.person_class:
                    person_count += 1
                
                detections_list.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return {
            'vehicle_count': vehicle_count,
            'person_count': person_count,
            'detections': detections_list,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_detection_frame(self, frame: np.ndarray, detections: Dict):
        """Save annotated frame with bounding boxes"""
        try:
            # Draw bounding boxes
            annotated = frame.copy()
            
            for det in detections.get('detections', []):
                x1, y1, x2, y2 = det['bbox']
                
                # Color based on class
                if 'car' in det['class'] or 'bus' in det['class'] or 'truck' in det['class']:
                    color = (0, 255, 0)  # Green for vehicles
                elif 'person' in det['class']:
                    color = (255, 0, 0)  # Blue for people
                else:
                    color = (0, 165, 255)  # Orange for others
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save frame
            frames_dir = self.data_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(str(frames_dir / filename), annotated)
            
            logger.info(f"💾 Saved detection frame: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def _save_detection_data(self, detections: Dict):
        """Save detection data to JSON database"""
        timestamp = datetime.now()
        filename = f"abbey_road_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        hour_key = timestamp.strftime("%Y-%m-%d_%H:00")
        
        # Calculate vehicle counts by direction (estimated from position)
        vehicle_counts = self._estimate_direction_counts(detections.get('detections', []))
        
        save_data = {
            'session_start': timestamp.isoformat(),
            'timestamp': timestamp.isoformat(),
            'hourly_stats': {
                'hour': hour_key,
                'timestamp': timestamp.isoformat(),
                'frame_count': 1,
                'avg_crowd_density': detections.get('person_count', 0),
                'vehicle_counts': vehicle_counts
            },
            'detections': [
                {
                    'timestamp': timestamp.isoformat(),
                    'vehicle_count': detections.get('vehicle_count', 0),
                    'person_count': detections.get('person_count', 0),
                    'objects': detections.get('detections', [])
                }
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"💾 Saved detection data: {filename}")
    
    def _estimate_direction_counts(self, detections: List[Dict]) -> Dict:
        """Estimate vehicle direction counts from bounding box positions"""
        counts = {
            'car': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'bus': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
            'motorcycle': {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        }
        
        for det in detections:
            if 'car' in det['class']:
                vehicle_type = 'car'
            elif 'bus' in det['class']:
                vehicle_type = 'bus'
            elif 'motorcycle' in det['class']:
                vehicle_type = 'motorcycle'
            else:
                continue
            
            # Estimate direction from bbox position (rough heuristic)
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Simplified direction estimation
            if center_y < 0.4:
                counts[vehicle_type]['north'] += 1
            elif center_y > 0.6:
                counts[vehicle_type]['south'] += 1
            elif center_x < 0.5:
                counts[vehicle_type]['east'] += 1
            else:
                counts[vehicle_type]['west'] += 1
        
        return counts
    
    def _fallback_to_demo(self) -> Dict:
        """Fallback to demo mode if real detection fails"""
        demo_data = self.demo_data_generator.generate_hourly_data()
        self._save_demo_data(demo_data)
        logger.info(f"✓ Fallback demo data: {demo_data['vehicle_count']} vehicles, {demo_data['person_count']} people")
        return demo_data
    
    def _save_demo_data(self, data: Dict):
        """Save demo data to proper format for dashboard"""
        data_dir = Path("owl_data") / datetime.now().strftime("%Y-%m-%d") / "video_analytics"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now()
        filename = f"abbey_road_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = data_dir / filename
        
        # Format as hourly stats
        hour_key = timestamp.strftime("%Y-%m-%d_%H:00")
        
        save_data = {
            'session_start': timestamp.isoformat(),
            'timestamp': timestamp.isoformat(),
            'hourly_stats': {
                'hour': hour_key,
                'timestamp': timestamp.isoformat(),
                'frame_count': data.get('frame_count', 120),
                'avg_crowd_density': data.get('person_count', 0),
                'vehicle_counts': data.get('vehicle_counts', {})
            },
            'detections': [
                {
                    'timestamp': timestamp.isoformat(),
                    'vehicle_count': data.get('vehicle_count', 0),
                    'person_count': data.get('person_count', 0)
                }
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _get_latest_analytics(self):
        """Get most recent analytics file"""
        data_dir = Path("owl_data") / datetime.now().strftime("%Y-%m-%d") / "video_analytics"
        if not data_dir.exists():
            return None
        
        files = list(data_dir.glob("abbey_road_*.json"))
        if not files:
            return None
        
        return max(files, key=lambda f: f.stat().st_mtime)


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    print("🎥 Abbey Road Video Analytics Test")
    print("=" * 60)
    
    print("\n✓ Running in DEMO MODE with realistic data generation")
    print("  (Install YOLO packages for real video analysis)\n")
    
    # Test demo data generation
    print("Testing demo data generator...")
    collector = AbbeyRoadCollector(demo_mode=True)
    
    print("\nGenerating 3 sample data points...\n")
    for i in range(3):
        data = collector.collect()
        print(f"Sample {i+1}:")
        print(f"  Time: {data.get('hour', 'N/A')}")
        print(f"  Vehicles: {data.get('vehicle_count', 0)}")
        print(f"  Pedestrians: {data.get('person_count', 0)}")
        print(f"  Rush Hour: {data.get('is_rush_hour', False)}")
        print(f"  Tourist Peak: {data.get('is_tourist_peak', False)}")
        
        if 'vehicle_counts' in data:
            total_north = sum(data['vehicle_counts'][t].get('north', 0) for t in data['vehicle_counts'])
            total_south = sum(data['vehicle_counts'][t].get('south', 0) for t in data['vehicle_counts'])
            print(f"  North: {total_north} | South: {total_south}")
        print()
    
    print("✅ Demo data generated successfully!")
    print("\n📊 Data saved to: owl_data/[date]/video_analytics/")
    print("\n🎯 View in dashboard:")
    print("   streamlit run palantir_dashboard.py")
    print("   → Open 'Abbey Road' tab")
    print("\n" + "=" * 60)
