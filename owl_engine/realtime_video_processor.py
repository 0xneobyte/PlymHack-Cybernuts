"""
Real-Time 24/7 Video Analytics Processor
Continuously captures and analyzes Abbey Road live stream
Detects and counts vehicles/people in real-time
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import subprocess
import time
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("YOLO not available! Install: pip install ultralytics")


class RealtimeVideoProcessor:
    """
    24/7 Real-Time Video Analytics
    - Continuously processes Abbey Road live stream
    - YOLO detection on every frame
    - Saves latest detection frame
    - Counts vehicles/people in real-time
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent / "owl_data"
        self.youtube_url = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
        
        # Load YOLO model
        if not YOLO_AVAILABLE:
            raise Exception("YOLO not available! Install: pip install ultralytics")
        
        model_path = Path(__file__).parent / 'yolov8n.pt'
        self.model = YOLO(str(model_path))
        logger.info("[OK] YOLO model loaded")
        
        # COCO class IDs
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.person_class = 0
        
        # Stats tracking
        self.total_vehicles_today = 0
        self.total_people_today = 0
        self.hourly_vehicle_count = 0
        self.hourly_people_count = 0
        self.current_hour = datetime.now().hour
        self.session_start = datetime.now()
        self.frames_processed = 0
        
        # Frame rotation for dashboard (keep last 20 frames for smooth display)
        self.frame_buffer_size = 20
        self.current_frame_index = 0
        
        # Create output directory
        self.setup_output_dir()
        
    def setup_output_dir(self):
        """Setup output directory for today"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.data_dir = self.base_dir / date_str / "video_analytics"
        self.frames_dir = self.data_dir / "realtime_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats file
        self.stats_file = self.data_dir / "realtime_stats.json"
        logger.info(f"[OK] Output directory: {self.frames_dir}")
    
    def get_stream_url(self):
        """Get direct stream URL from YouTube"""
        try:
            cmd = ['python', '-m', 'yt_dlp', '-f', 'best', '--get-url', self.youtube_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                stream_url = result.stdout.strip()
                logger.info("[OK] Got YouTube stream URL")
                return stream_url
            else:
                logger.error("[ERROR] Failed to get YouTube stream URL")
                return None
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return None
    
    def detect_objects(self, frame):
        """Run YOLO detection on frame"""
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
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    vehicle_count += 1
                elif cls == self.person_class:
                    person_count += 1
                
                detections_list.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return vehicle_count, person_count, detections_list
    
    def annotate_frame(self, frame, detections):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Color based on class
            if any(v in det['class'].lower() for v in ['car', 'bus', 'truck', 'motorcycle']):
                color = (0, 255, 0)  # Green for vehicles
            elif 'person' in det['class'].lower():
                color = (255, 0, 0)  # Blue for people
            else:
                color = (0, 165, 255)  # Orange for others
            
            # Draw rectangle and label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def save_latest_frame(self, frame, vehicle_count, person_count):
        """Save rotating frames for smooth dashboard display"""
        # Save to rotating buffer (frame_0.jpg to frame_19.jpg)
        frame_path = self.frames_dir / f"frame_{self.current_frame_index}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Also save as latest.jpg for compatibility
        latest_path = self.frames_dir / "latest.jpg"
        cv2.imwrite(str(latest_path), frame)
        
        # Rotate frame index
        self.current_frame_index = (self.current_frame_index + 1) % self.frame_buffer_size
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'vehicle_count': vehicle_count,
            'person_count': person_count,
            'frames_processed': self.frames_processed,
            'session_uptime_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'hourly_vehicle_count': self.hourly_vehicle_count,
            'hourly_people_count': self.hourly_people_count,
            'total_vehicles_today': self.total_vehicles_today,
            'total_people_today': self.total_people_today
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_stats(self, vehicle_count, person_count):
        """Update running statistics"""
        current_hour = datetime.now().hour
        
        # Reset hourly counts if new hour
        if current_hour != self.current_hour:
            logger.info(f"📊 Hour {self.current_hour}: {self.hourly_vehicle_count} vehicles, {self.hourly_people_count} people")
            self.current_hour = current_hour
            self.hourly_vehicle_count = 0
            self.hourly_people_count = 0
        
        # Update counts
        self.hourly_vehicle_count += vehicle_count
        self.hourly_people_count += person_count
        self.total_vehicles_today += vehicle_count
        self.total_people_today += person_count
    
    def process_stream(self):
        """Main processing loop - runs 24/7"""
        logger.info("[START] Starting real-time video processing...")
        logger.info("[INFO] Press Ctrl+C to stop")
        
        last_reconnect = datetime.now()
        reconnect_interval = timedelta(minutes=30)  # Reconnect every 30 min to refresh stream
        
        while True:
            try:
                # Get fresh stream URL
                stream_url = self.get_stream_url()
                if not stream_url:
                    logger.error("[RETRY] Failed to get stream URL, retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                
                # Open video stream
                cap = cv2.VideoCapture(stream_url)
                
                if not cap.isOpened():
                    logger.error("[RETRY] Could not open stream, retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                
                logger.info("[OK] Connected to live stream")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                logger.info(f"[OK] Stream FPS: {fps}")
                
                frame_skip = max(1, int(fps / 2))  # Process 2 frames per second
                frame_count = 0
                
                # Process frames
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning("[WARN] Frame read failed, reconnecting...")
                        break
                    
                    frame_count += 1
                    
                    # Process every Nth frame (2 FPS processing rate)
                    if frame_count % frame_skip == 0:
                        # Run detection
                        vehicle_count, person_count, detections = self.detect_objects(frame)
                        
                        # Annotate frame
                        annotated = self.annotate_frame(frame, detections)
                        
                        # Update stats
                        self.update_stats(vehicle_count, person_count)
                        self.frames_processed += 1
                        
                        # Save latest frame (overwrites previous)
                        self.save_latest_frame(annotated, vehicle_count, person_count)
                        
                        # Log every 10 detections
                        if self.frames_processed % 10 == 0:
                            logger.info(f"[OK] Processed {self.frames_processed} frames | Latest: {vehicle_count}V {person_count}P | Today: {self.total_vehicles_today}V {self.total_people_today}P")
                    
                    # Check if need to reconnect (refresh stream)
                    if datetime.now() - last_reconnect > reconnect_interval:
                        logger.info("[REFRESH] Reconnecting to refresh stream...")
                        last_reconnect = datetime.now()
                        break
                
                cap.release()
                
            except KeyboardInterrupt:
                logger.info("\n[STOP] Stopping real-time processor...")
                break
            except Exception as e:
                logger.error(f"[ERROR] {e}")
                logger.info("[RETRY] Retrying in 10 seconds...")
                time.sleep(10)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🦉 OWL ENGINE - REAL-TIME 24/7 VIDEO ANALYTICS")
    print("="*70)
    print("Continuously processing Abbey Road live stream")
    print("YOLO detection on every frame")
    print("Counting vehicles and people in real-time")
    print("Saving latest detection frame for dashboard")
    print("\nPress Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    processor = RealtimeVideoProcessor()
    processor.process_stream()
