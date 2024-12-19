import torch
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import subprocess
from dataclasses import dataclass
from typing import Tuple, List, Dict
import logging
import pyttsx3
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_tracker.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Detection:
    id: int
    box: Tuple[int, int, int, int]  # x, y, w, h
    class_id: int
    confidence: float
    status: str

class Notifier:
    def __init__(self, enable_voice=True):
        self.enable_voice = enable_voice
        self.last_notification = defaultdict(float)
        self.notification_cooldown = 5  # seconds
        if self.enable_voice:
            try:
                self.engine = pyttsx3.init()
            except Exception as e:
                logging.warning(f"Failed to initialize text-to-speech: {e}")
                self.enable_voice = False

    def speak(self, text, event_type="default"):
        current_time = time.time()
        if current_time - self.last_notification[event_type] >= self.notification_cooldown:
            logging.info(f"Notification: {text}")
            if self.enable_voice:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logging.error(f"Failed to speak: {e}")
            self.last_notification[event_type] = current_time

class VehicleTracker:
    def __init__(self, 
                 model_path: str = 'yolov8s.pt',
                 confidence_threshold: float = 0.4,
                 max_disappeared: int = 300,
                 iou_threshold: float = 0.3):
        self.notifier = Notifier()
        self._initialize_device()
        self._load_model(model_path)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
        # Tracking state
        self.next_vehicle_id = 0
        self.vehicles: Dict[int, dict] = {}
        self.vehicle_history = defaultdict(list)
        
        # Vehicle classes in YOLO v8 (car, bus, truck)
        self.vehicle_classes = {2: "car", 5: "bus", 7: "truck"}
        
        # Performance metrics
        self.fps_stats = []
        self.last_fps_print = time.time()
        
        logging.info("Vehicle tracker initialized successfully")

    def _initialize_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")
        self.notifier.speak(f"System initialized using {self.device} device")

    def _load_model(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logging.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        if frame is None:
            raise ValueError("Empty frame received")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(frame_rgb, verbose=False)
        current_detections = self._process_detections(results)
        
        # Update tracking
        self._update_tracking(current_detections)
        
        # Draw results
        annotated_frame = self._draw_results(frame.copy())
        
        return annotated_frame, current_detections

    def _process_detections(self, results) -> List[Tuple[int, int, int, int]]:
        current_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > self.confidence_threshold and cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    current_detections.append((
                        int(x1), int(y1), 
                        int(x2 - x1), int(y2 - y1)  # convert to width, height
                    ))
        
        return current_detections

    def _update_tracking(self, current_detections: List[Tuple[int, int, int, int]]):
        # Update disappearance counters
        for vehicle_id in self.vehicles:
            self.vehicles[vehicle_id]["disappeared"] += 1

        # Match new detections to existing vehicles
        matched_vehicles = set()
        matched_detections = set()

        for idx, detection in enumerate(current_detections):
            best_iou = self.iou_threshold
            best_match = None

            for vehicle_id, vehicle_info in self.vehicles.items():
                if vehicle_id in matched_vehicles:
                    continue

                iou = self._calculate_iou(detection, vehicle_info["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = vehicle_id

            if best_match is not None:
                self._update_vehicle(best_match, detection)
                matched_vehicles.add(best_match)
                matched_detections.add(idx)

        # Add new vehicles
        for idx, detection in enumerate(current_detections):
            if idx not in matched_detections:
                self._add_new_vehicle(detection)

        # Remove stale vehicles
        self._remove_stale_vehicles()

    def _update_vehicle(self, vehicle_id: int, box: Tuple[int, int, int, int]):
        self.vehicles[vehicle_id]["box"] = box
        self.vehicles[vehicle_id]["disappeared"] = 0
        self.vehicle_history[vehicle_id].append(time.time())

    def _add_new_vehicle(self, box: Tuple[int, int, int, int]):
        self.notifier.speak("New vehicle detected", "arrival")
        self.vehicles[self.next_vehicle_id] = {
            "box": box,
            "disappeared": 0,
            "first_seen": time.time()
        }
        self.vehicle_history[self.next_vehicle_id].append(time.time())
        self.next_vehicle_id += 1

    def _remove_stale_vehicles(self):
        for vehicle_id in list(self.vehicles.keys()):
            if self.vehicles[vehicle_id]["disappeared"] > self.max_disappeared:
                duration = time.time() - self.vehicles[vehicle_id]["first_seen"]
                self.notifier.speak(f"Vehicle departed after {int(duration)} seconds", "departure")
                logging.info(f"Vehicle {vehicle_id} tracked for {duration:.2f} seconds")
                del self.vehicles[vehicle_id]

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0

    def _draw_results(self, frame: np.ndarray) -> np.ndarray:
        for vehicle_id, vehicle_info in self.vehicles.items():
            if vehicle_info["disappeared"] == 0:
                x, y, w, h = vehicle_info["box"]
                
                # Calculate duration
                duration = time.time() - vehicle_info["first_seen"]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add status text
                status = f"ID: {vehicle_id} ({duration:.1f}s)"
                cv2.putText(frame, status, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add FPS counter
        fps = len(self.fps_stats) / (self.fps_stats[-1] - self.fps_stats[0]) if len(self.fps_stats) > 1 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

def get_video_source():
    """Initialize RTSP camera connection with proper settings"""
    # RTSP camera configuration
    rtsp_url = 'rtsp://admin:Aa80808080@192.168.68.155:554'
    
    # Configure OpenCV capture with RTSP settings
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logging.error(f"Failed to connect to RTSP camera at {rtsp_url}")
        return None
        
    # Optimize RTSP stream settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Use H264 codec
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    logging.info("Successfully connected to RTSP camera")
    return cap

def main():
    tracker = VehicleTracker()
    
    # Initialize RTSP camera
    cap = get_video_source()
    if cap is None:
        return
    
    target_fps = 30
    frame_time = 1/target_fps
    last_frame_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # Add timeout for frame reading
            if time.time() - last_frame_time > 5:  # 5 seconds timeout
                logging.error("Frame reading timeout, attempting to reconnect...")
                cap.release()
                cap = get_video_source()
                if cap is None:
                    break
                last_frame_time = time.time()
                continue
            
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame")
                continue
                
            last_frame_time = time.time()
            
            # Process frame
            processed_frame, _ = tracker.process_frame(frame)
            
            # Display output
            cv2.imshow('Vehicle Detection', processed_frame)
            
            # Maintain target FPS
            processing_time = time.time() - loop_start
            delay = max(1, int((frame_time - processing_time) * 1000))
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Application terminated")

if __name__ == "__main__":
    main()