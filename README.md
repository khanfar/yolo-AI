# Vehicle Tracking System
Developed by Khanfar System

## Overview
This system is a real-time vehicle detection and tracking solution that utilizes YOLOv8 (You Only Look Once) for object detection and custom tracking algorithms for vehicle monitoring. The system processes video streams from RTSP cameras and provides real-time analytics and notifications.

## Features
- Real-time vehicle detection using YOLOv8
- Multi-vehicle tracking with unique IDs
- Support for multiple vehicle classes (cars, buses, trucks)
- Real-time performance metrics (FPS monitoring)
- Voice notifications for vehicle events
- RTSP camera stream support with automatic reconnection
- GPU acceleration support (CUDA)

## Technical Architecture

### Core Components

#### 1. Detection System (`Detection` class)
```python
@dataclass
class Detection:
    id: int                              # Unique identifier for each detection
    box: Tuple[int, int, int, int]       # Bounding box coordinates (x, y, w, h)
    class_id: int                        # Vehicle class identifier
    confidence: float                    # Detection confidence score
    status: str                          # Current status of the detection
```

#### 2. Notification System (`Notifier` class)
- Handles voice notifications for vehicle events
- Implements cooldown mechanism to prevent notification spam
- Uses text-to-speech for audio feedback
- Logging capabilities for event tracking

#### 3. Vehicle Tracker (`VehicleTracker` class)
Main processing unit that handles:
- Model initialization and device selection (CPU/CUDA/MPS)
- Real-time frame processing
- Vehicle detection and tracking
- Performance monitoring
- Visual output generation

### Key Functions and Their Roles

#### Device Management
- `_initialize_device()`: Selects optimal processing device (CUDA GPU if available)
- `_load_model()`: Initializes YOLOv8 model with specified weights

#### Detection Processing
- `process_frame()`: Main processing pipeline for each video frame
- `_process_detections()`: Processes raw YOLOv8 detections
- `_update_tracking()`: Updates vehicle tracking information

#### Tracking Algorithm
- `_calculate_iou()`: Calculates Intersection over Union for object tracking
- `_update_vehicle()`: Updates tracked vehicle information
- `_add_new_vehicle()`: Handles new vehicle detection
- `_remove_stale_vehicles()`: Manages vehicle tracking termination

#### Video Stream Handling
- `get_video_source()`: Manages RTSP camera connection
- Implements automatic reconnection
- Optimizes stream settings for performance

### Implementation Details

#### Vehicle Classification
```python
vehicle_classes = {
    2: "car",
    5: "bus",
    7: "truck"
}
```

#### Performance Optimization
- Buffer size optimization for RTSP streams
- Frame processing timing control
- GPU acceleration when available
- Efficient memory management for tracking data

#### Error Handling
- Connection timeout detection
- Automatic stream reconnection
- Graceful error recovery
- Comprehensive logging system

## Dependencies
- PyTorch: Deep learning framework
- Ultralytics YOLOv8: Object detection model
- OpenCV: Image processing and video handling
- NumPy: Numerical computations
- pyttsx3: Text-to-speech functionality

## Configuration Parameters
- `confidence_threshold`: 0.4 (default)
- `max_disappeared`: 300 frames
- `iou_threshold`: 0.3
- Target FPS: 30
- Notification cooldown: 5 seconds

## Usage Example
```python
tracker = VehicleTracker(
    model_path='yolov8s.pt',
    confidence_threshold=0.4,
    max_disappeared=300,
    iou_threshold=0.3
)
```

## RTSP Camera Configuration
```python
rtsp_url = 'rtsp://admin:password@camera_ip:554'
# Video settings
- Resolution: 1920x1080
- Codec: H264
- Buffer Size: 1 frame
```

## Performance Considerations
1. GPU Acceleration
   - Automatically utilizes CUDA if available
   - Falls back to MPS or CPU if necessary

2. Memory Management
   - Efficient tracking data structures
   - Automatic cleanup of stale tracking data
   - Optimized frame buffer management

3. Error Recovery
   - Automatic reconnection for lost streams
   - Graceful handling of frame processing errors
   - Comprehensive logging for debugging

## Development Guidelines
1. Code Modification Rules:
   - Don't remove or delete anything not related to the task
   - Don't create missing functions when editing code
   - Deep analyze before making any changes
   - Keep original code as backup reference

2. Best Practices:
   - Follow type hints for better code clarity
   - Maintain comprehensive logging
   - Handle errors gracefully
   - Optimize for performance where possible

---
© 2024 Khanfar System. All rights reserved.
