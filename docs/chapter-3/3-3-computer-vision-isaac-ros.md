---
sidebar_position: 3
sidebar_label: "Lesson 3.3: Computer Vision Isaac ROS"
title: "Computer Vision with Isaac ROS: Real-Time Perception for Humanoid Robots"
description: "Implement hardware-accelerated object detection, semantic segmentation, and instance segmentation for real-time robot perception"
duration: 120
difficulty: Advanced
hardware: ["Jetson Orin Nano 8GB", "RealSense D435i", "ROS 2 Humble", "NVIDIA Isaac ROS"]
prerequisites: ["Lesson 3.2: VSLAM & Navigation"]
---

# Lesson 3.3: Computer Vision with Isaac ROS - Real-Time Perception

:::info Lesson Overview
**Duration**: 120 minutes | **Difficulty**: Advanced | **Hardware**: Jetson Orin Nano + RealSense D435i

**Prerequisites**: Lesson 3.2 (VSLAM & Navigation) - understanding of Isaac ROS ecosystem

**Learning Outcome**: Deploy hardware-accelerated computer vision models for real-time object detection, segmentation, and scene understanding on edge devices
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand computer vision tasks: detection, segmentation, instance segmentation
- Deploy pre-trained YOLO detection models on Jetson
- Implement semantic segmentation for scene understanding
- Use Isaac ROS perception nodes for hardware acceleration
- Integrate TensorRT for inference optimization
- Process camera streams with real-time perception
- Debug perception pipelines with RViz visualization
- Measure and optimize inference latency on edge hardware
- Build robot perception systems combining SLAM and vision
- Handle perception failures gracefully in robot applications

## Hardware Requirements

:::note Perception on Edge Devices
Computer vision inference is computationally expensive. Edge deployment requires careful optimization:
- **Real-time**: Object detection must run at 15-30 FPS
- **Low latency**: &lt;100ms from image capture to decision
- **Resource constrained**: Jetson Orin Nano has 8GB VRAM shared with OS and SLAM
:::

### Jetson Orin Nano Specifications for Vision

| Component | Specification | Impact |
|-----------|---------------|--------|
| **GPU Cores** | 1024 NVIDIA CUDA cores | Can run ~8 TFLOPS INT8 inference |
| **VRAM** | 8 GB LPDDR5 (shared) | SLAM: 2-3GB, Vision: 2-3GB, OS: ~1GB |
| **Memory Bandwidth** | 102 GB/s | Sufficient for HD image processing |
| **Power Budget** | 15W typical | Fanless operation possible; passive cooling adequate |
| **Encoder/Decoder** | Hardware H.264/H.265 | Can offload video encoding to NVENC |

### Recommended Camera

**Intel RealSense D435i** (Used throughout course)
- **Resolution**: 1280×720 @ 30 FPS (RGB + Depth)
- **Depth Accuracy**: ±2% @ 2m range
- **Field of View**: 87.3° × 58° (RGB), 85.2° × 58° (Depth)
- **IMU**: 6-axis (accelerometer + gyroscope)
- **USB**: 3.1 bus (5Gbps) - sufficient for 720p @ 30Hz

---

## Part 1: Computer Vision Fundamentals for Robotics

### Vision Tasks Overview

```
Raw Image Input
    ↓
┌───────────────────────────────────────────┐
│  COMPUTER VISION PROCESSING PIPELINE      │
└───────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Detection (Bounding Boxes)  │ Segmentation (Per-Pixel)      │
├─────────────────────────────┼───────────────────────────────┤
│ • YOLO                       │ • Semantic Segmentation       │
│ • Faster R-CNN               │   (class labels per pixel)    │
│ • SSD                        │ • Instance Segmentation       │
│ • EfficientDet               │   (individual objects)        │
│ • Detectron2                 │ • Panoptic Segmentation       │
└─────────────────────────────┴───────────────────────────────┘
    ↓
Robot Decision Making (Grasp, Navigate, Interact)
```

### Object Detection with YOLO

**YOLO (You Only Look Once)** is the dominant real-time detection framework for robotics:

- **Single-shot detection**: Predicts all objects in one forward pass
- **Real-time speed**: YOLOv8n runs at 40-60 FPS on Jetson Orin Nano
- **Various scales**: Nano (fastest), Small, Medium, Large (most accurate)
- **Training-friendly**: Can fine-tune on custom robot datasets

**YOLO Detection Output**
```
Detections:
├─ Class: "humanoid" (confidence: 0.94)
│  └─ Bounding box: [x1, y1, x2, y2] = [125, 87, 342, 510]
├─ Class: "obstacle" (confidence: 0.87)
│  └─ Bounding box: [x1, y1, x2, y2] = [450, 200, 580, 380]
└─ Class: "target_object" (confidence: 0.91)
   └─ Bounding box: [x1, y1, x2, y2] = [200, 150, 280, 320]
```

### Semantic Segmentation

**Pixel-level classification** - assigns a class label to every pixel in the image:

```
Input Image    Segmentation Map
┌─────────┐   ┌──────────────┐
│ ▓▓▓▓▓   │   │ 0 0 1 1 1 2  │
│ ▓░▓▓▓   │   │ 0 0 1 1 1 2  │
│ ░░░▓▓   │   │ 0 0 0 1 2 2  │
└─────────┘   └──────────────┘
             Classes: 0=Background, 1=Robot, 2=Obstacle
```

**Use cases for robots:**
- **Scene understanding**: What surfaces can the robot walk on?
- **Obstacle avoidance**: Which pixels are navigable?
- **Manipulation**: What parts of scene are graspable?

### Instance Segmentation

**Per-object pixel masks** - combines detection and segmentation:

```
Detection (Bounding Box) + Segmentation Mask = Instance Map
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ ┌──────┐     │   │ ▓▓▓▓▓▓       │   │ 1 1 1 1 1    │
│ │ obj1 │     │   │ ▓▓▓▓▓▓       │   │ 1 1 1 1 1    │
│ └──────┐     │   │ ▓▓▓▓▓▓       │   │ 1 1 1 1 1    │
│   ┌────┴──┐  │   │   ▓▓▓▓▓▓     │   │   2 2 2 2    │
│   │ obj2  │  │   │   ▓▓▓▓▓▓     │   │   2 2 2 2    │
│   └───────┘  │   │   ▓▓▓▓▓▓     │   │   2 2 2 2    │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Part 2: Isaac ROS Perception Architecture

### Isaac ROS Components for Vision

```
ROS 2 Network
    ↓
┌─────────────────────────────────────┐
│   Isaac ROS Perception Pipeline     │
├─────────────────────────────────────┤
│                                     │
│  Input: /camera/color/image_raw    │
│         /camera/depth/image_rect   │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ Image Preprocessing Node     │  │
│  │ • Resize, normalize          │  │
│  │ • Color space conversion     │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ Inference Engine (TensorRT)  │  │
│  │ • GPU acceleration            │  │
│  │ • INT8/FP16 quantization      │  │
│  │ • Batch processing            │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │ Post-processing Node         │  │
│  │ • NMS (Non-Max Suppression)  │  │
│  │ • Coordinate transformation  │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  Output: /perception/detections    │
│          /perception/segmentation  │
│                                     │
└─────────────────────────────────────┘
```

### TensorRT Optimization

NVIDIA TensorRT converts standard models (PyTorch, ONNX) to optimized engines:

| Metric | PyTorch Float32 | TensorRT INT8 | Speedup |
|--------|-----------------|---------------|---------|
| Latency (YOLOv8s) | 45 ms | 12 ms | 3.75× |
| Memory | 256 MB | 64 MB | 4× less |
| Throughput | 22 FPS | 83 FPS | 3.75× |
| Accuracy Loss | - | &lt;1% | Acceptable |

**Quantization Trade-offs:**
```
FP32 (Full Precision)
└─ Latency: 45ms, Accuracy: 100%

FP16 (Half Precision)
└─ Latency: 25ms, Accuracy: 99.8%

INT8 (Integer Quantization)
└─ Latency: 12ms, Accuracy: 99.0%
```

---

## Part 3: Setting Up Computer Vision on Jetson Orin Nano

### Step 1: Install NVIDIA Container Toolkit (Recommended)

Using Docker ensures consistent dependencies and easy model updates:

```bash
# Add Docker repository
curl https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Install Isaac ROS Perception Packages

```bash
# Create workspace
mkdir -p ~/isaac_ros_dev/src
cd ~/isaac_ros_dev

# Clone Isaac ROS repositories
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_perception.git src/isaac_ros_perception
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_proc.git src/isaac_ros_image_proc

# Install dependencies
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y

# Build workspace
colcon build --symlink-install
source install/setup.bash
```

### Step 3: Download Pre-trained Models

```bash
# Create models directory
mkdir -p ~/models/yolo ~/models/segmentation

# Download YOLOv8 Nano (ONNX format)
cd ~/models/yolo
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Convert to ONNX (requires ultralytics package)
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
# Output: yolov8n.onnx (6.3 MB)

# Convert ONNX to TensorRT Engine
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
# Outputs: yolov8n.engine (optimized for INT8, 1.5 MB)
```

### Step 4: Validate Perception Stack

```bash
# Terminal 1: RealSense camera
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:="640,480,30" \
  enable_depth:=true

# Terminal 2: Check topics
ros2 topic list
# Expected output:
# /camera/color/camera_info
# /camera/color/image_raw
# /camera/depth/image_rect_raw

# Terminal 3: Verify camera publishing
ros2 run image_view image_view \
  --ros-args -r image:=/camera/color/image_raw
```

---

## Part 4: Real-Time Object Detection

### YOLO Detection Node

```python
#!/usr/bin/env python3
"""
YOLOv8 Real-Time Object Detection for ROS 2
Runs inference on Jetson with TensorRT acceleration
"""

import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import time

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Initialize YOLO with TensorRT engine (Nano model for Jetson)
        self.model = YOLO('~/models/yolo/yolov8n.engine', task='detect')
        self.bridge = CvBridge()
        self.inference_times = []

        # Subscriptions and publications
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            '/perception/detections_debug',
            10
        )

        # Parameters (adjustable via ROS 2 CLI)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('enable_debug', True)

        self.get_logger().info('YOLO Detector initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            height, width = cv_image.shape[:2]

            # Run inference
            start_time = time.time()

            results = self.model.predict(
                source=cv_image,
                conf=self.get_parameter('confidence_threshold').value,
                iou=self.get_parameter('nms_threshold').value,
                imgsz=self.get_parameter('input_size').value,
                device=0,  # GPU device
                verbose=False
            )

            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)

            # Keep running average of last 30 inferences
            if len(self.inference_times) > 30:
                self.inference_times.pop(0)

            avg_latency = np.mean(self.inference_times)
            fps = 1000 / avg_latency if avg_latency > 0 else 0

            # Parse detections
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            result = results[0]  # Single image

            if result.boxes is not None:
                for box in result.boxes:
                    detection = Detection2D()
                    detection.header = msg.header

                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detection.bbox.center.x = float((x1 + x2) / 2)
                    detection.bbox.center.y = float((y1 + y2) / 2)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    # Class and confidence
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = result.names[class_id]

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_name = class_name
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)

                    detections_msg.detections.append(detection)

            # Publish detections
            self.detections_pub.publish(detections_msg)

            # Debug visualization (optional, CPU-intensive)
            if self.get_parameter('enable_debug').value:
                annotated_frame = result.plot()

                # Add performance metrics
                cv2.putText(
                    annotated_frame,
                    f'Latency: {avg_latency:.1f}ms | FPS: {fps:.1f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Publish debug image
                debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

            # Log statistics every 30 frames
            if len(self.inference_times) % 30 == 0:
                self.get_logger().info(
                    f'Detections: {len(detections_msg.detections)} | '
                    f'Latency: {avg_latency:.1f}ms | FPS: {fps:.1f}'
                )

        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    detector = YOLODetector()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Performance Metrics on Jetson Orin Nano:**
```
YOLOv8 Nano Model:
├─ Model size: 6.3 MB
├─ Parameters: 3.2M
├─ FLOPs: 8.7 G (inference at 640×480)
├─ TensorRT Engine: 1.5 MB
├─ Latency (INT8): 12-15 ms
├─ FPS (input 640×480 @ 30Hz): 60-80 FPS
└─ Memory usage: 180-220 MB
```

---

## Part 5: Semantic Segmentation

### Semantic Segmentation Configuration

```yaml
# semantic_segmentation_config.yaml
semantic_segmentation_node:
  ros__parameters:
    # Input topics
    image_topic: '/camera/color/image_raw'
    camera_info_topic: '/camera/color/camera_info'

    # Model configuration
    model_engine_file_path: '~/models/segmentation/deeplabv3_nano.engine'
    input_size: [512, 512]  # Input resolution for model

    # Inference settings
    confidence_threshold: 0.5
    device_id: 0  # GPU device

    # Number of classes in segmentation
    num_classes: 21  # PASCAL VOC dataset
    # Classes: background, person, bicycle, car, dog, cat, etc.

    # Output settings
    output_colormap: 'pascal_voc'  # Standard robotics colormap
    publish_mask: true
    publish_colored: true

    # Performance tuning
    enable_tensorrt: true
    quantization_type: 'INT8'  # FP32, FP16, INT8
    max_batch_size: 1
```

### Semantic Segmentation Node

```python
#!/usr/bin/env python3
"""
Semantic Segmentation for Scene Understanding
Classifies each pixel into semantic categories
"""

import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class SemanticSegmentation(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')

        self.bridge = CvBridge()

        # PASCAL VOC colormap (standard for robotics)
        self.colormap = self.create_pascal_voc_colormap()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.segment_callback,
            10
        )

        # Publications
        self.mask_pub = self.create_publisher(
            Image,
            '/perception/segmentation/mask',
            10
        )

        self.colored_pub = self.create_publisher(
            Image,
            '/perception/segmentation/colored',
            10
        )

        # Load pre-trained DeepLabV3 model
        # For custom robotics classes, fine-tune on robot dataset
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'deeplabv3_mobilenet_v3_large',
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info(f'Segmentation model loaded on {self.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')

    def segment_callback(self, msg):
        """Process image and generate segmentation mask"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            h, w = cv_image.shape[:2]

            # Preprocess: resize to 512×512 (DeepLabV3 standard input)
            input_tensor = self.preprocess_image(cv_image)

            # Inference
            start = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
            latency = (time.time() - start) * 1000

            # Post-process: get predicted class per pixel
            logits = output['out']  # Shape: [1, num_classes, 512, 512]
            predicted_mask = torch.argmax(logits, dim=1)[0]  # Shape: [512, 512]

            # Resize to original image dimensions
            predicted_mask_np = predicted_mask.cpu().numpy().astype(np.uint8)
            segmentation_map = cv2.resize(
                predicted_mask_np,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            # Create colored visualization
            colored_seg = self.colorize_segmentation(segmentation_map)

            # Publish raw mask
            mask_msg = self.bridge.cv2_to_imgmsg(segmentation_map, 'mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            # Publish colored version
            colored_msg = self.bridge.cv2_to_imgmsg(colored_seg, 'rgb8')
            colored_msg.header = msg.header
            self.colored_pub.publish(colored_msg)

            if segmentation_map.size % 100 == 0:
                self.get_logger().info(
                    f'Segmentation latency: {latency:.1f}ms'
                )

        except Exception as e:
            self.get_logger().error(f'Segmentation error: {str(e)}')

    def preprocess_image(self, cv_image):
        """Normalize and resize for model"""
        import torch
        from torchvision import transforms

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Standard ImageNet normalization
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transforms
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(rgb_image)
        tensor = preprocess(pil_image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def create_pascal_voc_colormap(self):
        """Generate PASCAL VOC colormap (21 classes)"""
        # Standard colormap used in computer vision
        colormap = np.zeros((21, 3), dtype=np.uint8)

        colors = [
            (0, 0, 0),        # background
            (128, 0, 0),      # person
            (0, 128, 0),      # bicycle
            (128, 128, 0),    # car
            (0, 0, 128),      # dog
            (128, 0, 128),    # cat
            (0, 128, 128),    # cow
            (128, 128, 128),  # horse
            (64, 0, 0),       # sheep
            (192, 0, 0),      # airplane
            (64, 128, 0),     # train
            (192, 128, 0),    # boat
            (64, 0, 128),     # bottle
            (192, 0, 128),    # chair
            (64, 128, 128),   # table
            (192, 128, 128),  # dog
            (0, 64, 0),       # cat
            (128, 64, 0),     # monitor
            (0, 192, 0),      # keyboard
            (128, 192, 0),    # mouse
            (0, 64, 128),     # sofa
        ]

        for i, color in enumerate(colors[:21]):
            colormap[i] = color

        return colormap

    def colorize_segmentation(self, mask):
        """Convert grayscale mask to colored using colormap"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in np.unique(mask):
            colored[mask == class_id] = self.colormap[class_id]
        return colored

def main(args=None):
    rclpy.init(args=args)
    seg_node = SemanticSegmentation()
    rclpy.spin(seg_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Part 6: Instance Segmentation (Mask R-CNN)

### Instance Segmentation Node

```python
#!/usr/bin/env python3
"""
Instance Segmentation: Detection + Segmentation Masks
Identifies individual objects with pixel-precise masks
"""

import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import time

class InstanceSegmentation(Node):
    def __init__(self):
        super().__init__('instance_segmentation')

        self.bridge = CvBridge()

        # Load Mask R-CNN (pre-trained on COCO)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # COCO class names (80 classes)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant',
            # ... (60 more classes)
        ]

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.instance_segment_callback,
            10
        )

        # Publications
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/perception/instance_detections',
            10
        )

        self.mask_pub = self.create_publisher(
            Image,
            '/perception/instance_masks',
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/perception/instance_visualization',
            10
        )

        self.get_logger().info('Instance Segmentation node initialized')

    def instance_segment_callback(self, msg):
        """Process image and generate instance masks"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            h, w = cv_image.shape[:2]

            # Preprocess: convert to tensor
            image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Inference
            start = time.time()
            with torch.no_grad():
                predictions = self.model(image_tensor)
            latency = (time.time() - start) * 1000

            pred = predictions[0]
            masks = pred['masks']  # Shape: [num_objects, 1, H, W]
            boxes = pred['boxes']
            labels = pred['labels']
            scores = pred['scores']

            # Filter by confidence threshold
            confidence_threshold = 0.5
            keep_idx = torch.where(scores > confidence_threshold)[0]

            masks = masks[keep_idx]
            boxes = boxes[keep_idx]
            labels = labels[keep_idx]
            scores = scores[keep_idx]

            # Create instance ID map (different color per object)
            instance_map = np.zeros((h, w), dtype=np.uint8)
            colored_instances = np.zeros((h, w, 3), dtype=np.uint8)

            # Generate detections message
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            # Colors for visualization (distinct per instance)
            colors = [
                (0, 255, 0),      # Green
                (0, 0, 255),      # Red
                (255, 0, 0),      # Blue
                (0, 255, 255),    # Cyan
                (255, 0, 255),    # Magenta
                (255, 255, 0),    # Yellow
            ]

            for inst_id, (mask, box, label, score) in enumerate(
                zip(masks, boxes, labels, scores)
            ):
                # Get mask binary
                mask_np = mask.squeeze(0).cpu().numpy() > 0.5
                instance_map[mask_np] = inst_id + 1

                # Assign color
                color = colors[inst_id % len(colors)]
                colored_instances[mask_np] = color

                # Create detection message
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.x = float((box[0] + box[2]) / 2)
                detection.bbox.center.y = float((box[1] + box[3]) / 2)
                detection.bbox.size_x = float(box[2] - box[0])
                detection.bbox.size_y = float(box[3] - box[1])

                # Class info
                class_name = self.coco_classes[label.item() - 1] if label < len(self.coco_classes) else 'unknown'
                from vision_msgs.msg import ObjectHypothesisWithPose
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_name = class_name
                hyp.hypothesis.score = float(score.item())
                detection.results.append(hyp)

                detections_msg.detections.append(detection)

            # Publish results
            self.detections_pub.publish(detections_msg)

            mask_msg = self.bridge.cv2_to_imgmsg(instance_map, 'mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            # Publish visualization
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            viz = cv2.addWeighted(cv_image_bgr, 0.7,
                                 cv2.cvtColor(colored_instances, cv2.COLOR_RGB2BGR), 0.3, 0)

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'{self.coco_classes[label.item()-1]}: {score.item():.2f}'
                cv2.putText(viz, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            viz_msg = self.bridge.cv2_to_imgmsg(viz, 'bgr8')
            viz_msg.header = msg.header
            self.visualization_pub.publish(viz_msg)

            self.get_logger().info(
                f'Instances: {len(detections_msg.detections)} | '
                f'Latency: {latency:.1f}ms'
            )

        except Exception as e:
            self.get_logger().error(f'Instance segmentation error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    inst_seg = InstanceSegmentation()
    rclpy.spin(inst_seg)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Part 7: Combining Perception with SLAM and Navigation

### Full Perception Pipeline Launch File

```python
# perception_pipeline.launch.py
"""
Complete perception stack: SLAM + Vision + Navigation
Runs all components with proper topic routing
"""

from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Camera launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'rgb_camera.color_profile': '640,480,30',
            'depth_module.depth_profile': '640,480,30',
            'enable_depth': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'align_depth': 'true',
        }.items()
    )

    # SLAM node
    slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        parameters=[{
            'camera_info_topic': '/camera/color/camera_info',
            'rgb_image_topic': '/camera/color/image_raw',
            'depth_image_topic': '/camera/depth/image_rect_raw',
            'enable_loop_closure': True,
            'enable_depth_regularization': True,
            'map_type': 'occupancy_grid',
        }],
        remappings=[
            ('camera_info_in', '/camera/color/camera_info'),
            ('image_rgb', '/camera/color/image_raw'),
        ]
    )

    # Object detection
    yolo_node = Node(
        package='perception_nodes',  # Your custom package
        executable='yolo_detector.py',
        parameters=[{
            'confidence_threshold': 0.5,
            'input_size': 640,
            'enable_debug': True,
        }]
    )

    # Semantic segmentation
    segmentation_node = Node(
        package='perception_nodes',
        executable='semantic_segmentation.py',
        parameters=[{
            'output_colormap': 'pascal_voc',
            'publish_mask': True,
        }]
    )

    # Instance segmentation
    instance_node = Node(
        package='perception_nodes',
        executable='instance_segmentation.py'
    )

    # Navigation stack
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/bringup_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': os.path.join(
                get_package_share_directory('my_robot'),
                'config', 'nav2_params.yaml'
            ),
        }.items()
    )

    # RViz visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('my_robot'),
            'rviz', 'perception.rviz'
        )]
    )

    return LaunchDescription([
        realsense_launch,
        slam_node,
        yolo_node,
        segmentation_node,
        instance_node,
        nav2_launch,
        rviz_node,
    ])
```

### Python Integration: Perception-Based Robot Control

```python
#!/usr/bin/env python3
"""
High-level robot control using perception outputs
Example: Navigate to detected objects and grasp them
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, Twist
import math

class PerceptionControlSystem(Node):
    def __init__(self):
        super().__init__('perception_control')

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscriptions to perception outputs
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/perception/detections',
            self.detections_callback,
            10
        )

        # Velocity publisher for reactive control
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.target_object = None
        self.robot_state = 'idle'  # idle, navigating, grasping

        self.get_logger().info('Perception control system initialized')

    def detections_callback(self, msg):
        """React to detected objects"""

        if len(msg.detections) == 0:
            self.get_logger().debug('No objects detected')
            return

        # Find "target_object" class
        for detection in msg.detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_name
                confidence = detection.results[0].hypothesis.score

                if class_name == 'target_object' and confidence > 0.7:
                    self.target_object = detection
                    self.approach_object(detection)
                    break

    def approach_object(self, detection):
        """Navigate to detected object"""

        # Get object position in image
        center_x = detection.bbox.center.x
        center_y = detection.bbox.center.y

        # Image-space error
        image_width = 640
        error_x = center_x - image_width / 2

        if abs(error_x) > 50:
            # Object not centered - rotate to face it
            twist = Twist()
            twist.angular.z = float(0.5 * error_x / image_width)
            self.vel_pub.publish(twist)
        else:
            # Object centered - move towards it
            twist = Twist()
            twist.linear.x = 0.3  # Move forward at 30 cm/s
            self.vel_pub.publish(twist)

    def send_navigation_goal(self, goal_x, goal_y):
        """Send autonomous navigation goal via Nav2"""

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(0)  # 0 radians
        goal.pose.pose.orientation.w = math.cos(0)

        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    controller = PerceptionControlSystem()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Hands-On Exercise: Real-Time Object Detection and Segmentation

### Exercise 1: Deploy YOLO and Measure Performance

```bash
# Terminal 1: Start RealSense camera
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:="640,480,30" \
  enable_depth:=true

# Terminal 2: Run YOLO detector
ros2 run perception_nodes yolo_detector.py

# Terminal 3: Monitor performance
ros2 topic hz /perception/detections
# Expected: 20-25 Hz (60-80 FPS inference rate)

# Terminal 4: View detections
ros2 run rqt_image_view rqt_image_view \
  --topics /perception/detections_debug

# Terminal 5: Profile resource usage
nvidia-smi -l 1  # Monitor GPU usage
# Expected: 2-3 GB VRAM, 40-50% GPU utilization

# Challenge: Modify yolo_detector.py to use YOLOv8s (medium model)
# Compare latency difference
```

### Exercise 2: Create Custom YOLO Training Dataset

```python
#!/usr/bin/env python3
"""
Collect training data from robot camera for custom object detection
"""

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.bridge = CvBridge()
        self.dataset_dir = f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.frame_count = 0

        self.create_subscription(Image, '/camera/color/image_raw', self.save_frame, 10)

    def save_frame(self, msg):
        """Save frames for manual annotation"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Save every 10th frame
        if self.frame_count % 10 == 0:
            filename = os.path.join(self.dataset_dir, f'frame_{self.frame_count:06d}.jpg')
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f'Saved {filename}')

        self.frame_count += 1

# After collecting 500 frames, use label-img for annotation:
# pip install labelImg
# labelImg <dataset_dir> --yolo

# Then fine-tune YOLO:
# yolo detect train data=custom_data.yaml epochs=100 device=0
```

### Exercise 3: Debug Common Perception Errors

| Error | Cause | Fix |
|-------|-------|-----|
| **Very low FPS (&lt;5)** | Model too large for Jetson | Use YOLOv8n instead of YOLOv8m |
| **Out of memory crash** | Models competing for VRAM | Reduce batch size, disable SLAM |
| **High latency (>100ms)** | No TensorRT optimization | Convert ONNX to TensorRT engine |
| **Missed detections** | Low confidence threshold | Reduce confidence_threshold parameter |
| **False positives** | Model overtrained on wrong data | Fine-tune on robot's target objects |
| **Topic lag** | ROS 2 network bottleneck | Use compressed image transport |

---

## Key Takeaways

✅ **Computer Vision Tasks**
- Detection identifies objects (bounding boxes)
- Segmentation classifies pixels (scene understanding)
- Instance segmentation combines both (precise object masks)

✅ **Edge Inference Optimization**
- TensorRT reduces inference latency by 3-4×
- INT8 quantization sacrifices &lt;1% accuracy for 3-4× speedup
- Model selection matters: YOLOv8n for Jetson, YOLOv8l for powerful GPUs

✅ **ROS 2 Integration Patterns**
- Perception nodes subscribe to camera topics
- Publish structured detection/segmentation messages
- Navigation and manipulation subscribe to perception outputs
- RViz visualizes perception pipeline for debugging

✅ **Real-Time Constraints**
- Jetson Orin Nano can run YOLO at 60+ FPS
- Combined SLAM + Vision + Navigation requires careful resource allocation
- Memory management is critical (8GB VRAM shared between OS, SLAM, vision)

✅ **Debugging Perception**
- Always profile latency (nvidia-smi, ros2 node info)
- Use RViz to visualize intermediate outputs
- Test inference speed offline before deploying to robot
- Keep separate confidence thresholds for different use cases

---

## Further Reading

### Official Documentation
- [NVIDIA Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [NVIDIA Isaac Sim](https://docs.nvidia.com/isaac-sim/)
- [ROS 2 Vision Messages](https://github.com/ros-perception/vision_msgs)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)

### Research Papers
- YOLO: [You Only Look Once - Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- Mask R-CNN: [Instance Segmentation by Detecting Objects and Segmenting Masks](https://arxiv.org/abs/1703.06870)
- DeepLabV3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

### Open Source Tools
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Detectron2 (Facebook)](https://github.com/facebookresearch/detectron2)
- [OpenCV for Computer Vision](https://github.com/opencv/opencv)

---

## Next Lesson

**Lesson 4.1: Vision-Language-Action (VLA) Models**

Now that your humanoid robot can perceive its environment (SLAM + Object Detection), we'll add language understanding. You'll learn to:
- Integrate large language models (GPT-4, Gemini, open-source alternatives)
- Parse natural language commands into robot actions
- Build end-to-end "visual question answering" for robots
- Deploy LLMs efficiently on edge devices with quantization

**Example Workflow:**
```
User: "Pick up the red cube and place it on the table"
    ↓
[Speech-to-text] (not covered in this course)
    ↓
"Pick up the red cube and place it on the table"
    ↓
[Vision-Language Model]
    ├─ Detect: Red cube at (234, 456, 567, 789)
    ├─ Parse: Action sequence [grasp, place]
    ├─ Plan: Approach cube → Open gripper → Move to table → Close gripper
    ↓
[Robot Execution]
    ├─ Navigate to cube location (SLAM + Nav2)
    ├─ Grasp cube (Manipulation controller)
    ├─ Navigate to table (SLAM + Nav2)
    ├─ Place cube (Gripper control)
    ↓
✅ Task Complete
```
