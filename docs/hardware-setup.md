---
sidebar_position: 1001
sidebar_label: "Hardware Setup"
title: "Hardware Setup & Configuration"
description: "Guide for setting up sensors and peripherals on Jetson Orin Nano"
---

# Hardware Setup & Configuration

This guide covers setting up optional sensors and peripherals for the Physical AI course.

## Prerequisites

- NVIDIA Jetson Orin Nano with ROS 2 Humble installed
- Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- Network connectivity (for downloading packages)

## Intel RealSense D435i Camera

The RealSense D435i is an industrial-grade depth camera with IMU (Inertial Measurement Unit).

**Capabilities**:
- Stereo depth sensing (up to 10m range)
- RGB color camera
- IR (infrared) sensors
- Built-in IMU (accelerometer + gyroscope)
- USB 3.0 interface

### Installation

#### 1. Install RealSense SDK

```bash
# Add Intel RealSense GPG key
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6270CDCC

# Add RealSense repository
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-xenial main" -u

# Install SDK
sudo apt update
sudo apt install librealsense2-dev librealsense2-utils
```

#### 2. Install ROS 2 RealSense Wrapper

```bash
sudo apt install ros-humble-realsense2-camera
```

#### 3. Verify Installation

```bash
# Check camera is detected
lsusb | grep Intel

# Expected output:
# Bus 001 Device 006: ID 8086:0b07 Intel Corp. RealSense D435

# Launch camera viewer
realsense-viewer

# In another terminal, check ROS topics
ros2 topic list | grep camera
```

### Basic Usage

#### Launch RealSense in ROS 2

```bash
# Terminal 1: Launch camera
ros2 launch realsense2_camera rs_launch.py

# Terminal 2: View available topics
ros2 topic list

# Expected topics:
# /camera/color/camera_info
# /camera/color/image_raw
# /camera/depth/camera_info
# /camera/depth/image_rect_raw
# /camera/imu
```

#### Access Camera Data in Python

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import cv2

class RealsenseListener(Node):
    def __init__(self):
        super().__init__('realsense_listener')

        # Subscribe to depth and color images
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/camera/imu',
            self.imu_callback,
            10
        )

        self.bridge = CvBridge()

    def depth_callback(self, msg):
        # Convert ROS Image to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(msg)
        self.get_logger().info(f'Depth image shape: {depth_image.shape}')

    def color_callback(self, msg):
        color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.get_logger().info(f'Color image shape: {color_image.shape}')

    def imu_callback(self, msg):
        acc = msg.linear_acceleration
        self.get_logger().info(f'Acceleration: x={acc.x:.2f}, y={acc.y:.2f}, z={acc.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    listener = RealsenseListener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Troubleshooting RealSense

| Issue | Cause | Solution |
|-------|-------|----------|
| Camera not detected | USB port issue | Try different USB 3.0 port (blue ports) |
| "Permission denied" | USB rules missing | Add udev rules and reconnect |
| Low frame rate | USB bandwidth | Use dedicated USB 3.0 hub |
| Depth holes | Distance or occlusion | Move closer (1-3m optimal range) |

#### Add udev Rules for RealSense

```bash
# Download rules
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

# Install rules
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/

# Reconnect camera (unplug and replug)
```

## ReSpeaker USB 4-Mic Array

The ReSpeaker is a USB microphone array with 4 microphones for directional audio.

**Capabilities**:
- 4 omnidirectional microphones
- Beamforming and direction-of-arrival estimation
- LED ring (controllable)
- USB audio interface
- Plug-and-play (UPnP)

### Installation

#### 1. Install Audio Drivers

```bash
# Clone Seeed repository
git clone https://github.com/respeaker/seeed-voicecard.git
cd seeed-voicecard

# Run installer (USB version)
sudo ./install.sh --with-respeaker-usb-4mic-array

# Reboot system
sudo reboot
```

#### 2. Verify Installation

```bash
# List audio devices
arecord -l

# Expected output shows ReSpeaker device

# Test recording
arecord -D plughw:CARD=seeed4miclinux,DEV=0 -f cd test.wav

# Test playback
aplay -D plughw:CARD=seeed4miclinux,DEV=0 test.wav
```

#### 3. Install ROS 2 Audio Packages

```bash
# Install audio capture and processing
sudo apt install ros-humble-audio-capture
sudo apt install python3-soundfile python3-sounddevice
```

### Basic Usage

#### Capture Audio in ROS 2

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
import numpy as np

class AudioListener(Node):
    def __init__(self):
        super().__init__('audio_listener')
        self.subscription = self.create_subscription(
            AudioData,
            'audio',
            self.audio_callback,
            10
        )

    def audio_callback(self, msg):
        # Convert bytes to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        self.get_logger().info(f'Received {len(audio_data)} audio samples')

def main(args=None):
    rclpy.init(args=args)
    listener = AudioListener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Control LED Ring

```python
#!/usr/bin/env python3
import usb.core
import usb.util

class ReSpeakerLED:
    def __init__(self):
        # Find ReSpeaker device
        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if not self.dev:
            raise Exception("ReSpeaker USB not found")

    def set_led(self, index, r, g, b):
        """Set LED at position to RGB color"""
        msg = [0xFF] + [0] * 7
        msg[index + 1] = (r << 4) | (g << 2) | b
        self.dev.ctrl_transfer(0xc0, 0, index, 0, msg, 6)

    def set_all_leds(self, r, g, b):
        """Set all LEDs to same color"""
        for i in range(4):
            self.set_led(i, r, g, b)

    def pulse(self):
        """Pulse effect"""
        for brightness in range(0, 8):
            self.set_all_leds(brightness, brightness, 0)  # Yellow pulse

if __name__ == '__main__':
    led = ReSpeakerLED()
    led.pulse()
```

### Troubleshooting ReSpeaker

| Issue | Cause | Solution |
|-------|-------|----------|
| Device not found | Driver not installed | Re-run installer and reboot |
| No audio input | Permissions issue | Add user to audio group: `sudo usermod -a -G audio $USER` |
| Low volume | Gain setting | Adjust with `alsamixer` |
| LED not responsive | USB issue | Try different USB port |

#### Add User to Audio Group

```bash
# Add current user to audio group
sudo usermod -a -G audio $USER

# Log out and log back in (or use: su - $USER)
# Verify
groups $USER  # Should show 'audio'
```

## USB Port Configuration

### Jetson Orin Nano USB Ports

```
                   [Back]
          [USB 3.0] [USB 3.0]
            Port 1   Port 2        <- Use for cameras/sensors

          [USB 2.0] [USB 2.0]
            Port 3   Port 4        <- Use for keyboards/mice

         [Power Input]
```

**Recommendations**:
- RealSense D435i → USB 3.0 ports (1 or 2)
- ReSpeaker → USB 2.0 or 3.0 (less bandwidth-sensitive)
- Keyboard/mouse → USB 2.0 ports (3 or 4)

### USB Hub Configuration

For multiple devices, use a powered USB 3.0 hub:

```bash
# List USB devices
lsusb -t

# Check port usage
lsusb -v

# Monitor USB bandwidth
sudo apt install usbutils
lsusb -H  # Show USB hierarchy
```

## Power Management

### Jetson Power Limits

```bash
# Check current power mode
sudo tegrastats

# Set to maximum performance (uses more power)
sudo nvpmodel -m 0  # MAXN mode (15W)

# Set to balanced mode (default, lower power)
sudo nvpmodel -m 1  # Balanced mode
```

### USB Power Consumption

```
Device              | Typical Current | Max Current
--------------------|-----------------|------------
RealSense D435i     | 600 mA          | 1.0 A
ReSpeaker USB 4-mic | 200 mA          | 300 mA
External SSD        | 500-800 mA      | 1.5 A
--------------------|-----------------|------------
Total recommended   | < 2.5 A         | (with hub)
```

**Note**: Jetson power supply = 5V/4A (20W typical). Use powered USB hub for multiple devices.

## ROS 2 Sensor Integration

### Launch Configuration for Multiple Sensors

Create `sensors.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # RealSense camera
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            output='screen',
            parameters=[{'camera_name': 'camera'}]
        ),

        # Audio capture
        Node(
            package='audio_common',
            executable='audio_capture_node',
            output='screen',
            parameters=[{
                'audio_topic': '/audio',
                'device_id': 'plughw:CARD=seeed4miclinux,DEV=0'
            }]
        ),
    ])
```

Launch with:
```bash
ros2 launch sensors_launch sensors.launch.py
```

## Performance Monitoring

```bash
# Monitor CPU/GPU usage
sudo tegrastats

# Monitor power consumption
sudo tegrastats --show Power

# Monitor thermal temperature
cat /sys/class/thermal/thermal_zone*/temp

# Monitor USB bandwidth
watch -n 1 'cat /proc/net/dev'
```

## Sensor Calibration

### RealSense Calibration

```bash
# Launch calibration tool
realsense-viewer

# In GUI:
# 1. Select Tools → Calibration
# 2. Follow on-screen instructions
# 3. Save calibration file
```

### Camera Intrinsics

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

class CameraInfoReader(Node):
    def __init__(self):
        super().__init__('camera_info_reader')
        self.subscription = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

    def camera_info_callback(self, msg):
        self.get_logger().info(
            f'Resolution: {msg.width}x{msg.height}\n'
            f'Focal length: {msg.K[0]:.2f}, {msg.K[4]:.2f}\n'
            f'Principal point: ({msg.K[2]:.2f}, {msg.K[5]:.2f})'
        )

def main(args=None):
    rclpy.init(args=args)
    reader = CameraInfoReader()
    rclpy.spin(reader)
    reader.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Conclusion

After setting up hardware:

1. **Test each sensor independently** with provided tools
2. **Integrate with ROS 2** using launch files
3. **Verify topics and data quality** with CLI tools
4. **Monitor performance** during operation

For sensor-specific lessons in Chapters 3-4, refer back to this guide for troubleshooting and configuration.

---

**Need help?** See [Getting Started](getting-started.md) or post in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
