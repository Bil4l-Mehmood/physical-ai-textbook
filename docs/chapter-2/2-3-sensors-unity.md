---
sidebar_position: 3
sidebar_label: "Lesson 2.3: Sensors & Unity"
title: "Sensor Simulation and High-Fidelity Visualization with Unity"
description: "Simulate realistic sensors in Gazebo and render robots with photorealistic graphics in Unity"
duration: 90
difficulty: Intermediate
hardware: ["Ubuntu 22.04 LTS", "ROS 2 Humble", "Gazebo Harmonic", "Unity 2022+ (optional)"]
prerequisites: ["Lesson 2.1: Gazebo Fundamentals", "Lesson 2.2: URDF/SDF & Physics"]
---

# Lesson 2.3: Sensor Simulation & Unity Integration

:::info Lesson Overview
**Duration**: 90 minutes | **Difficulty**: Intermediate | **Hardware**: Ubuntu 22.04 + ROS 2 Humble + Gazebo + Unity (optional)

**Prerequisites**: Lessons 2.1 and 2.2

**Learning Outcome**: Simulate realistic sensors in Gazebo and visualize robots with high-fidelity graphics in Unity
:::

## Learning Objectives

By the end of this lesson, you will be able to:

- Understand sensor types and their simulation models
- Configure camera sensors with realistic parameters
- Simulate LiDAR point clouds for navigation
- Simulate IMU (accelerometer, gyroscope) data
- Publish sensor data to ROS 2 topics
- Integrate Gazebo with Unity for photorealistic rendering
- Use ROS-Unity bridge for real-time data streaming
- Optimize sensor simulation performance

## Part 1: Gazebo Sensor Simulation

:::note Why Simulate Sensors?
Real sensors are expensive and slow to debug. Simulation allows:
- **Cost reduction**: No hardware damage during development
- **Iteration speed**: Test algorithms at 100x faster than real-time
- **Repeatability**: Identical conditions for each test
- **Realism**: Add noise/lag to match real hardware
- **Scaling**: Simulate multiple robots simultaneously
:::

### Camera Sensor

```xml
<!-- Add to SDF model -->
<link name="camera_link">
  <inertial>
    <mass>0.1</mass>
    <inertia>
      <ixx>0.001</ixx>
      <iyy>0.001</iyy>
      <izz>0.001</izz>
    </inertia>
  </inertial>

  <visual name="visual">
    <geometry>
      <box>
        <size>0.05 0.05 0.05</size>
      </box>
    </geometry>
  </visual>

  <collision name="collision">
    <geometry>
      <box>
        <size>0.05 0.05 0.05</size>
      </box>
    </geometry>
  </collision>

  <!-- Camera Sensor Plugin -->
  <sensor name="camera" type="camera">
    <pose>0 0 0 0 0 0</pose>
    <update_rate>30</update_rate>  <!-- 30 Hz -->
    <camera>
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.01</near>  <!-- Minimum distance in meters -->
        <far>300</far>     <!-- Maximum distance in meters -->
      </clip>
      <!-- Lens distortion (optional) -->
      <lens>
        <type>quadratic</type>
        <scale_to_hfov>true</scale_to_hfov>
        <cutoff_angle>1.5707</cutoff_angle>
        <intrinsics>
          <fx>554.254</fx>   <!-- Focal length in pixels -->
          <fy>554.254</fy>
          <cx>320.0</cx>     <!-- Principal point -->
          <cy>240.0</cy>
          <s>0</s>           <!-- Skew -->
        </intrinsics>
      </lens>
    </camera>

    <!-- ROS 2 Plugin for publishing camera data -->
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <!-- Camera topics will be published under this namespace -->
        <namespace>camera</namespace>
        <remapping>image_raw:=image</remapping>
        <remapping>camera_info:=info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>  <!-- For stereo -->
      <distortion_k1>0.0</distortion_k1>   <!-- Lens distortion -->
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</link>

<!-- Joint to attach camera to robot -->
<joint name="camera_joint" type="fixed">
  <parent>base_link</parent>
  <child>camera_link</child>
  <pose>0.05 0 0.1 0 0 0</pose>  <!-- Mount on front-top of robot -->
</joint>
```

### LiDAR Sensor (2D Laser Scan)

```xml
<!-- 2D LiDAR for range measurements -->
<link name="lidar_link">
  <inertial>
    <mass>0.05</mass>
    <inertia>
      <ixx>0.0001</ixx>
      <iyy>0.0001</iyy>
      <izz>0.0001</izz>
    </inertia>
  </inertial>

  <visual name="visual">
    <geometry>
      <cylinder>
        <radius>0.03</radius>
        <length>0.05</length>
      </cylinder>
    </geometry>
  </visual>

  <collision name="collision">
    <geometry>
      <cylinder>
        <radius>0.03</radius>
        <length>0.05</length>
      </cylinder>
    </geometry>
  </collision>

  <!-- Ray sensor (LIDAR) -->
  <sensor name="lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <update_rate>40</update_rate>  <!-- 40 Hz -->
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>      <!-- 360 rays -->
          <resolution>1</resolution>  <!-- 1 degree per ray -->
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>1</samples>        <!-- 2D LIDAR only -->
        </vertical>
      </scan>
      <range>
        <min>0.2</min>              <!-- 20 cm minimum range -->
        <max>30</max>               <!-- 30 m maximum range -->
        <resolution>0.01</resolution>
      </range>
      <!-- Add noise to match real hardware -->
      <noise>
        <type>gaussian</type>
        <mean>0</mean>
        <stddev>0.01</stddev>  <!-- 1 cm std dev -->
      </noise>
    </ray>

    <!-- ROS 2 Plugin for LaserScan topic -->
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</link>

<joint name="lidar_joint" type="fixed">
  <parent>base_link</parent>
  <child>lidar_link</child>
  <pose>0 0 0.15 0 0 0</pose>
</joint>
```

### IMU Sensor (Accelerometer + Gyroscope)

```xml
<!-- IMU for motion sensing -->
<link name="imu_link">
  <inertial>
    <mass>0.01</mass>
    <inertia>
      <ixx>0.00001</ixx>
      <iyy>0.00001</iyy>
      <izz>0.00001</izz>
    </inertia>
  </inertial>

  <!-- IMU Sensor -->
  <sensor name="imu" type="imu">
    <pose>0 0 0 0 0 0</pose>
    <update_rate>200</update_rate>  <!-- 200 Hz -->
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.002</stddev>  <!-- 0.2 deg/s noise -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.002</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.002</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.017</stddev>  <!-- 0.017 m/s^2 noise -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>

    <!-- ROS 2 Plugin for IMU data -->
    <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
      <frame_name>imu_link</frame_name>
    </plugin>
  </sensor>
</link>

<joint name="imu_joint" type="fixed">
  <parent>base_link</parent>
  <child>imu_link</child>
</joint>
```

---

## Part 2: Reading Sensor Data in ROS 2

### Python Script to Process Camera Images

```python
#!/usr/bin/env python3
"""
Subscribe to camera images from Gazebo simulation
and process them with OpenCV
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraProcessor(Node):
    """Process camera images from Gazebo"""

    def __init__(self):
        super().__init__('camera_processor')

        # Subscribe to camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        # OpenCV bridge for ROS Image ↔ OpenCV Mat conversion
        self.bridge = CvBridge()

        # For video output
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None

        self.get_logger().info('Camera processor initialized')

    def image_callback(self, msg):
        """Process incoming camera frame"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Image processing example: edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # Display processed image
            cv2.imshow('Camera Feed', cv_image)
            cv2.imshow('Edges', edges)
            cv2.waitKey(1)

            # Log statistics
            self.get_logger().debug(
                f'Image: {cv_image.shape}, Mean color: {cv_image.mean():.1f}'
            )

        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    processor = CameraProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Python Script to Process LiDAR Data

```python
#!/usr/bin/env python3
"""
Subscribe to LiDAR laser scans and detect obstacles
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LiDARProcessor(Node):
    """Process LiDAR scans from Gazebo"""

    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to laser scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.scan_callback,
            10
        )

        self.obstacle_threshold = 1.0  # Detect obstacles within 1 meter
        self.get_logger().info('LiDAR processor initialized')

    def scan_callback(self, msg):
        """Process incoming laser scan"""
        try:
            # msg.ranges is array of distances (float)
            ranges = np.array(msg.ranges)

            # Filter out invalid readings
            valid_ranges = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]

            # Detect obstacles
            obstacles = valid_ranges[valid_ranges < self.obstacle_threshold]

            if len(obstacles) > 0:
                min_distance = np.min(obstacles)
                self.get_logger().warn(
                    f'⚠️  Obstacle detected! Minimum distance: {min_distance:.2f}m'
                )
                # Could trigger emergency stop or obstacle avoidance here

                # Find direction of closest obstacle
                closest_idx = np.argmin(ranges)
                angle = msg.angle_min + closest_idx * msg.angle_increment
                self.get_logger().info(f'Obstacle at angle: {angle:.2f} rad')

            else:
                self.get_logger().debug('No obstacles in range')

            # Statistics
            self.get_logger().debug(
                f'LiDAR: {len(valid_ranges)} valid readings, '
                f'min={np.min(valid_ranges):.2f}m, '
                f'max={np.max(valid_ranges):.2f}m'
            )

        except Exception as e:
            self.get_logger().error(f'Scan processing failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    processor = LiDARProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Python Script to Process IMU Data

```python
#!/usr/bin/env python3
"""
Subscribe to IMU data and detect motion/orientation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import math


class IMUProcessor(Node):
    """Process IMU data from Gazebo"""

    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.get_logger().info('IMU processor initialized')

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        # Linear acceleration (m/s^2)
        accel = msg.linear_acceleration
        accel_magnitude = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

        # Angular velocity (rad/s)
        angular = msg.angular_velocity
        angular_magnitude = math.sqrt(angular.x**2 + angular.y**2 + angular.z**2)

        # Orientation (quaternion)
        orientation = msg.orientation

        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        self.get_logger().info(
            f'Acceleration: {accel_magnitude:.2f} m/s^2 | '
            f'Rotation: {angular_magnitude:.2f} rad/s | '
            f'Orientation: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}'
        )

    @staticmethod
    def quaternion_to_euler(x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (rotation around X axis)
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sin_roll, cos_roll)

        # Pitch (rotation around Y axis)
        sin_pitch = 2.0 * (w * y - z * x)
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = math.asin(sin_pitch)

        # Yaw (rotation around Z axis)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(sin_yaw, cos_yaw)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Part 3: Unity Integration for High-Fidelity Visualization

:::tip Why Use Both Gazebo and Unity?
- **Gazebo**: Fast physics simulation, perfect for algorithm development
- **Unity**: Photorealistic rendering, human-in-the-loop testing, demos
- **Together**: Realistic simulation + beautiful visualization
:::

### Setting Up ROS-Unity Bridge

```bash
# Install ROS-Unity bridge in Unity project
# Download from GitHub: https://github.com/Unity-Technologies/ROS-TCP-Connector

# In Unity:
# 1. Window → TextMesh Pro → Import TMP Essential Resources
# 2. Assets → Import Package → Custom Package (ROS-TCP-Connector.unitypackage)
# 3. Add ROSConnection GameObject to scene
# 4. Configure ROS Master URI
```

### Unity Script for Robot Control

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using UnityEngine;

public class RobotController : MonoBehaviour
{
    private ROSConnection ros;

    // Robot transform
    private Transform robotBase;

    // Sensor displays
    private Texture2D cameraImage;
    private RawImage imageDisplay;

    void Start()
    {
        // Get ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Register subscribers for sensor topics
        ros.Subscribe<Image>("/camera/image", CameraImageCallback);
        ros.Subscribe<LaserScan>("/lidar/scan", LiDARCallback);
        ros.Subscribe<Imu>("/imu/data", IMUCallback);

        // Find robot in scene
        robotBase = transform.Find("robot_base");
        imageDisplay = GetComponent<RawImage>();
    }

    void Update()
    {
        // Example: Move robot with keyboard input
        float moveX = Input.GetAxis("Horizontal");
        float moveZ = Input.GetAxis("Vertical");

        robotBase.Translate(new Vector3(moveX, 0, moveZ) * 5f * Time.deltaTime);
    }

    void CameraImageCallback(Image image)
    {
        // Convert ROS Image to Unity Texture2D
        // Process and display in UI
        Debug.Log($"Camera received: {image.width}x{image.height}");
    }

    void LiDARCallback(LaserScan scan)
    {
        Debug.Log($"LiDAR received: {scan.ranges.Length} rays");

        // Visualize point cloud
        for (int i = 0; i < scan.ranges.Length; i++)
        {
            float distance = (float)scan.ranges[i];
            float angle = (float)(scan.angle_min + i * scan.angle_increment);

            // Convert polar to Cartesian
            float x = Mathf.Cos(angle) * distance;
            float z = Mathf.Sin(angle) * distance;

            // Draw point in world
            Debug.DrawLine(robotBase.position, robotBase.position + new Vector3(x, 0, z), Color.green);
        }
    }

    void IMUCallback(Imu imuData)
    {
        Debug.Log($"IMU acceleration: {imuData.linear_acceleration.x}");

        // Could update robot orientation based on IMU
        // robotBase.rotation = ...
    }
}
```

---

## Hands-On Exercise

**Task**: Create a robot with camera, LiDAR, and IMU sensors and process all sensor data.

### Step 1: Create Sensor-Equipped Robot SDF

```bash
# Create complete world with sensors
mkdir -p ~/gazebo_ws/src/worlds
cat > ~/gazebo_ws/src/worlds/sensors_world.sdf << 'EOF'
# Copy complete SDF with camera, LiDAR, IMU from Parts 1-2
EOF
```

### Step 2: Create ROS 2 Sensor Processing Nodes

```bash
mkdir -p ~/gazebo_ws/src/sensor_processing
# Copy camera_processor.py, lidar_processor.py, imu_processor.py
chmod +x ~/gazebo_ws/src/sensor_processing/*.py
```

### Step 3: Launch All Sensors

```bash
# Terminal 1: Gazebo with sensors
ros2 launch gazebo_ros gazebo.launch.py world:=$HOME/gazebo_ws/src/worlds/sensors_world.sdf

# Terminal 2: Camera processor
python3 ~/gazebo_ws/src/sensor_processing/camera_processor.py

# Terminal 3: LiDAR processor
python3 ~/gazebo_ws/src/sensor_processing/lidar_processor.py

# Terminal 4: IMU processor
python3 ~/gazebo_ws/src/sensor_processing/imu_processor.py
```

### Step 4: Verify Sensor Data

```bash
# Monitor all active topics
ros2 topic list

# View camera images
ros2 topic echo /camera/image_raw | head -20

# View LiDAR ranges
ros2 topic echo /lidar/scan | head -20

# View IMU data
ros2 topic echo /imu/data | head -20
```

### Exercises

1. **Obstacle Detection**: Detect when LiDAR range < 1 meter
2. **Motion Detection**: Log when IMU acceleration > 2 m/s²
3. **Object Detection**: Apply OpenCV blob detection to camera images
4. **Fall Detection**: Use IMU to detect if robot falls over
5. **Unity Integration**: Create a simple Unity scene displaying sensor data

---

## Common Sensor Simulation Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No camera images | Plugin not loaded | Check `libgazebo_ros_camera.so` path |
| LiDAR no data | Ray sensor misconfigured | Verify `<scan>` horizontal samples |
| IMU values zero | World lacks gravity | Ensure gravity in physics `<gravity>0 0 -9.81</gravity>` |
| Noise too high | stddev too large | For cameras use 0.01, for IMU use 0.002 |
| Sensor slow | Update rate too high | Start with 30 Hz for camera, 40 for LiDAR |
| ROS topic empty | Plugin namespace wrong | Check topic names match plugin config |

---

## Key Takeaways

✅ **Sensor Types**: Camera (RGB), LiDAR (range), IMU (motion)

✅ **Realistic Simulation**: Add noise, lag, and field-of-view constraints

✅ **ROS 2 Publishing**: Gazebo plugins automatically publish to ROS topics

✅ **Data Processing**: Use Python + OpenCV for real-time sensor processing

✅ **Hybrid Simulation**: Gazebo for physics + Unity for photorealism

✅ **Sensor Fusion**: Combine camera + LiDAR + IMU for robust perception

✅ **Optimization**: Balance accuracy vs. computational cost

---

## Further Reading

- 📖 [Gazebo Sensor Plugins](https://gazebosim.org/api/gazebo_plugins/index.html)
- 📖 [ROS-Unity Bridge Documentation](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- 📖 [Sensor Message Formats](https://docs.ros.org/en/humble/Concepts/Intermediate/About-ROS-2-Messages.html#sensor-messages)
- 📖 [OpenCV with ROS 2](https://docs.ros.org/en/humble/Tutorials/Computer-Vision/OpenCV-Python-Example.html)

---

**Chapter 2 Complete!** Next: Chapter 3 - The AI-Robot Brain (NVIDIA Isaac)

**Questions?** See [FAQ](../faq.md) or [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
