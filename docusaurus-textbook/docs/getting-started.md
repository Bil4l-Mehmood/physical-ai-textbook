---
sidebar_position: 1000
sidebar_label: "Getting Started"
title: "Getting Started with the Physical AI Course"
description: "Setup and prerequisites for the Physical AI & Humanoid Robotics textbook course"
---

# Getting Started with Physical AI

This guide walks you through setting up your development environment to complete the Physical AI & Humanoid Robotics course.

## System Requirements

### Hardware
- **Primary**: NVIDIA Jetson Orin Nano (8GB or higher recommended)
- **Alternative**: Any x86_64 Linux system for development/testing
- **Recommended Add-ons**:
  - Intel RealSense D435i (camera)
  - ReSpeaker USB Mic Array (microphone)

### Software
- **Operating System**: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- **ROS 2 Humble** (LTS release)
- **Python 3.10+**
- **Node.js 18.x+** (for Docusaurus, optional for viewing locally)

## Step 1: Install ROS 2 Humble on Jetson Orin Nano

### Option A: Official NVIDIA Method (Recommended)

NVIDIA provides a pre-configured SD card image with ROS 2 Humble installed.

1. **Download SD Card Image**:
   - Visit [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/downloads)
   - Select "Jetson Orin Nano Developer Kit"
   - Download "JetPack 5.1.2" or later

2. **Flash to microSD Card**:
   ```bash
   # On your host machine (not on Jetson)
   # Download Balena Etcher: https://www.balena.io/etcher/
   # Flash the .img file to your microSD card
   ```

3. **Boot Jetson and Configure**:
   - Insert microSD into Jetson Orin Nano
   - Connect to monitor, keyboard, mouse, and power
   - Complete initial setup wizard
   - Update system: `sudo apt update && sudo apt upgrade -y`

### Option B: Manual Installation

If you need to install ROS 2 on an existing system:

```bash
# Add ROS 2 repository
sudo apt update
sudo apt install curl gnupg lsb-release ubuntu-keyring
curl -sSL https://repo.ros2.org/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop-full

# Add ROS 2 setup to bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc

# Verify installation
ros2 --version
```

## Step 2: Set Up Development Environment

### Create Workspace

```bash
# Create ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Install Build Tools

```bash
# Install colcon (ROS 2 build system)
sudo apt install python3-colcon-common-extensions

# Install additional development tools
sudo apt install build-essential cmake git python3-dev python3-pip

# Install URDF tools
sudo apt install ros-humble-urdf-parser ros-humble-xacro

# Install visualization tools (optional)
sudo apt install ros-humble-rviz2 ros-humble-rqt
```

### Configure Shell

```bash
# Add ROS 2 setup to shell initialization
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

## Step 3: Verify Installation

Run these commands to verify everything is installed correctly:

```bash
# Check ROS 2 version
ros2 --version

# Check Python version (should be 3.10+)
python3 --version

# Test ROS 2 communication
# Terminal 1
ros2 run demo_nodes_cpp talker

# Terminal 2
ros2 run demo_nodes_cpp listener
```

Expected output: "Listener receives messages from talker"

## Step 4: Clone Course Repository

```bash
# Clone the textbook repository
cd ~/ros2_ws/src
git clone https://github.com/physical-ai-course/physical-ai-textbook.git

# Or download as ZIP and extract
# https://github.com/physical-ai-course/physical-ai-textbook/archive/refs/heads/main.zip
```

## Step 5: Organize Examples

Create a working directory for code examples:

```bash
# Create examples workspace
mkdir -p ~/ai_robotics/chapter_1
cd ~/ai_robotics/chapter_1

# Copy course examples
cp -r ~/ros2_ws/src/physical-ai-textbook/docusaurus-textbook/docs/chapter-1/examples/* .

# Verify files
ls -la
```

## Course Structure

The course is organized into 4 chapters over 13 weeks:

### Chapter 1: The Nervous System (Weeks 1-3)
**Focus**: ROS 2 fundamentals and robot description

- **Lesson 1.1**: Foundations of Physical AI (theory)
- **Lesson 1.2**: ROS 2 Core Concepts (publish/subscribe)
- **Lesson 1.3**: rclpy Packages & Launch Files (packages)
- **Lesson 1.4**: URDF/XACRO Basics (robot structure)

**Skills**: Node creation, topic communication, package development, robot modeling

### Chapter 2: The Digital Twin (Weeks 4-6)
**Focus**: Simulation and digital representation

- **Lesson 2.1**: Introduction to Simulation (Isaac Sim)
- **Lesson 2.2**: Sim-to-Real Transfer
- **Lesson 2.3**: Robot Kinematics (planned)

**Skills**: Physics simulation, parameter tuning, reality gap mitigation

### Chapter 3: The AI Brain (Weeks 7-10)
**Focus**: Machine learning and intelligent perception

- **Lesson 3.1**: Reinforcement Learning (DRL, PPO, SAC)
- **Lesson 3.2**: Perception & Sensor Fusion (RealSense)
- **Lesson 3.3**: Computer Vision with Isaac ROS (planned)

**Skills**: RL policy training, sensor integration, vision processing

### Chapter 4: Embodied Intelligence (Weeks 11-13)
**Focus**: Advanced multimodal AI and human-robot interaction

- **Lesson 4.1**: LLM Integration (language models)
- **Lesson 4.2**: Conversational AI & VLA
- **Lesson 4.3**: HRI & Safety Frameworks

**Skills**: LLM integration, multi-modal reasoning, safe robot control

## First Lesson: Hands-On Setup

Complete these exercises to verify your setup:

### Exercise 1: Run ROS 2 Publisher/Subscriber (Lesson 1.2)

```bash
# Terminal 1
cd ~/ai_robotics/chapter_1
source /opt/ros/humble/setup.bash
python3 ros2_publisher.py

# Terminal 2
source /opt/ros/humble/setup.bash
python3 ros2_subscriber.py
```

**Expected**: Subscriber receives messages from publisher every 0.5 seconds

### Exercise 2: Inspect ROS 2 System (Lesson 1.2)

```bash
source /opt/ros/humble/setup.bash

# List active nodes
ros2 node list

# List topics
ros2 topic list

# Show topic details
ros2 topic info /topic

# Monitor messages
ros2 topic echo /topic
```

**Expected**: See `minimal_publisher` and `minimal_subscriber` nodes active on `/topic`

### Exercise 3: Validate URDF (Lesson 1.4)

```bash
# Test URDF syntax
cd ~/ai_robotics/chapter_1
urdf_parser simple_robot.urdf

# Convert XACRO to URDF
xacro simple_robot.urdf.xacro > simple_robot_generated.urdf

# View structure
urdf_to_graphviz simple_robot.urdf > robot.dot
cat robot.dot  # Shows kinematic chain
```

**Expected**: No errors, URDF parses successfully

## Common Setup Issues

### Issue: "Module not found: rclpy"

**Solution**:
```bash
source /opt/ros/humble/setup.bash
python3 -c "import rclpy; print('OK')"
```

### Issue: "ros2 command not found"

**Solution**: ROS 2 not sourced
```bash
source /opt/ros/humble/setup.bash

# Make permanent
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
```

### Issue: "Permission denied" for Jetson access

**Solution**: Add user to dialout group (for USB/serial)
```bash
sudo usermod -a -G dialout $USER
# Restart shell or reboot
```

### Issue: "USB Devices not visible"

**Solution**: Install USB rules
```bash
sudo apt install ros-humble-librealsense2-dev  # For RealSense

# Or manually: Add udev rules and restart
sudo apt install libusb-1.0-0
```

## Optional: Docusaurus Local Server

To view the textbook locally with live updates:

```bash
# Install Node.js 18+ (if needed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install nodejs

# Navigate to textbook
cd ~/ros2_ws/src/physical-ai-textbook/docusaurus-textbook

# Install dependencies
npm install

# Start development server
npm run start

# View at http://localhost:3000
```

## Next Steps

After completing setup:

1. **Read Lesson 1.1** (Foundations of Physical AI) to understand core concepts
2. **Complete Lesson 1.2** (ROS 2 Core Concepts) with hands-on exercises
3. **Work through Lessons 1.3-1.4** to build packages and robot descriptions
4. **Join the community** at [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)

## Additional Resources

### Official Documentation
- [ROS 2 Humble](https://docs.ros.org/en/humble/) - Core ROS 2 documentation
- [NVIDIA Jetson Orin Nano](https://developer.nvidia.com/jetson-orin-nano-developer-kit) - Hardware documentation
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) - Physics simulation platform
- [URDF Documentation](http://wiki.ros.org/urdf) - Robot description format

### Community
- [ROS 2 Discourse](https://discourse.ros.org/) - Q&A forum
- [ROS 2 GitHub Issues](https://github.com/ros2/ros2/issues) - Bug reports and features
- [Course GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions) - Course-specific help

### Learning Resources
- [ROS 2 Beginner Tutorials](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries.html)
- [Writing URDF from Scratch](http://wiki.ros.org/urdf/Tutorials/Building%20a%20Visual%20Robot%20Model%20with%20URDF%20from%20Scratch)
- [NVIDIA Robotics Tutorials](https://developer.nvidia.com/robotics)

## Hardware Setup (Optional)

If using Intel RealSense camera or ReSpeaker microphone:

### RealSense D435i Camera

```bash
# Install RealSense SDK
sudo apt install librealsense2-dev

# Install ROS 2 wrapper
sudo apt install ros-humble-realsense2-camera

# Verify camera
realsense-viewer
```

### ReSpeaker USB Mic Array

```bash
# Install audio drivers
git clone https://github.com/respeaker/seeed-voicecard.git
cd seeed-voicecard
sudo ./install.sh --with-respeaker-usb-4mic-array

# Reboot required
sudo reboot
```

## Getting Help

- **Technical Issues**: Post in [GitHub Discussions](https://github.com/physical-ai-course/physical-ai-textbook/discussions)
- **Jetson-specific**: Check [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- **ROS 2 Questions**: Ask on [ROS 2 Discourse](https://discourse.ros.org/)

---

**Ready to start?** Proceed to [Lesson 1.1: Foundations of Physical AI](chapter-1/1-1-foundations-pai.md)
