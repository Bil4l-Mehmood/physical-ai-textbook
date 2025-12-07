/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

/** @type {import('@docusaurus/types').SidebarsConfig} */
const sidebars = {
  // But you can create a sidebar manually
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Home',
    },
    {
      type: 'category',
      label: 'Chapter 1: The Robotic Nervous System',
      position: 1,
      collapsible: true,
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'chapter-1/1-1-foundations-pai',
          label: 'Lesson 1.1: Foundations of Physical AI & Embodied Intelligence',
        },
        {
          type: 'doc',
          id: 'chapter-1/1-2-ros2-core',
          label: 'Lesson 1.2: ROS 2 Core Concepts: Nodes, Topics, Services',
        },
        {
          type: 'doc',
          id: 'chapter-1/1-3-rclpy-packages',
          label: 'Lesson 1.3: Building rclpy Packages and Launch Files',
        },
        {
          type: 'doc',
          id: 'chapter-1/1-4-urdf-xacro',
          label: 'Lesson 1.4: Defining the Physical Body: URDF/XACRO',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 2: The Digital Twin',
      position: 2,
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'chapter-2/2-1-gazebo-fundamentals',
          label: 'Lesson 2.1: Gazebo Fundamentals - Physics Simulation for Robots',
        },
        {
          type: 'doc',
          id: 'chapter-2/2-2-urdf-sdf-physics',
          label: 'Lesson 2.2: URDF/SDF & Physics - Advanced Configuration',
        },
        {
          type: 'doc',
          id: 'chapter-2/2-3-sensors-unity',
          label: 'Lesson 2.3: Sensors & Unity - Integration and Visualization',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 3: The AI-Robot Brain',
      position: 3,
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'chapter-3/3-1-isaac-sim-basics',
          label: 'Lesson 3.1: NVIDIA Isaac Sim - Photorealistic Simulation',
        },
        {
          type: 'doc',
          id: 'chapter-3/3-2-vslam-navigation',
          label: 'Lesson 3.2: VSLAM & Navigation - Autonomous Mapping',
        },
        {
          type: 'doc',
          id: 'chapter-3/3-3-computer-vision-isaac-ros',
          label: 'Lesson 3.3: Computer Vision with Isaac ROS - Real-Time Perception',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 4: Vision-Language-Action',
      position: 4,
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'chapter-4/4-1-llm-brain',
          label: 'Lesson 4.1: LLM Integration for Robotics - Cognitive Core',
        },
        {
          type: 'doc',
          id: 'chapter-4/4-2-conversational-ai',
          label: 'Lesson 4.2: Conversational AI & Vision-Language-Action (VLA)',
        },
        {
          type: 'doc',
          id: 'chapter-4/4-3-hri-safety',
          label: 'Lesson 4.3: Human-Robot Interaction & Safety',
        },
      ],
    },
  ],
};

module.exports = sidebars;
