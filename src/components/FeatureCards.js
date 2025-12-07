import React from 'react';
import styles from './FeatureCards.module.css';

const features = [
  {
    icon: '🧠',
    title: 'AI-Powered Learning',
    description: 'Master reinforcement learning, computer vision, and LLMs for robotics applications.',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  },
  {
    icon: '🤖',
    title: 'Hands-On Hardware',
    description: 'Work directly with NVIDIA Jetson Orin Nano, RealSense cameras, and robotic platforms.',
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
  },
  {
    icon: '🎮',
    title: 'Simulation First',
    description: 'Learn sim-to-real workflows using NVIDIA Isaac Sim before deploying to hardware.',
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
  },
  {
    icon: '⚡',
    title: 'ROS 2 Ecosystem',
    description: 'Build production-ready robots using industry-standard ROS 2 middleware and tools.',
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
  },
  {
    icon: '🎯',
    title: 'Project-Based',
    description: 'Complete real-world projects from autonomous navigation to human-robot interaction.',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
  },
  {
    icon: '📚',
    title: 'Comprehensive',
    description: 'From basics to advanced topics - everything you need in one complete curriculum.',
    gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
  },
];

export default function FeatureCards() {
  return (
    <div className={styles.featuresSection}>
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>Why This Textbook?</h2>
        <p className={styles.sectionSubtitle}>
          A comprehensive, hands-on approach to mastering Physical AI and Humanoid Robotics
        </p>
      </div>

      <div className={styles.featuresGrid}>
        {features.map((feature, index) => (
          <div
            key={index}
            className={styles.featureCard}
            style={{ animationDelay: `${index * 0.1}s` }}>
            <div
              className={styles.featureIcon}
              style={{ background: feature.gradient }}>
              {feature.icon}
            </div>
            <h3 className={styles.featureTitle}>{feature.title}</h3>
            <p className={styles.featureDescription}>{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
