import React from 'react';
import styles from './HardwareShowcase.module.css';

const hardware = [
  {
    name: 'NVIDIA Jetson Orin Nano',
    category: 'Compute Platform',
    icon: '💻',
    specs: ['8GB RAM', '1024-core GPU', 'AI Performance: 40 TOPS'],
    color: '#76b900',
  },
  {
    name: 'Intel RealSense D435i',
    category: 'Depth Camera',
    icon: '📷',
    specs: ['Depth + RGB', 'IMU Sensor', 'Up to 90 FPS'],
    color: '#0071c5',
  },
  {
    name: 'ReSpeaker Mic Array',
    category: 'Audio Input',
    icon: '🎙️',
    specs: ['4-Mic Array', 'Voice Recognition', 'USB 2.0'],
    color: '#e74c3c',
  },
  {
    name: 'NVIDIA Isaac Sim',
    category: 'Simulation',
    icon: '🌐',
    specs: ['PhotoRealistic', 'Physics Engine', 'ROS 2 Integration'],
    color: '#9b59b6',
  },
];

const software = [
  { name: 'ROS 2 Humble', icon: '🤖' },
  { name: 'Python 3.10+', icon: '🐍' },
  { name: 'Ubuntu 22.04', icon: '🐧' },
  { name: 'PyTorch', icon: '🔥' },
  { name: 'OpenCV', icon: '👁️' },
  { name: 'CUDA', icon: '⚡' },
];

export default function HardwareShowcase() {
  return (
    <div className={styles.showcaseSection}>
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>Hardware & Software Stack</h2>
        <p className={styles.sectionSubtitle}>
          Industry-standard tools and cutting-edge hardware for real-world robotics development
        </p>
      </div>

      <div className={styles.hardwareGrid}>
        {hardware.map((item, index) => (
          <div
            key={index}
            className={styles.hardwareCard}
            style={{ animationDelay: `${index * 0.1}s` }}>
            <div
              className={styles.hardwareIcon}
              style={{ background: item.color }}>
              {item.icon}
            </div>
            <div className={styles.hardwareInfo}>
              <div className={styles.category}>{item.category}</div>
              <h3 className={styles.hardwareName}>{item.name}</h3>
              <div className={styles.specsList}>
                {item.specs.map((spec, idx) => (
                  <div key={idx} className={styles.specItem}>
                    <span className={styles.checkmark}>✓</span>
                    {spec}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className={styles.softwareSection}>
        <h3 className={styles.softwareTitle}>Software & Frameworks</h3>
        <div className={styles.softwareGrid}>
          {software.map((item, index) => (
            <div
              key={index}
              className={styles.softwareTag}
              style={{ animationDelay: `${index * 0.05}s` }}>
              <span className={styles.softwareIcon}>{item.icon}</span>
              <span className={styles.softwareName}>{item.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
