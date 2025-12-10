import React, { useState } from 'react';
import styles from './SignupModal.module.css';

export default function SignupModal({ isOpen, onClose, onSignupSuccess }) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    softwareBackground: '',
    hardwareBackground: '',
    learningGoals: '',
  });

  const handleSubmit = (e) => {
    e.preventDefault();

    // Save user data to localStorage
    const userData = {
      ...formData,
      signupDate: new Date().toISOString(),
      userId: Date.now().toString(),
    };

    localStorage.setItem('physicalai_user', JSON.stringify(userData));
    localStorage.setItem('physicalai_logged_in', 'true');

    alert('Welcome to Physical AI Course! 🤖');

    // Call success callback if provided
    if (onSignupSuccess) {
      onSignupSuccess();
    } else {
      onClose();
      window.location.reload(); // Refresh to show personalized content
    }
  };

  if (!isOpen) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <button className={styles.closeButton} onClick={onClose}>✕</button>

        <h2>🚀 Welcome to Physical AI & Humanoid Robotics!</h2>
        <p>Tell us about yourself to personalize your learning experience</p>

        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label>Full Name *</label>
            <input
              type="text"
              placeholder="Enter your name"
              value={formData.name}
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              required
            />
          </div>

          <div className={styles.formGroup}>
            <label>Email *</label>
            <input
              type="email"
              placeholder="your.email@example.com"
              value={formData.email}
              onChange={(e) => setFormData({...formData, email: e.target.value})}
              required
            />
          </div>

          <div className={styles.formGroup}>
            <label>Software Background *</label>
            <select
              value={formData.softwareBackground}
              onChange={(e) => setFormData({...formData, softwareBackground: e.target.value})}
              required
            >
              <option value="">Select your background...</option>
              <option value="beginner">Beginner - New to programming</option>
              <option value="some-python">Some Python experience</option>
              <option value="intermediate">Intermediate - Built projects before</option>
              <option value="advanced">Advanced - Professional developer</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Hardware Background *</label>
            <select
              value={formData.hardwareBackground}
              onChange={(e) => setFormData({...formData, hardwareBackground: e.target.value})}
              required
            >
              <option value="">Select your background...</option>
              <option value="none">No hardware experience</option>
              <option value="arduino">Worked with Arduino/Raspberry Pi</option>
              <option value="robotics">Some robotics experience</option>
              <option value="professional">Professional roboticist</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>What do you want to learn? *</label>
            <textarea
              placeholder="e.g., Build a robot, Learn ROS 2, Understand AI in robotics..."
              value={formData.learningGoals}
              onChange={(e) => setFormData({...formData, learningGoals: e.target.value})}
              rows={3}
              required
            />
          </div>

          <button type="submit" className={styles.submitButton}>
            Start Learning 🎯
          </button>
        </form>

        <p className={styles.privacyNote}>
          🔒 Your data is stored locally and never shared
        </p>
      </div>
    </div>
  );
}
