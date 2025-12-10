import React, { useState } from 'react';
import styles from './styles.module.css';

export default function PersonalizeButton({ chapterTitle }) {
  const [personalized, setPersonalized] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePersonalize = async () => {
    setLoading(true);

    // Get user data from localStorage
    const userData = JSON.parse(localStorage.getItem('physicalai_user') || '{}');

    if (!userData.name) {
      alert('Please sign up first to personalize content!');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/personalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapter: chapterTitle,
          software_background: userData.softwareBackground,
          hardware_background: userData.hardwareBackground,
          learning_goals: userData.learningGoals,
          user_name: userData.name,
        }),
      });

      const data = await response.json();
      setPersonalized(data.content);
    } catch (error) {
      console.error('Personalization error:', error);
      setPersonalized('Unable to personalize right now. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.personalizeContainer}>
      <button
        className={styles.personalizeButton}
        onClick={handlePersonalize}
        disabled={loading}
      >
        {loading ? '⏳ Personalizing...' : '🎯 Personalize This Chapter for Me'}
      </button>

      {personalized && (
        <div className={styles.personalizedContent}>
          <h3>📚 Your Personalized Learning Path</h3>
          <div className={styles.content}>{personalized}</div>
        </div>
      )}
    </div>
  );
}
