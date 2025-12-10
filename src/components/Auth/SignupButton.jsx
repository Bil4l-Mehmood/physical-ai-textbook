import React, { useState } from 'react';
import SignupModal from './SignupModal';
import styles from './SignupButton.module.css';

export default function SignupButton() {
  const [showModal, setShowModal] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(
    typeof window !== 'undefined' && localStorage.getItem('physicalai_logged_in') === 'true'
  );

  const handleLogout = () => {
    if (confirm('Are you sure you want to logout?')) {
      localStorage.removeItem('physicalai_user');
      localStorage.removeItem('physicalai_logged_in');
      setIsLoggedIn(false);
      window.location.reload();
    }
  };

  if (isLoggedIn) {
    const userData = JSON.parse(localStorage.getItem('physicalai_user') || '{}');
    return (
      <div className={styles.userInfo}>
        <span className={styles.userName}>👋 {userData.name}</span>
        <button onClick={handleLogout} className={styles.logoutButton}>
          Logout
        </button>
      </div>
    );
  }

  return (
    <>
      <button
        className={styles.signupButton}
        onClick={() => setShowModal(true)}
      >
        🚀 Sign Up / Login
      </button>
      <SignupModal isOpen={showModal} onClose={() => setShowModal(false)} />
    </>
  );
}
