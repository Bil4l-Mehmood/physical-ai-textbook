import React, { useState, useEffect } from 'react';
import SignupModal from '../Auth/SignupModal';
import styles from './styles.module.css';

export default function NavbarAuth() {
  const [showModal, setShowModal] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userData, setUserData] = useState({});

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const loggedIn = localStorage.getItem('physicalai_logged_in') === 'true';
      setIsLoggedIn(loggedIn);
      if (loggedIn) {
        const data = JSON.parse(localStorage.getItem('physicalai_user') || '{}');
        setUserData(data);
      }
    }
  }, []);

  const handleLogout = () => {
    if (confirm('Are you sure you want to logout?')) {
      localStorage.removeItem('physicalai_user');
      localStorage.removeItem('physicalai_logged_in');
      setIsLoggedIn(false);
      setUserData({});
      window.location.reload();
    }
  };

  const handleModalClose = () => {
    setShowModal(false);
    // Check if user just logged in
    if (typeof window !== 'undefined') {
      const loggedIn = localStorage.getItem('physicalai_logged_in') === 'true';
      if (loggedIn) {
        setIsLoggedIn(true);
        const data = JSON.parse(localStorage.getItem('physicalai_user') || '{}');
        setUserData(data);
      }
    }
  };

  if (isLoggedIn) {
    return (
      <div className={styles.userContainer}>
        <span className={styles.userName}>👋 {userData.name}</span>
        <button onClick={handleLogout} className={styles.logoutBtn}>
          Logout
        </button>
      </div>
    );
  }

  return (
    <>
      <div className={styles.authButtons}>
        <button
          className={styles.signInBtn}
          onClick={() => setShowModal(true)}
        >
          Sign In
        </button>
        <button
          className={styles.signUpBtn}
          onClick={() => setShowModal(true)}
        >
          Sign Up
        </button>
      </div>
      <SignupModal isOpen={showModal} onClose={handleModalClose} />
    </>
  );
}