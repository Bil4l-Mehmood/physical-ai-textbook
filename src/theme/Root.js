import React, { useEffect, useState } from 'react';
import { useLocation, useHistory } from '@docusaurus/router';
import ChatWidget from '../components/RAGChatbot/ChatWidget';
import SignupModal from '../components/Auth/SignupModal';

// Root component wraps all pages in Docusaurus
// This is the recommended way to add global components
export default function Root({ children }) {
  const location = useLocation();
  const history = useHistory();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication status on route change
  useEffect(() => {
    const userData = localStorage.getItem('physicalai_user');
    const isAuth = !!userData;
    setIsAuthenticated(isAuth);

    // Check if current route is protected (docs pages and getting-started)
    const isProtectedRoute =
      location.pathname.startsWith('/docs/') ||
      location.pathname === '/docs/getting-started' ||
      location.pathname.startsWith('/docs/chapter-');

    // If trying to access protected route without authentication, show modal
    if (isProtectedRoute && !isAuth) {
      setShowAuthModal(true);
    }
  }, [location.pathname]);

  // Handle successful signup - close modal and allow navigation
  const handleSignupSuccess = () => {
    setShowAuthModal(false);
    setIsAuthenticated(true);
  };

  // Handle modal close - redirect to homepage if not authenticated
  const handleModalClose = () => {
    setShowAuthModal(false);
    if (!isAuthenticated && (location.pathname.startsWith('/docs/') || location.pathname === '/docs/getting-started')) {
      history.push('/');
    }
  };

  // If trying to access protected content without auth, show auth wall
  const shouldShowContent = !location.pathname.startsWith('/docs/') || isAuthenticated;

  return (
    <>
      {shouldShowContent ? children : (
        <div style={{
          minHeight: '80vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          padding: '2rem'
        }}>
          <div>
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🔒</div>
            <h1>Authentication Required</h1>
            <p style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>
              Please sign up or sign in to access course content.
            </p>
          </div>
        </div>
      )}
      <ChatWidget />

      {showAuthModal && (
        <SignupModal
          isOpen={showAuthModal}
          onClose={handleModalClose}
          onSignupSuccess={handleSignupSuccess}
        />
      )}
    </>
  );
}
