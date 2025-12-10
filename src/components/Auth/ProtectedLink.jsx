import React, { useState } from 'react';
import { useHistory } from '@docusaurus/router';
import SignupModal from './SignupModal';

/**
 * ProtectedLink - A link component that requires authentication
 * Shows SignupModal if user is not authenticated
 */
export default function ProtectedLink({ to, children, className, style, onClick }) {
  const [showSignupModal, setShowSignupModal] = useState(false);
  const history = useHistory();

  const handleClick = (e) => {
    e.preventDefault();

    // Check if user is authenticated
    const userData = localStorage.getItem('physicalai_user');

    if (userData) {
      // User is authenticated, navigate to destination
      if (onClick) {
        onClick(e);
      }
      history.push(to);
    } else {
      // User is not authenticated, show signup modal
      setShowSignupModal(true);
    }
  };

  return (
    <>
      <a
        href={to}
        onClick={handleClick}
        className={className}
        style={style}
      >
        {children}
      </a>

      {showSignupModal && (
        <SignupModal
          isOpen={showSignupModal}
          onClose={() => setShowSignupModal(false)}
        />
      )}
    </>
  );
}
