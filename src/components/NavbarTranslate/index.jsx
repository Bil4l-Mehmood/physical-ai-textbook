import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

export default function NavbarTranslate() {
  const [showDropdown, setShowDropdown] = useState(false);
  const [currentLang, setCurrentLang] = useState('English');

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'ur', name: 'اردو (Urdu)', flag: '🇵🇰' },
    // Add more languages as needed
  ];

  const handleLanguageSelect = (lang) => {
    setCurrentLang(lang.name);
    setShowDropdown(false);

    // Store selected language preference
    if (typeof window !== 'undefined') {
      localStorage.setItem('physicalai_language', lang.code);
      // Trigger a custom event that lesson pages can listen to
      window.dispatchEvent(new CustomEvent('languageChanged', { detail: lang.code }));
    }
  };

  return (
    <div className={styles.translateContainer}>
      <button
        className={styles.translateBtn}
        onClick={() => setShowDropdown(!showDropdown)}
        onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
      >
        🌐 Translate
      </button>

      {showDropdown && (
        <div className={styles.dropdown}>
          {languages.map((lang) => (
            <button
              key={lang.code}
              className={styles.langOption}
              onClick={() => handleLanguageSelect(lang)}
            >
              <span className={styles.flag}>{lang.flag}</span>
              <span>{lang.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}