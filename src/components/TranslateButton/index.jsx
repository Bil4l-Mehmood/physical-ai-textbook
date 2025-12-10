import React, { useState } from 'react';
import styles from './styles.module.css';

export default function TranslateButton({ content, chapterTitle }) {
  const [translated, setTranslated] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showOriginal, setShowOriginal] = useState(true);

  const handleTranslate = async () => {
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/translate/urdu', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: content || chapterTitle,
          chapter_title: chapterTitle,
          page_url: window.location.href, // Send current page URL for backend tracking
        }),
      });

      const data = await response.json();
      setTranslated(data.translated);
      setShowOriginal(false);

      // Log successful file save (if backend returns the path)
      if (data.saved_to) {
        console.log('✅ Translation saved to backend:', data.filename);
      }
    } catch (error) {
      console.error('Translation error:', error);
      setTranslated('ترجمہ فی الحال دستیاب نہیں ہے۔ براہ کرم بعد میں کوشش کریں۔');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.translateContainer}>
      <div className={styles.buttonGroup}>
        <button
          className={styles.translateButton}
          onClick={handleTranslate}
          disabled={loading}
        >
          {loading ? '⏳ Translating...' : '🌐 اردو میں پڑھیں (Read in Urdu)'}
        </button>

        {translated && (
          <button
            className={styles.toggleButton}
            onClick={() => setShowOriginal(!showOriginal)}
          >
            {showOriginal ? 'Show Urdu' : 'Show English'}
          </button>
        )}
      </div>

      {translated && !showOriginal && (
        <div className={styles.urduContent} dir="rtl">
          <h3>📖 اردو ترجمہ</h3>
          <div className={styles.content}>{translated}</div>
          <p className={styles.note}>
            نوٹ: یہ خودکار ترجمہ ہے۔ تکنیکی اصطلاحات انگریزی میں رکھی گئی ہیں۔
          </p>
        </div>
      )}
    </div>
  );
}
