import React, { useState, useRef, useEffect } from 'react';
import styles from './ChatWidget.module.css';

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  const API_URL = 'http://localhost:8000';

  // Detect text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();
      if (text.length > 10) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      // Determine endpoint based on whether text is selected
      const endpoint = selectedText
        ? `${API_URL}/api/chat/selection`
        : `${API_URL}/api/chat`;

      const body = selectedText
        ? { question: input, selected_text: selectedText }
        : { question: input, top_k: 5 };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Format assistant message
      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        responseTime: data.response_time_ms,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSelectedText(''); // Clear selection after use
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error.message}. Make sure the backend server is running at ${API_URL}`,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Chat Button */}
      <button
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chat"
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          {/* Header */}
          <div className={styles.chatHeader}>
            <h3>📚 AI Textbook Assistant</h3>
            <p>Ask me about Physical AI & Humanoid Robotics</p>
            {selectedText && (
              <div className={styles.selectionBadge}>
                ✓ Text selected ({selectedText.length} chars)
              </div>
            )}
          </div>

          {/* Messages */}
          <div className={styles.chatMessages}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <p>👋 Hi! I can help you with questions about this textbook.</p>
                <p><strong>Tips:</strong></p>
                <ul>
                  <li>Ask me anything about ROS 2, robotics, or AI</li>
                  <li>Select text on the page to ask questions about specific sections</li>
                </ul>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`${styles.message} ${styles[msg.role]}`}>
                <div className={styles.messageContent}>
                  {msg.content}

                  {/* Citations */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className={styles.citations}>
                      <strong>Sources:</strong>
                      <ul>
                        {msg.citations.map((cite, i) => (
                          <li key={i}>
                            <a href={cite.source_url} target="_blank" rel="noopener noreferrer">
                              {cite.chapter_title} - {cite.section}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Response time */}
                  {msg.responseTime && (
                    <div className={styles.responseTime}>
                      Answered in {msg.responseTime}ms
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className={`${styles.message} ${styles.assistant}`}>
                <div className={styles.messageContent}>
                  <div className={styles.loader}>Thinking...</div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className={styles.chatInput}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                selectedText
                  ? 'Ask about the selected text...'
                  : 'Ask a question about the textbook...'
              }
              rows={2}
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading || !input.trim()}>
              {loading ? '⏳' : '➤'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
