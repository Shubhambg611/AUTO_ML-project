// static/js/app.js
import React from 'react';
import { createRoot } from 'react-dom/client';
import ChatInterface from './components/ChatInterface';

console.log('App.js loaded');

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    const chatRoot = document.getElementById('chat-root');
    console.log('Chat root element:', chatRoot);

    if (chatRoot) {
        try {
            console.log('Creating React root');
            const root = createRoot(chatRoot);
            console.log('Rendering ChatInterface');
            root.render(
                <React.StrictMode>
                    <ChatInterface />
                </React.StrictMode>
            );
            console.log('Render complete');
        } catch (error) {
            console.error('Error mounting chat interface:', error);
        }
    } else {
        console.error('Chat root element not found!');
    }
});