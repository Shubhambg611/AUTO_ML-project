// static/js/app.js
import React from 'react';
import { createRoot } from 'react-dom/client';
import ChatInterface from './components/ChatInterface';
import MLProcessExplainer from './components/MLProcessExplainer';

console.log('App.js loaded');

// Initialize global components object
window.Components = {
    MLProcessExplainer,
    ChatInterface
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    
    // Initialize Chat Interface
    const chatRoot = document.getElementById('chat-root');
    if (chatRoot) {
        try {
            console.log('Creating React root for chat');
            const chatRootInstance = createRoot(chatRoot);
            console.log('Rendering ChatInterface');
            chatRootInstance.render(
                <React.StrictMode>
                    <ChatInterface />
                </React.StrictMode>
            );
            console.log('Chat render complete');
        } catch (error) {
            console.error('Error mounting chat interface:', error);
        }
    }

    // Initialize Process Explainer
    const processExplainerRoot = document.getElementById('process-explainer-root');
    if (processExplainerRoot) {
        try {
            console.log('Creating React root for process explainer');
            const explainerRootInstance = createRoot(processExplainerRoot);
            console.log('Rendering MLProcessExplainer');
            explainerRootInstance.render(
                <React.StrictMode>
                    <MLProcessExplainer />
                </React.StrictMode>
            );
            console.log('Process explainer render complete');
        } catch (error) {
            console.error('Error mounting process explainer:', error);
        }
    }
});

// Export components for use in other files
export { ChatInterface, MLProcessExplainer };