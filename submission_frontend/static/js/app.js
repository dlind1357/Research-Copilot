// Front-end state and configuration
const config = {
    apiEndpoint: '/api/chat',
    typingSpeedMs: 15,          // Simulated streaming character delay
    isStreamingMode: false       // Set to true to switch to actual Server-Sent Events later
};

// DOM Cache
const dom = {
    chatForm: document.getElementById('chatForm'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    conversationPanel: document.getElementById('conversationPanel'),
    welcomeView: document.getElementById('welcomeView'),
    messagesContainer: document.getElementById('messagesContainer'),
    typingIndicator: document.getElementById('typingIndicator')
};

// Auto-resize textarea to expand with multi-line inputs
dom.messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight - 4) + 'px';
});

// Suggested click triggers
function applySuggestion(text) {
    dom.messageInput.value = text;
    dom.messageInput.style.height = 'auto';
    dom.messageInput.style.height = (dom.messageInput.scrollHeight - 4) + 'px';
    dom.messageInput.focus();
}

// Global Keyboard Shortcut listeners
dom.messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Stop newline
        dom.chatForm.dispatchEvent(new Event('submit'));
    }
});

// Handle Form Submission
dom.chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const query = dom.messageInput.value.trim();
    if (!query) return;

    // Reset input layout
    dom.messageInput.value = '';
    dom.messageInput.style.height = 'auto';

    // Hide welcome panel on initial interaction
    if (dom.welcomeView) {
        dom.welcomeView.style.display = 'none';
    }

    // Render User Query
    appendMessage('user', query);
    scrollPanelToBottom();

    // Toggle loading states
    setLoadingState(true);

    try {
        if (config.isStreamingMode) {
            // Future streaming / SSE execution placeholder
            await executeStreamingChat(query);
        } else {
            // Regular POST with simulated streaming output
            const response = await fetchChatPlaceholder(query);
            await appendMessageWithSimulatedStream('assistant', response.response);
        }
    } catch (err) {
        console.error('Chat error:', err);
        appendMessage('assistant', '⚠️ An error occurred while communicating with the service. Please try again.');
    } finally {
        setLoadingState(false);
        scrollPanelToBottom();
    }
});

// Appends user or direct text messages instantly
function appendMessage(sender, text) {
    const timestamp = getFormattedTime();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${sender}`;

    const parsedHTML = parseSimpleMarkdown(text);

    messageDiv.innerHTML = `
        <div class="${sender}-avatar">
            <i class="lucide-icon" data-lucide="${sender === 'user' ? 'user' : 'bot'}"></i>
        </div>
        <div class="bubble">
            <div class="message-content">${parsedHTML}</div>
            <span class="time-stamp">${timestamp}</span>
        </div>
    `;

    dom.messagesContainer.appendChild(messageDiv);
    
    // Re-initialize newly rendered icons
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Simulated Streaming Token Printing
function appendMessageWithSimulatedStream(sender, text) {
    return new Promise((resolve) => {
        const timestamp = getFormattedTime();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${sender}`;

        messageDiv.innerHTML = `
            <div class="${sender}-avatar">
                <i class="lucide-icon" data-lucide="${sender === 'user' ? 'user' : 'bot'}"></i>
            </div>
            <div class="bubble">
                <div class="message-content"></div>
                <span class="time-stamp">${timestamp}</span>
            </div>
        `;

        dom.messagesContainer.appendChild(messageDiv);
        const contentBox = messageDiv.querySelector('.message-content');

        if (window.lucide) {
            window.lucide.createIcons();
        }

        let currentIdx = 0;
        
        // Emulates streaming by outputting segments incrementally
        function printNextChar() {
            if (currentIdx < text.length) {
                // Read chunks of characters at a time for smoother output flow
                const chunkSize = Math.min(3, text.length - currentIdx);
                const part = text.substring(0, currentIdx + chunkSize);
                contentBox.innerHTML = parseSimpleMarkdown(part);
                currentIdx += chunkSize;
                scrollPanelToBottom();
                setTimeout(printNextChar, config.typingSpeedMs);
            } else {
                // Ensure complete parsing of raw text
                contentBox.innerHTML = parseSimpleMarkdown(text);
                scrollPanelToBottom();
                resolve();
            }
        }

        printNextChar();
    });
}

// Fetch helper from standard JSON API
async function fetchChatPlaceholder(message) {
    const res = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
    });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

// Future implementation area for genuine Server Sent Events (SSE)
async function executeStreamingChat(message) {
    // Left open for the user to connect their EventSource API cleanly later:
    // const eventSource = new EventSource(`/api/chat/stream?msg=${encodeURIComponent(message)}`);
    // eventSource.onmessage = (event) => { ... }
}

// UI state management helper
function setLoadingState(isLoading) {
    if (isLoading) {
        dom.typingIndicator.style.display = 'flex';
        dom.sendBtn.disabled = true;
        dom.messageInput.disabled = true;
    } else {
        dom.typingIndicator.style.display = 'none';
        dom.sendBtn.disabled = false;
        dom.messageInput.disabled = false;
        dom.messageInput.focus();
    }
}

// Auto-Scroll helper
function scrollPanelToBottom() {
    dom.conversationPanel.scrollTop = dom.conversationPanel.scrollHeight;
}

// Utility formatting time helper
function getFormattedTime() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Simple on-the-fly markdown parser
function parseSimpleMarkdown(markdown) {
    let html = markdown;

    // Code Blocks (```code```)
    html = html.replace(/```([\s\S]+?)```/g, '<pre><code>$1</code></pre>');

    // Inline Code (`code`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold (**text**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Links ([text](url)) - Convert to a blank-target anchor tag
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // Split into lines to parse lists and paragraphs properly
    const lines = html.split('\n');
    let inUserList = false; // for '-' or '*' bullets
    let inNumberedList = false; // for '1.' bullets
    let processedLines = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Match bullets: - or *
        const listMatch = line.match(/^(\s*)([-*])\s+(.+)$/);
        // Match numbered list: 1. or 2.
        const numMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);

        if (listMatch) {
            if (inNumberedList) {
                processedLines.push('</ol>');
                inNumberedList = false;
            }
            if (!inUserList) {
                processedLines.push('<ul>');
                inUserList = true;
            }
            processedLines.push(`<li>${listMatch[3]}</li>`);
        } else if (numMatch) {
            if (inUserList) {
                processedLines.push('</ul>');
                inUserList = false;
            }
            if (!inNumberedList) {
                processedLines.push('<ol>');
                inNumberedList = true;
            }
            processedLines.push(`<li>${numMatch[3]}</li>`);
        } else {
            // Close any active list
            if (inUserList) {
                processedLines.push('</ul>');
                inUserList = false;
            }
            if (inNumberedList) {
                processedLines.push('</ol>');
                inNumberedList = false;
            }
            
            // If empty line, add paragraph break or keep it
            if (line.trim() === '') {
                processedLines.push('<br/>');
            } else {
                processedLines.push(line);
            }
        }
    }
    
    // Close remaining open list tags
    if (inUserList) processedLines.push('</ul>');
    if (inNumberedList) processedLines.push('</ol>');

    // Join lines
    html = processedLines.join('\n');

    // Handle paragraph splits with multiple line breaks
    html = html.replace(/(<br\s*\/?>\s*){2,}/g, '</p><p>');
    
    // If we have paragraph breaks, wrap the entire output nicely
    if (html.includes('</p>')) {
        html = '<p>' + html + '</p>';
        html = html.replace(/<p>\s*<\/p>/g, '');
    }

    return html;
}
